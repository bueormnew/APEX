"""
ECHO (Engrama Compresivo de Alta Ortogonalidad)
Implementación nativa y estabilizada en PyTorch para sustituir FFNs en bloques de mezcla de secuencias.

Características clave:
1. Product Key Memory: Descompone la búsqueda en dos mitades (q1, q2) sobre K1 y K2, permitiendo n_keys^2 combinaciones con 2 * n_keys comparaciones.
2. Micro-expertos factorizados de bajo rango (r): W_down, W_up por lado.
3. Softmax continuo conjunto sobre las k * k combinaciones elegidas (o suave ponderado) para estabilidad y diferenciabilidad end-to-end.
4. Compresor compartido de cuello de botella (bottleneck) para abstracción semántica.
5. Compuerta SiLU multiplicativa.
6. Pérdida auxiliar de balanceo de carga (Load Balancing Loss) y regularización de entropía para evitar colapso de expertos.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = torch.mean(x ** 2, dim=-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight


class ECHO(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_keys: int = 64,
        top_k: int = 4,
        rank: int = 16,
        bottleneck_dim: Optional[int] = None,
        dropout: float = 0.0,
        load_balance_coef: float = 0.01,
        temperature: float = 1.0,
    ):
        """
        Args:
            d_model: Dimensión del modelo (d). Debe ser par para dividir en d/2.
            n_keys: Número de claves por cada lado (N = n_keys^2 combinaciones totales).
            top_k: Número de claves seleccionadas en cada mitad (k).
            rank: Rango de cada micro-experto (r).
            bottleneck_dim: Dimensión intermedia del compresor compartido. Si None, k * k * 2.
            dropout: Tasa de dropout.
            load_balance_coef: Coeficiente alpha para pérdida auxiliar de balanceo.
            temperature: Temperatura para la softmax de los scores de claves.
        """
        super().__init__()
        assert d_model % 2 == 0, "d_model debe ser divisible por 2 para product keys"
        self.d_model = d_model
        self.half_dim = d_model // 2
        self.n_keys = n_keys
        self.top_k = min(top_k, n_keys)
        self.rank = rank
        self.load_balance_coef = load_balance_coef
        self.temperature = temperature
        
        # Dimensión recomendada en el análisis (§8.7): escalar con combinaciones activas
        if bottleneck_dim is None:
            self.bottleneck_dim = max(32, self.top_k * self.top_k * 2)
        else:
            self.bottleneck_dim = bottleneck_dim

        # Consulta rápida
        self.q_norm = RMSNorm(d_model)
        self.W_q = nn.Linear(d_model, d_model, bias=False)

        # Claves K1 y K2 inicializadas a escala unitaria / norma esperada tras RMSNorm
        scale_key = 1.0 / math.sqrt(self.half_dim)
        self.K1 = nn.Parameter(torch.randn(n_keys, self.half_dim) * scale_key)
        self.K2 = nn.Parameter(torch.randn(n_keys, self.half_dim) * scale_key)

        # Micro-expertos factorizados de bajo rango
        # down: [n_keys, d_model, rank]
        # up:   [n_keys, rank, d_model]
        scale_down = 1.0 / math.sqrt(d_model)
        scale_up = 1.0 / math.sqrt(rank)
        self.down1 = nn.Parameter(torch.randn(n_keys, d_model, rank) * scale_down)
        self.up1 = nn.Parameter(torch.randn(n_keys, rank, d_model) * scale_up)
        self.down2 = nn.Parameter(torch.randn(n_keys, d_model, rank) * scale_down)
        self.up2 = nn.Parameter(torch.randn(n_keys, rank, d_model) * scale_up)

        # Compresor compartido (bottleneck)
        self.comp_norm = RMSNorm(d_model)
        self.W_o1 = nn.Linear(d_model, self.bottleneck_dim, bias=False)
        self.W_o2 = nn.Linear(self.bottleneck_dim, d_model, bias=False)

        # Compuerta y proyección de salida
        self.W_gate = nn.Linear(d_model, d_model, bias=False)
        self.W_out = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Variable para almacenar la métrica de balanceo y entropía
        self.last_aux_loss = torch.tensor(0.0)
        self.last_entropy = 1.0

    def forward(
        self, 
        h: torch.Tensor, 
        return_aux_loss: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass end-to-end de ECHO.
        Args:
            h: [B, T, d_model] o [B, d_model]
            return_aux_loss: si True, retorna el dict de métricas incluyendo loss_balance.
        Returns:
            out: [B, T, d_model]
            metrics: dict con aux_loss, entropy_k1, entropy_k2
        """
        orig_shape = h.shape
        if h.dim() == 2:
            h = h.unsqueeze(1)
        B, T, D = h.shape
        N_tokens = B * T
        h_flat = h.reshape(N_tokens, D)

        # 1. Consulta rápida - Product Keys
        q = self.W_q(self.q_norm(h_flat))  # [N, D]
        q1, q2 = torch.chunk(q, 2, dim=-1)  # [N, D/2] cada uno

        # Scores sobre K1 y K2: [N, n_keys]
        s1 = torch.matmul(q1, self.K1.t()) / (math.sqrt(self.half_dim) * self.temperature)
        s2 = torch.matmul(q2, self.K2.t()) / (math.sqrt(self.half_dim) * self.temperature)

        # Softmax completa para calcular pérdida de balanceo y evitar colapso
        probs_s1 = F.softmax(s1, dim=-1) # [N, n_keys]
        probs_s2 = F.softmax(s2, dim=-1) # [N, n_keys]

        # Monitoreo de entropía normalizada y cálculo de pérdida auxiliar
        # Penaliza varianza en el uso agregado por batch
        mean_usage_1 = probs_s1.mean(dim=0) # [n_keys]
        mean_usage_2 = probs_s2.mean(dim=0) # [n_keys]
        
        # Coeficiente de variación / dispersión de carga
        loss_balance = (
            self.n_keys * torch.sum(mean_usage_1 ** 2) - 1.0 +
            self.n_keys * torch.sum(mean_usage_2 ** 2) - 1.0
        )
        aux_loss = self.load_balance_coef * F.relu(loss_balance)

        # 2. Selección top-k por lado
        top1_val, top1_idx = torch.topk(s1, self.top_k, dim=-1) # [N, k]
        top2_val, top2_idx = torch.topk(s2, self.top_k, dim=-1) # [N, k]

        # Softmax conjunta sobre las k*k combinaciones
        # combo_scores: [N, k, k]
        combo_scores = top1_val.unsqueeze(2) + top2_val.unsqueeze(1)
        combo_scores_flat = combo_scores.view(N_tokens, -1)
        p_combo = F.softmax(combo_scores_flat, dim=-1).view(N_tokens, self.top_k, self.top_k)

        # Pesos marginales de cada mitad
        w1 = p_combo.sum(dim=2) # [N, k]
        w2 = p_combo.sum(dim=1) # [N, k]

        # 3. Recuperación composicional de micro-expertos
        # Gather de matrices de expertos para cada token:
        # down1: [n_keys, D, rank], top1_idx: [N, k] -> [N, k, D, rank]
        down1_sel = self.down1[top1_idx] # [N, k, D, r]
        up1_sel = self.up1[top1_idx]     # [N, k, r, D]
        down2_sel = self.down2[top2_idx] # [N, k, D, r]
        up2_sel = self.up2[top2_idx]     # [N, k, r, D]

        # Aplicar Expert 1:
        # h_flat: [N, D] -> expandido para einsum: [N, 1, D]
        # x @ down1_sel: [N, k, r]
        z1 = F.silu(torch.einsum("nd,nkdr->nkr", h_flat, down1_sel))
        e1 = torch.einsum("nkr,nkrd->nkd", z1, up1_sel) # [N, k, D]

        # Aplicar Expert 2:
        z2 = F.silu(torch.einsum("nd,nkdr->nkr", h_flat, down2_sel))
        e2 = torch.einsum("nkr,nkrd->nkd", z2, up2_sel) # [N, k, D]

        # Suma ponderada de la memoria recuperada
        m = torch.einsum("nk,nkd->nd", w1, e1) + torch.einsum("nk,nkd->nd", w2, e2) # [N, D]

        # 4. Compresor compartido (bottleneck)
        comp = self.W_o2(F.silu(self.W_o1(self.comp_norm(m))))
        m_final = m + comp

        # 5. Compuerta y salida
        gate = F.silu(self.W_gate(h_flat))
        out = self.W_out(gate * m_final)
        out = self.dropout(out)

        out = out.view(orig_shape)
        
        metrics = {
            "aux_loss": aux_loss,
            "loss_balance": loss_balance.detach(),
            "mean_entropy": (- (mean_usage_1 * torch.log(mean_usage_1 + 1e-9)).sum() / math.log(self.n_keys)).detach()
        }
        self.last_aux_loss = aux_loss
        self.last_entropy = metrics["mean_entropy"].item()

        return out, metrics
