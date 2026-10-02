"""
Hop-Mix v1: Bloque de mezcla O(log N) guiado por atención e integrado con ECHO.
Sustituye los FFNs tradicionales por ECHO para proporcionar memoria de alta capacidad y estabilidad.
Totalmente compatible con inferencia causal autoregresiva (generación token por token con cache).
"""

import math
from typing import Optional, Tuple, Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from echo import ECHO, RMSNorm


def make_geometric_hops(max_len: int, base: int = 2) -> List[int]:
    """Genera saltos geométricos fijos h_k = base^0, base^1, ..., < max_len."""
    hops = []
    h = 1
    while h < max_len:
        hops.append(h)
        h *= base
    return hops


class HopMixCache:
    """
    Caché para inferencia autoregresiva token a token en HopMix.
    Almacena m_j (memoria comprimida o completa), z_j (salience), routes y route_w.
    """
    def __init__(self, max_seq_len: int, d_mem: int, n_heads: int, r_routes: int, device: torch.device, dtype: torch.dtype):
        self.max_seq_len = max_seq_len
        self.d_mem = d_mem
        self.n_heads = n_heads
        self.r_routes = r_routes
        self.device = device
        self.dtype = dtype

        self.m = None      # [B, max_seq_len, d_mem]
        self.z = None      # [B, max_seq_len, n_heads]
        self.routes = None # [B, max_seq_len, r_routes] (long)
        self.route_w = None# [B, max_seq_len, r_routes] (float)
        self.curr_len = 0

    def append(self, m_t: torch.Tensor, z_t: torch.Tensor, routes_t: Optional[torch.Tensor] = None, route_w_t: Optional[torch.Tensor] = None):
        """
        Agrega un nuevo token en tiempo t.
        m_t: [B, 1, d_mem]
        z_t: [B, 1, n_heads]
        """
        B = m_t.shape[0]
        if self.m is None:
            self.m = torch.zeros(B, self.max_seq_len, self.d_mem, device=self.device, dtype=self.dtype)
            self.z = torch.zeros(B, self.max_seq_len, self.n_heads, device=self.device, dtype=self.dtype)
            self.routes = torch.full((B, self.max_seq_len, self.r_routes), -1, device=self.device, dtype=torch.long)
            self.route_w = torch.zeros(B, self.max_seq_len, self.r_routes, device=self.device, dtype=self.dtype)
            self.curr_len = 0

        pos = self.curr_len
        self.m[:, pos : pos + 1] = m_t
        self.z[:, pos : pos + 1] = z_t

        if routes_t is not None:
            self.routes[:, pos : pos + 1] = routes_t
        if route_w_t is not None:
            self.route_w[:, pos : pos + 1] = route_w_t

        self.curr_len += 1


class HopMix(nn.Module):
    """
    Bloque Hop-Mix con ranuras fijas geométricas O(log N) + ranuras guiadas + pointer jumps.
    """
    def __init__(
        self,
        d_model: int,
        max_seq_len: int = 4096,
        hops: Optional[List[int]] = None,
        n_routes: int = 2,
        pointer_jump: bool = True,
        n_gate_heads: int = 4,
        mem_dim: Optional[int] = None,
        use_salience: bool = True,
        n_dist_buckets: int = 24,
        init_scale: float = 1e-2,
        source_dropout: float = 0.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.hops = hops if hops is not None else make_geometric_hops(max_seq_len, base=2)
        self.K_hops = len(self.hops)
        self.n_routes = n_routes
        self.pointer_jump = pointer_jump
        self.r_guided = n_routes * (2 if pointer_jump else 1)
        self.S_slots = self.K_hops + self.r_guided
        self.H = n_gate_heads
        self.mem_dim = mem_dim if mem_dim is not None else d_model
        assert self.mem_dim % self.H == 0, "mem_dim debe ser divisible por n_gate_heads"
        self.d_h = self.mem_dim // self.H
        self.use_salience = use_salience
        self.n_dist_buckets = n_dist_buckets
        self.source_dropout = source_dropout

        # Normalización previa
        self.norm = RMSNorm(d_model, eps=eps)

        # Proyección de compresión de memoria (opcional W_down / W_up)
        if mem_dim is not None and mem_dim != d_model:
            self.W_down = nn.Linear(d_model, mem_dim, bias=False)
            self.W_up = nn.Linear(mem_dim, d_model, bias=False)
        else:
            self.W_down = None
            self.W_up = None

        # Salience: contenido fuente
        self.W_sal = nn.Linear(d_model, self.H, bias=False)

        # Compuerta: W_g produce logits [H, S] por token
        self.W_g = nn.Linear(d_model, self.H * self.S_slots, bias=False)
        self.b_gate = nn.Parameter(torch.zeros(self.H, self.S_slots))
        self.null_logit = nn.Parameter(torch.zeros(self.H))  # nu_h: sumidero nulo para estabilidad

        # Escalar de peso de ruta beta_s * log(w) para ranuras guiadas
        if self.r_guided > 0:
            self.beta_routes = nn.Parameter(torch.ones(self.r_guided) * 0.1)
            self.dist_bias = nn.Parameter(torch.zeros(self.r_guided, self.n_dist_buckets, self.H))
        else:
            self.beta_routes = None
            self.dist_bias = None

        # Factor por ranura a_s: [S, mem_dim]
        self.a_scale = nn.Parameter(torch.ones(self.S_slots, self.mem_dim))

        # Parámetro residual gamma
        self.gamma = nn.Parameter(torch.ones(d_model) * init_scale)

    def _bucket_distance(self, dist: torch.Tensor) -> torch.Tensor:
        """Asigna distancia a bucket log2."""
        clamped_dist = torch.clamp(dist, min=1)
        b = torch.clamp((torch.log2(clamped_dist.float())).long(), 0, self.n_dist_buckets - 1)
        return b

    def forward(
        self,
        x: torch.Tensor,
        routes: Optional[torch.Tensor] = None,
        route_w: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward paralelo completo sobre secuencia [B, T, D].
        """
        B, T, D = x.shape
        n = self.norm(x) # [B, T, D]

        # 1. Memoria m y salience z
        m = self.W_down(n) if self.W_down is not None else n # [B, T, d_m]
        z = self.W_sal(n) # [B, T, H]

        device = x.device
        positions = torch.arange(T, device=device) # [T]

        # 2. Construir matrices de índices y validez para cada ranura s en 0..S-1
        # Ranuras fijas: i - h_k
        src_indices = []
        valid_masks = []
        is_guided = []

        for h in self.hops:
            idx = positions - h # [T]
            v = idx >= 0
            idx_clamped = torch.clamp(idx, min=0)
            # Expand to [B, T]
            src_indices.append(idx_clamped.unsqueeze(0).expand(B, -1))
            valid_masks.append(v.unsqueeze(0).expand(B, -1))
            is_guided.append(False)

        # Ranuras guiadas de atención
        if self.n_routes > 0 and routes is not None:
            # routes: [B, T, r]
            for r_i in range(self.n_routes):
                r_idx = routes[:, :, r_i] # [B, T]
                v = (r_idx >= 0) & (r_idx <= positions.unsqueeze(0))
                r_idx_clamped = torch.clamp(r_idx, min=0)
                src_indices.append(r_idx_clamped)
                valid_masks.append(v)
                is_guided.append(True)

            if self.pointer_jump:
                for r_i in range(self.n_routes):
                    r_idx = routes[:, :, r_i]
                    v_first = (r_idx >= 0) & (r_idx <= positions.unsqueeze(0))
                    r_idx_clamped = torch.clamp(r_idx, min=0)
                    # Gather pointer jump: routes[b, r_idx_clamped, 0]
                    pj_idx = torch.gather(routes[:, :, 0], 1, r_idx_clamped)
                    v_pj = v_first & (pj_idx >= 0) & (pj_idx <= r_idx_clamped)
                    pj_idx_clamped = torch.clamp(pj_idx, min=0)
                    src_indices.append(pj_idx_clamped)
                    valid_masks.append(v_pj)
                    is_guided.append(True)
        else:
            # Si no hay rutas pasadas, rellenar ranuras guiadas como no válidas
            for _ in range(self.r_guided):
                src_indices.append(torch.zeros(B, T, dtype=torch.long, device=device))
                valid_masks.append(torch.zeros(B, T, dtype=torch.bool, device=device))
                is_guided.append(True)

        # Stacks: [B, T, S]
        all_indices = torch.stack(src_indices, dim=-1) # [B, T, S]
        all_val = torch.stack(valid_masks, dim=-1)     # [B, T, S]

        # 3. Logits de compuerta
        # W_g(n): [B, T, H * S] -> [B, T, H, S]
        query_term = self.W_g(n).view(B, T, self.H, self.S_slots)
        gate_logits = query_term + self.b_gate.unsqueeze(0).unsqueeze(0) # [B, T, H, S]

        # Añadir salience z de la fuente: gather z[b, src_idx, :]
        # z: [B, T, H], all_indices: [B, T, S]
        # expand z: [B, T, 1, H]
        z_expanded = z.unsqueeze(2).expand(-1, -1, self.S_slots, -1)
        # gather across sequence dimension T:
        indices_for_gather = all_indices.unsqueeze(-1).expand(-1, -1, -1, self.H) # [B, T, S, H]
        z_src = torch.gather(z.unsqueeze(2).expand(-1, -1, self.S_slots, -1), 1, indices_for_gather) # [B, T, S, H]
        z_src = z_src.permute(0, 1, 3, 2) # [B, T, H, S]
        if self.use_salience:
            gate_logits = gate_logits + z_src

        # Añadir beta * log(w) y sesgo por distancia en ranuras guiadas
        if self.r_guided > 0 and routes is not None and route_w is not None:
            # route_w: [B, T, r]
            for g_i in range(self.r_guided):
                slot_idx = self.K_hops + g_i
                src_i = all_indices[:, :, slot_idx] # [B, T]
                dist = torch.clamp(positions.unsqueeze(0) - src_i, min=0)
                b_dist = self._bucket_distance(dist) # [B, T]
                # bias: [B, T, H]
                # dist_bias[g_i]: [buckets, H]
                d_b = self.dist_bias[g_i, b_dist] # [B, T, H]
                gate_logits[:, :, :, slot_idx] += d_b

                # log weight si es ranura directa
                if g_i < self.n_routes and route_w is not None:
                    w = torch.clamp(route_w[:, :, g_i], min=1e-8)
                    gate_logits[:, :, :, slot_idx] += (self.beta_routes[g_i] * torch.log(w)).unsqueeze(-1)

        # Enmascarar ranuras no válidas con valor finito grande (-1e30) para evitar NaNs
        mask_val = all_val.unsqueeze(2).expand(-1, -1, self.H, -1) # [B, T, H, S]
        gate_logits = torch.where(mask_val, gate_logits, torch.tensor(-1e30, device=device, dtype=gate_logits.dtype))

        # Softmax con el sumidero nulo nu_h incorporado
        # Cat nu_h como la ranura S+1: [B, T, H, S + 1]
        nu = self.null_logit.view(1, 1, self.H, 1).expand(B, T, -1, -1)
        full_logits = torch.cat([gate_logits, nu], dim=-1)
        full_gates = F.softmax(full_logits, dim=-1)
        gates = full_gates[:, :, :, :self.S_slots] # [B, T, H, S]

        # 4. Mezcla ponderada de m
        # m: [B, T, d_m] dividida en H grupos de d_h
        m_grouped = m.view(B, T, self.H, self.d_h)
        a_grouped = self.a_scale.view(self.S_slots, self.H, self.d_h) # [S, H, d_h]

        # Gather de m para cada ranura
        # indices_m: [B, T, S, H, d_h]
        indices_m = all_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.H, self.d_h)
        m_expanded = m_grouped.unsqueeze(2).expand(-1, -1, self.S_slots, -1, -1)
        m_src = torch.gather(m_expanded, 1, indices_m) # [B, T, S, H, d_h]

        # Escalar a_s * m_src: [B, T, S, H, d_h]
        scaled_m = m_src * a_grouped.unsqueeze(0).unsqueeze(0)

        # Ponderar con compuerta g: [B, T, H, S] -> [B, T, S, H, 1]
        g_expanded = gates.permute(0, 1, 3, 2).unsqueeze(-1)
        y = torch.sum(g_expanded * scaled_m, dim=2) # [B, T, H, d_h]
        y = y.view(B, T, self.mem_dim)

        # Proyección de retorno si hubo mem_dim
        if self.W_up is not None:
            y = self.W_up(y)

        # Conexión residual escalada con gamma
        out = x + self.gamma * y
        return out

    def step(
        self,
        x_t: torch.Tensor,
        cache: HopMixCache,
        routes_t: Optional[torch.Tensor] = None,
        route_w_t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Paso incremental autoregresivo de un único token (decoding eficiente O(S)).
        x_t: [B, 1, D]
        """
        B, _, D = x_t.shape
        t = cache.curr_len
        device = x_t.device

        n_t = self.norm(x_t)
        m_t = self.W_down(n_t) if self.W_down is not None else n_t # [B, 1, d_m]
        z_t = self.W_sal(n_t) # [B, 1, H]

        cache.append(m_t, z_t, routes_t, route_w_t)

        # Evaluar ranuras en tiempo t
        src_indices = []
        valid_masks = []
        for h in self.hops:
            src = t - h
            valid = (src >= 0)
            src_indices.append(max(0, src))
            valid_masks.append(valid)

        if self.n_routes > 0 and routes_t is not None:
            for r_i in range(self.n_routes):
                r_idx = routes_t[:, 0, r_i].item() if routes_t.dim() == 3 else routes_t[0, r_i].item()
                v = (r_idx >= 0) and (r_idx <= t)
                src_indices.append(max(0, r_idx))
                valid_masks.append(v)

            if self.pointer_jump:
                for r_i in range(self.n_routes):
                    r_idx = routes_t[:, 0, r_i].item() if routes_t.dim() == 3 else routes_t[0, r_i].item()
                    if 0 <= r_idx <= t and cache.routes is not None:
                        pj_idx = cache.routes[0, r_idx, 0].item()
                        v = (pj_idx >= 0) and (pj_idx <= r_idx)
                        src_indices.append(max(0, pj_idx))
                        valid_masks.append(v)
                    else:
                        src_indices.append(0)
                        valid_masks.append(False)
        else:
            for _ in range(self.r_guided):
                src_indices.append(0)
                valid_masks.append(False)

        # Logits de compuerta en tiempo t
        query = self.W_g(n_t).view(B, 1, self.H, self.S_slots)
        gate_logits = query + self.b_gate.view(1, 1, self.H, self.S_slots)

        for s_idx, (src, val) in enumerate(zip(src_indices, valid_masks)):
            if not val:
                gate_logits[:, :, :, s_idx] = -1e30
            else:
                if self.use_salience:
                    gate_logits[:, :, :, s_idx] += cache.z[:, src : src + 1, :]

        nu = self.null_logit.view(1, 1, self.H, 1).expand(B, 1, -1, -1)
        full_logits = torch.cat([gate_logits, nu], dim=-1)
        full_gates = F.softmax(full_logits, dim=-1)
        gates = full_gates[:, :, :, :self.S_slots] # [B, 1, H, S]

        # Acumular mezcla
        y = torch.zeros(B, 1, self.H, self.d_h, device=device, dtype=x_t.dtype)
        a_grouped = self.a_scale.view(self.S_slots, self.H, self.d_h)
        for s_idx, (src, val) in enumerate(zip(src_indices, valid_masks)):
            if val:
                m_src = cache.m[:, src : src + 1].view(B, 1, self.H, self.d_h)
                g_s = gates[:, :, :, s_idx : s_idx + 1] # [B, 1, H, 1]
                y += g_s * (a_grouped[s_idx].unsqueeze(0).unsqueeze(0) * m_src)

        y = y.view(B, 1, self.mem_dim)
        if self.W_up is not None:
            y = self.W_up(y)

        out_t = x_t + self.gamma * y
        return out_t


class HopMixECHOBlock(nn.Module):
    """
    Bloque Transformer de salto O(log N) que integra HopMix para mezcla de secuencias
    y ECHO de manera nativa como reemplazo total y estable del bloque FFN.
    """
    def __init__(
        self,
        d_model: int,
        max_seq_len: int = 4096,
        n_routes: int = 2,
        pointer_jump: bool = True,
        n_gate_heads: int = 4,
        echo_n_keys: int = 64,
        echo_top_k: int = 4,
        echo_rank: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.mixer = HopMix(
            d_model=d_model,
            max_seq_len=max_seq_len,
            n_routes=n_routes,
            pointer_jump=pointer_jump,
            n_gate_heads=n_gate_heads,
        )
        self.echo_norm = RMSNorm(d_model)
        self.echo = ECHO(
            d_model=d_model,
            n_keys=echo_n_keys,
            top_k=echo_top_k,
            rank=echo_rank,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        routes: Optional[torch.Tensor] = None,
        route_w: Optional[torch.Tensor] = None,
        return_aux_loss: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # 1. Mezcla de contexto causal multiescala O(log N)
        h = self.mixer(x, routes=routes, route_w=route_w)
        # 2. Bloque de memoria asociativa factorizada ECHO (en lugar de FFN)
        echo_out, metrics = self.echo(self.echo_norm(h), return_aux_loss=return_aux_loss)
        out = h + echo_out
        return out, metrics

    def step(
        self,
        x_t: torch.Tensor,
        cache: HopMixCache,
        routes_t: Optional[torch.Tensor] = None,
        route_w_t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Paso incremental autoregresivo del bloque completo HopMix + ECHO."""
        h_t = self.mixer.step(x_t, cache, routes_t=routes_t, route_w_t=route_w_t)
        echo_out, _ = self.echo(self.echo_norm(h_t))
        return h_t + echo_out
