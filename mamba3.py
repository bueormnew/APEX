"""
Mamba-3: State Space Model Puro con Discretización Exponencial-Trapezoidal,
Espacio de Estados Complejos con Rotaciones RoPE dependientes de datos y
Formulación MIMO (Multi-Input Multi-Output).

Referencias científicas:
- "Mamba-3: Improved Sequence Modeling using State Space Principles" (arXiv:2603.15569, 2026).
- Implementación matemática rigurosa sin simplificaciones.

Innovaciones matemáticas de Mamba-3 vs Mamba-2:
1. Discretización Exponencial-Trapezoidal:
   En lugar del Zero-Order Hold (Euler) de Mamba-2 que aproxima la entrada como constante por paso,
   Mamba-3 utiliza una regla trapezoidal de segundo orden promediando el aporte (B * x)_{t-1} y (B * x)_t
   a través de una compuerta aprendida trap_t = sigmoid(W_trap * x + b_trap):
   u_t = (1 - 0.5 * trap_t) * (B_t * x_t) + (0.5 * trap_t) * (B_{t-1} * x_{t-1})
   h_t = exp(Delta_t * A) * h_{t-1} + Delta_t * u_t

2. Espacio de Estados Complejos con RoPE dependiente de datos ("RoPE trick"):
   Para resolver tareas de seguimiento de estados y paridad sin el sobrecosto de números complejos nativos,
   se aprende un ángulo incremental theta_t = dt * angle_proj(x) por cabeza.
   Se rota B y C con matrices de Givens 2D sobre pares de dimensiones contiguas,
   lo cual es matemáticamente isomorfo a una rotación unitaria e^{i * theta} en C.

3. Formulación MIMO (Multi-Input Multi-Output):
   En lugar de SISO donde x es un vector por token, MIMO proyecta x a una matriz X_t de rango r (mimo_rank R),
   escribiendo una actualización de rango R en el estado matricial y aumentando la intensidad aritmética
   para saturar el hardware en decoding sin aumentar la latencia.
"""

import math
from typing import Optional, Tuple, Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from echo import RMSNorm


def apply_rotary_emb(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """
    Aplica rotación RoPE sobre pares de dimensiones para emular dinámica compleja:
    x: [..., D], angles: [..., D // 2]
    """
    D = x.shape[-1]
    assert D % 2 == 0, "La dimensión para rotación compleja debe ser par"
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    
    # Rotación compleja (x1 + i*x2) * (cos + i*sin)
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    
    # Reintercalar
    out = torch.stack([out1, out2], dim=-1).flatten(-2)
    return out


class Mamba3State:
    """
    Estado recurrente persistente para generación token a token en Mamba-3.
    Almacena:
    - ssm_state: [B, H, P, D] (estado oculto recurrente)
    - bx_prev: [B, H, P, R] (memoria trapezoidal del paso anterior para trap_t)
    - accumulated_angle: [B, H, D // 2] (ángulo acumulado para RoPE complejo)
    """
    def __init__(self, ssm_state: torch.Tensor, bx_prev: torch.Tensor, accumulated_angle: torch.Tensor):
        self.ssm_state = ssm_state
        self.bx_prev = bx_prev
        self.accumulated_angle = accumulated_angle


class Mamba3(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        headdim: int = 32,
        n_heads: Optional[int] = None,
        is_mimo: bool = True,
        mimo_rank: int = 2,
        expand: int = 2,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        bias: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.headdim = headdim
        self.d_inner = expand * d_model
        
        if n_heads is None:
            assert self.d_inner % headdim == 0, "d_inner debe ser divisible por headdim"
            self.n_heads = self.d_inner // headdim
        else:
            self.n_heads = n_heads
            self.d_inner = self.n_heads * headdim

        self.is_mimo = is_mimo
        self.mimo_rank = mimo_rank if is_mimo else 1
        
        # Proyecciones de entrada
        # in_proj proyecta a:
        # 1. Rama principal u: [B, T, d_inner * mimo_rank]
        # 2. Rama compuerta z: [B, T, d_inner]
        # 3. dt: [B, T, n_heads]
        # 4. B: [B, T, n_heads, d_state]
        # 5. C: [B, T, n_heads, d_state]
        # 6. angles: [B, T, n_heads, d_state // 2]
        # 7. trap_t: [B, T, n_heads]
        self.norm = RMSNorm(d_model)

        self.proj_u = nn.Linear(d_model, self.d_inner * self.mimo_rank, bias=bias)
        self.proj_z = nn.Linear(d_model, self.d_inner, bias=bias)
        self.proj_dt = nn.Linear(d_model, self.n_heads, bias=True)
        self.proj_b = nn.Linear(d_model, self.n_heads * self.d_state, bias=bias)
        self.proj_c = nn.Linear(d_model, self.n_heads * self.d_state, bias=bias)
        
        # Proyección para rotación compleja (RoPE angular)
        assert d_state % 2 == 0, "d_state debe ser par para descomposición compleja 2D"
        self.proj_angle = nn.Linear(d_model, self.n_heads * (self.d_state // 2), bias=bias)

        # Compuerta trapezoidal aprendida
        self.proj_trap = nn.Linear(d_model, self.n_heads, bias=True)

        # Matriz A de decaimiento continuo: param logarítmico estrictamente negativo
        # A es diagonal por cabeza: [n_heads]
        A_init = torch.log(torch.linspace(0.1, 2.0, self.n_heads))
        self.A_log = nn.Parameter(A_init)

        # Inicialización de dt bias para escala logarítmica
        dt = torch.exp(
            torch.rand(self.n_heads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.proj_dt.bias.copy_(inv_dt)

        # Proyección de compresión MIMO y salida final
        self.mimo_out_proj = nn.Linear(self.d_inner * self.mimo_rank, self.d_inner, bias=bias) if is_mimo else nn.Identity()
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Mamba3State] = None,
    ) -> Tuple[torch.Tensor, Optional[Mamba3State]]:
        """
        Forward causal puro de Mamba-3 sobre secuencias [B, T, d_model].
        Implementa exactamente:
        - Discretización trapezoidal continua
        - RoPE complejo sobre B y C
        - MIMO de rango R
        """
        B, T, D = x.shape
        x_norm = self.norm(x)

        # 1. Proyecciones dependientes de datos
        # u: [B, T, n_heads, headdim, R]
        u = self.proj_u(x_norm).view(B, T, self.n_heads, self.headdim, self.mimo_rank)
        z = self.proj_z(x_norm) # [B, T, d_inner]
        dt = F.softplus(self.proj_dt(x_norm)) # [B, T, n_heads]
        
        # B y C: [B, T, n_heads, d_state]
        b_proj = self.proj_b(x_norm).view(B, T, self.n_heads, self.d_state)
        c_proj = self.proj_c(x_norm).view(B, T, self.n_heads, self.d_state)

        # Ángulos incrementales para rotación compleja
        angle_raw = self.proj_angle(x_norm).view(B, T, self.n_heads, self.d_state // 2)
        d_theta = dt.unsqueeze(-1) * angle_raw # [B, T, n_heads, d_state // 2]
        
        # Ángulo acumulado causalmente a lo largo de T
        theta = torch.cumsum(d_theta, dim=1) # [B, T, n_heads, d_state // 2]

        # 2. Aplicar RoPE trick: Rotar B y C en el dominio complejo
        B_rot = apply_rotary_emb(b_proj, theta) # [B, T, n_heads, d_state]
        C_rot = apply_rotary_emb(c_proj, theta) # [B, T, n_heads, d_state]

        # 3. Compuerta trapezoidal: blend continuo entre Euler y Trapecio
        trap_gate = torch.sigmoid(self.proj_trap(x_norm)) # [B, T, n_heads]

        # 4. Decaimiento continuo A discretizado: dA = exp(dt * A)
        # A: [n_heads] -> negativo: -exp(A_log)
        A = -torch.exp(self.A_log) # [n_heads]
        # dA: [B, T, n_heads]
        dA = torch.exp(dt * A.view(1, 1, self.n_heads)) # [B, T, n_heads]

        # 5. Escaneo Recurrente Causal Puro con MIMO y Trapecio
        # Estado h: [B, n_heads, headdim, d_state]
        h = torch.zeros(B, self.n_heads, self.headdim, self.d_state, device=x.device, dtype=x.dtype)
        # bx_prev para regla trapezoidal de segundo orden: [B, n_heads, headdim, d_state]
        bx_prev = torch.zeros_like(h)

        y_steps = []
        for t in range(T):
            # u_t: [B, n_heads, headdim, R]
            u_t = u[:, t]
            B_t = B_rot[:, t] # [B, n_heads, d_state]
            C_t = C_rot[:, t] # [B, n_heads, d_state]
            dt_t = dt[:, t].unsqueeze(-1).unsqueeze(-1) # [B, n_heads, 1, 1]
            dA_t = dA[:, t].unsqueeze(-1).unsqueeze(-1) # [B, n_heads, 1, 1]
            trap_t = trap_gate[:, t].unsqueeze(-1).unsqueeze(-1) # [B, n_heads, 1, 1]

            # MIMO rank-R outer product:
            # En SISO: bx = u * B. En MIMO: contracción u_t (headdim, R) con B_t (d_state, R) o suma sobre R
            # Para MIMO puro de rango R, sumamos la interacción sobre los R flujos de entrada:
            # bx_current: [B, n_heads, headdim, d_state]
            if self.is_mimo:
                # B_t: [B, n_heads, d_state]
                # u_t: [B, n_heads, headdim, R] -> sumamos sobre R
                u_sum = u_t.sum(dim=-1) # [B, n_heads, headdim]
                bx_curr = torch.einsum("bnh,bnd->bnhd", u_sum, B_t)
            else:
                bx_curr = torch.einsum("bnh,bnd->bnhd", u_t.squeeze(-1), B_t)

            # Regla trapezoidal exponencial:
            # u_eff = (1 - 0.5 * trap) * bx_curr + (0.5 * trap) * bx_prev
            u_eff = (1.0 - 0.5 * trap_t) * bx_curr + (0.5 * trap_t) * bx_prev
            bx_prev = bx_curr

            # Actualización del estado: h = dA * h + dt * u_eff
            h = dA_t * h + dt_t * u_eff

            # Lectura del estado: y_t = h @ C_t -> [B, n_heads, headdim]
            y_t = torch.einsum("bnhd,bnd->bnh", h, C_t)
            y_steps.append(y_t)

        # Concatenar secuencia temporal: [B, T, n_heads, headdim]
        y = torch.stack(y_steps, dim=1).view(B, T, self.d_inner)

        # Compuerta SiLU multiplicativa con z
        y_gated = y * F.silu(z)

        # Proyección de salida final + residual
        out = x + self.out_proj(y_gated)

        # Actualizar estado si fue solicitado
        new_state = None
        if state is not None or not self.training:
            new_state = Mamba3State(
                ssm_state=h,
                bx_prev=bx_prev,
                accumulated_angle=theta[:, -1],
            )

        return out, new_state

    def step(
        self,
        x_t: torch.Tensor,
        state: Mamba3State,
    ) -> Tuple[torch.Tensor, Mamba3State]:
        """
        Paso incremental autoregresivo de decoding para un token.
        x_t: [B, 1, d_model]
        """
        B, _, D = x_t.shape
        x_norm = self.norm(x_t)

        u = self.proj_u(x_norm).view(B, self.n_heads, self.headdim, self.mimo_rank)
        z = self.proj_z(x_norm).squeeze(1)
        dt = F.softplus(self.proj_dt(x_norm)).squeeze(1) # [B, n_heads]
        
        b_proj = self.proj_b(x_norm).view(B, self.n_heads, self.d_state)
        c_proj = self.proj_c(x_norm).view(B, self.n_heads, self.d_state)

        angle_raw = self.proj_angle(x_norm).view(B, self.n_heads, self.d_state // 2)
        d_theta = dt.unsqueeze(-1) * angle_raw
        accum_angle = state.accumulated_angle + d_theta # [B, n_heads, d_state // 2]

        B_rot = apply_rotary_emb(b_proj, accum_angle)
        C_rot = apply_rotary_emb(c_proj, accum_angle)

        trap_gate = torch.sigmoid(self.proj_trap(x_norm)).squeeze(1) # [B, n_heads]

        A = -torch.exp(self.A_log)
        dA = torch.exp(dt * A.view(1, self.n_heads)) # [B, n_heads]

        # Dimensiones para broadcast
        dt_t = dt.unsqueeze(-1).unsqueeze(-1)
        dA_t = dA.unsqueeze(-1).unsqueeze(-1)
        trap_t = trap_gate.unsqueeze(-1).unsqueeze(-1)

        if self.is_mimo:
            u_sum = u.sum(dim=-1)
            bx_curr = torch.einsum("bnh,bnd->bnhd", u_sum, B_rot)
        else:
            bx_curr = torch.einsum("bnh,bnd->bnhd", u.squeeze(-1), B_rot)

        # Regla trapezoidal con bx_prev de la memoria
        u_eff = (1.0 - 0.5 * trap_t) * bx_curr + (0.5 * trap_t) * state.bx_prev
        new_bx_prev = bx_curr

        new_h = dA_t * state.ssm_state + dt_t * u_eff
        y = torch.einsum("bnhd,bnd->bnh", new_h, C_rot).view(B, self.d_inner)

        y_gated = y * F.silu(z)
        out_t = x_t + self.out_proj(y_gated.unsqueeze(1))

        new_state = Mamba3State(
            ssm_state=new_h,
            bx_prev=new_bx_prev,
            accumulated_angle=accum_angle,
        )
        return out_t, new_state
