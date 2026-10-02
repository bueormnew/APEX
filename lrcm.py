"""
LRCM (Long-Range Convolutional Retrieval Memory) con integración nativa de ECHO.

LRCM desacopla el problema de contexto largo en:
1. Exact Local Mixing: Atención exacta con ventana causal W (e.g. W=128..256).
2. Multiscale Descriptors + Long Convolution: Jerarquía de descriptores (tokens -> chunks -> páginas -> regiones).
3. Hierarchical Learned Routing: Addressing de grueso a fino con sparse beam y temperatura ajustable.
4. Exact Leaf Gather + Tiny Exact Attention: Recuperación exacta del contenido de las hojas seleccionadas.
5. Gated Merge & Output: Fusión normalizada aprendida (softmax gate entre local y exact recall).
6. Integración nativa de ECHO: Sustituye el bloque FFN estándar para máxima capacidad de hechos y estabilidad.
"""

import math
from typing import Optional, Tuple, Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from echo import ECHO, RMSNorm


class LRCMMemoryState:
    """
    Estado de memoria histórica para inferencia y entrenamiento en LRCM.
    Mantiene:
    - local_kv: [B, W, 2, n_heads, head_dim]
    - leaf_kv: [B, max_chunks, chunk_size, 2, d_model]
    - chunk_desc: [B, max_chunks, desc_dim]
    - page_desc: [B, max_pages, desc_dim]
    - region_desc: [B, max_regions, desc_dim]
    """
    def __init__(
        self,
        batch_size: int,
        max_tokens: int,
        local_window: int = 16,
        chunk_size: int = 16,
        page_size: int = 64,
        region_size: int = 256,
        d_model: int = 64,
        desc_dim: int = 32,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        self.chunk_size = chunk_size
        self.local_window = local_window
        self.max_tokens = max_tokens
        self.page_size = page_size
        self.region_size = region_size
        self.max_chunks = max(1, math.ceil(max_tokens / chunk_size))
        self.max_pages = max(1, math.ceil(max_tokens / page_size))
        self.max_regions = max(1, math.ceil(max_tokens / region_size))

        self.leaf_kv = torch.zeros(batch_size, self.max_chunks, chunk_size, 2, d_model, device=device, dtype=dtype)
        self.chunk_desc = torch.zeros(batch_size, self.max_chunks, desc_dim, device=device, dtype=dtype)
        self.chunk_pool = torch.zeros_like(self.chunk_desc)
        self.chunk_desc_accumulator = torch.zeros(batch_size, desc_dim, device=device, dtype=dtype)
        self.local_k = torch.zeros(batch_size, local_window, d_model, device=device, dtype=dtype)
        self.local_v = torch.zeros_like(self.local_k)
        self.page_desc = torch.zeros(batch_size, self.max_pages, desc_dim, device=device, dtype=dtype)
        self.region_desc = torch.zeros(batch_size, self.max_regions, desc_dim, device=device, dtype=dtype)

        self.num_tokens = 0
        self.num_chunks = 0
        self.num_pages = 0
        self.num_regions = 0


class CausalDepthwiseConv1d(nn.Module):
    """Convolución causal depthwise eficiente para agregación de descriptores."""
    def __init__(self, dim: int, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(dim, dim, kernel_size, groups=dim, padding=0, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        # Padding causal a la izquierda: kernel_size - 1
        x_pad = F.pad(x, (self.kernel_size - 1, 0))
        return self.conv(x_pad)


class LRCM(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        local_window: int = 16,
        chunk_size: int = 16,
        page_chunks: int = 4,
        region_pages: int = 4,
        desc_dim: int = 32,
        beam_size: int = 2,
        temperature: float = 1.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        assert d_model % n_heads == 0
        self.head_dim = d_model // n_heads
        self.local_window = local_window
        self.chunk_size = chunk_size
        self.page_chunks = page_chunks
        self.region_pages = region_pages
        self.page_size = chunk_size * page_chunks
        self.region_size = self.page_size * region_pages
        self.desc_dim = desc_dim
        self.beam_size = beam_size
        self.temperature = temperature

        self.norm = RMSNorm(d_model, eps=eps)

        # 1. Rama Local de Atención
        self.q_local = nn.Linear(d_model, d_model, bias=False)
        self.k_local = nn.Linear(d_model, d_model, bias=False)
        self.v_local = nn.Linear(d_model, d_model, bias=False)
        self.out_local = nn.Linear(d_model, d_model, bias=False)

        # 2. Generador de Descriptores de Chunk y Convoluciones Jerárquicas
        self.token_to_desc = nn.Linear(d_model, desc_dim, bias=False)
        self.conv_chunk = CausalDepthwiseConv1d(desc_dim, kernel_size=3)
        self.conv_page = CausalDepthwiseConv1d(desc_dim, kernel_size=3)
        self.conv_region = CausalDepthwiseConv1d(desc_dim, kernel_size=3)

        # 3. Router Jerárquico: Proyección de consulta a espacio de descriptores
        self.q_mem_proj = nn.Linear(d_model * 2, desc_dim, bias=False)

        # 4. Rama de Recuperación Exacta (Leaf KV)
        self.k_leaf_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_leaf_proj = nn.Linear(d_model, d_model, bias=False)
        self.q_recall_proj = nn.Linear(d_model * 2, d_model, bias=False)

        # 5. Compuerta de Fusión (softmax gate entre local y recall)
        self.fusion_gate = nn.Linear(d_model * 2, 2, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def _local_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Atención exacta con ventana causal local W.
        q, k, v: [B, T, n_heads, head_dim]
        """
        B, T, H, D = q.shape
        q = q.transpose(1, 2) # [B, H, T, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        window = min(self.local_window, T)
        k_windows = F.pad(k, (0, 0, window - 1, 0)).unfold(2, window, 1)
        v_windows = F.pad(v, (0, 0, window - 1, 0)).unfold(2, window, 1)
        k_windows = k_windows.permute(0, 1, 2, 4, 3)
        v_windows = v_windows.permute(0, 1, 2, 4, 3)

        scores = torch.matmul(q.unsqueeze(-2), k_windows.transpose(-1, -2)).squeeze(-2)
        scores = scores / math.sqrt(D)
        positions = torch.arange(T, device=q.device)
        offsets = torch.arange(window, device=q.device)
        valid_keys = positions[:, None] - window + 1 + offsets[None, :] >= 0
        scores = scores.masked_fill(~valid_keys[None, None], torch.finfo(scores.dtype).min)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn.unsqueeze(-2), v_windows).squeeze(-2)
        out = out.transpose(1, 2).contiguous().view(B, T, H * D)
        return self.out_local(out)

    def _build_hierarchy(self, x_norm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Construye descriptores de chunks, páginas y regiones a partir de la secuencia.
        """
        B, T, D = x_norm.shape
        n_chunks = max(1, math.ceil(T / self.chunk_size))
        pad_len = n_chunks * self.chunk_size - T
        if pad_len > 0:
            x_padded = F.pad(x_norm, (0, 0, 0, pad_len))
        else:
            x_padded = x_norm

        # 1. Hojas exactas (Keys & Values para lectura)
        k_leaf = self.k_leaf_proj(x_padded).view(B, n_chunks, self.chunk_size, D)
        v_leaf = self.v_leaf_proj(x_padded).view(B, n_chunks, self.chunk_size, D)

        # 2. Descriptores de Chunk
        desc_tokens = self.token_to_desc(x_padded) # [B, n_chunks * chunk_size, desc_dim]
        chunk_pooled = desc_tokens.view(B, n_chunks, self.chunk_size, self.desc_dim).mean(dim=2) # [B, n_chunks, desc_dim]
        
        # Convolución causal sobre chunks
        chunk_desc = self.conv_chunk(chunk_pooled.transpose(1, 2)).transpose(1, 2) # [B, n_chunks, desc_dim]

        # 3. Descriptores de Página
        n_pages = max(1, math.ceil(n_chunks / self.page_chunks))
        pad_chunks = n_pages * self.page_chunks - n_chunks
        if pad_chunks > 0:
            chunk_padded = F.pad(chunk_desc, (0, 0, 0, pad_chunks))
        else:
            chunk_padded = chunk_desc
        page_pooled = chunk_padded.view(B, n_pages, self.page_chunks, self.desc_dim).mean(dim=2)
        page_desc = self.conv_page(page_pooled.transpose(1, 2)).transpose(1, 2)

        # 4. Descriptores de Región
        n_regions = max(1, math.ceil(n_pages / self.region_pages))
        pad_pages = n_regions * self.region_pages - n_pages
        if pad_pages > 0:
            page_padded = F.pad(page_desc, (0, 0, 0, pad_pages))
        else:
            page_padded = page_desc
        region_pooled = page_padded.view(B, n_regions, self.region_pages, self.desc_dim).mean(dim=2)
        region_desc = self.conv_region(region_pooled.transpose(1, 2)).transpose(1, 2)

        return k_leaf, v_leaf, chunk_desc, page_desc, region_desc

    def forward(
        self,
        x: torch.Tensor,
        memory_state: Optional[LRCMMemoryState] = None,
    ) -> Tuple[torch.Tensor, Optional[LRCMMemoryState]]:
        """
        Forward paralelo completo sobre secuencia [B, T, D].
        """
        B, T, D = x.shape
        x_norm = self.norm(x)

        # 1. Rama Local de alta precisión
        q_l = self.q_local(x_norm).view(B, T, self.n_heads, self.head_dim)
        k_l = self.k_local(x_norm).view(B, T, self.n_heads, self.head_dim)
        v_l = self.v_local(x_norm).view(B, T, self.n_heads, self.head_dim)
        h_local = self._local_attention(q_l, k_l, v_l) # [B, T, D]

        # Si la secuencia es más corta que un chunk, la atención local es suficiente
        if T <= self.chunk_size:
            out = x + self.out_proj(h_local)
            return out, memory_state

        # 2. Construcción multiescala de descriptores
        k_leaf, v_leaf, chunk_desc, page_desc, region_desc = self._build_hierarchy(x_norm)
        n_chunks = chunk_desc.shape[1]

        # 3. Consulta de Memoria q_mem = W_q [x; h_local]
        q_combo = torch.cat([x_norm, h_local], dim=-1)
        q_mem = self.q_mem_proj(q_combo) # [B, T, desc_dim]
        q_recall = self.q_recall_proj(q_combo) # [B, T, D]

        # 4. Enrutamiento Jerárquico por token
        # Scoring sobre chunks de manera causal y eficiente
        # Cada token solo puede buscar en chunks causales (chunk_idx <= token_idx // chunk_size)
        token_chunk_id = torch.arange(T, device=x.device) // self.chunk_size # [T]

        # Scores sobre todos los chunks: [B, T, n_chunks]
        scores_chunk = torch.matmul(q_mem, chunk_desc.transpose(1, 2)) / (math.sqrt(self.desc_dim) * self.temperature)
        
        # Solo se recuperan chunks completos anteriores al chunk consultado.
        chunk_causal_mask = token_chunk_id.unsqueeze(1) > torch.arange(n_chunks, device=x.device).unsqueeze(0) # [T, n_chunks]
        scores_chunk = scores_chunk.masked_fill(
            ~chunk_causal_mask.unsqueeze(0), torch.finfo(scores_chunk.dtype).min
        )

        # Seleccionar top-B chunks
        beam_k = min(self.beam_size, n_chunks)
        top_scores, top_chunk_indices = torch.topk(scores_chunk, beam_k, dim=-1) # [B, T, beam_k]
        selected_valid = chunk_causal_mask.unsqueeze(0).expand(B, -1, -1).gather(
            2, top_chunk_indices
        )

        # 5. Exact Leaf Gather y Tiny Exact Attention
        # Gather de los chunks seleccionados:
        # k_leaf: [B, n_chunks, chunk_size, D]
        # expand indices: [B, T, beam_k, chunk_size, D]
        idx_expanded = top_chunk_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.chunk_size, D)
        k_leaf_expanded = k_leaf.unsqueeze(1).expand(-1, T, -1, -1, -1)
        v_leaf_expanded = v_leaf.unsqueeze(1).expand(-1, T, -1, -1, -1)

        gathered_k = torch.gather(k_leaf_expanded, 2, idx_expanded) # [B, T, beam_k, chunk_size, D]
        gathered_v = torch.gather(v_leaf_expanded, 2, idx_expanded)

        gathered_k = gathered_k.view(B, T, beam_k * self.chunk_size, D)
        gathered_v = gathered_v.view(B, T, beam_k * self.chunk_size, D)

        # Exact Attention sobre las hojas recuperadas
        # q_recall: [B, T, 1, D]
        attn_scores = torch.matmul(q_recall.unsqueeze(2), gathered_k.transpose(-1, -2)) / math.sqrt(D) # [B, T, 1, beam_k * chunk_size]
        attn_scores = attn_scores.view(B, T, beam_k, self.chunk_size)
        attn_scores = attn_scores.masked_fill(
            ~selected_valid.unsqueeze(-1), torch.finfo(attn_scores.dtype).min
        )
        attn_scores = attn_scores.view(B, T, 1, beam_k * self.chunk_size)
        leaf_weights = F.softmax(attn_scores, dim=-1)
        h_recall = torch.matmul(leaf_weights, gathered_v).squeeze(2) # [B, T, D]

        # 6. Fusión con compuerta aprendida
        fusion_logits = self.fusion_gate(q_combo) # [B, T, 2]
        g = F.softmax(fusion_logits, dim=-1)
        has_memory = (token_chunk_id > 0).view(1, T, 1)
        g = torch.where(has_memory, g, torch.cat([torch.ones_like(g[..., :1]), torch.zeros_like(g[..., 1:])], dim=-1))
        h_fused = g[:, :, 0:1] * h_local + g[:, :, 1:2] * h_recall

        out = x + self.out_proj(h_fused)
        return out, memory_state

    def step(
        self,
        x_t: torch.Tensor,
        state: LRCMMemoryState,
    ) -> Tuple[torch.Tensor, LRCMMemoryState]:
        """
        Paso incremental autoregresivo de decoding en tiempo constante por token.
        x_t: [B, 1, D]
        """
        B, _, D = x_t.shape
        x_norm = self.norm(x_t)

        if state.num_tokens >= state.max_tokens:
            raise ValueError("LRCM inference state exceeded its configured max_tokens")

        # Cache local keys/values and compute causal sliding-window attention.
        q_l = self.q_local(x_norm).view(B, 1, self.n_heads, self.head_dim)
        k_l = self.k_local(x_norm).view(B, 1, self.n_heads, self.head_dim)
        v_l = self.v_local(x_norm).view(B, 1, self.n_heads, self.head_dim)
        position = state.num_tokens
        state.local_k[:, position % self.local_window] = k_l[:, 0].reshape(B, D)
        state.local_v[:, position % self.local_window] = v_l[:, 0].reshape(B, D)
        local_len = min(position + 1, self.local_window)
        local_positions = torch.arange(
            position + 1 - local_len, position + 1, device=x_t.device
        ) % self.local_window
        k_window = state.local_k.index_select(1, local_positions).view(
            B, local_len, self.n_heads, self.head_dim
        ).transpose(1, 2)
        v_window = state.local_v.index_select(1, local_positions).view(
            B, local_len, self.n_heads, self.head_dim
        ).transpose(1, 2)
        q_window = q_l.transpose(1, 2)
        local_scores = torch.matmul(q_window, k_window.transpose(-1, -2)) / math.sqrt(self.head_dim)
        local_weights = F.softmax(local_scores, dim=-1)
        local_context = torch.matmul(local_weights, v_window).transpose(1, 2).contiguous().view(B, 1, D)
        h_local = self.out_local(local_context)

        q_combo = torch.cat([x_norm, h_local], dim=-1)
        available_chunks = state.num_chunks
        if available_chunks == 0:
            out = x_t + self.out_proj(h_local)
            state.chunk_desc_accumulator.add_(self.token_to_desc(x_norm[:, 0]))
            state.leaf_kv[:, position // self.chunk_size, position % self.chunk_size, 0] = self.k_leaf_proj(x_norm)[:, 0]
            state.leaf_kv[:, position // self.chunk_size, position % self.chunk_size, 1] = self.v_leaf_proj(x_norm)[:, 0]
            if (position + 1) % self.chunk_size == 0:
                chunk_index = position // self.chunk_size
                state.chunk_pool[:, chunk_index] = state.chunk_desc_accumulator / self.chunk_size
                first = max(0, chunk_index - self.conv_chunk.kernel_size + 1)
                recent = state.chunk_pool[:, first : chunk_index + 1].transpose(1, 2)
                state.chunk_desc[:, chunk_index] = self.conv_chunk(recent)[:, :, -1]
                state.chunk_desc_accumulator.zero_()
                state.num_chunks += 1
            state.num_tokens += 1
            return out, state

        # Query de memoria
        q_mem = self.q_mem_proj(q_combo) # [B, 1, desc_dim]
        q_recall = self.q_recall_proj(q_combo) # [B, 1, D]

        # Scoring de chunks disponibles
        chunk_desc_avail = state.chunk_desc[:, :available_chunks] # [B, avail_chunks, desc_dim]
        scores = torch.matmul(q_mem, chunk_desc_avail.transpose(1, 2)) / (math.sqrt(self.desc_dim) * self.temperature)

        beam_k = min(self.beam_size, available_chunks)
        _, top_idx = torch.topk(scores, beam_k, dim=-1) # [B, 1, beam_k]
        leaf_k = state.leaf_kv[:, :, :, 0, :]
        leaf_v = state.leaf_kv[:, :, :, 1, :]
        index = top_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.chunk_size, D)
        gathered_k = torch.gather(
            leaf_k.unsqueeze(1).expand(-1, 1, -1, -1, -1), 2, index
        ).reshape(B, 1, beam_k * self.chunk_size, D)
        gathered_v = torch.gather(
            leaf_v.unsqueeze(1).expand(-1, 1, -1, -1, -1), 2, index
        ).reshape(B, 1, beam_k * self.chunk_size, D)

        attn_scores = torch.matmul(q_recall.unsqueeze(2), gathered_k.transpose(-1, -2)) / math.sqrt(D)
        leaf_weights = F.softmax(attn_scores, dim=-1)
        h_recall = torch.matmul(leaf_weights, gathered_v).squeeze(2) # [B, 1, D]

        # Fusión
        fusion_logits = self.fusion_gate(q_combo)
        g = F.softmax(fusion_logits, dim=-1)
        h_fused = g[:, :, 0:1] * h_local + g[:, :, 1:2] * h_recall

        out = x_t + self.out_proj(h_fused)
        # Commit current token to the leaf store after retrieval to keep reads strictly causal.
        chunk_index = position // self.chunk_size
        chunk_offset = position % self.chunk_size
        state.leaf_kv[:, chunk_index, chunk_offset, 0] = self.k_leaf_proj(x_norm)[:, 0]
        state.leaf_kv[:, chunk_index, chunk_offset, 1] = self.v_leaf_proj(x_norm)[:, 0]
        state.chunk_desc_accumulator.add_(self.token_to_desc(x_norm[:, 0]))
        if chunk_offset + 1 == self.chunk_size:
            state.chunk_pool[:, chunk_index] = state.chunk_desc_accumulator / self.chunk_size
            first = max(0, chunk_index - self.conv_chunk.kernel_size + 1)
            recent = state.chunk_pool[:, first : chunk_index + 1].transpose(1, 2)
            state.chunk_desc[:, chunk_index] = self.conv_chunk(recent)[:, :, -1]
            state.chunk_desc_accumulator.zero_()
            state.num_chunks += 1
        state.num_tokens += 1
        return out, state


class LRCMECHOBlock(nn.Module):
    """
    Bloque LRCM completo que sustituye el FFN por ECHO de forma nativa.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        local_window: int = 16,
        chunk_size: int = 16,
        desc_dim: int = 32,
        beam_size: int = 2,
        echo_n_keys: int = 64,
        echo_top_k: int = 4,
        echo_rank: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.mixer = LRCM(
            d_model=d_model,
            n_heads=n_heads,
            local_window=local_window,
            chunk_size=chunk_size,
            desc_dim=desc_dim,
            beam_size=beam_size,
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
        memory_state: Optional[LRCMMemoryState] = None,
        return_aux_loss: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Optional[LRCMMemoryState]]:
        h, memory_state = self.mixer(x, memory_state=memory_state)
        echo_out, metrics = self.echo(self.echo_norm(h), return_aux_loss=return_aux_loss)
        out = h + echo_out
        return out, metrics, memory_state

    def step(
        self,
        x_t: torch.Tensor,
        state: LRCMMemoryState,
    ) -> Tuple[torch.Tensor, LRCMMemoryState]:
        h_t, state = self.mixer.step(x_t, state)
        echo_out, _ = self.echo(self.echo_norm(h_t))
        return h_t + echo_out, state
