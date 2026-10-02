"""
Hybrid causal autoregressive language model integrating:
- LRCM (con memoria convolucional de largo alcance + lectura de hojas exactas + ECHO nativo)
- Hop-Mix (con saltos multiescala O(log N) + rutas guiadas + ECHO nativo)
- Mamba-3 (SSM de segundo orden con discretización trapezoidal, RoPE complejo y MIMO)

Soporta secuencias configurables como:
['lrcm', 'hopmix', 'mamba3', 'mamba3', 'hopmix', 'lrcm']
Entrenamiento end-to-end con next-token loss causal + inferencia autoregresiva generativa token a token (.generate()).
"""

import math
from typing import Optional, Tuple, Dict, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from echo import ECHO, RMSNorm
from hopmix import HopMix, HopMixECHOBlock, HopMixCache
from lrcm import LRCM, LRCMECHOBlock, LRCMMemoryState
from mamba3 import Mamba3, Mamba3State


class HybridLMConfig:
    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 256,
        max_seq_len: int = 2048,
        layer_pattern: Optional[List[str]] = None,
        # Configuración ECHO
        echo_n_keys: int = 32,
        echo_top_k: int = 4,
        echo_rank: int = 16,
        echo_balance_coef: float = 0.01,
        # Configuración HopMix
        hop_routes: int = 2,
        hop_gate_heads: int = 4,
        hop_pointer_jump: bool = True,
        # Configuración LRCM
        lrcm_heads: int = 4,
        lrcm_local_window: int = 32,
        lrcm_chunk_size: int = 16,
        lrcm_desc_dim: int = 32,
        lrcm_beam: int = 2,
        # Configuración Mamba-3
        mamba_d_state: int = 32,
        mamba_headdim: int = 32,
        mamba_mimo_rank: int = 2,
        mamba_is_mimo: bool = True,
        dropout: float = 0.0,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        # Patrón híbrido sugerido: LRCM -> HOP -> MAMBA -> MAMBA -> HOP -> LRCM
        self.layer_pattern = layer_pattern or ["lrcm", "hopmix", "mamba3", "mamba3", "hopmix", "lrcm"]
        
        self.echo_n_keys = echo_n_keys
        self.echo_top_k = echo_top_k
        self.echo_rank = echo_rank
        self.echo_balance_coef = echo_balance_coef
        
        self.hop_routes = hop_routes
        self.hop_gate_heads = hop_gate_heads
        self.hop_pointer_jump = hop_pointer_jump

        self.lrcm_heads = lrcm_heads
        self.lrcm_local_window = lrcm_local_window
        self.lrcm_chunk_size = lrcm_chunk_size
        self.lrcm_desc_dim = lrcm_desc_dim
        self.lrcm_beam = lrcm_beam

        self.mamba_d_state = mamba_d_state
        self.mamba_headdim = mamba_headdim
        self.mamba_mimo_rank = mamba_mimo_rank
        self.mamba_is_mimo = mamba_is_mimo
        self.dropout = dropout


class HybridBlock(nn.Module):
    """
    Bloque universal unificado que envuelve cualquiera de las tres arquitecturas:
    'hopmix', 'lrcm', 'mamba3' garantizando compatibilidad end-to-end.
    """
    def __init__(self, block_type: str, config: HybridLMConfig):
        super().__init__()
        self.block_type = block_type.lower()
        self.d_model = config.d_model

        if self.block_type == "hopmix":
            self.module = HopMixECHOBlock(
                d_model=config.d_model,
                max_seq_len=config.max_seq_len,
                n_routes=config.hop_routes,
                pointer_jump=config.hop_pointer_jump,
                n_gate_heads=config.hop_gate_heads,
                echo_n_keys=config.echo_n_keys,
                echo_top_k=config.echo_top_k,
                echo_rank=config.echo_rank,
                dropout=config.dropout,
            )
        elif self.block_type == "lrcm":
            self.module = LRCMECHOBlock(
                d_model=config.d_model,
                n_heads=config.lrcm_heads,
                local_window=config.lrcm_local_window,
                chunk_size=config.lrcm_chunk_size,
                desc_dim=config.lrcm_desc_dim,
                beam_size=config.lrcm_beam,
                echo_n_keys=config.echo_n_keys,
                echo_top_k=config.echo_top_k,
                echo_rank=config.echo_rank,
                dropout=config.dropout,
            )
        elif self.block_type == "mamba3":
            self.module = Mamba3(
                d_model=config.d_model,
                d_state=config.mamba_d_state,
                headdim=config.mamba_headdim,
                is_mimo=config.mamba_is_mimo,
                mimo_rank=config.mamba_mimo_rank,
            )
        else:
            raise ValueError(f"Tipo de bloque desconocido: {block_type}")

    def forward(
        self,
        x: torch.Tensor,
        routes: Optional[torch.Tensor] = None,
        route_w: Optional[torch.Tensor] = None,
        return_aux_loss: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        if self.block_type == "hopmix":
            out, metrics = self.module(x, routes=routes, route_w=route_w, return_aux_loss=return_aux_loss)
            return out, metrics
        elif self.block_type == "lrcm":
            out, metrics, _ = self.module(x, return_aux_loss=return_aux_loss)
            return out, metrics
        elif self.block_type == "mamba3":
            out, _ = self.module(x)
            return out, None
        else:
            raise RuntimeError(f"Bloque desconocido: {self.block_type}")

    def step(
        self,
        x_t: torch.Tensor,
        layer_state: any,
        routes_t: Optional[torch.Tensor] = None,
        route_w_t: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, any]:
        if self.block_type == "hopmix":
            out = self.module.step(x_t, layer_state, routes_t=routes_t, route_w_t=route_w_t)
            return out, layer_state
        elif self.block_type == "lrcm":
            out, new_state = self.module.step(x_t, layer_state)
            return out, new_state
        elif self.block_type == "mamba3":
            out, new_state = self.module.step(x_t, layer_state)
            return out, new_state
        else:
            raise RuntimeError(f"Bloque desconocido: {self.block_type}")


class HybridCausalLM(nn.Module):
    """
    Modelo de Lenguaje Causal Autoregresivo Híbrido end-to-end con LRCM, Hop-Mix y Mamba-3.
    """
    def __init__(self, config: HybridLMConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.d_model = config.d_model

        # Embeddings de entrada
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)

        # Capas apiladas según el patrón solicitado
        self.layers = nn.ModuleList([
            HybridBlock(block_type=b_type, config=config)
            for b_type in config.layer_pattern
        ])

        # Normalización final y cabezal LM
        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying opcional entre embedding y lm_head
        self.lm_head.weight = self.embedding.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Forward paralelo completo para pretraining / fine-tuning.
        input_ids: [B, T]
        labels: [B, T] opcional para calcular CrossEntropyLoss
        """
        B, T = input_ids.shape
        x = self.embedding(input_ids) # [B, T, d_model]

        total_aux_loss = torch.tensor(0.0, device=input_ids.device)
        routes = None
        route_w = None

        # Procesar a través de la secuencia de bloques
        for layer in self.layers:
            x, metrics = layer(x, routes=routes, route_w=route_w, return_aux_loss=True)
            if metrics is not None and "aux_loss" in metrics:
                total_aux_loss = total_aux_loss + metrics["aux_loss"]

        x = self.final_norm(x)
        logits = self.lm_head(x) # [B, T, vocab_size]

        loss = None
        if labels is not None:
            # Shift causally: logits[:-1] vs labels[1:]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            ce_loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            loss = ce_loss + total_aux_loss

        if return_dict:
            return {
                "loss": loss,
                "logits": logits,
                "aux_loss": total_aux_loss,
            }
        return logits, loss

    def init_inference_states(self, batch_size: int, device: torch.device) -> List[any]:
        """Inicializa cachés/estados recurrentes para cada capa durante decoding."""
        states = []
        for layer in self.layers:
            if layer.block_type == "hopmix":
                state = HopMixCache(
                    max_seq_len=self.config.max_seq_len,
                    d_mem=self.config.d_model,
                    n_heads=self.config.hop_gate_heads,
                    r_routes=self.config.hop_routes * (2 if self.config.hop_pointer_jump else 1),
                    device=device,
                    dtype=torch.float32,
                )
            elif layer.block_type == "lrcm":
                state = LRCMMemoryState(
                    batch_size=batch_size,
                    max_tokens=self.config.max_seq_len,
                    chunk_size=self.config.lrcm_chunk_size,
                    d_model=self.config.d_model,
                    desc_dim=self.config.lrcm_desc_dim,
                    device=device,
                )
            elif layer.block_type == "mamba3":
                state = Mamba3State(
                    ssm_state=torch.zeros(
                        batch_size,
                        layer.module.n_heads,
                        layer.module.headdim,
                        layer.module.d_state,
                        device=device,
                    ),
                    bx_prev=torch.zeros(
                        batch_size,
                        layer.module.n_heads,
                        layer.module.headdim,
                        layer.module.d_state,
                        device=device,
                    ),
                    accumulated_angle=torch.zeros(
                        batch_size,
                        layer.module.n_heads,
                        layer.module.d_state // 2,
                        device=device,
                    ),
                )
            states.append(state)
        return states

    def step(
        self,
        token_id_t: torch.Tensor,
        states: List[any],
    ) -> Tuple[torch.Tensor, List[any]]:
        """
        Paso de generación de un token [B, 1].
        """
        x_t = self.embedding(token_id_t) # [B, 1, d_model]
        new_states = []

        routes_t = None
        route_w_t = None

        for idx, layer in enumerate(self.layers):
            state = states[idx]
            x_t, new_state = layer.step(x_t, state, routes_t=routes_t, route_w_t=route_w_t)
            new_states.append(new_state)

        x_t = self.final_norm(x_t)
        logits_t = self.lm_head(x_t) # [B, 1, vocab_size]
        return logits_t, new_states

    @torch.no_grad()
    def generate(
        self,
        prompt_tokens: torch.Tensor,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:
        """
        Generación autoregresiva token a token.
        """
        self.eval()
        B, T = prompt_tokens.shape
        device = prompt_tokens.device
        states = self.init_inference_states(batch_size=B, device=device)

        # Prellenar estados con el prompt
        for t in range(T - 1):
            curr_token = prompt_tokens[:, t:t+1]
            _, states = self.step(curr_token, states)

        curr_token = prompt_tokens[:, -1:]
        generated = [prompt_tokens]

        for _ in range(max_new_tokens):
            logits_t, states = self.step(curr_token, states) # [B, 1, vocab_size]
            logits_t = logits_t[:, -1, :] / max(temperature, 1e-5)

            if top_k > 0:
                vals, _ = torch.topk(logits_t, min(top_k, logits_t.shape[-1]))
                logits_t[logits_t < vals[:, -1, None]] = -float("Inf")

            probs = F.softmax(logits_t, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1) # [B, 1]
            generated.append(next_token)
            curr_token = next_token

        return torch.cat(generated, dim=1)
