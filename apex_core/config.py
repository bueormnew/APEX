"""
APEX Configuration and Block Presets.
Provee 3 maquetas / presets de bloques profesionales:
1. APEX_PYRAMID (Topología ganadora: HOP -> LRCM -> MAMBA -> MAMBA -> LRCM -> HOP)
2. APEX_OMNI (Topología asimétrica: LRCM -> HOP -> MAMBA -> MAMBA -> LRCM -> MAMBA)
3. APEX_DEEP_MAMBA (Topología intensiva en razonamiento: LRCM -> MAMBA -> MAMBA -> MAMBA -> HOP -> LRCM)
"""

from typing import List, Optional, Dict, Any
from enum import Enum


class BlockPreset(str, Enum):
    APEX_PYRAMID = "apex_pyramid"
    APEX_OMNI = "apex_omni"
    APEX_DEEP_MAMBA = "apex_deep_mamba"


PRESET_PATTERNS: Dict[BlockPreset, List[str]] = {
    BlockPreset.APEX_PYRAMID: ["hopmix", "lrcm", "mamba3", "mamba3", "lrcm", "hopmix"],
    BlockPreset.APEX_OMNI: ["lrcm", "hopmix", "mamba3", "mamba3", "lrcm", "mamba3"],
    BlockPreset.APEX_DEEP_MAMBA: ["lrcm", "mamba3", "mamba3", "mamba3", "hopmix", "lrcm"],
}


class APEXConfig:
    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 256,
        max_seq_len: int = 2048,
        preset: Optional[BlockPreset] = BlockPreset.APEX_PYRAMID,
        layer_pattern: Optional[List[str]] = None,
        # Configuración nativa ECHO (en reemplazo total de FFN)
        echo_n_keys: int = 32,
        echo_top_k: int = 4,
        echo_rank: int = 16,
        echo_balance_coef: float = 0.01,
        # Configuración Hop-Mix
        hop_routes: int = 2,
        hop_gate_heads: int = 4,
        hop_pointer_jump: bool = True,
        # Configuración LRCM
        lrcm_heads: int = 4,
        lrcm_local_window: int = 32,
        lrcm_chunk_size: int = 16,
        lrcm_desc_dim: int = 32,
        lrcm_beam: int = 2,
        # Configuración Mamba-3 (Puro: Discretización Trapezoidal + RoPE Complejo + MIMO)
        mamba_d_state: int = 32,
        mamba_headdim: int = 32,
        mamba_mimo_rank: int = 2,
        mamba_is_mimo: bool = True,
        dropout: float = 0.0,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.preset = preset

        if layer_pattern is not None:
            self.layer_pattern = [p.lower() for p in layer_pattern]
        elif preset in PRESET_PATTERNS:
            self.layer_pattern = list(PRESET_PATTERNS[preset])
        else:
            self.layer_pattern = list(PRESET_PATTERNS[BlockPreset.APEX_PYRAMID])

        # Parámetros ECHO
        self.echo_n_keys = echo_n_keys
        self.echo_top_k = echo_top_k
        self.echo_rank = echo_rank
        self.echo_balance_coef = echo_balance_coef

        # Parámetros Hop-Mix
        self.hop_routes = hop_routes
        self.hop_gate_heads = hop_gate_heads
        self.hop_pointer_jump = hop_pointer_jump

        # Parámetros LRCM
        self.lrcm_heads = lrcm_heads
        self.lrcm_local_window = lrcm_local_window
        self.lrcm_chunk_size = lrcm_chunk_size
        self.lrcm_desc_dim = lrcm_desc_dim
        self.lrcm_beam = lrcm_beam

        # Parámetros Mamba-3
        self.mamba_d_state = mamba_d_state
        self.mamba_headdim = mamba_headdim
        self.mamba_mimo_rank = mamba_mimo_rank
        self.mamba_is_mimo = mamba_is_mimo

        self.dropout = dropout

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "max_seq_len": self.max_seq_len,
            "preset": self.preset.value if isinstance(self.preset, BlockPreset) else str(self.preset),
            "layer_pattern": self.layer_pattern,
            "echo_n_keys": self.echo_n_keys,
            "echo_top_k": self.echo_top_k,
            "echo_rank": self.echo_rank,
            "echo_balance_coef": self.echo_balance_coef,
            "hop_routes": self.hop_routes,
            "hop_gate_heads": self.hop_gate_heads,
            "hop_pointer_jump": self.hop_pointer_jump,
            "lrcm_heads": self.lrcm_heads,
            "lrcm_local_window": self.lrcm_local_window,
            "lrcm_chunk_size": self.lrcm_chunk_size,
            "lrcm_desc_dim": self.lrcm_desc_dim,
            "lrcm_beam": self.lrcm_beam,
            "mamba_d_state": self.mamba_d_state,
            "mamba_headdim": self.mamba_headdim,
            "mamba_mimo_rank": self.mamba_mimo_rank,
            "mamba_is_mimo": self.mamba_is_mimo,
            "dropout": self.dropout,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "APEXConfig":
        preset_val = d.get("preset")
        if preset_val in [p.value for p in BlockPreset]:
            preset_enum = BlockPreset(preset_val)
        else:
            preset_enum = None

        return cls(
            vocab_size=d.get("vocab_size", 50257),
            d_model=d.get("d_model", 256),
            max_seq_len=d.get("max_seq_len", 2048),
            preset=preset_enum,
            layer_pattern=d.get("layer_pattern"),
            echo_n_keys=d.get("echo_n_keys", 32),
            echo_top_k=d.get("echo_top_k", 4),
            echo_rank=d.get("echo_rank", 16),
            echo_balance_coef=d.get("echo_balance_coef", 0.01),
            hop_routes=d.get("hop_routes", 2),
            hop_gate_heads=d.get("hop_gate_heads", 4),
            hop_pointer_jump=d.get("hop_pointer_jump", True),
            lrcm_heads=d.get("lrcm_heads", 4),
            lrcm_local_window=d.get("lrcm_local_window", 32),
            lrcm_chunk_size=d.get("lrcm_chunk_size", 16),
            lrcm_desc_dim=d.get("lrcm_desc_dim", 32),
            lrcm_beam=d.get("lrcm_beam", 2),
            mamba_d_state=d.get("mamba_d_state", 32),
            mamba_headdim=d.get("mamba_headdim", 32),
            mamba_mimo_rank=d.get("mamba_mimo_rank", 2),
            mamba_is_mimo=d.get("mamba_is_mimo", True),
            dropout=d.get("dropout", 0.0),
        )
