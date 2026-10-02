"""
APEX Model Wrapper:
Provee la clase `APEXModel` con API orientada a objetos:
- Instanciación limpia y profesional: `APEXModel(config)` o `APEXModel.from_preset("apex_pyramid")`
- Guardado y carga en formato único unificado `.apex` (contiene pesos, configuración e hiperparámetros)
- Soporte para pretraining/fine-tuning y generación autorregresiva de alta velocidad
- Compatible al 100% con los bloques intactos de `echo`, `hopmix`, `lrcm`, `mamba3` y `hybrid_model`.
"""

import os
import io
import json
import zipfile
from typing import Optional, Dict, Any, List, Union, Tuple
import torch
import torch.nn as nn

from hybrid_model import HybridCausalLM, HybridLMConfig
from .config import APEXConfig, BlockPreset


class APEXModel(nn.Module):
    """
    Envoltorio profesional para modelos APEX.
    Permite instanciar, entrenar, exportar a `.apex` y generar texto.
    """
    def __init__(self, config: APEXConfig):
        super().__init__()
        self.config = config
        
        # Mapeo a la configuración interna probada
        self._internal_cfg = HybridLMConfig(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            max_seq_len=config.max_seq_len,
            layer_pattern=config.layer_pattern,
            echo_n_keys=config.echo_n_keys,
            echo_top_k=config.echo_top_k,
            echo_rank=config.echo_rank,
            echo_balance_coef=config.echo_balance_coef,
            hop_routes=config.hop_routes,
            hop_gate_heads=config.hop_gate_heads,
            hop_pointer_jump=config.hop_pointer_jump,
            lrcm_heads=config.lrcm_heads,
            lrcm_local_window=config.lrcm_local_window,
            lrcm_chunk_size=config.lrcm_chunk_size,
            lrcm_desc_dim=config.lrcm_desc_dim,
            lrcm_beam=config.lrcm_beam,
            mamba_d_state=config.mamba_d_state,
            mamba_headdim=config.mamba_headdim,
            mamba_mimo_rank=config.mamba_mimo_rank,
            mamba_is_mimo=config.mamba_is_mimo,
            dropout=config.dropout,
        )
        self.core = HybridCausalLM(self._internal_cfg)

    @classmethod
    def from_preset(cls, preset: Union[str, BlockPreset], **kwargs) -> "APEXModel":
        """Instancia un modelo APEX preconfigurado."""
        if isinstance(preset, str):
            preset = BlockPreset(preset.lower())
        cfg = APEXConfig(preset=preset, **kwargs)
        return cls(cfg)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
        return self.core(input_ids=input_ids, labels=labels, return_dict=return_dict)

    @torch.no_grad()
    def generate(
        self,
        prompt_tokens: torch.Tensor,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:
        """Generación autorregresiva token a token con cachés sincronizadas."""
        return self.core.generate(
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save_apex(self, file_path: str):
        """
        Exporta el modelo completo en el formato unificado `.apex`.
        El archivo .apex es un contenedor que almacena:
        - `config.json`: Configuración exacta e hiperparámetros.
        - `weights.pt`: Tensores de pesos y sesgos de la red.
        - `metadata.json`: Metadatos del framework, versión y arquitectura.
        """
        if not file_path.endswith(".apex"):
            file_path += ".apex"

        config_dict = self.config.to_dict()
        metadata = {
            "format": "APEX_CONTAINER_V1",
            "framework": "apex_core",
            "version": "0.1.0",
            "total_params": self.count_parameters(),
            "architecture": self.config.layer_pattern,
        }

        # Serializar tensores a bytes en memoria
        weights_buffer = io.BytesIO()
        torch.save(self.state_dict(), weights_buffer)
        weights_bytes = weights_buffer.getvalue()

        # Empaquetar todo en el archivo .apex
        with zipfile.ZipFile(file_path, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
            zip_file.writestr("config.json", json.dumps(config_dict, indent=2))
            zip_file.writestr("metadata.json", json.dumps(metadata, indent=2))
            zip_file.writestr("weights.pt", weights_bytes)

        return file_path

    @classmethod
    def load_apex(cls, file_path: str, device: Union[str, torch.device] = "cpu") -> "APEXModel":
        """
        Carga un modelo completo desde un archivo `.apex`.
        Restaura instantáneamente la configuración y los pesos exactos.
        """
        if not os.path.exists(file_path):
            if os.path.exists(file_path + ".apex"):
                file_path = file_path + ".apex"
            else:
                raise FileNotFoundError(f"Archivo .apex no encontrado: {file_path}")

        with zipfile.ZipFile(file_path, "r") as zip_file:
            config_str = zip_file.read("config.json").decode("utf-8")
            config_dict = json.loads(config_str)
            weights_bytes = zip_file.read("weights.pt")

        config = APEXConfig.from_dict(config_dict)
        model = cls(config)

        weights_buffer = io.BytesIO(weights_bytes)
        state_dict = torch.load(weights_buffer, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        return model
