"""
APEX Trainer:
Motor de entrenamiento profesional para modelos APEX.
Soporta múltiples técnicas reales de entrenamiento de vanguardia:
1. Optimizador AdamW acoplado con Weight Decay desacoplado.
2. Cosine Annealing Learning Rate Schedule con Warmup lineal.
3. Gradient Clipping por norma L2 para máxima estabilidad numérica.
4. Integración nativa de Pérdida Auxiliar (Load Balancing Loss de micro-expertos ECHO).
5. Estrategias de Fine-Tuning: Ajuste completo o congelamiento selectivo (por ejemplo, fine-tuning solo de memoria asociativa ECHO o capas superiores).
"""

import time
import math
from typing import Optional, Dict, Any, List, Union, Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .model import APEXModel


@dataclass
class TrainingConfig:
    learning_rate: float = 3e-4
    min_learning_rate: float = 1e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 50
    total_steps: Optional[int] = None
    epochs: int = 3
    lr_scheduler_type: str = "cosine"  # "cosine", "linear", "constant"
    aux_loss_weight: float = 0.01  # Coeficiente para la pérdida balanceada de ECHO
    optimizer_type: str = "adamw"      # "adamw", "adam", "sgd"
    device: str = "auto"
    eval_every: int = 100
    save_checkpoint_path: Optional[str] = None
    log_interval: int = 10


class APEXTrainer:
    """
    Entrenador de alta eficiencia para APEXModel.
    """
    def __init__(
        self,
        model: APEXModel,
        config: TrainingConfig,
    ):
        self.model = model
        self.config = config

        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)

        self.model.to(self.device)
        self.optimizer = self._build_optimizer()
        self.scheduler = None  # Se inicializará cuando se conozcan los pasos totales
        
        self.history = {
            "step": [],
            "train_loss": [],
            "aux_loss": [],
            "total_loss": [],
            "learning_rate": [],
            "val_loss": [],
            "val_perplexity": [],
        }

    def _build_optimizer(self) -> torch.optim.Optimizer:
        # Separar parámetros con y sin weight decay (no aplicar decay a bias o LayerNorm)
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim <= 1 or "bias" in name or "norm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optim_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        if self.config.optimizer_type.lower() == "adamw":
            return torch.optim.AdamW(
                optim_groups,
                lr=self.config.learning_rate,
                betas=(0.9, 0.95),
                eps=1e-8,
            )
        elif self.config.optimizer_type.lower() == "adam":
            return torch.optim.Adam(
                optim_groups,
                lr=self.config.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8,
            )
        elif self.config.optimizer_type.lower() == "sgd":
            return torch.optim.SGD(
                optim_groups,
                lr=self.config.learning_rate,
                momentum=0.9,
            )
        else:
            raise ValueError(f"Optimizador no reconocido: {self.config.optimizer_type}")

    def _build_scheduler(self, total_steps: int):
        warmup_steps = self.config.warmup_steps
        min_lr = self.config.min_learning_rate
        max_lr = self.config.learning_rate

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / float(max(1, warmup_steps))
            if self.config.lr_scheduler_type == "constant":
                return 1.0
            
            # Progreso post-warmup
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            progress = min(max(progress, 0.0), 1.0)
            
            if self.config.lr_scheduler_type == "linear":
                factor = 1.0 - progress
            else:  # Cosine por defecto
                factor = 0.5 * (1.0 + math.cos(math.pi * progress))
                
            lr_scale = (min_lr + (max_lr - min_lr) * factor) / max_lr
            return lr_scale

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)

    def freeze_components(self, target: str):
        """
        Congela componentes para fine-tuning eficiente:
        - 'backbone': congela todo excepto la cabeza LM final.
        - 'mamba_only': congela LRCM y HopMix, entrenando solo los bloques recurrentes Mamba-3.
        - 'echo_only': entrena exclusivamente los bloques de memoria ECHO (reemplazo de FFN).
        """
        for param in self.model.parameters():
            param.requires_grad = True

        if target == "backbone":
            for name, param in self.model.core.named_parameters():
                if "head" not in name:
                    param.requires_grad = False
        elif target == "mamba_only":
            for name, param in self.model.core.named_parameters():
                if "mamba" not in name and "head" not in name:
                    param.requires_grad = False
        elif target == "echo_only":
            for name, param in self.model.core.named_parameters():
                if "echo" not in name and "head" not in name:
                    param.requires_grad = False

    def train_step(self, input_ids: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """Ejecuta un paso de entrenamiento con grad clipping y aux loss de ECHO."""
        self.model.train()
        self.optimizer.zero_grad()

        input_ids = input_ids.to(self.device)
        labels = labels.to(self.device)

        out = self.model(input_ids=input_ids, labels=labels, return_dict=True)
        lm_loss = out["loss"]
        aux_loss = out.get("aux_loss", torch.tensor(0.0, device=self.device))

        total_loss = lm_loss + self.config.aux_loss_weight * aux_loss
        total_loss.backward()

        # Gradient Clipping
        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        current_lr = self.optimizer.param_groups[0]["lr"]

        return {
            "lm_loss": lm_loss.item(),
            "aux_loss": aux_loss.item() if isinstance(aux_loss, torch.Tensor) else float(aux_loss),
            "total_loss": total_loss.item(),
            "lr": current_lr,
        }

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        callbacks: Optional[List[Callable]] = None,
    ) -> Dict[str, List[float]]:
        """Ciclo completo de entrenamiento multi-época."""
        total_batches = len(train_dataloader) * self.config.epochs
        total_steps = self.config.total_steps or total_batches
        self.scheduler = self._build_scheduler(total_steps)

        global_step = 0
        best_val_loss = float("inf")

        print(f"=== Iniciando Entrenamiento APEX ===")
        print(f"Dispositivo: {self.device} | Parámetros entrenables: {self.model.trainable_parameters():,}")
        print(f"Épocas: {self.config.epochs} | Pasos totales: {total_steps} | LR: {self.config.learning_rate}")

        start_time = time.time()
        for epoch in range(self.config.epochs):
            epoch_loss = 0.0
            for batch_idx, batch in enumerate(train_dataloader):
                if isinstance(batch, (tuple, list)):
                    input_ids, labels = batch[0], batch[1]
                elif isinstance(batch, dict):
                    input_ids = batch["input_ids"]
                    labels = batch.get("labels", input_ids)
                else:
                    input_ids = batch
                    labels = input_ids

                step_metrics = self.train_step(input_ids, labels)
                global_step += 1
                epoch_loss += step_metrics["total_loss"]

                self.history["step"].append(global_step)
                self.history["train_loss"].append(step_metrics["lm_loss"])
                self.history["aux_loss"].append(step_metrics["aux_loss"])
                self.history["total_loss"].append(step_metrics["total_loss"])
                self.history["learning_rate"].append(step_metrics["lr"])

                if global_step % self.config.log_interval == 0:
                    print(
                        f"Época {epoch+1}/{self.config.epochs} | Paso {global_step}/{total_steps} | "
                        f"Loss: {step_metrics['lm_loss']:.4f} (Aux ECHO: {step_metrics['aux_loss']:.4f}) | "
                        f"LR: {step_metrics['lr']:.2e}"
                    )

                if val_dataloader is not None and global_step % self.config.eval_every == 0:
                    val_metrics = self.evaluate(val_dataloader)
                    self.history["val_loss"].append(val_metrics["val_loss"])
                    self.history["val_perplexity"].append(val_metrics["perplexity"])
                    print(f"--> [Validación] Val Loss: {val_metrics['val_loss']:.4f} | Perplejidad: {val_metrics['perplexity']:.2f}")

                    if val_metrics["val_loss"] < best_val_loss and self.config.save_checkpoint_path:
                        best_val_loss = val_metrics["val_loss"]
                        self.model.save_apex(self.config.save_checkpoint_path)
                        print(f"--> Modelo guardado en {self.config.save_checkpoint_path} (Mejor Val Loss)")

                if callbacks:
                    for cb in callbacks:
                        cb(self, global_step, step_metrics)

                if global_step >= total_steps:
                    break
            if global_step >= total_steps:
                break

        total_duration = time.time() - start_time
        print(f"=== Entrenamiento APEX completado en {total_duration:.2f}s ===")
        return self.history

    @torch.no_grad()
    def evaluate(self, val_dataloader: DataLoader) -> Dict[str, float]:
        """Calcula pérdida de validación y perplejidad."""
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0

        for batch in val_dataloader:
            if isinstance(batch, (tuple, list)):
                input_ids, labels = batch[0], batch[1]
            elif isinstance(batch, dict):
                input_ids = batch["input_ids"]
                labels = batch.get("labels", input_ids)
            else:
                input_ids = batch
                labels = input_ids

            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)

            out = self.model(input_ids=input_ids, labels=labels, return_dict=True)
            loss = out["loss"]
            n_tokens = labels.numel()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens

        avg_loss = total_loss / max(1, total_tokens)
        perplexity = math.exp(min(avg_loss, 20.0))  # Prevenir overflow en exp

        return {
            "val_loss": avg_loss,
            "perplexity": perplexity,
        }
