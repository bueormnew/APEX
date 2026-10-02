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
import os
import tempfile
import random
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
    device_ids: Optional[List[int]] = None
    mixed_precision: str = "no"  # "no", "fp16", "bf16"
    eval_every: int = 100
    save_checkpoint_path: Optional[str] = None
    training_checkpoint_path: Optional[str] = None
    checkpoint_every: int = 0
    resume_checkpoint_path: Optional[str] = None
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
        self.execution_model = self.model
        if (
            self.device.type == "cuda"
            and config.device_ids
            and len(config.device_ids) > 1
        ):
            if self.device.index not in config.device_ids:
                raise ValueError("device must be the primary device in device_ids")
            self.execution_model = nn.DataParallel(
                self.model, device_ids=config.device_ids, output_device=self.device.index
            )
        self.optimizer = self._build_optimizer()
        self.scheduler = None  # Se inicializará cuando se conozcan los pasos totales
        self._effective_total_steps = 0
        if config.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError("mixed_precision must be 'no', 'fp16', or 'bf16'")
        if config.mixed_precision != "no" and self.device.type != "cuda":
            raise ValueError("mixed precision training currently requires CUDA")
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=self.device.type == "cuda" and config.mixed_precision == "fp16"
        )
        
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
        self.execution_model.train()
        self.optimizer.zero_grad(set_to_none=True)

        input_ids = input_ids.to(self.device)
        labels = labels.to(self.device)

        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(self.config.mixed_precision)
        autocast = (
            torch.autocast(device_type="cuda", dtype=dtype)
            if dtype is not None
            else torch.autocast(device_type=self.device.type, enabled=False)
        )
        with autocast:
            out = self.execution_model(input_ids=input_ids, labels=labels, return_dict=True)
        lm_loss = out["lm_loss"].mean() if out.get("lm_loss") is not None else out["loss"].mean()
        aux_loss = out.get("aux_loss", torch.tensor(0.0, device=self.device))
        if isinstance(aux_loss, torch.Tensor):
            aux_loss = aux_loss.mean()

        total_loss = lm_loss + self.config.aux_loss_weight * aux_loss
        self.scaler.scale(total_loss).backward()

        # Gradient Clipping
        if self.config.max_grad_norm > 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

        self.scaler.step(self.optimizer)
        self.scaler.update()
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
        if total_steps <= 0:
            raise ValueError("Training must include at least one optimization step")
        if self.config.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.config.resume_checkpoint_path and not os.path.isfile(self.config.resume_checkpoint_path):
            raise FileNotFoundError(
                f"Resume checkpoint does not exist: {self.config.resume_checkpoint_path}"
            )
        self._effective_total_steps = total_steps
        self.scheduler = self._build_scheduler(total_steps)

        global_step = 0
        start_epoch = 0
        resume_batch = 0
        best_val_loss = float("inf")
        if self.config.resume_checkpoint_path and os.path.exists(self.config.resume_checkpoint_path):
            progress = self.load_training_checkpoint(self.config.resume_checkpoint_path)
            global_step = progress["step"]
            if total_steps > progress.get("total_steps", total_steps):
                self._extend_scheduler(global_step, total_steps)
            start_epoch, resume_batch = divmod(global_step, max(1, len(train_dataloader)))
        last_epoch = start_epoch

        print(f"=== Iniciando Entrenamiento APEX ===")
        print(f"Dispositivo: {self.device} | Parámetros entrenables: {self.model.trainable_parameters():,}")
        print(f"Épocas: {self.config.epochs} | Pasos totales: {total_steps} | LR: {self.config.learning_rate}")

        start_time = time.time()
        for epoch in range(start_epoch, self.config.epochs):
            if global_step >= total_steps:
                break
            last_epoch = epoch
            epoch_loss = 0.0
            for batch_idx, batch in enumerate(train_dataloader):
                if epoch == start_epoch and batch_idx < resume_batch:
                    continue
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

                if (
                    val_dataloader is not None
                    and self.config.eval_every > 0
                    and global_step % self.config.eval_every == 0
                ):
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

                if (
                    self.config.training_checkpoint_path
                    and self.config.checkpoint_every > 0
                    and global_step % self.config.checkpoint_every == 0
                ):
                    self.save_training_checkpoint(
                        self.config.training_checkpoint_path, global_step, epoch
                    )

                if global_step >= total_steps:
                    break
            if global_step >= total_steps:
                break

        total_duration = time.time() - start_time
        print(f"=== Entrenamiento APEX completado en {total_duration:.2f}s ===")
        if self.config.training_checkpoint_path:
            self.save_training_checkpoint(self.config.training_checkpoint_path, global_step, last_epoch)
        return self.history

    def _extend_scheduler(self, current_step: int, total_steps: int):
        """Continue an extended run from its saved LR without jumping back up."""
        current_lr = self.optimizer.param_groups[0]["lr"]
        start_scale = current_lr / self.config.learning_rate
        target_scale = self.config.min_learning_rate / self.config.learning_rate
        remaining_steps = max(1, total_steps - current_step)

        def lr_lambda(step: int) -> float:
            progress = min(max((step - current_step) / remaining_steps, 0.0), 1.0)
            decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return target_scale + (start_scale - target_scale) * decay

        self.scheduler.lr_lambdas = [lr_lambda for _ in self.optimizer.param_groups]

    def save_training_checkpoint(self, file_path: str, step: int, epoch: int) -> str:
        """Atomically save model and optimizer state for resuming a training run."""
        destination = os.path.abspath(file_path)
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "scaler": self.scaler.state_dict(),
            "step": step,
            "epoch": epoch,
            "total_steps": self._effective_total_steps,
            "history": self.history,
            "config": self.model.config.to_dict(),
            "training_config": {
                "learning_rate": self.config.learning_rate,
                "min_learning_rate": self.config.min_learning_rate,
                "weight_decay": self.config.weight_decay,
                "max_grad_norm": self.config.max_grad_norm,
                "warmup_steps": self.config.warmup_steps,
                "lr_scheduler_type": self.config.lr_scheduler_type,
                "aux_loss_weight": self.config.aux_loss_weight,
                "optimizer_type": self.config.optimizer_type,
                "mixed_precision": self.config.mixed_precision,
            },
            "python_rng_state": random.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        fd, temporary = tempfile.mkstemp(
            prefix="apex-checkpoint-", suffix=".pt", dir=os.path.dirname(destination)
        )
        os.close(fd)
        try:
            torch.save(checkpoint, temporary)
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.remove(temporary)
        return destination

    def load_training_checkpoint(self, file_path: str) -> Dict[str, int]:
        """Restore model, optimizer, scheduler, scaler and progress from a trusted local checkpoint."""
        checkpoint = torch.load(file_path, map_location=self.device, weights_only=False)
        if checkpoint["config"] != self.model.config.to_dict():
            raise ValueError("Checkpoint model configuration does not match the current model")
        training_keys = (
            "learning_rate", "min_learning_rate", "weight_decay", "max_grad_norm",
            "warmup_steps", "lr_scheduler_type", "aux_loss_weight", "optimizer_type",
            "mixed_precision",
        )
        expected_training = {
            key: getattr(self.config, key) for key in training_keys
        }
        if checkpoint.get("training_config", expected_training) != expected_training:
            raise ValueError("Checkpoint training hyperparameters do not match the current run")
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scheduler is not None and checkpoint["scheduler"] is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.scaler.load_state_dict(checkpoint.get("scaler", {}))
        self.history = checkpoint.get("history", self.history)
        if "python_rng_state" in checkpoint:
            random.setstate(checkpoint["python_rng_state"])
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if torch.cuda.is_available() and checkpoint.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng_state"]])
        return {
            "step": int(checkpoint["step"]),
            "epoch": int(checkpoint["epoch"]),
            "total_steps": int(checkpoint.get("total_steps", 0)),
        }

    @torch.no_grad()
    def evaluate(self, val_dataloader: DataLoader) -> Dict[str, float]:
        """Calcula pérdida de validación y perplejidad."""
        was_training = self.model.training
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

            out = self.execution_model(input_ids=input_ids, labels=labels, return_dict=True)
            loss = out["lm_loss"].mean() if out.get("lm_loss") is not None else out["loss"].mean()
            n_tokens = int((labels[..., 1:] != -100).sum().item())
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens

        self.model.train(was_training)
        avg_loss = total_loss / max(1, total_tokens)
        perplexity = math.exp(min(avg_loss, 20.0))  # Prevenir overflow en exp

        return {
            "val_loss": avg_loss,
            "perplexity": perplexity,
        }
