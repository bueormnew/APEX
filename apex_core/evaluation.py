"""
APEX Evaluation & Benchmarking Suite:
Provee herramientas profesionales para evaluar modelos APEX:
1. `MetricsMonitor`: Mide throughput (tokens/segundo), latencia por token (ms/token) y uso de memoria (VRAM/RAM).
2. `BenchmarkHarness`: Evalúa capacidad de respuesta ante pares Pregunta-Respuesta (QA), Needle-in-a-Haystack y Exact Match.
"""

import time
import os
from typing import List, Dict, Any, Optional, Callable, Union
import torch
import torch.nn as nn

from .model import APEXModel


class MetricsMonitor:
    """
    Monitor de rendimiento y métricas de hardware para modelos APEX.
    """
    @staticmethod
    def get_memory_info(device: torch.device) -> Dict[str, float]:
        """Retorna consumo de memoria en MB (VRAM si CUDA, RSS si CPU)."""
        if device.type == "cuda":
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
            reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
            return {"allocated_mb": allocated, "reserved_mb": reserved}
        else:
            try:
                import psutil
                process = psutil.Process(os.getpid())
                rss_mb = process.memory_info().rss / (1024 ** 2)
                return {"allocated_mb": rss_mb, "reserved_mb": rss_mb}
            except ImportError:
                return {"allocated_mb": 0.0, "reserved_mb": 0.0}

    @classmethod
    def benchmark_throughput(
        cls,
        model: APEXModel,
        seq_len: int = 128,
        batch_size: int = 4,
        num_batches: int = 10,
        warmup_batches: int = 3,
        device: Union[str, torch.device] = "auto",
    ) -> Dict[str, Any]:
        """
        Mide el throughput de procesamiento en tokens/segundo tanto en Forward como en Forward+Backward.
        """
        if device == "auto":
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            dev = torch.device(device)

        model.to(dev)
        dummy_inputs = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=dev)
        dummy_labels = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=dev)

        # Warmup
        model.eval()
        with torch.no_grad():
            for _ in range(warmup_batches):
                _ = model(dummy_inputs)

        if dev.type == "cuda":
            torch.cuda.synchronize()

        # Medición de Forward Throughput
        start_time = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_batches):
                _ = model(dummy_inputs)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        forward_time = time.perf_counter() - start_time
        total_tokens = batch_size * seq_len * num_batches
        forward_tok_sec = total_tokens / forward_time

        # Medición de Forward + Backward (Training Throughput)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        for _ in range(warmup_batches):
            optimizer.zero_grad()
            out = model(dummy_inputs, labels=dummy_labels)
            out["loss"].backward()
            optimizer.step()

        if dev.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()
        for _ in range(num_batches):
            optimizer.zero_grad()
            out = model(dummy_inputs, labels=dummy_labels)
            out["loss"].backward()
            optimizer.step()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        train_time = time.perf_counter() - start_time
        train_tok_sec = total_tokens / train_time

        mem_info = cls.get_memory_info(dev)

        return {
            "device": str(dev),
            "batch_size": batch_size,
            "seq_len": seq_len,
            "forward_tokens_per_sec": forward_tok_sec,
            "train_tokens_per_sec": train_tok_sec,
            "forward_latency_ms": (forward_time / num_batches) * 1000.0,
            "memory_mb": mem_info["allocated_mb"],
        }

    @classmethod
    def benchmark_generation(
        cls,
        model: APEXModel,
        prompt_len: int = 16,
        gen_tokens: int = 32,
        device: Union[str, torch.device] = "auto",
    ) -> Dict[str, Any]:
        """
        Mide la velocidad de generación token por token autorregresiva.
        """
        if device == "auto":
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            dev = torch.device(device)

        model.to(dev)
        prompt = torch.randint(0, model.config.vocab_size, (1, prompt_len), device=dev)

        # Warmup
        _ = model.generate(prompt, max_new_tokens=4)
        if dev.type == "cuda":
            torch.cuda.synchronize()

        # Medición
        start_time = time.perf_counter()
        _ = model.generate(prompt, max_new_tokens=gen_tokens)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        gen_time = time.perf_counter() - start_time

        tok_per_sec = gen_tokens / gen_time
        ms_per_tok = (gen_time / gen_tokens) * 1000.0

        return {
            "device": str(dev),
            "prompt_length": prompt_len,
            "generated_tokens": gen_tokens,
            "generation_time_sec": gen_time,
            "generation_tokens_per_sec": tok_per_sec,
            "latency_ms_per_token": ms_per_tok,
        }


class BenchmarkHarness:
    """
    Arnés de evaluación para evaluar capacidades de razonamiento,
    recuperación de información (Needle) y evaluación estándar de Pregunta/Respuesta (QA).
    """
    def __init__(self, model: APEXModel):
        self.model = model

    def evaluate_qa(
        self,
        qa_pairs: List[Dict[str, Any]],
        tokenizer_fn: Optional[Callable[[str], torch.Tensor]] = None,
        detokenizer_fn: Optional[Callable[[torch.Tensor], str]] = None,
        max_new_tokens: int = 16,
        match_mode: str = "contains",
    ) -> Dict[str, Any]:
        """
        Evalúa un conjunto de preguntas y respuestas.
        Cada elemento en qa_pairs debe tener las llaves:
        - 'question': texto o tensor con la entrada
        - 'target': texto o tensor esperado
        
        match_mode:
        - 'exact': coincidencia exacta estricta
        - 'contains': la respuesta generada contiene el objetivo
        """
        self.model.eval()
        device = next(self.model.parameters()).device

        total = len(qa_pairs)
        correct = 0
        details = []

        for item in qa_pairs:
            q = item["question"]
            expected = item["target"]

            # Tokenización
            if isinstance(q, str):
                if tokenizer_fn is None:
                    # Tokenizador simple por caracteres de respaldo si no se proporciona uno externo
                    q_tokens = torch.tensor([[ord(c) % self.model.config.vocab_size for c in q]], device=device)
                else:
                    q_tokens = tokenizer_fn(q).to(device)
            elif isinstance(q, torch.Tensor):
                q_tokens = q.to(device)
                if q_tokens.ndim == 1:
                    q_tokens = q_tokens.unsqueeze(0)
            else:
                q_tokens = torch.tensor([q], device=device)

            # Generar respuesta
            generated_seq = self.model.generate(
                prompt_tokens=q_tokens,
                max_new_tokens=max_new_tokens,
                temperature=0.0,  # Greedy
            )
            # Extraer solo los tokens nuevos generados
            new_tokens = generated_seq[0, q_tokens.shape[1]:]

            # Decodificar
            if isinstance(expected, str):
                if detokenizer_fn is None:
                    gen_text = "".join([chr(t.item()) for t in new_tokens if 32 <= t.item() <= 126])
                else:
                    gen_text = detokenizer_fn(new_tokens)

                is_match = False
                if match_mode == "exact":
                    is_match = gen_text.strip().lower() == expected.strip().lower()
                elif match_mode == "contains":
                    is_match = expected.strip().lower() in gen_text.lower()
                
                details.append({
                    "question": q,
                    "expected": expected,
                    "generated": gen_text,
                    "is_correct": is_match,
                })
            else:
                # Comparación basada en tensores
                expected_tensor = torch.tensor(expected, device=device)
                min_len = min(new_tokens.shape[0], expected_tensor.shape[0])
                is_match = (new_tokens[:min_len] == expected_tensor[:min_len]).all().item()
                details.append({
                    "question": str(q),
                    "expected": expected_tensor.tolist(),
                    "generated": new_tokens.tolist(),
                    "is_correct": is_match,
                })

            if is_match:
                correct += 1

        accuracy = (correct / max(1, total)) * 100.0
        return {
            "total_items": total,
            "correct_items": correct,
            "accuracy_percent": accuracy,
            "details": details,
        }

    def evaluate_needle_recall(
        self,
        haystack_size: int = 256,
        needle_token: int = 777,
        query_token: int = 888,
        depth_ratio: float = 0.5,
    ) -> bool:
        """
        Prueba sintética de aguja en el pajar:
        Inserta `needle_token` en una posición oculta del contexto, y luego solicita recordar ese valor con `query_token`.
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        vocab = self.model.config.vocab_size

        # Secuencia base
        seq = torch.randint(10, vocab - 10, (1, haystack_size), device=device)
        needle_pos = int(haystack_size * depth_ratio)
        seq[0, needle_pos] = needle_token
        seq[0, -1] = query_token

        gen = self.model.generate(prompt_tokens=seq, max_new_tokens=1, temperature=0.0)
        predicted_token = gen[0, -1].item()

        return predicted_token == needle_token
