"""
Evaluación Definitiva: Entrenamiento y Benchmark de los Dos Modelos Finales
para LLMs de Producción Reales (Rápidos, Inteligentes y de Alta Capacidad).

Candidatos Finales Seleccionados basados en la evidencia empírica:
1. Modelo 1 - 'HYBRID-APEX' (Pirámide de Fluidez y Recuperación Multiescala):
   Patrón: HOPMIX -> LRCM -> MAMBA3 -> MAMBA3 -> LRCM -> HOPMIX
   - Entrada/Salida con difusión O(log N) ultrarrápida.
   - Anclaje y lectura exacta jerárquica con LRCM + memoria asociativa ECHO.
   - Núcleo de razonamiento continuo de 2º orden con Mamba-3.

2. Modelo 2 - 'HYBRID-OMNI' (Arquitectura Asimétrica de Alto Rendimiento y Razonamiento):
   Patrón: LRCM -> HOPMIX -> MAMBA3 -> MAMBA3 -> LRCM -> MAMBA3
   - Entrada con atención local precisa W=16 y compresión de chunks.
   - Difusión multiescala inmediata con HopMix.
   - Núcleo denso Mamba-3 + fase de recuperación exacta LRCM + cabezal de razonamiento dinámico Mamba-3 final.

Protocolo de Evaluación:
- Dataset Causal Enriquecido (Lenguaje natural estructurado + tareas sintéticas entrelazadas).
- Entrenamiento completo con optimizador AdamW y pérdida combinada (Causal CE + Balanceo ECHO).
- Métricas:
  * Loss Inicial vs Loss Final (% Reducción)
  * Tiempo de Entrenamiento Total (segundos)
  * Perplejidad en Validación (PPL)
  * Aguja en un Pajar (% Exact Needle Retrieval)
  * Cambio de Significado / Seguimiento de Estado (% State Tracking Accuracy)
  * Velocidad de Decodificación Token a Token (tokens/segundo)
  * Consumo de Memoria Pico (MB)
"""

import sys
import time
import math
import random
import tracemalloc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from hybrid_model import HybridLMConfig, HybridCausalLM


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_rich_production_dataset(num_samples: int, seq_len: int, vocab_size: int):
    """
    Genera un corpus sintético rico que combina:
    - Patrones sintácticos causales (Markov estructurado)
    - Tareas de recuperación a largo alcance (Needle in a Haystack)
    - Tareas de actualización de variables / estado
    """
    data = torch.zeros(num_samples, seq_len, dtype=torch.long)
    for i in range(num_samples):
        mode = i % 3
        if mode == 0:
            # Lenguaje estructurado / Markov sintáctico
            curr = random.randint(10, vocab_size - 1)
            for t in range(seq_len):
                data[i, t] = curr
                if random.random() < 0.70:
                    curr = (curr * 7 + 13) % (vocab_size - 10) + 10
                else:
                    curr = random.randint(10, vocab_size - 1)
        elif mode == 1:
            # Tarea de Aguja en un Pajar
            tokens = torch.randint(10, vocab_size - 20, (seq_len,))
            insert_pos = random.randint(2, seq_len // 3)
            secret_val = random.randint(100, vocab_size - 1)
            tokens[insert_pos] = 1 # key
            tokens[insert_pos + 1] = secret_val
            tokens[-2] = 2 # query
            tokens[-1] = 1
            data[i] = tokens
        else:
            # Cambio de Significado / Seguimiento de Variables
            tokens = torch.randint(10, vocab_size - 20, (seq_len,))
            pos1 = random.randint(2, seq_len // 4)
            old_val = random.randint(50, 100)
            tokens[pos1] = 3 # var
            tokens[pos1 + 1] = 4 # assign
            tokens[pos1 + 2] = old_val

            pos2 = random.randint(seq_len // 2, seq_len - 5)
            new_val = random.randint(101, vocab_size - 1)
            tokens[pos2] = 3
            tokens[pos2 + 1] = 4
            tokens[pos2 + 2] = new_val

            tokens[-2] = 5 # query
            tokens[-1] = 3
            data[i] = tokens

    return data


def generate_eval_testsets(seq_len: int, vocab_size: int):
    # Test LM
    val_lm = torch.randint(10, vocab_size, (16, seq_len))
    
    # Test Needle
    needle_tokens = torch.randint(10, vocab_size - 20, (24, seq_len))
    needle_targets = torch.zeros(24, dtype=torch.long)
    for i in range(24):
        pos = random.randint(2, seq_len // 3)
        val = random.randint(100, vocab_size - 1)
        needle_tokens[i, pos] = 1
        needle_tokens[i, pos + 1] = val
        needle_tokens[i, -2] = 2
        needle_tokens[i, -1] = 1
        needle_targets[i] = val

    # Test State
    state_tokens = torch.randint(10, vocab_size - 20, (24, seq_len))
    state_targets = torch.zeros(24, dtype=torch.long)
    for i in range(24):
        p1 = random.randint(2, seq_len // 4)
        v1 = random.randint(50, 100)
        state_tokens[i, p1] = 3
        state_tokens[i, p1 + 1] = 4
        state_tokens[i, p1 + 2] = v1

        p2 = random.randint(seq_len // 2, seq_len - 5)
        v2 = random.randint(101, vocab_size - 1)
        state_tokens[i, p2] = 3
        state_tokens[i, p2 + 1] = 4
        state_tokens[i, p2 + 2] = v2

        state_tokens[i, -2] = 5
        state_tokens[i, -1] = 3
        state_targets[i] = v2

    return val_lm, needle_tokens, needle_targets, state_tokens, state_targets


def run_production_duel():
    print("=" * 85)
    print("DUELO DEFINITIVO: ENTRENAMIENTO DE LOS 2 MODELOS DE PRODUCCIÓN REAL")
    print("Buscando el equilibrio óptimo entre Rapidez, Inteligencia y Calidad de Lenguaje")
    print("=" * 85)

    vocab_size = 256
    seq_len = 64
    d_model = 64
    batch_size = 8
    epochs = 18 # Entrenamiento más profundo para convergencia sólida
    device = torch.device("cpu")

    models_config = {
        "HYBRID-APEX (Pyramid Balanced)": [
            "hopmix", "lrcm", "mamba3", "mamba3", "lrcm", "hopmix"
        ],
        "HYBRID-OMNI (Asymmetric Full-Stack)": [
            "lrcm", "hopmix", "mamba3", "mamba3", "lrcm", "mamba3"
        ],
    }

    set_seed(500)
    train_data = generate_rich_production_dataset(num_samples=48, seq_len=seq_len, vocab_size=vocab_size).to(device)
    val_lm, needle_tok, needle_tgt, state_tok, state_tgt = generate_eval_testsets(seq_len, vocab_size)
    val_lm = val_lm.to(device)
    needle_tok, needle_tgt = needle_tok.to(device), needle_tgt.to(device)
    state_tok, state_tgt = state_tok.to(device), state_tgt.to(device)

    final_results = {}

    for name, pattern in models_config.items():
        set_seed(500)
        cfg = HybridLMConfig(
            vocab_size=vocab_size,
            d_model=d_model,
            max_seq_len=seq_len,
            layer_pattern=pattern,
            echo_n_keys=16,
            echo_top_k=2,
            echo_rank=8,
            lrcm_chunk_size=8,
            lrcm_local_window=16,
            lrcm_desc_dim=16,
            mamba_d_state=24,
            mamba_headdim=16,
            mamba_mimo_rank=1,
        )
        model = HybridCausalLM(cfg).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        optimizer = optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=1e-4)

        tracemalloc.start()
        start_time = time.perf_counter()

        initial_loss = None
        final_loss = None

        # Entrenamiento intensivo
        for epoch in range(epochs):
            for b in range(0, train_data.shape[0], batch_size):
                bx = train_data[b : b + batch_size]
                optimizer.zero_grad()
                out = model(bx, labels=bx)
                loss = out["loss"]
                loss.backward()
                optimizer.step()

                if initial_loss is None:
                    initial_loss = loss.item()
                final_loss = loss.item()

        train_time_s = time.perf_counter() - start_time
        curr_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Evaluación
        model.eval()
        with torch.no_grad():
            val_out = model(val_lm, labels=val_lm)
            ce_loss = val_out["loss"] - val_out["aux_loss"]
            perplexity = math.exp(min(ce_loss.item(), 20.0))

            # Aguja en un pajar
            n_logits = model(needle_tok)["logits"]
            pred_needle = torch.argmax(n_logits[:, -1, :], dim=-1)
            needle_acc = (pred_needle == needle_tgt).float().mean().item() * 100

            # Cambio de Significado / Estado
            s_logits = model(state_tok)["logits"]
            pred_state = torch.argmax(s_logits[:, -1, :], dim=-1)
            state_acc = (pred_state == state_tgt).float().mean().item() * 100

            # Velocidad de Decodificación
            t_gen_start = time.perf_counter()
            _ = model.generate(prompt_tokens=val_lm[0:1, :4], max_new_tokens=10, temperature=0.7)
            t_gen_end = time.perf_counter()
            decode_tok_s = 10.0 / (t_gen_end - t_gen_start)

        loss_delta = ((initial_loss - final_loss) / initial_loss) * 100

        final_results[name] = {
            "pattern": " -> ".join([p.upper() for p in pattern]),
            "params": total_params,
            "loss_init": initial_loss,
            "loss_final": final_loss,
            "loss_delta": loss_delta,
            "train_time_s": train_time_s,
            "perplexity": perplexity,
            "needle_acc": needle_acc,
            "state_acc": state_acc,
            "decode_tok_s": decode_tok_s,
            "peak_mem_mb": peak_mem / (1024 * 1024),
        }

        print(f"-> {name} completado | Loss: {initial_loss:.4f} -> {final_loss:.4f} (-{loss_delta:.1f}%) | Tiempo: {train_time_s:.2f}s | PPL: {perplexity:.2f} | Needle: {needle_acc:.1f}% | State: {state_acc:.1f}% | Dec: {decode_tok_s:.1f} tok/s")

    # Tabla resumen definitiva
    print("\n" + "=" * 135)
    print("TABLA COMPARATIVA DEFINITIVA DE MODELOS DE PRODUCCIÓN")
    print("=" * 135)
    header = f"{'Modelo Final':<36} | {'Params':<8} | {'Loss Init':<9} | {'Loss Fin':<9} | {'% Reduc':<8} | {'T.Entr(s)':<9} | {'PPL':<7} | {'Needle':<7} | {'State':<7} | {'Dec(t/s)':<8} | {'Mem(MB)':<7}"
    print(header)
    print("-" * 135)

    for k, v in final_results.items():
        row = f"{k:<36} | {v['params']:<8,} | {v['loss_init']:<9.4f} | {v['loss_final']:<9.4f} | {v['loss_delta']:<7.1f}% | {v['train_time_s']:<9.2f} | {v['perplexity']:<7.2f} | {v['needle_acc']:<6.1f}% | {v['state_acc']:<6.1f}% | {v['decode_tok_s']:<8.1f} | {v['peak_mem_mb']:<7.2f}"
        print(row)

    return final_results


if __name__ == "__main__":
    run_production_duel()
