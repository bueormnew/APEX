"""
Script de investigación empírica: 5 Nuevas Mezclas Híbridas (Hop-Mix, LRCM, Mamba-3, ECHO).

Objetivo:
Evaluar 5 arquitecturas híbridas distintas a las probadas anteriormente para descubrir
la mejor relación estructural en cuanto a:
- Pérdida inicial (Loss Init)
- Pérdida final (Loss Fin)
- % Reducción de Pérdida
- Perplejidad (PPL)
- Tiempo de entrenamiento total (s)
- Precisión en Aguja en un Pajar (% Needle Recall)
- Precisión en Cambio de Significado / Seguimiento de Estado (% State Tracking)
- Velocidad de decodificación token a token (Decode tok/s)
- Memoria RAM/VRAM pico (MB)

Arquitecturas evaluadas (6 capas cada una):
1. Mezcla 1 - "Mamba Core Asimétrico": LRCM -> Mamba3 -> Mamba3 -> Mamba3 -> HopMix -> LRCM
2. Mezcla 2 - "HopMix Backbone": HopMix -> HopMix -> Mamba3 -> Mamba3 -> LRCM -> LRCM
3. Mezcla 3 - "Trio Intercalado": LRCM -> HopMix -> Mamba3 -> LRCM -> HopMix -> Mamba3
4. Mezcla 4 - "Mamba Sandwich": Mamba3 -> HopMix -> LRCM -> LRCM -> HopMix -> Mamba3
5. Mezcla 5 - "Pirámide Invertida": HopMix -> LRCM -> Mamba3 -> Mamba3 -> LRCM -> HopMix
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


def generate_causal_lm_data(num_samples: int, seq_len: int, vocab_size: int):
    data = torch.zeros(num_samples, seq_len, dtype=torch.long)
    for i in range(num_samples):
        curr = random.randint(10, vocab_size - 1)
        for t in range(seq_len):
            data[i, t] = curr
            if random.random() < 0.65:
                curr = (curr * 7 + 13) % (vocab_size - 10) + 10
            else:
                curr = random.randint(10, vocab_size - 1)
    return data


def generate_needle_in_haystack_batch(num_samples: int, seq_len: int, vocab_size: int):
    tokens = torch.randint(10, vocab_size - 20, (num_samples, seq_len))
    targets = torch.zeros(num_samples, dtype=torch.long)
    needle_key = 1
    query_token = 2

    for i in range(num_samples):
        insert_pos = random.randint(2, max(3, seq_len // 3))
        secret_val = random.randint(100, vocab_size - 1)
        tokens[i, insert_pos] = needle_key
        tokens[i, insert_pos + 1] = secret_val

        tokens[i, -2] = query_token
        tokens[i, -1] = needle_key
        targets[i] = secret_val

    return tokens, targets


def generate_state_tracking_batch(num_samples: int, seq_len: int, vocab_size: int):
    tokens = torch.randint(10, vocab_size - 20, (num_samples, seq_len))
    targets = torch.zeros(num_samples, dtype=torch.long)
    var_token = 3
    assign_token = 4
    query_token = 5

    for i in range(num_samples):
        pos1 = random.randint(2, seq_len // 4)
        old_val = random.randint(50, 100)
        tokens[i, pos1] = var_token
        tokens[i, pos1 + 1] = assign_token
        tokens[i, pos1 + 2] = old_val

        pos2 = random.randint(seq_len // 2, seq_len - 5)
        new_val = random.randint(101, vocab_size - 1)
        tokens[i, pos2] = var_token
        tokens[i, pos2 + 1] = assign_token
        tokens[i, pos2 + 2] = new_val

        tokens[i, -2] = query_token
        tokens[i, -1] = var_token
        targets[i] = new_val

    return tokens, targets


def run_5_hybrid_benchmark():
    print("=" * 80)
    print("INICIANDO INVESTIGACIÓN EMPÍRICA: 5 NUEVAS MEZCLAS HÍBRIDAS")
    print("Evaluando aprendizaje causal, tareas de contexto, velocidad y tiempo de entrenamiento")
    print("=" * 80)

    vocab_size = 256
    seq_len = 64
    d_model = 64
    batch_size = 8
    epochs = 15
    device = torch.device("cpu")

    # 5 Nuevas configuraciones arquitectónicas
    mezclas = {
        "Mezcla 1 (Mamba-Heavy Core)": [
            "lrcm", "mamba3", "mamba3", "mamba3", "hopmix", "lrcm"
        ],
        "Mezcla 2 (HopMix Backbone)": [
            "hopmix", "hopmix", "mamba3", "mamba3", "lrcm", "lrcm"
        ],
        "Mezcla 3 (Trio Intercalado)": [
            "lrcm", "hopmix", "mamba3", "lrcm", "hopmix", "mamba3"
        ],
        "Mezcla 4 (Mamba Sandwich)": [
            "mamba3", "hopmix", "lrcm", "lrcm", "hopmix", "mamba3"
        ],
        "Mezcla 5 (Pirámide Invertida)": [
            "hopmix", "lrcm", "mamba3", "mamba3", "lrcm", "hopmix"
        ],
    }

    # Datos fijos y reproducibles para comparación justa
    set_seed(300)
    train_lm_data = generate_causal_lm_data(num_samples=40, seq_len=seq_len, vocab_size=vocab_size).to(device)
    val_lm_data = generate_causal_lm_data(num_samples=16, seq_len=seq_len, vocab_size=vocab_size).to(device)

    needle_tokens, needle_targets = generate_needle_in_haystack_batch(num_samples=24, seq_len=seq_len, vocab_size=vocab_size)
    needle_tokens, needle_targets = needle_tokens.to(device), needle_targets.to(device)

    state_tokens, state_targets = generate_state_tracking_batch(num_samples=24, seq_len=seq_len, vocab_size=vocab_size)
    state_tokens, state_targets = state_tokens.to(device), state_targets.to(device)

    results = {}

    for name, pattern in mezclas.items():
        set_seed(300)
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
        optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

        tracemalloc.start()
        start_time = time.perf_counter()

        initial_loss = None
        final_loss = None

        # 1. Bucle de Entrenamiento
        for epoch in range(epochs):
            for b in range(0, train_lm_data.shape[0], batch_size):
                bx = train_lm_data[b : b + batch_size]
                optimizer.zero_grad()
                out = model(bx, labels=bx)
                loss = out["loss"]
                loss.backward()
                optimizer.step()

                if initial_loss is None:
                    initial_loss = loss.item()
                final_loss = loss.item()

            # Tareas complementarias de aguja y cambio de estado
            for b in range(0, needle_tokens.shape[0] // 2, batch_size):
                bx = needle_tokens[b : b + batch_size]
                optimizer.zero_grad()
                out = model(bx, labels=bx)
                out["loss"].backward()
                optimizer.step()

        train_time_s = time.perf_counter() - start_time
        curr_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # 2. Evaluación en Test Set
        model.eval()
        with torch.no_grad():
            val_out = model(val_lm_data, labels=val_lm_data)
            ce_loss = val_out["loss"] - val_out["aux_loss"]
            perplexity = math.exp(min(ce_loss.item(), 20.0))

            # Aguja en un pajar (Needle Recall)
            n_logits = model(needle_tokens)["logits"]
            pred_needle = torch.argmax(n_logits[:, -1, :], dim=-1)
            needle_acc = (pred_needle == needle_targets).float().mean().item() * 100

            # Cambio de Significado (State Tracking)
            s_logits = model(state_tokens)["logits"]
            pred_state = torch.argmax(s_logits[:, -1, :], dim=-1)
            state_acc = (pred_state == state_targets).float().mean().item() * 100

            # Velocidad de Decodificación token a token
            t_gen_start = time.perf_counter()
            _ = model.generate(prompt_tokens=val_lm_data[0:1, :4], max_new_tokens=10, temperature=0.7)
            t_gen_end = time.perf_counter()
            decode_tok_s = 10.0 / (t_gen_end - t_gen_start)

        loss_delta = ((initial_loss - final_loss) / initial_loss) * 100

        results[name] = {
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

        print(f"-> {name} | Loss: {initial_loss:.2f} -> {final_loss:.2f} (-{loss_delta:.1f}%) | Tiempo: {train_time_s:.2f}s | PPL: {perplexity:.1f} | Needle: {needle_acc:.1f}% | State: {state_acc:.1f}% | Dec: {decode_tok_s:.1f} tok/s")

    # Imprimir tabla comparativa estructurada
    print("\n" + "=" * 125)
    print("TABLA COMPARATIVA FINAL: 5 NUEVAS MEZCLAS HÍBRIDAS")
    print("=" * 125)
    header = f"{'Mezcla':<28} | {'Params':<8} | {'Loss Init':<9} | {'Loss Fin':<9} | {'% Reduc':<8} | {'T.Entr(s)':<9} | {'PPL':<7} | {'Needle':<7} | {'State':<7} | {'Dec(t/s)':<8} | {'Mem(MB)':<7}"
    print(header)
    print("-" * 125)

    for k, v in results.items():
        row = f"{k:<28} | {v['params']:<8,} | {v['loss_init']:<9.4f} | {v['loss_final']:<9.4f} | {v['loss_delta']:<7.1f}% | {v['train_time_s']:<9.2f} | {v['perplexity']:<7.2f} | {v['needle_acc']:<6.1f}% | {v['state_acc']:<6.1f}% | {v['decode_tok_s']:<8.1f} | {v['peak_mem_mb']:<7.2f}"
        print(row)

    return results


if __name__ == "__main__":
    run_5_hybrid_benchmark()
