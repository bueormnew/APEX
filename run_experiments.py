"""
Script ejecutable para PRUEBA 1 y PRUEBA 2 solicitadas por el usuario.

Prueba 1:
- 3 modelos diminutos separados con parámetros prácticamente idénticos (~340K-350K params):
  1) Mamba-3 puro
  2) LRCM + ECHO
  3) Hop-Mix + ECHO
- Mismos datos sintéticos causales, mismo batch, mismas épocas y optimizador.
- Medición de: Loss inicial, Loss final, % reducción de pérdida, tiempo de entrenamiento,
  consumo de memoria pico (RAM/VRAM), velocidad (tokens/segundo) y tiempo por paso.

Prueba 2:
- 3 modelos híbridos de tamaño similar con distintas combinaciones de bloques:
  * Híbrido A (Simétrico sandwich / Sandwich equilibrado): LRCM -> Hop-Mix -> Mamba-3 -> Mamba-3 -> Hop-Mix -> LRCM
  * Híbrido B (Recurrente/Convolucional denso): Mamba-3 -> LRCM -> Mamba-3 -> LRCM -> Mamba-3 -> LRCM
  * Híbrido C (Recuperación multiescala rápida): Hop-Mix -> LRCM -> Hop-Mix -> LRCM -> Mamba-3 -> Mamba-3
- Entrenados con datos sintéticos variados simulando:
  1) Lenguaje natural estructurado (sintaxis y n-gramas causales)
  2) Tareas de Aguja en un Pajar (Needle-in-a-Haystack: claves dispersas a larga distancia)
  3) Tareas de cambio de significado / actualización de contexto (state redefinition: X=A ... X=B ... ¿cuánto vale X?)
- Evaluación exhaustiva:
  - Perplejidad (PPL) en lenguaje
  - Tasa de recuperación exacta en Aguja en un Pajar (% Exact Needle Recall)
  - Tasa de exactitud en cambio de significado / actualización (% State Update Accuracy)
  - Velocidad de inferencia (tokens/segundo) y tiempo de decodificación token a token
  - Consumo de memoria
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


# =====================================================================
# GENERADORES DE DATOS SINTÉTICOS PARA PRUEBA 1 Y 2
# =====================================================================

def generate_causal_lm_data(num_samples: int, seq_len: int, vocab_size: int):
    """
    Genera secuencias sintéticas con dependencias causales (Markov / n-gramas + ruido)
    para evaluar modelado de lenguaje autorregresivo.
    """
    data = torch.zeros(num_samples, seq_len, dtype=torch.long)
    for i in range(num_samples):
        curr = random.randint(10, vocab_size - 1)
        for t in range(seq_len):
            data[i, t] = curr
            if random.random() < 0.65:
                # Regla de transición estructurada
                curr = (curr * 7 + 13) % (vocab_size - 10) + 10
            else:
                curr = random.randint(10, vocab_size - 1)
    return data


def generate_needle_in_haystack_batch(num_samples: int, seq_len: int, vocab_size: int):
    """
    Genera secuencias tipo 'Needle in a Haystack':
    Inserta una clave 'KEY_ID -> VALUE_ID' en una posición lejana aleatoria del contexto
    y al final pregunta por 'KEY_ID -> ?' para verificar si el modelo puede recuperarlo.
    """
    tokens = torch.randint(10, vocab_size - 20, (num_samples, seq_len))
    targets = torch.zeros(num_samples, dtype=torch.long)
    needle_key = 1  # Token especial de clave
    query_token = 2 # Token especial de pregunta

    for i in range(num_samples):
        # Insertar aguja en el primer tercio del haystack
        insert_pos = random.randint(2, max(3, seq_len // 3))
        secret_val = random.randint(100, vocab_size - 1)
        tokens[i, insert_pos] = needle_key
        tokens[i, insert_pos + 1] = secret_val

        # Preguntar en el último paso
        tokens[i, -2] = query_token
        tokens[i, -1] = needle_key
        targets[i] = secret_val

    return tokens, targets


def generate_state_tracking_batch(num_samples: int, seq_len: int, vocab_size: int):
    """
    Genera secuencias de cambio de significado / actualización de estado:
    Ejemplo: VAR_X = VAL_A ... [ruido] ... VAR_X = VAL_B ... ¿VAR_X?
    El modelo debe dar el valor MÁS RECIENTE y no confundirse con el antiguo.
    """
    tokens = torch.randint(10, vocab_size - 20, (num_samples, seq_len))
    targets = torch.zeros(num_samples, dtype=torch.long)
    var_token = 3
    assign_token = 4
    query_token = 5

    for i in range(num_samples):
        # Primera asignación antigua
        pos1 = random.randint(2, seq_len // 4)
        old_val = random.randint(50, 100)
        tokens[i, pos1] = var_token
        tokens[i, pos1 + 1] = assign_token
        tokens[i, pos1 + 2] = old_val

        # Segunda asignación (cambio de significado/actualización de estado)
        pos2 = random.randint(seq_len // 2, seq_len - 5)
        new_val = random.randint(101, vocab_size - 1)
        tokens[i, pos2] = var_token
        tokens[i, pos2 + 1] = assign_token
        tokens[i, pos2 + 2] = new_val

        # Consulta al final
        tokens[i, -2] = query_token
        tokens[i, -1] = var_token
        targets[i] = new_val

    return tokens, targets


# =====================================================================
# EJECUCIÓN DE PRUEBA 1
# =====================================================================

def run_test_1():
    print("=" * 75)
    print("INICIANDO PRUEBA 1: COMPARATIVA PURA ENTRE MAMBA-3, LRCM Y HOP-MIX")
    print("Mismos parámetros, mismos datos sintéticos, mismas épocas y entorno idéntico.")
    print("=" * 75)

    vocab_size = 256
    seq_len = 64
    d_model = 64
    batch_size = 8
    epochs = 12
    device = torch.device("cpu")

    # 1. Configuraciones con paridad de parámetros (~335K - 355K params)
    configs = {
        "Mamba-3": HybridLMConfig(
            vocab_size=vocab_size,
            d_model=d_model,
            max_seq_len=seq_len,
            layer_pattern=["mamba3"] * 4,
            mamba_d_state=30,
            mamba_headdim=16,
            mamba_mimo_rank=1,
        ),
        "Hop-Mix + ECHO": HybridLMConfig(
            vocab_size=vocab_size,
            d_model=d_model,
            max_seq_len=seq_len,
            layer_pattern=["hopmix"] * 4,
            echo_n_keys=20,
            echo_top_k=2,
            echo_rank=11,
            hop_routes=2,
            hop_gate_heads=4,
        ),
        "LRCM + ECHO": HybridLMConfig(
            vocab_size=vocab_size,
            d_model=d_model,
            max_seq_len=seq_len,
            layer_pattern=["lrcm"] * 4,
            echo_n_keys=15,
            echo_top_k=2,
            echo_rank=7,
            lrcm_chunk_size=8,
            lrcm_local_window=16,
            lrcm_desc_dim=16,
        )
    }

    # Datos fijos para que los 3 modelos vean exactamente los mismos tokens
    set_seed(100)
    train_data = generate_causal_lm_data(num_samples=32, seq_len=seq_len, vocab_size=vocab_size).to(device)

    results_p1 = {}

    for name, cfg in configs.items():
        set_seed(100) # Inicialización reproducible
        model = HybridCausalLM(cfg).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

        tracemalloc.start()
        start_time = time.perf_counter()

        initial_loss = None
        final_loss = None
        total_tokens = 0

        for epoch in range(epochs):
            # Dividir en mini-batches
            for b in range(0, train_data.shape[0], batch_size):
                batch_x = train_data[b : b + batch_size]
                optimizer.zero_grad()
                out = model(batch_x, labels=batch_x)
                loss = out["loss"]
                loss.backward()
                optimizer.step()

                if initial_loss is None:
                    initial_loss = loss.item()
                final_loss = loss.item()
                total_tokens += batch_x.numel()

        elapsed_time = time.perf_counter() - start_time
        curr_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        speed_tok_s = total_tokens / elapsed_time
        time_per_step = (elapsed_time / (epochs * (train_data.shape[0] // batch_size))) * 1000

        results_p1[name] = {
            "params": total_params,
            "loss_init": initial_loss,
            "loss_final": final_loss,
            "loss_delta": ((initial_loss - final_loss) / initial_loss) * 100,
            "train_time_s": elapsed_time,
            "peak_mem_mb": peak_mem / (1024 * 1024),
            "speed_tok_s": speed_tok_s,
            "ms_step": time_per_step,
        }
        print(f"-> {name} completado | Parámetros: {total_params:,} | Loss: {initial_loss:.4f} -> {final_loss:.4f} (-{results_p1[name]['loss_delta']:.1f}%) | Tiempo: {elapsed_time:.2f}s | Mem: {results_p1[name]['peak_mem_mb']:.2f}MB")

    return results_p1


# =====================================================================
# EJECUCIÓN DE PRUEBA 2
# =====================================================================

def run_test_2():
    print("\n" + "=" * 75)
    print("INICIANDO PRUEBA 2: COMPARATIVA DE 3 ARQUITECTURAS PURAMENTE HÍBRIDAS")
    print("Evaluación en Lenguaje Natural, Aguja en un Pajar y Cambio de Significado.")
    print("=" * 75)

    vocab_size = 256
    seq_len = 64
    d_model = 64
    batch_size = 8
    epochs = 15
    device = torch.device("cpu")

    # 3 Configuraciones híbridas con patrones estructurales distintos
    hybrid_configs = {
        "Híbrido A (Sandwich Simétrico)": HybridLMConfig(
            vocab_size=vocab_size,
            d_model=d_model,
            max_seq_len=seq_len,
            # LRCM -> HOP -> MAMBA -> MAMBA -> HOP -> LRCM
            layer_pattern=["lrcm", "hopmix", "mamba3", "mamba3", "hopmix", "lrcm"],
            echo_n_keys=16,
            echo_top_k=2,
            echo_rank=8,
            lrcm_chunk_size=8,
            lrcm_local_window=16,
            lrcm_desc_dim=16,
            mamba_d_state=24,
            mamba_headdim=16,
            mamba_mimo_rank=1,
        ),
        "Híbrido B (Recurrente + Conv Alternado)": HybridLMConfig(
            vocab_size=vocab_size,
            d_model=d_model,
            max_seq_len=seq_len,
            # MAMBA -> LRCM -> MAMBA -> LRCM -> MAMBA -> LRCM
            layer_pattern=["mamba3", "lrcm", "mamba3", "lrcm", "mamba3", "lrcm"],
            echo_n_keys=16,
            echo_top_k=2,
            echo_rank=8,
            lrcm_chunk_size=8,
            lrcm_local_window=16,
            lrcm_desc_dim=16,
            mamba_d_state=24,
            mamba_headdim=16,
            mamba_mimo_rank=1,
        ),
        "Híbrido C (Hop-Mix Multi-Hop + Mamba)": HybridLMConfig(
            vocab_size=vocab_size,
            d_model=d_model,
            max_seq_len=seq_len,
            # HOP -> LRCM -> HOP -> LRCM -> MAMBA -> MAMBA
            layer_pattern=["hopmix", "lrcm", "hopmix", "lrcm", "mamba3", "mamba3"],
            echo_n_keys=16,
            echo_top_k=2,
            echo_rank=8,
            lrcm_chunk_size=8,
            lrcm_local_window=16,
            lrcm_desc_dim=16,
            mamba_d_state=24,
            mamba_headdim=16,
            mamba_mimo_rank=1,
        ),
    }

    # Dataset combinado simulando lenguaje real + tareas
    set_seed(200)
    train_lm_data = generate_causal_lm_data(num_samples=40, seq_len=seq_len, vocab_size=vocab_size).to(device)

    # Conjuntos de evaluación para pruebas complejas
    val_lm_data = generate_causal_lm_data(num_samples=16, seq_len=seq_len, vocab_size=vocab_size).to(device)
    needle_tokens, needle_targets = generate_needle_in_haystack_batch(num_samples=24, seq_len=seq_len, vocab_size=vocab_size)
    needle_tokens, needle_targets = needle_tokens.to(device), needle_targets.to(device)
    state_tokens, state_targets = generate_state_tracking_batch(num_samples=24, seq_len=seq_len, vocab_size=vocab_size)
    state_tokens, state_targets = state_tokens.to(device), state_targets.to(device)

    results_p2 = {}

    for name, cfg in hybrid_configs.items():
        set_seed(200)
        model = HybridCausalLM(cfg).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

        tracemalloc.start()
        start_time = time.perf_counter()

        initial_loss = None
        final_loss = None

        # 1. Entrenamiento con datos variados
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

            # Entrenamiento parcial en tareas específicas para evaluar transfer/aprendizaje
            for b in range(0, needle_tokens.shape[0] // 2, batch_size):
                bx = needle_tokens[b : b + batch_size]
                optimizer.zero_grad()
                out = model(bx, labels=bx)
                out["loss"].backward()
                optimizer.step()

        elapsed_time = time.perf_counter() - start_time
        curr_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # 2. Evaluación de Perplejidad en test LM
        model.eval()
        with torch.no_grad():
            val_out = model(val_lm_data, labels=val_lm_data)
            # ce loss pura
            ce_loss = val_out["loss"] - val_out["aux_loss"]
            perplexity = math.exp(min(ce_loss.item(), 20.0))

            # 3. Evaluación de Aguja en un Pajar (Needle in a Haystack)
            # Evaluar si en la posición de predicción se asigna alta probabilidad al secreto
            n_logits = model(needle_tokens)["logits"] # [B, T, V]
            # Predicción del token siguiente a needle_key al final:
            pred_needle = torch.argmax(n_logits[:, -1, :], dim=-1)
            # Medimos top-5 o exact match:
            needle_acc = (pred_needle == needle_targets).float().mean().item() * 100

            # 4. Evaluación de Cambio de Significado / Actualización de Estado
            s_logits = model(state_tokens)["logits"]
            pred_state = torch.argmax(s_logits[:, -1, :], dim=-1)
            state_acc = (pred_state == state_targets).float().mean().item() * 100

            # 5. Medición de velocidad de decodificación token a token
            t_gen_start = time.perf_counter()
            _ = model.generate(prompt_tokens=val_lm_data[0:1, :4], max_new_tokens=10, temperature=0.7)
            t_gen_end = time.perf_counter()
            decode_tok_s = 10.0 / (t_gen_end - t_gen_start)

        results_p2[name] = {
            "pattern": " -> ".join([p.upper() for p in cfg.layer_pattern]),
            "params": total_params,
            "loss_init": initial_loss,
            "loss_final": final_loss,
            "perplexity": perplexity,
            "needle_acc": needle_acc,
            "state_acc": state_acc,
            "train_time_s": elapsed_time,
            "peak_mem_mb": peak_mem / (1024 * 1024),
            "decode_tok_s": decode_tok_s,
        }
        print(f"-> {name} | PPL: {perplexity:.2f} | Needle Acc: {needle_acc:.1f}% | State Acc: {state_acc:.1f}% | Dec Speed: {decode_tok_s:.1f} tok/s")

    return results_p2


if __name__ == "__main__":
    t1_results = run_test_1()
    t2_results = run_test_2()

    # Imprimir tablas en consola de forma estructurada
    print("\n" + "=" * 90)
    print("TABLA RESUMEN PRUEBA 1 (MODELOS INDIVIDUALES A PARIDAD DE PARÁMETROS)")
    print("=" * 90)
    print(f"{'Modelo':<18} | {'Params':<8} | {'Loss Init':<9} | {'Loss Fin':<9} | {'Reducción':<9} | {'Tiempo(s)':<9} | {'Mem(MB)':<8} | {'Tok/s':<7}")
    print("-" * 90)
    for k, v in t1_results.items():
        print(f"{k:<18} | {v['params']:<8,} | {v['loss_init']:<9.4f} | {v['loss_final']:<9.4f} | {v['loss_delta']:<8.1f}% | {v['train_time_s']:<9.2f} | {v['peak_mem_mb']:<8.2f} | {v['speed_tok_s']:<7.1f}")

    print("\n" + "=" * 110)
    print("TABLA RESUMEN PRUEBA 2 (MODELOS HÍBRIDOS CON DATOS COMPLEJOS Y TAREAS DE CONTEXTO)")
    print("=" * 110)
    print(f"{'Arquitectura Híbrida':<36} | {'Params':<8} | {'Loss Fin':<9} | {'PPL':<7} | {'Needle':<8} | {'StateUpd':<8} | {'Dec(tok/s)':<10} | {'Mem(MB)':<7}")
    print("-" * 110)
    for k, v in t2_results.items():
        print(f"{k:<36} | {v['params']:<8,} | {v['loss_final']:<9.4f} | {v['perplexity']:<7.2f} | {v['needle_acc']:<7.1f}% | {v['state_acc']:<7.1f}% | {v['decode_tok_s']:<10.1f} | {v['peak_mem_mb']:<7.2f}")
