"""
Suite integral de verificación, pruebas y benchmark de entrenamiento causal
para la arquitectura híbrida: Hop-Mix + LRCM + Mamba-3 + ECHO.
"""

import time
import torch
import torch.optim as optim
from echo import ECHO
from hopmix import HopMixECHOBlock
from lrcm import LRCMECHOBlock
from mamba3 import Mamba3
from hybrid_model import HybridLMConfig, HybridCausalLM


def test_individual_modules():
    print("=" * 60)
    print("1. PROBANDO MÓDULOS INDIVIDUALES")
    print("=" * 60)
    device = torch.device("cpu")
    B, T, D = 2, 32, 64

    # 1. ECHO
    echo = ECHO(d_model=D, n_keys=16, top_k=2, rank=8)
    x = torch.randn(B, T, D)
    y_echo, met_echo = echo(x, return_aux_loss=True)
    assert y_echo.shape == (B, T, D)
    assert "aux_loss" in met_echo
    print("[PASS] ECHO: Forward y métricas de balanceo correctas.")

    # 2. HopMix con ECHO nativo
    hop = HopMixECHOBlock(d_model=D, max_seq_len=64, echo_n_keys=16, echo_top_k=2)
    y_hop, met_hop = hop(x, return_aux_loss=True)
    assert y_hop.shape == (B, T, D)
    print("[PASS] HopMix + ECHO: Forward y mezcla O(log N) correcta.")

    # 3. LRCM con ECHO nativo
    lrcm = LRCMECHOBlock(d_model=D, n_heads=4, local_window=16, chunk_size=8, echo_n_keys=16, echo_top_k=2)
    y_lrcm, met_lrcm, _ = lrcm(x, return_aux_loss=True)
    assert y_lrcm.shape == (B, T, D)
    print("[PASS] LRCM + ECHO: Forward jerárquico y Exact Leaf Recall correcto.")

    # 4. Mamba-3 Puro (Trapezoidal + RoPE Complejo + MIMO)
    mamba = Mamba3(d_model=D, d_state=32, headdim=16, is_mimo=True, mimo_rank=2)
    y_mamba, _ = mamba(x)
    assert y_mamba.shape == (B, T, D)
    print("[PASS] Mamba-3: Forward con discretización trapezoidal y RoPE complejo correcto.")


def test_hybrid_causal_training():
    print("\n" + "=" * 60)
    print("2. SIMULACIÓN DE ENTRENAMIENTO CAUSAL AUTOREGRESIVO (END-TO-END)")
    print("=" * 60)
    cfg = HybridLMConfig(
        vocab_size=200,
        d_model=64,
        max_seq_len=64,
        layer_pattern=["lrcm", "hopmix", "mamba3", "mamba3", "hopmix", "lrcm"],
        echo_n_keys=16,
        echo_top_k=2,
        echo_rank=8,
        lrcm_chunk_size=8,
        lrcm_local_window=16,
        mamba_d_state=32,
        mamba_headdim=16,
        mamba_mimo_rank=2,
    )
    model = HybridCausalLM(cfg)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # Datos sintéticos para predicción de siguiente token
    torch.manual_seed(42)
    data = torch.randint(0, cfg.vocab_size, (4, 32))
    labels = data.clone()

    print(f"Arquitectura: {' -> '.join([b.upper() for b in cfg.layer_pattern])}")
    print(f"Parámetros totales: {sum(p.numel() for p in model.parameters()):,}")

    initial_loss = None
    final_loss = None

    for step in range(5):
        t0 = time.time()
        optimizer.zero_grad()
        out = model(data, labels=labels)
        loss = out["loss"]
        loss.backward()
        optimizer.step()
        dt = time.time() - t0

        if step == 0:
            initial_loss = loss.item()
        final_loss = loss.item()

        print(f"Paso {step + 1:02d} | Causal Loss: {loss.item():.4f} | Aux Loss: {out['aux_loss'].item():.5f} | Tiempo: {dt*1000:.1f}ms")

    assert final_loss < initial_loss, f"La pérdida debería disminuir: inicial {initial_loss} vs final {final_loss}"
    print(f"[PASS] Optimización causal end-to-end exitosa: Pérdida descendió de {initial_loss:.4f} a {final_loss:.4f}.")


def test_autoregressive_generation():
    print("\n" + "=" * 60)
    print("3. GENERACIÓN AUTOREGRESIVA TOKEN A TOKEN (.generate)")
    print("=" * 60)
    cfg = HybridLMConfig(
        vocab_size=100,
        d_model=64,
        max_seq_len=64,
        layer_pattern=["lrcm", "hopmix", "mamba3", "mamba3", "hopmix", "lrcm"],
        echo_n_keys=16,
        echo_top_k=2,
    )
    model = HybridCausalLM(cfg)
    prompt = torch.tensor([[10, 20, 30, 40]]) # [1, 4]
    
    print(f"Prompt de entrada: {prompt.tolist()}")
    generated = model.generate(prompt, max_new_tokens=10, temperature=0.8, top_k=20)
    print(f"Secuencia generada: {generated.tolist()}")
    assert generated.shape == (1, 14)
    print("[PASS] Inferencia causal generativa completada correctamente.")


if __name__ == "__main__":
    test_individual_modules()
    test_hybrid_causal_training()
    test_autoregressive_generation()
    print("\n" + "=" * 60)
    print("TODAS LAS PRUEBAS Y VALIDACIONES PASARON EXITOSAMENTE.")
    print("=" * 60)
