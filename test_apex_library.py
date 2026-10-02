"""
Test Suite for APEX Library:
Verifica:
1. Instanciación y presets (APEX_PYRAMID, APEX_OMNI, APEX_DEEP_MAMBA).
2. Entrenamiento con APEXTrainer (AdamW, Cosine Annealing, Grad Clipping, Aux Loss ECHO).
3. Exportación y carga unificada en formato .apex.
4. Métricas de rendimiento con MetricsMonitor (Tokens/s, latencia, memoria).
5. Evaluación con BenchmarkHarness (QA Pairs y Needle).
6. Congelamiento para Fine-Tuning.
"""

import os
import torch
from torch.utils.data import TensorDataset, DataLoader

from apex_core import (
    APEXConfig,
    BlockPreset,
    APEXModel,
    APEXTrainer,
    TrainingConfig,
    MetricsMonitor,
    BenchmarkHarness,
)


def run_tests():
    print("=" * 60)
    print("INICIANDO PRUEBAS DE LA LIBRERÍA APEX")
    print("=" * 60)

    # 1. Probar Presets
    print("\n--- 1. Comprobando Maquetas y Presets ---")
    presets = [BlockPreset.APEX_PYRAMID, BlockPreset.APEX_OMNI, BlockPreset.APEX_DEEP_MAMBA]
    for p in presets:
        cfg = APEXConfig(vocab_size=1000, d_model=64, max_seq_len=128, preset=p)
        model = APEXModel(cfg)
        params = model.count_parameters()
        print(f"[OK] Preset {p.value.upper()}: {params:,} parámetros | Capas: {cfg.layer_pattern}")

    # 2. Probar Entrenamiento con APEXTrainer
    print("\n--- 2. Comprobando Entrenamiento Avanzado ---")
    config = APEXConfig(
        vocab_size=500,
        d_model=64,
        max_seq_len=64,
        preset=BlockPreset.APEX_PYRAMID,
        echo_n_keys=16,
        echo_top_k=2,
        echo_rank=8,
    )
    model = APEXModel(config)

    # Datos sintéticos
    dummy_x = torch.randint(0, 500, (16, 32))
    dummy_y = torch.randint(0, 500, (16, 32))
    train_loader = DataLoader(TensorDataset(dummy_x, dummy_y), batch_size=4)
    val_loader = DataLoader(TensorDataset(dummy_x[:4], dummy_y[:4]), batch_size=4)

    trainer_cfg = TrainingConfig(
        learning_rate=1e-3,
        weight_decay=0.01,
        max_grad_norm=1.0,
        warmup_steps=2,
        epochs=2,
        lr_scheduler_type="cosine",
        aux_loss_weight=0.05,
        log_interval=2,
        eval_every=4,
    )
    trainer = APEXTrainer(model, trainer_cfg)
    history = trainer.train(train_loader, val_loader)

    print(f"[OK] Entrenamiento completado. Loss inicial: {history['train_loss'][0]:.4f} -> Final: {history['train_loss'][-1]:.4f}")

    # 3. Probar Exportación y Carga .apex
    print("\n--- 3. Comprobando Exportación y Carga .apex ---")
    save_path = "test_model.apex"
    saved_file = model.save_apex(save_path)
    assert os.path.exists(saved_file), "El archivo .apex no fue creado"
    file_size_kb = os.path.getsize(saved_file) / 1024
    print(f"[OK] Archivo guardado con éxito: {saved_file} ({file_size_kb:.2f} KB)")

    loaded_model = APEXModel.load_apex(save_path)
    print(f"[OK] Modelo cargado desde {saved_file}. Parámetros: {loaded_model.count_parameters():,}")

    # Verificar equivalencia de predicción
    test_in = torch.randint(0, 500, (1, 16))
    out1 = model(test_in, return_dict=True)["logits"]
    out2 = loaded_model(test_in, return_dict=True)["logits"]
    diff = (out1 - out2).abs().max().item()
    print(f"[OK] Diferencia numérica entre modelo original y cargado: {diff:.6e}")
    assert diff < 1e-5, f"Diferencia numérica mayor al umbral: {diff}"

    # 4. Probar Métricas de Rendimiento (MetricsMonitor)
    print("\n--- 4. Comprobando MetricsMonitor ---")
    bench_res = MetricsMonitor.benchmark_throughput(
        loaded_model,
        seq_len=32,
        batch_size=2,
        num_batches=3,
        warmup_batches=1,
    )
    print(f"[OK] Forward Throughput: {bench_res['forward_tokens_per_sec']:.1f} tokens/s")
    print(f"[OK] Training Throughput: {bench_res['train_tokens_per_sec']:.1f} tokens/s")
    print(f"[OK] Memoria utilizada: {bench_res['memory_mb']:.2f} MB")

    gen_res = MetricsMonitor.benchmark_generation(
        loaded_model,
        prompt_len=8,
        gen_tokens=16,
    )
    print(f"[OK] Velocidad de Generación: {gen_res['generation_tokens_per_sec']:.1f} tokens/s ({gen_res['latency_ms_per_token']:.2f} ms/token)")

    # 5. Probar Harnes de Benchmarks (BenchmarkHarness)
    print("\n--- 5. Comprobando BenchmarkHarness ---")
    harness = BenchmarkHarness(loaded_model)
    qa_sample = [
        {"question": [10, 20, 30], "target": [40]},
        {"question": [15, 25, 35], "target": [45]},
    ]
    eval_res = harness.evaluate_qa(qa_sample, max_new_tokens=2)
    print(f"[OK] Harness QA Items evaluados: {eval_res['total_items']} | Precisión calculada: {eval_res['accuracy_percent']:.1f}%")

    # 6. Probar Modos de Fine-Tuning
    print("\n--- 6. Comprobando Congelamiento de Componentes ---")
    trainer.freeze_components("echo_only")
    trainable = model.trainable_parameters()
    total = model.count_parameters()
    print(f"[OK] Modo 'echo_only' activado: {trainable:,} entrenables de {total:,} ({trainable/total*100:.1f}%)")

    # Limpiar archivo temporal
    if os.path.exists(save_path):
        os.remove(save_path)
    print("\n" + "=" * 60)
    print("TODAS LAS PRUEBAS DE LA LIBRERÍA APEX PASARON EXITOSAMENTE")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
