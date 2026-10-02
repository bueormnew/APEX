"""Run correctness, dual-GPU integration, and focused Triton benchmarks."""

import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn as nn

from apex_triton import mamba3_decode_step, mamba3_scan
from hybrid_model import HybridCausalLM, HybridLMConfig


def reference_scan(u, b, c, dt, trap, a):
    batch, steps, heads, width = u.shape
    state_size = b.shape[-1]
    h = torch.zeros(batch, heads, width, state_size, device=u.device)
    bx_prev = torch.zeros_like(h)
    outputs = []
    for step in range(steps):
        bx = u[:, step, :, :, None] * b[:, step, :, None, :]
        trap_t = trap[:, step, :, None, None]
        dt_t = dt[:, step, :, None, None]
        h = (
            torch.exp(dt_t * a[None, :, None, None]) * h
            + dt_t * ((1 - 0.5 * trap_t) * bx + 0.5 * trap_t * bx_prev)
        )
        outputs.append((h * c[:, step, :, None, :]).sum(dim=-1))
        bx_prev = bx
    return torch.stack(outputs, dim=1)


def timed_ms(operation, warmup=5, repeats=20):
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        operation()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples)


def benchmark_training_scan(device):
    torch.manual_seed(29)
    batch, steps, heads, width, state_size = 4, 96, 4, 32, 32
    u = torch.randn(batch, steps, heads, width, device=device, requires_grad=True)
    b = torch.randn(batch, steps, heads, state_size, device=device, requires_grad=True)
    c = torch.randn_like(b, requires_grad=True)
    dt = torch.rand(batch, steps, heads, device=device, requires_grad=True)
    trap = torch.rand_like(dt, requires_grad=True)
    a = (-torch.rand(heads, device=device)).requires_grad_()
    probe = torch.randn(batch, steps, heads, width, device=device)

    expected = reference_scan(u, b, c, dt, trap, a)
    actual = mamba3_scan(u.contiguous(), b.contiguous(), c.contiguous(),
                         dt.contiguous(), trap.contiguous(), a.contiguous())
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)

    def torch_train_step():
        for tensor in (u, b, c, dt, trap, a):
            tensor.grad = None
        reference_scan(u, b, c, dt, trap, a).mul(probe).sum().backward()

    def triton_train_step():
        for tensor in (u, b, c, dt, trap, a):
            tensor.grad = None
        mamba3_scan(u.contiguous(), b.contiguous(), c.contiguous(),
                    dt.contiguous(), trap.contiguous(), a.contiguous()).mul(probe).sum().backward()

    torch_ms = timed_ms(torch_train_step, warmup=1, repeats=5)
    triton_ms = timed_ms(triton_train_step, warmup=2, repeats=10)
    return {
        "shape": [batch, steps, heads, width, state_size],
        "torch_forward_backward_ms": round(torch_ms, 3),
        "triton_forward_backward_ms": round(triton_ms, 3),
        "speedup": round(torch_ms / triton_ms, 3),
    }


def benchmark_decode(device):
    torch.manual_seed(31)
    batch, heads, width, state_size = 16, 4, 32, 32
    u = torch.randn(batch, heads, width, device=device)
    b = torch.randn(batch, heads, state_size, device=device)
    c = torch.randn_like(b)
    dt = torch.rand(batch, heads, device=device)
    trap = torch.rand_like(dt)
    a = -torch.rand(heads, device=device)
    h_initial = torch.randn(batch, heads, width, state_size, device=device)
    bx_initial = torch.randn_like(h_initial)
    reference_h = h_initial.clone()
    reference_bx = bx_initial.clone()

    def torch_step():
        bx = u.unsqueeze(-1) * b.unsqueeze(-2)
        trap_t = trap[..., None, None]
        dt_t = dt[..., None, None]
        h = torch.exp(dt_t * a[None, :, None, None]) * reference_h
        h = h + dt_t * ((1 - 0.5 * trap_t) * bx + 0.5 * trap_t * reference_bx)
        y = (h * c.unsqueeze(-2)).sum(dim=-1)
        reference_h.copy_(h)
        reference_bx.copy_(bx)
        return y

    actual_h = h_initial.clone()
    actual_bx = bx_initial.clone()
    actual = mamba3_decode_step(u, b, c, dt, trap, a, actual_h, actual_bx)
    torch.testing.assert_close(actual, torch_step(), rtol=2e-4, atol=2e-5)
    actual_h.copy_(h_initial)
    actual_bx.copy_(bx_initial)

    def triton_step():
        return mamba3_decode_step(u, b, c, dt, trap, a, actual_h, actual_bx)

    torch_ms = timed_ms(torch_step, warmup=10, repeats=50)
    triton_ms = timed_ms(triton_step, warmup=10, repeats=50)
    return {
        "batch": batch,
        "torch_step_ms": round(torch_ms, 4),
        "triton_step_ms": round(triton_ms, 4),
        "speedup": round(torch_ms / triton_ms, 3),
    }


class _LossAndLogits(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, labels):
        result = self.model(input_ids, labels=labels)
        return result["loss"].reshape(1), result["logits"]


def verify_dual_gpu_training():
    config = HybridLMConfig(
        vocab_size=128,
        d_model=32,
        max_seq_len=64,
        layer_pattern=["hopmix", "lrcm", "mamba3"],
        echo_n_keys=8,
        echo_top_k=2,
        echo_rank=4,
        hop_gate_heads=4,
        lrcm_heads=4,
        lrcm_local_window=8,
        lrcm_chunk_size=8,
        lrcm_desc_dim=8,
        lrcm_beam=1,
        mamba_d_state=16,
        mamba_headdim=16,
    )
    model = HybridCausalLM(config)
    parallel = nn.DataParallel(model, device_ids=[0, 1]).cuda(0)
    optimizer = torch.optim.AdamW(parallel.parameters(), lr=1e-3)
    input_ids = torch.randint(0, config.vocab_size, (8, 32), device="cuda:0")
    labels = input_ids.clone()

    optimizer.zero_grad(set_to_none=True)
    losses, logits = parallel(input_ids, labels)
    loss = losses.mean()
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    assert torch.cuda.memory_reserved(0) > 0 and torch.cuda.memory_reserved(1) > 0
    return {
        "devices": [torch.cuda.get_device_name(i) for i in (0, 1)],
        "batch": list(input_ids.shape),
        "logits": list(logits.shape),
        "loss": round(loss.item(), 6),
        "gpu0_reserved_mb": round(torch.cuda.memory_reserved(0) / 2**20, 2),
        "gpu1_reserved_mb": round(torch.cuda.memory_reserved(1) / 2**20, 2),
    }


def verify_dual_gpu_generation():
    config = HybridLMConfig(
        vocab_size=64,
        d_model=32,
        max_seq_len=32,
        layer_pattern=["mamba3"],
        mamba_d_state=16,
        mamba_headdim=16,
        mamba_is_mimo=False,
    )
    results = []
    for device_index in (0, 1):
        model = HybridCausalLM(config).to(f"cuda:{device_index}").eval()
        prompt = torch.tensor([[1, 2, 3]], device=f"cuda:{device_index}")
        generated = model.generate(prompt, max_new_tokens=4, top_k=8)
        torch.cuda.synchronize(device_index)
        assert generated.shape == (1, 7)
        results.append({
            "device": torch.cuda.get_device_name(device_index),
            "generated_shape": list(generated.shape),
        })
        del model, prompt, generated
        torch.cuda.empty_cache()
    return results


def main():
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("This benchmark requires Kaggle's dual-T4 GPU machine.")
    if not __import__("apex_triton").TRITON_AVAILABLE:
        raise RuntimeError("Triton is not installed in the Kaggle runtime.")

    result = {
        "torch": torch.__version__,
        "triton": __import__("triton").__version__,
        "training_scan_gpu0": benchmark_training_scan(torch.device("cuda:0")),
        "decode_step_per_gpu": {
            str(index): benchmark_decode(torch.device(f"cuda:{index}"))
            for index in (0, 1)
        },
        "hybrid_data_parallel_training": verify_dual_gpu_training(),
        "autoregressive_generation_per_gpu": verify_dual_gpu_generation(),
    }
    output_path = Path("/kaggle/working/apex_triton_benchmark.json")
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved benchmark report to {output_path}")


if __name__ == "__main__":
    main()
