"""CUDA correctness tests for the optional Mamba-3 Triton kernels."""

import pytest
import torch

from apex_triton import TRITON_AVAILABLE, mamba3_decode_step, mamba3_scan


pytestmark = pytest.mark.skipif(
    not TRITON_AVAILABLE or not torch.cuda.is_available(),
    reason="Triton CUDA kernels require a CUDA-enabled PyTorch and Triton",
)
DEVICE_INDICES = list(range(torch.cuda.device_count())) or [0]


def _reference_scan(u, b, c, dt, trap, a):
    batch, steps, heads, width = u.shape
    state_size = b.shape[-1]
    h = u.new_zeros(batch, heads, width, state_size)
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


@pytest.mark.parametrize("device_index", DEVICE_INDICES)
def test_mamba3_scan_matches_reference_outputs_and_gradients(device_index):
    device = torch.device(f"cuda:{device_index}")
    torch.manual_seed(17)
    shapes = ((2, 5, 2, 3), (2, 5, 2, 4), (2, 5, 2), (2,))
    u = torch.randn(*shapes[0], device=device, requires_grad=True)
    b = torch.randn(*shapes[1], device=device, requires_grad=True)
    c = torch.randn(*shapes[1], device=device, requires_grad=True)
    dt = torch.rand(*shapes[2], device=device, requires_grad=True)
    trap = torch.rand(*shapes[2], device=device, requires_grad=True)
    a = (-torch.rand(*shapes[3], device=device)).requires_grad_()
    inputs = (u, b, c, dt, trap, a)

    expected = _reference_scan(*inputs)
    actual = mamba3_scan(*inputs)
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)

    probe = torch.randn_like(expected)
    expected_grads = torch.autograd.grad((expected * probe).sum(), inputs)
    actual_grads = torch.autograd.grad((actual * probe).sum(), inputs)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=3e-4, atol=3e-5)


@pytest.mark.parametrize("device_index", DEVICE_INDICES)
def test_mamba3_decode_step_matches_reference_and_updates_state(device_index):
    device = torch.device(f"cuda:{device_index}")
    torch.manual_seed(23)
    batch, heads, width, state_size = 2, 2, 3, 4
    u = torch.randn(batch, heads, width, device=device)
    b = torch.randn(batch, heads, state_size, device=device)
    c = torch.randn_like(b)
    dt = torch.rand(batch, heads, device=device)
    trap = torch.rand_like(dt)
    a = -torch.rand(heads, device=device)
    h_state = torch.randn(batch, heads, width, state_size, device=device)
    bx_prev = torch.randn_like(h_state)

    expected_bx = u.unsqueeze(-1) * b.unsqueeze(-2)
    trap_t = trap[..., None, None]
    dt_t = dt[..., None, None]
    expected_h = (
        torch.exp(dt_t * a[None, :, None, None]) * h_state
        + dt_t * ((1 - 0.5 * trap_t) * expected_bx + 0.5 * trap_t * bx_prev)
    )
    expected_y = (expected_h * c.unsqueeze(-2)).sum(dim=-1)

    actual_y = mamba3_decode_step(u, b, c, dt, trap, a, h_state, bx_prev)
    assert actual_y is not None
    torch.testing.assert_close(actual_y, expected_y, rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(h_state, expected_h, rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(bx_prev, expected_bx, rtol=2e-4, atol=2e-5)
