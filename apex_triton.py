"""Optional Triton kernels for the Mamba-3 recurrent core.

The kernels fuse the recurrent state update and readout. Projections, RoPE,
gating, and the remaining APEX blocks stay in PyTorch.
"""

from typing import Optional

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


TRITON_AVAILABLE = triton is not None


if TRITON_AVAILABLE:
    @triton.jit
    def _mamba_scan_forward(
        U, B, C, DT, TRAP, A, Y, H_STATES, BX_STATES,
        T: tl.constexpr, H: tl.constexpr, P: tl.constexpr, N: tl.constexpr,
        BP: tl.constexpr, BN: tl.constexpr,
    ):
        bh = tl.program_id(0)
        batch = bh // H
        head = bh % H
        p = tl.arange(0, BP)
        n = tl.arange(0, BN)
        p_mask = p < P
        n_mask = n < N

        h_state = tl.zeros((BP, BN), tl.float32)
        bx_prev = tl.zeros((BP, BN), tl.float32)
        a_cont = tl.load(A + head).to(tl.float32)

        for t in range(T):
            u = tl.load(U + ((batch * T + t) * H + head) * P + p, p_mask, 0).to(tl.float32)
            b = tl.load(B + ((batch * T + t) * H + head) * N + n, n_mask, 0).to(tl.float32)
            c = tl.load(C + ((batch * T + t) * H + head) * N + n, n_mask, 0).to(tl.float32)
            dt = tl.load(DT + (batch * T + t) * H + head).to(tl.float32)
            trap = tl.load(TRAP + (batch * T + t) * H + head).to(tl.float32)

            bx = u[:, None] * b[None, :]
            q = (1.0 - 0.5 * trap) * bx + (0.5 * trap) * bx_prev
            a = tl.exp(dt * a_cont)
            h_state = a * h_state + dt * q
            y = tl.sum(h_state * c[None, :], axis=1)

            tl.store(Y + ((batch * T + t) * H + head) * P + p, y, p_mask)
            state_offset = (((batch * T + t) * H + head) * P + p[:, None]) * N + n[None, :]
            state_mask = p_mask[:, None] & n_mask[None, :]
            tl.store(H_STATES + state_offset, h_state, state_mask)
            tl.store(BX_STATES + state_offset, bx, state_mask)
            bx_prev = bx


    @triton.jit
    def _mamba_scan_backward(
        U, B, C, DT, TRAP, A, DY, H_STATES, BX_STATES,
        DU, DB, DC, DDT, DTRAP, DA,
        T: tl.constexpr, H: tl.constexpr, P: tl.constexpr, N: tl.constexpr,
        BP: tl.constexpr, BN: tl.constexpr,
    ):
        bh = tl.program_id(0)
        batch = bh // H
        head = bh % H
        p = tl.arange(0, BP)
        n = tl.arange(0, BN)
        p_mask = p < P
        n_mask = n < N

        gh_next = tl.zeros((BP, BN), tl.float32)
        gbx_next = tl.zeros((BP, BN), tl.float32)
        a_cont = tl.load(A + head).to(tl.float32)
        da_acc = tl.full((), 0.0, tl.float32)

        for rev_t in range(T):
            t = T - 1 - rev_t
            u = tl.load(U + ((batch * T + t) * H + head) * P + p, p_mask, 0).to(tl.float32)
            b = tl.load(B + ((batch * T + t) * H + head) * N + n, n_mask, 0).to(tl.float32)
            c = tl.load(C + ((batch * T + t) * H + head) * N + n, n_mask, 0).to(tl.float32)
            dt = tl.load(DT + (batch * T + t) * H + head).to(tl.float32)
            trap = tl.load(TRAP + (batch * T + t) * H + head).to(tl.float32)
            dy = tl.load(DY + ((batch * T + t) * H + head) * P + p, p_mask, 0).to(tl.float32)

            offset = (((batch * T + t) * H + head) * P + p[:, None]) * N + n[None, :]
            h_cur = tl.load(H_STATES + offset, p_mask[:, None] & n_mask[None, :], 0).to(tl.float32)
            bx = tl.load(BX_STATES + offset, p_mask[:, None] & n_mask[None, :], 0).to(tl.float32)
            if t > 0:
                prev_offset = ((((batch * T + t - 1) * H + head) * P + p[:, None]) * N + n[None, :])
                h_prev = tl.load(H_STATES + prev_offset, p_mask[:, None] & n_mask[None, :], 0).to(tl.float32)
                bx_prev = tl.load(BX_STATES + prev_offset, p_mask[:, None] & n_mask[None, :], 0).to(tl.float32)
            else:
                h_prev = tl.zeros((BP, BN), tl.float32)
                bx_prev = tl.zeros((BP, BN), tl.float32)

            q = (1.0 - 0.5 * trap) * bx + (0.5 * trap) * bx_prev
            a = tl.exp(dt * a_cont)
            gh = gh_next + dy[:, None] * c[None, :]
            gq = gh * dt
            gbx = gbx_next + gq * (1.0 - 0.5 * trap)
            gbx_prev = gq * (0.5 * trap)

            du = tl.sum(gbx * b[None, :], axis=1)
            db = tl.sum(gbx * u[:, None], axis=0)
            dc = tl.sum(dy[:, None] * h_cur, axis=0)
            ddt = tl.sum(tl.sum(gh * q, axis=1), axis=0)
            ddt += tl.sum(tl.sum(gh * h_prev, axis=1), axis=0) * a * a_cont
            dtrap = tl.sum(tl.sum(gq * (0.5 * (bx_prev - bx)), axis=1), axis=0)
            da_acc += tl.sum(tl.sum(gh * h_prev, axis=1), axis=0) * a * dt

            tl.store(DU + ((batch * T + t) * H + head) * P + p, du, p_mask)
            tl.store(DB + ((batch * T + t) * H + head) * N + n, db, n_mask)
            tl.store(DC + ((batch * T + t) * H + head) * N + n, dc, n_mask)
            tl.store(DDT + (batch * T + t) * H + head, ddt)
            tl.store(DTRAP + (batch * T + t) * H + head, dtrap)

            gh_next = gh * a
            gbx_next = gbx_prev

        tl.atomic_add(DA + head, da_acc)


    @triton.jit
    def _mamba_decode_step(
        U, B, C, DT, TRAP, A, H_STATE, BX_PREV, Y,
        H: tl.constexpr, P: tl.constexpr, N: tl.constexpr,
        BP: tl.constexpr, BN: tl.constexpr,
    ):
        bh = tl.program_id(0)
        batch = bh // H
        head = bh % H
        p = tl.arange(0, BP)
        n = tl.arange(0, BN)
        p_mask = p < P
        n_mask = n < N

        u = tl.load(U + batch * H * P + head * P + p, p_mask, 0).to(tl.float32)
        b = tl.load(B + batch * H * N + head * N + n, n_mask, 0).to(tl.float32)
        c = tl.load(C + batch * H * N + head * N + n, n_mask, 0).to(tl.float32)
        dt = tl.load(DT + batch * H + head).to(tl.float32)
        trap = tl.load(TRAP + batch * H + head).to(tl.float32)
        a_cont = tl.load(A + head).to(tl.float32)
        offset = ((batch * H + head) * P + p[:, None]) * N + n[None, :]
        mask = p_mask[:, None] & n_mask[None, :]
        h_prev = tl.load(H_STATE + offset, mask, 0).to(tl.float32)
        bx_prev = tl.load(BX_PREV + offset, mask, 0).to(tl.float32)

        bx = u[:, None] * b[None, :]
        q = (1.0 - 0.5 * trap) * bx + (0.5 * trap) * bx_prev
        h_new = tl.exp(dt * a_cont) * h_prev + dt * q
        y = tl.sum(h_new * c[None, :], axis=1)
        tl.store(H_STATE + offset, h_new, mask)
        tl.store(BX_PREV + offset, bx, mask)
        tl.store(Y + batch * H * P + head * P + p, y, p_mask)


class _MambaScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, b, c, dt, trap, a):
        batch, steps, heads, width = u.shape
        state_size = b.shape[-1]
        y = torch.empty_like(u)
        states = torch.empty(
            (batch, steps, heads, width, state_size), device=u.device, dtype=u.dtype
        )
        bx_states = torch.empty_like(states)
        with torch.cuda.device(u.device):
            _mamba_scan_forward[(batch * heads,)](
                u, b, c, dt, trap, a, y, states, bx_states,
                steps, heads, width, state_size,
                triton.next_power_of_2(width), triton.next_power_of_2(state_size),
                num_warps=4,
            )
        ctx.save_for_backward(u, b, c, dt, trap, a, states, bx_states)
        ctx.shape = (batch, steps, heads, width, state_size)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        u, b, c, dt, trap, a, states, bx_states = ctx.saved_tensors
        batch, steps, heads, width, state_size = ctx.shape
        grads = [
            torch.empty_like(u), torch.empty_like(b), torch.empty_like(c),
            torch.empty_like(dt), torch.empty_like(trap), torch.zeros_like(a),
        ]
        with torch.cuda.device(u.device):
            _mamba_scan_backward[(batch * heads,)](
                u, b, c, dt, trap, a, grad_y.contiguous(), states, bx_states,
                *grads,
                steps, heads, width, state_size,
                triton.next_power_of_2(width), triton.next_power_of_2(state_size),
                num_warps=4,
            )
        return tuple(grads)


def mamba3_scan(
    u: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    dt: torch.Tensor,
    trap: torch.Tensor,
    a: torch.Tensor,
) -> torch.Tensor:
    """Run the Mamba-3 trapezoidal recurrent scan with a Triton autograd kernel."""
    tensors = (u, b, c, dt, trap, a)
    if not TRITON_AVAILABLE or not u.is_cuda:
        raise RuntimeError("mamba3_scan requires Triton and CUDA tensors")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("mamba3_scan inputs must be contiguous")
    if any(tensor.device != u.device for tensor in tensors):
        raise ValueError("mamba3_scan inputs must be on the same CUDA device")
    batch, steps, heads, width = u.shape
    state_size = b.shape[-1] if b.ndim == 4 else -1
    if (
        b.shape != (batch, steps, heads, state_size)
        or c.shape != b.shape
        or dt.shape != (batch, steps, heads)
        or trap.shape != dt.shape
        or a.shape != (heads,)
    ):
        raise ValueError("mamba3_scan received incompatible tensor shapes")
    return _MambaScan.apply(*tensors)


def mamba3_decode_step(
    u: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    dt: torch.Tensor,
    trap: torch.Tensor,
    a: torch.Tensor,
    h_state: torch.Tensor,
    bx_prev: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Fuse one Mamba-3 state update and readout; returns None without Triton/CUDA."""
    if not TRITON_AVAILABLE or not u.is_cuda:
        return None
    tensors = (u, b, c, dt, trap, a, h_state, bx_prev)
    if any(tensor.device != u.device for tensor in tensors):
        raise ValueError("mamba3_decode_step inputs must be on the same CUDA device")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("mamba3_decode_step inputs must be contiguous")
    batch, heads, width = u.shape
    state_size = b.shape[-1]
    if (
        b.shape != (batch, heads, state_size)
        or c.shape != b.shape
        or dt.shape != (batch, heads)
        or trap.shape != dt.shape
        or a.shape != (heads,)
        or h_state.shape != (batch, heads, width, state_size)
        or bx_prev.shape != h_state.shape
    ):
        raise ValueError("mamba3_decode_step received incompatible tensor shapes")
    y = torch.empty_like(u)
    with torch.cuda.device(u.device):
        _mamba_decode_step[(batch * heads,)](
            u, b, c, dt, trap, a, h_state, bx_prev, y,
            heads, width, state_size,
            triton.next_power_of_2(width), triton.next_power_of_2(state_size),
            num_warps=4,
        )
    return y
