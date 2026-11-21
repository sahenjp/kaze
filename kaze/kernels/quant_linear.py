"""Quantized linear kernels: weight-only int4 / int8 / fp16.

Decode (M <= 16) is a GEMV: weights are streamed exactly once, so the kernel is
bandwidth bound and the only things that matter are coalesced 32-bit nibble
loads, enough CTAs (split-K), and zero wasted traffic.  Prefill (M large) is a
tiled dequant-on-the-fly GEMM over the same layout.

Layout recap (see kaze/quant.py):
    packed : int32 / int8 [N, K/8 or K]
    scales : fp16 [N, K/GROUP]
    w[n, k] = (nibble - 8) * scales[n, k // GROUP]
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from kaze.quant import QuantLinear


@triton.jit
def _unpack_int4(wp, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """wp [BLOCK_N, BLOCK_K//8] int32 -> [BLOCK_N, BLOCK_K] fp16 in [-8, 7]."""
    shifts = tl.arange(0, 8) * 4
    nib = (wp[:, :, None] >> shifts[None, None, :]) & 0xF
    return tl.reshape(nib, (BLOCK_N, BLOCK_K)).to(tl.float16) - 8.0


# --------------------------------------------------------------------- decode
@triton.jit
def gemv_int4_kernel(X, WQ, SC, BIAS, Y, RES, N, K,
                     BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP: tl.constexpr,
                     TDIM: tl.constexpr, HAS_BIAS: tl.constexpr, HAS_RES: tl.constexpr):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    offs_kb = tl.arange(0, BLOCK_K // 2)
    if TDIM == 1:
        # batch 1: matvec over one row of X.  The nibble pairs stay split instead of being
        # interleaved back into a [BLOCK_N, BLOCK_K] tile, which also skips tl.interleave
        # (a layout conversion) and the 16-row dot.  Low nibble of byte j is k=2j, high
        # nibble is k=2j+1, so X is split the same way.
        acc1 = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            b = tl.load(WQ + offs_n[:, None] * (K // 2) + (k0 // 2) + offs_kb[None, :],
                        mask=mask_n[:, None], other=0)
            s = tl.load(SC + offs_n * (K // GROUP) + (k0 // GROUP), mask=mask_n, other=0.0)
            x = tl.load(X + k0 + tl.arange(0, BLOCK_K)).to(tl.float32)
            xe, xo = tl.split(tl.reshape(x, (BLOCK_K // 2, 2)))
            bi = b.to(tl.int32)
            raw = (tl.sum(((bi & 0xF).to(tl.float32) - 8.0) * xe[None, :], axis=1)
                   + tl.sum((((bi >> 4) & 0xF).to(tl.float32) - 8.0) * xo[None, :], axis=1))
            acc1 += raw * s
        if HAS_BIAS:
            acc1 += tl.load(BIAS + offs_n, mask=mask_n, other=0.0)
        if HAS_RES:
            acc1 += tl.load(RES + offs_n, mask=mask_n, other=0.0).to(tl.float32)
            tl.store(RES + offs_n, acc1.to(tl.float16), mask=mask_n)
        else:
            tl.store(Y + offs_n, acc1.to(tl.float16), mask=mask_n)
        return
    offs_m = tl.arange(0, TDIM)
    acc = tl.zeros((BLOCK_N, TDIM), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        b = tl.load(WQ + offs_n[:, None] * (K // 2) + (k0 // 2) + offs_kb[None, :],
                    mask=mask_n[:, None], other=0)
        s = tl.load(SC + offs_n * (K // GROUP) + (k0 // GROUP), mask=mask_n, other=0.0)
        bi = b.to(tl.int32)
        sb = s * 8.0
        lo = (bi & 0xF).to(tl.float16) * s[:, None] - sb[:, None]
        hi = ((bi >> 4) & 0xF).to(tl.float16) * s[:, None] - sb[:, None]
        w = tl.interleave(lo, hi)
        xs = tl.load(X + offs_m[:, None] * K + k0 + tl.arange(0, BLOCK_K)[None, :])
        acc += tl.dot(w, tl.trans(xs))
    if HAS_BIAS:
        acc += tl.load(BIAS + offs_n, mask=mask_n, other=0.0)[:, None]
    if HAS_RES:
        ptr = offs_m[None, :] * N + offs_n[:, None]
        acc += tl.load(RES + ptr, mask=mask_n[:, None], other=0.0).to(tl.float32)
        tl.store(RES + ptr, acc.to(tl.float16), mask=mask_n[:, None])
    else:
        tl.store(Y + offs_m[None, :] * N + offs_n[:, None], acc.to(tl.float16),
                 mask=mask_n[:, None])


@triton.jit
def gemv_int8_kernel(X, WQ, SC, BIAS, Y, RES, N, K,
                     BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP: tl.constexpr,
                     SUB: tl.constexpr,
                     TDIM: tl.constexpr, HAS_BIAS: tl.constexpr, HAS_RES: tl.constexpr):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    if TDIM == 1:
        # batch 1: one row of X, and no tl.dot.  tl.dot needs 16 output columns, so the
        # generic path computes 15 rows nobody reads AND makes every CTA re-read all 16
        # rows of X from L2 (for qkv that is more bytes than the weights).  The matvec
        # path reads X once and lets split-K fill the SMs: 5.4 -> 3.7 us on 0.5B qkv.
        acc1 = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for k0 in range(0, K, GROUP):
            raw = tl.zeros((BLOCK_N,), dtype=tl.float32)
            for j in tl.static_range(SUB):
                kk = k0 + j * BLOCK_K
                w = tl.load(WQ + offs_n[:, None] * K + kk + tl.arange(0, BLOCK_K)[None, :],
                            mask=mask_n[:, None], other=0).to(tl.float32)
                x = tl.load(X + kk + tl.arange(0, BLOCK_K)).to(tl.float32)
                raw += tl.sum(w * x[None, :], axis=1)
            s = tl.load(SC + offs_n * (K // GROUP) + (k0 // GROUP), mask=mask_n, other=0.0)
            acc1 += raw * s
        if HAS_BIAS:
            acc1 += tl.load(BIAS + offs_n, mask=mask_n, other=0.0)
        if HAS_RES:
            acc1 += tl.load(RES + offs_n, mask=mask_n, other=0.0).to(tl.float32)
            tl.store(RES + offs_n, acc1.to(tl.float16), mask=mask_n)
        else:
            tl.store(Y + offs_n, acc1.to(tl.float16), mask=mask_n)
        return
    offs_m = tl.arange(0, TDIM)
    acc = tl.zeros((BLOCK_N, TDIM), dtype=tl.float32)
    # one scale per GROUP lanes of K: accumulate the raw int8 products for the whole
    # group, then apply the scale once (instead of multiplying every weight element)
    for k0 in range(0, K, GROUP):
        acc_g = tl.zeros((BLOCK_N, TDIM), dtype=tl.float32)
        for j in tl.static_range(SUB):
            kk = k0 + j * BLOCK_K
            w = tl.load(WQ + offs_n[:, None] * K + kk + tl.arange(0, BLOCK_K)[None, :],
                        mask=mask_n[:, None], other=0).to(tl.float16)
            xs = tl.load(X + offs_m[:, None] * K + kk + tl.arange(0, BLOCK_K)[None, :])
            acc_g += tl.dot(w, tl.trans(xs))
        s = tl.load(SC + offs_n * (K // GROUP) + (k0 // GROUP), mask=mask_n, other=0.0)
        acc += acc_g * s[:, None]
    if HAS_BIAS:
        acc += tl.load(BIAS + offs_n, mask=mask_n, other=0.0)[:, None]
    if HAS_RES:
        ptr = offs_m[None, :] * N + offs_n[:, None]
        acc += tl.load(RES + ptr, mask=mask_n[:, None], other=0.0).to(tl.float32)
        tl.store(RES + ptr, acc.to(tl.float16), mask=mask_n[:, None])
    else:
        tl.store(Y + offs_m[None, :] * N + offs_n[:, None], acc.to(tl.float16),
                 mask=mask_n[:, None])


@triton.jit
def gemv_fp16_kernel(X, W, BIAS, Y, RES, N, K,
                     BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                     TDIM: tl.constexpr, HAS_BIAS: tl.constexpr, HAS_RES: tl.constexpr):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    offs_m = tl.arange(0, TDIM)
    acc = tl.zeros((BLOCK_N, TDIM), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        w = tl.load(W + offs_n[:, None] * K + offs_k[None, :],
                    mask=mask_n[:, None] & mask_k[None, :], other=0.0)
        xs = tl.load(X + offs_m[:, None] * K + offs_k[None, :],
                     mask=mask_k[None, :], other=0.0)
        acc += tl.dot(w, tl.trans(xs))
    if HAS_BIAS:
        acc += tl.load(BIAS + offs_n, mask=mask_n, other=0.0)[:, None]
    if HAS_RES:
        ptr = offs_m[None, :] * N + offs_n[:, None]
        acc += tl.load(RES + ptr, mask=mask_n[:, None], other=0.0).to(tl.float32)
        tl.store(RES + ptr, acc.to(tl.float16), mask=mask_n[:, None])
    else:
        tl.store(Y + offs_m[None, :] * N + offs_n[:, None], acc.to(tl.float16),
                 mask=mask_n[:, None])


# ----------------------------------------------------------------- split-K path
@triton.jit
def gemv_int4_splitk_kernel(X, WQ, SC, Y, N, K,
                            BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                            GROUP: tl.constexpr, TDIM: tl.constexpr,
                            SPLIT_K: tl.constexpr):
    pid = tl.program_id(0)
    sp = tl.program_id(1)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    offs_kb = tl.arange(0, BLOCK_K // 2)
    n_kblocks = K // BLOCK_K
    if TDIM == 1:
        acc1 = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for ki in range(sp, n_kblocks, SPLIT_K):
            k0 = ki * BLOCK_K
            b = tl.load(WQ + offs_n[:, None] * (K // 2) + (k0 // 2) + offs_kb[None, :],
                        mask=mask_n[:, None], other=0)
            s = tl.load(SC + offs_n * (K // GROUP) + (k0 // GROUP), mask=mask_n, other=0.0)
            x = tl.load(X + k0 + tl.arange(0, BLOCK_K)).to(tl.float32)
            xe, xo = tl.split(tl.reshape(x, (BLOCK_K // 2, 2)))
            bi = b.to(tl.int32)
            raw = (tl.sum(((bi & 0xF).to(tl.float32) - 8.0) * xe[None, :], axis=1)
                   + tl.sum((((bi >> 4) & 0xF).to(tl.float32) - 8.0) * xo[None, :], axis=1))
            acc1 += raw * s
        tl.store(Y + sp * N + offs_n, acc1, mask=mask_n)
        return
    offs_m = tl.arange(0, TDIM)
    acc = tl.zeros((BLOCK_N, TDIM), dtype=tl.float32)
    for ki in range(sp, n_kblocks, SPLIT_K):
        k0 = ki * BLOCK_K
        b = tl.load(WQ + offs_n[:, None] * (K // 2) + (k0 // 2) + offs_kb[None, :],
                    mask=mask_n[:, None], other=0)
        s = tl.load(SC + offs_n * (K // GROUP) + (k0 // GROUP), mask=mask_n, other=0.0)
        bi = b.to(tl.int32)
        sb = s * 8.0
        lo = (bi & 0xF).to(tl.float16) * s[:, None] - sb[:, None]
        hi = ((bi >> 4) & 0xF).to(tl.float16) * s[:, None] - sb[:, None]
        w = tl.interleave(lo, hi)
        xs = tl.load(X + offs_m[:, None] * K + k0 + tl.arange(0, BLOCK_K)[None, :])
        acc += tl.dot(w, tl.trans(xs))
    tl.store(Y + sp * (TDIM * N) + offs_m[None, :] * N + offs_n[:, None], acc,
             mask=mask_n[:, None])


@triton.jit
def gemv_int8_splitk_kernel(X, WQ, SC, Y, N, K,
                            BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                            GROUP: tl.constexpr, TDIM: tl.constexpr,
                            SPLIT_K: tl.constexpr):
    pid = tl.program_id(0)
    sp = tl.program_id(1)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    n_kblocks = K // BLOCK_K
    if TDIM == 1:
        # batch 1 matvec: same reasoning as gemv_int8_kernel, and the partial buffer the
        # epilogue reads shrinks from [SPLIT_K, 16, N] to [SPLIT_K, 1, N] fp32.
        acc1 = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for ki in range(sp, n_kblocks, SPLIT_K):
            k0 = ki * BLOCK_K
            w = tl.load(WQ + offs_n[:, None] * K + k0 + tl.arange(0, BLOCK_K)[None, :],
                        mask=mask_n[:, None], other=0).to(tl.float32)
            x = tl.load(X + k0 + tl.arange(0, BLOCK_K)).to(tl.float32)
            s = tl.load(SC + offs_n * (K // GROUP) + (k0 // GROUP), mask=mask_n, other=0.0)
            acc1 += tl.sum(w * x[None, :], axis=1) * s
        tl.store(Y + sp * N + offs_n, acc1, mask=mask_n)
        return
    offs_m = tl.arange(0, TDIM)
    acc = tl.zeros((BLOCK_N, TDIM), dtype=tl.float32)
    for ki in range(sp, n_kblocks, SPLIT_K):
        k0 = ki * BLOCK_K
        w = tl.load(WQ + offs_n[:, None] * K + k0 + tl.arange(0, BLOCK_K)[None, :],
                    mask=mask_n[:, None], other=0).to(tl.float16)
        s = tl.load(SC + offs_n * (K // GROUP) + (k0 // GROUP), mask=mask_n, other=0.0)
        w = w * s[:, None]
        xs = tl.load(X + offs_m[:, None] * K + k0 + tl.arange(0, BLOCK_K)[None, :])
        acc += tl.dot(w, tl.trans(xs))
    tl.store(Y + sp * (TDIM * N) + offs_m[None, :] * N + offs_n[:, None], acc,
             mask=mask_n[:, None])


@triton.jit
def gemv_epilogue_kernel(Y32, BIAS, Y, RES, N, BLOCK: tl.constexpr,
                         HAS_BIAS: tl.constexpr, HAS_RES: tl.constexpr,
                         TDIM: tl.constexpr = 16, SPLIT_K: tl.constexpr = 1):
    m = tl.program_id(0)
    n = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = n < N
    y = tl.zeros((BLOCK,), dtype=tl.float32)
    for sp in tl.static_range(SPLIT_K):
        y += tl.load(Y32 + sp * (TDIM * N) + m * N + n, mask=mask, other=0.0)
    if HAS_BIAS:
        y += tl.load(BIAS + n, mask=mask, other=0.0)
    if HAS_RES:
        y += tl.load(RES + m * N + n, mask=mask, other=0.0)
        tl.store(RES + m * N + n, y.to(tl.float16), mask=mask)
    else:
        tl.store(Y + m * N + n, y.to(tl.float16), mask=mask)


# ---------------------------------------------------------------------- prefill
@triton.jit
def gemm_int4_kernel(A, WQ, SC, BIAS, Y, M, N, K,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                     GROUP: tl.constexpr, HAS_BIAS: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    offs_k8 = tl.arange(0, BLOCK_K // 8)
    for k0 in range(0, K, BLOCK_K):
        wp = tl.load(WQ + offs_n[:, None] * (K // 8) + (k0 // 8) + offs_k8[None, :],
                     mask=mask_n[:, None], other=0)
        w = _unpack_int4(wp, BLOCK_N, BLOCK_K)
        s = tl.load(SC + offs_n * (K // GROUP) + (k0 // GROUP), mask=mask_n, other=0.0)
        w = w * s[:, None]
        a = tl.load(A + offs_m[:, None] * K + k0 + tl.arange(0, BLOCK_K)[None, :],
                    mask=mask_m[:, None], other=0.0)
        acc += tl.dot(a, tl.trans(w))
    if HAS_BIAS:
        acc += tl.load(BIAS + offs_n, mask=mask_n, other=0.0)[None, :]
    tl.store(Y + offs_m[:, None] * N + offs_n[None, :], acc.to(tl.float16),
             mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def gemm_int8_kernel(A, WQ, SC, BIAS, Y, M, N, K,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                     GROUP: tl.constexpr, HAS_BIAS: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        w = tl.load(WQ + offs_n[:, None] * K + k0 + tl.arange(0, BLOCK_K)[None, :],
                    mask=mask_n[:, None], other=0).to(tl.float16)
        s = tl.load(SC + offs_n * (K // GROUP) + (k0 // GROUP), mask=mask_n, other=0.0)
        w = w * s[:, None]
        a = tl.load(A + offs_m[:, None] * K + k0 + tl.arange(0, BLOCK_K)[None, :],
                    mask=mask_m[:, None], other=0.0)
        acc += tl.dot(a, tl.trans(w))
    if HAS_BIAS:
        acc += tl.load(BIAS + offs_n, mask=mask_n, other=0.0)[None, :]
    tl.store(Y + offs_m[:, None] * N + offs_n[None, :], acc.to(tl.float16),
             mask=mask_m[:, None] & mask_n[None, :])


# --------------------------------------------------------------------- dispatch
GEMM_WARPS = 4
GEMM_STAGES = 3
# int8 activations + integer tensor cores.  Correct to cos 0.99998 but it changes the
# exact bit pattern of every decode, which the engine's "verify == decode" invariants
# rely on, so it stays opt-in until a batched serving profile wants it.
W8A8_DECODE = False
# int4 weights + int8 activations (integer MMAs).  Three attempts, all neutral
# end-to-end: unfused (255.0 vs 254.9 on 1.5B), batch>=32 only (12,821 vs 12,852),
# and with the activation quantise fused into rmsnorm (253.7 vs 255.5 on 1.5B,
# 157.4 vs 157.9 on 3B).  The kernel is bandwidth bound, so cheaper ALU work buys
# nothing.  The fused path and the pre-quantised gemv arguments stay in place for a
# future kernel that actually needs int8 activations.
SPLITK_STAGES = 2      # stages 3 measured -3.4% (3B) / -2.4% (7B), so 2 stands
# "tuned" is the in-graph table in _cfg; "legacy" reproduces the launch-latency table so
# one env var (or set_cfg_mode from a benchmark) is enough to A/B the two end to end.
# "rows16cfg" keeps the new kernels but uses the 16-row table for batch 1 as well, so the
# batch-1 table can be A/B-ed on its own.
_CFG_MODES = ("tuned", "legacy", "rows16cfg")
# Cap on split-K for the wide buckets (batch >= 16).  The split-K partials are written and
# read back per layer, so at 32 rows their traffic is no longer negligible; this knob is
# only for A/B measurements.
_SPLITK_MAX = int(os.environ.get("KAZE_SPLITK_MAX", "8"))
_CFG_MODE = os.environ.get("KAZE_KERNEL_CFG", "tuned")


def set_cfg_mode(mode: str) -> None:
    """Switch the kernel config table in-process (for A/B; not used by the engine)."""
    global _CFG_MODE
    if mode not in _CFG_MODES:
        raise ValueError(f"unknown kernel cfg mode: {mode}")
    _CFG_MODE = mode
W4A8_DECODE = os.environ.get("KAZE_W4A8", "0") == "1"
# int8 weights on the integer tensor-core prefill path (int8 activations too).  Measured
# 0.5B int8: prefill 28.2k -> 51.6k tok/s, but the logits stop matching the fp16-activation
# path (layer-0 hidden cos 0.946, last hidden 0.347, logits 0.83) because RMSNorm outputs
# have outliers that one int8 scale per 128 lanes cannot hold.  int8 is the quality default
# (KL 0.0003), so this stays off until a finer activation scheme (per-32 or per-channel)
# fits the tensor-core layout.
W8A8_PREFILL = os.environ.get("KAZE_PREFILL_W8A8", "0") == "1"
# The down_proj input is silu(gate)*up, whose dynamic range is far worse than a RMSNorm
# output, so int8 activations there cost much more accuracy than they buy.  Keep it fp16
# unless explicitly enabled.
W8A8_DOWN_PREFILL = os.environ.get("KAZE_PREFILL_W8A8_DOWN", "0") == "1"
# Decode-time gate_up on int8 activations + int4 weights.  The RMSNorm output before
# gate_up has a tight range (unlike the down_proj input), so per-128-group int8 holds it:
# token-identical output over 8 streams x 24 tokens on 0.5B / 3B / 7B, and the kernel is
# 252 -> 208 us at TDIM=32 (bench/swiglu_w8a8_ab.py), worth +3.5% on 7B at 32 streams and
# neutral (0.15%) on 0.5B.  KAZE_W8A8_SWIGLU=0 restores the fp16-activation path.
W8A8_SWIGLU_DECODE = os.environ.get("KAZE_W8A8_SWIGLU", "1") == "1"


def _cfg_legacy(n: int, k: int, kind: str) -> tuple[int, int, int, int]:
    """The table that was tuned with launch-inclusive latency (kept for A/B only)."""
    if kind == "int4":
        if n >= 8192:
            return (128, 1, 4, 2) if k >= 2048 else (64, 1, 4, 2)
        if k >= 8192:
            return (64, 8, 4, 2)
        if k >= 4096:
            return (64, 4, 4, 2)
        return (64, 4, 4, 2)
    if n >= 65536:
        return (32, 1, 2, 4)
    if n >= 8192:
        return (64, 1, 4, 3)
    if k >= 4096:
        return (64, 8, 4, 2)
    if k > n:
        return (32, 1, 4, 3)
    return (64 if n >= 2048 else 32, 1, 4, 2)


def _cfg(n: int, k: int, kind: str, rows: int = 16) -> tuple[int, int, int, int, int]:
    """(block_n, split_k, num_warps, num_stages, block_k) for one decode GEMV.

    Retuned on Triton 3.9 / sm_86 with bench/kernel_lab.py, which times each kernel
    *inside a captured CUDA graph*.  That matters: with the launch cost included, split-K
    looked like a loss because its epilogue is a second launch, but inside a graph the
    epilogue is ~1 us and split-K is what fills 46 SMs for the narrow shapes (qkv, o_proj,
    down_proj).  Measured in-graph, per shape, 0.5B/1.5B/3B, int8 and int4:

        qkv   0.5B 7.4 -> 5.4 us, 1.5B 12.2 -> 8.6, 3B 18.3 -> 15.6
        o     0.5B 7.2 -> 4.9 us, 1.5B 12.8 -> 7.3, 3B 17.0 -> 11.9
        head  int4 0.5B 381 -> 186 us (the old (32,1,2,4) tile was the whole problem)

    The gate_up tile stays wide (128) at k >= 2048 for int4: that one was measured
    end-to-end in the engine in round 26 and end-to-end beats the microbenchmark there.
    """
    if _CFG_MODE == "legacy":
        return _cfg_legacy(n, k, kind) + (min(128, k),)
    if rows == 1 and _CFG_MODE != "rows16cfg":
        return _cfg_rows1(n, k, kind)
    if rows >= 32 and kind == "int4" and _CFG_MODE != "rows16cfg":
        return _cfg_rows32(n, k)
    if kind == "int4":
        if n >= 65536:                                  # lm_head: one pass over the vocab
            return (64, 1, 4, 2, 128) if k <= 1024 else (64, 1, 4, 4, 128)
        if n >= 4096 and n > k:                         # gate_up
            return (128, 1, 4, 2, 128) if k >= 2048 else (64, 1, 4, 2, 128)
        if k > 2 * n:                                   # down_proj: 14 CTAs -> 112
            return (64, 8, 4, 2, min(128, k))
        if k == n:                                      # o_proj
            return (64, 8, 4, 2, 64 if k >= 896 else 128)
        return (64, 8, 4, 2, 128)                       # qkv
    if n >= 65536:
        if k <= 1024:
            return (128, 1, 8, 2, 128)
        return (64, 1, 4, 2, 64) if k <= 1536 else (128, 1, 4, 4, 64)
    if n >= 4096 and n > k:                             # gate_up
        return (64, 1, 4, 2, 128) if k <= 1536 else (32, 1, 2, 3, 128)
    if k > 2 * n:                                       # down_proj
        if k <= 4864:
            return (32, 8, 2, 3, 64)
        return (64, 8, 4, 2, 128) if k <= 9000 else (64, 2, 4, 3, 128)
    if k == n:                                          # o_proj
        if n <= 1024:
            return (64, 8, 4, 2, 64)
        return (128, 8, 4, 2, 64) if n <= 1536 else (32, 1, 2, 3, 128)
    if n <= 1536:                                       # qkv
        return (64, 8, 4, 3, 128)
    return (64, 4, 4, 2, 128) if k <= 1536 else (64, 1, 4, 4, 128)


def _cfg_rows32(n: int, k: int) -> tuple[int, int, int, int, int]:
    """Shape config for the wide buckets (batch >= 32), int4 only.

    At 32+ rows the tl.dot has enough work that filling the SMs with split-K is no longer
    the win it is at batch 1: qkv prefers split_k=1 (which also removes its epilogue
    launch), o_proj wants 4 splits with a 32-row tile, down_proj keeps 8 but with 2 warps.
    Measured in-graph on 0.5B int4 (bench/kernel_lab.py micro --tdim 32):

        qkv   64/8/4/2 -> 32/1/4/3   7.5 -> 6.9 us
        o     64/8/4/2 -> 32/4/4/2   6.4 -> 6.0 us
        down  64/8/4/2 -> 64/8/2/2  15.4 -> 13.8 us
        gu    64/1/4/2 -> 64/1/4/3  22.3 -> 21.0 us
    """
    if n >= 65536:                                      # lm_head
        return (64, 1, 2, 3, 128) if k <= 1024 else (64, 1, 4, 3, 128)
    # 7B-scale shapes (hidden 4096): the 0.5B-derived branches below classify the 7B
    # lm_head (32000 x 4096) and qkv (6144 x 4096) as "gate_up" and give them a 64-column
    # tile.  Swept in-graph at tdim=32 (bench/sweep_batched_gemv_7b.py):
    #   head 32000x4096: (64,1,4,3) 308 -> (128,1,4,3) 240 us (-22%)
    #   qkv   6144x4096: (64,1,4,3)  86 -> (32,4,4,2)    64 us (-25%)
    #   down  4096x14336: the generic (64,8,2,2) is already the best of 9 tiles (132 us)
    if k >= 3072:
        if n >= 16000:                                  # 7B lm_head
            return (128, 1, 4, 3, 128)
        if k > 2 * n:                                   # 7B down_proj
            return (64, 8, 2, 2, 128)
        return (32, 4, 4, 2, 128)                       # 7B qkv / o_proj
    if n >= 4096 and n > k:                             # gate_up
        return (64, 1, 4, 3, 128)
    if k > 2 * n:                                       # down_proj
        return (64, 8, 2, 2, 128)
    if k == n:                                          # o_proj
        return (32, 4, 4, 2, 128)
    return (32, 1, 4, 3, 128)                           # qkv


def _cfg_rows1(n: int, k: int, kind: str) -> tuple[int, int, int, int, int]:
    """Shape config for the batch-1 matvec path (TDIM == 1).

    Separate table because the trade-off is different: with one row of X there is no
    16x activation re-read, so narrow tiles (block_n 16-32) win by giving the scheduler
    more CTAs.  Measured in-graph after the TDIM=1 kernels landed (0.5B int4):

        gate_up   64/1/4/2 -> 16/1/4/3   15.9 -> 12.5 us (+21%)
        down       64/8/4/2 -> 16/8/2/2   8.3 ->  7.4 us (+11%)
        qkv        64/8/4/2 -> 16/8/2/2   4.2 ->  3.9 us (+7%)
        o_proj     64/8/4/2 -> 32/8/4/2   3.8 ->  3.6 us (+5%)
    """
    if kind == "int4":
        if n >= 65536:                                  # lm_head
            return (32, 1, 8, 2, 128) if k <= 1024 else (64, 1, 4, 4, 128)
        if n >= 8192 and n > k:                         # gate_up (qkv n < 8192 everywhere)
            return (128, 1, 8, 2, 128) if k >= 2048 else (16, 1, 4, 3, 128)
        if k > 2 * n:                                   # down_proj
            # 7B-scale down (n >= 3072, k >= 3072): split-K 4 with 2 warps measured
            # in-graph 91.6 -> 77.9 us at [4096, 14336] (bench/ingraph_gemv_ab.py).
            if k >= 3072 and n >= 3072:
                return (32, 4, 2, 2, 128)
            # 3B-scale down (n = 2048, k = 11008): stages 2 beats 4 in-graph
            # (bench/ingraph_all_shapes.py: 35.9 -> 31.3 us); 1.5B (n = 1536) keeps 4.
            if n >= 2048 and k >= 8192:
                return (32, 8, 2, 2, 128)
            return (16, 8, 2, 2, 128) if n <= 1024 else (32, 8, 4, 2, 128)
        if k == n:                                      # o_proj
            return (16, 8, 2, 2, 128) if n >= 1536 else (32, 8, 4, 2, 64)
        return (16, 8, 2, 2, 128)                       # qkv
    if n >= 65536:                                      # lm_head
        return (128, 1, 8, 2, 128) if k <= 1024 else (64, 1, 4, 2, 64)
    if n >= 4096 and n > k:                             # gate_up
        return (32, 1, 8, 4, 128) if k >= 2048 else (16, 1, 2, 2, 128)
    if k > 2 * n:                                       # down_proj
        return (16, 8, 2, 2, 64) if k <= 4864 else (32, 8, 2, 2, 128)
    if k == n:                                          # o_proj
        return (16, 8, 4, 3, 128) if n >= 1536 else (16, 8, 4, 2, 128)
    if k >= 2048:                                       # qkv, deep K
        return (16, 8, 2, 3, 64)
    return (16, 8, 4, 2, 64)                            # qkv


def gemv(x: torch.Tensor, lin: QuantLinear, tdim: int, row_off: int = 0,
         n_override: int | None = None, block_n: int | None = None,
         block_k: int | None = None, out: torch.Tensor | None = None,
         acc: torch.Tensor | None = None, split_k: int | None = None,
         num_warps: int | None = None, add_to: torch.Tensor | None = None,
         xq: torch.Tensor | None = None,
         xs: torch.Tensor | None = None,
         defer_epilogue: bool = False) -> torch.Tensor:
    """x [tdim, K] fp16 -> [tdim, n] fp16 (rows >= batch size are padding).

    row_off / n_override address one expert inside a stacked [E*n, k] tensor.
    """
    n = n_override if n_override is not None else lin.n
    k = lin.k
    packed = lin.packed if row_off == 0 else lin.packed[row_off:]
    scales = lin.scales if (row_off == 0 or lin.scales is None) else lin.scales[row_off:]
    if out is None:
        out = torch.empty((tdim, n), dtype=torch.float16, device=x.device)
    cfg_bn, cfg_sk, cfg_nw, cfg_ns, cfg_bk = _cfg(n, k, lin.kind, rows=tdim)
    block_k = min(block_k or cfg_bk, lin.group, k)
    if lin.kind == "fp16":
        block_k = max(16, triton.next_power_of_2(block_k))
    # The table is per shape role, not per shape value: qkv / o_proj / gate_up / down_proj
    # / lm_head are told apart by (n, k) and each gets one config per weight kind.
    # Split-K is only used where the CTA count would otherwise leave SMs idle (o_proj,
    # down_proj, qkv on 0.5B); the epilogue is a second kernel but inside a CUDA graph it
    # costs ~1 us, and the partials are summed in a fixed order, so decode stays exact.
    # Round-59's lesson still holds for the *gate_up* tile, which is the one place where
    # the end-to-end measurement disagreed with the microbenchmark (see _cfg).
    if cfg_sk > _SPLITK_MAX:
        cfg_sk = _SPLITK_MAX
    if lin.kind == "fp16" or row_off:
        cfg_sk = 1
    split_k = split_k if split_k is not None else cfg_sk
    block_n = block_n or cfg_bn
    num_warps = num_warps or cfg_nw
    if split_k > 1:
        # deterministic partials: [SPLIT_K, tdim, n] fp32, overwritten every call
        if acc is None:
            key = (tdim, split_k)
            acc = lin.scratch.get(key)
            if acc is None or acc.shape != (split_k, tdim, n):
                acc = torch.zeros((split_k, tdim, n), dtype=torch.float32, device=x.device)
                lin.scratch[key] = acc
        kern = gemv_int4_splitk_kernel if lin.kind == "int4" else gemv_int8_splitk_kernel
        wptr = packed.view(torch.int8) if lin.kind == "int4" else packed
        kern[(triton.cdiv(n, block_n), split_k)](
            x, wptr, scales, acc, n, k, BLOCK_N=block_n, BLOCK_K=block_k,
            GROUP=lin.group, TDIM=tdim, SPLIT_K=split_k,
            num_warps=num_warps,
            num_stages=cfg_ns)
        if defer_epilogue:
            # The consumer (rmsnorm / rope_scatter) folds the partials in the same sp
            # ascending order this epilogue would, so the result is bit identical and one
            # launch per layer disappears (72 per decode step in a 24-layer model).
            # acc must be passed by the caller: the buffer has to be alive for the graph.
            if add_to is not None:
                raise ValueError("defer_epilogue leaves the residual to the consumer")
            return acc
        # 256-wide tiles leave only 4 CTAs for a 896-column output; the reduction is
        # latency bound, not bandwidth bound, so narrower tiles fill the device
        blk = 64 if n <= 2048 else 256
        gemv_epilogue_kernel[(tdim, triton.cdiv(n, blk))](
            acc, lin.bias if lin.bias is not None else out, out,
            add_to if add_to is not None else out, n, BLOCK=blk,
            HAS_BIAS=lin.bias is not None, HAS_RES=add_to is not None,
            TDIM=tdim, SPLIT_K=split_k, num_warps=4)
        return add_to if add_to is not None else out
    grid = (triton.cdiv(n, block_n),)
    bias = lin.bias if lin.bias is not None else out
    target = out if add_to is None else add_to
    res_ptr = add_to if add_to is not None else out
    # integer MMAs only pay off once the dot has enough rows: at TDIM=1 the matvec path
    # (no dot at all) is faster, so keep it for the batch-1 bucket
    w4a8 = lin.kind == "int4" and W4A8_DECODE and tdim >= 16
    if w4a8 and (xq is None or xs is None):
        xq, xs = quantize_act(x, lin.group)
    if w4a8:
        gemv_int4_w4a8_kernel[grid](xq, xs, packed.view(torch.int8), scales, bias, target,
                                    res_ptr, n, k,
                                    BLOCK_N=block_n, BLOCK_K=block_k, GROUP=lin.group,
                                    XGROUP=lin.group, TDIM=tdim,
                                    HAS_BIAS=lin.bias is not None,
                                    HAS_RES=add_to is not None,
                                    num_warps=num_warps, num_stages=cfg_ns)
    elif lin.kind == "int4":
        gemv_int4_kernel[grid](x, packed.view(torch.int8), scales, bias, target, res_ptr, n, k,
                               BLOCK_N=block_n, BLOCK_K=block_k, GROUP=lin.group,
                               TDIM=tdim, HAS_BIAS=lin.bias is not None,
                               HAS_RES=add_to is not None,
                               num_warps=num_warps, num_stages=cfg_ns)
    elif lin.kind == "int8" and tdim >= 16 and W8A8_DECODE:
        # at 16+ rows the fp16 dot is compute bound: quantise the activations and use
        # the integer tensor cores instead (the weights are already int8)
        xq, xs = quantize_act(x, 64)
        gemv_int8_w8a8_kernel[grid](xq, xs, packed, scales, bias, target, res_ptr, n, k,
                                    BLOCK_N=block_n, BLOCK_K=block_k, GROUP=lin.group,
                                    XGROUP=64, TDIM=tdim, HAS_BIAS=lin.bias is not None,
                                    HAS_RES=add_to is not None,
                                    num_warps=num_warps, num_stages=cfg_ns)
    elif lin.kind == "int8":
        gemv_int8_kernel[grid](x, packed, scales, bias, target, res_ptr, n, k,
                               BLOCK_N=block_n, BLOCK_K=block_k, GROUP=lin.group,
                               SUB=max(1, lin.group // block_k),
                               TDIM=tdim, HAS_BIAS=lin.bias is not None,
                               HAS_RES=add_to is not None,
                               num_warps=num_warps, num_stages=cfg_ns)
    else:
        gemv_fp16_kernel[grid](x, packed, bias, target, res_ptr, n, k,
                               BLOCK_N=block_n, BLOCK_K=block_k,
                               TDIM=tdim, HAS_BIAS=lin.bias is not None,
                               HAS_RES=add_to is not None,
                               num_warps=num_warps, num_stages=cfg_ns)
    return out


def gemm(a: torch.Tensor, lin: QuantLinear, block_m: int = 64, block_n: int | None = None,
         block_k: int = 64, row_off: int = 0, n_override: int | None = None,
         aq: torch.Tensor | None = None, as_: torch.Tensor | None = None,
         add_to: torch.Tensor | None = None, fp16_acts: bool = False) -> torch.Tensor:
    """Row offset / output-count override let one stacked tensor hold every expert."""
    m, k = a.shape
    n = n_override if n_override is not None else lin.n
    # Measured and rejected: routing narrow chunks to the split-K GEMV kernels is
    # SLOWER end to end (16-row prefill 1931 -> 1493 tok/s), because those kernels
    # serialize on split partials while the tiled GEMM still streams weights once.
    # NOTE: narrow-m tiles (16,128)/(64,32) are 1.4-2.5x faster per call at m=16 (down_proj
    # 173 -> 69 us), but the end-to-end effect could not be confirmed: a 16-row prefill
    # chunk stays at ~1.8k tok/s and the 32-request burst swung 0.39..1.98 s between runs
    # (the GPU idles at 210 MHz and a sub-second burst catches the clock ramp).  Keep the
    # tuned tile until the prefills themselves are batched; see docs/scorecard.md round 107.
    if block_n is None:
        block_n = 128 if n >= 4096 else 64
    if _GEMM_TILE is not None:
        block_m, block_n = _GEMM_TILE[0], _GEMM_TILE[1]
    if lin.kind == "fp16":
        wv = lin.packed[row_off:row_off + n]
        out = torch.nn.functional.linear(a, wv)
        if lin.bias is not None:
            out = out + lin.bias
        # every path must honour add_to: the caller relies on it for the residual, and a
        # silently dropped residual is a 0.5-cosine bug, not a small numeric difference
        return out + add_to if add_to is not None else out
    if (lin.kind in ("int4", "int8") and row_off == 0 and m >= 64 and not fp16_acts
            and (lin.kind == "int4" or W8A8_PREFILL)):
        # prefill is compute bound (~34 TFLOPS on the fp16 path); the integer tensor
        # cores reach ~47 TFLOPS, measured 25-40% faster per GEMM.  Both weight kinds use
        # the same kernel: int8 just skips the nibble unpack (IS_INT4=False).  The caller
        # may pass activations already quantised (fused into the producer).
        # BLOCK_K must equal one quantisation group, otherwise a single scale is applied
        # to two groups' worth of weights (group=64 gave KL 11.6 instead of 0.09).
        if aq is None or as_ is None:
            aq, as_ = quantize_act(a, lin.group)
        dest = (add_to if add_to is not None
                else torch.empty((m, n), dtype=torch.float16, device=a.device))
        grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
        gemm_int4_w8a8_kernel[grid](aq, as_, lin.packed.view(torch.int8), lin.scales,
                                    lin.bias if lin.bias is not None else dest, dest,
                                    m, n, k, BLOCK_M=block_m, BLOCK_N=block_n,
                                    BLOCK_K=lin.group, GROUP=lin.group, XGROUP=lin.group,
                                    HAS_BIAS=lin.bias is not None,
                                    HAS_RES=add_to is not None,
                                    IS_INT4=lin.kind == "int4",
                                    num_warps=_GEMM_TILE[2] if _GEMM_TILE else GEMM_WARPS,
                                    num_stages=_GEMM_TILE[3] if _GEMM_TILE else GEMM_STAGES)
        return dest
    packed = lin.packed if row_off == 0 else lin.packed[row_off:]
    scales = lin.scales if (row_off == 0 or lin.scales is None) else lin.scales[row_off:]
    if lin.group % block_k != 0:
        block_k = min(block_k, lin.group)
    out = torch.empty((m, n), dtype=torch.float16, device=a.device)
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    bias = lin.bias if lin.bias is not None else out
    kernel = gemm_int4_kernel if lin.kind == "int4" else gemm_int8_kernel
    kernel[grid](a, packed, scales, bias, out, m, n, k,
                 BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
                 GROUP=lin.group, HAS_BIAS=lin.bias is not None,
                 num_warps=GEMM_WARPS, num_stages=GEMM_STAGES)
    return out + add_to if add_to is not None else out


def quant_linear(x: torch.Tensor, lin: QuantLinear, tdim: int) -> torch.Tensor:
    if x.shape[0] <= 16:
        return gemv(x, lin, 16)
    return gemm(x, lin)


# ------------------------------------------------------------------ swiglu
@triton.jit
def gemv_int8_swiglu_kernel(X, WQ, SC, Y, NH, K,
                            BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                            GROUP: tl.constexpr, TDIM: tl.constexpr):
    """gate_up GEMV that writes silu(gate) * up directly.

    NH is half of N: rows [0, NH) are the gate, [NH, 2*NH) the up projection, so
    one program computes both halves of its output rows and the activation never
    round-trips through memory.
    """
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < NH
    if TDIM == 1:
        # batch 1: gate and up are two matvecs over the same single X row.  Same trick as
        # gemv_int8_kernel -- no tl.dot, one row of X, scale applied once per group.
        # evict_first on the weights is worth ~2% here (rounded off the same L2 the KV
        # cache wants); measured 25.2 -> 22.4 us on 0.5B gate_up.
        ag = tl.zeros((BLOCK_N,), dtype=tl.float32)
        au = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            gk = k0 // GROUP
            s_g = tl.load(SC + offs_n * (K // GROUP) + gk, mask=mask_n, other=0.0)
            s_u = tl.load(SC + (offs_n + NH) * (K // GROUP) + gk, mask=mask_n, other=0.0)
            pg = WQ + offs_n[:, None] * K + k0 + tl.arange(0, BLOCK_K)[None, :]
            pu = WQ + (offs_n + NH)[:, None] * K + k0 + tl.arange(0, BLOCK_K)[None, :]
            wg = tl.load(pg, mask=mask_n[:, None], other=0,
                         eviction_policy="evict_first").to(tl.float32)
            wu = tl.load(pu, mask=mask_n[:, None], other=0,
                         eviction_policy="evict_first").to(tl.float32)
            x = tl.load(X + k0 + tl.arange(0, BLOCK_K)).to(tl.float32)
            ag += tl.sum(wg * x[None, :], axis=1) * s_g
            au += tl.sum(wu * x[None, :], axis=1) * s_u
        act1 = ag * tl.sigmoid(ag) * au
        tl.store(Y + offs_n, act1.to(tl.float16), mask=mask_n)
        return
    offs_m = tl.arange(0, TDIM)
    acc_g = tl.zeros((BLOCK_N, TDIM), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_N, TDIM), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        gk = (k0 // GROUP)
        s_g = tl.load(SC + offs_n * (K // GROUP) + gk, mask=mask_n, other=0.0)
        s_u = tl.load(SC + (offs_n + NH) * (K // GROUP) + gk, mask=mask_n, other=0.0)
        wg = tl.load(WQ + offs_n[:, None] * K + k0 + tl.arange(0, BLOCK_K)[None, :],
                     mask=mask_n[:, None], other=0).to(tl.float16) * s_g[:, None]
        wu = tl.load(WQ + (offs_n + NH)[:, None] * K + k0 + tl.arange(0, BLOCK_K)[None, :],
                     mask=mask_n[:, None], other=0).to(tl.float16) * s_u[:, None]
        xs = tl.load(X + offs_m[:, None] * K + k0 + tl.arange(0, BLOCK_K)[None, :])
        xt = tl.trans(xs)
        acc_g += tl.dot(wg, xt)
        acc_u += tl.dot(wu, xt)
    act = acc_g * tl.sigmoid(acc_g) * acc_u
    tl.store(Y + offs_m[None, :] * NH + offs_n[:, None], act.to(tl.float16), mask=mask_n[:, None])


@triton.jit
def gemv_int4_swiglu_kernel(X, WB, SC, Y, NH, K,
                            BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                            GROUP: tl.constexpr, TDIM: tl.constexpr):
    """int4 gate_up GEMV that writes silu(gate) * up directly (interleaved unpack)."""
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < NH
    offs_kb = tl.arange(0, BLOCK_K // 2)
    if TDIM == 1:
        # batch 1: two matvecs over one row of X, nibble pairs kept split (no interleave)
        ag = tl.zeros((BLOCK_N,), dtype=tl.float32)
        au = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            gk = k0 // GROUP
            kb0 = k0 // 2
            s_g = tl.load(SC + offs_n * (K // GROUP) + gk, mask=mask_n, other=0.0)
            s_u = tl.load(SC + (offs_n + NH) * (K // GROUP) + gk, mask=mask_n, other=0.0)
            bg = tl.load(WB + offs_n[:, None] * (K // 2) + kb0 + offs_kb[None, :],
                         mask=mask_n[:, None], other=0,
                         eviction_policy="evict_first")
            bu = tl.load(WB + (offs_n + NH)[:, None] * (K // 2) + kb0 + offs_kb[None, :],
                         mask=mask_n[:, None], other=0,
                         eviction_policy="evict_first")
            x = tl.load(X + k0 + tl.arange(0, BLOCK_K)).to(tl.float32)
            xe, xo = tl.split(tl.reshape(x, (BLOCK_K // 2, 2)))
            gi = bg.to(tl.int32)
            ui = bu.to(tl.int32)
            ag += (tl.sum(((gi & 0xF).to(tl.float32) - 8.0) * xe[None, :], axis=1)
                   + tl.sum(((((gi >> 4) & 0xF).to(tl.float32)) - 8.0) * xo[None, :], axis=1)) * s_g
            au += (tl.sum(((ui & 0xF).to(tl.float32) - 8.0) * xe[None, :], axis=1)
                   + tl.sum(((((ui >> 4) & 0xF).to(tl.float32)) - 8.0) * xo[None, :], axis=1)) * s_u
        act1 = ag * tl.sigmoid(ag) * au
        tl.store(Y + offs_n, act1.to(tl.float16), mask=mask_n)
        return
    offs_m = tl.arange(0, TDIM)
    acc_g = tl.zeros((BLOCK_N, TDIM), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_N, TDIM), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        gk = k0 // GROUP
        kb0 = k0 // 2
        s_g = tl.load(SC + offs_n * (K // GROUP) + gk, mask=mask_n, other=0.0)
        s_u = tl.load(SC + (offs_n + NH) * (K // GROUP) + gk, mask=mask_n, other=0.0)
        bg = tl.load(WB + offs_n[:, None] * (K // 2) + kb0 + offs_kb[None, :],
                     mask=mask_n[:, None], other=0)
        bu = tl.load(WB + (offs_n + NH)[:, None] * (K // 2) + kb0 + offs_kb[None, :],
                     mask=mask_n[:, None], other=0)
        gi = bg.to(tl.int32)
        ui = bu.to(tl.int32)
        sbg = s_g * 8.0
        sbu = s_u * 8.0
        wg = tl.interleave((gi & 0xF).to(tl.float16) * s_g[:, None] - sbg[:, None],
                           ((gi >> 4) & 0xF).to(tl.float16) * s_g[:, None] - sbg[:, None])
        wu = tl.interleave((ui & 0xF).to(tl.float16) * s_u[:, None] - sbu[:, None],
                           ((ui >> 4) & 0xF).to(tl.float16) * s_u[:, None] - sbu[:, None])
        xs = tl.load(X + offs_m[:, None] * K + k0 + tl.arange(0, BLOCK_K)[None, :])
        xt = tl.trans(xs)
        acc_g += tl.dot(wg, xt)
        acc_u += tl.dot(wu, xt)
    act = acc_g * tl.sigmoid(acc_g) * acc_u
    tl.store(Y + offs_m[None, :] * NH + offs_n[:, None], act.to(tl.float16), mask=mask_n[:, None])


@triton.jit
def gemm_swiglu_kernel(A, WQ, SC, Y, M, NH, K,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                       GROUP: tl.constexpr, IS_INT4: tl.constexpr):
    """gate_up GEMM that writes silu(gate) * up, so the prefill never round-trips
    the 2*intermediate activation through memory (half a layer's bytes)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < NH
    acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    offs_kb = tl.arange(0, BLOCK_K // 2)
    for k0 in range(0, K, BLOCK_K):
        if IS_INT4:
            kb = k0 // 2
            bg = tl.load(WQ + offs_n[:, None] * (K // 2) + kb + offs_kb[None, :],
                         mask=mask_n[:, None], other=0)
            bu = tl.load(WQ + (offs_n + NH)[:, None] * (K // 2) + kb + offs_kb[None, :],
                         mask=mask_n[:, None], other=0)
            sg = tl.load(SC + offs_n * (K // GROUP) + (k0 // GROUP), mask=mask_n, other=0.0)
            su = tl.load(SC + (offs_n + NH) * (K // GROUP) + (k0 // GROUP), mask=mask_n, other=0.0)
            gi = bg.to(tl.int32)
            ui = bu.to(tl.int32)
            sbg = sg * 8.0
            sbu = su * 8.0
            wg = tl.interleave((gi & 0xF).to(tl.float16) * sg[:, None] - sbg[:, None],
                               ((gi >> 4) & 0xF).to(tl.float16) * sg[:, None] - sbg[:, None])
            wu = tl.interleave((ui & 0xF).to(tl.float16) * su[:, None] - sbu[:, None],
                               ((ui >> 4) & 0xF).to(tl.float16) * su[:, None] - sbu[:, None])
        else:
            wg = tl.load(WQ + offs_n[:, None] * K + k0 + tl.arange(0, BLOCK_K)[None, :],
                         mask=mask_n[:, None], other=0).to(tl.float16)
            wu = tl.load(WQ + (offs_n + NH)[:, None] * K + k0 + tl.arange(0, BLOCK_K)[None, :],
                         mask=mask_n[:, None], other=0).to(tl.float16)
            sg = tl.load(SC + offs_n * (K // GROUP) + (k0 // GROUP), mask=mask_n, other=0.0)
            su = tl.load(SC + (offs_n + NH) * (K // GROUP) + (k0 // GROUP), mask=mask_n, other=0.0)
            wg = wg * sg[:, None]
            wu = wu * su[:, None]
        a = tl.load(A + offs_m[:, None] * K + k0 + tl.arange(0, BLOCK_K)[None, :],
                    mask=mask_m[:, None], other=0.0)
        acc_g += tl.dot(a, tl.trans(wg))
        acc_u += tl.dot(a, tl.trans(wu))
    act = acc_g * tl.sigmoid(acc_g) * acc_u
    tl.store(Y + offs_m[:, None] * NH + offs_n[None, :], act.to(tl.float16),
             mask=mask_m[:, None] & mask_n[None, :])


def gemm_swiglu(a: torch.Tensor, lin: QuantLinear, nh: int,
                block_m: int = 64, block_n: int = 64, block_k: int = 64,
                out: torch.Tensor | None = None,
                aq: torch.Tensor | None = None,
                as_: torch.Tensor | None = None,
                out_q: torch.Tensor | None = None,
                out_s: torch.Tensor | None = None) -> torch.Tensor:
    """a [M, K] -> act [M, nh] with act = silu(gate) * up (gate_up layout)."""
    m, k = a.shape
    if out is not None and out.shape[0] != m:
        raise ValueError(f"out rows {out.shape[0]} != {m}")
    if (lin.kind in ("int4", "int8") and m >= 64
            and (lin.kind == "int4" or W8A8_PREFILL)):
        # fused gate_up on integer tensor cores.  BLOCK_M=32 was the old ceiling: with 64
        # the two int8 dots plus the nibble unpack spilled registers at 4 warps and the
        # kernel ran 40% slower than fp16; with 8 warps (64,128,8,2) it is the fastest
        # tile.  int8 weights skip the unpack entirely.
        if aq is None or as_ is None:
            aq, as_ = quantize_act(a, lin.group)
        nh = lin.n // 2
        out = torch.empty((m, nh), dtype=torch.float16, device=a.device)
        # Tile measured in-graph at m=512 (bench/gemm_bench.py): (64, 128, 8 warps, 2 stages)
        # 150 us vs 183 us for (32, 64, 4, 2) on 0.5B gate_up, and the two int8 dots still
        # fit the register budget at BLOCK_M=64 with 8 warps.  KAZE_SWIGLU_TILE overrides
        # it for A/B measurements ("bm,bn,warps,stages").
        bm, bn, nw, ns = _SWIGLU_TILE
        grid = (triton.cdiv(m, bm), triton.cdiv(nh, bn))
        # the fused int8 copy only lines up with the down_proj groups when the tile is one
        # group wide; fall back to a separate quantize_act otherwise
        has_outq = out_q is not None and out_s is not None and bn == lin.group
        gemm_swiglu_w8a8_kernel[grid](aq, as_, lin.packed.view(torch.int8), lin.scales, out,
                                      out_q if has_outq else out,
                                      out_s if has_outq else lin.scales,
                                      m, nh, k, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=lin.group,
                                      GROUP=lin.group, XGROUP=lin.group,
                                      IS_INT4=lin.kind == "int4", HAS_OUTQ=has_outq,
                                      num_warps=nw, num_stages=ns)
        return out
    if out is None:
        out = torch.empty((m, nh), dtype=torch.float16, device=a.device)
    if lin.kind == "fp16":
        gu = a @ lin.packed.t()
        g, u = gu[:, :nh], gu[:, nh:]
        return (g * torch.sigmoid(g) * u).to(torch.float16)
    block_k = min(block_k, lin.group, k)
    w = lin.packed if lin.kind == "int8" else lin.packed.view(torch.int8)
    gemm_swiglu_kernel[(triton.cdiv(m, block_m), triton.cdiv(nh, block_n))](
        a, w, lin.scales, out, m, nh, k,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, GROUP=lin.group,
        IS_INT4=(lin.kind == "int4"), num_warps=4, num_stages=3)
    return out


def gemv_swiglu(x: torch.Tensor, lin: QuantLinear, tdim: int, nh: int,
                block_n: int = 64, block_k: int | None = None,
                out: torch.Tensor | None = None, num_warps: int = 4,
                num_stages: int = 2, xq: torch.Tensor | None = None,
                xs: torch.Tensor | None = None) -> torch.Tensor:
    """x [tdim, K] -> act [tdim, nh] with act = silu(gate) * up."""
    k = lin.k
    block_k = min(block_k or lin.group, lin.group, k)
    if out is None:
        out = torch.empty((tdim, nh), dtype=torch.float16, device=x.device)
    if lin.kind == "int8":
        gemv_int8_swiglu_kernel[(triton.cdiv(nh, block_n),)](
            x, lin.packed, lin.scales, out, nh, k, BLOCK_N=block_n, BLOCK_K=block_k,
            GROUP=lin.group, TDIM=tdim, num_warps=num_warps, num_stages=num_stages)
        return out
    if lin.kind == "int4" and xq is not None and xs is not None:
        gemv_int4_swiglu_w8a8_kernel[(triton.cdiv(nh, 64),)](
            xq, xs, lin.packed.view(torch.int8), lin.scales, out, nh, k,
            BLOCK_N=64, BLOCK_K=lin.group, GROUP=lin.group, XGROUP=lin.group,
            TDIM=tdim, num_warps=4, num_stages=2)
        return out
    if lin.kind == "int4":
        cfg_bn, _, cfg_nw, cfg_ns, cfg_bk = _cfg(2 * nh, k, lin.kind, rows=tdim)
        # each iteration loads only BLOCK_N x 64 words (the 128-element quantisation
        # group caps BLOCK_K), so deeper pipelining is the only way to keep more loads
        # in flight.  Measured: 7B (nh=14336) 62.0 -> 64.1 tok/s with stages 4;
        # 0.5B (nh=4864) 640 -> 601, so only wide gate_up tensors get the deeper pipe.
        # bn=32 for small nh (more CTAs) measured -11% end to end: every CTA re-reads
        # the whole activation row, so halving the tile doubles that traffic.
        # bn=128 is measured per model below (7B has enough columns to fill it).
        # 7B-scale gate_up (nh >= 14000) is the exception to the bn=128 rule, but the best
        # tile depends on the batch: at TDIM=1 the sweep gives (32,4,4) 191 vs 231 us
        # (-17%), while at TDIM=32 the same tile costs 355 vs 253 us for (64,4,4)
        # (+40%) because a 32-column tile is too narrow for the dot GEMM.
        # (bench/swiglu_ingraph_ab.py for tdim=1, bench/swiglu_batched_ab.py for 16/32.)
        if nh >= 14000:
            if tdim >= 16:
                cfg_bn, cfg_nw, cfg_ns = 64, 4, 4
            else:
                cfg_bn, cfg_nw, cfg_ns = 32, 4, 4
        else:
            cfg_ns = 2
        gemv_int4_swiglu_kernel[(triton.cdiv(nh, cfg_bn),)](
            x, lin.packed.view(torch.int8), lin.scales, out, nh, k, BLOCK_N=cfg_bn,
            BLOCK_K=block_k, GROUP=lin.group, TDIM=tdim, num_warps=cfg_nw,
            num_stages=cfg_ns)
        return out
    # fallback: two GEMVs + silu
    gate = gemv(x, lin, tdim)
    g = gate[:, :nh].contiguous()
    u = gate[:, nh:].contiguous()
    from kaze.kernels.act import silu_mul
    return silu_mul(g, u, out=out)


@triton.jit
def moe_router_kernel(X, R, IDX, W, E, H, K,
                      BLOCK_H: tl.constexpr, BLOCK_E: tl.constexpr, TDIM: tl.constexpr,
                      NORM_TOPK: tl.constexpr):
    """Router logits + top-k in one pass, writing indices and (normalised) weights.

    Everything is preallocated so the MoE decode step can be captured in a CUDA
    graph; torch.topk would allocate and break the capture.
    """
    row = tl.program_id(0)
    offs_h = tl.arange(0, BLOCK_H)
    mask_h = offs_h < H
    x = tl.load(X + row * H + offs_h, mask=mask_h, other=0.0).to(tl.float32)
    offs_e = tl.arange(0, BLOCK_E)
    mask_e = offs_e < E
    acc = tl.zeros((BLOCK_E,), tl.float32)
    for h0 in range(0, H, BLOCK_H):
        r = tl.load(R + offs_e[:, None] * H + h0 + offs_h[None, :],
                    mask=mask_e[:, None] & ((h0 + offs_h)[None, :] < H), other=0.0).to(tl.float32)
        xv = tl.load(X + row * H + h0 + offs_h, mask=(h0 + offs_h) < H, other=0.0).to(tl.float32)
        acc += tl.sum(r * xv[None, :], axis=1)
    logits = tl.where(mask_e, acc, -float("inf"))
    probs = tl.exp(logits - tl.max(logits, axis=0))
    logits = probs / tl.sum(probs, axis=0)
    total = 0.0
    for j in range(K):
        best = tl.max(logits, axis=0)
        bidx = tl.argmax(logits, axis=0)
        tl.store(IDX + row * K + j, bidx.to(tl.int32))
        tl.store(W + row * K + j, best)
        logits = tl.where(offs_e == bidx, -1e30, logits)
        total += best
    if NORM_TOPK:
        for j in range(K):
            v = tl.load(W + row * K + j)
            tl.store(W + row * K + j, v / total)


# ------------------------------------------------------------------ MoE
@triton.jit
def moe_gemv_int8_kernel(X, WQ, SC, IDX, OUT, N, K, NEXP,
                         BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                         GROUP: tl.constexpr, PER_ROW: tl.constexpr,
                         SAME_ROWS: tl.constexpr, IS_FP16: tl.constexpr = False):
    """One program computes BLOCK_N outputs of one (token, expert) pair.

    MoE decode is bandwidth bound: only the selected experts are read, so the
    cost per token is top_k * expert size, and an FMA reduction is enough -- the
    tensor cores would be idle anyway at batch 1.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    e = tl.load(IDX + pid_m).to(tl.int64)
    if SAME_ROWS:
        x_row = pid_m
    else:
        x_row = pid_m // PER_ROW
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    acc = tl.zeros((BLOCK_N,), tl.float32)
    base = e * N * K
    sbase = e * N * (K // GROUP)
    for k0 in range(0, K, BLOCK_K):
        w = tl.load(WQ + base + offs_n[:, None] * K + k0 + tl.arange(0, BLOCK_K)[None, :],
                    mask=mask_n[:, None], other=0).to(tl.float16)
        if not IS_FP16:
            s = tl.load(SC + sbase + offs_n * (K // GROUP) + (k0 // GROUP), mask=mask_n, other=0.0)
            w = w * s[:, None]
        xv = tl.load(X + x_row * K + k0 + tl.arange(0, BLOCK_K))
        acc += tl.sum(w.to(tl.float32) * xv[None, :], axis=1)
    tl.store(OUT + pid_m * N + offs_n, acc.to(tl.float16), mask=mask_n)


def moe_gemv(x: torch.Tensor, lin: QuantLinear, idx: torch.Tensor, n_exp: int,
             n_out: int, same_rows: bool = False, per_row: int = 1,
             block_n: int = 128, out: torch.Tensor | None = None) -> torch.Tensor:
    """x [rows, K] with idx [rows*per_row] expert ids -> [rows*per_row, n_out]."""
    pairs = idx.numel()
    if out is None:
        out = torch.empty((pairs, n_out), dtype=torch.float16, device=x.device)
    moe_gemv_int8_kernel[(pairs, triton.cdiv(n_out, block_n))](
        x, lin.packed, lin.scales if lin.scales is not None else lin.packed, idx, out, n_out,
        lin.k, n_exp, BLOCK_N=block_n, BLOCK_K=min(lin.group, 128, lin.k), GROUP=lin.group,
        PER_ROW=per_row, SAME_ROWS=same_rows, IS_FP16=(lin.kind == "fp16"),
        num_warps=2, num_stages=2)
    return out

def moe_router(x: torch.Tensor, router: torch.Tensor, k: int, n_exp: int,
               norm_topk: bool, idx: torch.Tensor, w: torch.Tensor) -> None:
    """x [tdim, H] -> idx/w [tdim*k] (in place, no allocations)."""
    tdim, h = x.shape
    moe_router_kernel[(tdim,)](
        x, router, idx, w, n_exp, h, k,
        BLOCK_H=triton.next_power_of_2(min(h, 1024)),
        BLOCK_E=triton.next_power_of_2(n_exp), TDIM=tdim,
        NORM_TOPK=norm_topk, num_warps=4)


@triton.jit
def quant_act_kernel(X, Q, SC, M, K, BLOCK_K: tl.constexpr, GROUP: tl.constexpr,
                     TDIM: tl.constexpr):
    """int8 quantisation of the activations that feed the W8A8 prefill GEMMs.

    One program covers BLOCK_K columns of one row and computes one amax per GROUP inside
    that tile, instead of a program per (row, group).  The old shape (one 128-column
    program, 32 threads) made a 512-token prefill spend 998 us here - 8% of the step -
    because each thread moved 8 bytes and the grid was 3.5k tiny CTAs.
    """
    pid = tl.program_id(0)          # row
    pidk = tl.program_id(1)         # k block
    offs_k = pidk * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = offs_k < K
    x = tl.load(X + pid * K + offs_k, mask=mask, other=0.0).to(tl.float32)
    ng: tl.constexpr = BLOCK_K // GROUP
    xg = tl.reshape(x, (ng, GROUP))
    s = tl.maximum(tl.max(tl.abs(xg), axis=1), 1e-8) / 127.0
    v = xg / s[:, None]
    q = tl.reshape((v + tl.where(v >= 0, 0.5, -0.5)).to(tl.int8), (BLOCK_K,))
    tl.store(Q + pid * K + offs_k, q, mask=mask)
    g0 = pidk * ng
    # mask the tail: when K is not a multiple of BLOCK_K the last tile owns fewer than ng
    # groups, and an unmasked store wrote into the next row's first scale (a race that
    # made the prefill GEMMs nondeterministic: identical calls, different logits)
    gidx = g0 + tl.arange(0, ng)
    tl.store(SC + pid * tl.cdiv(K, GROUP) + gidx, s, mask=gidx < tl.cdiv(K, GROUP))


def _tile_from_env(name: str, default: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    raw = os.environ.get(name)
    if not raw:
        return default
    bm, bn, nw, ns = (int(v) for v in raw.split(","))
    return bm, bn, nw, ns


# prefill gate_up tile (block_m, block_n, num_warps, num_stages)
_SWIGLU_TILE = _tile_from_env("KAZE_SWIGLU_TILE", (64, 128, 8, 2))
# prefill GEMM tile (block_m, block_n, num_warps, num_stages).  Measured end to end on
# 0.5B int4 prefill: shape-derived default (block_n 64 at n=896) 43.7k tok/s -> (64,128,4,3)
# 46.4k (+6%).  BLOCK_M 128 loses 20% (14.4k -> 35.5k) because the tile no longer divides
# the m=512 chunk evenly across 46 SMs.  KAZE_GEMM_TILE overrides it for A/B.
# round 147: stages 3 -> 2, swept in-graph at every big layer (bench/ingraph_prefill_ab.py,
# stages 2 wins 1.3-6.7% per kernel) and confirmed end to end: 0.5B 47.3k -> 48.2k (+2.1%),
# 3B 10.2k -> 10.3k (+1.0%), interleaved A/B with separated IQRs.
_GEMM_TILE = (lambda raw: tuple(int(v) for v in raw.split(",")) if raw else (64, 128, 4, 2))(
    os.environ.get("KAZE_GEMM_TILE"))


def quantize_act(x: torch.Tensor, group: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    m, k = x.shape
    q = torch.empty((m, k), dtype=torch.int8, device=x.device)
    sc = torch.empty((m, (k + group - 1) // group), dtype=torch.float32, device=x.device)
    # 512-column tile (4 groups of 128) with 4 warps: 16 B per thread of coherent load and
    # a quarter of the launches.  The tile must be a multiple of the scale group.
    block_k = max(group, 512 // group * group)
    block_k = min(block_k, max(group, triton.next_power_of_2(k)))
    quant_act_kernel[(m, triton.cdiv(k, block_k))](x, q, sc, m, k,
                                                   BLOCK_K=block_k, GROUP=group,
                                                   TDIM=m, num_warps=4)
    return q, sc


@triton.jit
def gemv_int8_w8a8_kernel(XQ, XS, WQ, SC, BIAS, Y, RES, N, K,
                          BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                          GROUP: tl.constexpr, XGROUP: tl.constexpr,
                          TDIM: tl.constexpr, HAS_BIAS: tl.constexpr,
                          HAS_RES: tl.constexpr):
    """Decode GEMV with int8 weights *and* int8 activations (integer MMA)."""
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    offs_m = tl.arange(0, TDIM)
    acc = tl.zeros((BLOCK_N, TDIM), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        w = tl.load(WQ + offs_n[:, None] * K + k0 + tl.arange(0, BLOCK_K)[None, :],
                    mask=mask_n[:, None], other=0).to(tl.int8)
        xq = tl.load(XQ + offs_m[:, None] * K + k0 + tl.arange(0, BLOCK_K)[None, :])
        dots = tl.dot(w, tl.trans(xq), out_dtype=tl.int32).to(tl.float32)
        ws = tl.load(SC + offs_n * (K // GROUP) + (k0 // GROUP), mask=mask_n, other=0.0)
        xs = tl.load(XS + offs_m * (K // XGROUP) + (k0 // XGROUP))
        acc += dots * ws[:, None] * xs[None, :]
    if HAS_BIAS:
        acc += tl.load(BIAS + offs_n, mask=mask_n, other=0.0)[:, None]
    if HAS_RES:
        ptr = offs_m[None, :] * N + offs_n[:, None]
        acc += tl.load(RES + ptr, mask=mask_n[:, None], other=0.0).to(tl.float32)
        tl.store(RES + ptr, acc.to(tl.float16), mask=mask_n[:, None])
    else:
        tl.store(Y + offs_m[None, :] * N + offs_n[:, None], acc.to(tl.float16),
                 mask=mask_n[:, None])


@triton.jit
def gemv_int4_w4a8_kernel(XQ, XS, WQ, SC, BIAS, Y, RES, N, K,
                          BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                          GROUP: tl.constexpr, XGROUP: tl.constexpr,
                          TDIM: tl.constexpr, HAS_BIAS: tl.constexpr,
                          HAS_RES: tl.constexpr):
    """int4 weights with int8 activations: nibbles unpack to int8 and feed the integer
    MMAs, so the weights never touch fp16 and the scale is applied once per group."""
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    offs_m = tl.arange(0, TDIM)
    acc = tl.zeros((BLOCK_N, TDIM), dtype=tl.float32)
    half: tl.constexpr = BLOCK_K // 2
    for k0 in range(0, K, BLOCK_K):
        b = tl.load(WQ + offs_n[:, None] * (K // 2) + (k0 // 2) + tl.arange(0, half)[None, :],
                    mask=mask_n[:, None], other=0)
        bi = b.to(tl.int32)
        lo = (bi & 0xF).to(tl.int8) - 8
        hi = ((bi >> 4) & 0xF).to(tl.int8) - 8
        w_i8 = tl.interleave(lo, hi)
        xq = tl.load(XQ + offs_m[:, None] * K + k0 + tl.arange(0, BLOCK_K)[None, :])
        dots = tl.dot(w_i8, tl.trans(xq), out_dtype=tl.int32).to(tl.float32)
        s = tl.load(SC + offs_n * (K // GROUP) + (k0 // GROUP), mask=mask_n, other=0.0)
        xs = tl.load(XS + offs_m * (K // XGROUP) + (k0 // XGROUP))
        acc += dots * s[:, None] * xs[None, :]
    if HAS_BIAS:
        acc += tl.load(BIAS + offs_n, mask=mask_n, other=0.0)[:, None]
    if HAS_RES:
        ptr = offs_m[None, :] * N + offs_n[:, None]
        acc += tl.load(RES + ptr, mask=mask_n[:, None], other=0.0).to(tl.float32)
        tl.store(RES + ptr, acc.to(tl.float16), mask=mask_n[:, None])
    else:
        tl.store(Y + offs_m[None, :] * N + offs_n[:, None], acc.to(tl.float16),
                 mask=mask_n[:, None])


@triton.jit
def gemm_int4_w8a8_kernel(AQ, AS, WQ, SC, BIAS, Y, M, N, K,
                          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                          BLOCK_K: tl.constexpr, GROUP: tl.constexpr,
                          XGROUP: tl.constexpr, HAS_BIAS: tl.constexpr,
                          HAS_RES: tl.constexpr = False, IS_INT4: tl.constexpr = True):
    """Prefill GEMM on integer tensor cores: int4/int8 weights, int8 activations.

    Prefill is compute bound (measured ~70% of the fp16 tensor peak), and the integer
    path doubles the math throughput.  Activations are quantised per row per XGROUP
    lanes outside the kernel; the scales are applied to the int32 accumulator, so the
    inner loop is pure integer MMA plus the nibble unpack.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_n = offs_n < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    half: tl.constexpr = BLOCK_K // 2
    for k0 in range(0, K, BLOCK_K):
        aq = tl.load(AQ + offs_m[:, None] * K + k0 + offs_k[None, :],
                     mask=mask_m[:, None], other=0).to(tl.int8)
        if IS_INT4:
            b = tl.load(WQ + offs_n[:, None] * (K // 2) + (k0 // 2)
                        + tl.arange(0, half)[None, :], mask=mask_n[:, None], other=0)
            bi = b.to(tl.int32)
            lo = (bi & 0xF).to(tl.int8) - 8
            hi = ((bi >> 4) & 0xF).to(tl.int8) - 8
            w_i8 = tl.interleave(lo, hi)
        else:
            w_i8 = tl.load(WQ + offs_n[:, None] * K + k0 + offs_k[None, :],
                           mask=mask_n[:, None], other=0).to(tl.int8)
        dots = tl.dot(aq, tl.trans(w_i8), out_dtype=tl.int32).to(tl.float32)
        sw = tl.load(SC + offs_n * (K // GROUP) + (k0 // GROUP), mask=mask_n, other=0.0)
        sa = tl.load(AS + offs_m * (K // XGROUP) + (k0 // XGROUP), mask=mask_m, other=0.0)
        acc += dots * sa[:, None] * sw[None, :]
    if HAS_BIAS:
        acc += tl.load(BIAS + offs_n, mask=mask_n, other=0.0)[None, :]
    if HAS_RES:
        # residual add fused into the epilogue: saves one launch and one fp16 round trip
        # of the whole hidden chunk per prefill layer (two per layer)
        res_ptr = offs_m[:, None] * N + offs_n[None, :]
        res_mask = mask_m[:, None] & mask_n[None, :]
        acc += tl.load(Y + res_ptr, mask=res_mask, other=0.0).to(tl.float32)
    tl.store(Y + offs_m[:, None] * N + offs_n[None, :], acc.to(tl.float16),
             mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def gemm_swiglu_w8a8_kernel(AQ, AS, WQ, SC, Y, YQ, YS, M, NH, K,
                            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                            BLOCK_K: tl.constexpr, GROUP: tl.constexpr,
                            XGROUP: tl.constexpr, IS_INT4: tl.constexpr = True,
                            HAS_OUTQ: tl.constexpr = False):
    """Fused gate_up + silu on integer tensor cores (prefill, int4 or int8 weights).

    With HAS_OUTQ the kernel also writes the int8 copy of its output plus one scale per
    row per BLOCK_N lanes, so the following down_proj GEMM can skip quantize_act.  That
    only works when BLOCK_N equals the down_proj's XGROUP (128), which is the tile used.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_n = offs_n < NH
    acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    half: tl.constexpr = BLOCK_K // 2
    for k0 in range(0, K, BLOCK_K):
        aq = tl.load(AQ + offs_m[:, None] * K + k0 + offs_k[None, :],
                     mask=mask_m[:, None], other=0).to(tl.int8)
        sa = tl.load(AS + offs_m * (K // XGROUP) + (k0 // XGROUP), mask=mask_m, other=0.0)
        if IS_INT4:
            bg = tl.load(WQ + offs_n[:, None] * (K // 2) + (k0 // 2)
                         + tl.arange(0, half)[None, :], mask=mask_n[:, None], other=0)
            bu = tl.load(WQ + (offs_n + NH)[:, None] * (K // 2) + (k0 // 2)
                         + tl.arange(0, half)[None, :], mask=mask_n[:, None], other=0)
            gi = bg.to(tl.int32)
            ui = bu.to(tl.int32)
            wg = tl.interleave((gi & 0xF).to(tl.int8) - 8, ((gi >> 4) & 0xF).to(tl.int8) - 8)
            wu = tl.interleave((ui & 0xF).to(tl.int8) - 8, ((ui >> 4) & 0xF).to(tl.int8) - 8)
        else:
            wg = tl.load(WQ + offs_n[:, None] * K + k0 + offs_k[None, :],
                         mask=mask_n[:, None], other=0).to(tl.int8)
            wu = tl.load(WQ + (offs_n + NH)[:, None] * K + k0 + offs_k[None, :],
                         mask=mask_n[:, None], other=0).to(tl.int8)
        sg = tl.load(SC + offs_n * (K // GROUP) + (k0 // GROUP), mask=mask_n, other=0.0)
        su = tl.load(SC + (offs_n + NH) * (K // GROUP) + (k0 // GROUP), mask=mask_n, other=0.0)
        acc_g += tl.dot(aq, tl.trans(wg), out_dtype=tl.int32).to(tl.float32) * sa[:, None] * sg[None, :]
        acc_u += tl.dot(aq, tl.trans(wu), out_dtype=tl.int32).to(tl.float32) * sa[:, None] * su[None, :]
    act = acc_g * tl.sigmoid(acc_g) * acc_u
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(Y + offs_m[:, None] * NH + offs_n[None, :], act.to(tl.float16), mask=out_mask)
    if HAS_OUTQ:
        s = tl.maximum(tl.max(tl.where(out_mask, tl.abs(act), 0.0), axis=1), 1e-8) / 127.0
        v = act / s[:, None]
        q = v + tl.where(v >= 0, 0.5, -0.5)
        tl.store(YQ + offs_m[:, None] * NH + offs_n[None, :], q.to(tl.int8), mask=out_mask)
        tl.store(YS + offs_m * (NH // BLOCK_N) + pid_n, s, mask=mask_m)


@triton.jit
def gemv_int4_swiglu_w8a8_kernel(XQ, XS, WQ, SC, Y, NH, K,
                                 BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                                 GROUP: tl.constexpr, XGROUP: tl.constexpr,
                                 TDIM: tl.constexpr):
    """Decode gate_up on integer tensor cores.

    The fp16 variant spends its time converting two nibble tiles per iteration and
    keeping them in registers; int8 tiles are half the size and feed tl.dot directly.
    """
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < NH
    offs_m = tl.arange(0, TDIM)
    acc_g = tl.zeros((BLOCK_N, TDIM), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_N, TDIM), dtype=tl.float32)
    half: tl.constexpr = BLOCK_K // 2
    for k0 in range(0, K, BLOCK_K):
        gk = k0 // GROUP
        kb0 = k0 // 2
        bg = tl.load(WQ + offs_n[:, None] * (K // 2) + kb0 + tl.arange(0, half)[None, :],
                     mask=mask_n[:, None], other=0)
        bu = tl.load(WQ + (offs_n + NH)[:, None] * (K // 2) + kb0
                     + tl.arange(0, half)[None, :], mask=mask_n[:, None], other=0)
        gi = bg.to(tl.int32)
        ui = bu.to(tl.int32)
        wg = tl.interleave((gi & 0xF).to(tl.int8) - 8, ((gi >> 4) & 0xF).to(tl.int8) - 8)
        wu = tl.interleave((ui & 0xF).to(tl.int8) - 8, ((ui >> 4) & 0xF).to(tl.int8) - 8)
        xq = tl.load(XQ + offs_m[:, None] * K + k0 + tl.arange(0, BLOCK_K)[None, :])
        sg = tl.load(SC + offs_n * (K // GROUP) + gk, mask=mask_n, other=0.0)
        su = tl.load(SC + (offs_n + NH) * (K // GROUP) + gk, mask=mask_n, other=0.0)
        xs = tl.load(XS + offs_m * (K // XGROUP) + (k0 // XGROUP))
        dg = tl.dot(wg, tl.trans(xq), out_dtype=tl.int32).to(tl.float32)
        du = tl.dot(wu, tl.trans(xq), out_dtype=tl.int32).to(tl.float32)
        acc_g += dg * sg[:, None] * xs[None, :]
        acc_u += du * su[:, None] * xs[None, :]
    act = acc_g * tl.sigmoid(acc_g) * acc_u
    tl.store(Y + offs_m[None, :] * NH + offs_n[:, None], act.to(tl.float16),
             mask=mask_n[:, None])
