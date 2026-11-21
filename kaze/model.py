"""Kaze transformer: Llama / Qwen2 style dense decoder.

Two forward paths share the same weights and kernels:

* prefill - a chunk of a prompt.  Uses the tiled dequant GEMM, writes K/V into
  the paged cache, and runs causal attention over the gathered prefix.
* decode  - one token for each of B sequences.  Uses the GEMV kernel, the paged
  decode attention kernel, and static buffers so the whole step can be captured
  in a CUDA graph.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import torch

from kaze.config import ModelConfig
from kaze.kernels import attn as K_attn
from kaze.kernels import norm as K_norm
from kaze.kernels import quant_linear as K_lin
from kaze.kernels.act import silu_mul
from kaze.kernels.fused import rope_scatter
from kaze.quant import QuantLinear


# Fold the split-K epilogue inside the consumer kernels (rmsnorm / rope_scatter) instead of
# launching gemv_epilogue_kernel once per layer.  Only for the batch-1 bucket: at tdim>=16
# the partials are 8x wider and the fuse was measured to be a wash.  KAZE_FUSE_EPILOGUE=0
# or set_fuse_epilogue(False) restores the two-kernel path for A/B.
FUSE_EPILOGUE = os.environ.get("KAZE_FUSE_EPILOGUE", "1") != "0"
# Quantise prefill activations inside the rmsnorm that produces them, instead of a
# separate quant_act launch per GEMM (96 launches per 512-token prefill, 7% of the step).
FUSE_PREFILL_ACTQUANT = os.environ.get("KAZE_FUSE_PREFILL_ACTQUANT", "1") != "0"
# Add the residual inside the int4 prefill GEMM epilogue instead of a separate torch add
# (2 launches and 2 fp16 round trips of the hidden chunk per layer).
FUSE_PREFILL_RESIDUAL = os.environ.get("KAZE_FUSE_PREFILL_RESIDUAL", "1") != "0"
# Fold rope_scatter into the decode attention kernel for the single-pass (short) path:
# one launch fewer per layer, and at batch >= 16 the step is latency bound so the launch
# count matters more than the bandwidth the fused kernel re-reads.
FUSE_ROPE_ATTN = os.environ.get("KAZE_FUSE_ROPE_ATTN", "1") != "0"
# The gate_up kernel can write the down_proj's int8 activations itself (it already has
# them in registers) when its column tile equals one quantisation group.
_SWIGLU_BN = K_lin._SWIGLU_TILE[1]


def set_fuse_epilogue(flag: bool) -> None:
    """Switch the epilogue fold at runtime (benchmarks only; recapture graphs after)."""
    global FUSE_EPILOGUE
    FUSE_EPILOGUE = bool(flag)


def set_fuse_prefill_actquant(flag: bool) -> None:
    global FUSE_PREFILL_ACTQUANT
    FUSE_PREFILL_ACTQUANT = bool(flag)


def _splitk_of(lin, tdim: int, buf, fused: bool):
    """(split_k, partial buffer) if this GEMV can hand its epilogue to the consumer."""
    if not fused or lin is None:
        return 1, None
    # gemv() refuses split-K for fp16 weights (cfg_sk forced to 1) but the config table
    # still reports depth 8 for them, so trusting the table here handed the consumer a
    # partials buffer that gemv never writes: fp16 decode read zeros and diverged from
    # prefill at cos 0.09.  Mirror gemv's guard exactly.
    if lin.kind == "fp16":
        return 1, None
    depth = K_lin._cfg(lin.n, lin.k, lin.kind, rows=tdim)[1]
    if depth <= 1 or buf.shape[0] < depth or buf.shape[2] != lin.n or buf.shape[1] != tdim:
        return 1, None
    return depth, buf[:depth]


@dataclass
class LayerWeights:
    attn_norm: torch.Tensor
    wqkv: QuantLinear
    wo: QuantLinear
    ffn_norm: torch.Tensor
    wgate_up: QuantLinear
    wdown: QuantLinear
    q_norm: torch.Tensor | None = None     # Qwen3-style per-head RMSNorm on q
    k_norm: torch.Tensor | None = None
    # mixture of experts (Qwen2-MoE / Qwen3-MoE / Mixtral style)
    router: torch.Tensor | None = None     # [E, H] fp16
    w_gu_e: QuantLinear | None = None      # stacked [E*2I, H]
    w_down_e: QuantLinear | None = None    # stacked [E*H, I]
    moe_i: int = 0


class KazeModel:
    def __init__(self, cfg: ModelConfig, embed: torch.Tensor, layers: list[LayerWeights],
                 final_norm: torch.Tensor, lm_head: QuantLinear, device: torch.device,
                 dtype: torch.dtype = torch.float16):
        self.cfg = cfg
        self.device = device
        self.dtype = dtype
        self.embed = embed                       # [V, H] fp16
        self.layers = layers
        self.final_norm = final_norm
        self.lm_head = lm_head
        self.cos, self.sin = make_rope_tables(cfg, device)
        self.buffers: dict[int, "DecodeBuffers"] = {}
        # block-sparse decode attention (0.0 = dense, the default)
        # Split-K factor for long-context decode.  The grid is (batch, KV heads, splits),
        # so 8 leaves a 2-KV-head model at 16 CTAs on a 46-SM GPU (33% of the device) and
        # long-context decode is bandwidth bound on exactly those CTAs: 48/kvh splits fill
        # it for batch 1, and larger batches are already wide enough.  KAZE_ATTN_SPLITS
        # overrides it for A/B measurements (same value must reach DecodeBuffers).
        self.num_splits = int(os.environ.get(
            "KAZE_ATTN_SPLITS", max(8, -(-48 // max(1, cfg.num_key_value_heads)))))
        self.batched_gemm = True          # use tensor-core GEMM at decode batch >= 16
        self.sparse_reuse = 8          # recompute the selection every N decode steps
        self._sel_cache: dict[int, tuple] = {}
        self.sparse_frac = 0.0
        self.sparse_min_len = 1024
        self.sparse_recent = 128
        # deep layers spread their attention wider, so their budget is larger
        self.sparse_frac_deep = 0.0

    # ---------------------------------------------------------------- helpers
    @property
    def head_dim(self) -> int:
        return self.cfg.head_dim

    @property
    def scale(self) -> float:
        return self.head_dim ** -0.5

    def rope(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """x [M, H, D]; positions [M] -> rotated copy."""
        cos = self.cos.index_select(0, positions).unsqueeze(1)   # [M, 1, D/2]
        sin = self.sin.index_select(0, positions).unsqueeze(1)
        d2 = self.head_dim // 2
        xf = x.float()
        x1 = xf[..., :d2]
        x2 = xf[..., d2:]
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        return torch.cat((o1, o2), dim=-1).to(x.dtype)

    def _mlp(self, hidden: torch.Tensor, layer: LayerWeights, xn: torch.Tensor,
             aq: torch.Tensor | None = None, as_: torch.Tensor | None = None,
             out_q: torch.Tensor | None = None,
             out_s: torch.Tensor | None = None) -> torch.Tensor:
        cfg = self.cfg
        if layer.router is not None:
            out = hidden.clone()
            self._moe_prefill(layer, xn, out)
            return out
        inter = cfg.intermediate_size
        if layer.wgate_up.kind in ("int8", "int4"):
            # the kernel only writes the int8 copy when its column tile equals one
            # quantisation group; keep the flag in sync or the down GEMM reads garbage
            # the fused int8 copy needs the w8a8 gate_up path (m >= 64) and a column tile
            # that equals one quantisation group
            fused_outq = (xn.shape[0] >= 64 and out_q is not None and out_s is not None
                          and _SWIGLU_BN == (layer.wdown.group or 0))
            if aq is not None and (layer.wgate_up.kind == "int4"
                                   or (layer.wgate_up.kind == "int8" and K_lin.W8A8_PREFILL)):
                act = K_lin.gemm_swiglu(xn, layer.wgate_up, inter, aq=aq, as_=as_,
                                        out_q=out_q, out_s=out_s)
            else:
                act = K_lin.gemm_swiglu(xn, layer.wgate_up, inter)
                fused_outq = False
            # int8 activations on the silu product lose far too much accuracy (see
            # W8A8_DOWN_PREFILL), so the down projection keeps fp16 activations unless
            # the model is int4, where the weight error already dominates
            fp16_acts = (layer.wdown.kind == "int8" and not K_lin.W8A8_DOWN_PREFILL)
            if fused_outq:
                # the swiglu kernel already produced the int8 activations for this GEMM
                if FUSE_PREFILL_RESIDUAL and layer.wdown.kind == "int4":
                    return K_lin.gemm(act, layer.wdown, add_to=hidden,
                                      aq=out_q, as_=out_s)
                return hidden + K_lin.gemm(act, layer.wdown, aq=out_q, as_=out_s)
            if FUSE_PREFILL_RESIDUAL and layer.wdown.kind == "int4" and act.shape[0] >= 64:
                return K_lin.gemm(act, layer.wdown, add_to=hidden, fp16_acts=fp16_acts)
            down = K_lin.gemm(act, layer.wdown, fp16_acts=fp16_acts)
        else:
            gu = K_lin.gemm(xn, layer.wgate_up)
            act = silu_mul(gu[:, :inter], gu[:, inter:], stride_g=2 * inter)
            down = K_lin.gemm(act, layer.wdown)
        return hidden + down

    # --------------------------------------------------------------- prefill
    def _moe_decode(self, layer, xn2, hidden, cfg, bufs=None) -> None:
        """Router -> top-k -> only the selected experts are streamed.

        The router and the top-k run in one Triton kernel writing into
        preallocated buffers, so a MoE decode step can be captured in a CUDA
        graph (torch.topk would allocate inside the capture).
        """
        k = cfg.num_experts_per_tok
        tdim = xn2.shape[0]
        if bufs is not None:
            idx_flat = bufs.moe_idx[: tdim * k]
            w_flat = bufs.moe_w[: tdim * k]
            K_lin.moe_router(xn2, layer.router, k, cfg.num_experts, cfg.norm_topk_prob,
                             idx_flat, w_flat)
            w = w_flat.view(tdim, k)
            flat = idx_flat
        else:
            logits = torch.nn.functional.linear(xn2.float(), layer.router.float())
            probs = torch.softmax(logits, dim=-1)
            w, idx = torch.topk(probs, k, dim=-1)
            if cfg.norm_topk_prob:
                w = w / w.sum(-1, keepdim=True)
            flat = idx.reshape(-1).to(torch.int32)
        i = layer.moe_i
        gu = K_lin.moe_gemv(xn2, layer.w_gu_e, flat, cfg.num_experts, 2 * i, per_row=k)
        g, u = gu[:, :i], gu[:, i:]
        act = g * torch.sigmoid(g) * u
        dn = K_lin.moe_gemv(act, layer.w_down_e, flat, cfg.num_experts, cfg.hidden_size,
                            same_rows=True)
        y = (dn.float().view(tdim, k, cfg.hidden_size) * w.float()[:, :, None]).sum(1)
        hidden.add_(y.to(hidden.dtype))

    def _moe_prefill(self, layer, xn2, hidden) -> None:
        """One GEMM per expert that any token picked, so each expert is read once."""
        cfg = self.cfg
        k = cfg.num_experts_per_tok
        p = xn2.shape[0]
        logits = torch.nn.functional.linear(xn2.float(), layer.router.float())
        probs = torch.softmax(logits, dim=-1)
        w, idx = torch.topk(probs, k, dim=-1)
        if cfg.norm_topk_prob:
            w = w / w.sum(-1, keepdim=True)
        i, h = layer.moe_i, cfg.hidden_size
        out = torch.zeros((p, h), dtype=torch.float16, device=xn2.device)
        for e in range(cfg.num_experts):
            sel = idx == e
            rows = sel.any(-1).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            xe = xn2.index_select(0, rows)
            gu = K_lin.gemm(xe, layer.w_gu_e, row_off=e * 2 * i, n_override=2 * i)
            g, u = gu[:, :i], gu[:, i:]
            act = g * torch.sigmoid(g) * u
            dn = K_lin.gemm(act, layer.w_down_e, row_off=e * h, n_override=h)
            wsel = (sel[rows].to(torch.float32) * w[rows]).sum(-1, keepdim=True)
            out.index_add_(0, rows, dn * wsel.to(dn.dtype))
        hidden.add_(out)

    @torch.no_grad()
    def prefill_hidden(self, ids, positions, cache, block_table, q_start, upto):
        """Hidden state after layer 'upto' (debugging aid: compare against HF)."""
        k_flat_all, v_flat_all = cache.k_flat, cache.v_flat
        seq_len = q_start + ids.shape[0]
        slots_new = cache.slots_of(block_table, q_start, seq_len)
        cfg = self.cfg
        p = ids.shape[0]
        hq, hkv, d = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        hidden = self.embed[ids]
        q_buf = torch.zeros((p, hq, d), dtype=torch.float16, device=self.device)
        for li in range(upto + 1):
            layer = self.layers[li]
            xn = K_norm.rmsnorm(hidden, layer.attn_norm, cfg.rms_norm_eps)
            qkv = K_lin.gemm(xn, layer.wqkv)
            ks_l, vs_l = (cache.ks[li], cache.vs[li]) if getattr(cache, "q8", False) else (None, None)
            self._rope(cache, li, qkv, q_buf, k_flat_all[li], v_flat_all[li],
                         slots_new, positions, hq, hkv, d, p,
                         q_norm=layer.q_norm, k_norm=layer.k_norm, eps=cfg.rms_norm_eps,
                         k_scale=ks_l, v_scale=vs_l)
            ctx = self._prefill_attn(cache, li, q_buf, cache.k[li], cache.v[li], block_table,
                                            q_start, seq_len, cache.block_size, hq // hkv,
                                            d, self.scale)
            hidden = hidden + K_lin.gemm(ctx.reshape(p, hq * d), layer.wo)
            xn2 = K_norm.rmsnorm(hidden, layer.ffn_norm, cfg.rms_norm_eps)
            if layer.router is not None:
                out = hidden.clone()
                self._moe_prefill(layer, xn2, out)
                hidden = out
            else:
                if layer.wgate_up.kind in ("int8", "int4"):
                    act = K_lin.gemm_swiglu(xn2, layer.wgate_up, cfg.intermediate_size)
                else:
                    gu = K_lin.gemm(xn2, layer.wgate_up)
                    g, u = gu[:, :cfg.intermediate_size], gu[:, cfg.intermediate_size:]
                    act = g * torch.sigmoid(g) * u
                hidden = hidden + K_lin.gemm(act, layer.wdown)
        return hidden

    def prefill_logits(self, ids: torch.Tensor, positions: torch.Tensor,
                       cache, block_table: torch.Tensor, q_start: int,
                       last_only: bool = True) -> torch.Tensor:
        """One prefill chunk: ids [P] are tokens q_start .. q_start+P-1 of a
        sequence whose KV lives in cache at block_table."""
        k_flat_all = cache.k_flat
        v_flat_all = cache.v_flat
        seq_len = q_start + ids.shape[0]
        slots_new = cache.slots_of(block_table, q_start, seq_len)
        cfg = self.cfg
        device = self.device
        hidden = self.embed.index_select(0, ids)                  # [P, H]
        hq, hkv, d = cfg.num_attention_heads, cfg.num_key_value_heads, self.head_dim
        qs = hq * d
        kvs = hkv * d
        p = ids.shape[0]
        new_slots = slots_new

        q_buf = torch.empty((p, hq, d), dtype=torch.float16, device=hidden.device)
        qs_t = torch.tensor([q_start], dtype=torch.int32, device=hidden.device)
        sl_t = torch.tensor([seq_len], dtype=torch.int32, device=hidden.device)
        # The int4 prefill GEMMs quantise their activations first; doing that inside the
        # rmsnorm that produces them removes one launch and one full read of xn per GEMM
        # (96 launches in a 512-token prefill, measured at 8% of the step).
        act_q = None
        act_q2 = None
        # only allocate the int8 activation copy when a GEMM will actually consume it
        # (int4 always; int8 only when the W8A8 prefill path is enabled, which is off by
        # default because of the accuracy loss measured in kernels/quant_linear.py)
        uses_aq = any(l.wqkv.kind == "int4"
                      or (l.wqkv.kind == "int8" and K_lin.W8A8_PREFILL)
                      for l in self.layers)
        if FUSE_PREFILL_ACTQUANT and uses_aq:
            act_q = torch.empty((p, cfg.hidden_size), dtype=torch.int8, device=hidden.device)
            act_s = torch.empty((p, (cfg.hidden_size + 127) // 128), dtype=torch.float32,
                                device=hidden.device)
            inter = cfg.intermediate_size
            if inter % 128 == 0:
                act_q2 = torch.empty((p, inter), dtype=torch.int8, device=hidden.device)
                act_s2 = torch.empty((p, inter // 128), dtype=torch.float32, device=hidden.device)
        for li, layer in enumerate(self.layers):
            k_flat = k_flat_all[li]
            v_flat = v_flat_all[li]
            if act_q is not None and layer.wqkv.group == 128 and (layer.wqkv.kind == "int4"
                                      or (layer.wqkv.kind == "int8" and K_lin.W8A8_PREFILL)):
                xn = K_norm.rmsnorm(hidden, layer.attn_norm, cfg.rms_norm_eps,
                                    out_q=act_q, out_s=act_s)
                qkv = K_lin.gemm(xn, layer.wqkv, aq=act_q, as_=act_s)   # [P, qs+2kvs]
            else:
                xn = K_norm.rmsnorm(hidden, layer.attn_norm, cfg.rms_norm_eps)
                qkv = K_lin.gemm(xn, layer.wqkv)
            self._rope(cache, li, qkv, q_buf, k_flat, v_flat,
                         new_slots, positions, hq, hkv, d, p,
                         q_norm=layer.q_norm, k_norm=layer.k_norm, eps=cfg.rms_norm_eps)
            if self.sparse_frac > 0:
                cache.update_repr(li, new_slots)
            attn = self._prefill_attn(cache, li, q_buf, cache.k[li], cache.v[li], block_table,
                                             qs_t, sl_t, cache.block_size,
                                             hq // hkv, d, self.scale)
            if (FUSE_PREFILL_RESIDUAL and layer.wo.kind == "int4" and p >= 64):
                K_lin.gemm(attn.view(p, qs), layer.wo, add_to=hidden)
            else:
                hidden = hidden + K_lin.gemm(attn.view(p, qs), layer.wo)
            if (act_q is not None and layer.wgate_up is not None
                    and layer.wgate_up.group == 128
                    and layer.wgate_up.kind in ("int4", "int8")):
                xn2 = K_norm.rmsnorm(hidden, layer.ffn_norm, cfg.rms_norm_eps,
                                     out_q=act_q, out_s=act_s)
                hidden = self._mlp(hidden, layer, xn2, aq=act_q, as_=act_s,
                                   out_q=act_q2, out_s=act_s2 if act_q2 is not None else None)
            else:
                xn2 = K_norm.rmsnorm(hidden, layer.ffn_norm, cfg.rms_norm_eps)
                hidden = self._mlp(hidden, layer, xn2)

        xn = K_norm.rmsnorm(hidden, self.final_norm, cfg.rms_norm_eps)
        if last_only:
            pad = torch.zeros((16, xn.shape[1]), dtype=xn.dtype, device=xn.device)
            pad[0].copy_(xn[-1])
            return K_lin.gemv(pad, self.lm_head, 16)[:1]
        return K_lin.gemm(xn, self.lm_head)

    def prefill_batched(self, ids: torch.Tensor, positions: torch.Tensor,
                        slots: torch.Tensor, cache, block_tables: torch.Tensor,
                        lens: torch.Tensor, seq_of_group: torch.Tensor,
                        last_rows: torch.Tensor) -> torch.Tensor:
        """One forward for prefill chunks of several sequences.

        ids / positions / slots are [M] with the chunks concatenated along the row axis;
        row i belongs to sequence seq_of_group[i // 16] (groups never straddle a
        sequence) and lens[g] is that sequence total key count.  Returns [B, vocab] with
        the logits of last_rows[b] (the last row of sequence b).
        """
        cfg = self.cfg
        dev = self.device
        hidden = self.embed.index_select(0, ids)
        hq, hkv, d = cfg.num_attention_heads, cfg.num_key_value_heads, self.head_dim
        qs = hq * d
        p = ids.shape[0]
        q_buf = torch.empty((p, hq, d), dtype=torch.float16, device=dev)
        attn_buf = torch.empty((p, hq, d), dtype=torch.float16, device=dev)
        uses_aq = any(l.wqkv.kind == "int4" for l in self.layers)
        act_q = act_s = act_q2 = act_s2 = None
        if FUSE_PREFILL_ACTQUANT and uses_aq:
            act_q = torch.empty((p, cfg.hidden_size), dtype=torch.int8, device=dev)
            act_s = torch.empty((p, (cfg.hidden_size + 127) // 128),
                                dtype=torch.float32, device=dev)
            inter = cfg.intermediate_size
            if inter % 128 == 0:
                act_q2 = torch.empty((p, inter), dtype=torch.int8, device=dev)
                act_s2 = torch.empty((p, inter // 128), dtype=torch.float32, device=dev)
        for li, layer in enumerate(self.layers):
            if act_q is not None and layer.wqkv.kind == "int4":
                xn = K_norm.rmsnorm(hidden, layer.attn_norm, cfg.rms_norm_eps,
                                    out_q=act_q, out_s=act_s)
                qkv = K_lin.gemm(xn, layer.wqkv, aq=act_q, as_=act_s)
            else:
                xn = K_norm.rmsnorm(hidden, layer.attn_norm, cfg.rms_norm_eps)
                qkv = K_lin.gemm(xn, layer.wqkv)
            self._rope(cache, li, qkv, q_buf, cache.k_flat[li], cache.v_flat[li],
                       slots, positions, hq, hkv, d, p,
                       q_norm=layer.q_norm, k_norm=layer.k_norm, eps=cfg.rms_norm_eps)
            K_attn.attn_prefill_batch(q_buf, cache.k[li], cache.v[li], block_tables,
                                      positions, lens, seq_of_group, p,
                                      cache.block_size, hq // hkv, d, self.scale,
                                      out=attn_buf)
            if FUSE_PREFILL_RESIDUAL and layer.wo.kind == "int4" and p >= 64:
                K_lin.gemm(attn_buf.view(p, qs), layer.wo, add_to=hidden)
            else:
                hidden = hidden + K_lin.gemm(attn_buf.view(p, qs), layer.wo)
            if act_q is not None and layer.wgate_up is not None:
                xn2 = K_norm.rmsnorm(hidden, layer.ffn_norm, cfg.rms_norm_eps,
                                     out_q=act_q, out_s=act_s)
                hidden = self._mlp(hidden, layer, xn2, aq=act_q, as_=act_s,
                                   out_q=act_q2, out_s=act_s2)
            else:
                xn2 = K_norm.rmsnorm(hidden, layer.ffn_norm, cfg.rms_norm_eps)
                hidden = self._mlp(hidden, layer, xn2)
        xn = K_norm.rmsnorm(hidden, self.final_norm, cfg.rms_norm_eps)
        return K_lin.gemm(xn.index_select(0, last_rows), self.lm_head)

    # ---------------------------------------------------------------- decode
    def decode_logits(self, ids: torch.Tensor, positions: torch.Tensor,
                      slots: torch.Tensor, block_tables: torch.Tensor,
                      seq_lens: torch.Tensor, cache,
                      tdim: int | None = None, bufs: "DecodeBuffers | None" = None,
                      attn_mode: str = "long") -> torch.Tensor:
        cfg = self.cfg
        b = ids.shape[0]
        # allocate the buffers first: reading bufs.tdim before the None check raised for
        # any caller that did not pass bufs (the engine always does, so it went unnoticed)
        if bufs is None:
            bufs = self.get_buffers(tdim or 16)
        tdim = tdim or bufs.tdim
        hq, hkv, d = cfg.num_attention_heads, cfg.num_key_value_heads, self.head_dim
        qs, kvs = hq * d, hkv * d
        inter = cfg.intermediate_size
        k_flat_all = cache.k_flat
        v_flat_all = cache.v_flat

        if positions.numel() < tdim:
            bufs.pos[:b].copy_(positions)
            positions = bufs.pos
        hidden = bufs.hidden
        bufs.active[:b].copy_(self.embed.index_select(0, ids))
        hidden[:b].copy_(bufs.active[:b])

        # Fold the split-K epilogue in the consumer (one launch per layer less).  pending
        # carries the down_proj partials into the next layer's attn_norm.
        # Fold at every tdim: at batch >= 16 the step is *latency* bound (same graph:
        # 3.7 ms with a sync per step vs 1.9 ms pipelined), so removing the 48 epilogue
        # launches is worth more than the wider partial reads it costs.  The cos-0.18
        # regression that once pushed this back to tdim == 1 was not the fold at all: it
        # was the fp16 partials mismatch fixed in _splitk_of (round 128).
        fused = FUSE_EPILOGUE
        pending = None
        for li, layer in enumerate(self.layers):
            k_flat = k_flat_all[li]
            v_flat = v_flat_all[li]
            q8 = K_lin.W4A8_DECODE and layer.wqkv.kind == "int4"
            part, depth, pb = pending if pending is not None else (None, 1, None)
            xn = K_norm.rmsnorm(hidden, layer.attn_norm, cfg.rms_norm_eps, out=bufs.xn,
                                out_q=bufs.xn_q if q8 else None,
                                out_s=bufs.xn_s if q8 else None,
                                partials=part, bias=pb, res=hidden if part is not None else None)
            pending = None
            sk, qkv_part = _splitk_of(layer.wqkv, tdim, bufs.sk_qkv, fused)
            K_lin.gemv(xn, layer.wqkv, tdim, out=bufs.qkv,
                       xq=bufs.xn_q if q8 else None, xs=bufs.xn_s if q8 else None,
                       acc=qkv_part, defer_epilogue=qkv_part is not None)
            # bufs.qkv stays the shape carrier for rope_scatter; when the epilogue is
            # deferred the projection is read from the partials instead of this buffer.
            # tdim == 1 only, measured (bench/rope_fusion_ab32.py): the fused kernel wins
            # at batch 1 (+2.5%, one launch less per layer) but loses ~2.7% at batch 32 --
            # the fold + attention loop in one kernel costs registers, and 128 fat CTAs
            # hide latency worse than 32 light + 128 light ones.
            fuse_rope = (FUSE_ROPE_ATTN and qkv_part is not None
                         and attn_mode == "short" and not getattr(cache, "q8", False)
                         and layer.q_norm is None and layer.k_norm is None
                         and self.sparse_frac <= 0 and self.sparse_frac_deep <= 0)
            if not fuse_rope:
                self._rope(cache, li, bufs.qkv, bufs.q, k_flat, v_flat, slots, positions,
                           hq, hkv, d, b,
                           q_norm=layer.q_norm, k_norm=layer.k_norm, eps=cfg.rms_norm_eps,
                           partials=qkv_part, bias=layer.wqkv.bias)
            n_layer = len(self.layers)
            frac = self.sparse_frac + (self.sparse_frac_deep - self.sparse_frac) * (li / max(1, n_layer - 1))
            sparse = frac > 0 and int(seq_lens[:b].max()) > self.sparse_min_len
            if sparse:
                cache.update_repr(li, slots[:b])
                n_valid = (int(seq_lens[:b].max()) + cache.mb - 1) // cache.mb
                seq_now = int(seq_lens[:b].max())
                cached = self._sel_cache.get(li)
                if cached is not None and seq_now - cached[1] < self.sparse_reuse and cached[2] == frac:
                    sel = cached[0]
                else:
                    sel = K_attn.select_micro_blocks(bufs.q[:b], cache.ensure_repr()[li],
                                                     n_valid,
                                                     frac, self.sparse_recent)
                    self._sel_cache[li] = (sel, seq_now, frac)
                K_attn.attn_decode_sparse_split(
                    bufs.q[:b], cache.k[li], cache.v[li], block_tables, seq_lens, sel,
                    bufs.block_size, hq // hkv, d, self.scale,
                    acc_buf=bufs.acc[:b], ml_buf=bufs.ml[:b],
                    out=bufs.attn[:b].view(b, hq, d), num_splits=bufs.acc.shape[2])
            elif fuse_rope:
                K_attn.rope_attn_decode(qkv_part, layer.wqkv.bias, self.cos, self.sin,
                                        cache.k[li], cache.v[li], block_tables,
                                        slots, positions, seq_lens, hq, hkv, d,
                                        self.scale, bufs.attn[:b].view(b, hq, d),
                                        bufs.block_size)
            else:
                ks_l, vs_l = cache.scales(li)
                K_attn.attn_decode(bufs.q[:b], cache.k[li], cache.v[li], block_tables,
                                   seq_lens, bufs.block_size, hq // hkv, d,
                                   out=bufs.attn[:b].view(b, hq, d),
                                   acc_buf=bufs.acc[:b], ml_buf=bufs.ml[:b],
                                   scale=self.scale, mode=attn_mode,
                                   k_scale=ks_l, v_scale=vs_l)
            sko, o_part = _splitk_of(layer.wo, tdim, bufs.sk_h, fused)
            K_lin.gemv(bufs.attn, layer.wo, tdim, out=bufs.o,
                       add_to=None if o_part is not None else hidden,
                       acc=o_part, defer_epilogue=o_part is not None)
            # tdim >= 16 only: the tensor-core path is 252 -> 208 us at TDIM=32 but
            # 13.6 -> 16.9 us at TDIM=1 (bench/swiglu_tdim1_ab.py), so batch-1
            # decode keeps the fp16-activation unpack kernel.
            q8m = (K_lin.W8A8_SWIGLU_DECODE and tdim >= 16
                   and layer.wgate_up is not None
                   and layer.wgate_up.kind == "int4")
            xn2 = K_norm.rmsnorm(hidden, layer.ffn_norm, cfg.rms_norm_eps, out=bufs.xn,
                                 out_q=bufs.xn_q if q8m else None,
                                 out_s=bufs.xn_s if q8m else None,
                                 partials=o_part, bias=layer.wo.bias,
                                 res=hidden if o_part is not None else None)
            if layer.router is not None:
                self._moe_decode(layer, xn2, hidden, cfg, bufs)
            else:
                act = K_lin.gemv_swiglu(xn2, layer.wgate_up, tdim, inter, out=bufs.act,
                                        xq=bufs.xn_q if q8m else None,
                                        xs=bufs.xn_s if q8m else None)
                skd, d_part = _splitk_of(layer.wdown, tdim, bufs.sk_h, fused)
                K_lin.gemv(act, layer.wdown, tdim, out=bufs.o,
                           add_to=None if d_part is not None else hidden,
                           acc=d_part, defer_epilogue=d_part is not None)
                if d_part is not None:
                    pending = (d_part, skd, layer.wdown.bias)

        part, depth, pb = pending if pending is not None else (None, 1, None)
        xn = K_norm.rmsnorm(hidden, self.final_norm, cfg.rms_norm_eps, out=bufs.xn,
                            partials=part, bias=pb, res=hidden if part is not None else None)
        logits = K_lin.gemv(xn, self.lm_head, tdim, out=bufs.logits)
        return logits[:b]


    # --------------------------------------------------------------- verify
    def verify_logits_multi(self, ids, positions, slots, cache, block_tables, groups,
                            tdim: int = 16, n_real: int | None = None):
        """Verify several sequences in one forward.

        Row layout is packed: sequence g owns rows [groups[g].lo, groups[g].hi) and
        attends to its own paged prefix, so the linear layers run once for the
        whole batch and only the (cheap) attention is per sequence.
        """
        cfg = self.cfg
        n = ids.shape[0] if n_real is None else n_real
        hq, hkv, d = cfg.num_attention_heads, cfg.num_key_value_heads, self.head_dim
        qs = hq * d
        inter = cfg.intermediate_size
        bufs = self.get_buffers(tdim)
        q8 = False          # verification always runs the fp16-activation path
        hidden = bufs.hidden
        bufs.active[:n].copy_(self.embed.index_select(0, ids[:n]))
        hidden[:n].copy_(bufs.active[:n])
        if not hasattr(self, "_vgrp"):
            self._vgrp = torch.zeros(64, dtype=torch.int32, device=self.device)
        for li, layer in enumerate(self.layers):
            q8 = K_lin.W4A8_DECODE and layer.wqkv.kind == "int4"
            xn = K_norm.rmsnorm(hidden, layer.attn_norm, cfg.rms_norm_eps, out=bufs.xn,
                                out_q=bufs.xn_q if q8 else None,
                                out_s=bufs.xn_s if q8 else None)
            qkv = K_lin.gemv(xn, layer.wqkv, tdim, out=bufs.qkv,
                             xq=bufs.xn_q if q8 else None, xs=bufs.xn_s if q8 else None)
            self._rope(cache, li, qkv, bufs.q, cache.k_flat[li], cache.v_flat[li],
                         slots, positions, hq, hkv, d, n,
                         q_norm=layer.q_norm, k_norm=layer.k_norm, eps=cfg.rms_norm_eps)
            for (lo, hi, b, q_start, seq_len) in groups:
                if hi <= lo:
                    continue
                attn = self._prefill_attn(cache, li, 
                    bufs.q[lo:hi], cache.k[li], cache.v[li], block_tables[b:b + 1],
                    q_start, seq_len, cache.block_size, hq // hkv, d, self.scale)
                bufs.attn[lo:hi].copy_(attn.view(hi - lo, qs))
            K_lin.gemv(bufs.attn, layer.wo, tdim, out=bufs.o, add_to=hidden)
            # tdim >= 16 only: the tensor-core path is 252 -> 208 us at TDIM=32 but
            # 13.6 -> 16.9 us at TDIM=1 (bench/swiglu_tdim1_ab.py), so batch-1
            # decode keeps the fp16-activation unpack kernel.
            q8m = (K_lin.W8A8_SWIGLU_DECODE and tdim >= 16
                   and layer.wgate_up is not None
                   and layer.wgate_up.kind == "int4")
            xn2 = K_norm.rmsnorm(hidden, layer.ffn_norm, cfg.rms_norm_eps, out=bufs.xn,
                                 out_q=bufs.xn_q if q8m else None,
                                 out_s=bufs.xn_s if q8m else None)
            if layer.router is not None:
                self._moe_decode(layer, xn2, hidden, cfg, bufs)
            else:
                act = K_lin.gemv_swiglu(xn2, layer.wgate_up, tdim, inter, out=bufs.act,
                                        xq=bufs.xn_q if q8m else None,
                                        xs=bufs.xn_s if q8m else None)
                K_lin.gemv(act, layer.wdown, tdim, out=bufs.o, add_to=hidden)
        xn = K_norm.rmsnorm(hidden, self.final_norm, cfg.rms_norm_eps, out=bufs.xn)
        return K_lin.gemv(xn, self.lm_head, tdim, out=bufs.logits)

    def verify_logits(self, ids: torch.Tensor, positions: torch.Tensor, slots: torch.Tensor,
                      cache, block_table: torch.Tensor, qs_t: torch.Tensor, sl_t: torch.Tensor,
                      tdim: int = 16, n_real: int | None = None) -> torch.Tensor:
        """Forward n <= 16 tokens of one sequence (speculative verification).

        Same weights and kernels as decode, but n queries share one pass: query i
        (global position q_start + i) attends causally to the prefix plus the
        tokens before it in this chunk.  Every input is a pre-filled static
        tensor, so the whole call can be captured in a CUDA graph.  Returns
        logits [tdim, vocab] (rows >= n are padding).
        """
        cfg = self.cfg
        n = ids.shape[0] if n_real is None else n_real
        hq, hkv, d = cfg.num_attention_heads, cfg.num_key_value_heads, self.head_dim
        qs, kvs, inter = hq * d, hkv * d, cfg.intermediate_size
        bufs = self.get_buffers(tdim)
        q8 = False          # verification always runs the fp16-activation path
        hidden = bufs.hidden
        bufs.active[:n].copy_(self.embed.index_select(0, ids[:n]))
        hidden[:n].copy_(bufs.active[:n])
        for li, layer in enumerate(self.layers):
            q8 = K_lin.W4A8_DECODE and layer.wqkv.kind == "int4"
            xn = K_norm.rmsnorm(hidden, layer.attn_norm, cfg.rms_norm_eps, out=bufs.xn,
                                out_q=bufs.xn_q if q8 else None,
                                out_s=bufs.xn_s if q8 else None)
            qkv = K_lin.gemv(xn, layer.wqkv, tdim, out=bufs.qkv,
                             xq=bufs.xn_q if q8 else None, xs=bufs.xn_s if q8 else None)
            self._rope(cache, li, qkv, bufs.q, cache.k_flat[li], cache.v_flat[li],
                         slots, positions, hq, hkv, d, n,
                         q_norm=layer.q_norm, k_norm=layer.k_norm, eps=cfg.rms_norm_eps)
            attn = self._prefill_attn(cache, li, bufs.q[:n], cache.k[li], cache.v[li], block_table,
                                             qs_t, sl_t, cache.block_size,
                                             hq // hkv, d, self.scale)
            bufs.attn[:n].copy_(attn.view(n, qs))
            K_lin.gemv(bufs.attn, layer.wo, tdim, out=bufs.o, add_to=hidden)
            # tdim >= 16 only: the tensor-core path is 252 -> 208 us at TDIM=32 but
            # 13.6 -> 16.9 us at TDIM=1 (bench/swiglu_tdim1_ab.py), so batch-1
            # decode keeps the fp16-activation unpack kernel.
            q8m = (K_lin.W8A8_SWIGLU_DECODE and tdim >= 16
                   and layer.wgate_up is not None
                   and layer.wgate_up.kind == "int4")
            xn2 = K_norm.rmsnorm(hidden, layer.ffn_norm, cfg.rms_norm_eps, out=bufs.xn,
                                 out_q=bufs.xn_q if q8m else None,
                                 out_s=bufs.xn_s if q8m else None)
            if layer.router is not None:
                self._moe_decode(layer, xn2, hidden, cfg, bufs)
            else:
                act = K_lin.gemv_swiglu(xn2, layer.wgate_up, tdim, inter, out=bufs.act,
                                        xq=bufs.xn_q if q8m else None,
                                        xs=bufs.xn_s if q8m else None)
                K_lin.gemv(act, layer.wdown, tdim, out=bufs.o, add_to=hidden)
        xn = K_norm.rmsnorm(hidden, self.final_norm, cfg.rms_norm_eps, out=bufs.xn)
        return K_lin.gemv(xn, self.lm_head, tdim, out=bufs.logits)

    def get_buffers(self, tdim: int) -> "DecodeBuffers":
        b = self.buffers.get(tdim)
        if b is None:
            b = DecodeBuffers(self.cfg, self.device, tdim, num_splits=self.num_splits)
            self.buffers[tdim] = b
        return b

    # ------------------------------------------------------------- reference
    def to_dense(self, dtype: torch.dtype = torch.float16) -> "DenseReference":
        return DenseReference(self, dtype)

    def _kv_scales(self, cache, layer: int):
        """(k_scale, v_scale) for this layer, or (None, None) for fp16/fp8 caches."""
        return cache.scales(layer)

    def _prefill_attn(self, cache, layer, *args, **kwargs):
        ks, vs = self._kv_scales(cache, layer)
        return K_attn.attn_prefill_paged(*args, k_scale=ks, v_scale=vs, **kwargs)

    def _rope(self, cache, layer, qkv, out_q, k_flat, v_flat, slots, positions, *args,
              partials=None, bias=None, **kw):
        ks, vs = self._kv_scales(cache, layer)
        # a caller (prefill_hidden) may pass the scales itself; ours wins, so drop theirs
        kw.pop("k_scale", None)
        kw.pop("v_scale", None)
        return rope_scatter(qkv, self.cos, self.sin, out_q, k_flat, v_flat, slots, positions,
                            *args, **kw, k_scale=ks, v_scale=vs,
                            partials=partials, bias=bias)

    def weight_bytes(self) -> int:
        n = self.embed.numel() * self.embed.element_size()
        for l in self.layers:
            n += l.attn_norm.numel() * 2 + l.ffn_norm.numel() * 2
            for q in (l.wqkv, l.wo, l.wgate_up, l.wdown):
                if q is not None:
                    n += q.nbytes
            # MoE layers keep the experts stacked instead of a single MLP
            if getattr(l, "w_gu_e", None) is not None:
                n += l.w_gu_e.nbytes + l.w_down_e.nbytes
        n += self.final_norm.numel() * 2 + self.lm_head.nbytes
        return n


class DecodeBuffers:
    """Static per-batch-size scratch space (sizes fixed -> CUDA graph friendly)."""

    def __init__(self, cfg: ModelConfig, device: torch.device, tdim: int, block_size: int = 16,
                 num_splits: int = 8):
        h = cfg.hidden_size
        hq, hkv, d = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        qs, kvs = hq * d, hkv * d
        inter = cfg.intermediate_size
        self.tdim = tdim
        self.block_size = block_size
        self.active = torch.zeros((tdim, h), dtype=torch.float16, device=device)
        self.pos = torch.zeros((tdim,), dtype=torch.long, device=device)
        self.hidden = torch.zeros((tdim, h), dtype=torch.float16, device=device)
        self.xn = torch.zeros((tdim, h), dtype=torch.float16, device=device)
        self.xn_q = torch.zeros((tdim, h), dtype=torch.int8, device=device)
        self.xn_s = torch.zeros((tdim, h // 128), dtype=torch.float32, device=device)
        self.act_q = torch.zeros((tdim, inter), dtype=torch.int8, device=device)
        self.act_s = torch.zeros((tdim, (inter + 127) // 128), dtype=torch.float32, device=device)
        self.qkv = torch.zeros((tdim, qs + 2 * kvs), dtype=torch.float16, device=device)
        self.attn = torch.zeros((tdim, qs), dtype=torch.float16, device=device)
        self.q = torch.zeros((tdim, hq, d), dtype=torch.float16, device=device)
        self.o = torch.zeros((tdim, h), dtype=torch.float16, device=device)
        self.gu = torch.zeros((tdim, 2 * inter), dtype=torch.float16, device=device)
        self.act = torch.zeros((tdim, inter), dtype=torch.float16, device=device)
        self.logits = torch.zeros((tdim, cfg.vocab_size), dtype=torch.float16, device=device)
        kvh = cfg.num_key_value_heads
        self.moe_idx = torch.zeros((tdim * max(cfg.num_experts_per_tok, 1),), dtype=torch.int32,
                                   device=device)
        self.moe_w = torch.zeros((tdim * max(cfg.num_experts_per_tok, 1),), dtype=torch.float32,
                                 device=device)
        gqa = hq // hkv
        group_heads = max(16, 1 << (gqa - 1).bit_length())
        self.acc = torch.zeros((tdim, kvh, num_splits, group_heads, d), dtype=torch.float32,
                               device=device)
        self.ml = torch.zeros((tdim, kvh, num_splits, 2, group_heads), dtype=torch.float32,
                              device=device)
        # split-K partials that the consumers (rime norm / rope) fold themselves instead of
        # a separate epilogue kernel.  Depth 8 is the largest split the config table asks
        # for; qkv has its own width, o_proj and down_proj share the hidden-width one
        # (down_proj writes it, the next layer's attn_norm reads it, then o_proj overwrites).
        self.sk_depth = 8
        self.sk_qkv = torch.zeros((self.sk_depth, tdim, qs + 2 * kvs), dtype=torch.float32,
                                  device=device)
        self.sk_h = torch.zeros((self.sk_depth, tdim, h), dtype=torch.float32, device=device)


class DenseReference:
    """Float reference used by the test suite (no Triton, no paging)."""

    def __init__(self, model: KazeModel, dtype: torch.dtype = torch.float32):
        self.model = model
        cfg = model.cfg
        self.cfg = cfg
        self.dtype = dtype
        dev = model.device
        self.embed = model.embed.to(dtype)
        self.layers = []
        for l in model.layers:
            self.layers.append(dict(
                attn_norm=l.attn_norm.to(dtype),
                wqkv=l.wqkv.dequantize(dtype),
                bqkv=None if l.wqkv.bias is None else l.wqkv.bias.to(dtype),
                wo=l.wo.dequantize(dtype),
                bo=None if l.wo.bias is None else l.wo.bias.to(dtype),
                ffn_norm=l.ffn_norm.to(dtype),
                wgate_up=l.wgate_up.dequantize(dtype),
                bgu=None if l.wgate_up.bias is None else l.wgate_up.bias.to(dtype),
                wdown=l.wdown.dequantize(dtype),
                bdown=None if l.wdown.bias is None else l.wdown.bias.to(dtype),
            ))
        self.final_norm = model.final_norm.to(dtype)
        self.lm_head = model.lm_head.dequantize(dtype)
        self.lm_bias = None if model.lm_head.bias is None else model.lm_head.bias.to(dtype)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        hq, hkv, d = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        qs, kvs, inter = hq * d, hkv * d, cfg.intermediate_size
        dev = ids.device
        x = self.embed.index_select(0, ids).to(self.dtype)
        p = ids.shape[0]
        pos = torch.arange(p, device=dev)
        cos = self.model.cos.index_select(0, pos).unsqueeze(1).to(self.dtype)
        sin = self.model.sin.index_select(0, pos).unsqueeze(1).to(self.dtype)
        mask = torch.full((p, p), float("-inf"), dtype=self.dtype, device=dev).triu(1)
        for l in self.layers:
            h = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps) * l["attn_norm"]
            qkv = h @ l["wqkv"].T
            if l["bqkv"] is not None:
                qkv = qkv + l["bqkv"]
            q = qkv[:, :qs].view(p, hq, d)
            k = qkv[:, qs:qs + kvs].view(p, hkv, d)
            v = qkv[:, qs + kvs:].view(p, hkv, d)
            q1, q2 = q[..., :d // 2], q[..., d // 2:]
            q = torch.cat((q1 * cos - q2 * sin, q2 * cos + q1 * sin), dim=-1)
            k1, k2 = k[..., :d // 2], k[..., d // 2:]
            k = torch.cat((k1 * cos - k2 * sin, k2 * cos + k1 * sin), dim=-1)
            k = k.repeat_interleave(hq // hkv, dim=1)
            v = v.repeat_interleave(hq // hkv, dim=1)
            qq = q.transpose(0, 1)[None]
            kk = k.transpose(0, 1)[None]
            vv = v.transpose(0, 1)[None]
            w = torch.softmax(qq @ kk.transpose(-1, -2) * (d ** -0.5) + mask, dim=-1)
            a = (w @ vv).squeeze(0).transpose(0, 1).reshape(p, qs)
            x = x + a @ l["wo"].T + (0 if l["bo"] is None else l["bo"])
            h = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps) * l["ffn_norm"]
            gu = h @ l["wgate_up"].T
            if l["bgu"] is not None:
                gu = gu + l["bgu"]
            act = torch.nn.functional.silu(gu[:, :inter]) * gu[:, inter:]
            x = x + act @ l["wdown"].T + (0 if l["bdown"] is None else l["bdown"])
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps) * self.final_norm
        out = x @ self.lm_head.T
        if self.lm_bias is not None:
            out = out + self.lm_bias
        return out


def make_rope_tables(cfg: ModelConfig, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin tables, honouring llama3-style piecewise frequency scaling."""
    d = cfg.head_dim
    inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, d, 2, device=device, dtype=torch.float32) / d))
    scaling = cfg.rope_scaling or {}
    rtype = scaling.get("rope_type") or scaling.get("type")
    # the table has to cover the extended context, not just the native one
    max_ctx = int(cfg.max_position_embeddings)
    if rtype in ("yarn", "linear", "dynamic"):
        max_ctx = max(max_ctx,
                      int(float(scaling.get("original_max_position_embeddings",
                                            cfg.max_position_embeddings))
                          * float(scaling.get("factor", 1.0))))
    if rtype == "llama3":
        factor = float(scaling.get("factor", 8.0))
        low_f = float(scaling.get("low_freq_factor", 1.0))
        high_f = float(scaling.get("high_freq_factor", 4.0))
        old_ctx = float(scaling.get("original_max_position_embeddings", 8192))
        low = old_ctx / low_f
        high = old_ctx / high_f
        wavelen = 2.0 * math.pi / inv
        smooth = torch.clamp((old_ctx / wavelen - low_f) / (high_f - low_f), 0.0, 1.0)
        inv = torch.where(wavelen < high, inv,
                          torch.where(wavelen > low, inv / factor,
                                      (1.0 - smooth) * inv / factor + smooth * inv))
    elif rtype in ("linear", "dynamic"):
        inv = inv / float(scaling.get("factor", 1.0))
    elif rtype == "yarn":
        # YaRN: interpolate the low-frequency dims, keep the high ones, and scale the
        # resulting cos/sin by the attention factor (Peng et al.).  Same formula as
        # transformers' _compute_yarn_parameters.
        factor = float(scaling.get("factor", 4.0))
        theta = float(cfg.rope_theta)
        beta_fast = float(scaling.get("beta_fast", 32.0))
        beta_slow = float(scaling.get("beta_slow", 1.0))
        orig = float(scaling.get("original_max_position_embeddings", cfg.max_position_embeddings))

        def corr_dim(num_rot: float) -> float:
            return d * math.log(orig / (num_rot * 2.0 * math.pi)) / (2.0 * math.log(theta))

        lo = max(0, int(math.floor(corr_dim(beta_fast))))
        hi = min(d // 2 - 1, int(math.ceil(corr_dim(beta_slow))))
        ramp = torch.clamp(
            (torch.arange(d // 2, device=device, dtype=torch.float32) - lo)
            / max(1, hi - lo), 0.0, 1.0)
        inv = inv / factor * ramp + inv * (1.0 - ramp)
        mscale = 0.1 * math.log(factor) + 1.0 if factor > 1 else 1.0
        inv = inv.clone()
        t = torch.arange(max_ctx, device=device, dtype=torch.float32)
        freqs = torch.outer(t, inv)
        return (freqs.cos() * mscale).contiguous(), (freqs.sin() * mscale).contiguous()
    t = torch.arange(max_ctx, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv)
    return freqs.cos().contiguous(), freqs.sin().contiguous()
