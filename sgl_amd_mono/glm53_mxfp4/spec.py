# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3 Quark MXFP4 (amd/GLM-5.3-Quark-MXFP4-AttnFP8) with an FP8 KV cache on
gfx950: the vLLM MonoKernel of vllm-project/vllm#60831 behind SGLang's monokernel
interface (RFC sgl-project/sglang#43949).

Every covered layer is one persistent launch for the whole decoder layer
(``gfx950/kernel.py``): the input norm, q_a and kv_a, the q and kv norms, the FP8
MLA cache write, q_b, the indexer and its exact top-2048 on the layers that run
it, the sparse MLA over the FP8 cache, W_UV and o_proj with the attention TP
all-reduce, the post-attention norm, the router, the shared and routed MXFP4
experts with their inputs quantized to MXFP8, and the MoE TP all-reduce.

The forward is ``DeepseekV2ForCausalLM.forward`` for a routed decode step, as in
``sgl_amd_mono.glm53.spec``: SGLang's embedding, dense layers 0 to 2, final norm
and logits processor run around the launches of the 75 MoE layers. Layer 3 is
the first covered layer. It takes the stream layer 2 wrote and, being a layer
that reuses the previous top-k, layer 2's indices. A pad row (SGLang's padding
slot 0, or a row of the kernel width past the step) writes no cache and attends
no key.

The kernels read the experts and SGLang's caches in place. ``gfx950/weights.py``
lists the copies made at build. The caches need no copy and no change of
addressing: the MLA cache row is 576 FP8 bytes (512 latent, then 64 RoPE) in both
vLLM and SGLang on ROCm, and an index-cache page is 64 x 128 FP8 values in 16 x
16-byte tiles followed by 64 f32 scales in both. The one format difference is
SGLang's Hadamard rotation of the cached index keys, which the kernel's
``index_hadamard`` option applies.
"""

from __future__ import annotations

import importlib.util
import logging

import msgspec
import torch
from sglang.srt.monokernels.spec import (
    DeviceLimits,
    MonoSpec,
    Numerics,
    NumericsKind,
    Requirements,
)

from sgl_amd_mono.glm53 import HF_CONFIG

logger = logging.getLogger(__name__)

# Kernel widths. A step of fewer rows runs at the next width, the other rows
# being pad rows. A one-row launch is not built (vLLM pads it to two rows too).
WIDTHS = (2, 4, 8)
ROWS = tuple(range(1, max(WIDTHS) + 1))
TP_SIZES = (4, 8)
# The most rows a step runs on the kernels at a TP size. At TP4 on MI355X the
# 8-row launch decodes at 12.4 ms a token against 11.6 ms for SGLang's own
# forward (1024 input tokens, 2026-10-21), so wider steps at TP4 run SGLang's
# forward. TP8 is not measured and keeps every width.
TP_MAX_ROWS = {4: 4}
# One workgroup per CU, all resident at once.
BLOCKS = 256
# SGLang's DSA caches on ROCm use 64-token pages.
PAGE_SIZE = 64
TOPK = 2048
INDEX_PAGE_BYTES = PAGE_SIZE * (128 + 4)
KV_ROW = 512 + 64
# vLLM bounds every mailbox wait at 20,000,000 re-polls, about 4 s, so one
# second is about 5,000,000 re-polls.
POLLS_PER_SECOND = 5_000_000

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/arg_groups/fields/*.py": (
        "page_size",
        "kv_cache_dtype",
        "enable_hisparse",
        "expert_distribution_recorder_mode",
    ),
    "srt/distributed/parallel_state.py": ("cpu_group",),
    "srt/environ.py": (
        "SGLANG_DSA_FUSE_TOPK",
        "SGLANG_SIMULATE_UNIFORM_EXPERTS",
        "SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS",
    ),
    "srt/configs/model_config.py": ("context_len",),
    "srt/layers/attention/dsa/utils.py": (
        "hadamard_preserved",
        "aiter_can_use_preshuffle_paged_mqa",
    ),
    "srt/layers/attention/dsa_backend.py": ("forward_metadata", "real_page_table"),
    "srt/layers/layer_boundary/__init__.py": ("get_attn_tp_context",),
    "srt/layers/layer_boundary/adapters/attention.py": ("maybe_input_scattered",),
    "srt/layers/layer_boundary/residual/batch.py": (
        "start",
        "fold",
        "take_output",
        "set_written",
        "complete_output",
        "final_norm",
    ),
    "srt/layers/radix_attention.py": ("k_scale_float",),
    "srt/mem_cache/memory_pool.py": ("get_key_buffer", "get_index_k_with_scale_buffer"),
    "srt/model_executor/forward_batch_info.py": (
        "out_cache_loc",
        "seq_lens",
        "forward_mode",
        "is_idle",
    ),
    "srt/model_executor/forward_context.py": (
        "get_attn_backend",
        "get_token_to_kv_pool",
    ),
    "srt/models/deepseek_v2.py": (
        "model",
        "embed_tokens",
        "layers",
        "start_layer",
        "end_layer",
        "norm",
        "self_attn",
        "attn_mqa",
        "indexer",
        "gemm_output_zero_allocator_size",
        "next_full_attention_layer_id",
        "logits_processor",
        "lm_head",
        "forward(prev_topk_indices)",
    ),
    "srt/monokernels/spec.py": (
        "DeviceLimits",
        "MonoSpec",
        "Numerics",
        "NumericsKind",
        "Requirements",
        "widths",
        "deadline_s",
        "copies",
    ),
    "srt/runtime_context.py": (
        "get_parallel",
        "process_model_config",
        '"tp_rank"',
        '"tp_size"',
        '"tp_group"',
    ),
    "srt/utils/common.py": ("BumpAllocator",),
}

# Each covered layer against SGLang's own layer on the same input and caches
# (SGLANG_AMD_MONO_CHECK=1 in sgl_amd_mono.glm53.check): the stream it writes and
# the cache rows. The kernel quantizes the expert inputs to MXFP8 where SGLang
# quantizes them to MXFP4, runs the indexer's K projection in bf16 where SGLang
# runs it in block FP8, and sums the TP all-reduces in bf16, so the stream bound
# is looser than the cache bounds.
NUMERICS = Numerics(
    NumericsKind.TOLERANCE,
    {
        "layer.streams.cosine": 0.95,
        "layer.kv.cosine": 0.99,
        "layer.index.cosine": 0.99,
    },
)


def width(rows: int) -> int:
    """The kernel width a step of ``rows`` rows runs at."""
    return next(w for w in WIDTHS if w >= rows)


def covered_layers(config) -> list[int]:
    """The MoE layers. SGLang runs the dense layers 0 to 2 on its own path."""
    return list(range(config.first_k_dense_replace, config.num_hidden_layers))


def is_quark_mxfp4(hf_config) -> bool:
    """Whether the checkpoint is AMD Quark's MXFP4 export: MXFP4 weights by
    default, with per-layer overrides (block FP8 for the attention linears)."""
    quant = getattr(hf_config, "quantization_config", None) or {}
    if quant.get("quant_method") != "quark":
        return False
    weight = (quant.get("global_quant_config") or {}).get("weight") or {}
    return weight.get("dtype") == "fp4" and weight.get("group_size") == 32


def _attention_is_block_fp8(hf_config) -> bool:
    """Whether the Quark export overrides the attention linears with 128 x 128
    block FP8, as amd/GLM-5.3-Quark-MXFP4-AttnFP8 does."""
    quant = getattr(hf_config, "quantization_config", None) or {}
    for name, layer_cfg in (quant.get("layer_quant_config") or {}).items():
        if "self_attn" not in name:
            continue
        weight = (layer_cfg or {}).get("weight") or {}
        if (
            weight.get("dtype") == "fp8_e4m3"
            and weight.get("qscheme") == "per_block"
            and list(weight.get("block_size") or ()) in ([128, 128], [1, 128, 128])
        ):
            return True
    return False


def config_disable_reasons(cfg, hf_config) -> list[str]:
    reasons = [
        f"needs GLM-5.3's {key} = {want!r}, got {getattr(hf_config, key, None)!r}"
        for key, want in HF_CONFIG.items()
        if getattr(hf_config, key, None) != want
    ]
    if getattr(hf_config, "moe_layer_freq", 1) != 1:
        reasons.append("needs every layer after the dense ones to be MoE")
    # SGLang scales the softmax by YaRN's mscale, and the kernel's scale is fixed.
    rope = (
        getattr(hf_config, "rope_parameters", None)
        or getattr(hf_config, "rope_scaling", None)
        or {}
    )
    rope_type = rope.get("rope_type", rope.get("type", "default"))
    if rope_type != "default":
        reasons.append(f"RoPE {rope_type!r}: the kernel's softmax scale is unscaled")
    if cfg.page_size not in (None, PAGE_SIZE):
        reasons.append(f"--page-size {cfg.page_size}: the DSA caches need {PAGE_SIZE}")
    if not is_quark_mxfp4(hf_config):
        reasons.append("needs the Quark MXFP4 checkpoint (MXFP4 experts)")
    elif not _attention_is_block_fp8(hf_config):
        reasons.append("needs the Quark export's attention linears in 128 x 128 block FP8")
    if cfg.kv_cache_dtype != "fp8_e4m3":
        reasons.append(
            f"--kv-cache-dtype {cfg.kv_cache_dtype}: the kernel reads and writes an "
            "FP8 E4M3 MLA cache (--kv-cache-dtype fp8_e4m3)"
        )
    if importlib.util.find_spec("flydsl") is None:
        reasons.append("needs FlyDSL")
    return reasons


def _where(ids: list) -> str:
    return f"layer {ids[0]}" if len(ids) == 1 else f"{len(ids)} layers from {ids[0]}"


def model_disable_reasons(model, hf_config, copies: dict) -> list[str]:
    from sglang.srt.layers.attention.dsa.utils import (
        aiter_can_use_preshuffle_paged_mqa,
        hadamard_preserved,
    )
    from sglang.srt.runtime_context import get_parallel

    from sgl_amd_mono.glm53_mxfp4.gfx950.weights import experts_reason

    reasons = []
    covered = covered_layers(hf_config)
    # The routed and shared experts' intermediate width on this TP rank.
    inter = hf_config.moe_intermediate_size // get_parallel().tp_size
    rotated, plain, bad_scale = [], [], []
    for i in covered:
        layer = model.model.layers[i]
        why = experts_reason(layer.mlp, inter)
        if why is not None:
            reasons.append(f"layer {i}: {why}")
        attn = layer.self_attn
        k_scale = getattr(attn.attn_mqa, "k_scale_float", None)
        if k_scale not in (None, 1.0):
            bad_scale.append(i)
        if attn.indexer is not None:
            (rotated if hadamard_preserved(attn.indexer) else plain).append(i)
    if bad_scale:
        reasons.append(
            f"the MLA cache of {_where(bad_scale)} has a KV scale other than 1, and "
            "the kernel writes and reads the FP8 cache unscaled"
        )
    if rotated and plain:
        reasons.append(
            f"the indexers of {_where(rotated)} cache Hadamard-rotated keys and those "
            f"of {_where(plain)} unrotated keys: the kernel is built for one format"
        )
    if not aiter_can_use_preshuffle_paged_mqa():
        reasons.append(
            "SGLang keeps the index cache in its legacy one-token pages: the kernel "
            "reads the 64-token preshuffled pages"
        )
    first = model.model.layers[covered[0]]
    prev = model.model.layers[covered[0] - 1]
    if first.self_attn.indexer is None and prev.self_attn.indexer is None:
        reasons.append(
            f"layer {covered[0]} reuses the top-k of layer {covered[0] - 1}, which "
            "does not compute it"
        )
    return reasons


class _Layer:
    """One covered layer: its module, its weights, its ordinal among the layers of
    its kind (indexer or shared top-k) and its op per kernel width."""

    def __init__(self, i: int, module, weights, index: bool):
        self.i, self.module, self.w, self.index = i, module, weights, index
        self.kind = "index" if index else "shared"
        self.ordinal = 0
        self.ops: dict = {}  # width -> Glm5MonoKernel
        self.caches = None  # (kv [slots, 576] fp8, index [pages, 64, 132] uint8)


class _Step(msgspec.Struct):
    """The rows of one decode step, as every launch of that step reads them."""

    positions: torch.Tensor  # int64 [W]: a pad row's is 0
    slots: torch.Tensor  # int64 [W]: the cache slot of the row's new token, -1 for a pad row
    indptr: torch.Tensor  # int32 [W + 1]: row s attends top-k entries indptr[s] .. indptr[s + 1]
    req_ids: torch.Tensor  # int32 [W]: the page-table row of each row
    page_table: torch.Tensor  # int32 [S, pages]: SGLang's real page table
    topk: torch.Tensor  # int32 [W * 2048 + 1]: the compact top-k, as KV slots


class _Built:
    """One rank's monokernels: an op per (covered layer, width), the runtimes
    shared by the layers of one (width, kind), two output buffers per runtime,
    the step's rows per width, and the forward."""

    def __init__(self, lm, layers: dict, runtimes: dict, cos, sin, tp: int, device):
        from sgl_amd_mono.glm53_mxfp4.gfx950.config import HIDDEN

        self.lm, self.model = lm, lm.model
        self._layers, self.runtimes = layers, runtimes
        self.first, self.last = min(layers), max(layers)
        self.max_rows = TP_MAX_ROWS.get(tp, max(WIDTHS))
        self.cos, self.sin = cos, sin
        self.cur_pos = torch.zeros(1, dtype=torch.int32, device=device)
        self.topk_cols = torch.arange(TOPK, dtype=torch.int64, device=device)
        self.steps, self.pad_h, self.outs = {}, {}, {}
        i32, i64, bf16 = torch.int32, torch.int64, torch.bfloat16
        for w in sorted({w for w, _ in runtimes}):
            self.steps[w] = _Step(
                positions=torch.zeros(w, dtype=i64, device=device),
                slots=torch.full((w,), -1, dtype=i64, device=device),
                indptr=torch.zeros(w + 1, dtype=i32, device=device),
                req_ids=torch.zeros(w, dtype=i32, device=device),
                page_table=torch.zeros(w, 1, dtype=i32, device=device),
                # The last entry takes the scattered copies of a row's unused keys.
                topk=torch.zeros(w * TOPK + 1, dtype=i32, device=device),
            )
            self.pad_h[w] = torch.zeros(w, HIDDEN, dtype=bf16, device=device)
        for key in runtimes:
            # A layer writes the buffer of its ordinal's parity. The layer before
            # it is either of the other kind, which writes the other runtime's
            # buffers, or of this kind with the other parity, so a launch never
            # writes the tensor it reads.
            self.outs[key] = [
                torch.zeros(key[0], HIDDEN, dtype=bf16, device=device) for _ in range(2)
            ]

    def forward(
        self,
        plan,
        input_ids,
        positions,
        forward_batch,
        input_embeds=None,
        pp_proxy_tensors=None,
    ):
        """``DeepseekV2ForCausalLM.forward`` for a routed decode step."""
        lm = self.lm
        if plan.key.rows > self.max_rows:
            return type(lm).forward(
                lm, input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors
            )
        from sglang.srt.layers.layer_boundary import get_attn_tp_context

        with get_attn_tp_context().maybe_input_scattered(forward_batch):
            hidden = self._backbone(
                plan, input_ids, positions, forward_batch, input_embeds
            )
        return lm.logits_processor(input_ids, hidden, lm.lm_head, forward_batch, None)

    def _backbone(self, plan, input_ids, positions, forward_batch, input_embeds):
        """``DeepseekV2Model.forward``: the embedding and the dense layers on
        SGLang's path, the covered layers as launches, then the final norm."""
        from sglang.srt.layers.layer_boundary.residual.batch import (
            complete_output,
            final_norm,
            start,
        )
        from sglang.srt.utils.common import BumpAllocator

        m = self.model
        hidden = m.embed_tokens(input_ids) if input_embeds is None else input_embeds
        start(forward_batch)
        zero = BumpAllocator(
            buffer_size=(m.end_layer - m.start_layer) * 2,
            dtype=torch.float32,
            device=hidden.device,
        )
        gemm_zero = None
        if m.gemm_output_zero_allocator_size > 0:
            gemm_zero = BumpAllocator(
                buffer_size=m.gemm_output_zero_allocator_size,
                dtype=torch.float32,
                device=hidden.device,
            )
        topk_indices = None
        for i in range(m.start_layer, m.end_layer):
            L = self._layers.get(i)
            if L is not None:
                hidden = self._layer(
                    L, plan, positions, hidden, forward_batch, topk_indices
                )
                topk_indices = None
                continue
            hidden, topk_indices = m.layers[i](
                positions,
                hidden,
                forward_batch,
                zero,
                gemm_zero,
                None,
                prev_topk_indices=topk_indices,
                next_full_attention_layer_id=m.next_full_attention_layer_id.get(i),
            )
        hidden = complete_output(hidden, forward_batch)
        if not forward_batch.forward_mode.is_idle():
            hidden = final_norm(hidden, forward_batch, m.norm)
        return hidden

    def _layer(self, L: _Layer, plan, positions, hidden, forward_batch, topk_indices):
        """One covered layer's launch. Its input is the stream the previous layer
        wrote (the first covered layer) or the previous launch's output. It
        returns the launch's output, or the written stream after the last layer."""
        from sglang.srt.layers.layer_boundary.residual.batch import (
            fold,
            set_written,
            take_output,
        )

        S = plan.key.rows
        W = width(S)
        step = self._step(plan, positions, forward_batch, S, W)
        if L.i == self.first:
            # Layer 2's output still owes its residual add to the next input norm.
            h = take_output(fold(hidden, forward_batch), forward_batch)
            if S < W:
                pad = self.pad_h[W]
                pad[:S].copy_(h)
                pad[S:].zero_()
                h = pad
            elif not h.is_contiguous():
                h = h.contiguous()
            if not L.index:
                self._take_topk(topk_indices, step, S, W)
        else:
            h = hidden
        kv, idx = self._bind(L)
        out = self.outs[W, L.kind][L.ordinal % 2]
        op = L.ops[W]
        if L.index:
            op.forward(
                h,
                self.cur_pos,
                kv,
                kv,
                None,
                self.cos,
                self.sin,
                x_out=out,
                layer=L.ordinal,
                advance=False,
                index_cache=idx,
                positions=step.positions,
                slot_mapping=step.slots,
                sparse_kv_indptr=step.indptr,
                block_table=step.page_table,
                req_ids=step.req_ids,
                out_indices=step.topk,
            )
        else:
            op.forward(
                h,
                self.cur_pos,
                kv,
                kv,
                step.topk,
                self.cos,
                self.sin,
                x_out=out,
                layer=L.ordinal,
                advance=False,
                positions=step.positions,
                slot_mapping=step.slots,
                sparse_kv_indptr=step.indptr,
            )
        if L.i != self.last:
            return out
        for (w, _), rt in self.runtimes.items():
            if w == W:
                rt.advance_step()
        return set_written(out[:S], forward_batch)

    def _take_topk(self, topk_indices, step: _Step, S: int, W: int) -> None:
        """Layer 2's top-k (SGLang's [S, 2048] KV slots, -1 past each row's keys)
        into the compact buffer the next launch reads, row s at indptr[s]."""
        if topk_indices is None:
            raise RuntimeError("GLM-5.3 MXFP4 monokernel: no top-k from the layer before")
        start = step.indptr[:S].to(torch.int64).unsqueeze(1)
        count = (step.indptr[1 : S + 1] - step.indptr[:S]).to(torch.int64).unsqueeze(1)
        dst = torch.where(self.topk_cols < count, start + self.topk_cols, W * TOPK)
        step.topk.scatter_(
            0, dst.reshape(-1), topk_indices[:S].reshape(-1).to(torch.int32)
        )

    @staticmethod
    def _bind(L: _Layer):
        """The layer's MLA cache as FP8 [slots, 576] and, on an indexer layer, its
        index cache as uint8 [pages, 64, 132]. They are views of SGLang's pool,
        taken again when the pool tensors move (memory profiling runs real steps
        on caches that are replaced later)."""
        from sglang.srt.model_executor.forward_context import get_token_to_kv_pool

        pool = get_token_to_kv_pool()
        kv = pool.get_key_buffer(L.i)
        idx = pool.get_index_k_with_scale_buffer(L.i) if L.index else None
        if (
            L.caches is not None
            and L.caches[0].data_ptr() == kv.data_ptr()
            and (idx is None or L.caches[1].data_ptr() == idx.data_ptr())
        ):
            return L.caches
        if kv.element_size() != 1 or kv.shape[-1] != KV_ROW or not kv.is_contiguous():
            raise RuntimeError(
                f"GLM-5.3 MXFP4 monokernel: layer {L.i}: MLA cache {kv.dtype} "
                f"{tuple(kv.shape)}, needs contiguous FP8 [slots, 1, {KV_ROW}]"
            )
        kv = kv.view(torch.float8_e4m3fn).view(-1, KV_ROW)
        if idx is not None:
            if (
                idx.dtype != torch.uint8
                or not idx.is_contiguous()
                or idx.numel() % INDEX_PAGE_BYTES
            ):
                raise RuntimeError(
                    f"GLM-5.3 MXFP4 monokernel: layer {L.i}: index cache {idx.dtype} "
                    f"{tuple(idx.shape)}, needs contiguous uint8 pages of "
                    f"{INDEX_PAGE_BYTES} bytes"
                )
            idx = idx.view(-1, PAGE_SIZE, INDEX_PAGE_BYTES // PAGE_SIZE)
        L.caches = (kv, idx)
        return L.caches

    def _step(self, plan, positions, forward_batch, S: int, W: int) -> _Step:
        """The step's rows for every launch, built once a step (inside the graph
        when the step is captured)."""
        if "step" not in plan.state:
            from sglang.srt.model_executor.forward_context import get_attn_backend

            st = self.steps[W]
            loc = forward_batch.out_cache_loc[:S].to(torch.int64)
            valid = loc != 0
            st.slots[:S] = torch.where(valid, loc, -1)
            st.slots[S:] = -1
            st.positions[:S] = positions[:S].to(torch.int64)
            st.positions[S:] = 0
            seq = forward_batch.seq_lens[:S].to(torch.int32)
            count = torch.where(valid, seq.clamp(max=TOPK), 0)
            # indptr[0] stays 0 from its allocation.
            st.indptr[1 : S + 1] = torch.cumsum(count, 0, dtype=torch.int32)
            st.indptr[S + 1 :] = st.indptr[S]
            st.req_ids[:S] = torch.arange(S, dtype=torch.int32, device=loc.device)
            st.req_ids[S:] = 0
            md = get_attn_backend().forward_metadata
            plan.state["step"] = msgspec.structs.replace(
                st, page_table=md.real_page_table[:S]
            )
        return plan.state["step"]


def build(model, hf_config, ctx) -> _Built:
    from sglang.srt.layers.attention.dsa.utils import hadamard_preserved
    from sglang.srt.runtime_context import get_parallel, process_model_config

    from sgl_amd_mono.glm53.gfx950.runner import (
        compile_only,
        ensure_writable_flydsl_cache,
    )
    from sgl_amd_mono.glm53.gfx950.weights import rope_tables
    from sgl_amd_mono.glm53_mxfp4.gfx950.config import AttentionWeight, KvCacheLayout
    from sgl_amd_mono.glm53_mxfp4.gfx950.layout import INDEX_KEYS_PER_TASK
    from sgl_amd_mono.glm53_mxfp4.gfx950.op import (
        Glm5MonoKernel,
        prepare_glm5_weights,
    )
    from sgl_amd_mono.glm53_mxfp4.gfx950.packing import pack_bf16, pack_fp8
    from sgl_amd_mono.glm53_mxfp4.gfx950.weights import layer_weights

    ensure_writable_flydsl_cache()
    p = get_parallel()
    tp, rank = p.tp_size, p.tp_rank
    device = model.model.layers[0].input_layernorm.weight.device
    copies = ctx.copies
    covered = covered_layers(hf_config)
    modules = {i: model.model.layers[i] for i in covered}
    layers = {
        i: _Layer(
            i,
            m,
            layer_weights(m, rank, tp, with_indexer=m.self_attn.indexer is not None),
            index=m.self_attn.indexer is not None,
        )
        for i, m in modules.items()
    }
    rotations = {
        hadamard_preserved(m.self_attn.indexer)
        for m in modules.values()
        if m.self_attn.indexer is not None
    }
    if len(rotations) != 1:
        raise ValueError("GLM-5.3 MXFP4 monokernel: the indexers' key formats differ")
    index_hadamard = rotations.pop()
    cos, sin = rope_tables(list(modules.values()), copies)
    context_len = process_model_config().context_len
    # The copied table is SGLang's at build, and a longer served context would
    # read past it.
    if cos.shape[0] < context_len:
        raise ValueError(
            f"GLM-5.3 MXFP4 monokernel: the rotary table holds {cos.shape[0]} "
            f"positions, the server's context is {context_len}"
        )
    # The keys an indexer row scores at most: the served context's, in whole
    # tasks. It sizes the score scratch only (8 bytes a key a row).
    max_seq = -(-context_len // INDEX_KEYS_PER_TASK) * INDEX_KEYS_PER_TASK
    cap = TP_MAX_ROWS.get(tp, max(WIDTHS))
    widths = sorted({width(r) for r in ctx.widths if r <= cap})
    if not widths:
        raise ValueError(
            f"GLM-5.3 MXFP4 monokernel: TP{tp} runs at most {cap} rows on the "
            f"kernels, and this server's decode graphs route {sorted(ctx.widths)}"
        )
    kinds = {
        "index": [i for i in covered if layers[i].index],
        "shared": [i for i in covered if not layers[i].index],
    }
    for ids in kinds.values():
        for n, i in enumerate(ids):
            layers[i].ordinal = n
    # The packed attention, router and indexer tensors of a layer, once for all
    # widths.
    prepared = {}
    for i, L in layers.items():
        packed = prepare_glm5_weights(L.w, AttentionWeight.FP8_BLOCK128)
        if L.index:
            packed["w_index_k"] = pack_bf16(L.w.t["w_index_k"])
            packed["w_index_q"] = pack_fp8(L.w.t["w_index_q"])
            packed["w_index_w"] = pack_bf16(L.w.t["w_index_w"])
        for name in ("w_qkv_a", "w_q_b", "w_uk", "w_uv", "w_o", "w_r"):
            copies["vLLM MonoKernel packed attention and router", i, name] = packed[name]
        for name in ("w_index_k", "w_index_q", "w_index_w"):
            if name in packed:
                copies["vLLM MonoKernel packed indexer", i, name] = packed[name]
        for name in ("g_index_k", "b_index_k", "bias"):
            if name in L.w.t:
                copies["vLLM MonoKernel f32 vectors", i, name] = L.w.t[name]
        prepared[i] = packed
    poll_limit = max(1, int(ctx.deadline_s * POLLS_PER_SECOND))
    index_options = dict(
        index_max_seq=max_seq,
        index_paged=True,
        index_block_size=PAGE_SIZE,
        index_block_bytes=INDEX_PAGE_BYTES,
        index_shuffled=True,
        index_k_bf16=True,
        index_hadamard=index_hadamard,
    )
    runtimes = {}
    group = p.tp_group.cpu_group
    # The peer handshake of each runtime is collective, so every rank builds them
    # in the same order.
    for w in widths:
        for kind, ids in kinds.items():
            for i in ids:
                L = layers[i]
                op = Glm5MonoKernel(
                    L.w,
                    w,
                    rank=rank,
                    npes=tp,
                    group=group,
                    topk=TOPK,
                    launches_per_step=len(ids),
                    with_indexer=L.index,
                    attention_weight=AttentionWeight.FP8_BLOCK128,
                    kv_cache_layout=KvCacheLayout.ATOM,
                    kv_cache_dtype="fp8",
                    prepared_weights=prepared[i],
                    runtime=runtimes.get((w, kind)),
                    native_fp4_mfma=True,
                    poll_limit=poll_limit,
                    poll_early_out=False,
                    **(index_options if L.index else {}),
                )
                if (w, kind) in runtimes:
                    # The layers of one (width, kind) build the same kernel, so
                    # they share the first one's launcher and its compiled code.
                    op.launch = runtimes[w, kind].launch
                else:
                    # SGLang's convention for a failed wait: stop every rank with
                    # a GPU fault at address 0x0. The kernel stores its expired-poll
                    # flags at these two addresses, so pointing both at the null
                    # page turns an expired wait into that fault.
                    for name in ("poll_err_addr", "poll_xrank_addr"):
                        off = op.scr_layout[name]
                        op.scratch[off : off + 8].view(torch.int64).fill_(0)
                    runtimes[w, kind] = op
                L.ops[w] = op
    # Every op reads the packed tensors, so drop the row-major copies, which the
    # builds above needed for their scale geometry.
    for i, L in layers.items():
        for name, t in prepared[i].items():
            if name in L.w.t and name not in ("w_ug", "s_ug", "w_dn", "s_dn"):
                L.w.t[name] = t
    # Every rank builds every width. FlyDSL's disk cache hands a rank the kernels
    # another rank built (a lock per kernel), so each rank starts at its own share.
    order = [(w, k) for w in widths for k in kinds]
    start = len(order) * rank // tp
    for key in order[start:] + order[:start]:
        compile_only(runtimes[key].launch)
    logger.info(
        "GLM-5.3 MXFP4 monokernel (vllm-project/vllm#60831): layers %d..%d, %d with "
        "the indexer, widths %s, index-key Hadamard %s, poll limit %d",
        min(covered),
        max(covered),
        len(kinds["index"]),
        widths,
        "on" if index_hadamard else "off",
        poll_limit,
    )
    return _Built(model, layers, runtimes, cos, sin, tp, device)


SPEC = MonoSpec(
    name="glm53_mxfp4_vllm_mono",
    devices={
        "gfx950": DeviceLimits(widths=ROWS, tp_sizes=TP_SIZES, resident_blocks=BLOCKS)
    },
    features=frozenset(),
    requires=Requirements(
        dtypes=frozenset({"bfloat16"}),
        attention_backends=frozenset({"dsa"}),
        server_args={
            "kv_cache_dtype": frozenset({"fp8_e4m3"}),
            "moe_runner_backend": frozenset({"auto", "aiter"}),
            # HiSparse maps the logical KV slots to device rows, and the kernels
            # write raw slots.
            "enable_hisparse": frozenset({False}),
            # The launches record no expert distribution.
            "expert_distribution_recorder_mode": frozenset({None}),
        },
        env={
            # Layer 3 reads layer 2's top-k as KV slots. Unfused top-k hands out
            # positions.
            "SGLANG_DSA_FUSE_TOPK": True,
            # Benchmark routing overrides: the kernels route with the model's
            # own router.
            "SGLANG_SIMULATE_UNIFORM_EXPERTS": False,
            "SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS": False,
        },
    ),
    numerics=NUMERICS,
    model_disable_reasons=model_disable_reasons,
    build=build,
    api_version=1,  # sglang.srt.monokernels.spec.MONOKERNEL_API_VERSION
    config_disable_reasons=config_disable_reasons,
)
