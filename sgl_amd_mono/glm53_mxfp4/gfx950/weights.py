# SPDX-License-Identifier: Apache-2.0
"""A covered layer of amd/GLM-5.3-Quark-MXFP4-AttnFP8 as the vLLM MonoKernel reads
it, built from SGLang's loaded modules (``LayerWeights`` of ``storage.py``).

This plays the part of vllm/models/deepseek_v32/amd/mono/weights.py of
vllm-project/vllm#60831. The routed and shared MXFP4 experts are SGLang's tensors
in place: SGLang shuffles them for AITER's (16, 16) tiles and its scales with
``e8m0_shuffle``, which is the ATOM layout the kernel reads, and it keeps the
shared expert in slot 256. Everything else is a copy, made once at build: the
block-FP8 attention projections row-major (SGLang keeps them in AITER's (16, 16)
layout, and the kernel packs its own tiles from row-major values), W_UK and W_UV
from SGLang's per-tensor FP8 ``w_kc`` and ``w_vc``, the indexer's K projection
dequantized to bf16 (the vLLM kernel reads it as bf16), and the indexer's k
LayerNorm in f32.
"""

from __future__ import annotations

from typing import Optional

import torch

from .config import (
    GLM5_GLOBAL_HEADS,
    HIDDEN,
    KV_LORA,
    NOPE_DIM,
    PE_DIM,
    Q_LORA,
    V_DIM,
    Mxfp4ScaleLayout,
    Mxfp4WeightLayout,
    glm5_tp_config,
)
from .layout import INDEX_DIM, INDEX_HEADS
from .storage import LayerWeights, _unshuffle_linear_weight

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/models/deepseek_v2.py": (
        "self_attn",
        "mlp",
        "input_layernorm",
        "post_attention_layernorm",
        "fused_qkv_a_proj_with_mqa",
        "q_a_layernorm",
        "kv_a_layernorm",
        "q_b_proj",
        "o_proj",
        "w_kc",
        "w_vc",
        "w_scale",
        "indexer",
        "gate",
        "e_score_correction_bias",
        "experts",
    ),
    "srt/layers/attention/dsa/dsa_indexer.py": (
        "wq_b",
        "wk",
        "weights_proj",
        "k_norm",
    ),
    "srt/layers/quantization/fp8.py": (
        "weight_scale_inv",
        "aiter_bpreshuffled",
        "is_shuffled",
    ),
    "srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py": (
        "w13_weight",
        "w13_weight_scale",
        "w2_weight",
        "w2_weight_scale",
        "is_shuffled",
    ),
}

N_EXPERTS = 256
# SGLang appends the fused shared expert to the routed ones, as expert 256.
PHYSICAL_EXPERTS = N_EXPERTS + 1
QKV_A = Q_LORA + KV_LORA + PE_DIM
BLOCK = 128


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def _need(name: str, t: torch.Tensor, shape: tuple, dtypes: tuple) -> torch.Tensor:
    """Return ``t`` if it is contiguous with this shape and one of these dtypes,
    or raise a ValueError that names what was found instead."""
    if tuple(t.shape) != shape or t.dtype not in dtypes or not t.is_contiguous():
        raise ValueError(
            f"needs {name} as contiguous {'/'.join(map(str, dtypes))} {shape}, got "
            f"{t.dtype} {tuple(t.shape)} strides {tuple(t.stride())}"
        )
    return t


def _shuffled(linear) -> bool:
    """Whether SGLang rewrote this linear's weight into AITER's (16, 16) layout.
    SGLang sets one of these two markers on every linear it shuffles."""
    return bool(
        getattr(linear, "aiter_bpreshuffled", False)
        or getattr(linear.weight, "is_shuffled", False)
    )


def block_fp8(name: str, linear, rows: int, cols: int):
    """A block-FP8 linear as row-major FP8 E4M3 [rows, cols] values and its f32
    128 x 128 block scales. A weight SGLang shuffled for AITER is unshuffled into
    a new tensor, because the kernel packs its tiles from row-major values."""
    weight = linear.weight
    scale = getattr(linear, "weight_scale_inv", None)
    if scale is None:
        raise ValueError(f"needs {name} as block FP8, but it has no weight_scale_inv")
    _need(f"{name} weight", weight, (rows, cols), (torch.float8_e4m3fn,))
    _need(
        f"{name} scale",
        scale,
        (_cdiv(rows, BLOCK), _cdiv(cols, BLOCK)),
        (torch.float32,),
    )
    if _shuffled(linear):
        # The unshuffle reads the marker from the tensor it is given.
        marked = weight.detach().view(torch.float8_e4m3fn)
        marked.is_shuffled = True
        weight = _unshuffle_linear_weight(marked)
    return weight.detach().contiguous(), scale.detach().contiguous()


def dequantize_block_fp8(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Row-major FP8 values times their 128 x 128 block scales, in bf16."""
    rows, cols = weight.shape
    full = scale.repeat_interleave(BLOCK, 0)[:rows].repeat_interleave(BLOCK, 1)
    return (weight.float() * full[:, :cols]).to(torch.bfloat16).contiguous()


def uk_uv(attn, heads: int):
    """W_UK [H * 512, 192] and W_UV [H * 256, 512] row-major from SGLang's per-tensor
    FP8 ``w_kc`` and ``w_vc``, each with a scale tensor that repeats SGLang's one
    ``w_scale``. These are the values SGLang's own absorbed attention multiplies
    by (SGLang requantizes the checkpoint's block-FP8 kv_b_proj to one scale)."""
    w_kc, w_vc, w_scale = attn.w_kc, attn.w_vc, getattr(attn, "w_scale", None)
    if w_kc.dtype != torch.float8_e4m3fn or w_scale is None or w_scale.numel() != 1:
        raise ValueError(
            "needs w_kc and w_vc as per-tensor FP8 E4M3 with a scalar w_scale, got "
            f"{w_kc.dtype} and {None if w_scale is None else tuple(w_scale.shape)}"
        )
    if tuple(w_kc.shape) != (heads, NOPE_DIM, KV_LORA) or tuple(w_vc.shape) != (
        heads,
        KV_LORA,
        V_DIM,
    ):
        raise ValueError(
            f"needs w_kc [{heads}, {NOPE_DIM}, {KV_LORA}] and w_vc [{heads}, "
            f"{KV_LORA}, {V_DIM}], got {tuple(w_kc.shape)} and {tuple(w_vc.shape)}"
        )
    w_uk = w_kc.transpose(1, 2).reshape(heads * KV_LORA, NOPE_DIM).contiguous()
    w_uv = w_vc.transpose(1, 2).reshape(heads * V_DIM, KV_LORA).contiguous()
    scale = float(w_scale.float().item())
    dev = w_uk.device
    # The kernel reads W_UK scales per 128 rows and 64 columns, and W_UV scales
    # per 128 rows (uv_scale_rows) and 128 columns.
    s_uk = torch.full(
        (heads * KV_LORA // BLOCK, NOPE_DIM // 64), scale, dtype=torch.float32, device=dev
    )
    s_uv = torch.full(
        (heads * V_DIM // BLOCK, KV_LORA // BLOCK), scale, dtype=torch.float32, device=dev
    )
    return w_uk, s_uk, w_uv, s_uv


def indexer_weights(attn) -> dict:
    """The indexer tensors of a layer that computes its top-k: the K projection
    dequantized to bf16 [128, 6144], weights_proj bf16 [32, 6144], wq_b row-major
    block FP8 [4096, 2048] with its scales, and the k LayerNorm in f32."""
    ind = attn.indexer
    w_k, s_k = block_fp8("indexer.wk", ind.wk, INDEX_DIM, HIDDEN)
    w_q, s_q = block_fp8("indexer.wq_b", ind.wq_b, INDEX_HEADS * INDEX_DIM, Q_LORA)
    w_w = _need(
        "indexer.weights_proj",
        ind.weights_proj.weight,
        (INDEX_HEADS, HIDDEN),
        (torch.bfloat16,),
    )
    ln = ind.k_norm
    eps = getattr(ln, "variance_epsilon", getattr(ln, "eps", None))
    if eps != 1e-6 or ln.bias is None:
        raise ValueError(
            f"needs the indexer's k_norm as a LayerNorm with eps 1e-6 and a bias, "
            f"got eps {eps!r}"
        )
    return {
        "w_index_k": dequantize_block_fp8(w_k, s_k),
        "w_index_w": w_w.detach(),
        "w_index_q": w_q,
        "s_index_q": s_q,
        "g_index_k": ln.weight.detach().float().contiguous(),
        "b_index_k": ln.bias.detach().float().contiguous(),
    }


def experts_reason(moe, inter: int) -> Optional[str]:
    """Why the kernel cannot read this MoE's MXFP4 experts in place, or None."""
    experts = getattr(moe, "experts", None)
    w13 = getattr(experts, "w13_weight", None)
    if w13 is None:
        return "no fused experts (w13_weight)"
    if w13.dtype != torch.uint8:
        return f"needs MXFP4 experts stored as uint8, got {w13.dtype}"
    if w13.shape[0] != PHYSICAL_EXPERTS:
        return (
            f"needs the shared expert fused as expert {N_EXPERTS} "
            f"({PHYSICAL_EXPERTS} experts), got {w13.shape[0]}"
        )
    shapes = {
        "w13_weight": (PHYSICAL_EXPERTS, 2 * inter, HIDDEN // 2),
        "w13_weight_scale": (PHYSICAL_EXPERTS, 2 * inter, HIDDEN // 32),
        "w2_weight": (PHYSICAL_EXPERTS, HIDDEN, inter // 2),
        "w2_weight_scale": (PHYSICAL_EXPERTS, HIDDEN, inter // 32),
    }
    for name, shape in shapes.items():
        t = getattr(experts, name, None)
        if t is None or tuple(t.shape) != shape or not t.is_contiguous():
            got = None if t is None else tuple(t.shape)
            return f"needs {name} as contiguous {shape} (the TP shard), got {got}"
    for name in ("w13_weight", "w2_weight"):
        if not getattr(getattr(experts, name), "is_shuffled", False):
            return (
                f"needs {name} in AITER's (16, 16) MXFP4 layout, which SGLang applies "
                "with SGLANG_USE_AITER=1 on gfx950"
            )
    return None


def moe_weights(moe, inter: int) -> dict:
    """The router, its correction bias and the experts of one MoE layer."""
    why = experts_reason(moe, inter)
    if why is not None:
        raise ValueError(why)
    experts, gate = moe.experts, moe.gate
    router = _need(
        "gate", gate.weight, (N_EXPERTS, HIDDEN), (torch.bfloat16, torch.float32)
    )
    if router.dtype != torch.bfloat16 and not torch.equal(
        router.to(torch.bfloat16).float(), router
    ):
        # SGLang routes an f32 gate in f32, and the kernel's router GEMV is bf16.
        raise ValueError("needs router weights exactly representable in bf16")
    bias = _need(
        "e_score_correction_bias",
        gate.e_score_correction_bias,
        (N_EXPERTS,),
        (torch.float32, torch.bfloat16),
    )

    def fp4(t: torch.Tensor) -> torch.Tensor:
        # A view of SGLang's tensor that carries the marker the kernel's storage
        # check reads. No bytes are copied.
        view = t.detach().view(torch.uint8)
        view.is_shuffled = True
        return view

    return {
        "w_r": router.detach().to(torch.bfloat16).contiguous(),
        "bias": bias.detach().float().contiguous(),
        "w_ug": fp4(experts.w13_weight),
        "s_ug": experts.w13_weight_scale.detach().view(torch.uint8),
        "w_dn": fp4(experts.w2_weight),
        "s_dn": experts.w2_weight_scale.detach().view(torch.uint8),
    }


def layer_weights(layer, rank: int, tp: int, with_indexer: bool) -> LayerWeights:
    """One covered layer's tensors, in the names of the vLLM kernel."""
    cfg = glm5_tp_config(tp)
    attn = layer.self_attn
    heads = GLM5_GLOBAL_HEADS // tp
    bf16 = (torch.bfloat16,)
    tensors = {
        "g_in": _need("input_layernorm", layer.input_layernorm.weight, (HIDDEN,), bf16),
        "g_q": _need("q_a_layernorm", attn.q_a_layernorm.weight, (Q_LORA,), bf16),
        "g_kv": _need("kv_a_layernorm", attn.kv_a_layernorm.weight, (KV_LORA,), bf16),
        "g_post": _need(
            "post_attention_layernorm",
            layer.post_attention_layernorm.weight,
            (HIDDEN,),
            bf16,
        ),
    }
    tensors["w_qkv_a"], tensors["s_qkv_a"] = block_fp8(
        "fused_qkv_a_proj_with_mqa", attn.fused_qkv_a_proj_with_mqa, QKV_A, HIDDEN
    )
    tensors["w_q_b"], tensors["s_q_b"] = block_fp8(
        "q_b_proj", attn.q_b_proj, heads * (NOPE_DIM + PE_DIM), Q_LORA
    )
    tensors["w_o"], tensors["s_o"] = block_fp8(
        "o_proj", attn.o_proj, HIDDEN, heads * V_DIM
    )
    (
        tensors["w_uk"],
        tensors["s_uk"],
        tensors["w_uv"],
        tensors["s_uv"],
    ) = uk_uv(attn, heads)
    tensors.update(moe_weights(layer.mlp, cfg.inter))
    if with_indexer:
        tensors.update(indexer_weights(attn))
    return LayerWeights(
        heads,
        tensors,
        cfg,
        rank,
        tp,
        mxfp4_weight_layout=Mxfp4WeightLayout.ATOM,
        mxfp4_scale_layout=Mxfp4ScaleLayout.ATOM,
        physical_experts=PHYSICAL_EXPERTS,
    )
