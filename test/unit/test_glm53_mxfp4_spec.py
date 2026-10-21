"""The GLM-5.3 Quark MXFP4 spec (sgl_amd_mono.glm53_mxfp4) and the selector that
shares GlmMoeDsaForCausalLM with the block-FP8 spec, on CPU: which spec each
checkpoint gets, which KV cache dtype each one accepts, the layers the MXFP4
forward covers, and its row routing."""

from types import SimpleNamespace

from sglang.srt.monokernels import spec as registry

import sgl_amd_mono
from sgl_amd_mono import glm_moe_dsa
from sgl_amd_mono.glm53 import HF_CONFIG
from sgl_amd_mono.glm53 import spec as glm53
from sgl_amd_mono.glm53_mxfp4 import spec as mxfp4

_BLOCK_FP8 = {
    "dtype": "fp8_e4m3",
    "qscheme": "per_block",
    "block_size": [128, 128],
}

# amd/GLM-5.3-Quark-MXFP4-AttnFP8's quantization (its config.json, cut to the
# fields the spec reads)
QUARK = {
    "quant_method": "quark",
    "global_quant_config": {
        "weight": {"dtype": "fp4", "qscheme": "per_group", "group_size": 32}
    },
    "layer_quant_config": {
        "model.layers.23.self_attn.o_proj": {"weight": _BLOCK_FP8},
        "model.layers.6.self_attn.indexer.wq_b": {"weight": _BLOCK_FP8},
    },
}


def _hf(quantization_config):
    return SimpleNamespace(
        **HF_CONFIG,
        num_hidden_layers=78,
        first_k_dense_replace=3,
        moe_layer_freq=1,
        quantization_config=quantization_config,
        rope_parameters={"rope_theta": 8000000, "rope_type": "default"},
        architectures=["GlmMoeDsaForCausalLM"],
    )


MXFP4 = _hf(QUARK)
FP8 = _hf({"quant_method": "fp8", "weight_block_size": [128, 128]})


def _server(kv_cache_dtype):
    # the server arguments the specs' requirements and config checks read
    return SimpleNamespace(
        page_size=None,
        kv_cache_dtype=kv_cache_dtype,
        moe_runner_backend="auto",
        enable_hisparse=False,
        expert_distribution_recorder_mode=None,
    )


def test_the_plugin_registers_the_selector():
    sgl_amd_mono.register_specs()
    found, why = registry.find_spec(
        backend=sgl_amd_mono.BACKEND,
        architectures=["GlmMoeDsaForCausalLM"],
        isa="gfx950",
    )
    assert found is glm_moe_dsa.SPEC, why


def test_each_checkpoint_gets_its_spec():
    assert glm_moe_dsa.chosen(MXFP4) is mxfp4.SPEC
    assert glm_moe_dsa.chosen(FP8) is glm53.SPEC


def test_the_selector_accepts_what_either_spec_accepts():
    kv = glm_moe_dsa.SPEC.requires.server_args["kv_cache_dtype"]
    assert {"auto", "bfloat16", "fp8_e4m3"} <= set(kv)


def test_each_spec_keeps_its_own_kv_cache_dtype():
    reasons = glm_moe_dsa.SPEC.config_disable_reasons
    assert reasons(_server("fp8_e4m3"), MXFP4) == []
    text = " ".join(reasons(_server("auto"), MXFP4))
    assert "--kv-cache-dtype" in text and "glm53_mxfp4" in text
    assert reasons(_server("auto"), FP8) == []
    text = " ".join(reasons(_server("fp8_e4m3"), FP8))
    assert "kv_cache_dtype" in text and "glm53_hip" in text


def test_a_quark_checkpoint_without_block_fp8_attention_is_refused():
    plain = dict(QUARK, layer_quant_config={})
    text = " ".join(mxfp4.config_disable_reasons(_server("fp8_e4m3"), _hf(plain)))
    assert "block FP8" in text


def test_the_mxfp4_forward_covers_the_moe_layers_at_the_next_width():
    assert mxfp4.covered_layers(MXFP4) == list(range(3, 78))
    # a one-row launch never completes on gfx950, so one row runs at width 2
    assert [mxfp4.width(r) for r in mxfp4.ROWS] == [2, 2, 4, 4, 8, 8, 8, 8]


class _Lm:
    """A model whose own forward records that it ran."""

    def __init__(self):
        self.model, self.ran = SimpleNamespace(), []

    def forward(self, *args):
        self.ran.append(args)
        return "the model's forward"


def test_tp4_runs_at_most_four_rows_on_the_kernels():
    # At TP4 the 8-row launch is slower than SGLang's forward, so a step of five
    # or more rows runs the model's own forward
    runtimes = {(w, k): None for w in (2, 4) for k in ("index", "shared")}
    layers = {i: None for i in range(3, 78)}
    lm = _Lm()
    built = mxfp4._Built(lm, layers, runtimes, None, None, 4, "cpu")
    key = registry.ShapeKey(forward_mode="decode", batch_size=5, tokens_per_request=1)
    assert built.forward(registry.Plan(key=key), "ids", "pos", "batch") == (
        "the model's forward"
    )
    assert lm.ran == [("ids", "pos", "batch", None, None)]
    assert mxfp4._Built(lm, layers, runtimes, None, None, 8, "cpu").max_rows == 8
