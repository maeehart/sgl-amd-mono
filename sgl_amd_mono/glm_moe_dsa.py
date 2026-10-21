# SPDX-License-Identifier: Apache-2.0
"""GlmMoeDsaForCausalLM on --monokernel-backend amd: one registry entry for the
two GLM-5.3 specs, chosen by the checkpoint.

SGLang's registry takes one spec per HF architecture, and GLM-5.3 ships both as
the block-FP8 checkpoint (zai-org/GLM-5.3, served by ``sgl_amd_mono.glm53``) and
as AMD's Quark export with MXFP4 experts and block-FP8 attention
(amd/GLM-5.3-Quark-MXFP4-AttnFP8, served by ``sgl_amd_mono.glm53_mxfp4``, which
runs the kernel of vllm-project/vllm#60831). SGLang checks a spec's
``requires`` before anything else, so this spec requires what both accept, and
``config_disable_reasons`` then holds the chosen spec to its own requirements.
"""

from __future__ import annotations

from sglang.srt.monokernels.spec import MonoSpec, Numerics, Requirements

from sgl_amd_mono.glm53 import spec as fp8
from sgl_amd_mono.glm53_mxfp4 import spec as mxfp4

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/monokernels/spec.py": ("MonoSpec", "Numerics", "Requirements"),
}


def chosen(hf_config) -> MonoSpec:
    """The spec of this checkpoint: the Quark MXFP4 one or the block-FP8 one."""
    return mxfp4.SPEC if mxfp4.is_quark_mxfp4(hf_config) else fp8.SPEC


def _config_disable_reasons(cfg, hf_config) -> list[str]:
    spec = chosen(hf_config)
    reasons = [
        f"{arg} {getattr(cfg, arg)!r} is not one of {sorted(map(str, ok))} "
        f"({spec.name})"
        for arg, ok in spec.requires.server_args.items()
        if ok and getattr(cfg, arg) not in ok
    ]
    return reasons + spec.config_disable_reasons(cfg, hf_config)


def _model_disable_reasons(model, hf_config, copies: dict) -> list[str]:
    return chosen(hf_config).model_disable_reasons(model, hf_config, copies)


def _build(model, hf_config, ctx):
    return chosen(hf_config).build(model, hf_config, ctx)


def _union(a: Requirements, b: Requirements) -> Requirements:
    """What either spec accepts. Both specs need the same dtypes, attention
    backend, HF config fields and environment, which this asserts."""
    assert (a.dtypes, a.attention_backends, dict(a.env)) == (
        b.dtypes,
        b.attention_backends,
        dict(b.env),
    )
    args = {
        k: a.server_args.get(k, frozenset()) | b.server_args.get(k, frozenset())
        for k in set(a.server_args) | set(b.server_args)
    }
    return Requirements(
        dtypes=a.dtypes,
        attention_backends=a.attention_backends,
        server_args=args,
        hf_config=dict(a.hf_config),
        env=dict(a.env),
    )


def _looser(a: Numerics, b: Numerics) -> Numerics:
    """Each bound at the looser of the two specs' values."""
    assert a.kind == b.kind
    bounds = dict(a.bounds)
    for name, v in b.bounds.items():
        if name not in bounds:
            bounds[name] = v
        elif name.endswith(".cosine"):
            bounds[name] = min(bounds[name], v)
        else:
            bounds[name] = max(bounds[name], v)
    return Numerics(a.kind, bounds)


assert fp8.SPEC.devices == mxfp4.SPEC.devices
assert fp8.SPEC.features == mxfp4.SPEC.features
assert fp8.SPEC.load_tensor is None and mxfp4.SPEC.load_tensor is None

SPEC = MonoSpec(
    name="glm53_hip (block FP8) or glm53_mxfp4_vllm_mono (Quark MXFP4)",
    devices=fp8.SPEC.devices,
    features=fp8.SPEC.features,
    requires=_union(fp8.SPEC.requires, mxfp4.SPEC.requires),
    numerics=_looser(fp8.SPEC.numerics, mxfp4.SPEC.numerics),
    model_disable_reasons=_model_disable_reasons,
    build=_build,
    api_version=1,  # sglang.srt.monokernels.spec.MONOKERNEL_API_VERSION
    config_disable_reasons=_config_disable_reasons,
)
