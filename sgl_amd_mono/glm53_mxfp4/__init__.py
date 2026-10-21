# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3 Quark MXFP4 (amd/GLM-5.3-Quark-MXFP4-AttnFP8) with an FP8 KV cache: the
GLM-5 decode MonoKernel of vllm-project/vllm#60831 at 6deee07c52, ported to
SGLang's monokernel interface. The spec and its SGLang glue are in ``spec``, the
kernel, its host wrapper and the weight readers in ``gfx950``.

The kernel files come from vllm/models/deepseek_v32/amd/ops/mono/ of that PR,
which adapted ROCm/aiter#6173 (Apache-2.0, Copyright (c) 2025 FlyDSL Project
Contributors); each file names its origin and what changed.
"""
