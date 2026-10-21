# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copied from vllm-project/vllm#60831 at 6deee07c52
# (vllm/models/deepseek_v32/amd/ops/mono/dispatch.py), imports made package-relative.
# Adapted from ROCm/aiter#6173 at c39b56c36 (MIT License),
# Copyright (C) 2026, Advanced Micro Devices, Inc.:
# aiter/ops/flydsl/kernels/glm5_mono/dispatch.py

"""Error raised when the MonoKernel cannot take a model or runtime state."""

from __future__ import annotations


class MonoUnsupported(Exception):
    """The loaded model or runtime state cannot use the native path."""
