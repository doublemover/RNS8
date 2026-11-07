"""Relevant execution-variable names shared by request and native observations."""

from __future__ import annotations

ENVIRONMENT_PREFIXES = (
    "RNS8_",
    "OMP_",
    "HIP_",
    "ROCR_",
    "ROCP_",
    "ROCPROF",
    "HSA_",
    "KMP_",
    "PERF_",
    "MULL_",
)
ENVIRONMENT_KEYS = frozenset(("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH", "LD_PRELOAD"))


def relevant_environment_key(key: str) -> bool:
    return key.startswith(ENVIRONMENT_PREFIXES) or key in ENVIRONMENT_KEYS
