"""Shareable session environment: explicit allowlist, never a full process dump."""
from __future__ import annotations

import re
from typing import Mapping

VISIBILITY_KEYS = ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL", "CUDA_VISIBLE_DEVICES")
OVERRIDE_KEYS = ("HSA_OVERRIDE_GFX_VERSION", "HSA_XNACK", "HSA_ENABLE_SDMA")
PATH_KEYS = ("ROCM_PATH", "HIP_PATH", "HIP_CLANG_PATH", "HIP_DEVICE_LIB_PATH", "LD_LIBRARY_PATH", "LIBRARY_PATH", "CPATH")
ALLOWLIST = (*VISIBILITY_KEYS, *OVERRIDE_KEYS, *PATH_KEYS, "OMP_NUM_THREADS", "SLURM_JOB_ID", "SLURM_STEP_ID")
BUILD_ENV_KEYS = ("PATH", "HOME", "USERPROFILE", "TMP", "TEMP", "TMPDIR", "SystemRoot", "SYSTEMROOT",
                  "COMSPEC", "PATHEXT", "ProgramFiles", "ProgramFiles(x86)", "ProgramData", "LOCALAPPDATA",
                  "INCLUDE", "LIB", "LIBPATH", "VSINSTALLDIR", "VCINSTALLDIR", "VCToolsInstallDir",
                  "WindowsSdkDir", "WindowsSDKVersion", "VSCMD_ARG_TGT_ARCH", "LANG", "LC_ALL", "TZ")


def safe_value(value: str | None) -> str | None:
    if value is None:
        return None
    # Path variables occasionally contain authenticated URLs. Do not serialize
    # credentials, query parameters, control characters or unbounded values.
    if len(value) > 4096 or re.search(r"[\x00-\x1f\x7f]", value) or re.search(r"https?://", value, re.I):
        return "<redacted: unsafe environment value>"
    return value


def snapshot(environment: Mapping[str, str]) -> dict[str, str | None]:
    return {key: safe_value(environment.get(key)) for key in ALLOWLIST}


def execution_blockers(environment: Mapping[str, str]) -> list[str]:
    blockers = []
    for key in VISIBILITY_KEYS:
        if key in environment:
            blockers.append(key + " is inherited; ordinal remapping is ambiguous. Use an unmasked allocation and explicit --devices.")
    for key in OVERRIDE_KEYS:
        if key in environment:
            blockers.append(key + " is inherited; diagnostic runtime overrides cannot qualify a baseline session.")
    return blockers


def isolated_environment(environment: Mapping[str, str], device: int) -> dict[str, str]:
    if type(device) is not int or device < 0:
        raise ValueError("device must be a nonnegative explicit runtime ordinal")
    child = dict(environment)
    for key in VISIBILITY_KEYS:
        child.pop(key, None)
    child["HIP_VISIBLE_DEVICES"] = str(device)
    return child


def build_environment(environment: Mapping[str, str]) -> dict[str, str]:
    """Forward required local process/build values, never cloud/auth variables."""
    return {key: value for key in (*BUILD_ENV_KEYS, *ALLOWLIST)
            if (value := environment.get(key)) is not None and safe_value(value) == value}
