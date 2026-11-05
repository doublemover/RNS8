"""Compiler-affecting environment shared by native build/analysis receipts."""

from __future__ import annotations

import os
from collections.abc import Mapping

COMPILER_ENVIRONMENT = frozenset(
    (
        "PATH",
        "INCLUDE",
        "LIB",
        "LIBPATH",
        "VCINSTALLDIR",
        "VCTOOLSINSTALLDIR",
        "WINDOWSSDKDIR",
        "WINDOWSSDKVERSION",
        "CPATH",
        "CPLUS_INCLUDE_PATH",
        "C_INCLUDE_PATH",
        "LIBRARY_PATH",
        "SDKROOT",
        "HIP_PATH",
        "ROCM_PATH",
        "HIP_CLANG_PATH",
        "HIP_DEVICE_LIB_PATH",
        "HIP_PLATFORM",
        "LLVM_PATH",
        "CL",
        "_CL_",
        "CCC_OVERRIDE_OPTIONS",
        "CLANG_CONFIG_FILE_USER_DIR",
        "CLANG_CONFIG_FILE_SYSTEM_DIR",
        "GCC_EXEC_PREFIX",
        "COMPILER_PATH",
        "HIPCC_COMPILE_FLAGS_APPEND",
        "HIPCC_LINK_FLAGS_APPEND",
        "CC",
        "CXX",
        "CFLAGS",
        "CXXFLAGS",
        "CPPFLAGS",
        "LDFLAGS",
        "TEMP",
        "TMP",
        "TMPDIR",
        "SYSTEMROOT",
        "WINDIR",
        "SYSTEMDRIVE",
    )
)


def compiler_environment(environment: Mapping[str, str]) -> dict[str, str]:
    if any(
        type(name) is not str or not name or type(value) is not str
        for name, value in environment.items()
    ):
        raise ValueError("compiler environment must map nonempty string names to string values")

    def relevant(name: str) -> bool:
        key = name.upper() if os.name == "nt" else name
        return key in COMPILER_ENVIRONMENT or key.startswith(
            ("HIP_", "HIPCC_", "ROCM_", "LLVM_", "CCC_", "CLANG_", "GCC_")
        )

    return {name: value for name, value in environment.items() if relevant(name)}
