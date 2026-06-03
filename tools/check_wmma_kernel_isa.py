#!/usr/bin/env python3
"""Validate rocWMMA accelerator kernel ISA evidence from a compiled HIP object."""

from __future__ import annotations

import sys
from pathlib import Path

from isa_common import (
    FORBIDDEN_DIVIDE_RE,
    FORBIDDEN_INT32_GLOBAL_STORE_RE,
    disassemble_code_object,
    extracted_device_code_object,
    forbidden_mnemonic_lines,
    parse_isa_tool_config,
    symbol_count_matching_marker,
)


WMMA_SYMBOL_MARKER = "rocwmma_i8_residue_gemm"
REQUIRED_WMMA_MNEMONIC = "v_wmma_i32_16x16x16_iu8"


def wmma_symbol_count(objdump: str, code_object: Path) -> int:
    return symbol_count_matching_marker(objdump, code_object, WMMA_SYMBOL_MARKER)


def scan_disassembly(objdump: str, code_object: Path, amdgpu_target: str) -> tuple[int, list[str], list[str]]:
    disassembly = disassemble_code_object(objdump, code_object, amdgpu_target)
    wmma_count = disassembly.count(REQUIRED_WMMA_MNEMONIC)
    forbidden_stores = forbidden_mnemonic_lines(disassembly, FORBIDDEN_INT32_GLOBAL_STORE_RE)
    forbidden_divides = forbidden_mnemonic_lines(disassembly, FORBIDDEN_DIVIDE_RE)
    return wmma_count, forbidden_stores, forbidden_divides


def main() -> int:
    config = parse_isa_tool_config(
        __doc__,
        "Compiled rocWMMA HIP host object containing .hip_fatbin",
        "rocWMMA HIP object",
    )
    with extracted_device_code_object(config, "rns8-wmma-isa-", "wmma_backend_kernels.fatbin") as code_object:
        symbol_count = wmma_symbol_count(config.objdump, code_object)
        if symbol_count <= 0:
            raise RuntimeError(f"missing rocWMMA {WMMA_SYMBOL_MARKER} device kernel symbols")
        wmma_count, forbidden_stores, forbidden_divides = scan_disassembly(config.objdump, code_object, config.target)
        if wmma_count <= 0:
            raise RuntimeError(f"rocWMMA object does not contain required {REQUIRED_WMMA_MNEMONIC} matrix instructions")
        if forbidden_stores:
            raise RuntimeError(
                "rocWMMA object contains forbidden INT32 global/buffer stores:\n" + "\n".join(forbidden_stores[:20])
            )
        if forbidden_divides:
            raise RuntimeError(
                "rocWMMA object contains forbidden reciprocal/divide/remainder instructions:\n"
                + "\n".join(forbidden_divides[:20])
            )

    print("rocWMMA ISA check: PASS")
    print(f"object: {config.host_object}")
    print(f"target: {config.target}")
    print(f"- rocWMMA fused kernel symbols: {symbol_count}")
    print(f"- {REQUIRED_WMMA_MNEMONIC} instructions: {wmma_count}")
    print("- no global_store_dword/buffer_store_dword instructions")
    print("- no v/s reciprocal, divide, or remainder instructions")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - command-line tool should print concise failure cause.
        print(f"rocWMMA ISA check: FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
