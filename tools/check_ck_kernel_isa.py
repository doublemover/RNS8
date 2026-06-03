#!/usr/bin/env python3
"""Validate CK accelerator kernel ISA evidence from a compiled HIP object."""

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
    symbols_matching_markers,
)


CK_SYMBOL_MARKER = "kernel_gemm_wmma"
REQUIRED_WMMA_MNEMONIC = "v_wmma_i32_16x16x16_iu8"


def ck_symbols(objdump: str, code_object: Path) -> list[str]:
    return symbols_matching_markers(objdump, code_object, [CK_SYMBOL_MARKER], "CK GEMM WMMA symbols")


def scan_disassembly(
    objdump: str,
    code_object: Path,
    amdgpu_target: str,
    symbols: list[str],
) -> tuple[int, list[str], list[str]]:
    wmma_count = 0
    forbidden_stores: list[str] = []
    forbidden_divides: list[str] = []
    for symbol in symbols:
        disassembly = disassemble_code_object(objdump, code_object, amdgpu_target, symbol)
        wmma_count += disassembly.count(REQUIRED_WMMA_MNEMONIC)
        forbidden_stores.extend(
            f"{symbol}: {line}" for line in forbidden_mnemonic_lines(disassembly, FORBIDDEN_INT32_GLOBAL_STORE_RE)
        )
        forbidden_divides.extend(
            f"{symbol}: {line}" for line in forbidden_mnemonic_lines(disassembly, FORBIDDEN_DIVIDE_RE)
        )
    return wmma_count, forbidden_stores, forbidden_divides


def main() -> int:
    config = parse_isa_tool_config(__doc__, "Compiled CK HIP host object containing .hip_fatbin", "CK HIP object")
    with extracted_device_code_object(config, "rns8-ck-isa-", "ck_backend_kernels.fatbin") as code_object:
        symbols = ck_symbols(config.objdump, code_object)
        wmma_count, forbidden_stores, forbidden_divides = scan_disassembly(
            config.objdump,
            code_object,
            config.target,
            symbols,
        )
        if wmma_count <= 0:
            raise RuntimeError(f"CK object does not contain required {REQUIRED_WMMA_MNEMONIC} matrix instructions")
        if forbidden_stores:
            raise RuntimeError(
                "CK object contains forbidden INT32 global/buffer stores:\n" + "\n".join(forbidden_stores[:20])
            )
        if forbidden_divides:
            raise RuntimeError(
                "CK object contains forbidden reciprocal/divide/remainder instructions:\n"
                + "\n".join(forbidden_divides[:20])
            )

    print("CK ISA check: PASS")
    print(f"object: {config.host_object}")
    print(f"target: {config.target}")
    print(f"- CK GEMM WMMA symbols: {len(symbols)}")
    print(f"- {REQUIRED_WMMA_MNEMONIC} instructions: {wmma_count}")
    print("- no global_store_dword/buffer_store_dword instructions")
    print("- no v/s reciprocal, divide, or remainder instructions")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - command-line tool should print concise failure cause.
        print(f"CK ISA check: FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
