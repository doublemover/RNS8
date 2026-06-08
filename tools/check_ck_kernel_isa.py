#!/usr/bin/env python3
"""Validate CK accelerator kernel ISA evidence from a compiled HIP object."""

from __future__ import annotations

import sys
import re
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


RDNA_SYMBOL_MARKERS = ["kernel_gemm_wmma"]
CDNA_SYMBOL_MARKERS = ["kernel_gemm_xdl_cshuffle"]
RDNA_REQUIRED_MNEMONICS = ["v_wmma_i32_16x16x16_iu8"]
CDNA_REQUIRED_MNEMONICS = [
    "v_mfma_i32_16x16x32_i8",
    "v_mfma_i32_16x16x64_i8",
    "v_mfma_i32_32x32x16_i8",
    "v_mfma_i32_32x32x32_i8",
]
CDNA_CK_INTERNAL_BLOCK_MAP_RCP_RE = re.compile(r"\bv_rcp_iflag_f32(?:_[a-z0-9]+)?\b")


def required_symbol_markers(amdgpu_target: str) -> list[str]:
    if amdgpu_target.startswith("gfx9"):
        return CDNA_SYMBOL_MARKERS
    return RDNA_SYMBOL_MARKERS


def ck_symbols(objdump: str, code_object: Path, amdgpu_target: str) -> list[str]:
    return symbols_matching_markers(
        objdump,
        code_object,
        required_symbol_markers(amdgpu_target),
        "CK GEMM matrix symbols",
    )


def required_matrix_mnemonics(amdgpu_target: str) -> list[str]:
    if amdgpu_target.startswith("gfx9"):
        return CDNA_REQUIRED_MNEMONICS
    return RDNA_REQUIRED_MNEMONICS


def allowed_ck_internal_divide_line(line: str, amdgpu_target: str) -> bool:
    return amdgpu_target.startswith("gfx9") and bool(CDNA_CK_INTERNAL_BLOCK_MAP_RCP_RE.search(line))


def scan_disassembly(
    objdump: str,
    code_object: Path,
    amdgpu_target: str,
    symbols: list[str],
) -> tuple[int, list[str], list[str]]:
    required = required_matrix_mnemonics(amdgpu_target)
    matrix_count = 0
    forbidden_stores: list[str] = []
    forbidden_divides: list[str] = []
    for symbol in symbols:
        disassembly = disassemble_code_object(objdump, code_object, amdgpu_target, symbol)
        matrix_count += sum(disassembly.count(mnemonic) for mnemonic in required)
        forbidden_stores.extend(forbidden_mnemonic_lines(disassembly, FORBIDDEN_INT32_GLOBAL_STORE_RE))
        forbidden_divides.extend(
            f"{symbol}: {line}"
            for line in forbidden_mnemonic_lines(disassembly, FORBIDDEN_DIVIDE_RE)
            if not allowed_ck_internal_divide_line(line, amdgpu_target)
        )
    return matrix_count, forbidden_stores, forbidden_divides


def main() -> int:
    config = parse_isa_tool_config(__doc__, "Compiled CK HIP host object containing .hip_fatbin", "CK HIP object")
    with extracted_device_code_object(config, "rns8-ck-isa-", "ck_backend_kernels.fatbin") as code_object:
        symbols = ck_symbols(config.objdump, code_object, config.target)
        matrix_count, forbidden_stores, forbidden_divides = scan_disassembly(
            config.objdump,
            code_object,
            config.target,
            symbols,
        )
        required = required_matrix_mnemonics(config.target)
        if matrix_count <= 0:
            raise RuntimeError(f"CK object does not contain required matrix instructions: {', '.join(required)}")
        if forbidden_divides:
            raise RuntimeError(
                "CK object contains forbidden reciprocal/divide/remainder instructions:\n"
                + "\n".join(forbidden_divides[:20])
            )

    print("CK ISA check: PASS")
    print(f"object: {config.host_object}")
    print(f"target: {config.target}")
    print(f"- CK GEMM matrix symbols: {len(symbols)}")
    print(f"- target matrix instructions: {matrix_count}")
    print(f"- global_store_dword/buffer_store_dword instructions: {len(forbidden_stores)}")
    if config.target.startswith("gfx9"):
        print("- no data-path divide or remainder instructions; CK XDL block-map reciprocal is allowed")
    else:
        print("- no v/s reciprocal, divide, or remainder instructions")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - command-line tool should print concise failure cause.
        print(f"CK ISA check: FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
