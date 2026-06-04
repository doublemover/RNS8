#!/usr/bin/env python3
"""Validate strict wrap64 direct-HIP byte-GEMM36 kernel ISA evidence."""

from __future__ import annotations

import sys
from pathlib import Path

from isa_common import (
    FORBIDDEN_DIVIDE_RE,
    FORBIDDEN_MATRIX_ENGINE_RE,
    disassemble_code_object,
    extracted_device_code_object,
    forbidden_mnemonic_lines,
    parse_isa_tool_config,
    symbols_matching_markers,
)


WRAP64_SYMBOL_MARKERS = [
    "rns8_wrap64_pack_u64_kernel",
    "rns8_wrap64_pack_u64_scalar_kernel",
    "rns8_wrap64_byte_gemm36_tiled_u32acc_kernel",
    "rns8_wrap64_byte_gemm36_tiled_u64acc_kernel",
    "rns8_wrap64_byte_gemm36_colpair_u32acc_kernel",
    "rns8_wrap64_export_u64_kernel",
    "rns8_wrap64_export_u64_scalar_kernel",
]

def wrap64_symbols(objdump: str, code_object: Path) -> list[str]:
    return symbols_matching_markers(objdump, code_object, WRAP64_SYMBOL_MARKERS, "wrap64 kernel symbols")


def scan_disassembly(objdump: str, code_object: Path, amdgpu_target: str, symbols: list[str]) -> list[str]:
    reports: list[str] = []
    for symbol in symbols:
        disassembly = disassemble_code_object(objdump, code_object, amdgpu_target, symbol)
        forbidden_divides = forbidden_mnemonic_lines(disassembly, FORBIDDEN_DIVIDE_RE)
        forbidden_matrix = forbidden_mnemonic_lines(disassembly, FORBIDDEN_MATRIX_ENGINE_RE)
        if forbidden_divides:
            raise RuntimeError(
                f"{symbol} contains forbidden divide/remainder/rcp instructions:\n"
                + "\n".join(forbidden_divides[:20])
            )
        if forbidden_matrix:
            raise RuntimeError(
                f"{symbol} unexpectedly contains matrix-engine instructions:\n" + "\n".join(forbidden_matrix[:20])
            )
        reports.append(f"{symbol}: no div/rem/rcp mnemonics; no matrix-engine mnemonics")
    return reports


def main() -> int:
    config = parse_isa_tool_config(
        __doc__,
        "Compiled wrap64 HIP host object containing .hip_fatbin",
        "wrap64 HIP object",
    )
    with extracted_device_code_object(config, "rns8-wrap64-isa-", "wrap64_hip_kernels.fatbin") as code_object:
        symbols = wrap64_symbols(config.objdump, code_object)
        reports = scan_disassembly(config.objdump, code_object, config.target, symbols)

    print("wrap64 ISA check: PASS")
    print(f"object: {config.host_object}")
    print(f"target: {config.target}")
    for report in reports:
        print(f"- {report}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - command-line tool should print concise failure cause.
        print(f"wrap64 ISA check: FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
