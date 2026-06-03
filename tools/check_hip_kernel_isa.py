#!/usr/bin/env python3
"""Validate selected direct-HIP kernel ISA properties from a compiled object."""

from __future__ import annotations

import sys
from pathlib import Path

from isa_common import (
    FORBIDDEN_DIVIDE_RE,
    disassemble_code_object,
    extracted_device_code_object,
    forbidden_mnemonic_lines,
    parse_isa_tool_config,
    symbols_matching_markers,
)


GEMM_SYMBOL_MARKERS = [
    "rns8_ring_gemm_i8_i32_tiled_kernel",
    "rns8_ring_gemm_i8_i32_scheduled_kernel",
]

REQUIRED_RECIPROCAL_MNEMONIC = "v_mul_hi_u32"


def scan_disassembly(objdump: str, code_object: Path, amdgpu_target: str, symbols: list[str]) -> list[str]:
    reports: list[str] = []
    for symbol in symbols:
        disassembly = disassemble_code_object(objdump, code_object, amdgpu_target, symbol)
        forbidden = forbidden_mnemonic_lines(disassembly, FORBIDDEN_DIVIDE_RE)
        if forbidden:
            raise RuntimeError(
                f"{symbol} contains forbidden divide/remainder/rcp instructions:\n" + "\n".join(forbidden)
            )
        if REQUIRED_RECIPROCAL_MNEMONIC not in disassembly:
            raise RuntimeError(f"{symbol} does not contain required {REQUIRED_RECIPROCAL_MNEMONIC} instruction")
        reports.append(f"{symbol}: no div/rem/rcp mnemonics; contains {REQUIRED_RECIPROCAL_MNEMONIC}")
    return reports


def main() -> int:
    config = parse_isa_tool_config(__doc__, "Compiled HIP host object containing .hip_fatbin", "HIP object")
    with extracted_device_code_object(config, "rns8-hip-isa-", "hip_direct_kernels.fatbin") as code_object:
        symbols = symbols_matching_markers(config.objdump, code_object, GEMM_SYMBOL_MARKERS, "GEMM kernel symbols")
        reports = scan_disassembly(config.objdump, code_object, config.target, symbols)

    print("HIP ISA check: PASS")
    print(f"object: {config.host_object}")
    print(f"target: {config.target}")
    for report in reports:
        print(f"- {report}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - command-line tool should print concise failure cause.
        print(f"HIP ISA check: FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
