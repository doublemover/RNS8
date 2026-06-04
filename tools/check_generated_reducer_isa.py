#!/usr/bin/env python3
"""Validate generated Direct-HIP prefix pack/reducer ISA from a compiled object."""

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


GENERATED_SYMBOL_MARKERS = [
    "rns8_pack_i64_fixed_prefix_kernel",
    "rns8_pack_u64_fixed_prefix_kernel",
    "rns8_ring_gemm_i8_i32_scheduled_kernel",
    "rns8_ring_gemm_native_i64_i32_grouped_prefix9_kernel",
]
NO_DIVIDE_SYMBOL_MARKERS = [
    "rns8_ring_gemm_i8_i32_scheduled_kernel",
]


def scan_generated_symbols(objdump: str, code_object: Path, target: str, symbols: list[str]) -> list[str]:
    reports = []
    for symbol in symbols:
        if not any(marker in symbol for marker in NO_DIVIDE_SYMBOL_MARKERS):
            reports.append(f"{symbol}: expected generated prefix symbol is present")
            continue
        disassembly = disassemble_code_object(objdump, code_object, target, symbol)
        forbidden = forbidden_mnemonic_lines(disassembly, FORBIDDEN_DIVIDE_RE)
        if forbidden:
            raise RuntimeError(
                f"{symbol} contains forbidden divide/remainder/rcp instructions:\n" + "\n".join(forbidden)
            )
        reports.append(f"{symbol}: generated prefix path has no div/rem/rcp mnemonics")
    return reports


def main() -> int:
    config = parse_isa_tool_config(
        __doc__,
        "Compiled Direct-HIP host object containing .hip_fatbin",
        "Direct-HIP object",
    )
    with extracted_device_code_object(config, "rns8-generated-reducer-isa-", "hip_direct_kernels.fatbin") as code_object:
        symbols = symbols_matching_markers(
            config.objdump,
            code_object,
            GENERATED_SYMBOL_MARKERS,
            "generated prefix pack/reducer symbols",
        )
        reports = scan_generated_symbols(config.objdump, code_object, config.target, symbols)

    print("Generated reducer ISA check: PASS")
    print(f"object: {config.host_object}")
    print(f"target: {config.target}")
    for report in reports:
        print(f"- {report}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - command-line tool should print concise failure cause.
        print(f"Generated reducer ISA check: FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
