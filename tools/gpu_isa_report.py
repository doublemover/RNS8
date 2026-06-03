#!/usr/bin/env python3
"""Emit temp-only ISA summaries for compiled RNS8 HIP objects."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import re
import sys
from pathlib import Path
from typing import Any

from isa_common import (
    IsaToolConfig,
    device_function_symbols,
    disassemble_code_object,
    extracted_device_code_object,
    mnemonic_lines,
    sibling_tool,
)


BACKEND_OBJECT_MARKERS = {
    "direct-hip": ["hip_direct_kernels"],
    "hipblaslt": ["hipblaslt_kernels"],
    "ck": ["ck_backend_kernels"],
    "wmma": ["wmma_backend_kernels"],
    "wrap64": ["wrap64_hip_kernels"],
    "vector-alu": ["vector_alu_kernels", "hip_vector_alu_baseline_kernels"],
}
BACKEND_SYMBOL_MARKERS = {
    "direct-hip": ["rns8_ring_gemm_i8_i32", "rns8_export", "finite", "exact_wide"],
    "hipblaslt": ["rns8_hipblaslt", "pack_transpose", "reduce_i32_to_centered"],
    "ck": ["kernel_gemm_wmma", "ck_"],
    "wmma": ["rocwmma_i8_residue_gemm", "rocwmma_wrap64_byte_gemm36_candidate"],
    "wrap64": ["rns8_wrap64"],
    "vector-alu": ["gemm_i64_kernel", "gemm_u64_kernel", "rns8_vector_alu"],
}
GLOBAL_STORE_RE = re.compile(r"\b(?:global|buffer)_store\b|\b(?:global|buffer)_store_[a-z0-9_]+\b")
VGPR_RE = re.compile(r"(?:VGPRs?|\.amdhsa_next_free_vgpr)\D+(\d+)", re.IGNORECASE)
SGPR_RE = re.compile(r"(?:SGPRs?|\.amdhsa_next_free_sgpr)\D+(\d+)", re.IGNORECASE)
OCCUPANCY_RE = re.compile(r"occupancy\D+(\d+)", re.IGNORECASE)


def discover_objects(build_tree: Path, backend: str) -> list[Path]:
    markers = BACKEND_OBJECT_MARKERS.get(backend, [])
    candidates = sorted(
        path
        for path in build_tree.rglob("*")
        if path.is_file() and path.suffix.lower() in {".obj", ".o"} and (not markers or any(marker in path.name for marker in markers))
    )
    return candidates


def backend_for_object(path: Path, requested_backend: str) -> str:
    if requested_backend != "all":
        return requested_backend
    name = path.name.lower()
    for backend, markers in BACKEND_OBJECT_MARKERS.items():
        if any(marker in name for marker in markers):
            return backend
    return "all"


def selected_symbols(symbols: list[str], backend: str) -> tuple[list[str], str | None]:
    markers = BACKEND_SYMBOL_MARKERS.get(backend, [])
    if not markers:
        return symbols, None
    selected = [symbol for symbol in symbols if any(marker in symbol for marker in markers)]
    if selected:
        return selected, None
    return symbols, f"no {backend} symbol markers matched; reported all device symbols"


def integer_matches(pattern: re.Pattern[str], text: str) -> list[int]:
    values = []
    for match in pattern.finditer(text):
        try:
            values.append(int(match.group(1)))
        except (IndexError, ValueError):
            continue
    return values


def scan_disassembly(disassembly: str) -> dict[str, Any]:
    mnemonics = mnemonic_lines(disassembly)
    counts = {
        "wmma": 0,
        "mfma": 0,
        "global_store": 0,
        "lds_mentions": 0,
        "wait_instructions": 0,
        "instruction_lines": len(mnemonics),
    }
    for line, mnemonic in mnemonics:
        lowered_line = line.lower()
        lowered_mnemonic = mnemonic.lower()
        if lowered_mnemonic.startswith("v_wmma"):
            counts["wmma"] += 1
        if lowered_mnemonic.startswith("v_mfma"):
            counts["mfma"] += 1
        if GLOBAL_STORE_RE.search(lowered_mnemonic):
            counts["global_store"] += 1
        if lowered_mnemonic.startswith("ds_") or "lds" in lowered_line:
            counts["lds_mentions"] += 1
        if lowered_mnemonic.startswith("s_wait") or lowered_mnemonic in {"s_barrier", "s_sleep"}:
            counts["wait_instructions"] += 1
    vgprs = integer_matches(VGPR_RE, disassembly)
    sgprs = integer_matches(SGPR_RE, disassembly)
    occupancy = integer_matches(OCCUPANCY_RE, disassembly)
    return {
        **counts,
        "vgpr_count": max(vgprs) if vgprs else None,
        "sgpr_count": max(sgprs) if sgprs else None,
        "occupancy": max(occupancy) if occupancy else None,
    }


def sum_symbol_reports(symbol_reports: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ["wmma", "mfma", "global_store", "lds_mentions", "wait_instructions", "instruction_lines"]
    totals = {key: sum(int(report["counts"].get(key) or 0) for report in symbol_reports) for key in keys}
    for key in ["vgpr_count", "sgpr_count", "occupancy"]:
        values = [report["counts"].get(key) for report in symbol_reports if report["counts"].get(key) is not None]
        totals[key] = max(values) if values else None
    return totals


def report_object(config: IsaToolConfig, backend: str, rga_path: Path | None) -> dict[str, Any]:
    with extracted_device_code_object(config, "rns8-gpu-isa-", f"{config.host_object.stem}.fatbin") as code_object:
        symbols = device_function_symbols(config.objdump, code_object)
        symbols_to_report, note = selected_symbols(symbols, backend)
        symbol_reports = []
        for symbol in symbols_to_report:
            disassembly = disassemble_code_object(config.objdump, code_object, config.target, symbol)
            symbol_reports.append({"symbol": symbol, "counts": scan_disassembly(disassembly)})
        return {
            "object": str(config.host_object),
            "target": config.target,
            "backend": backend,
            "tools": {
                "llvm_objcopy": config.objcopy,
                "llvm_objdump": config.objdump,
                "clang_offload_bundler": config.bundler,
                "rga": str(rga_path) if rga_path is not None else None,
                "rga_status": "not_run_optional",
            },
            "code_object_note": note,
            "device_symbol_count": len(symbols),
            "reported_symbol_count": len(symbol_reports),
            "symbols": symbol_reports,
            "instruction_totals": sum_symbol_reports(symbol_reports),
        }


def write_report(report: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_backend = str(report.get("backend") or "unknown").replace("/", "_")
    stem = Path(str(report["object"])).stem
    target = str(report.get("target") or "unknown")
    out_path = out_dir / f"{stem}-{target}-{safe_backend}-isa-summary.json"
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def json_safe_config(config: IsaToolConfig) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object", dest="objects", action="append", type=Path, help="compiled HIP host object")
    parser.add_argument("--build-tree", type=Path, help="discover HIP objects under a CMake build tree")
    parser.add_argument(
        "--backend",
        default="all",
        choices=["all", "direct-hip", "hipblaslt", "ck", "wmma", "wrap64", "vector-alu"],
    )
    parser.add_argument("--target", required=True, help="AMDGPU target id, for example gfx1100")
    parser.add_argument("--hipcc", type=Path, help="HIP compiler path; sibling LLVM tools are preferred")
    parser.add_argument("--llvm-objcopy", help="Override llvm-objcopy path")
    parser.add_argument("--llvm-objdump", help="Override llvm-objdump path")
    parser.add_argument("--clang-offload-bundler", help="Override clang-offload-bundler path")
    parser.add_argument("--rga", type=Path, help="Optional RGA CLI path recorded in the report; not run by default")
    parser.add_argument("--scratch-root", type=Path, default=Path("temp"), help="ignored scratch directory")
    parser.add_argument("--out-dir", type=Path, default=Path("temp") / "isa-reports", help="temp-only output directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    objects = list(args.objects or [])
    if args.build_tree:
        if not args.build_tree.exists():
            raise RuntimeError(f"build tree does not exist: {args.build_tree}")
        objects.extend(discover_objects(args.build_tree, args.backend))
    unique_objects = []
    seen = set()
    for obj in objects:
        key = str(obj.resolve()) if obj.exists() else str(obj)
        if key not in seen:
            unique_objects.append(obj)
            seen.add(key)
    if not unique_objects:
        raise RuntimeError("no HIP objects supplied or discovered")

    hipcc = args.hipcc
    config_template = {
        "target": args.target,
        "objcopy": args.llvm_objcopy or sibling_tool(hipcc, "llvm-objcopy"),
        "objdump": args.llvm_objdump or sibling_tool(hipcc, "llvm-objdump"),
        "bundler": args.clang_offload_bundler or sibling_tool(hipcc, "clang-offload-bundler"),
        "scratch_root": args.scratch_root,
    }

    outputs = []
    for obj in unique_objects:
        if not obj.exists():
            raise RuntimeError(f"HIP object does not exist: {obj}")
        backend = backend_for_object(obj, args.backend)
        config = IsaToolConfig(host_object=obj, **config_template)
        report = report_object(config, backend, args.rga)
        report["config"] = json_safe_config(config)
        outputs.append(write_report(report, args.out_dir))

    print("GPU ISA report: PASS")
    for out_path in outputs:
        print(f"- {out_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - command-line tool should print concise failure cause.
        print(f"GPU ISA report: FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
