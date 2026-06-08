#!/usr/bin/env python3
"""Summarize AMD matrix-instruction calculator evidence for RNS8 targets."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


REPORT_POLICY = "matrix_instruction_calculator_evidence_only_not_a_correctness_or_performance_claim"
DEFAULT_CALCULATOR = Path("temp") / "amd_matrix_instruction_calculator" / "matrix_calculator.py"
DEFAULT_ARCHITECTURES = ("gfx942", "gfx1100")
DETAIL_INT_RE = re.compile(r"^\s*(?P<key>M|N|K|blocks|Ops|Execution cycles|VALU co-execution cycles possible):\s*(?P<value>-?\d+)")
DETAIL_THROUGHPUT_RE = re.compile(r"^\s*Ops/(?P<scope>CU|WGP)/cycle:\s*(?P<value>\d+)")
DETAIL_BOOL_RE = re.compile(r"^\s*(?P<key>Can co-execute with VALU|Sparse A matrix):\s*(?P<value>True|False)")
DETAIL_GPR_RE = re.compile(r"^\s*GPRs required for (?P<role>A|B|C|D):\s*(?P<value>\d+)")
DETAIL_ARCH_RE = re.compile(r"^\s*Architecture:\s*(?P<value>.+?)\s*$")
DETAIL_INSTRUCTION_RE = re.compile(r"^\s*Instruction:\s*(?P<value>.+?)\s*$")


@dataclass(frozen=True)
class MatrixInstruction:
    architecture_query: str
    architecture_reported: str | None
    instruction: str
    m: int | None
    n: int | None
    k: int | None
    blocks: int | None
    ops: int | None
    execution_cycles: int | None
    ops_per_cycle: int | None
    ops_per_cycle_metric: str | None
    ops_per_cu_cycle: int | None
    ops_per_wgp_cycle: int | None
    can_coexecute_with_valu: bool | None
    valu_coexecution_cycles_possible: int | None
    gprs_a: int | None
    gprs_b: int | None
    gprs_c: int | None
    gprs_d: int | None
    sparse_a_matrix: bool | None
    category: str


def run_calculator(calculator: Path, args: list[str]) -> str:
    command = [sys.executable, str(calculator), *args]
    completed = subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return completed.stdout


def parse_instruction_list(text: str) -> list[str]:
    instructions: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("v_"):
            instructions.append(stripped)
    return instructions


def instruction_category(instruction: str) -> str:
    lowered = instruction.lower()
    if lowered.startswith("v_smfmac") or lowered.startswith("v_swmmac"):
        if "_i32_" in lowered and lowered.endswith("_i8"):
            return "sparse_i8_i32_matrix_core"
        return "sparse_other_matrix_core"
    if "_i32_" in lowered and (lowered.endswith("_i8") or lowered.endswith("_iu8")):
        return "dense_i8_i32_matrix_core"
    return "other_matrix_core"


def parse_detail(architecture: str, requested_instruction: str, text: str) -> MatrixInstruction:
    values: dict[str, Any] = {
        "architecture_query": architecture,
        "architecture_reported": None,
        "instruction": requested_instruction,
        "m": None,
        "n": None,
        "k": None,
        "blocks": None,
        "ops": None,
        "execution_cycles": None,
        "ops_per_cycle": None,
        "ops_per_cycle_metric": None,
        "ops_per_cu_cycle": None,
        "ops_per_wgp_cycle": None,
        "can_coexecute_with_valu": None,
        "valu_coexecution_cycles_possible": None,
        "gprs_a": None,
        "gprs_b": None,
        "gprs_c": None,
        "gprs_d": None,
        "sparse_a_matrix": None,
        "category": instruction_category(requested_instruction),
    }
    int_map = {
        "M": "m",
        "N": "n",
        "K": "k",
        "blocks": "blocks",
        "Ops": "ops",
        "Execution cycles": "execution_cycles",
        "VALU co-execution cycles possible": "valu_coexecution_cycles_possible",
    }
    bool_map = {
        "Can co-execute with VALU": "can_coexecute_with_valu",
        "Sparse A matrix": "sparse_a_matrix",
    }
    for line in text.splitlines():
        arch_match = DETAIL_ARCH_RE.match(line)
        if arch_match:
            values["architecture_reported"] = arch_match.group("value")
            continue
        instruction_match = DETAIL_INSTRUCTION_RE.match(line)
        if instruction_match:
            values["instruction"] = instruction_match.group("value").lower()
            values["category"] = instruction_category(str(values["instruction"]))
            continue
        int_match = DETAIL_INT_RE.match(line)
        if int_match:
            values[int_map[int_match.group("key")]] = int(int_match.group("value"))
            continue
        throughput_match = DETAIL_THROUGHPUT_RE.match(line)
        if throughput_match:
            value = int(throughput_match.group("value"))
            scope = throughput_match.group("scope").lower()
            values["ops_per_cycle"] = value
            values["ops_per_cycle_metric"] = f"ops_per_{scope}_cycle"
            values[f"ops_per_{scope}_cycle"] = value
            continue
        bool_match = DETAIL_BOOL_RE.match(line)
        if bool_match:
            values[bool_map[bool_match.group("key")]] = bool_match.group("value") == "True"
            continue
        gpr_match = DETAIL_GPR_RE.match(line)
        if gpr_match:
            values[f"gprs_{gpr_match.group('role').lower()}"] = int(gpr_match.group("value"))
    return MatrixInstruction(**values)


def _sort_score(instruction: MatrixInstruction) -> tuple[int, int, int, str]:
    throughput = instruction.ops_per_cycle or 0
    accumulator_regs = instruction.gprs_d if instruction.gprs_d is not None else instruction.gprs_c
    if accumulator_regs is None:
        accumulator_regs = 999
    input_regs = (instruction.gprs_a or 0) + (instruction.gprs_b or 0)
    return (-throughput, accumulator_regs, input_regs, instruction.instruction)


def _dense_recommendations(instructions: list[MatrixInstruction]) -> list[dict[str, Any]]:
    dense = [item for item in instructions if item.category == "dense_i8_i32_matrix_core"]
    ranked = sorted(dense, key=_sort_score)
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(ranked):
        reason = "highest_dense_i8_throughput_lowest_accumulator_register_pressure"
        if index > 0 and ranked and item.ops_per_cu_cycle == ranked[0].ops_per_cu_cycle:
            reason = "same_dense_i8_throughput_higher_accumulator_register_pressure"
        rows.append(
            {
                "instruction": item.instruction,
                "rank": index + 1,
                "reason": reason,
                "ops_per_cycle": item.ops_per_cycle,
                "ops_per_cycle_metric": item.ops_per_cycle_metric,
                "ops_per_cu_cycle": item.ops_per_cu_cycle,
                "ops_per_wgp_cycle": item.ops_per_wgp_cycle,
                "tile": {"m": item.m, "n": item.n, "k": item.k},
                "registers": {"a": item.gprs_a, "b": item.gprs_b, "c": item.gprs_c, "d": item.gprs_d},
            }
        )
    return rows


def _sparse_notes(instructions: list[MatrixInstruction]) -> list[dict[str, Any]]:
    sparse = [item for item in instructions if item.category == "sparse_i8_i32_matrix_core"]
    return [
        {
            "instruction": item.instruction,
            "ops_per_cycle": item.ops_per_cycle,
            "ops_per_cycle_metric": item.ops_per_cycle_metric,
            "ops_per_cu_cycle": item.ops_per_cu_cycle,
            "ops_per_wgp_cycle": item.ops_per_wgp_cycle,
            "tile": {"m": item.m, "n": item.n, "k": item.k},
            "eligibility": "future_sparse_only_requires_explicit_4_to_2_A_matrix_compression_contract",
        }
        for item in sorted(sparse, key=_sort_score)
    ]


def summarize_architecture(calculator: Path, architecture: str) -> dict[str, Any]:
    list_output = run_calculator(calculator, ["--architecture", architecture, "--list-instructions"])
    listed = parse_instruction_list(list_output)
    selected = [
        instruction
        for instruction in listed
        if instruction_category(instruction) in {"dense_i8_i32_matrix_core", "sparse_i8_i32_matrix_core"}
    ]
    details = [
        parse_detail(
            architecture,
            instruction,
            run_calculator(calculator, ["--architecture", architecture, "--instruction", instruction, "--detail-instruction"]),
        )
        for instruction in selected
    ]
    dense = [item for item in details if item.category == "dense_i8_i32_matrix_core"]
    sparse = [item for item in details if item.category == "sparse_i8_i32_matrix_core"]
    return {
        "architecture_query": architecture,
        "listed_instruction_count": len(listed),
        "dense_i8_i32_instruction_count": len(dense),
        "sparse_i8_i32_instruction_count": len(sparse),
        "dense_i8_i32_instructions": [asdict(item) for item in sorted(dense, key=_sort_score)],
        "sparse_i8_i32_instructions": [asdict(item) for item in sorted(sparse, key=_sort_score)],
        "dense_i8_i32_recommendations": _dense_recommendations(details),
        "sparse_i8_i32_notes": _sparse_notes(details),
    }


def build_report(calculator: Path, architectures: list[str]) -> dict[str, Any]:
    version = None
    try:
        version = run_calculator(calculator, ["--version"]).strip() or None
    except (OSError, subprocess.CalledProcessError):
        version = None
    summaries = [summarize_architecture(calculator, architecture) for architecture in architectures]
    return {
        "schema_version": 1,
        "policy": REPORT_POLICY,
        "calculator": str(calculator),
        "calculator_version": version,
        "architectures": summaries,
        "rns8_implications": [
            "Use this report to choose matrix-core candidates and register-layout investigations; still verify compiled objects with gpu_isa_report and exact CPU comparisons.",
            "Dense RNS8 int8xint8-to-int32 GEMM on CDNA should prioritize dense MFMA i8 instructions before sparse SMFMAC unless inputs carry an explicit sparse compression contract.",
            "Sparse SMFMAC is not a drop-in correctness-preserving path for current dense bounded/exact/finite RNS8 contracts.",
        ],
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# AMD Matrix Instruction Report",
        "",
        f"Policy: `{report['policy']}`",
        "",
    ]
    for arch in report["architectures"]:
        lines.extend(
            [
                f"## {arch['architecture_query']}",
                "",
                "| rank | instruction | tile | ops/cycle | A/B/C/D GPRs | reason |",
                "| --- | --- | --- | ---: | --- | --- |",
            ]
        )
        for row in arch["dense_i8_i32_recommendations"]:
            tile = row["tile"]
            regs = row["registers"]
            lines.append(
                f"| {row['rank']} | `{row['instruction']}` | {tile['m']}x{tile['n']}x{tile['k']} | "
                f"{row['ops_per_cycle']} {row['ops_per_cycle_metric']} | "
                f"{regs['a']}/{regs['b']}/{regs['c']}/{regs['d']} | {row['reason']} |"
            )
        if arch["sparse_i8_i32_notes"]:
            lines.extend(["", "Sparse instructions are future-only unless the input contract includes A-side compression metadata:", ""])
            for row in arch["sparse_i8_i32_notes"]:
                tile = row["tile"]
                lines.append(
                    f"- `{row['instruction']}`: {tile['m']}x{tile['n']}x{tile['k']}, "
                    f"{row['ops_per_cycle']} {row['ops_per_cycle_metric']}, {row['eligibility']}"
                )
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calculator", type=Path, default=DEFAULT_CALCULATOR)
    parser.add_argument(
        "--architectures",
        default=",".join(DEFAULT_ARCHITECTURES),
        help="comma-separated architecture aliases, for example gfx942,gfx1100",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("temp") / "amd-matrix-instruction-reports")
    parser.add_argument("--markdown", action="store_true", help="also write a Markdown summary")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.calculator.exists():
        raise RuntimeError(f"AMD matrix instruction calculator not found: {args.calculator}")
    architectures = [item.strip() for item in args.architectures.split(",") if item.strip()]
    if not architectures:
        raise RuntimeError("at least one architecture is required")
    report = build_report(args.calculator, architectures)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "amd-matrix-instruction-report.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("AMD matrix instruction report: PASS")
    print(f"- {json_path}")
    if args.markdown:
        markdown_path = args.out_dir / "amd-matrix-instruction-report.md"
        write_markdown(report, markdown_path)
        print(f"- {markdown_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - command-line tool should print concise failure cause.
        print(f"AMD matrix instruction report: FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
