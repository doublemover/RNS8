#!/usr/bin/env python3
"""Summarize AMD matrix-instruction calculator evidence for RNS8 targets."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


REPORT_POLICY = "matrix_instruction_calculator_evidence_only_not_a_correctness_or_performance_claim"
DEFAULT_CALCULATOR = Path("temp") / "amd_matrix_instruction_calculator" / "matrix_calculator.py"
DEFAULT_ARCHITECTURES = ("gfx942", "gfx1100", "gfx1200", "gfx1201")
DETAIL_INT_RE = re.compile(r"^\s*(?P<key>M|N|K|blocks|Ops|Execution cycles|VALU co-execution cycles possible):\s*(?P<value>-?\d+)")
DETAIL_THROUGHPUT_RE = re.compile(r"^\s*Ops/(?P<scope>CU|WGP)/cycle:\s*(?P<value>\d+)")
DETAIL_BOOL_RE = re.compile(
    r"^\s*(?P<key>Can co-execute with VALU|Sparse A matrix|CBSZ and ABID bits supported|BLGP bits supported|OPSEL supported|NEG bits supported):\s*(?P<value>True|False)"
)
DETAIL_GPR_RE = re.compile(r"^\s*GPRs required for (?P<role>A|B|C|D):\s*(?P<value>\d+)")
DETAIL_ARCH_RE = re.compile(r"^\s*Architecture:\s*(?P<value>.+?)\s*$")
DETAIL_INSTRUCTION_RE = re.compile(r"^\s*Instruction:\s*(?P<value>.+?)\s*$")
DETAIL_WAVE_REGISTER_USAGE_RE = re.compile(r"^\s*Wave(?P<wave>32|64) register usage:\s*$")
DETAIL_REGISTER_USAGE_RE = re.compile(r"^\s*Register usage:\s*$")
DETAIL_DATA_TYPE_SECTION_RE = re.compile(r"^\s*Register data types:\s*$")
DETAIL_DATA_TYPE_RE = re.compile(r"^\s*(?P<field>Src0|Src1|Src2|Vdst):\s*(?P<value>.+?)\s*$")
INSTRUCTION_FAMILY_RE = re.compile(r"^v_(?P<family>mfma|smfmac|wmma|swmmac)_")
INTEGER_OPERAND_SUFFIXES = ("_i8", "_iu8", "_iu4")


@dataclass
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
    instruction_family: str | None
    matrix_sparsity: str
    operand_type: str | None
    integer_operand_bits: int | None
    operand_signedness: str | None
    rns8_integer_candidate: bool
    sparse_explicit_contract_required: bool
    wavefront_register_usage: dict[str, dict[str, int]] = field(default_factory=dict)
    preferred_register_usage: dict[str, int | None] = field(default_factory=dict)
    register_data_types: dict[str, str] = field(default_factory=dict)
    modifier_support: dict[str, bool] = field(default_factory=dict)
    rdna_integer_modifier_constraints: dict[str, str] = field(default_factory=dict)
    rns8_semantic_requirements: dict[str, str] = field(default_factory=dict)
    layout_artifacts: list[dict[str, Any]] = field(default_factory=list)


def run_calculator(calculator: Path, args: list[str]) -> str:
    command = [sys.executable, str(calculator), *args]
    completed = subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return completed.stdout


def parse_instruction_list(text: str) -> list[str]:
    instructions: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("v_"):
            instructions.append(stripped.lower())
    return instructions


def _operand_type(instruction: str) -> str | None:
    lowered = instruction.lower()
    for suffix in INTEGER_OPERAND_SUFFIXES:
        if lowered.endswith(suffix):
            return suffix[1:]
    return None


def _integer_operand_bits(operand: str | None) -> int | None:
    if operand in {"i8", "iu8"}:
        return 8
    if operand == "iu4":
        return 4
    return None


def _operand_signedness(operand: str | None) -> str | None:
    if operand == "i8":
        return "signed"
    if operand in {"iu8", "iu4"}:
        return "signed_or_unsigned_selected_by_instruction_modifier"
    return None


def instruction_traits(instruction: str) -> dict[str, Any]:
    lowered = instruction.lower()
    family_match = INSTRUCTION_FAMILY_RE.match(lowered)
    family = family_match.group("family") if family_match else None
    sparsity = "sparse" if family in {"smfmac", "swmmac"} else "dense" if family in {"mfma", "wmma"} else "other"
    operand = _operand_type(lowered)
    integer_bits = _integer_operand_bits(operand)
    integer_candidate = "_i32_" in lowered and operand is not None and family in {"mfma", "smfmac", "wmma", "swmmac"}
    if integer_candidate:
        bit_label = "i8" if integer_bits == 8 else "i4"
        category = f"{sparsity}_{bit_label}_i32_matrix_core"
    elif sparsity == "sparse":
        category = "sparse_other_matrix_core"
    elif sparsity == "dense":
        category = "dense_other_matrix_core"
    else:
        category = "other_matrix_core"
    return {
        "category": category,
        "instruction_family": family,
        "matrix_sparsity": sparsity,
        "operand_type": operand,
        "integer_operand_bits": integer_bits,
        "operand_signedness": _operand_signedness(operand),
        "rns8_integer_candidate": integer_candidate,
        "sparse_explicit_contract_required": bool(integer_candidate and sparsity == "sparse"),
    }


def instruction_category(instruction: str) -> str:
    return str(instruction_traits(instruction)["category"])


def _rdna_integer_constraints(family: str | None, operand: str | None) -> dict[str, str]:
    if family not in {"wmma", "swmmac"} or operand not in {"iu8", "iu4"}:
        return {}
    return {
        "NEG[0]": "A operand signedness: 0 means unsigned, 1 means signed",
        "NEG[1]": "B operand signedness: 0 means unsigned, 1 means signed",
        "NEG[2]": "must be zero for integer WMMA/SWMMAC",
        "NEG_HI": "must be zero for integer WMMA/SWMMAC",
        "opsel": "RDNA4 SWMMAC OPSEL selects sparse compression index groups; not a dense RNS8 contract",
    }


def _rns8_semantic_requirements(family: str | None, operand: str | None, sparsity: str) -> dict[str, str]:
    centered_operand = "signed 4-bit A and B interpretation" if operand == "iu4" else "signed int8 A and B interpretation"
    wrap64_requirement = (
        "not a direct byte-limb wrap64 path; use only for explicit 4-bit experiments"
        if operand == "iu4"
        else "requires unsigned byte A and B interpretation or an explicit signed-byte correction path"
    )
    requirements = {
        "centered_residue_inputs": f"requires {centered_operand}",
        "wrap64_byte_limb_inputs": wrap64_requirement,
    }
    if family in {"mfma", "smfmac"} and operand == "i8":
        requirements["cdna_wrap64_note"] = "CDNA i8 MFMA/SMFMAC is signed-input evidence; strict byte-limb wrap64 needs the dedicated byte-limb backend or correction"
    if sparsity == "sparse":
        requirements["sparse_contract"] = "requires the explicit RNS8 sparse-A 4:2 structured-K contract, canonical A-side compression indices, dense B, and CPU sparse-reference parity; dense GEMM must not route here"
    return requirements


def _preferred_register_usage(wavefront_usage: dict[str, dict[str, int]]) -> tuple[str | None, dict[str, int | None]]:
    if "64" in wavefront_usage:
        chosen = "64"
    elif "32" in wavefront_usage:
        chosen = "32"
    elif "default" in wavefront_usage:
        chosen = "default"
    else:
        chosen = None
    usage = wavefront_usage.get(chosen or "", {})
    return chosen, {
        "a": usage.get("a"),
        "b": usage.get("b"),
        "c": usage.get("c"),
        "d": usage.get("d"),
    }


def parse_detail(architecture: str, requested_instruction: str, text: str) -> MatrixInstruction:
    traits = instruction_traits(requested_instruction)
    values: dict[str, Any] = {
        "architecture_query": architecture,
        "architecture_reported": None,
        "instruction": requested_instruction.lower(),
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
        **traits,
        "wavefront_register_usage": {},
        "preferred_register_usage": {},
        "register_data_types": {},
        "modifier_support": {},
        "rdna_integer_modifier_constraints": {},
        "rns8_semantic_requirements": {},
        "layout_artifacts": [],
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
        "CBSZ and ABID bits supported": "cbsz_abid_supported",
        "BLGP bits supported": "blgp_supported",
        "OPSEL supported": "opsel_supported",
        "NEG bits supported": "neg_supported",
    }
    current_register_scope: str | None = None
    in_data_types = False
    for line in text.splitlines():
        arch_match = DETAIL_ARCH_RE.match(line)
        if arch_match:
            values["architecture_reported"] = arch_match.group("value")
            continue
        instruction_match = DETAIL_INSTRUCTION_RE.match(line)
        if instruction_match:
            instruction = instruction_match.group("value").lower()
            values["instruction"] = instruction
            values.update(instruction_traits(instruction))
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
            target = bool_map[bool_match.group("key")]
            value = bool_match.group("value") == "True"
            if target in {"can_coexecute_with_valu", "sparse_a_matrix"}:
                values[target] = value
            else:
                values["modifier_support"][target] = value
            continue
        wave_register_match = DETAIL_WAVE_REGISTER_USAGE_RE.match(line)
        if wave_register_match:
            current_register_scope = wave_register_match.group("wave")
            values["wavefront_register_usage"].setdefault(current_register_scope, {})
            in_data_types = False
            continue
        if DETAIL_REGISTER_USAGE_RE.match(line):
            current_register_scope = "default"
            values["wavefront_register_usage"].setdefault(current_register_scope, {})
            in_data_types = False
            continue
        if DETAIL_DATA_TYPE_SECTION_RE.match(line):
            in_data_types = True
            current_register_scope = None
            continue
        data_type_match = DETAIL_DATA_TYPE_RE.match(line) if in_data_types else None
        if data_type_match:
            values["register_data_types"][data_type_match.group("field")] = data_type_match.group("value").strip()
            continue
        gpr_match = DETAIL_GPR_RE.match(line)
        if gpr_match:
            scope = current_register_scope or "default"
            role = gpr_match.group("role").lower()
            value = int(gpr_match.group("value"))
            values["wavefront_register_usage"].setdefault(scope, {})[role] = value
            continue

    preferred_wavefront, preferred = _preferred_register_usage(values["wavefront_register_usage"])
    values["preferred_register_usage"] = {"wavefront": preferred_wavefront, **preferred}
    values["gprs_a"] = preferred.get("a")
    values["gprs_b"] = preferred.get("b")
    values["gprs_c"] = preferred.get("c")
    values["gprs_d"] = preferred.get("d")
    values["rdna_integer_modifier_constraints"] = _rdna_integer_constraints(
        values["instruction_family"], values["operand_type"]
    )
    values["rns8_semantic_requirements"] = _rns8_semantic_requirements(
        values["instruction_family"], values["operand_type"], values["matrix_sparsity"]
    )
    return MatrixInstruction(**values)


def _sort_score(instruction: MatrixInstruction) -> tuple[int, int, int, str]:
    throughput = instruction.ops_per_cycle or 0
    accumulator_regs = instruction.gprs_d if instruction.gprs_d is not None else instruction.gprs_c
    if accumulator_regs is None:
        accumulator_regs = 999
    input_regs = (instruction.gprs_a or 0) + (instruction.gprs_b or 0)
    return (-throughput, accumulator_regs, input_regs, instruction.instruction)


def _register_summary(instruction: MatrixInstruction) -> dict[str, int | str | None]:
    return {
        "wavefront": instruction.preferred_register_usage.get("wavefront"),
        "a": instruction.gprs_a,
        "b": instruction.gprs_b,
        "c": instruction.gprs_c,
        "d": instruction.gprs_d,
    }


def _dense_recommendations(instructions: list[MatrixInstruction]) -> list[dict[str, Any]]:
    dense = [
        item
        for item in instructions
        if item.rns8_integer_candidate and item.matrix_sparsity == "dense"
    ]
    ranked = sorted(dense, key=_sort_score)
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(ranked):
        reason = "highest_dense_integer_throughput_lowest_accumulator_register_pressure"
        if index > 0 and ranked and item.ops_per_cycle == ranked[0].ops_per_cycle:
            reason = "same_dense_integer_throughput_higher_accumulator_register_pressure"
        rows.append(
            {
                "instruction": item.instruction,
                "rank": index + 1,
                "reason": reason,
                "family": item.instruction_family,
                "operand_type": item.operand_type,
                "operand_signedness": item.operand_signedness,
                "ops_per_cycle": item.ops_per_cycle,
                "ops_per_cycle_metric": item.ops_per_cycle_metric,
                "ops_per_cu_cycle": item.ops_per_cu_cycle,
                "ops_per_wgp_cycle": item.ops_per_wgp_cycle,
                "tile": {"m": item.m, "n": item.n, "k": item.k},
                "registers": _register_summary(item),
                "wavefront_register_usage": item.wavefront_register_usage,
                "rns8_semantic_requirements": item.rns8_semantic_requirements,
                "layout_artifacts": item.layout_artifacts,
            }
        )
    return rows


def _sparse_notes(instructions: list[MatrixInstruction]) -> list[dict[str, Any]]:
    sparse = [
        item
        for item in instructions
        if item.rns8_integer_candidate and item.matrix_sparsity == "sparse"
    ]
    return [
        {
            "instruction": item.instruction,
            "family": item.instruction_family,
            "operand_type": item.operand_type,
            "operand_signedness": item.operand_signedness,
            "ops_per_cycle": item.ops_per_cycle,
            "ops_per_cycle_metric": item.ops_per_cycle_metric,
            "ops_per_cu_cycle": item.ops_per_cu_cycle,
            "ops_per_wgp_cycle": item.ops_per_wgp_cycle,
            "tile": {"m": item.m, "n": item.n, "k": item.k},
            "registers": _register_summary(item),
            "wavefront_register_usage": item.wavefront_register_usage,
            "rdna_integer_modifier_constraints": item.rdna_integer_modifier_constraints,
            "eligibility": "requires_explicit_4_to_2_A_matrix_compression_contract",
            "layout_artifacts": item.layout_artifacts,
        }
        for item in sorted(sparse, key=_sort_score)
    ]


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "unknown"


def _relative(path: Path, base: Path) -> str:
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()


def _write_calculator_artifact(
    calculator: Path,
    args: list[str],
    path: Path,
    report_root: Path,
    *,
    kind: str,
    matrix: str,
    wavefront: str | None,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    command_args = [sys.executable, str(calculator), *args]
    try:
        output = run_calculator(calculator, args)
    except subprocess.CalledProcessError as exc:
        error_path = path.with_suffix(path.suffix + ".error.txt")
        error_path.write_text(
            "\n".join(
                [
                    "command: " + " ".join(command_args),
                    "stdout:",
                    exc.stdout or "",
                    "stderr:",
                    exc.stderr or "",
                ]
            ).rstrip()
            + "\n",
            encoding="utf-8",
        )
        return {
            "kind": kind,
            "matrix": matrix,
            "wavefront": wavefront,
            "status": "failed",
            "path": _relative(error_path, report_root),
            "command_args": args,
        }
    path.write_text(output, encoding="utf-8")
    return {
        "kind": kind,
        "matrix": matrix,
        "wavefront": wavefront,
        "status": "present",
        "path": _relative(path, report_root),
        "command_args": args,
    }


def _layout_wavefronts(instruction: MatrixInstruction) -> list[str | None]:
    wavefronts = [item for item in ("32", "64") if item in instruction.wavefront_register_usage]
    return wavefronts or [None]


def _representative_d_coordinates(instruction: MatrixInstruction) -> list[tuple[int, int]]:
    m = instruction.m or 0
    n = instruction.n or 0
    if m <= 0 or n <= 0:
        return []
    candidates = [(0, 0), (0, n - 1), (m - 1, 0), (m - 1, n - 1), (m // 2, n // 2)]
    deduped: list[tuple[int, int]] = []
    for item in candidates:
        if item not in deduped:
            deduped.append(item)
    return deduped


def capture_layout_artifacts(
    calculator: Path,
    instruction: MatrixInstruction,
    out_dir: Path,
    report_root: Path,
) -> list[dict[str, Any]]:
    arch_dir = out_dir / _safe_name(instruction.architecture_query)
    instr_dir = arch_dir / _safe_name(instruction.instruction)
    artifacts: list[dict[str, Any]] = []
    matrices = {"a": "--A-matrix", "b": "--B-matrix", "d": "--D-matrix"}
    if instruction.matrix_sparsity == "sparse":
        matrices["compression"] = "--compression"
    for wavefront in _layout_wavefronts(instruction):
        wave_args = ["--wavefront", wavefront] if wavefront else []
        wave_suffix = f"-wave{wavefront}" if wavefront else ""
        for matrix, matrix_flag in matrices.items():
            base_args = [
                "--architecture",
                instruction.architecture_query,
                "--instruction",
                instruction.instruction,
                matrix_flag,
                *wave_args,
            ]
            artifacts.append(
                _write_calculator_artifact(
                    calculator,
                    [*base_args, "--register-layout", "--csv"],
                    instr_dir / f"{matrix}{wave_suffix}-register-layout.csv",
                    report_root,
                    kind="register_layout_csv",
                    matrix=matrix,
                    wavefront=wavefront,
                )
            )
            artifacts.append(
                _write_calculator_artifact(
                    calculator,
                    [*base_args, "--matrix-layout", "--csv"],
                    instr_dir / f"{matrix}{wave_suffix}-matrix-layout.csv",
                    report_root,
                    kind="matrix_layout_csv",
                    matrix=matrix,
                    wavefront=wavefront,
                )
            )
        for i, j in _representative_d_coordinates(instruction):
            artifacts.append(
                _write_calculator_artifact(
                    calculator,
                    [
                        "--architecture",
                        instruction.architecture_query,
                        "--instruction",
                        instruction.instruction,
                        "--D-matrix",
                        *wave_args,
                        "--get-register",
                        "--I-coordinate",
                        str(i),
                        "--J-coordinate",
                        str(j),
                        "--output-calculation",
                    ],
                    instr_dir / f"d{wave_suffix}-output-i{i}-j{j}.txt",
                    report_root,
                    kind="d_output_calculation",
                    matrix="d",
                    wavefront=wavefront,
                )
            )
    return artifacts


def summarize_architecture(
    calculator: Path,
    architecture: str,
    *,
    layout_root: Path | None = None,
    report_root: Path | None = None,
) -> dict[str, Any]:
    list_output = run_calculator(calculator, ["--architecture", architecture, "--list-instructions"])
    listed = parse_instruction_list(list_output)
    selected = [
        instruction
        for instruction in listed
        if instruction_traits(instruction)["rns8_integer_candidate"]
    ]
    details = [
        parse_detail(
            architecture,
            instruction,
            run_calculator(calculator, ["--architecture", architecture, "--instruction", instruction, "--detail-instruction"]),
        )
        for instruction in selected
    ]
    if layout_root is not None and report_root is not None:
        for item in details:
            item.layout_artifacts = capture_layout_artifacts(calculator, item, layout_root, report_root)
    dense = [item for item in details if item.matrix_sparsity == "dense"]
    sparse = [item for item in details if item.matrix_sparsity == "sparse"]
    dense_i8 = [item for item in dense if item.integer_operand_bits == 8]
    sparse_i8 = [item for item in sparse if item.integer_operand_bits == 8]
    dense_i4 = [item for item in dense if item.integer_operand_bits == 4]
    sparse_i4 = [item for item in sparse if item.integer_operand_bits == 4]
    return {
        "architecture_query": architecture,
        "listed_instruction_count": len(listed),
        "rns8_integer_instruction_count": len(details),
        "dense_integer_i32_instruction_count": len(dense),
        "sparse_integer_i32_instruction_count": len(sparse),
        "dense_i8_i32_instruction_count": len(dense_i8),
        "sparse_i8_i32_instruction_count": len(sparse_i8),
        "dense_i4_i32_instruction_count": len(dense_i4),
        "sparse_i4_i32_instruction_count": len(sparse_i4),
        "dense_integer_i32_instructions": [asdict(item) for item in sorted(dense, key=_sort_score)],
        "sparse_integer_i32_instructions": [asdict(item) for item in sorted(sparse, key=_sort_score)],
        "dense_i8_i32_instructions": [asdict(item) for item in sorted(dense_i8, key=_sort_score)],
        "sparse_i8_i32_instructions": [asdict(item) for item in sorted(sparse_i8, key=_sort_score)],
        "dense_integer_i32_recommendations": _dense_recommendations(details),
        "sparse_integer_i32_notes": _sparse_notes(details),
        "dense_i8_i32_recommendations": _dense_recommendations(dense_i8),
        "sparse_i8_i32_notes": _sparse_notes(sparse_i8),
    }


def build_report(
    calculator: Path,
    architectures: list[str],
    *,
    out_dir: Path | None = None,
    capture_layouts: bool = True,
) -> dict[str, Any]:
    version = None
    try:
        version = run_calculator(calculator, ["--version"]).strip() or None
    except (OSError, subprocess.CalledProcessError):
        version = None
    layout_root = (out_dir / "layouts") if out_dir is not None and capture_layouts else None
    summaries = [
        summarize_architecture(calculator, architecture, layout_root=layout_root, report_root=out_dir)
        for architecture in architectures
    ]
    return {
        "schema_version": 2,
        "policy": REPORT_POLICY,
        "calculator": str(calculator),
        "calculator_version": version,
        "default_architectures": list(DEFAULT_ARCHITECTURES),
        "layout_capture_status": "enabled" if layout_root is not None else "disabled",
        "architectures": summaries,
        "rdna_integer_modifier_policy": {
            "NEG[0]": "A operand signedness: 0 unsigned, 1 signed",
            "NEG[1]": "B operand signedness: 0 unsigned, 1 signed",
            "NEG[2]": "must be zero for integer WMMA/SWMMAC",
            "NEG_HI": "must be zero for integer WMMA/SWMMAC",
        },
        "rns8_implications": [
            "Use this report to choose matrix-core candidates and register-layout investigations; still verify compiled objects with gpu_isa_report and exact CPU comparisons.",
            "Dense RNS8 int8xint8-to-int32 GEMM on CDNA should prioritize dense MFMA i8 instructions before sparse SMFMAC unless inputs carry an explicit sparse compression contract.",
            "RDNA integer WMMA/SWMMAC IU8/IU4 instructions require explicit signedness modifier handling; centered residues need signed A/B and wrap64 byte limbs need unsigned A/B where supported.",
            "Sparse SMFMAC/SWMMAC is not a drop-in correctness-preserving path for current dense bounded/exact/finite/wrap RNS8 contracts.",
        ],
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# AMD Matrix Instruction Report",
        "",
        f"Policy: `{report['policy']}`",
        "",
        "Calculator evidence guides backend experiments. Compiled ISA reports and exact CPU comparisons remain the proof.",
        "",
    ]
    for arch in report["architectures"]:
        lines.extend(
            [
                f"## {arch['architecture_query']}",
                "",
                "| rank | instruction | family | operand | tile | ops/cycle | preferred A/B/C/D GPRs | wave | reason |",
                "| --- | --- | --- | --- | --- | ---: | --- | --- | --- |",
            ]
        )
        for row in arch["dense_integer_i32_recommendations"]:
            tile = row["tile"]
            regs = row["registers"]
            lines.append(
                f"| {row['rank']} | `{row['instruction']}` | `{row['family']}` | `{row['operand_type']}` | "
                f"{tile['m']}x{tile['n']}x{tile['k']} | {row['ops_per_cycle']} {row['ops_per_cycle_metric']} | "
                f"{regs['a']}/{regs['b']}/{regs['c']}/{regs['d']} | {regs.get('wavefront')} | {row['reason']} |"
            )
        if arch["sparse_integer_i32_notes"]:
            lines.extend(["", "Sparse instructions require explicit A-side compression metadata:", ""])
            for row in arch["sparse_integer_i32_notes"]:
                tile = row["tile"]
                lines.append(
                    f"- `{row['instruction']}`: {tile['m']}x{tile['n']}x{tile['k']}, "
                    f"{row['ops_per_cycle']} {row['ops_per_cycle_metric']}, {row['eligibility']}"
                )
        artifact_count = sum(
            len(item.get("layout_artifacts") or [])
            for group_name in ("dense_integer_i32_instructions", "sparse_integer_i32_instructions")
            for item in arch.get(group_name, [])
        )
        lines.extend(
            [
                "",
                f"Layout artifacts: `{artifact_count}`",
                "",
            ]
        )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calculator", type=Path, default=DEFAULT_CALCULATOR)
    parser.add_argument(
        "--architectures",
        default=",".join(DEFAULT_ARCHITECTURES),
        help="comma-separated architecture aliases, for example gfx942,gfx1100,gfx1200,gfx1201",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("temp") / "amd-matrix-instruction-reports")
    parser.add_argument("--markdown", action="store_true", help="also write a Markdown summary")
    parser.add_argument("--no-layouts", action="store_true", help="skip register/matrix layout artifact capture")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.calculator.exists():
        raise RuntimeError(f"AMD matrix instruction calculator not found: {args.calculator}")
    architectures = [item.strip() for item in args.architectures.split(",") if item.strip()]
    if not architectures:
        raise RuntimeError("at least one architecture is required")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(args.calculator, architectures, out_dir=args.out_dir, capture_layouts=not args.no_layouts)
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
