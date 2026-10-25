#!/usr/bin/env python3
"""Emit temp-only ISA summaries for compiled RNS8 HIP objects."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import hashlib
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from benchmark_schema import load_capture, validate_capture
from isa_common import (
    IsaToolConfig,
    device_function_symbols,
    disassemble_code_object,
    extracted_device_code_object,
    mnemonic_lines,
    run_command,
    sibling_tool,
)


BACKEND_OBJECT_MARKERS = {
    "direct-hip": ["hip_direct_kernels"],
    "hipblaslt": ["hipblaslt_kernels"],
    "ck": ["ck_backend_kernels"],
    "rocwmma": ["rocwmma_backend_kernels"],
    "amdgpu-builtins": ["amdgpu_builtins_kernels"],
    "wrap64": ["wrap64_hip_kernels"],
    "vector-alu": ["vector_alu_kernels", "hip_vector_alu_baseline_kernels"],
}
CMAKE_OPTION_BACKENDS = {
    "RNS8_ENABLE_HIP": ["direct-hip", "wrap64", "vector-alu"],
    "RNS8_ENABLE_HIPBLASLT": ["hipblaslt"],
    "RNS8_ENABLE_CK": ["ck"],
    "RNS8_ENABLE_ROCWMMA": ["rocwmma"],
    "RNS8_ENABLE_AMDGPU_BUILTINS": ["amdgpu-builtins"],
}
BACKEND_SYMBOL_MARKERS = {
    "direct-hip": ["rns8_ring_gemm_i8_i32", "rns8_export", "finite", "exact_wide", "rns8_wrap_signed_radix", "rns8_wrap_native", "rns8_word_mod"],
    "hipblaslt": ["rns8_hipblaslt", "pack_transpose", "reduce_i32_to_centered"],
    "ck": ["kernel_gemm_wmma", "ck_"],
    "rocwmma": ["rocwmma_i8_residue_gemm", "rocwmma_wrap64_byte_gemm36_candidate"],
    "amdgpu-builtins": [
        "cdna3_dense_mfma_kernel",
        "rdna3_dense_wmma_kernel",
        "rdna4_dense_wmma_kernel",
        "cdna3_sparse_smfmac_kernel",
        "rdna4_sparse_swmmac_kernel",
        "rns8_amdgpu_builtin",
    ],
    "wrap64": ["rns8_wrap64"],
    "vector-alu": ["gemm_i64_kernel", "gemm_u64_kernel", "rns8_vector_alu"],
}
GLOBAL_STORE_RE = re.compile(r"\b(?:global|buffer)_store\b|\b(?:global|buffer)_store_[a-z0-9_]+\b")
MATRIX_MNEMONIC_RE = re.compile(r"^v_(?:mfma|smfmac|wmma|swmmac)[a-z0-9_]*$")
VGPR_RE = re.compile(r"(?:VGPRs?|\.amdhsa_next_free_vgpr)\D+(\d+)", re.IGNORECASE)
SGPR_RE = re.compile(r"(?:SGPRs?|\.amdhsa_next_free_sgpr)\D+(\d+)", re.IGNORECASE)
OCCUPANCY_RE = re.compile(r"occupancy\D+(\d+)", re.IGNORECASE)
METADATA_KERNEL_START_RE = re.compile(r"^\s*-\s+\.args:")
METADATA_NAME_RE = re.compile(r"^\s*\.name:\s*(?P<value>\S+)")
METADATA_INT_RE = re.compile(
    r"^\s*\.(?P<key>"
    r"group_segment_fixed_size|private_segment_fixed_size|max_flat_workgroup_size|"
    r"sgpr_count|sgpr_spill_count|vgpr_count|vgpr_spill_count|wavefront_size"
    r"):\s*(?P<value>\d+)\s*$"
)
METADATA_KEY_MAP = {
    "group_segment_fixed_size": "lds_bytes",
    "private_segment_fixed_size": "scratch_bytes",
    "max_flat_workgroup_size": "max_flat_workgroup_size",
    "sgpr_count": "sgpr_count",
    "sgpr_spill_count": "sgpr_spill_count",
    "vgpr_count": "vgpr_count",
    "vgpr_spill_count": "vgpr_spill_count",
    "wavefront_size": "wavefront_size",
}


def capture_metadata(path: Path, label: str | None) -> dict[str, Any]:
    capture = load_capture(path)
    if capture.get("schema")=="rns8_residue_dot_microcase_v1":
        from collect_residue_dot import validate_capture as validate_residue_capture
        validate_residue_capture(capture,dict(case=capture.get('case'),run_id=capture.get('run_id'),arithmetic=capture.get('arithmetic')),capture.get('hooks_enabled') is True)
        return dict(path=str(path),label=label,schema_version=capture['schema'],benchmark='residue-dot',benchmark_execution_mode=capture['case']['mode'],backend_selected='direct-hip',selected_kernel=capture['selected_kernel'],artifact_capture=capture['artifact_capture'],
            build_manifest_sha256=hashlib.sha256(json.dumps(capture['build_manifest'],sort_keys=True,separators=(",", ":")).encode()).hexdigest(),semantics=capture['case']['domain'],shape={k:capture['case'][k] for k in ('m','n','k')},target_id=capture['target'],target_namespace='amdgcn',review_group_key=None,pack_layout='centered_modulus_major',fusion_mode='same_scheduler_residue_inner_loop',generated_reducer_identity=None)
    if capture.get("schema")=="rns8_wrap64_microcase_v1":
        from collect_wrap64 import validate_capture as validate_wrap_capture
        case={key:capture.get(key) for key in ('candidate','scope','m','n','k','pattern','reuse_count')}
        case['device']=capture.get('device_requested')
        validate_wrap_capture(capture,case,capture.get('run_id'),capture.get('repeats'),capture.get('warmups'))
        backend='rocwmma' if case['candidate']=='rocwmma_corrected' else 'wrap64' if case['candidate']=='direct_v4' else 'direct-hip'
        return dict(path=str(path),label=label,schema_version=capture['schema'],benchmark='wrap64',benchmark_execution_mode=case['scope'],backend_selected=backend,selected_kernel=capture['kernel_selection'],artifact_capture=capture.get('artifact_capture'),
            build_manifest_sha256=hashlib.sha256(json.dumps(capture['build_manifest'],sort_keys=True,separators=(",", ":")).encode()).hexdigest(),semantics=capture['semantics'],shape={k:case[k] for k in ('m','n','k')},target_id=capture['target'],target_namespace='amdgcn',review_group_key=None,pack_layout=capture['compute_layout'],fusion_mode='per_pair_unsigned_ring_fold',generated_reducer_identity=capture['arithmetic_certificate_sha256'])
    if capture.get("schema")=="rns8_word_mod_microcase_v1":
        from collect_word_mod_calibration import validate_capture as validate_word_capture, FLOAT_METHODS
        keys=('device','method','scope','m','n','k','domain','modulus')
        case={key:capture.get(key) for key in keys}
        if case['method'] in FLOAT_METHODS:case.update({key:capture.get(key) for key in ('left_digits','right_digits','slab_k')})
        validate_word_capture(capture,case,capture.get('run_id'),capture.get('repeats'),capture.get('warmups'))
        recipe={key:capture.get(key) for key in ('floating_schedule','fp64_mfma_schedule','byte_wmma_schedule','canonical_dot4_schedule')}
        return dict(path=str(path),label=label,schema_version=capture['schema'],benchmark='word-modular',benchmark_execution_mode=case['scope'],backend_selected='direct-hip',selected_kernel=capture['arithmetic_kernel'],artifact_capture=capture.get('artifact_capture'),
            build_manifest_sha256=hashlib.sha256(json.dumps(capture['build_manifest'],sort_keys=True,separators=(",", ":")).encode()).hexdigest(),semantics=case['domain'],shape={k:case[k] for k in ('m','n','k')},target_id=capture['target'],target_namespace='amdgcn',review_group_key=None,pack_layout=(capture.get('fp64_mfma_schedule') or {}).get('layout','canonical_native_u64'),fusion_mode=case['method'],generated_reducer_identity=hashlib.sha256(json.dumps(recipe,sort_keys=True,separators=(",", ":")).encode()).hexdigest())
    validate_capture(capture, path)
    target = capture.get("target_variant")
    if not isinstance(target, dict):
        target = {}
    timing = capture.get("timing_metadata")
    if not isinstance(timing, dict):
        timing = {}
    return {
        "path": str(path),
        "label": label,
        "schema_version": capture.get("schema_version"),
        "benchmark": capture.get("benchmark"),
        "benchmark_execution_mode": capture.get("benchmark_execution_mode"),
        "backend_selected": capture.get("backend_selected"),
        "selected_kernel": capture.get("selected_kernel"),
        "artifact_capture": capture.get("artifact_capture"),
        "build_manifest_sha256": hashlib.sha256(json.dumps(capture.get("build_manifest"), sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "semantics": capture.get("semantics"),
        "shape": {"m": capture.get("m"), "n": capture.get("n"), "k": capture.get("k")},
        "target_id": target.get("target_id") or (capture.get("device") or {}).get("gcn_arch"),
        "target_namespace": target.get("target_namespace"),
        "review_group_key": target.get("review_group_key"),
        "pack_layout": timing.get("pack_layout"),
        "fusion_mode": timing.get("fusion_mode"),
        "generated_reducer_identity": timing.get("generated_reducer_identity"),
    }


def discover_objects(build_tree: Path, backend: str) -> list[Path]:
    if backend == "all":
        markers = sorted({marker for values in BACKEND_OBJECT_MARKERS.values() for marker in values})
    else:
        markers = BACKEND_OBJECT_MARKERS.get(backend, [])
    candidates = sorted(
        path
        for path in build_tree.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".obj", ".o"}
        and (not markers or any(marker in path.name for marker in markers))
    )
    return candidates


def read_cmake_cache(build_tree: Path | None) -> dict[str, str]:
    if build_tree is None:
        return {}
    cache_path = build_tree / "CMakeCache.txt"
    if not cache_path.exists():
        return {}
    values: dict[str, str] = {}
    for line in cache_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line or line.startswith("//") or line.startswith("#") or ":" not in line or "=" not in line:
            continue
        key, rest = line.split(":", 1)
        _type, value = rest.split("=", 1)
        values[key] = value.strip()
    return values


def expected_backends_from_cmake_cache(build_tree: Path | None, backend: str) -> set[str]:
    if backend != "all":
        return {backend}
    cache = read_cmake_cache(build_tree)
    expected: set[str] = set()
    for option, backends in CMAKE_OPTION_BACKENDS.items():
        if cache.get(option, "").upper() == "ON":
            expected.update(backends)
    return expected


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


def matrix_mnemonic_family(mnemonic: str) -> str | None:
    lowered = mnemonic.lower()
    for family in ("smfmac", "swmmac", "mfma", "wmma"):
        if lowered.startswith(f"v_{family}"):
            return family
    return None


def matrix_mnemonic_categories(mnemonic: str) -> set[str]:
    lowered = mnemonic.lower()
    family = matrix_mnemonic_family(lowered)
    categories: set[str] = set()
    if family is None:
        return categories
    categories.add("matrix_core")
    if family in {"mfma", "wmma"}:
        categories.add("dense_matrix_core")
    if family in {"smfmac", "swmmac"}:
        categories.add("sparse_matrix_core")
    if "_i32_" in lowered and lowered.endswith(("_i8", "_iu8", "_iu4")):
        categories.add("integer_i32_matrix_core")
        if family in {"mfma", "wmma"}:
            categories.add("dense_integer_i32_matrix_core")
        if family in {"smfmac", "swmmac"}:
            categories.add("sparse_integer_i32_matrix_core")
    if family == "mfma" and "_i32_" in lowered and lowered.endswith("_i8"):
        categories.add("mfma_dense_i8")
    if family == "smfmac" and "_i32_" in lowered and lowered.endswith("_i8"):
        categories.add("smfmac_sparse_i8")
    if family == "wmma" and "_i32_" in lowered and lowered.endswith(("_iu8", "_iu4")):
        categories.add("wmma_dense_integer")
    if family == "swmmac" and "_i32_" in lowered and lowered.endswith(("_iu8", "_iu4")):
        categories.add("swmmac_sparse_integer")
    return categories


def scan_disassembly(disassembly: str) -> dict[str, Any]:
    mnemonics = mnemonic_lines(disassembly)
    counts = {
        "wmma": 0,
        "mfma": 0,
        "smfmac": 0,
        "swmmac": 0,
        "matrix_instruction_count": 0,
        "dense_integer_matrix_instruction_count": 0,
        "sparse_integer_matrix_instruction_count": 0,
        "mfma_dense_i8": 0,
        "smfmac_sparse_i8": 0,
        "wmma_dense_integer": 0,
        "swmmac_sparse_integer": 0,
        "global_store": 0,
        "lds_mentions": 0,
        "wait_instructions": 0,
        "instruction_lines": len(mnemonics),
        "packed_dot4":0,"scalar_integer_multiply_add":0,"byte_extract_permute":0,"lds_read":0,"lds_write":0,
    }
    matrix_histogram: dict[str, int] = {}
    matrix_families: set[str] = set()
    for line, mnemonic in mnemonics:
        lowered_line = line.lower()
        lowered_mnemonic = mnemonic.lower()
        if lowered_mnemonic.startswith(('v_dot4_','v_dot4c_')):counts['packed_dot4']+=1
        if lowered_mnemonic.startswith(('v_mul_lo_i32','v_mul_lo_u32','v_mad_i32','v_mad_u32','v_mad_i24','v_mad_u24','v_mul_i32_i24','v_mul_u32_u24')):counts['scalar_integer_multiply_add']+=1
        if lowered_mnemonic.startswith(('v_bfe_','v_perm_','v_lshrrev_b32','v_lshlrev_b32','v_lshl_or_b32')):counts['byte_extract_permute']+=1
        if lowered_mnemonic.startswith(('ds_read', 'ds_load')):counts['lds_read']+=1
        if lowered_mnemonic.startswith(('ds_write', 'ds_store')):counts['lds_write']+=1
        family = matrix_mnemonic_family(lowered_mnemonic)
        if family is not None and MATRIX_MNEMONIC_RE.match(lowered_mnemonic):
            matrix_histogram[lowered_mnemonic] = matrix_histogram.get(lowered_mnemonic, 0) + 1
            matrix_families.add(family)
            counts["matrix_instruction_count"] += 1
            for category in matrix_mnemonic_categories(lowered_mnemonic):
                if category == "dense_integer_i32_matrix_core":
                    counts["dense_integer_matrix_instruction_count"] += 1
                elif category == "sparse_integer_i32_matrix_core":
                    counts["sparse_integer_matrix_instruction_count"] += 1
                elif category in {
                    "mfma_dense_i8",
                    "smfmac_sparse_i8",
                    "wmma_dense_integer",
                    "swmmac_sparse_integer",
                }:
                    counts[category] += 1
        if lowered_mnemonic.startswith("v_wmma"):
            counts["wmma"] += 1
        if lowered_mnemonic.startswith("v_swmmac"):
            counts["swmmac"] += 1
        if lowered_mnemonic.startswith("v_mfma"):
            counts["mfma"] += 1
        if lowered_mnemonic.startswith("v_smfmac"):
            counts["smfmac"] += 1
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
        "matrix_instruction_histogram": dict(sorted(matrix_histogram.items())),
        "matrix_instruction_families": sorted(matrix_families),
        "vgpr_count": max(vgprs) if vgprs else None,
        "sgpr_count": max(sgprs) if sgprs else None,
        "occupancy": max(occupancy) if occupancy else None,
    }


def readobj_for_objdump(objdump: str, override: str | None = None) -> str | None:
    if override:
        return override
    objdump_path = Path(objdump)
    suffix = ".exe" if sys.platform == "win32" else ""
    candidates = [
        objdump_path.with_name(f"llvm-readobj{suffix}"),
        objdump_path.with_name("llvm-readobj"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    try:
        return sibling_tool(None, "llvm-readobj")
    except RuntimeError:
        return None


def parse_amdgpu_metadata(readobj_output: str) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    current: dict[str, Any] | None = None
    for line in readobj_output.splitlines():
        if METADATA_KERNEL_START_RE.match(line):
            if current and current.get("name"):
                metadata[str(current["name"])] = current
            current = {}
            continue
        if current is None:
            continue
        name_match = METADATA_NAME_RE.match(line)
        if name_match:
            current["name"] = name_match.group("value")
            continue
        int_match = METADATA_INT_RE.match(line)
        if not int_match:
            continue
        key = METADATA_KEY_MAP[int_match.group("key")]
        current[key] = int(int_match.group("value"))
    if current and current.get("name"):
        metadata[str(current["name"])] = current
    return metadata


def code_object_metadata(readobj: str | None, code_object: Path) -> dict[str, dict[str, Any]]:
    if readobj is None:
        return {}
    try:
        output = run_command([readobj, "--notes", str(code_object)])
    except (OSError, RuntimeError, subprocess.SubprocessError):
        return {}
    return parse_amdgpu_metadata(output)


def merge_resource_metadata(counts: dict[str, Any], metadata: dict[str, Any] | None) -> dict[str, Any]:
    if not metadata:
        return counts
    merged = dict(counts)
    for key in (
        "vgpr_count",
        "sgpr_count",
        "lds_bytes",
        "scratch_bytes",
        "vgpr_spill_count",
        "sgpr_spill_count",
        "max_flat_workgroup_size",
        "wavefront_size",
    ):
        if metadata.get(key) is not None:
            merged[key] = metadata.get(key)
    return merged


def _merge_histograms(symbol_reports: list[dict[str, Any]], key: str) -> dict[str, int]:
    merged: dict[str, int] = {}
    for report in symbol_reports:
        histogram = report["counts"].get(key)
        if not isinstance(histogram, dict):
            continue
        for name, value in histogram.items():
            try:
                merged[str(name)] = merged.get(str(name), 0) + int(value)
            except (TypeError, ValueError):
                continue
    return dict(sorted(merged.items()))


def sum_symbol_reports(symbol_reports: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "wmma",
        "mfma",
        "smfmac",
        "swmmac",
        "matrix_instruction_count",
        "dense_integer_matrix_instruction_count",
        "sparse_integer_matrix_instruction_count",
        "mfma_dense_i8",
        "smfmac_sparse_i8",
        "wmma_dense_integer",
        "swmmac_sparse_integer",
        "global_store",
        "lds_mentions",
        "wait_instructions",
        "instruction_lines",
        "packed_dot4", "scalar_integer_multiply_add", "byte_extract_permute", "lds_read", "lds_write",
    ]
    totals = {key: sum(int(report["counts"].get(key) or 0) for report in symbol_reports) for key in keys}
    totals["matrix_instruction_histogram"] = _merge_histograms(symbol_reports, "matrix_instruction_histogram")
    totals["matrix_instruction_families"] = sorted(
        {
            str(family)
            for report in symbol_reports
            for family in (report["counts"].get("matrix_instruction_families") or [])
            if isinstance(family, str)
        }
    )
    for key in [
        "vgpr_count",
        "sgpr_count",
        "lds_bytes",
        "scratch_bytes",
        "vgpr_spill_count",
        "sgpr_spill_count",
        "max_flat_workgroup_size",
        "wavefront_size",
        "occupancy",
    ]:
        values = [report["counts"].get(key) for report in symbol_reports if report["counts"].get(key) is not None]
        totals[key] = max(values) if values else None
    return totals


def load_matrix_instruction_candidates(path: Path | None, target: str) -> dict[str, Any] | None:
    if path is None:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"matrix instruction report must be a JSON object: {path}")
    candidates: set[str] = set()
    matched_architectures: list[str] = []
    for arch in data.get("architectures") or []:
        if not isinstance(arch, dict):
            continue
        aliases = {str(arch.get("architecture_query") or "").lower()}
        for group_name in ("dense_integer_i32_instructions", "sparse_integer_i32_instructions", "dense_i8_i32_instructions", "sparse_i8_i32_instructions"):
            for item in arch.get(group_name) or []:
                if isinstance(item, dict) and isinstance(item.get("architecture_query"), str):
                    aliases.add(item["architecture_query"].lower())
                if isinstance(item, dict) and isinstance(item.get("architecture_reported"), str):
                    aliases.add(item["architecture_reported"].lower())
        if target.lower() not in aliases:
            continue
        matched_architectures.append(str(arch.get("architecture_query") or target))
        for group_name in ("dense_integer_i32_instructions", "sparse_integer_i32_instructions", "dense_i8_i32_instructions", "sparse_i8_i32_instructions"):
            for item in arch.get(group_name) or []:
                if isinstance(item, dict) and isinstance(item.get("instruction"), str):
                    candidates.add(item["instruction"].lower())
    return {
        "path": str(path),
        "target": target,
        "matched_architectures": sorted(set(matched_architectures)),
        "candidate_instruction_count": len(candidates),
        "candidate_instructions": sorted(candidates),
        "status": "present" if candidates else "no_matching_target_candidates",
    }


def correlate_matrix_instruction_report(report: dict[str, Any], candidates: dict[str, Any] | None) -> None:
    if candidates is None:
        report["matrix_instruction_calculator_evidence"] = {
            "status": "not_attached",
            "observed_in_calculator_candidates": {},
        }
        return
    candidate_set = set(candidates.get("candidate_instructions") or [])
    observed = set((report.get("instruction_totals") or {}).get("matrix_instruction_histogram") or {})
    report["matrix_instruction_calculator_evidence"] = {
        **candidates,
        "observed_instruction_count": len(observed),
        "observed_in_calculator_candidates": {name: name in candidate_set for name in sorted(observed)},
        "observed_missing_from_calculator_candidates": sorted(observed - candidate_set),
    }


def report_object(config: IsaToolConfig, backend: str, rga_path: Path | None, readobj_path: str | None) -> dict[str, Any]:
    object_hash = hashlib.sha256(config.host_object.read_bytes()).hexdigest()
    with extracted_device_code_object(config, "rns8-gpu-isa-", f"{config.host_object.stem}.fatbin") as code_object:
        code_hash = hashlib.sha256(code_object.read_bytes()).hexdigest()
        symbols = device_function_symbols(config.objdump, code_object)
        metadata = code_object_metadata(readobj_path, code_object)
        symbols_to_report, note = selected_symbols(symbols, backend)
        symbol_reports = []
        for symbol in symbols_to_report:
            disassembly = disassemble_code_object(config.objdump, code_object, config.target, symbol)
            symbol_metadata = metadata.get(symbol)
            symbol_reports.append(
                {
                    "symbol": symbol,
                    "counts": merge_resource_metadata(scan_disassembly(disassembly), symbol_metadata),
                    "metadata": symbol_metadata,
                }
            )
        if hashlib.sha256(config.host_object.read_bytes()).hexdigest() != object_hash:
            raise RuntimeError("inspected object changed during disassembly")
        return {
            "schema": "rns8_static_isa_v2",
            "object_sha256": object_hash,
            "code_object_sha256": code_hash,
            "static_inventory_only": True,
            "object": str(config.host_object),
            "target": config.target,
            "backend": backend,
            "tools": {
                "llvm_objcopy": config.objcopy,
                "llvm_objdump": config.objdump,
                "llvm_readobj": readobj_path,
                "clang_offload_bundler": config.bundler,
                "rga": str(rga_path) if rga_path is not None else None,
                "rga_status": "not_run_optional",
                "amdgpu_metadata_status": "present" if metadata else "missing",
            },
            "code_object_note": note,
            "device_symbol_count": len(symbols),
            "reported_symbol_count": len(symbol_reports),
            "symbols": symbol_reports,
            "instruction_totals": sum_symbol_reports(symbol_reports),
        }


def bind_selected_symbols(report: dict[str, Any], capture: dict[str, Any], artifact: Path,
                          symbols: list[str], config: IsaToolConfig) -> None:
    """Prove artifact membership, not dynamic execution, for an explicit mapping."""
    from evidence_database_lib.isa import normalized_target, normalized_backend
    if not symbols or len(set(symbols)) != len(symbols):
        raise RuntimeError("selected static symbol mapping must be nonempty and unique")
    available = {entry["symbol"] for entry in report.get("symbols", [])}
    if not set(symbols) <= available:
        raise RuntimeError("selected symbol is absent from the inspected code object")
    if normalized_backend(capture.get("backend_selected")) != normalized_backend(report.get("backend")):
        raise RuntimeError("capture and ISA backend differ")
    target = normalized_target(capture.get("target_id"))
    if target is None or target != normalized_target(report.get("target")):
        raise RuntimeError("capture and ISA concrete target differ or are unknown")
    identity = capture.get("artifact_capture") or {}
    artifact_hash = hashlib.sha256(artifact.read_bytes()).hexdigest()
    kind = "executable_sha256" if artifact_hash == identity.get("executable_sha256") else "shared_library_sha256"
    if identity.get("unchanged_during_run") is not True or artifact_hash != identity.get(kind):
        raise RuntimeError("artifact is not the unchanged executable/library in the capture")
    if artifact_hash == report.get("object_sha256"):
        member_hash = report["code_object_sha256"]
    else:
        artifact_config = IsaToolConfig(artifact, config.target, config.objcopy, config.objdump, config.bundler, config.scratch_root)
        with extracted_device_code_object(artifact_config, "rns8-isa-artifact-", "artifact.fatbin") as member:
            member_hash = hashlib.sha256(member.read_bytes()).hexdigest()
    if member_hash != report.get("code_object_sha256"):
        raise RuntimeError("inspected code object does not match the captured artifact's target code object")
    if hashlib.sha256(artifact.read_bytes()).hexdigest() != artifact_hash:
        raise RuntimeError("artifact changed during ISA association")
    if not isinstance(capture.get("selected_kernel"), str) or not capture["selected_kernel"]:
        raise RuntimeError("capture has no selected kernel identity")
    report["capture_binding"] = {
        "schema": "rns8_selected_static_isa_binding_v1", "artifact_kind": kind,
        "artifact_sha256": artifact_hash, "code_object_sha256": member_hash,
        "build_manifest_sha256": capture.get("build_manifest_sha256"),
        "target": target, "selected_kernel": capture["selected_kernel"],
        "symbols": sorted(symbols), "selection_evidence": "explicit_static_symbol_mapping",
        "artifact_membership_verified": True, "dynamic_execution_proven": False}
    report["static_inventory_only"] = False


def write_report(report: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_backend = str(report.get("backend") or "unknown").replace("/", "_")
    stem = Path(str(report["object"])).stem
    target = str(report.get("target") or "unknown")
    out_path = out_dir / f"{stem}-{str(report.get('object_sha256') or 'unbound')[:16]}-{target}-{safe_backend}-isa-summary.json"
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def write_manifest(
    *,
    args: argparse.Namespace,
    outputs: list[Path],
    skipped_objects: list[dict[str, Any]],
    expected_backends: set[str],
    reported_backends: set[str],
    out_dir: Path,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "backend": args.backend,
        "target": args.target,
        "reports": [str(path) for path in outputs],
        "report_count": len(outputs),
        "expected_backends": sorted(expected_backends),
        "reported_backends": sorted(reported_backends),
        "skipped_objects": skipped_objects,
        "skipped_object_count": len(skipped_objects),
    }
    manifest_path = out_dir / "gpu-isa-report-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def is_missing_hip_fatbin_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return ".hip_fatbin" in text and "not found" in text


def json_safe_config(config: IsaToolConfig) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object", dest="objects", action="append", type=Path, help="compiled HIP host object")
    parser.add_argument("--build-tree", type=Path, help="discover HIP objects under a CMake build tree")
    parser.add_argument(
        "--backend",
        default="all",
        choices=["all", "direct-hip", "hipblaslt", "ck", "rocwmma", "amdgpu-builtins", "wrap64", "vector-alu"],
    )
    parser.add_argument("--target", required=True, help="AMDGPU target id, for example gfx1100")
    parser.add_argument("--hipcc", type=Path, help="HIP compiler path; sibling LLVM tools are preferred")
    parser.add_argument("--llvm-objcopy", help="Override llvm-objcopy path")
    parser.add_argument("--llvm-objdump", help="Override llvm-objdump path")
    parser.add_argument("--llvm-readobj", help="Override llvm-readobj path")
    parser.add_argument("--clang-offload-bundler", help="Override clang-offload-bundler path")
    parser.add_argument("--rga", type=Path, help="Optional RGA CLI path recorded in the report; not run by default")
    parser.add_argument("--capture", type=Path, help="optional schema-v4 benchmark capture to validate and cross-link")
    parser.add_argument("--capture-label", help="human label for the cross-linked capture")
    parser.add_argument("--artifact", type=Path, help="exact captured executable/library; verify inspected code-object membership")
    parser.add_argument("--selected-symbol", action="append", default=[], help="exact static symbol in the selected kernel mapping; repeatable, requires --capture and --artifact")
    parser.add_argument(
        "--matrix-instruction-report",
        type=Path,
        help="optional amd_matrix_instruction_report.py JSON used to cross-link observed matrix mnemonics",
    )
    parser.add_argument("--scratch-root", type=Path, default=Path("temp"), help="ignored scratch directory")
    parser.add_argument("--out-dir", type=Path, default=Path("temp") / "isa-reports", help="temp-only output directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if bool(args.artifact) != bool(args.selected_symbol) or (args.selected_symbol and not args.capture):
        raise RuntimeError("qualified static association requires --capture, --artifact and --selected-symbol together")
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
    readobj_path = readobj_for_objdump(config_template["objdump"], args.llvm_readobj)
    linked_capture = capture_metadata(args.capture, args.capture_label) if args.capture is not None else None
    matrix_candidates = load_matrix_instruction_candidates(args.matrix_instruction_report, args.target)

    outputs = []
    skipped_objects: list[dict[str, Any]] = []
    reported_backends: set[str] = set()
    expected_backends = expected_backends_from_cmake_cache(args.build_tree, args.backend)
    if not expected_backends:
        expected_backends = {backend_for_object(obj, args.backend) for obj in unique_objects}
        expected_backends.discard("all")
    for obj in unique_objects:
        if not obj.exists():
            raise RuntimeError(f"HIP object does not exist: {obj}")
        backend = backend_for_object(obj, args.backend)
        config = IsaToolConfig(host_object=obj, **config_template)
        try:
            report = report_object(config, backend, args.rga, readobj_path)
        except RuntimeError as exc:
            if not is_missing_hip_fatbin_error(exc):
                raise
            skipped_objects.append(
                {
                    "object": str(obj),
                    "backend": backend,
                    "reason": "missing_hip_fatbin_section",
                    "detail": str(exc).splitlines()[0],
                }
            )
            continue
        if linked_capture is not None:
            report["capture"] = linked_capture
            if args.artifact:
                bind_selected_symbols(report, linked_capture, args.artifact, args.selected_symbol, config)
        correlate_matrix_instruction_report(report, matrix_candidates)
        report["config"] = json_safe_config(config)
        outputs.append(write_report(report, args.out_dir))
        reported_backends.add(backend)

    missing_backends = sorted(expected_backends - reported_backends)
    if missing_backends:
        write_manifest(
            args=args,
            outputs=outputs,
            skipped_objects=skipped_objects,
            expected_backends=expected_backends,
            reported_backends=reported_backends,
            out_dir=args.out_dir,
        )
        raise RuntimeError(
            "no ISA report produced for expected backend(s): "
            + ", ".join(missing_backends)
            + "; skipped objects are recorded in gpu-isa-report-manifest.json"
        )
    if not outputs:
        write_manifest(
            args=args,
            outputs=outputs,
            skipped_objects=skipped_objects,
            expected_backends=expected_backends,
            reported_backends=reported_backends,
            out_dir=args.out_dir,
        )
        raise RuntimeError("no ISA reports produced; skipped objects are recorded in gpu-isa-report-manifest.json")
    manifest_path = write_manifest(
        args=args,
        outputs=outputs,
        skipped_objects=skipped_objects,
        expected_backends=expected_backends,
        reported_backends=reported_backends,
        out_dir=args.out_dir,
    )

    print("GPU ISA report: PASS")
    for out_path in outputs:
        print(f"- {out_path}")
    if skipped_objects:
        print(f"skipped_non_hip_objects={len(skipped_objects)} manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - command-line tool should print concise failure cause.
        print(f"GPU ISA report: FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
