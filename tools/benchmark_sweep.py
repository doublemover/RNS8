#!/usr/bin/env python3
"""Run and review reproducible rns8-bench capture sweeps."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark_schema import BenchmarkSchemaError, load_capture, validate_capture


BOUNDED_BACKENDS = ["cpu", "hip-direct", "hip-vector-alu-int64", "hipblaslt", "ck", "rocwmma"]
PUBLIC_ONESHOT_BACKENDS = ["cpu", "hip-direct"]
EXACT_WIDE_BACKENDS = ["cpu", "hip-direct", "hipblaslt", "ck", "rocwmma"]
FINITE_BACKENDS = ["cpu", "hip-direct", "hipblaslt", "ck", "rocwmma"]
WRAP64_BACKENDS = ["wrap64-byte-limb", "hip-direct"]
WRAP64_ROCWMMA_CANDIDATE_BACKEND = "rocwmma-wrap64-candidate"
BOUNDED_SEMANTICS = ["bounded-i64", "bounded-u64"]
EXACT_WIDE_SEMANTICS = ["exact-wide-signed", "exact-wide-unsigned"]
RNS_CHAIN_SEMANTICS = BOUNDED_SEMANTICS + EXACT_WIDE_SEMANTICS
PHASES = ["pack", "rns_gemm", "crt_export", "end_to_end"]
REVIEW_SCHEMA_VERSION = 3
PLACEHOLDER_TARGET_IDS = {"", "none", "cpu", "unknown", "not_applicable", "n/a", "null"}
RELEASE_MIN_WARMUPS = 3
RELEASE_MIN_REPEATS = 9
PROMOTABLE_RELEASE_SHAPES = [64, 128, 512, 1024]
EXPLORATORY_RELEASE_SHAPES = [2048, 4096, 8192]
DEFAULT_ADAPTIVE_CASES = [
    "tiny-adaptive:65,65,64,64,64",
    "medium-adaptive:1024,1024,1024,128,128",
]
ADAPTIVE_WORKLOAD_CASES = [
    "banded-adaptive-256:256,256,512,64,64,adaptive-bands",
    "banded-adaptive-1024:1024,1024,1024,128,128,adaptive-bands",
    "banded-rect-adaptive:512,1024,512,128,128,adaptive-bands",
]
DEFAULT_FINITE_RING_MODULI = [251, 255]
DEFAULT_FINITE_FIELD_MODULI = [251]
INPUT_PROFILES = {"uniform-small", "adaptive-bands"}
DEFAULT_EXACT_WIDE_LIMB_COUNT = 4
EXACT_WIDE_LIMB_VARIANTS = [1, 2, 3, 4, 8, 16, 32]


@dataclass(frozen=True)
class SweepCase:
    name: str
    m: int
    n: int
    k: int
    tile_m: int = 128
    tile_n: int = 128
    bound_mode: str = "global"
    input_profile: str = "uniform-small"
    require_adaptive: bool = False
    promotable: bool = True


@dataclass(frozen=True)
class ScenarioItem:
    family: str
    name: str
    semantics: str
    case: SweepCase
    evidence_scope: str
    output_domain: str
    rationale: str
    backends: tuple[str, ...] | None = None
    pack_mode: str = "per_repeat_repack"
    finite_moduli: tuple[int | None, ...] = (None,)
    exact_wide_limb_counts: tuple[int | None, ...] = (None,)
    residue_chain_length: int = 1
    output_ld_padding: int = 0
    oneshot: bool = False
    prefix_policy: str | None = None
    max_prefix: int | None = None
    bound_source: str | None = None
    include_wrap64_candidate: bool = False


@dataclass(frozen=True)
class SweepCommand:
    name: str
    command: list[str]
    output: Path
    scenario: dict[str, Any] | None = None


def parse_int(text: str, label: str) -> int:
    try:
        value = int(text, 10)
    except ValueError as exc:
        raise SystemExit(f"{label} must be an integer, got {text!r}") from exc
    if value <= 0:
        raise SystemExit(f"{label} must be positive, got {value}")
    return value


def parse_case(value: str, *, adaptive: bool = False, promotable: bool = True) -> SweepCase:
    name = ""
    body = value
    if ":" in value:
        name, body = value.split(":", 1)
    parts = [part.strip() for part in body.replace("x", ",").split(",") if part.strip()]
    min_expected = 5 if adaptive else 3
    max_expected = 6 if adaptive else 3
    if len(parts) < min_expected or len(parts) > max_expected:
        shape = "NAME:M,N,K,TILE_M,TILE_N[,INPUT_PROFILE]" if adaptive else "NAME:M,N,K"
        raise SystemExit(f"case must use {shape}, got {value!r}")
    m = parse_int(parts[0], "m")
    n = parse_int(parts[1], "n")
    k = parse_int(parts[2], "k")
    tile_m = parse_int(parts[3], "tile_m") if adaptive else 128
    tile_n = parse_int(parts[4], "tile_n") if adaptive else 128
    input_profile = parts[5] if adaptive and len(parts) == 6 else "uniform-small"
    if input_profile not in INPUT_PROFILES:
        raise SystemExit(f"adaptive input profile must be one of {sorted(INPUT_PROFILES)}, got {input_profile!r}")
    if not name:
        prefix = "adaptive" if adaptive else "shape"
        name = f"{prefix}-{m}x{n}x{k}"
    return SweepCase(
        name=name,
        m=m,
        n=n,
        k=k,
        tile_m=tile_m,
        tile_n=tile_n,
        bound_mode="per-tile" if adaptive else "global",
        input_profile=input_profile,
        require_adaptive=adaptive,
        promotable=promotable,
    )


def capture_contract_key(capture: dict[str, Any]) -> str:
    tile_bounds = capture.get("tile_bounds_u64")
    tile_hash = tile_bounds.get("hash_u64") if isinstance(tile_bounds, dict) else None
    wrap64_contract = capture.get("semantics") == "wrap_u64_mod_2_64"
    tile_m = "wrap64_semantic_contract" if wrap64_contract else capture.get("tile_m")
    tile_n = "wrap64_semantic_contract" if wrap64_contract else capture.get("tile_n")
    bound_source = capture_bound_source(capture)
    parts = [
        f"semantics={capture.get('semantics')}",
        f"finite_modulus={capture.get('finite_modulus')}",
        f"bound_kind={capture.get('bound_kind')}",
        f"bound_mode={capture.get('bound_mode')}",
        f"bound={capture.get('bound')}",
        f"bound_source={bound_source}",
        f"m={capture.get('m')}",
        f"n={capture.get('n')}",
        f"k={capture.get('k')}",
        f"output_logical_ld={capture.get('output_logical_ld', capture.get('n'))}",
        f"output_ld_padding={capture.get('output_ld_padding', 0)}",
        f"prefix={capture.get('prefix')}",
        f"layout={capture.get('layout')}",
        f"tile_m={tile_m}",
        f"tile_n={tile_n}",
        f"k_block={capture.get('k_block_size')}",
        f"exact_wide_limb_count={capture.get('exact_wide_limb_count')}",
        f"residue_chain_length={capture.get('residue_chain_length', 1)}",
        f"residue_output_mode={capture.get('residue_output_mode', 'host_export')}",
        f"seed={capture.get('seed')}",
        f"input_distribution={capture.get('input_distribution')}",
        f"reuse_packed_inputs={capture.get('reuse_packed_inputs') is True}",
        f"pack_mode={capture_pack_mode(capture)}",
        f"prepack_reuse_strategy={capture_prepack_reuse_strategy(capture)}",
        f"tile_hash={tile_hash}",
    ]
    return ";".join(str(part) for part in parts)


def median_phase(capture: dict[str, Any], phase: str) -> float | None:
    summary = capture.get("timing_summary_us")
    if not isinstance(summary, dict):
        return None
    item = summary.get(phase)
    if not isinstance(item, dict):
        return None
    value = item.get("median")
    return float(value) if isinstance(value, (int, float)) else None


def backend_id(capture: dict[str, Any]) -> str:
    if capture.get("backend_requested") == WRAP64_ROCWMMA_CANDIDATE_BACKEND:
        return WRAP64_ROCWMMA_CANDIDATE_BACKEND
    backend = str(capture.get("backend_selected"))
    if capture_execution_mode(capture) == "public_oneshot_transient_native_inputs":
        return f"{backend}-oneshot"
    return backend


def selected_kernel(capture: dict[str, Any]) -> str:
    kernel = capture.get("selected_kernel")
    return str(kernel) if kernel is not None else ""


def normalized_target_id(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in PLACEHOLDER_TARGET_IDS:
        return None
    return text


def backend_requires_gpu_target(backend: str) -> bool:
    return backend not in {"cpu-reference", "wrap64-byte-limb"}


def normalized_positive_int(value: Any) -> str | None:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return str(value)
    return None


def normalized_identity_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in PLACEHOLDER_TARGET_IDS:
        return None
    return text


def normalized_compiler_identity(capture: dict[str, Any]) -> str | None:
    compiler = capture_compiler(capture)
    compiler_id = normalized_identity_text(compiler.get("id"))
    compiler_version = normalized_identity_text(compiler.get("version"))
    if compiler_id is None or compiler_version is None:
        return None
    return f"{compiler_id} {compiler_version}"


def capture_backend_metadata(capture: dict[str, Any]) -> dict[str, Any]:
    metadata = capture.get("backend_metadata")
    return metadata if isinstance(metadata, dict) else {}


def capture_timing_metadata(capture: dict[str, Any]) -> dict[str, Any]:
    metadata = capture.get("timing_metadata")
    return metadata if isinstance(metadata, dict) else {}


def capture_execution_mode(capture: dict[str, Any]) -> str:
    mode = capture.get("benchmark_execution_mode")
    if not isinstance(mode, str):
        mode = capture_timing_metadata(capture).get("benchmark_execution_mode")
    if isinstance(mode, str):
        return mode
    if capture.get("backend_requested") == WRAP64_ROCWMMA_CANDIDATE_BACKEND:
        return "internal_wrap64_rocwmma_candidate"
    if capture.get("benchmark") in {"rns8_bounded_gemm_public_oneshot", "rns8_finite_u8_public_oneshot"}:
        return "public_oneshot_transient_native_inputs"
    return "persistent_resident_matrices"


def capture_pack_mode(capture: dict[str, Any]) -> str:
    timing = capture_timing_metadata(capture)
    mode = timing.get("pack_mode")
    if mode is None:
        mode = capture.get("pack_mode")
    if isinstance(mode, str):
        return mode
    return "prepacked_reuse" if capture.get("reuse_packed_inputs") is True else "per_repeat_repack"


def capture_prepack_reuse_strategy(capture: dict[str, Any]) -> str:
    timing = capture_timing_metadata(capture)
    strategy = timing.get("prepack_reuse_strategy")
    if strategy is None:
        strategy = capture.get("prepack_reuse_strategy")
    if isinstance(strategy, str):
        return strategy
    return "persistent_matrix_residency" if capture.get("reuse_packed_inputs") is True else "none"


def capture_bound_source(capture: dict[str, Any]) -> str:
    value = capture.get("bound_source")
    return value if isinstance(value, str) else "static_profile"


def requested_pack_mode(args: argparse.Namespace) -> str:
    if getattr(args, "reuse_packed_inputs", False):
        return "prepacked_reuse"
    reuse_a = bool(getattr(args, "reuse_packed_a", False))
    reuse_b = bool(getattr(args, "reuse_packed_b", False))
    if reuse_a and reuse_b:
        return "prepacked_reuse"
    if reuse_a:
        return "prepacked_reuse_a"
    if reuse_b:
        return "prepacked_reuse_b"
    return "per_repeat_repack"


def capture_prepack_reuse_operands(capture: dict[str, Any]) -> tuple[str, ...]:
    operands = capture.get("prepack_reuse_operands")
    if not isinstance(operands, list):
        timing = capture_timing_metadata(capture)
        operands = timing.get("prepack_reuse_operands")
    if isinstance(operands, list):
        return tuple(str(item) for item in operands)
    mode = capture_pack_mode(capture)
    if mode == "prepacked_reuse":
        return ("A", "B")
    if mode == "prepacked_reuse_a":
        return ("A",)
    if mode == "prepacked_reuse_b":
        return ("B",)
    return ()


def capture_device(capture: dict[str, Any]) -> dict[str, Any]:
    device = capture.get("device")
    return device if isinstance(device, dict) else {}


def capture_hip_toolchain(capture: dict[str, Any]) -> dict[str, Any]:
    toolchain = capture.get("hip_toolchain")
    return toolchain if isinstance(toolchain, dict) else {}


def capture_compiler(capture: dict[str, Any]) -> dict[str, Any]:
    compiler = capture.get("compiler")
    return compiler if isinstance(compiler, dict) else {}


def candidate_source_metadata(capture: dict[str, Any]) -> dict[str, Any]:
    metadata = capture_backend_metadata(capture)
    timing = capture_timing_metadata(capture)
    device = capture_device(capture)
    toolchain = capture_hip_toolchain(capture)
    compiler = capture_compiler(capture)
    return {
        "target_id": normalized_target_id(device.get("gcn_arch")),
        "device_name": device.get("name"),
        "hip_runtime_version": device.get("hip_runtime_version"),
        "hip_driver_version": device.get("hip_driver_version"),
        "hip_sdk_or_rocm_version": toolchain.get("hip_sdk_or_rocm_version"),
        "accelerator_library": metadata.get("accelerator_library"),
        "accelerator_version": metadata.get("accelerator_version"),
        "compiler": {"id": compiler.get("id"), "version": compiler.get("version")},
        "configured_amdgpu_targets": capture.get("configured_amdgpu_targets"),
        "git_commit": capture.get("git_commit"),
        "benchmark": capture.get("benchmark"),
        "benchmark_execution_mode": capture_execution_mode(capture),
        "seed": capture.get("seed"),
        "warmups": capture.get("warmups"),
        "repeats": capture.get("repeats"),
        "layout": capture.get("layout"),
        "output_logical_ld": capture.get("output_logical_ld", capture.get("n")),
        "output_ld_padding": capture.get("output_ld_padding", 0),
        "reuse_packed_inputs": capture.get("reuse_packed_inputs") is True,
        "pack_mode": capture_pack_mode(capture),
        "prepack_reuse_strategy": capture_prepack_reuse_strategy(capture),
        "prepack_setup_us": capture.get("prepack_setup_us"),
        "bound_source": capture_bound_source(capture),
        "bound_discovery": capture.get("bound_discovery"),
        "prefix": capture.get("prefix"),
        "selected_prefix": capture.get("selected_prefix"),
        "requested_max_prefix": capture.get("requested_max_prefix"),
        "contract_prefix_policy": capture.get("contract_prefix_policy"),
        "residue_plane_skip_fraction": capture.get("residue_plane_skip_fraction"),
        "k_block_size": capture.get("k_block_size"),
        "tile_m": capture.get("tile_m"),
        "tile_n": capture.get("tile_n"),
        "epilogue": capture.get("epilogue_type"),
        "workspace_bytes": metadata.get("workspace_required_bytes", 0),
        "autotune_key": metadata.get("autotune_key"),
        "event_source": timing.get("gpu_event_timing_source"),
        "event_source_scope": timing.get("gpu_event_timing_source_scope"),
        "event_status": timing.get("gpu_event_timing_status"),
        "event_phase_order": timing.get("gpu_event_phase_order"),
    }


def group_source_metadata(items: list[dict[str, Any]]) -> dict[str, Any]:
    source_items = [candidate_source_metadata(item) for item in items]
    targets = sorted(
        {
            str(item.get("target_id"))
            for item in source_items
            if item.get("target_id")
        }
    )
    hip_versions = sorted(
        {
            str(item.get("hip_sdk_or_rocm_version"))
            for item in source_items
            if item.get("hip_sdk_or_rocm_version")
        }
    )
    accelerator_versions = sorted(
        {
            str(item.get("accelerator_version"))
            for item in source_items
            if item.get("accelerator_version")
        }
    )
    configured_targets = sorted(
        {
            str(item.get("configured_amdgpu_targets"))
            for item in source_items
            if item.get("configured_amdgpu_targets")
        }
    )
    runtime_versions = sorted(
        {
            str(item.get("hip_runtime_version"))
            for item in source_items
            if item.get("hip_runtime_version")
        }
    )
    driver_versions = sorted(
        {
            str(item.get("hip_driver_version"))
            for item in source_items
            if item.get("hip_driver_version")
        }
    )
    compilers = sorted(
        {
            f"{compiler.get('id')} {compiler.get('version')}"
            for item in items
            if (compiler := capture_compiler(item)).get("id") or compiler.get("version")
        }
    )
    return {
        "target_ids": targets,
        "hip_sdk_or_rocm_versions": hip_versions,
        "accelerator_versions": accelerator_versions,
        "configured_amdgpu_targets": configured_targets,
        "hip_runtime_versions": runtime_versions,
        "hip_driver_versions": driver_versions,
        "compilers": compilers,
        "git_commits": sorted({str(item.get("git_commit")) for item in items if item.get("git_commit")}),
        "seeds": sorted({int(item.get("seed")) for item in items if isinstance(item.get("seed"), int)}),
        "warmups": sorted({int(item.get("warmups")) for item in items if isinstance(item.get("warmups"), int)}),
        "repeats": sorted({int(item.get("repeats")) for item in items if isinstance(item.get("repeats"), int)}),
        "reuse_packed_inputs": sorted({bool(item.get("reuse_packed_inputs") is True) for item in items}),
        "benchmark_execution_modes": sorted({capture_execution_mode(item) for item in items}),
        "pack_modes": sorted({capture_pack_mode(item) for item in items}),
        "prepack_reuse_strategies": sorted({capture_prepack_reuse_strategy(item) for item in items}),
        "prepack_reuse_operands": sorted({"/".join(capture_prepack_reuse_operands(item)) or "none" for item in items}),
        "event_sources": sorted(
            {
                str(capture_timing_metadata(item).get("gpu_event_timing_source") or "unavailable")
                for item in items
            }
        ),
    }


def cli_backend(backend: str) -> str:
    if backend == "hip-vector-alu-int64":
        return "hip-vector-alu-int64-runtime"
    return backend


def normalize_semantics(value: str) -> str:
    aliases = {
        "bounded_i64": "bounded-i64",
        "bounded_u64": "bounded-u64",
        "exact_wide_signed": "exact-wide-signed",
        "exact-wide-i64": "exact-wide-signed",
        "exact_wide_unsigned": "exact-wide-unsigned",
        "exact-wide-u64": "exact-wide-unsigned",
        "wrap_u64_mod_2_64": "wrap-u64",
        "finite-ring-u8": "finite-u8-ring",
        "finite_ring_u8": "finite-u8-ring",
        "finite-field-u8": "finite-u8-field",
        "finite_field_u8": "finite-u8-field",
    }
    return aliases.get(value, value)


def parse_backend_bench(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"--bench-for must use BACKEND=PATH, got {value!r}")
        backend, path = value.split("=", 1)
        if not backend or not path:
            raise SystemExit(f"--bench-for must use BACKEND=PATH, got {value!r}")
        result[backend] = Path(path)
    return result


def autotune_cache_path() -> Path:
    override = os.environ.get("RNS8_AUTOTUNE_CACHE_PATH")
    if override:
        return Path(override)
    if os.name == "nt":
        root = os.environ.get("LOCALAPPDATA") or os.environ.get("USERPROFILE")
        if root:
            suffix = ["rns8-gemm", "autotune.json"]
            if root == os.environ.get("USERPROFILE"):
                suffix = ["AppData", "Local", *suffix]
            return Path(root).joinpath(*suffix)
    root = os.environ.get("XDG_CACHE_HOME") or os.environ.get("HOME")
    if root:
        return Path(root) / ".cache" / "rns8-gemm" / "autotune.json"
    return Path("rns8-gemm") / "autotune.json"


def run_command(command: list[str], output: Path) -> bool:
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if completed.returncode != 0:
        failure = {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        output.with_suffix(".failed.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
        return False
    output.write_text(completed.stdout, encoding="utf-8")
    return True


def validate_paths(paths: list[Path]) -> list[dict[str, Any]]:
    captures: list[dict[str, Any]] = []
    for path in paths:
        try:
            data = load_capture(path)
            validate_capture(data, path)
        except BenchmarkSchemaError as exc:
            raise SystemExit(str(exc)) from exc
        data["_path"] = str(path)
        captures.append(data)
    return captures


def required_baselines(semantics: Any) -> list[str]:
    if semantics in {"bounded_i64", "bounded_u64"}:
        return ["cpu-reference", "hip-direct", "hip-vector-alu-int64"]
    if semantics in {"finite_ring_u8", "finite_field_u8"}:
        return ["cpu-reference", "hip-direct"]
    if semantics in {"exact_wide_signed", "exact_wide_unsigned"}:
        return ["cpu-reference", "hip-direct"]
    if semantics == "wrap_u64_mod_2_64":
        return ["wrap64-byte-limb", "hip-direct"]
    return []


def phase_ratios(item: dict[str, Any], direct: dict[str, Any] | None, vector: dict[str, Any] | None) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for phase in PHASES:
        value = median_phase(item, phase)
        direct_value = median_phase(direct, phase) if direct else None
        vector_value = median_phase(vector, phase) if vector else None
        result[phase] = {
            "median_us": value,
            "speedup_vs_direct_hip": (direct_value / value) if direct_value and value else None,
            "speedup_vs_vector_alu": (vector_value / value) if vector_value and value else None,
        }
    return result


def release_capture_satisfied(capture: dict[str, Any]) -> bool:
    return (
        isinstance(capture.get("warmups"), int)
        and isinstance(capture.get("repeats"), int)
        and capture["warmups"] >= RELEASE_MIN_WARMUPS
        and capture["repeats"] >= RELEASE_MIN_REPEATS
    )


def promotion_blockers(
    *,
    missing: list[str],
    semantics: Any,
    release_review_satisfied: bool,
    gpu_target_identity_complete: bool,
    gpu_target_compatible: bool,
    configured_target_identity_complete: bool,
    configured_target_compatible: bool,
    hip_toolchain_version_complete: bool,
    hip_toolchain_version_compatible: bool,
    hip_runtime_version_complete: bool,
    hip_runtime_version_compatible: bool,
    hip_driver_version_complete: bool,
    hip_driver_version_compatible: bool,
    compiler_identity_complete: bool,
    compiler_identity_compatible: bool,
    git_commit_identity_complete: bool,
    git_commit_identity_compatible: bool,
    warmup_count_complete: bool,
    warmup_count_compatible: bool,
    repeat_count_complete: bool,
    repeat_count_compatible: bool,
    duplicate_backends: list[str],
    accelerator: bool,
    internal_candidate: bool,
    prepacked_reuse: bool,
    oneshot_capture: bool,
    end_to_end: float | None,
    direct: float | None,
    vector: float | None,
) -> list[str]:
    blockers: list[str] = []
    if missing:
        blockers.append("missing_required_baselines")
    if not release_review_satisfied:
        blockers.append("not_release_review")
    if not gpu_target_identity_complete:
        blockers.append("missing_gpu_target_id")
    elif not gpu_target_compatible:
        blockers.append("gpu_target_mismatch")
    if not configured_target_identity_complete:
        blockers.append("missing_configured_gpu_target")
    elif not configured_target_compatible:
        blockers.append("configured_gpu_target_mismatch")
    if not hip_toolchain_version_complete:
        blockers.append("missing_hip_toolchain_version")
    elif not hip_toolchain_version_compatible:
        blockers.append("hip_toolchain_version_mismatch")
    if not hip_runtime_version_complete:
        blockers.append("missing_hip_runtime_version")
    elif not hip_runtime_version_compatible:
        blockers.append("hip_runtime_version_mismatch")
    if not hip_driver_version_complete:
        blockers.append("missing_hip_driver_version")
    elif not hip_driver_version_compatible:
        blockers.append("hip_driver_version_mismatch")
    if not compiler_identity_complete:
        blockers.append("missing_compiler_identity")
    elif not compiler_identity_compatible:
        blockers.append("compiler_identity_mismatch")
    if not git_commit_identity_complete:
        blockers.append("missing_git_commit")
    elif not git_commit_identity_compatible:
        blockers.append("git_commit_mismatch")
    if not warmup_count_complete:
        blockers.append("missing_warmup_count")
    elif not warmup_count_compatible:
        blockers.append("warmup_count_mismatch")
    if not repeat_count_complete:
        blockers.append("missing_repeat_count")
    elif not repeat_count_compatible:
        blockers.append("repeat_count_mismatch")
    if duplicate_backends:
        blockers.append("duplicate_backend_capture")
    if not accelerator:
        blockers.append("not_accelerator_backend")
    if internal_candidate:
        blockers.append("internal_candidate_not_public_backend")
    if prepacked_reuse:
        blockers.append("prepacked_reuse_not_autotune_promotable")
    if oneshot_capture:
        blockers.append("oneshot_api_capture_not_autotune_promotable")
    if end_to_end is None:
        blockers.append("missing_end_to_end_timing")
    if direct is None:
        blockers.append("missing_direct_hip_timing")
    elif end_to_end is not None and end_to_end >= direct:
        blockers.append("not_faster_than_direct_hip")
    if vector is not None and end_to_end is not None and end_to_end >= vector:
        blockers.append("not_faster_than_vector_alu")
    return blockers


def primary_loss_phase(item: dict[str, Any], direct: dict[str, Any] | None) -> str | None:
    if direct is None:
        return None
    worst_phase = None
    worst_ratio = 0.0
    for phase in PHASES:
        value = median_phase(item, phase)
        baseline = median_phase(direct, phase)
        if value is None or baseline is None or baseline <= 0:
            continue
        ratio = value / baseline
        if ratio > worst_ratio:
            worst_ratio = ratio
            worst_phase = phase
    return worst_phase


def bottleneck_classification(capture: dict[str, Any]) -> dict[str, Any]:
    end_to_end = median_phase(capture, "end_to_end")
    phase_values = {
        phase: value
        for phase in ("pack", "rns_gemm", "crt_export")
        if (value := median_phase(capture, phase)) is not None and value > 0
    }
    if not end_to_end or end_to_end <= 0 or not phase_values:
        return {"class": "unknown", "phase": None, "share": None}
    shares = {phase: value / end_to_end for phase, value in phase_values.items()}
    overhead_share = max(0.0, end_to_end - sum(phase_values.values())) / end_to_end
    phase, share = max(shares.items(), key=lambda item: item[1])
    if overhead_share >= 0.25 and overhead_share > share:
        return {"class": "launch_or_api_bound", "phase": "unattributed_overhead", "share": overhead_share}
    if share < 0.40:
        return {"class": "mixed_bound", "phase": phase, "share": share}
    return {
        "class": {"pack": "pack_bound", "rns_gemm": "compute_bound", "crt_export": "export_bound"}[phase],
        "phase": phase,
        "share": share,
    }


def review_captures(captures: list[dict[str, Any]], *, review_mode: str = "smoke") -> dict[str, Any]:
    if review_mode not in {"smoke", "release"}:
        raise ValueError(f"unsupported review mode: {review_mode}")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for capture in captures:
        grouped[capture_contract_key(capture)].append(capture)

    groups = []
    promotable_entries = []
    for key, items in sorted(grouped.items()):
        backend_counts: dict[str, int] = defaultdict(int)
        for item in items:
            backend_counts[backend_id(item)] += 1
        duplicate_backends = sorted(backend for backend, count in backend_counts.items() if count > 1)
        by_backend = {backend_id(item): item for item in items}
        semantics = items[0].get("semantics")
        required = required_baselines(semantics)
        missing = [backend for backend in required if backend not in by_backend]
        direct_capture = by_backend.get("hip-direct")
        vector_capture = by_backend.get("hip-vector-alu-int64")
        phase_medians = {
            f"{backend_id(item)}/{selected_kernel(item)}": {phase: median_phase(item, phase) for phase in PHASES}
            for item in items
        }
        gpu_targets = {
            backend: normalized_target_id(capture.get("device", {}).get("gcn_arch"))
            for backend, capture in by_backend.items()
            if backend_requires_gpu_target(backend)
        }
        missing_gpu_targets = sorted(backend for backend, target in gpu_targets.items() if target is None)
        gpu_target_identity_complete = not missing_gpu_targets
        gpu_target_values = {value for value in gpu_targets.values() if value}
        gpu_target_compatible = gpu_target_identity_complete and len(gpu_target_values) <= 1
        configured_gpu_targets = {
            backend: normalized_target_id(capture.get("configured_amdgpu_targets"))
            for backend, capture in by_backend.items()
            if backend_requires_gpu_target(backend)
        }
        missing_configured_gpu_targets = sorted(
            backend for backend, target in configured_gpu_targets.items() if target is None
        )
        configured_target_identity_complete = not missing_configured_gpu_targets
        configured_target_values = {target for target in configured_gpu_targets.values() if target}
        configured_target_compatible = configured_target_identity_complete and len(configured_target_values) <= 1
        hip_toolchain_versions = {
            backend: normalized_target_id(capture_hip_toolchain(capture).get("hip_sdk_or_rocm_version"))
            for backend, capture in by_backend.items()
            if backend_requires_gpu_target(backend)
        }
        missing_hip_toolchain_versions = sorted(
            backend for backend, version in hip_toolchain_versions.items() if version is None
        )
        hip_toolchain_version_complete = not missing_hip_toolchain_versions
        hip_toolchain_version_values = {version for version in hip_toolchain_versions.values() if version}
        hip_toolchain_version_compatible = (
            hip_toolchain_version_complete and len(hip_toolchain_version_values) <= 1
        )
        hip_runtime_versions = {
            backend: normalized_positive_int(capture_device(capture).get("hip_runtime_version"))
            for backend, capture in by_backend.items()
            if backend_requires_gpu_target(backend)
        }
        missing_hip_runtime_versions = sorted(
            backend for backend, version in hip_runtime_versions.items() if version is None
        )
        hip_runtime_version_complete = not missing_hip_runtime_versions
        hip_runtime_version_values = {version for version in hip_runtime_versions.values() if version}
        hip_runtime_version_compatible = hip_runtime_version_complete and len(hip_runtime_version_values) <= 1
        hip_driver_versions = {
            backend: normalized_positive_int(capture_device(capture).get("hip_driver_version"))
            for backend, capture in by_backend.items()
            if backend_requires_gpu_target(backend)
        }
        missing_hip_driver_versions = sorted(
            backend for backend, version in hip_driver_versions.items() if version is None
        )
        hip_driver_version_complete = not missing_hip_driver_versions
        hip_driver_version_values = {version for version in hip_driver_versions.values() if version}
        hip_driver_version_compatible = hip_driver_version_complete and len(hip_driver_version_values) <= 1
        compiler_identities = {backend: normalized_compiler_identity(capture) for backend, capture in by_backend.items()}
        missing_compiler_identities = sorted(
            backend for backend, identity in compiler_identities.items() if identity is None
        )
        compiler_identity_complete = not missing_compiler_identities
        compiler_identity_values = {identity for identity in compiler_identities.values() if identity}
        compiler_identity_compatible = compiler_identity_complete and len(compiler_identity_values) <= 1
        git_commits = {
            backend: normalized_identity_text(capture.get("git_commit")) for backend, capture in by_backend.items()
        }
        missing_git_commits = sorted(backend for backend, commit in git_commits.items() if commit is None)
        git_commit_identity_complete = not missing_git_commits
        git_commit_values = {commit for commit in git_commits.values() if commit}
        git_commit_identity_compatible = git_commit_identity_complete and len(git_commit_values) <= 1
        warmup_counts = {backend: normalized_positive_int(capture.get("warmups")) for backend, capture in by_backend.items()}
        missing_warmup_counts = sorted(backend for backend, count in warmup_counts.items() if count is None)
        warmup_count_complete = not missing_warmup_counts
        warmup_count_values = {count for count in warmup_counts.values() if count}
        warmup_count_compatible = warmup_count_complete and len(warmup_count_values) <= 1
        repeat_counts = {backend: normalized_positive_int(capture.get("repeats")) for backend, capture in by_backend.items()}
        missing_repeat_counts = sorted(backend for backend, count in repeat_counts.items() if count is None)
        repeat_count_complete = not missing_repeat_counts
        repeat_count_values = {count for count in repeat_counts.values() if count}
        repeat_count_compatible = repeat_count_complete and len(repeat_count_values) <= 1
        release_review_satisfied = review_mode == "release" and all(release_capture_satisfied(item) for item in items)
        candidates = []
        for item in items:
            backend = backend_id(item)
            metadata = capture_backend_metadata(item)
            accelerator = metadata.get("accelerator_backend") is True
            internal_candidate = backend == WRAP64_ROCWMMA_CANDIDATE_BACKEND
            oneshot_capture = capture_execution_mode(item) == "public_oneshot_transient_native_inputs"
            end_to_end = median_phase(item, "end_to_end")
            direct = median_phase(direct_capture, "end_to_end") if direct_capture else None
            vector = median_phase(vector_capture, "end_to_end") if vector_capture else None
            blockers = promotion_blockers(
                missing=missing,
                semantics=semantics,
                release_review_satisfied=release_review_satisfied,
                gpu_target_identity_complete=gpu_target_identity_complete,
                gpu_target_compatible=gpu_target_compatible,
                configured_target_identity_complete=configured_target_identity_complete,
                configured_target_compatible=configured_target_compatible,
                hip_toolchain_version_complete=hip_toolchain_version_complete,
                hip_toolchain_version_compatible=hip_toolchain_version_compatible,
                hip_runtime_version_complete=hip_runtime_version_complete,
                hip_runtime_version_compatible=hip_runtime_version_compatible,
                hip_driver_version_complete=hip_driver_version_complete,
                hip_driver_version_compatible=hip_driver_version_compatible,
                compiler_identity_complete=compiler_identity_complete,
                compiler_identity_compatible=compiler_identity_compatible,
                git_commit_identity_complete=git_commit_identity_complete,
                git_commit_identity_compatible=git_commit_identity_compatible,
                warmup_count_complete=warmup_count_complete,
                warmup_count_compatible=warmup_count_compatible,
                repeat_count_complete=repeat_count_complete,
                repeat_count_compatible=repeat_count_compatible,
                duplicate_backends=duplicate_backends,
                accelerator=accelerator,
                internal_candidate=internal_candidate,
                prepacked_reuse=capture_pack_mode(item) != "per_repeat_repack",
                oneshot_capture=oneshot_capture,
                end_to_end=end_to_end,
                direct=direct,
                vector=vector if semantics in {"bounded_i64", "bounded_u64"} else None,
            )
            promotable = not blockers
            candidate = {
                "backend": backend,
                "selected_kernel": selected_kernel(item),
                "capture": item.get("_path"),
                "source_metadata": candidate_source_metadata(item),
                "accelerator_backend": accelerator,
                "release_review_capture": release_capture_satisfied(item),
                "median_end_to_end_us": end_to_end,
                "phase_diagnostics": phase_ratios(item, direct_capture, vector_capture),
                "speedup_vs_direct_hip": (direct / end_to_end) if direct and end_to_end else None,
                "speedup_vs_vector_alu": (vector / end_to_end) if vector and end_to_end else None,
                "promotable": promotable,
                "promotion_blockers": blockers,
                "promotion_reason": "beats_required_same_contract_gpu_baselines" if promotable else "blocked",
                "primary_loss_phase_vs_direct_hip": None if promotable else primary_loss_phase(item, direct_capture),
                "bottleneck": bottleneck_classification(item),
                "cache_write_status": "eligible_after_review" if promotable else "not_eligible",
            }
            candidates.append(candidate)

        fastest = None
        promotable_candidates = [item for item in candidates if item["promotable"]]
        if promotable_candidates:
            fastest = min(promotable_candidates, key=lambda item: item["median_end_to_end_us"])
            for item in candidates:
                if item is not fastest and item["promotable"]:
                    item["promotable"] = False
                    item["promotion_blockers"] = ["not_fastest_promotable_accelerator"]
                    item["promotion_reason"] = "blocked"
                    item["cache_write_status"] = "not_eligible"
            source = by_backend.get(fastest["backend"])
            if source is not None:
                metadata = source.get("backend_metadata") if isinstance(source.get("backend_metadata"), dict) else {}
                promotable_entries.append(
                    {
                        "source_capture": source.get("_path"),
                        "autotune_key": metadata.get("autotune_key"),
                        "selected_backend": fastest["backend"],
                        "selected_kernel": fastest["selected_kernel"],
                        "median_end_to_end_us": fastest["median_end_to_end_us"],
                        "target_id": candidate_source_metadata(source).get("target_id"),
                        "hip_sdk_or_rocm_version": candidate_source_metadata(source).get("hip_sdk_or_rocm_version"),
                        "accelerator_library": metadata.get("accelerator_library"),
                        "accelerator_version": metadata.get("accelerator_version"),
                        "workspace_bytes": metadata.get("workspace_required_bytes", 0),
                        "winner_rationale": "fastest_promotable_same_contract_accelerator",
                        "cache_write_status": "pending",
                    }
                )

        groups.append(
            {
                "contract_key": key,
                "semantics": semantics,
                "finite_modulus": items[0].get("finite_modulus"),
                "shape": {"m": items[0].get("m"), "n": items[0].get("n"), "k": items[0].get("k")},
                "capture_count": len(items),
                "source_metadata": group_source_metadata(items),
                "review_mode": review_mode,
                "release_review_satisfied": release_review_satisfied,
                "release_review_requirements": {
                    "min_warmups": RELEASE_MIN_WARMUPS,
                    "min_repeats": RELEASE_MIN_REPEATS,
                },
                "required_baselines": required,
                "missing_required_baselines": missing,
                "gpu_targets": gpu_targets,
                "missing_gpu_targets": missing_gpu_targets,
                "gpu_target_identity_complete": gpu_target_identity_complete,
                "gpu_target_compatible": gpu_target_compatible,
                "configured_gpu_targets": configured_gpu_targets,
                "missing_configured_gpu_targets": missing_configured_gpu_targets,
                "configured_target_identity_complete": configured_target_identity_complete,
                "configured_target_compatible": configured_target_compatible,
                "hip_toolchain_versions": hip_toolchain_versions,
                "missing_hip_toolchain_versions": missing_hip_toolchain_versions,
                "hip_toolchain_version_complete": hip_toolchain_version_complete,
                "hip_toolchain_version_compatible": hip_toolchain_version_compatible,
                "hip_runtime_versions": hip_runtime_versions,
                "missing_hip_runtime_versions": missing_hip_runtime_versions,
                "hip_runtime_version_complete": hip_runtime_version_complete,
                "hip_runtime_version_compatible": hip_runtime_version_compatible,
                "hip_driver_versions": hip_driver_versions,
                "missing_hip_driver_versions": missing_hip_driver_versions,
                "hip_driver_version_complete": hip_driver_version_complete,
                "hip_driver_version_compatible": hip_driver_version_compatible,
                "compiler_identities": compiler_identities,
                "missing_compiler_identities": missing_compiler_identities,
                "compiler_identity_complete": compiler_identity_complete,
                "compiler_identity_compatible": compiler_identity_compatible,
                "git_commits": git_commits,
                "missing_git_commits": missing_git_commits,
                "git_commit_identity_complete": git_commit_identity_complete,
                "git_commit_identity_compatible": git_commit_identity_compatible,
                "warmup_counts": warmup_counts,
                "missing_warmup_counts": missing_warmup_counts,
                "warmup_count_complete": warmup_count_complete,
                "warmup_count_compatible": warmup_count_compatible,
                "repeat_counts": repeat_counts,
                "missing_repeat_counts": missing_repeat_counts,
                "repeat_count_complete": repeat_count_complete,
                "repeat_count_compatible": repeat_count_compatible,
                "duplicate_backends": duplicate_backends,
                "phase_medians_us": phase_medians,
                "fastest_promotable": fastest,
                "candidates": candidates,
            }
        )

    return {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "review_mode": review_mode,
        "release_review_requirements": {
            "min_warmups": RELEASE_MIN_WARMUPS,
            "min_repeats": RELEASE_MIN_REPEATS,
        },
        "reviewed_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "group_count": len(groups),
        "groups": groups,
        "promotable_autotune_entries": promotable_entries,
        "cache_write": {
            "requested": False,
            "path": None,
            "entries_written": 0,
            "status": "not_requested",
        },
    }


def cache_entry_from_capture(capture: dict[str, Any], validation_status: str) -> dict[str, Any]:
    metadata = capture.get("backend_metadata") if isinstance(capture.get("backend_metadata"), dict) else {}
    medians = capture.get("timing_summary_us") if isinstance(capture.get("timing_summary_us"), dict) else {}
    schedule = capture.get("schedule_metadata") if isinstance(capture.get("schedule_metadata"), dict) else {}
    tile_bounds = capture.get("tile_bounds_u64") if isinstance(capture.get("tile_bounds_u64"), dict) else {}
    selected_prefix = capture.get("selected_prefix", schedule.get("max_selected_prefix"))
    requested_max_prefix = capture.get("requested_max_prefix", capture.get("prefix"))
    prefix_policy = capture.get("contract_prefix_policy", "legacy_v4_unspecified")
    bound_source = capture_bound_source(capture)
    prefix_schedule_hash = (
        f"tile_rows={schedule.get('tile_rows')};tile_cols={schedule.get('tile_cols')};"
        f"selected_prefix={selected_prefix};requested_max_prefix={requested_max_prefix};"
        f"prefix_policy={prefix_policy};bound_source={bound_source};"
        f"groups={schedule.get('prefix_group_count')};"
        f"adaptive_prefix={int(bool(schedule.get('adaptive_prefix_active')))};"
        f"adaptive_skip={int(bool(schedule.get('adaptive_skip_active')))};"
        f"schedule_flags={schedule.get('flags', 0)};"
        f"zero_output_tiles={schedule.get('zero_output_tile_count', 0)};"
        f"zero_a_rows={schedule.get('zero_a_row_proof_count', 0)};"
        f"zero_b_cols={schedule.get('zero_b_col_proof_count', 0)};"
        f"zero_row_col_products={schedule.get('zero_row_col_product_count', 0)};"
        f"tile_bound_hash={tile_bounds.get('hash_u64', 0)}"
    )
    hip_toolchain = capture.get("hip_toolchain") if isinstance(capture.get("hip_toolchain"), dict) else {}
    device = capture.get("device") if isinstance(capture.get("device"), dict) else {}
    version = metadata.get("accelerator_version") or hip_toolchain.get("hip_sdk_or_rocm_version") or "unknown"

    def median(phase: str) -> float:
        item = medians.get(phase) if isinstance(medians, dict) else None
        if isinstance(item, dict) and isinstance(item.get("median"), (int, float)):
            return float(item["median"])
        return 0.0

    return {
        "key": metadata.get("autotune_key"),
        "selected_backend": capture.get("backend_selected"),
        "selected_kernel": capture.get("selected_kernel"),
        "target_id": normalized_target_id(device.get("gcn_arch")) or "cpu",
        "hip_sdk_or_library_version": version,
        "semantic_contract": capture.get("semantics"),
        "finite_modulus": capture.get("finite_modulus") or 0,
        "shape": {"m": capture.get("m"), "n": capture.get("n"), "k": capture.get("k")},
        "layout": capture.get("layout"),
        "prefix_schedule_hash": prefix_schedule_hash,
        "k_block_size": capture.get("k_block_size"),
        "tile_m": capture.get("tile_m"),
        "tile_n": capture.get("tile_n"),
        "epilogue": metadata.get("epilogue_mode"),
        "kernel_family": metadata.get("selected_kernel") or capture.get("selected_kernel"),
        "workspace_bytes": metadata.get("workspace_required_bytes", 0),
        "measured_medians_us": {
            "pack": median("pack"),
            "rns_gemm": median("rns_gemm"),
            "crt_export": median("crt_export"),
            "end_to_end": median("end_to_end"),
        },
        "performance_validated": True,
        "validation_status": validation_status,
        "schema_version": 1,
        "updated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def write_promoted_cache_entries(report: dict[str, Any], captures: list[dict[str, Any]], path: Path) -> int:
    promotable = report.get("promotable_autotune_entries")
    if not isinstance(promotable, list) or not promotable:
        return 0
    by_path = {str(capture.get("_path")): capture for capture in captures}
    entries = []
    for item in promotable:
        if not isinstance(item, dict):
            continue
        capture = by_path.get(str(item.get("source_capture")))
        if not capture:
            continue
        entry = cache_entry_from_capture(capture, "reviewed_release_same_contract_fastest_windows_gfx1100")
        if entry.get("key"):
            entries.append(entry)
    if not entries:
        return 0

    existing: dict[str, Any] = {"schema_version": 1, "entries": []}
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(existing, dict):
            existing = {"schema_version": 1, "entries": []}
    existing_entries = existing.get("entries")
    if not isinstance(existing_entries, list):
        existing_entries = []
    by_key = {entry.get("key"): entry for entry in existing_entries if isinstance(entry, dict) and entry.get("key")}
    for entry in entries:
        by_key[entry["key"]] = entry
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"schema_version": 1, "entries": list(by_key.values())}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for item in promotable:
        if isinstance(item, dict):
            item["cache_write_status"] = "written"
    return len(entries)


def attach_cache_write_status(report: dict[str, Any], requested: bool, path: Path, entries_written: int) -> None:
    report["cache_write"] = {
        "requested": requested,
        "path": str(path) if requested else None,
        "entries_written": entries_written,
        "status": "written" if requested and entries_written else "no_promotable_entries" if requested else "not_requested",
    }
    if not requested:
        for item in report.get("promotable_autotune_entries", []):
            if isinstance(item, dict):
                item["cache_write_status"] = "not_requested"
        return
    written_sources = {
        str(item.get("source_capture"))
        for item in report.get("promotable_autotune_entries", [])
        if isinstance(item, dict) and item.get("cache_write_status") == "written" and item.get("source_capture")
    }
    for group in report.get("groups", []):
        if not isinstance(group, dict):
            continue
        for candidate in group.get("candidates", []):
            if not isinstance(candidate, dict):
                continue
            if str(candidate.get("capture")) in written_sources:
                candidate["cache_write_status"] = "written"
            elif candidate.get("promotable"):
                candidate["cache_write_status"] = "pending"


def default_cases(args: argparse.Namespace) -> list[SweepCase]:
    if args.case:
        return [parse_case(value) for value in args.case]
    if args.shapes:
        return [parse_case(f"{shape},{shape},{shape}") for shape in args.shapes]
    shapes = PROMOTABLE_RELEASE_SHAPES if args.release_matrix else [64, 128]
    if args.include_exploratory_large:
        shapes = [*shapes, *EXPLORATORY_RELEASE_SHAPES]
    return [parse_case(f"{shape},{shape},{shape}", promotable=shape in PROMOTABLE_RELEASE_SHAPES) for shape in shapes]


def adaptive_cases(args: argparse.Namespace) -> list[SweepCase]:
    cases: list[SweepCase] = []
    if args.adaptive_case:
        cases.extend(parse_case(value, adaptive=True) for value in args.adaptive_case)
    if args.include_default_adaptive:
        cases.extend(parse_case(value, adaptive=True) for value in DEFAULT_ADAPTIVE_CASES)
    if args.include_adaptive_workloads:
        cases.extend(parse_case(value, adaptive=True) for value in ADAPTIVE_WORKLOAD_CASES)
    return cases


def wrap64_cases(args: argparse.Namespace) -> list[SweepCase]:
    if args.case:
        return [parse_case(value) for value in args.case]
    shapes = PROMOTABLE_RELEASE_SHAPES if args.release_matrix else [64]
    if args.include_exploratory_large:
        shapes = [*shapes, *EXPLORATORY_RELEASE_SHAPES]
    return [
        parse_case(f"wrap64-{shape}:{shape},{shape},{shape}", promotable=shape in PROMOTABLE_RELEASE_SHAPES)
        for shape in shapes
    ]


def wrap64_backends_for(args: argparse.Namespace) -> list[str]:
    backends = list(WRAP64_BACKENDS)
    if args.include_wrap64_rocwmma_candidate:
        backends.append(WRAP64_ROCWMMA_CANDIDATE_BACKEND)
    return backends


def finite_moduli_for(semantics: str, args: argparse.Namespace) -> list[int | None]:
    if semantics == "finite-u8-ring":
        return args.modulus or DEFAULT_FINITE_RING_MODULI
    if semantics == "finite-u8-field":
        return args.modulus or DEFAULT_FINITE_FIELD_MODULI
    return [None]


def exact_wide_limb_counts_for(semantics: str, args: argparse.Namespace) -> list[int | None]:
    if semantics not in EXACT_WIDE_SEMANTICS:
        return [None]
    if args.exact_wide_limbs:
        counts = args.exact_wide_limbs
    elif args.include_exact_wide_limb_variants:
        counts = EXACT_WIDE_LIMB_VARIANTS
    else:
        counts = [DEFAULT_EXACT_WIDE_LIMB_COUNT]
    invalid = [count for count in counts if count < 1 or count > 32]
    if invalid:
        raise SystemExit(f"--exact-wide-limbs values must be in [1, 32], got {invalid}")
    return list(dict.fromkeys(counts))


def scenario_catalog() -> dict[str, list[ScenarioItem]]:
    repeated_b_512 = parse_case("bounded-i64-512:512,512,512")
    repeated_b_1024 = parse_case("bounded-i64-1024:1024,1024,1024")
    exact_512 = parse_case("exact-wide-512:512,512,512")
    finite_128 = parse_case("finite-128:128,128,128")
    finite_512 = parse_case("finite-512:512,512,512")
    chain_128 = parse_case("chain-128:128,128,128")
    small_64 = parse_case("small-64:64,64,64")
    small_128 = parse_case("small-128:128,128,128")
    skinny_512 = parse_case("gemv-n1-512:512,1,512")
    skinny_1024 = parse_case("gemv-n1-1024:1024,1,1024")
    wrap64_512 = parse_case("wrap64-512:512,512,512")
    wrap64_1024 = parse_case("wrap64-1024:1024,1024,1024")
    large_2048 = parse_case("large-2048:2048,2048,2048", promotable=False)
    adaptive_256 = parse_case("adaptive-bands-256:256,256,512,64,64,adaptive-bands", adaptive=True)
    adaptive_rect = parse_case("adaptive-bands-rect:512,1024,512,128,128,adaptive-bands", adaptive=True)
    adaptive_1024 = parse_case("adaptive-bands-1024:1024,1024,1024,128,128,adaptive-bands", adaptive=True)

    bounded_gpu_backends = ("hip-direct", "hip-vector-alu-int64", "hipblaslt", "ck", "rocwmma")
    bounded_per_tile_backends = ("hip-direct", "hip-vector-alu-int64", "ck", "rocwmma")
    direct_oneshot_backends = ("cpu", "hip-direct")
    accelerator_backends = ("hip-direct", "hipblaslt", "ck", "rocwmma")

    return {
        "adaptive-bands": [
            ScenarioItem(
                "adaptive-bands",
                "bounded-i64-256",
                "bounded-i64",
                adaptive_256,
                "profile-driven per-tile bounded i64 adaptive prefix deletion",
                "host_export",
                "proves whether adaptive scans and active-prefix schedules delete enough residue work to beat fixed-prefix baselines",
                backends=bounded_per_tile_backends,
                bound_source="input-scan",
            ),
            ScenarioItem(
                "adaptive-bands",
                "bounded-u64-rect",
                "bounded-u64",
                adaptive_rect,
                "rectangular profile-driven bounded u64 adaptive prefix deletion",
                "host_export",
                "keeps adaptive evidence from being square-only and exposes row/column zero-summary behavior",
                backends=bounded_per_tile_backends,
                bound_source="input-scan",
            ),
            ScenarioItem(
                "adaptive-bands",
                "bounded-i64-1024",
                "bounded-i64",
                adaptive_1024,
                "large profile-driven bounded i64 adaptive prefix deletion",
                "host_export",
                "checks whether adaptive scheduling remains useful once launch overhead is less dominant",
                backends=bounded_per_tile_backends,
                bound_source="input-scan",
            ),
        ],
        "repeated-b": [
            ScenarioItem(
                "repeated-b",
                "bounded-i64-512",
                "bounded-i64",
                repeated_b_512,
                "same B operand reused across measured repeats",
                "host_export",
                "measures whether setup cost amortizes for the current repeated-B implementation surfaces",
                backends=bounded_gpu_backends,
                pack_mode="prepacked_reuse_b",
                prefix_policy="fixed-requested",
                max_prefix=9,
            ),
            ScenarioItem(
                "repeated-b",
                "bounded-i64-1024",
                "bounded-i64",
                repeated_b_1024,
                "same B operand reused across measured repeats",
                "host_export",
                "keeps hipBLASLt and rocWMMA repeated-B evidence tied to the current 1024 winner split",
                backends=bounded_gpu_backends,
                pack_mode="prepacked_reuse_b",
                prefix_policy="fixed-requested",
                max_prefix=9,
            ),
        ],
        "exact-wide-export": [
            ScenarioItem(
                "exact-wide-export",
                "signed-limbs4-512",
                "exact-wide-signed",
                exact_512,
                "signed exact-wide host limb export with full-width status-elided output",
                "exact_wide_signed_limbs",
                "profiles exact-wide export before expanding exact-wide GEMM variants",
                backends=EXACT_WIDE_BACKENDS,
                exact_wide_limb_counts=(4,),
            ),
            ScenarioItem(
                "exact-wide-export",
                "unsigned-limbs3-512",
                "exact-wide-unsigned",
                exact_512,
                "unsigned exact-wide compact full-width three-limb export",
                "exact_wide_unsigned_limbs",
                "compares compact 192-bit output against the default four-limb export contract",
                backends=EXACT_WIDE_BACKENDS,
                exact_wide_limb_counts=(3,),
            ),
            ScenarioItem(
                "exact-wide-export",
                "unsigned-limbs4-512",
                "exact-wide-unsigned",
                exact_512,
                "unsigned exact-wide default four-limb export",
                "exact_wide_unsigned_limbs",
                "keeps compact-output claims anchored to the current default-width baseline",
                backends=EXACT_WIDE_BACKENDS,
                exact_wide_limb_counts=(4,),
            ),
        ],
        "finite-distributions": [
            ScenarioItem(
                "finite-distributions",
                "ring-128",
                "finite-u8-ring",
                finite_128,
                "small finite-ring u8 canonical-output workload",
                "finite_u8_canonical_host_export",
                "checks whether fixed-modulus paths stay worthwhile when setup dominates",
                backends=FINITE_BACKENDS,
                finite_moduli=(251, 255),
            ),
            ScenarioItem(
                "finite-distributions",
                "field-512",
                "finite-u8-field",
                finite_512,
                "medium finite-field u8 canonical-output workload",
                "finite_u8_canonical_host_export",
                "keeps field-251 evidence separate from ring-251 and ring-255 claims",
                backends=FINITE_BACKENDS,
                finite_moduli=(251,),
            ),
        ],
        "rns-chain": [
            ScenarioItem(
                "rns-chain",
                "bounded-i64-chain3",
                "bounded-i64",
                chain_128,
                "three chained RNS GEMMs with final untimed checksum export",
                "residue_current_rns",
                "measures lazy-export benefit without confusing it with host-output timing",
                backends=accelerator_backends,
                residue_chain_length=3,
            ),
            ScenarioItem(
                "rns-chain",
                "exact-wide-signed-chain3",
                "exact-wide-signed",
                chain_128,
                "three chained exact-wide RNS GEMMs with final untimed checksum export",
                "residue_current_rns",
                "profiles exact-wide RNS-native chains separately from fixed-width limb export",
                backends=EXACT_WIDE_BACKENDS,
                residue_chain_length=3,
                exact_wide_limb_counts=(4,),
            ),
        ],
        "small-oneshot": [
            ScenarioItem(
                "small-oneshot",
                "bounded-i64-64-persistent",
                "bounded-i64",
                small_64,
                "small persistent bounded i64 baseline",
                "host_export",
                "compares persistent matrix setup against public one-shot transient input paths",
                backends=direct_oneshot_backends,
            ),
            ScenarioItem(
                "small-oneshot",
                "bounded-i64-64-oneshot",
                "bounded-i64",
                small_64,
                "small public one-shot bounded i64 call",
                "native_i64_host_output",
                "keeps small-shape one-shot evidence out of persistent RNS autotune claims",
                backends=direct_oneshot_backends,
                oneshot=True,
            ),
            ScenarioItem(
                "small-oneshot",
                "bounded-u64-128-persistent",
                "bounded-u64",
                small_128,
                "small persistent bounded u64 baseline",
                "host_export",
                "compares persistent matrix setup against public one-shot transient input paths",
                backends=direct_oneshot_backends,
            ),
            ScenarioItem(
                "small-oneshot",
                "bounded-u64-128-oneshot",
                "bounded-u64",
                small_128,
                "small public one-shot bounded u64 call",
                "native_u64_host_output",
                "keeps small-shape one-shot evidence out of persistent RNS autotune claims",
                backends=direct_oneshot_backends,
                oneshot=True,
            ),
        ],
        "skinny-gemv": [
            ScenarioItem(
                "skinny-gemv",
                "bounded-i64-n1-512",
                "bounded-i64",
                skinny_512,
                "skinny N=1 bounded i64 GEMV-like workload",
                "host_export",
                "exposes vector/native paths that square GEMM review can hide",
                backends=BOUNDED_BACKENDS,
            ),
            ScenarioItem(
                "skinny-gemv",
                "bounded-u64-n1-1024",
                "bounded-u64",
                skinny_1024,
                "skinny N=1 bounded u64 GEMV-like workload",
                "host_export",
                "checks the native vector-ALU GEMV path against RNS accelerators",
                backends=BOUNDED_BACKENDS,
            ),
        ],
        "wrap64-carry": [
            ScenarioItem(
                "wrap64-carry",
                "wrap64-512",
                "wrap-u64",
                wrap64_512,
                "strict mod 2^64 byte-limb carry-heavy workload",
                "low64_wrap_u64_host_output",
                "tracks byte-limb direct-HIP tuning separately from odd-modulus RNS claims",
                backends=tuple(WRAP64_BACKENDS),
                include_wrap64_candidate=True,
            ),
            ScenarioItem(
                "wrap64-carry",
                "wrap64-1024",
                "wrap-u64",
                wrap64_1024,
                "strict mod 2^64 byte-limb carry-heavy workload",
                "low64_wrap_u64_host_output",
                "keeps large wrap64 evidence pinned to direct byte-limb behavior",
                backends=tuple(WRAP64_BACKENDS),
                include_wrap64_candidate=True,
            ),
        ],
        "large-exploratory": [
            ScenarioItem(
                "large-exploratory",
                "bounded-i64-2048",
                "bounded-i64",
                large_2048,
                "large bounded i64 exploratory release-shape workload",
                "host_export",
                "detects whether launch overhead has stopped dominating and raw backend throughput is the limiter",
                backends=bounded_gpu_backends,
            ),
            ScenarioItem(
                "large-exploratory",
                "bounded-u64-2048",
                "bounded-u64",
                large_2048,
                "large bounded u64 exploratory release-shape workload",
                "host_export",
                "separates large-shape throughput evidence from promotable 64..1024 cache entries",
                backends=bounded_gpu_backends,
            ),
        ],
    }


def scenario_names() -> list[str]:
    return sorted(scenario_catalog())


def selected_scenario_items(args: argparse.Namespace) -> list[ScenarioItem]:
    requested = list(dict.fromkeys(getattr(args, "scenario", []) or []))
    if not requested:
        return []
    catalog = scenario_catalog()
    unknown = [name for name in requested if name != "all" and name not in catalog]
    if unknown:
        raise SystemExit(f"--scenario must be one of {scenario_names() + ['all']}, got {unknown}")
    names = scenario_names() if "all" in requested else requested
    items: list[ScenarioItem] = []
    for name in names:
        items.extend(catalog[name])
    return items


def scenario_args_for_item(args: argparse.Namespace, item: ScenarioItem) -> argparse.Namespace:
    scenario_args = argparse.Namespace(**vars(args))
    scenario_args.reuse_packed_inputs = item.pack_mode == "prepacked_reuse"
    scenario_args.reuse_packed_a = item.pack_mode == "prepacked_reuse_a"
    scenario_args.reuse_packed_b = item.pack_mode == "prepacked_reuse_b"
    scenario_args.residue_chain_length = item.residue_chain_length
    scenario_args.output_ld_padding = item.output_ld_padding
    scenario_args.prefix_policy = item.prefix_policy or getattr(args, "prefix_policy", None)
    scenario_args.max_prefix = item.max_prefix if item.max_prefix is not None else getattr(args, "max_prefix", None)
    scenario_args.bound_source = item.bound_source or getattr(args, "bound_source", None)
    return scenario_args


def scenario_backends_for_item(args: argparse.Namespace, item: ScenarioItem) -> list[str]:
    backends = list(item.backends or default_backends_for(item.semantics, item.case))
    if item.semantics == "wrap-u64" and item.include_wrap64_candidate and args.include_wrap64_rocwmma_candidate:
        backends.append(WRAP64_ROCWMMA_CANDIDATE_BACKEND)
    if args.backends:
        requested = set(args.backends)
        backends = [backend for backend in backends if backend in requested]
    return list(dict.fromkeys(backends))


def scenario_metadata(
    item: ScenarioItem,
    backend: str,
    modulus: int | None,
    exact_wide_limb_count: int | None,
    scenario_args: argparse.Namespace,
    *,
    oneshot: bool,
) -> dict[str, Any]:
    return {
        "family": item.family,
        "name": item.name,
        "semantics": item.semantics,
        "backend": backend,
        "modulus": modulus,
        "exact_wide_limb_count": exact_wide_limb_count,
        "shape": {"m": item.case.m, "n": item.case.n, "k": item.case.k},
        "tile": {"m": item.case.tile_m, "n": item.case.tile_n},
        "bound_mode": item.case.bound_mode,
        "input_profile": item.case.input_profile,
        "pack_mode": requested_pack_mode(scenario_args),
        "reuse_packed_inputs": requested_pack_mode(scenario_args) != "per_repeat_repack",
        "residue_chain_length": item.residue_chain_length,
        "output_ld_padding": item.output_ld_padding,
        "oneshot": oneshot,
        "evidence_scope": item.evidence_scope,
        "output_domain": item.output_domain,
        "rationale": item.rationale,
    }


def default_backends_for(semantics: str, case: SweepCase) -> list[str]:
    if semantics in {"bounded-i64", "bounded-u64"}:
        return ["cpu", "hip-direct", "hip-vector-alu-int64", "ck", "rocwmma"] if case.bound_mode == "per-tile" else BOUNDED_BACKENDS
    if semantics in {"finite-u8-ring", "finite-u8-field"}:
        return FINITE_BACKENDS
    if semantics in {"exact-wide-signed", "exact-wide-unsigned"}:
        return EXACT_WIDE_BACKENDS
    if semantics == "wrap-u64":
        return WRAP64_BACKENDS
    return []


def backend_allowed_for(semantics: str, case: SweepCase, backend: str) -> bool:
    if semantics == "wrap-u64":
        return backend in WRAP64_BACKENDS or backend == WRAP64_ROCWMMA_CANDIDATE_BACKEND
    if semantics in {"finite-u8-ring", "finite-u8-field"}:
        return backend in FINITE_BACKENDS
    if semantics in {"exact-wide-signed", "exact-wide-unsigned"}:
        return case.bound_mode == "global" and backend in EXACT_WIDE_BACKENDS
    if semantics in {"bounded-i64", "bounded-u64"}:
        if backend not in BOUNDED_BACKENDS:
            return False
        if case.bound_mode == "per-tile" and backend == "hipblaslt":
            return False
        return True
    return False


def capture_name(
    semantics: str,
    case: SweepCase,
    backend: str,
    modulus: int | None,
    pack_mode: str,
    exact_wide_limb_count: int | None = None,
    residue_chain_length: int = 1,
    output_ld_padding: int = 0,
    oneshot: bool = False,
) -> str:
    parts = [semantics, case.name, f"{case.m}x{case.n}x{case.k}"]
    if modulus is not None:
        parts.append(f"mod{modulus}")
    if semantics in EXACT_WIDE_SEMANTICS and exact_wide_limb_count not in (None, DEFAULT_EXACT_WIDE_LIMB_COUNT):
        parts.append(f"limbs{exact_wide_limb_count}")
    if semantics in RNS_CHAIN_SEMANTICS and residue_chain_length > 1:
        parts.append(f"chain{residue_chain_length}")
    if output_ld_padding > 0:
        parts.append(f"outpad{output_ld_padding}")
    if oneshot:
        parts.append("oneshot")
    if pack_mode == "prepacked_reuse":
        parts.append("reuse-packed")
    elif pack_mode == "prepacked_reuse_a":
        parts.append("reuse-packed-a")
    elif pack_mode == "prepacked_reuse_b":
        parts.append("reuse-packed-b")
    parts.append(backend)
    return "-".join(parts).replace("/", "_") + ".json"


def command_for(
    bench: Path,
    backend: str,
    semantics: str,
    case: SweepCase,
    modulus: int | None,
    exact_wide_limb_count: int | None,
    args: argparse.Namespace,
    *,
    oneshot: bool = False,
) -> list[str]:
    tile_m = 16 if semantics == "wrap-u64" and backend == WRAP64_ROCWMMA_CANDIDATE_BACKEND else case.tile_m
    tile_n = 16 if semantics == "wrap-u64" and backend == WRAP64_ROCWMMA_CANDIDATE_BACKEND else case.tile_n
    command = [
        str(bench),
        "--backend",
        cli_backend(backend),
        "--semantics",
        semantics,
        "--m",
        str(case.m),
        "--n",
        str(case.n),
        "--k",
        str(case.k),
        "--tile-m",
        str(tile_m),
        "--tile-n",
        str(tile_n),
        "--bound-mode",
        case.bound_mode,
        "--warmups",
        str(args.warmups),
        "--repeats",
        str(args.repeats),
        "--seed",
        str(args.seed),
    ]
    if case.require_adaptive and getattr(args, "prefix_policy", None) != "fixed-requested":
        command.append("--require-adaptive-execution")
    if case.input_profile != "uniform-small":
        command.extend(["--input-profile", case.input_profile])
    if modulus is not None:
        command.extend(["--modulus", str(modulus)])
    if exact_wide_limb_count is not None:
        command.extend(["--exact-wide-limbs", str(exact_wide_limb_count)])
    bound_source = getattr(args, "bound_source", None)
    if bound_source:
        command.extend(["--bound-source", bound_source])
    prefix_policy = getattr(args, "prefix_policy", None)
    max_prefix = getattr(args, "max_prefix", None)
    if prefix_policy:
        command.extend(["--prefix-policy", prefix_policy])
    if max_prefix is not None:
        command.extend(["--max-prefix", str(max_prefix)])
    if args.residue_chain_length > 1:
        command.extend(["--residue-chain-length", str(args.residue_chain_length)])
    output_ld_padding = int(getattr(args, "output_ld_padding", 0) or 0)
    if output_ld_padding > 0:
        command.extend(["--output-ld-padding", str(output_ld_padding)])
    if oneshot:
        command.append("--oneshot")
    pack_mode = requested_pack_mode(args)
    if pack_mode == "prepacked_reuse":
        command.append("--reuse-packed-inputs")
    elif pack_mode == "prepacked_reuse_a":
        command.append("--reuse-packed-a")
    elif pack_mode == "prepacked_reuse_b":
        command.append("--reuse-packed-b")
    return command


def default_sweep_command_entries(args: argparse.Namespace) -> list[SweepCommand]:
    backend_benches = parse_backend_bench(args.bench_for)
    commands: list[SweepCommand] = []
    semantics_values = [normalize_semantics(item) for item in (args.semantics or ["bounded-i64", "bounded-u64"])]
    cases = [*([] if args.adaptive_only else default_cases(args)), *adaptive_cases(args)]
    if args.adaptive_only and not cases:
        raise SystemExit("--adaptive-only requires --adaptive-case, --include-default-adaptive, or --include-adaptive-workloads")
    if (args.include_wrap64 or args.include_wrap64_rocwmma_candidate) and "wrap-u64" not in semantics_values:
        semantics_values.append("wrap-u64")
    if args.include_exact_wide:
        for exact_semantics in EXACT_WIDE_SEMANTICS:
            if exact_semantics not in semantics_values:
                semantics_values.append(exact_semantics)
    if (getattr(args, "prefix_policy", None) or getattr(args, "max_prefix", None) is not None) and any(
        semantics not in {"bounded-i64", "bounded-u64", "exact-wide-signed", "exact-wide-unsigned"}
        for semantics in semantics_values
    ):
        raise SystemExit("--prefix-policy and --max-prefix are only valid for bounded or exact-wide RNS sweeps")
    if getattr(args, "bound_source", None) == "input-scan" and any(
        semantics not in {"bounded-i64", "bounded-u64"} for semantics in semantics_values
    ):
        raise SystemExit("--bound-source input-scan is only valid for bounded RNS sweeps")
    if getattr(args, "max_prefix", None) is not None and args.max_prefix <= 0:
        raise SystemExit("--max-prefix must be positive")
    if int(getattr(args, "output_ld_padding", 0) or 0) < 0:
        raise SystemExit("--output-ld-padding must be nonnegative")
    if args.residue_chain_length < 1:
        raise SystemExit("--residue-chain-length must be positive")
    if args.residue_chain_length > 1:
        non_rns_chain = [semantics for semantics in semantics_values if semantics not in RNS_CHAIN_SEMANTICS]
        if non_rns_chain:
            raise SystemExit("--residue-chain-length > 1 currently requires bounded or exact-wide RNS semantics")
    include_oneshot = bool(getattr(args, "include_oneshot", False))
    oneshot_only = bool(getattr(args, "oneshot_only", False))
    if include_oneshot or oneshot_only:
        if args.adaptive_only:
            raise SystemExit("--include-oneshot cannot be combined with --adaptive-only")
        if requested_pack_mode(args) != "per_repeat_repack":
            raise SystemExit("--include-oneshot cannot be combined with packed-input reuse modes")
        if args.residue_chain_length > 1:
            raise SystemExit("--include-oneshot cannot be combined with --residue-chain-length > 1")
    for semantics in semantics_values:
        if semantics == "wrap-u64":
            if args.adaptive_only:
                continue
            active_cases = wrap64_cases(args)
        else:
            active_cases = cases
        for case in active_cases:
            if args.residue_chain_length > 1 and (case.m != case.n or case.n != case.k):
                raise SystemExit("--residue-chain-length > 1 currently requires square RNS cases")
            if args.residue_chain_length > 1 and semantics in BOUNDED_SEMANTICS and case.bound_mode != "global":
                raise SystemExit("bounded --residue-chain-length > 1 currently requires global bound mode")
            backends = args.backends or (
                wrap64_backends_for(args) if semantics == "wrap-u64" else default_backends_for(semantics, case)
            )
            for modulus in finite_moduli_for(semantics, args):
                for exact_wide_limb_count in exact_wide_limb_counts_for(semantics, args):
                    if not oneshot_only:
                        for backend in backends:
                            if not backend_allowed_for(semantics, case, backend):
                                continue
                            if (
                                args.residue_chain_length > 1
                                and semantics in BOUNDED_SEMANTICS
                                and backend in {"auto", "hip-vector-alu-int64"}
                            ):
                                continue
                            bench = backend_benches.get(backend, args.bench)
                            if bench is None:
                                raise SystemExit(f"no benchmark executable configured for backend {backend}")
                            name = capture_name(
                                semantics,
                                case,
                                backend,
                                modulus,
                                requested_pack_mode(args),
                                exact_wide_limb_count,
                                args.residue_chain_length,
                                int(getattr(args, "output_ld_padding", 0) or 0),
                            )
                            command = command_for(bench, backend, semantics, case, modulus, exact_wide_limb_count, args)
                            commands.append(SweepCommand(name, command, args.out_root / name))
                    if (include_oneshot or oneshot_only) and (
                        (semantics in BOUNDED_SEMANTICS and case.bound_mode == "global")
                        or semantics in {"finite-u8-ring", "finite-u8-field"}
                    ):
                        oneshot_backends = [
                            backend
                            for backend in (args.backends or PUBLIC_ONESHOT_BACKENDS)
                            if backend in PUBLIC_ONESHOT_BACKENDS
                        ]
                        for backend in oneshot_backends:
                            bench = backend_benches.get(backend, args.bench)
                            if bench is None:
                                raise SystemExit(f"no benchmark executable configured for backend {backend}")
                            name = capture_name(
                                semantics,
                                case,
                                backend,
                                modulus,
                                requested_pack_mode(args),
                                exact_wide_limb_count,
                                args.residue_chain_length,
                                int(getattr(args, "output_ld_padding", 0) or 0),
                                oneshot=True,
                            )
                            command = command_for(
                                bench,
                                backend,
                                semantics,
                                case,
                                modulus,
                                exact_wide_limb_count,
                                args,
                                oneshot=True,
                            )
                            commands.append(SweepCommand(name, command, args.out_root / name))
    return commands


def scenario_sweep_command_entries(args: argparse.Namespace) -> list[SweepCommand]:
    if any(
        [
            getattr(args, "semantics", None),
            getattr(args, "case", None),
            getattr(args, "adaptive_case", None),
            getattr(args, "shapes", None),
            getattr(args, "include_default_adaptive", False),
            getattr(args, "include_adaptive_workloads", False),
            getattr(args, "adaptive_only", False),
            getattr(args, "include_wrap64", False),
            getattr(args, "include_exact_wide", False),
            getattr(args, "include_oneshot", False),
            getattr(args, "oneshot_only", False),
            getattr(args, "reuse_packed_inputs", False),
            getattr(args, "reuse_packed_a", False),
            getattr(args, "reuse_packed_b", False),
            getattr(args, "release_matrix", False),
            getattr(args, "include_exploratory_large", False),
            getattr(args, "prefix_policy", None),
            getattr(args, "max_prefix", None) is not None,
            getattr(args, "bound_source", None),
            int(getattr(args, "output_ld_padding", 0) or 0) != 0,
            int(getattr(args, "residue_chain_length", 1) or 1) != 1,
        ]
    ):
        raise SystemExit("--scenario cannot be combined with manual sweep shape/semantics/reuse/include flags")

    backend_benches = parse_backend_bench(args.bench_for)
    commands: list[SweepCommand] = []
    for item in selected_scenario_items(args):
        scenario_args = scenario_args_for_item(args, item)
        backends = scenario_backends_for_item(args, item)
        if not backends:
            continue
        finite_moduli = finite_moduli_for(item.semantics, args) if args.modulus else list(item.finite_moduli)
        if args.exact_wide_limbs:
            exact_wide_counts = exact_wide_limb_counts_for(item.semantics, args)
        else:
            exact_wide_counts = list(item.exact_wide_limb_counts)
        for modulus in finite_moduli:
            for exact_wide_limb_count in exact_wide_counts:
                for backend in backends:
                    if not backend_allowed_for(item.semantics, item.case, backend):
                        continue
                    if (
                        item.residue_chain_length > 1
                        and item.semantics in BOUNDED_SEMANTICS
                        and backend in {"auto", "hip-vector-alu-int64"}
                    ):
                        continue
                    bench = backend_benches.get(backend, args.bench)
                    if bench is None:
                        raise SystemExit(f"no benchmark executable configured for backend {backend}")
                    base_name = capture_name(
                        item.semantics,
                        item.case,
                        backend,
                        modulus,
                        requested_pack_mode(scenario_args),
                        exact_wide_limb_count,
                        item.residue_chain_length,
                        item.output_ld_padding,
                        oneshot=item.oneshot,
                    )
                    name = f"{item.family}-{item.name}-{base_name}"
                    command = command_for(
                        bench,
                        backend,
                        item.semantics,
                        item.case,
                        modulus,
                        exact_wide_limb_count,
                        scenario_args,
                        oneshot=item.oneshot,
                    )
                    metadata = scenario_metadata(
                        item,
                        backend,
                        modulus,
                        exact_wide_limb_count,
                        scenario_args,
                        oneshot=item.oneshot,
                    )
                    output = args.out_root / "scenarios" / item.family / name
                    commands.append(SweepCommand(name, command, output, metadata))
    return commands


def sweep_command_entries(args: argparse.Namespace) -> list[SweepCommand]:
    return (
        scenario_sweep_command_entries(args)
        if getattr(args, "scenario", None)
        else default_sweep_command_entries(args)
    )


def sweep_commands(args: argparse.Namespace) -> list[tuple[str, list[str], Path]]:
    return [(entry.name, entry.command, entry.output) for entry in sweep_command_entries(args)]


def scenario_manifest(entries: list[SweepCommand], args: argparse.Namespace) -> dict[str, Any]:
    scenario_entries = [entry for entry in entries if entry.scenario is not None]
    families = sorted({str(entry.scenario["family"]) for entry in scenario_entries if entry.scenario})
    return {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scenario_request": list(getattr(args, "scenario", []) or []),
        "scenario_families": families,
        "capture_count": len(scenario_entries),
        "review_mode": args.review_mode,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "seed": args.seed,
        "entries": [
            {
                **(entry.scenario or {}),
                "capture_name": entry.name,
                "capture_path": str(entry.output),
                "command": entry.command,
            }
            for entry in scenario_entries
        ],
    }


def write_scenario_manifest(entries: list[SweepCommand], args: argparse.Namespace, out_root: Path) -> dict[str, str] | None:
    if not any(entry.scenario is not None for entry in entries):
        return None
    manifest = scenario_manifest(entries, args)
    json_path = out_root / "scenario_manifest.json"
    json_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# RNS8 Scenario Benchmark Manifest",
        "",
        f"- schema_version: `{manifest['schema_version']}`",
        f"- generated_utc: `{manifest['generated_utc']}`",
        f"- scenario_request: `{','.join(manifest['scenario_request'])}`",
        f"- scenario_families: `{','.join(manifest['scenario_families'])}`",
        f"- captures: `{manifest['capture_count']}`",
        f"- review_mode: `{manifest['review_mode']}`",
        f"- warmups: `{manifest['warmups']}`",
        f"- repeats: `{manifest['repeats']}`",
        f"- seed: `{manifest['seed']}`",
        "",
        "| family | item | semantics | shape | backend | pack | output_domain | evidence_scope | capture |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for entry in manifest["entries"]:
        shape = entry["shape"]
        lines.append(
            "| {family} | {name} | {semantics} | {m}x{n}x{k} | {backend} | {pack} | {domain} | {scope} | {capture} |".format(
                family=entry["family"],
                name=entry["name"],
                semantics=entry["semantics"],
                m=shape["m"],
                n=shape["n"],
                k=shape["k"],
                backend=entry["backend"],
                pack=entry["pack_mode"],
                domain=entry["output_domain"],
                scope=entry["evidence_scope"],
                capture=entry["capture_name"],
            )
        )
    markdown_path = out_root / "scenario_manifest.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"scenario_manifest": str(json_path), "scenario_markdown": str(markdown_path)}


def write_markdown_report(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# RNS8 Benchmark Sweep Review",
        "",
        f"- schema_version: `{report.get('schema_version')}`",
        f"- review_mode: `{report.get('review_mode')}`",
        f"- reviewed_utc: `{report.get('reviewed_utc')}`",
        f"- groups: `{report.get('group_count')}`",
        f"- promotable_autotune_entries: `{len(report.get('promotable_autotune_entries', []))}`",
        f"- cache_write: `{report.get('cache_write', {}).get('status')}`",
        "",
    ]
    for group in report.get("groups", []):
        shape = group.get("shape", {})
        modulus = group.get("finite_modulus")
        title = f"{group.get('semantics')} {shape.get('m')}x{shape.get('n')}x{shape.get('k')}"
        if modulus is not None:
            title += f" mod {modulus}"
        lines.extend([f"## {title}", ""])
        missing = group.get("missing_required_baselines") or []
        missing_targets = group.get("missing_gpu_targets") or []
        lines.append(f"- missing_required_baselines: `{','.join(missing) if missing else 'none'}`")
        lines.append(f"- missing_gpu_targets: `{','.join(missing_targets) if missing_targets else 'none'}`")
        lines.append(f"- gpu_target_compatible: `{group.get('gpu_target_compatible')}`")
        missing_configured = group.get("missing_configured_gpu_targets") or []
        lines.append(
            f"- missing_configured_gpu_targets: `{','.join(missing_configured) if missing_configured else 'none'}`"
        )
        lines.append(f"- configured_target_compatible: `{group.get('configured_target_compatible')}`")
        missing_versions = group.get("missing_hip_toolchain_versions") or []
        lines.append(
            f"- missing_hip_toolchain_versions: `{','.join(missing_versions) if missing_versions else 'none'}`"
        )
        lines.append(f"- hip_toolchain_version_compatible: `{group.get('hip_toolchain_version_compatible')}`")
        missing_runtime = group.get("missing_hip_runtime_versions") or []
        lines.append(
            f"- missing_hip_runtime_versions: `{','.join(missing_runtime) if missing_runtime else 'none'}`"
        )
        lines.append(f"- hip_runtime_version_compatible: `{group.get('hip_runtime_version_compatible')}`")
        missing_driver = group.get("missing_hip_driver_versions") or []
        lines.append(f"- missing_hip_driver_versions: `{','.join(missing_driver) if missing_driver else 'none'}`")
        lines.append(f"- hip_driver_version_compatible: `{group.get('hip_driver_version_compatible')}`")
        missing_compilers = group.get("missing_compiler_identities") or []
        lines.append(f"- missing_compiler_identities: `{','.join(missing_compilers) if missing_compilers else 'none'}`")
        lines.append(f"- compiler_identity_compatible: `{group.get('compiler_identity_compatible')}`")
        missing_git = group.get("missing_git_commits") or []
        lines.append(f"- missing_git_commits: `{','.join(missing_git) if missing_git else 'none'}`")
        lines.append(f"- git_commit_identity_compatible: `{group.get('git_commit_identity_compatible')}`")
        missing_warmups = group.get("missing_warmup_counts") or []
        lines.append(f"- missing_warmup_counts: `{','.join(missing_warmups) if missing_warmups else 'none'}`")
        lines.append(f"- warmup_count_compatible: `{group.get('warmup_count_compatible')}`")
        missing_repeats = group.get("missing_repeat_counts") or []
        lines.append(f"- missing_repeat_counts: `{','.join(missing_repeats) if missing_repeats else 'none'}`")
        lines.append(f"- repeat_count_compatible: `{group.get('repeat_count_compatible')}`")
        duplicates = group.get("duplicate_backends") or []
        lines.append(f"- duplicate_backends: `{','.join(duplicates) if duplicates else 'none'}`")
        lines.append(f"- release_review_satisfied: `{group.get('release_review_satisfied')}`")
        fastest = group.get("fastest_promotable")
        if fastest:
            lines.append(f"- fastest_promotable: `{fastest['backend']}/{fastest['selected_kernel']}`")
            lines.append(f"- winner_rationale: `{fastest.get('promotion_reason')}`")
        else:
            lines.append("- fastest_promotable: `none`")
        lines.append("")
        lines.append(
            "| backend | kernel | target | e2e median us | bottleneck | promotable | cache | blockers | primary loss phase |"
        )
        lines.append("|---|---|---|---:|---|---|---|---|---|")
        for candidate in group.get("candidates", []):
            blockers = ",".join(candidate.get("promotion_blockers") or [])
            source = candidate.get("source_metadata") if isinstance(candidate.get("source_metadata"), dict) else {}
            bottleneck = candidate.get("bottleneck") if isinstance(candidate.get("bottleneck"), dict) else {}
            bottleneck_text = bottleneck.get("class") or "unknown"
            if bottleneck.get("phase"):
                bottleneck_text += f"/{bottleneck.get('phase')}"
            lines.append(
                "| {backend} | {kernel} | {target} | {median} | {bottleneck} | {promotable} | {cache} | {blockers} | {loss} |".format(
                    backend=candidate.get("backend"),
                    kernel=candidate.get("selected_kernel"),
                    target=source.get("target_id"),
                    median=candidate.get("median_end_to_end_us"),
                    bottleneck=bottleneck_text,
                    promotable=candidate.get("promotable"),
                    cache=candidate.get("cache_write_status"),
                    blockers=blockers or "none",
                    loss=candidate.get("primary_loss_phase_vs_direct_hip") or "",
                )
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", type=Path, help="path to rns8-bench executable")
    parser.add_argument(
        "--bench-for",
        action="append",
        default=[],
        metavar="BACKEND=PATH",
        help="override rns8-bench executable for one backend, useful for opt-in accelerator build dirs",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("temp") / "benchmark-sweeps" / "windows-gfx1100",
        help="ignored output directory for raw captures and review report",
    )
    parser.add_argument("--capture", type=Path, action="append", default=[], help="existing capture to review")
    parser.add_argument("--review-only", action="store_true", help="only review --capture files")
    parser.add_argument(
        "--scenario",
        action="append",
        choices=[*scenario_names(), "all"],
        help="run a named scenario corpus family; repeatable, or use all",
    )
    parser.add_argument("--backend", dest="backends", action="append", help="backend to sweep; repeatable")
    parser.add_argument("--semantics", action="append", help="benchmark semantics to sweep; repeatable")
    parser.add_argument("--case", action="append", help="global case NAME:M,N,K; repeatable")
    parser.add_argument(
        "--adaptive-case",
        action="append",
        help="adaptive case NAME:M,N,K,TILE_M,TILE_N[,INPUT_PROFILE]; repeatable",
    )
    parser.add_argument("--shape", dest="shapes", action="append", help="legacy square shape to sweep; repeatable")
    parser.add_argument("--modulus", type=int, action="append", help="finite-u8 modulus; repeatable")
    parser.add_argument("--exact-wide-limbs", type=int, action="append", help="exact-wide output limb count; repeatable")
    parser.add_argument(
        "--bound-source",
        choices=["static-profile", "input-scan"],
        help="bounded RNS bound source to pass through to rns8-bench",
    )
    parser.add_argument(
        "--prefix-policy",
        choices=["minimum-proven", "fixed-requested"],
        help="bounded/exact-wide RNS prefix policy to pass through to rns8-bench",
    )
    parser.add_argument(
        "--max-prefix",
        type=int,
        help="bounded/exact-wide RNS requested maximum prefix to pass through to rns8-bench",
    )
    parser.add_argument(
        "--residue-chain-length",
        type=int,
        default=1,
        help="exact-wide residue-current GEMM chain length; values above 1 skip timed host export",
    )
    parser.add_argument(
        "--output-ld-padding",
        type=int,
        default=0,
        help="add padding columns to the benchmark host output leading dimension",
    )
    parser.add_argument(
        "--include-exact-wide-limb-variants",
        action="store_true",
        help="include exact-wide output limb counts 1, 2, 3, 4, 8, 16, and 32",
    )
    parser.add_argument("--include-default-adaptive", action="store_true", help="include default adaptive bounded cases")
    parser.add_argument(
        "--include-adaptive-workloads",
        action="store_true",
        help="include profile-driven adaptive bounded workload cases",
    )
    parser.add_argument("--adaptive-only", action="store_true", help="run only adaptive cases, skipping global cases")
    parser.add_argument("--include-wrap64", action="store_true", help="include wrap64 CPU/direct-HIP captures")
    parser.add_argument(
        "--include-rocwmma-wrap64-candidate",
        dest="include_wrap64_rocwmma_candidate",
        action="store_true",
        help="include the internal rocWMMA wrap64 byte-GEMM36 candidate in wrap64 sweeps",
    )
    parser.add_argument(
        "--include-exact-wide",
        action="store_true",
        help="include exact-wide signed and unsigned captures",
    )
    parser.add_argument(
        "--reuse-packed-inputs",
        action="store_true",
        help="pack A/B once before warmups and benchmark repeated GEMM/export against persistent packed inputs",
    )
    parser.add_argument(
        "--reuse-packed-a",
        action="store_true",
        help="pack A once before warmups and benchmark repeats that repack B",
    )
    parser.add_argument(
        "--reuse-packed-b",
        action="store_true",
        help="pack B once before warmups and benchmark repeats that repack A",
    )
    parser.add_argument(
        "--include-oneshot",
        action="store_true",
        help="also capture bounded/finite CPU/direct-HIP public one-shot API runs beside persistent matrix baselines",
    )
    parser.add_argument(
        "--oneshot-only",
        action="store_true",
        help="capture only bounded/finite CPU/direct-HIP public one-shot API runs for the selected global cases",
    )
    parser.add_argument("--release-matrix", action="store_true", help="use promotable release bounded shapes 64..1024")
    parser.add_argument("--include-exploratory-large", action="store_true", help="include 2048/4096/8192 exploratory shapes")
    parser.add_argument(
        "--review-mode",
        choices=["smoke", "release"],
        default="smoke",
        help="release mode is required before captures can produce performance_validated autotune entries",
    )
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--write-autotune-cache",
        action="store_true",
        help="write performance_validated cache entries only for fastest promotable release-reviewed captures",
    )
    parser.add_argument("--autotune-cache", type=Path, help="override autotune cache output path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_root = Path(args.out_root)
    args.out_root.mkdir(parents=True, exist_ok=True)

    capture_paths = list(args.capture)
    entries: list[SweepCommand] = []
    if not args.review_only:
        if args.bench is None:
            raise SystemExit("--bench is required unless --review-only is used")
        entries = sweep_command_entries(args)
        for entry in entries:
            if run_command(entry.command, entry.output):
                capture_paths.append(entry.output)

    captures = validate_paths(capture_paths)
    report = review_captures(captures, review_mode=args.review_mode)
    promoted = 0
    cache_path = args.autotune_cache or autotune_cache_path()
    if args.write_autotune_cache:
        promoted = write_promoted_cache_entries(report, captures, cache_path)
    attach_cache_write_status(report, args.write_autotune_cache, cache_path, promoted)
    report_path = args.out_root / "review_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path = args.out_root / "review_report.md"
    write_markdown_report(report, markdown_path)
    scenario_paths = write_scenario_manifest(entries, args, args.out_root)
    output = {
        "review_report": str(report_path),
        "markdown_report": str(markdown_path),
        "captures": len(captures),
        "promoted_cache_entries": promoted,
        "autotune_cache": str(cache_path) if args.write_autotune_cache else None,
    }
    if scenario_paths is not None:
        output.update(scenario_paths)
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
