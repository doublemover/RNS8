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
HOST_API_BATCH_BACKENDS = ["cpu", "hip-direct", "hipblaslt", "ck", "rocwmma"]
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
    residue_chain_final_export: bool = False
    output_ld_padding: int = 0
    host_api_batch_size: int = 1
    oneshot: bool = False
    native_to_rns_bridge: bool = False
    vector_to_rns_chain: bool = False
    prefix_policy: str | None = None
    max_prefix: int | None = None
    bound_source: str | None = None
    next_op_hint: str | None = None
    residue_channel_fusion: bool = False
    modulus_set: str = "default"
    tile_shape_variant: str = "default"
    export_variant: str = "default"
    reconstruction_variant: str = "default_garner"
    grouped_dispatch_tasks: int = 1
    hip_graph_replay: bool = False
    workload_proxy: str = "none"
    resident_lifetime: bool = False
    workspace_arena: bool = False
    adaptive_grouped_scheduler: bool = False
    streaming_overlap: bool = False
    release_gate: str = "none"
    verification_amortization: str = "none"
    include_wrap64_candidate: bool = False
    metadata: dict[str, Any] | None = None


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
    timing_metadata = capture.get("timing_metadata")
    output_policy = capture.get("output_policy")
    requested_next_op = capture.get("requested_next_op")
    residue_output_mode = capture.get("residue_output_mode", "host_export")
    if residue_output_mode == "host_export":
        next_op_contract = "host_export"
    else:
        next_op_contract = requested_next_op.get("resolved") if isinstance(requested_next_op, dict) else None
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
        f"residue_output_mode={residue_output_mode}",
        f"seed={capture.get('seed')}",
        f"input_distribution={capture.get('input_distribution')}",
        f"reuse_packed_inputs={capture.get('reuse_packed_inputs') is True}",
        f"pack_mode={capture_pack_mode(capture)}",
        f"next_op_contract={next_op_contract}",
        f"output_policy={output_policy.get('destination_layout') if isinstance(output_policy, dict) else None}",
        f"status_handling={output_policy.get('status_handling') if isinstance(output_policy, dict) else None}",
        f"fusion_mode={timing_metadata.get('fusion_mode') if isinstance(timing_metadata, dict) else None}",
        f"residue_group_width={timing_metadata.get('residue_group_width') if isinstance(timing_metadata, dict) else None}",
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
    execution_mode = capture_execution_mode(capture)
    if execution_mode == "public_oneshot_transient_native_inputs":
        return f"{backend}-oneshot"
    if execution_mode == "benchmark_host_api_batch":
        return f"{backend}-hostbatch"
    if execution_mode == "hip_graph_replay_resident_rns_chain":
        return f"{backend}-hipgraph"
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


def backend_family_id(backend: str) -> str:
    for suffix in ("-oneshot", "-hostbatch"):
        if backend.endswith(suffix):
            return backend[: -len(suffix)]
    return backend


def backend_requires_gpu_target(backend: str) -> bool:
    return backend_family_id(backend) not in {"cpu-reference", "wrap64-byte-limb"}


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


def _timeout_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


def run_command(command: list[str], output: Path, timeout_seconds: float | None = None) -> bool:
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        completed = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        failure = {
            "command": command,
            "returncode": None,
            "timed_out": True,
            "timeout_seconds": timeout_seconds,
            "stdout": _timeout_text(exc.stdout),
            "stderr": _timeout_text(exc.stderr),
        }
        output.with_suffix(".failed.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
        return False
    if completed.returncode != 0:
        failure = {
            "command": command,
            "returncode": completed.returncode,
            "timed_out": False,
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


def existing_capture_valid(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        data = load_capture(path)
        validate_capture(data, path)
    except (BenchmarkSchemaError, OSError, json.JSONDecodeError):
        return False
    return True


def execute_sweep_entries(
    entries: list[SweepCommand],
    args: argparse.Namespace,
    capture_paths: list[Path],
) -> dict[str, int]:
    stats = {
        "planned_captures": len(entries),
        "skipped_existing_captures": 0,
        "new_captures_attempted": 0,
        "new_captures_completed": 0,
        "deferred_captures": 0,
    }
    max_new = getattr(args, "max_new_captures", None)
    timeout_seconds = getattr(args, "capture_timeout_seconds", None)
    for entry in entries:
        if getattr(args, "skip_existing", False) and existing_capture_valid(entry.output):
            capture_paths.append(entry.output)
            stats["skipped_existing_captures"] += 1
            continue
        if max_new is not None and stats["new_captures_attempted"] >= max_new:
            stats["deferred_captures"] += 1
            continue
        stats["new_captures_attempted"] += 1
        if run_command(entry.command, entry.output, timeout_seconds=timeout_seconds):
            capture_paths.append(entry.output)
            stats["new_captures_completed"] += 1
    return stats


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
    host_api_batch_capture: bool,
    hip_graph_replay_capture: bool,
    gpu_events_available: bool,
    end_to_end: float | None,
    cpu: float | None,
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
    if host_api_batch_capture:
        blockers.append("host_api_batch_not_autotune_promotable")
    if hip_graph_replay_capture:
        blockers.append("hip_graph_replay_not_autotune_promotable")
    if accelerator and not gpu_events_available:
        blockers.append("missing_required_gpu_events")
    if end_to_end is None:
        blockers.append("missing_end_to_end_timing")
    if cpu is not None and end_to_end is not None and end_to_end >= cpu:
        blockers.append("not_faster_than_cpu_reference")
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


def capture_gpu_events_available(capture: dict[str, Any]) -> bool:
    timing = capture_timing_metadata(capture)
    return (
        timing.get("gpu_event_timing") is True
        and timing.get("gpu_event_timing_status") == "available"
        and bool(timing.get("gpu_event_timing_source"))
        and isinstance(timing.get("gpu_event_phase_order"), list)
    )


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
        cpu_capture = by_backend.get("cpu-reference")
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
            execution_mode = capture_execution_mode(item)
            oneshot_capture = execution_mode == "public_oneshot_transient_native_inputs"
            host_api_batch_capture = execution_mode == "benchmark_host_api_batch"
            hip_graph_replay_capture = execution_mode == "hip_graph_replay_resident_rns_chain"
            end_to_end = median_phase(item, "end_to_end")
            cpu = median_phase(cpu_capture, "end_to_end") if cpu_capture else None
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
                host_api_batch_capture=host_api_batch_capture,
                hip_graph_replay_capture=hip_graph_replay_capture,
                gpu_events_available=capture_gpu_events_available(item),
                end_to_end=end_to_end,
                cpu=cpu,
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
    layout_512 = parse_case("layout-512:512,512,512")
    finite_128 = parse_case("finite-128:128,128,128")
    finite_512 = parse_case("finite-512:512,512,512")
    finite_generic_512 = parse_case("finite-generic-512:512,512,512")
    finite_generic_2048 = parse_case("finite-generic-2048:2048,2048,2048", promotable=False)
    chain_128 = parse_case("chain-128:128,128,128")
    chain_256 = parse_case("chain-256:256,256,256")
    chain_512 = parse_case("chain-512:512,512,512")
    small_64 = parse_case("small-64:64,64,64")
    small_128 = parse_case("small-128:128,128,128")
    pack_heavy_128 = parse_case("pack-heavy-128x128x4096:128,128,4096")
    many_small_32 = parse_case("many-small-32:32,32,32")
    many_small_64 = parse_case("many-small-64:64,64,64")
    many_small_128 = parse_case("many-small-128:128,128,128")
    many_small_skinny = parse_case("many-small-skinny-128x1x1024:128,1,1024")
    skinny_512 = parse_case("gemv-n1-512:512,1,512")
    skinny_1024 = parse_case("gemv-n1-1024:1024,1,1024")
    skinny_longk = parse_case("gemv-n1-longk-256:256,1,4096")
    wrap64_512 = parse_case("wrap64-512:512,512,512")
    wrap64_1024 = parse_case("wrap64-1024:1024,1024,1024")
    large_2048 = parse_case("large-2048:2048,2048,2048", promotable=False)
    large_4096 = parse_case("large-4096:4096,4096,4096", promotable=False)
    large_finite_2048 = parse_case("finite-large-2048:2048,2048,2048", promotable=False)
    large_finite_4096 = parse_case("finite-large-4096:4096,4096,4096", promotable=False)
    large_exact_2048 = parse_case("exact-wide-large-2048:2048,2048,2048", promotable=False)
    large_exact_4096 = parse_case("exact-wide-large-4096:4096,4096,4096", promotable=False)
    large_wrap64_2048 = parse_case("wrap64-large-2048:2048,2048,2048", promotable=False)
    large_wrap64_4096 = parse_case("wrap64-large-4096:4096,4096,4096", promotable=False)
    algebra_rank_update = parse_case("rank-update-512x512x128:512,512,128")
    algebra_f4_dense = parse_case("f4-dense-512:512,512,512")
    algebra_fglm_mulmat = parse_case("fglm-mulmat-256:256,256,256")
    algebra_crt_export = parse_case("crt-export-512:512,512,512")
    fhe_ntt_pressure = parse_case("ntt-log12-proxy:4096,1,4096", promotable=False)
    fhe_key_switch = parse_case("key-switch-1024x256x1024:1024,256,1024")
    fhe_linear_layer = parse_case("linear-layer-1024x128x1024:1024,128,1024")
    fhe_chain = parse_case("ckks-chain-256:256,256,256")
    adaptive_256 = parse_case("adaptive-bands-256:256,256,512,64,64,adaptive-bands", adaptive=True)
    adaptive_rect = parse_case("adaptive-bands-rect:512,1024,512,128,128,adaptive-bands", adaptive=True)
    adaptive_1024 = parse_case("adaptive-bands-1024:1024,1024,1024,128,128,adaptive-bands", adaptive=True)
    bound_discovery_i64_256_static = SweepCase(
        "bound-discovery-i64-256-static-global",
        256,
        256,
        512,
        tile_m=64,
        tile_n=64,
        bound_mode="global",
        input_profile="adaptive-bands",
    )
    bound_discovery_i64_256_scan = SweepCase(
        "bound-discovery-i64-256-input-scan-global",
        256,
        256,
        512,
        tile_m=64,
        tile_n=64,
        bound_mode="global",
        input_profile="adaptive-bands",
    )
    bound_discovery_i64_256_proof = SweepCase(
        "bound-discovery-i64-256-proof-mask-per-tile",
        256,
        256,
        512,
        tile_m=64,
        tile_n=64,
        bound_mode="per-tile",
        input_profile="adaptive-bands",
        require_adaptive=True,
    )
    bound_discovery_u64_rect_static = SweepCase(
        "bound-discovery-u64-rect-static-global",
        512,
        1024,
        512,
        tile_m=128,
        tile_n=128,
        bound_mode="global",
        input_profile="adaptive-bands",
    )
    bound_discovery_u64_rect_scan = SweepCase(
        "bound-discovery-u64-rect-input-scan-global",
        512,
        1024,
        512,
        tile_m=128,
        tile_n=128,
        bound_mode="global",
        input_profile="adaptive-bands",
    )
    bound_discovery_u64_rect_proof = SweepCase(
        "bound-discovery-u64-rect-proof-mask-per-tile",
        512,
        1024,
        512,
        tile_m=128,
        tile_n=128,
        bound_mode="per-tile",
        input_profile="adaptive-bands",
        require_adaptive=True,
    )
    bound_discovery_i64_1024_static = SweepCase(
        "bound-discovery-i64-1024-static-global",
        1024,
        1024,
        1024,
        tile_m=128,
        tile_n=128,
        bound_mode="global",
        input_profile="adaptive-bands",
    )
    bound_discovery_i64_1024_scan = SweepCase(
        "bound-discovery-i64-1024-input-scan-global",
        1024,
        1024,
        1024,
        tile_m=128,
        tile_n=128,
        bound_mode="global",
        input_profile="adaptive-bands",
    )
    bound_discovery_i64_1024_proof = SweepCase(
        "bound-discovery-i64-1024-proof-mask-per-tile",
        1024,
        1024,
        1024,
        tile_m=128,
        tile_n=128,
        bound_mode="per-tile",
        input_profile="adaptive-bands",
        require_adaptive=True,
    )
    tile_512_64 = SweepCase("tile-512-64x64", 512, 512, 512, 64, 64, "global", "uniform-small", False, False)
    tile_512_256 = SweepCase("tile-512-256x128", 512, 512, 512, 256, 128, "global", "uniform-small", False, False)
    tile_1024_64 = SweepCase("tile-1024-64x128", 1024, 1024, 1024, 64, 128, "global", "uniform-small", False, False)
    tile_1024_256 = SweepCase("tile-1024-256x256", 1024, 1024, 1024, 256, 256, "global", "uniform-small", False, False)
    tile_finite_2048_64 = SweepCase("finite-2048-64x64", 2048, 2048, 2048, 64, 64, "global", "uniform-small", False, False)
    tile_finite_2048_256 = SweepCase("finite-2048-256x128", 2048, 2048, 2048, 256, 128, "global", "uniform-small", False, False)

    bounded_gpu_backends = ("hip-direct", "hip-vector-alu-int64", "hipblaslt", "ck", "rocwmma")
    bounded_release_backends = ("cpu", "hip-direct", "hip-vector-alu-int64", "hipblaslt", "ck", "rocwmma")
    bounded_per_tile_backends = ("cpu", "hip-direct", "hip-vector-alu-int64", "ck", "rocwmma")
    direct_oneshot_backends = ("cpu", "hip-direct")
    accelerator_backends = ("hip-direct", "hipblaslt", "ck", "rocwmma")

    return {
        "layout-search": [
            ScenarioItem(
                "layout-search",
                "bounded-i64-prefix9-final-export",
                "bounded-i64",
                layout_512,
                "bounded i64 fixed-prefix RNS final-export layout candidate",
                "host_export",
                "anchors layout search against the current host-export RNS contract before trying lazy or fused variants",
                backends=bounded_gpu_backends,
                prefix_policy="fixed-requested",
                max_prefix=9,
                next_op_hint="final-export",
                metadata={
                    "workflow_name": "end_to_end_layout_search",
                    "layout_role": "rns_residue_planes_final_export",
                    "pack_layout_family": "centered_rns_residue_planes",
                    "output_domain_requirement": "host_export",
                    "promotion_scope": "layout_comparison_only",
                },
            ),
            ScenarioItem(
                "layout-search",
                "bounded-i64-prefix9-rns-next",
                "bounded-i64",
                layout_512,
                "bounded i64 fixed-prefix RNS next-GEMM layout candidate",
                "host_export",
                "keeps next-op intent visible when comparing final-export and residue-current layout choices",
                backends=bounded_gpu_backends,
                prefix_policy="fixed-requested",
                max_prefix=9,
                next_op_hint="rns-gemm",
                metadata={
                    "workflow_name": "end_to_end_layout_search",
                    "layout_role": "rns_residue_planes_next_rns_gemm",
                    "pack_layout_family": "centered_rns_residue_planes",
                    "output_domain_requirement": "rns_gemm_continuation",
                    "promotion_scope": "layout_comparison_only",
                },
            ),
            ScenarioItem(
                "layout-search",
                "exact-wide-signed-prefix20-final-export",
                "exact-wide-signed",
                layout_512,
                "exact-wide signed prefix-20 final limb-output layout candidate",
                "exact_wide_signed_limbs",
                "separates fixed limb export layout from bounded RNS host-output behavior",
                backends=EXACT_WIDE_BACKENDS,
                prefix_policy="fixed-requested",
                max_prefix=20,
                exact_wide_limb_counts=(4,),
                next_op_hint="final-export",
                metadata={
                    "workflow_name": "end_to_end_layout_search",
                    "layout_role": "exact_wide_prefix20_limb_export",
                    "pack_layout_family": "centered_rns_residue_planes",
                    "output_domain_requirement": "host_limb_export",
                    "promotion_scope": "layout_comparison_only",
                },
            ),
            ScenarioItem(
                "layout-search",
                "exact-wide-signed-prefix20-rns-next",
                "exact-wide-signed",
                layout_512,
                "exact-wide signed prefix-20 RNS-next layout candidate",
                "residue_current_rns",
                "tests whether exact-wide work should stay residue-current before final limb export",
                backends=EXACT_WIDE_BACKENDS,
                prefix_policy="fixed-requested",
                max_prefix=20,
                exact_wide_limb_counts=(4,),
                next_op_hint="rns-gemm",
                metadata={
                    "workflow_name": "end_to_end_layout_search",
                    "layout_role": "exact_wide_prefix20_next_rns_gemm",
                    "pack_layout_family": "centered_rns_residue_planes",
                    "output_domain_requirement": "lazy_export",
                    "promotion_scope": "layout_comparison_only",
                },
            ),
            ScenarioItem(
                "layout-search",
                "finite-ring-hot-modulus-layout",
                "finite-u8-ring",
                layout_512,
                "finite-ring u8 hot-modulus layout candidate",
                "finite_u8_canonical_host_export",
                "keeps finite-ring residue layout decisions separate from bounded RNS prefix decisions",
                backends=FINITE_BACKENDS,
                finite_moduli=(251, 255, 256),
                next_op_hint="final-export",
                metadata={
                    "workflow_name": "end_to_end_layout_search",
                    "layout_role": "finite_u8_ring_residue_layout",
                    "pack_layout_family": "centered_i8_finite_residues",
                    "output_domain_requirement": "finite_host_export",
                    "promotion_scope": "layout_comparison_only",
                    "modulus_role": "hot_modulus",
                },
            ),
            ScenarioItem(
                "layout-search",
                "finite-field-hot-prime-layout",
                "finite-u8-field",
                layout_512,
                "finite-field u8 hot-prime layout candidate",
                "finite_u8_canonical_host_export",
                "separates prime-field layout behavior from composite finite-ring behavior",
                backends=FINITE_BACKENDS,
                finite_moduli=(251,),
                next_op_hint="final-export",
                metadata={
                    "workflow_name": "end_to_end_layout_search",
                    "layout_role": "finite_u8_prime_field_residue_layout",
                    "pack_layout_family": "centered_i8_finite_residues",
                    "output_domain_requirement": "finite_host_export",
                    "promotion_scope": "layout_comparison_only",
                    "modulus_role": "hot_prime",
                    "prime_or_composite": "prime",
                },
            ),
            ScenarioItem(
                "layout-search",
                "wrap64-direct-byte-layout",
                "wrap-u64",
                wrap64_512,
                "strict wrap64 byte-limb direct layout candidate",
                "low64_wrap_u64_host_output",
                "keeps byte-limb wraparound layout decisions separate from odd-modulus RNS and finite-u8 paths",
                backends=("hip-direct",),
                metadata={
                    "workflow_name": "end_to_end_layout_search",
                    "layout_role": "wrap64_byte_limb_low64_output",
                    "pack_layout_family": "unsigned_byte_limb_planes",
                    "output_domain_requirement": "low64_host_export",
                    "promotion_scope": "layout_comparison_only",
                },
            ),
        ],
        "generated-prefix-reducers": [
            *[
                ScenarioItem(
                    "generated-prefix-reducers",
                    f"bounded-i64-prefix{prefix}",
                    "bounded-i64",
                    small_128,
                    f"direct-HIP bounded i64 fixed-prefix {prefix} generated reducer evidence",
                    "host_export",
                    "proves fixed-prefix generated reducer identity and no-divide ISA gates across bounded prefixes",
                    backends=("hip-direct",),
                    prefix_policy="fixed-requested",
                    max_prefix=prefix,
                    next_op_hint="final-export",
                    metadata={
                        "workflow_name": "generated_prefix_reducer",
                        "prefix": prefix,
                        "isa_gate": "no_integer_divide_expected",
                    },
                )
                for prefix in (1, 3, 5, 9)
            ],
            *[
                ScenarioItem(
                    "generated-prefix-reducers",
                    f"bounded-u64-prefix{prefix}",
                    "bounded-u64",
                    small_128,
                    f"direct-HIP bounded u64 fixed-prefix {prefix} generated reducer evidence",
                    "host_export",
                    "mirrors signed-prefix reducer coverage for unsigned bounded semantics",
                    backends=("hip-direct",),
                    prefix_policy="fixed-requested",
                    max_prefix=prefix,
                    next_op_hint="final-export",
                    metadata={
                        "workflow_name": "generated_prefix_reducer",
                        "prefix": prefix,
                        "isa_gate": "no_integer_divide_expected",
                    },
                )
                for prefix in (1, 3, 5, 9)
            ],
            ScenarioItem(
                "generated-prefix-reducers",
                "exact-wide-signed-prefix20",
                "exact-wide-signed",
                small_128,
                "direct-HIP exact-wide fixed-prefix 20 generated reducer evidence",
                "exact_wide_signed_limbs",
                "keeps exact-wide prefix-20 generated reducer identity tied to a compact smoke shape",
                backends=("hip-direct",),
                prefix_policy="fixed-requested",
                max_prefix=20,
                exact_wide_limb_counts=(4,),
                next_op_hint="final-export",
                metadata={
                    "workflow_name": "generated_prefix_reducer",
                    "prefix": 20,
                    "isa_gate": "no_integer_divide_expected",
                },
            ),
        ],
        "multi-modulus-pack": [
            *[
                ScenarioItem(
                    "multi-modulus-pack",
                    f"bounded-i64-prefix{prefix}",
                    "bounded-i64",
                    pack_heavy_128,
                    f"pack-heavy direct-HIP bounded i64 fixed-prefix {prefix}",
                    "host_export",
                    "separates multi-modulus fixed-prefix pack cost from large output GEMM effects",
                    backends=("hip-direct",),
                    prefix_policy="fixed-requested",
                    max_prefix=prefix,
                    next_op_hint="final-export",
                    metadata={"workflow_name": "multi_modulus_pack", "prefix": prefix},
                )
                for prefix in (1, 3, 5, 9)
            ],
            ScenarioItem(
                "multi-modulus-pack",
                "exact-wide-prefix20",
                "exact-wide-unsigned",
                pack_heavy_128,
                "pack-heavy direct-HIP exact-wide fixed-prefix 20",
                "exact_wide_unsigned_limbs",
                "measures the prefix-20 pack/export shape without changing exact-wide semantics",
                backends=("hip-direct",),
                prefix_policy="fixed-requested",
                max_prefix=20,
                exact_wide_limb_counts=(4,),
                next_op_hint="final-export",
                metadata={"workflow_name": "multi_modulus_pack", "prefix": 20},
            ),
        ],
        "residue-channel-fusion": [
            ScenarioItem(
                "residue-channel-fusion",
                "bounded-i64-small64",
                "bounded-i64",
                small_64,
                "benchmark-only direct-HIP width-3 residue-channel fusion experiment",
                "host_export",
                "compares width-3 residue grouping against the ordinary transient uniform-small path at setup-dominated size",
                backends=("hip-direct",),
                prefix_policy="fixed-requested",
                max_prefix=9,
                next_op_hint="final-export",
                residue_channel_fusion=True,
                metadata={"workflow_name": "residue_channel_fusion", "residue_group_width": 3},
            ),
            ScenarioItem(
                "residue-channel-fusion",
                "bounded-u64-small128",
                "bounded-u64",
                small_128,
                "benchmark-only direct-HIP width-3 residue-channel fusion experiment",
                "host_export",
                "keeps unsigned bounded semantics represented in the fusion comparison surface",
                backends=("hip-direct",),
                prefix_policy="fixed-requested",
                max_prefix=9,
                next_op_hint="final-export",
                residue_channel_fusion=True,
                metadata={"workflow_name": "residue_channel_fusion", "residue_group_width": 3},
            ),
        ],
        "fused-pack-gemm-small": [
            ScenarioItem(
                "fused-pack-gemm-small",
                "bounded-i64-oneshot64",
                "bounded-i64",
                small_64,
                "public one-shot bounded i64 native-input pack+GEMM comparison",
                "host_export",
                "compares CPU, direct-HIP, and vector-ALU native-input surfaces where setup dominates",
                backends=("cpu", "hip-direct"),
                oneshot=True,
                next_op_hint="final-export",
            ),
            ScenarioItem(
                "fused-pack-gemm-small",
                "bounded-u64-persistent128",
                "bounded-u64",
                small_128,
                "persistent bounded u64 small-shape comparison including vector-ALU",
                "host_export",
                "contrasts persistent direct-HIP RNS, vector-ALU, and accelerator candidates at small shapes",
                backends=("cpu", "hip-direct", "hip-vector-alu-int64", "ck", "rocwmma"),
                next_op_hint="final-export",
            ),
            ScenarioItem(
                "fused-pack-gemm-small",
                "finite-ring-u8-oneshot64",
                "finite-u8-ring",
                small_64,
                "public one-shot finite-ring u8 native-input comparison",
                "finite_u8_canonical_host_export",
                "captures finite-u8 small one-shot behavior separately from persistent residue storage",
                backends=("cpu", "hip-direct"),
                finite_moduli=(251, 255),
                oneshot=True,
                next_op_hint="final-export",
            ),
            ScenarioItem(
                "fused-pack-gemm-small",
                "finite-field-u8-persistent128",
                "finite-u8-field",
                small_128,
                "persistent finite-field u8 small-shape accelerator comparison",
                "finite_u8_canonical_host_export",
                "keeps field-u8 small persistent evidence available for accelerator triage",
                backends=("cpu", "hip-direct", "hipblaslt", "ck", "rocwmma"),
                finite_moduli=(251,),
                next_op_hint="final-export",
            ),
        ],
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
        "bound-discovery": [
            ScenarioItem(
                "bound-discovery",
                "bounded-i64-256-static-global",
                "bounded-i64",
                bound_discovery_i64_256_static,
                "bound_discovery_static_profile_baseline",
                "host_export",
                "anchors adaptive-band input timing to the ordinary static global bound contract",
                backends=bounded_release_backends,
                bound_source="static-profile",
                metadata={
                    "workflow_name": "bound_discovery_proof_mask_release_matrix",
                    "bound_discovery_role": "static_global_baseline",
                    "promotion_scope": "baseline_only",
                    "validation_contract": "setup_inclusive_static_global_vs_input_scan_vs_proof_mask",
                },
            ),
            ScenarioItem(
                "bound-discovery",
                "bounded-i64-256-input-scan-global",
                "bounded-i64",
                bound_discovery_i64_256_scan,
                "bound_discovery_global_input_scan_candidate",
                "host_export",
                "measures whether one-time global input scanning reduces enough residue work to beat static bounds",
                backends=bounded_release_backends,
                bound_source="input-scan",
                metadata={
                    "workflow_name": "bound_discovery_proof_mask_release_matrix",
                    "bound_discovery_role": "input_scan_global_candidate",
                    "promotion_scope": "workload_comparison_only",
                    "validation_contract": "setup_inclusive_static_global_vs_input_scan_vs_proof_mask",
                },
            ),
            ScenarioItem(
                "bound-discovery",
                "bounded-i64-256-proof-mask-per-tile",
                "bounded-i64",
                bound_discovery_i64_256_proof,
                "bound_discovery_proof_mask_candidate",
                "host_export",
                "checks whether per-tile exact bounds plus zero row/column proofs beat static/global scan baselines",
                backends=bounded_per_tile_backends,
                bound_source="input-scan",
                metadata={
                    "workflow_name": "bound_discovery_proof_mask_release_matrix",
                    "bound_discovery_role": "proof_mask_per_tile_candidate",
                    "promotion_scope": "workload_comparison_only",
                    "validation_contract": "setup_inclusive_static_global_vs_input_scan_vs_proof_mask",
                    "proof_mask_expected": True,
                },
            ),
            ScenarioItem(
                "bound-discovery",
                "bounded-u64-rect-static-global",
                "bounded-u64",
                bound_discovery_u64_rect_static,
                "bound_discovery_static_profile_baseline",
                "host_export",
                "keeps rectangular unsigned adaptive-band input timing anchored to a static global bound baseline",
                backends=bounded_release_backends,
                bound_source="static-profile",
                metadata={
                    "workflow_name": "bound_discovery_proof_mask_release_matrix",
                    "bound_discovery_role": "static_global_baseline",
                    "promotion_scope": "baseline_only",
                    "validation_contract": "setup_inclusive_static_global_vs_input_scan_vs_proof_mask",
                },
            ),
            ScenarioItem(
                "bound-discovery",
                "bounded-u64-rect-input-scan-global",
                "bounded-u64",
                bound_discovery_u64_rect_scan,
                "bound_discovery_global_input_scan_candidate",
                "host_export",
                "tests global input-bound discovery on the rectangular bounded-u64 adaptive-band workload",
                backends=bounded_release_backends,
                bound_source="input-scan",
                metadata={
                    "workflow_name": "bound_discovery_proof_mask_release_matrix",
                    "bound_discovery_role": "input_scan_global_candidate",
                    "promotion_scope": "workload_comparison_only",
                    "validation_contract": "setup_inclusive_static_global_vs_input_scan_vs_proof_mask",
                },
            ),
            ScenarioItem(
                "bound-discovery",
                "bounded-u64-rect-proof-mask-per-tile",
                "bounded-u64",
                bound_discovery_u64_rect_proof,
                "bound_discovery_proof_mask_candidate",
                "host_export",
                "covers proof-mask behavior on the rectangular bounded-u64 adaptive-band workload",
                backends=bounded_per_tile_backends,
                bound_source="input-scan",
                metadata={
                    "workflow_name": "bound_discovery_proof_mask_release_matrix",
                    "bound_discovery_role": "proof_mask_per_tile_candidate",
                    "promotion_scope": "workload_comparison_only",
                    "validation_contract": "setup_inclusive_static_global_vs_input_scan_vs_proof_mask",
                    "proof_mask_expected": True,
                },
            ),
            ScenarioItem(
                "bound-discovery",
                "bounded-i64-1024-static-global",
                "bounded-i64",
                bound_discovery_i64_1024_static,
                "bound_discovery_static_profile_baseline",
                "host_export",
                "anchors the larger adaptive-band workload to the ordinary static global bound contract",
                backends=bounded_release_backends,
                bound_source="static-profile",
                metadata={
                    "workflow_name": "bound_discovery_proof_mask_release_matrix",
                    "bound_discovery_role": "static_global_baseline",
                    "promotion_scope": "baseline_only",
                    "validation_contract": "setup_inclusive_static_global_vs_input_scan_vs_proof_mask",
                },
            ),
            ScenarioItem(
                "bound-discovery",
                "bounded-i64-1024-input-scan-global",
                "bounded-i64",
                bound_discovery_i64_1024_scan,
                "bound_discovery_global_input_scan_candidate",
                "host_export",
                "checks whether global input scanning still pays off once launch overhead is less dominant",
                backends=bounded_release_backends,
                bound_source="input-scan",
                metadata={
                    "workflow_name": "bound_discovery_proof_mask_release_matrix",
                    "bound_discovery_role": "input_scan_global_candidate",
                    "promotion_scope": "workload_comparison_only",
                    "validation_contract": "setup_inclusive_static_global_vs_input_scan_vs_proof_mask",
                },
            ),
            ScenarioItem(
                "bound-discovery",
                "bounded-i64-1024-proof-mask-per-tile",
                "bounded-i64",
                bound_discovery_i64_1024_proof,
                "bound_discovery_proof_mask_candidate",
                "host_export",
                "checks larger-shape per-tile proof-mask behavior against global static and input-scan baselines",
                backends=bounded_per_tile_backends,
                bound_source="input-scan",
                metadata={
                    "workflow_name": "bound_discovery_proof_mask_release_matrix",
                    "bound_discovery_role": "proof_mask_per_tile_candidate",
                    "promotion_scope": "workload_comparison_only",
                    "validation_contract": "setup_inclusive_static_global_vs_input_scan_vs_proof_mask",
                    "proof_mask_expected": True,
                },
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
        "reuse-contract": [
            ScenarioItem(
                "reuse-contract",
                "bounded-i64-1024-baseline",
                "bounded-i64",
                repeated_b_1024,
                "large bounded i64 non-reuse baseline for explicit reuse contracts",
                "host_export",
                "anchors stable-A and stable-A+B reuse comparisons to the ordinary per-repeat pack contract",
                backends=bounded_release_backends,
                prefix_policy="fixed-requested",
                max_prefix=9,
                metadata={
                    "workflow_name": "reuse_contract_release_matrix",
                    "reuse_contract_role": "nonreuse_baseline",
                    "promotion_scope": "baseline_only",
                    "validation_contract": "setup_inclusive_nonreuse_vs_reuse_a_reuse_b_reuse_ab",
                },
            ),
            ScenarioItem(
                "reuse-contract",
                "bounded-u64-1024-baseline",
                "bounded-u64",
                repeated_b_1024,
                "large bounded u64 non-reuse baseline for explicit reuse contracts",
                "host_export",
                "anchors unsigned stable-A and stable-A+B reuse comparisons to the ordinary per-repeat pack contract",
                backends=bounded_release_backends,
                prefix_policy="fixed-requested",
                max_prefix=9,
                metadata={
                    "workflow_name": "reuse_contract_release_matrix",
                    "reuse_contract_role": "nonreuse_baseline",
                    "promotion_scope": "baseline_only",
                    "validation_contract": "setup_inclusive_nonreuse_vs_reuse_a_reuse_b_reuse_ab",
                },
            ),
            ScenarioItem(
                "reuse-contract",
                "bounded-i64-1024-reuse-a",
                "bounded-i64",
                repeated_b_1024,
                "bounded i64 workload with stable LHS packed once before warmups",
                "host_export",
                "checks whether stable-A setup amortizes against same-backend and fastest non-reuse baselines",
                backends=bounded_release_backends,
                pack_mode="prepacked_reuse_a",
                prefix_policy="fixed-requested",
                max_prefix=9,
                metadata={
                    "workflow_name": "reuse_contract_release_matrix",
                    "reuse_contract_role": "stable_a_candidate",
                    "promotion_scope": "explicit_reuse_contract_only",
                    "reuse_contract": "stable_lhs_prepacked_before_warmups",
                    "validation_contract": "setup_inclusive_nonreuse_vs_reuse_a_reuse_b_reuse_ab",
                    "setup_cost_policy": "prepack_setup_us_excluded_from_per_repeat_included_in_review",
                },
            ),
            ScenarioItem(
                "reuse-contract",
                "bounded-u64-1024-reuse-a",
                "bounded-u64",
                repeated_b_1024,
                "bounded u64 workload with stable LHS packed once before warmups",
                "host_export",
                "checks whether unsigned stable-A setup amortizes against same-backend and fastest non-reuse baselines",
                backends=bounded_release_backends,
                pack_mode="prepacked_reuse_a",
                prefix_policy="fixed-requested",
                max_prefix=9,
                metadata={
                    "workflow_name": "reuse_contract_release_matrix",
                    "reuse_contract_role": "stable_a_candidate",
                    "promotion_scope": "explicit_reuse_contract_only",
                    "reuse_contract": "stable_lhs_prepacked_before_warmups",
                    "validation_contract": "setup_inclusive_nonreuse_vs_reuse_a_reuse_b_reuse_ab",
                    "setup_cost_policy": "prepack_setup_us_excluded_from_per_repeat_included_in_review",
                },
            ),
            ScenarioItem(
                "reuse-contract",
                "bounded-i64-1024-reuse-b",
                "bounded-i64",
                repeated_b_1024,
                "bounded i64 workload with stable RHS packed once before warmups",
                "host_export",
                "checks whether stable-B setup amortizes in the same matrix as stable-A and stable-A+B",
                backends=bounded_release_backends,
                pack_mode="prepacked_reuse_b",
                prefix_policy="fixed-requested",
                max_prefix=9,
                metadata={
                    "workflow_name": "reuse_contract_release_matrix",
                    "reuse_contract_role": "stable_b_candidate",
                    "promotion_scope": "explicit_reuse_contract_only",
                    "reuse_contract": "stable_rhs_prepacked_before_warmups",
                    "validation_contract": "setup_inclusive_nonreuse_vs_reuse_a_reuse_b_reuse_ab",
                    "setup_cost_policy": "prepack_setup_us_excluded_from_per_repeat_included_in_review",
                },
            ),
            ScenarioItem(
                "reuse-contract",
                "bounded-u64-1024-reuse-b",
                "bounded-u64",
                repeated_b_1024,
                "bounded u64 workload with stable RHS packed once before warmups",
                "host_export",
                "checks whether unsigned stable-B setup amortizes in the same matrix as stable-A and stable-A+B",
                backends=bounded_release_backends,
                pack_mode="prepacked_reuse_b",
                prefix_policy="fixed-requested",
                max_prefix=9,
                metadata={
                    "workflow_name": "reuse_contract_release_matrix",
                    "reuse_contract_role": "stable_b_candidate",
                    "promotion_scope": "explicit_reuse_contract_only",
                    "reuse_contract": "stable_rhs_prepacked_before_warmups",
                    "validation_contract": "setup_inclusive_nonreuse_vs_reuse_a_reuse_b_reuse_ab",
                    "setup_cost_policy": "prepack_setup_us_excluded_from_per_repeat_included_in_review",
                },
            ),
            ScenarioItem(
                "reuse-contract",
                "bounded-i64-1024-reuse-ab",
                "bounded-i64",
                repeated_b_1024,
                "bounded i64 workload with stable LHS and RHS packed once before warmups",
                "host_export",
                "checks whether full stable-input setup amortizes before converting A+B reuse into a workload contract",
                backends=bounded_release_backends,
                pack_mode="prepacked_reuse",
                prefix_policy="fixed-requested",
                max_prefix=9,
                metadata={
                    "workflow_name": "reuse_contract_release_matrix",
                    "reuse_contract_role": "stable_ab_candidate",
                    "promotion_scope": "explicit_reuse_contract_only",
                    "reuse_contract": "stable_lhs_rhs_prepacked_before_warmups",
                    "validation_contract": "setup_inclusive_nonreuse_vs_reuse_a_reuse_b_reuse_ab",
                    "setup_cost_policy": "prepack_setup_us_excluded_from_per_repeat_included_in_review",
                },
            ),
            ScenarioItem(
                "reuse-contract",
                "bounded-u64-1024-reuse-ab",
                "bounded-u64",
                repeated_b_1024,
                "bounded u64 workload with stable LHS and RHS packed once before warmups",
                "host_export",
                "checks whether unsigned full stable-input setup amortizes before converting A+B reuse into a workload contract",
                backends=bounded_release_backends,
                pack_mode="prepacked_reuse",
                prefix_policy="fixed-requested",
                max_prefix=9,
                metadata={
                    "workflow_name": "reuse_contract_release_matrix",
                    "reuse_contract_role": "stable_ab_candidate",
                    "promotion_scope": "explicit_reuse_contract_only",
                    "reuse_contract": "stable_lhs_rhs_prepacked_before_warmups",
                    "validation_contract": "setup_inclusive_nonreuse_vs_reuse_a_reuse_b_reuse_ab",
                    "setup_cost_policy": "prepack_setup_us_excluded_from_per_repeat_included_in_review",
                },
            ),
            ScenarioItem(
                "reuse-contract",
                "bounded-i64-2048-baseline",
                "bounded-i64",
                large_2048,
                "large bounded i64 non-reuse baseline for explicit reuse contracts",
                "host_export",
                "keeps 2048 stable-input reuse tied to the same large-shape baseline matrix",
                backends=bounded_release_backends,
                metadata={
                    "workflow_name": "reuse_contract_release_matrix",
                    "reuse_contract_role": "nonreuse_baseline",
                    "promotion_scope": "baseline_only",
                    "validation_contract": "setup_inclusive_nonreuse_vs_reuse_a_reuse_b_reuse_ab",
                },
            ),
            ScenarioItem(
                "reuse-contract",
                "bounded-u64-2048-baseline",
                "bounded-u64",
                large_2048,
                "large bounded u64 non-reuse baseline for explicit reuse contracts",
                "host_export",
                "keeps unsigned 2048 stable-input reuse tied to the same large-shape baseline matrix",
                backends=bounded_release_backends,
                metadata={
                    "workflow_name": "reuse_contract_release_matrix",
                    "reuse_contract_role": "nonreuse_baseline",
                    "promotion_scope": "baseline_only",
                    "validation_contract": "setup_inclusive_nonreuse_vs_reuse_a_reuse_b_reuse_ab",
                },
            ),
            ScenarioItem(
                "reuse-contract",
                "bounded-i64-2048-reuse-a",
                "bounded-i64",
                large_2048,
                "large bounded i64 workload with stable LHS packed once before warmups",
                "host_export",
                "tests whether large stable-A reuse survives setup-inclusive review",
                backends=bounded_release_backends,
                pack_mode="prepacked_reuse_a",
                metadata={
                    "workflow_name": "reuse_contract_release_matrix",
                    "reuse_contract_role": "stable_a_candidate",
                    "promotion_scope": "explicit_reuse_contract_only",
                    "reuse_contract": "large_stable_lhs_prepacked_before_warmups",
                    "validation_contract": "setup_inclusive_nonreuse_vs_reuse_a_reuse_b_reuse_ab",
                    "setup_cost_policy": "prepack_setup_us_excluded_from_per_repeat_included_in_review",
                },
            ),
            ScenarioItem(
                "reuse-contract",
                "bounded-u64-2048-reuse-a",
                "bounded-u64",
                large_2048,
                "large bounded u64 workload with stable LHS packed once before warmups",
                "host_export",
                "tests whether large unsigned stable-A reuse survives setup-inclusive review",
                backends=bounded_release_backends,
                pack_mode="prepacked_reuse_a",
                metadata={
                    "workflow_name": "reuse_contract_release_matrix",
                    "reuse_contract_role": "stable_a_candidate",
                    "promotion_scope": "explicit_reuse_contract_only",
                    "reuse_contract": "large_stable_lhs_prepacked_before_warmups",
                    "validation_contract": "setup_inclusive_nonreuse_vs_reuse_a_reuse_b_reuse_ab",
                    "setup_cost_policy": "prepack_setup_us_excluded_from_per_repeat_included_in_review",
                },
            ),
            ScenarioItem(
                "reuse-contract",
                "bounded-i64-2048-reuse-b",
                "bounded-i64",
                large_2048,
                "large bounded i64 workload with stable RHS packed once before warmups",
                "host_export",
                "keeps large stable-B reuse in the same explicit contract matrix as A and A+B",
                backends=bounded_release_backends,
                pack_mode="prepacked_reuse_b",
                metadata={
                    "workflow_name": "reuse_contract_release_matrix",
                    "reuse_contract_role": "stable_b_candidate",
                    "promotion_scope": "explicit_reuse_contract_only",
                    "reuse_contract": "large_stable_rhs_prepacked_before_warmups",
                    "validation_contract": "setup_inclusive_nonreuse_vs_reuse_a_reuse_b_reuse_ab",
                    "setup_cost_policy": "prepack_setup_us_excluded_from_per_repeat_included_in_review",
                },
            ),
            ScenarioItem(
                "reuse-contract",
                "bounded-u64-2048-reuse-b",
                "bounded-u64",
                large_2048,
                "large bounded u64 workload with stable RHS packed once before warmups",
                "host_export",
                "keeps large unsigned stable-B reuse in the same explicit contract matrix as A and A+B",
                backends=bounded_release_backends,
                pack_mode="prepacked_reuse_b",
                metadata={
                    "workflow_name": "reuse_contract_release_matrix",
                    "reuse_contract_role": "stable_b_candidate",
                    "promotion_scope": "explicit_reuse_contract_only",
                    "reuse_contract": "large_stable_rhs_prepacked_before_warmups",
                    "validation_contract": "setup_inclusive_nonreuse_vs_reuse_a_reuse_b_reuse_ab",
                    "setup_cost_policy": "prepack_setup_us_excluded_from_per_repeat_included_in_review",
                },
            ),
            ScenarioItem(
                "reuse-contract",
                "bounded-i64-2048-reuse-ab",
                "bounded-i64",
                large_2048,
                "large bounded i64 workload with stable LHS and RHS packed once before warmups",
                "host_export",
                "tests whether large full stable-input reuse survives setup-inclusive review",
                backends=bounded_release_backends,
                pack_mode="prepacked_reuse",
                metadata={
                    "workflow_name": "reuse_contract_release_matrix",
                    "reuse_contract_role": "stable_ab_candidate",
                    "promotion_scope": "explicit_reuse_contract_only",
                    "reuse_contract": "large_stable_lhs_rhs_prepacked_before_warmups",
                    "validation_contract": "setup_inclusive_nonreuse_vs_reuse_a_reuse_b_reuse_ab",
                    "setup_cost_policy": "prepack_setup_us_excluded_from_per_repeat_included_in_review",
                },
            ),
            ScenarioItem(
                "reuse-contract",
                "bounded-u64-2048-reuse-ab",
                "bounded-u64",
                large_2048,
                "large bounded u64 workload with stable LHS and RHS packed once before warmups",
                "host_export",
                "tests whether large unsigned full stable-input reuse survives setup-inclusive review",
                backends=bounded_release_backends,
                pack_mode="prepacked_reuse",
                metadata={
                    "workflow_name": "reuse_contract_release_matrix",
                    "reuse_contract_role": "stable_ab_candidate",
                    "promotion_scope": "explicit_reuse_contract_only",
                    "reuse_contract": "large_stable_lhs_rhs_prepacked_before_warmups",
                    "validation_contract": "setup_inclusive_nonreuse_vs_reuse_a_reuse_b_reuse_ab",
                    "setup_cost_policy": "prepack_setup_us_excluded_from_per_repeat_included_in_review",
                },
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
        "finite-generic-moduli": [
            ScenarioItem(
                "finite-generic-moduli",
                "ring-prime-127-512",
                "finite-u8-ring",
                finite_generic_512,
                "finite-ring u8 generic prime modulus workload",
                "finite_u8_canonical_host_export",
                "checks non-hot prime modulus behavior separately from the specialized 251/255/256 accelerator path",
                backends=direct_oneshot_backends,
                finite_moduli=(127,),
                metadata={
                    "workflow_name": "finite_u8_generic_modulus",
                    "domain_family": "finite_ring_u8",
                    "phase_label": "generic_prime_modulus_probe",
                    "prime_or_composite": "prime",
                    "modulus_role": "generic_non_hot_prime",
                    "promotion_scope": "feature_boundary",
                },
            ),
            ScenarioItem(
                "finite-generic-moduli",
                "field-prime-127-512",
                "finite-u8-field",
                finite_generic_512,
                "finite-field u8 generic prime modulus workload",
                "finite_u8_canonical_host_export",
                "separates field-prime correctness and timing from the current field-251 hot path",
                backends=direct_oneshot_backends,
                finite_moduli=(127,),
                metadata={
                    "workflow_name": "finite_u8_generic_modulus",
                    "domain_family": "prime_field_u8",
                    "phase_label": "generic_prime_field_probe",
                    "prime_or_composite": "prime",
                    "modulus_role": "generic_non_hot_prime",
                    "promotion_scope": "feature_boundary",
                },
            ),
            ScenarioItem(
                "finite-generic-moduli",
                "ring-composite-253-512",
                "finite-u8-ring",
                finite_generic_512,
                "finite-ring u8 generic composite modulus workload",
                "finite_u8_canonical_host_export",
                "checks composite ring behavior without reusing the specialized 255/256 hot-modulus path",
                backends=direct_oneshot_backends,
                finite_moduli=(253,),
                metadata={
                    "workflow_name": "finite_u8_generic_modulus",
                    "domain_family": "finite_ring_u8",
                    "phase_label": "generic_composite_modulus_probe",
                    "prime_or_composite": "composite",
                    "modulus_role": "generic_non_hot_composite",
                    "promotion_scope": "feature_boundary",
                },
            ),
            ScenarioItem(
                "finite-generic-moduli",
                "ring-prime-127-2048",
                "finite-u8-ring",
                finite_generic_2048,
                "large finite-ring u8 generic prime modulus workload",
                "finite_u8_canonical_host_export",
                "pairs the large-shape matrix with a non-hot finite modulus to distinguish feature value from hot-modulus specialization",
                backends=direct_oneshot_backends,
                finite_moduli=(127,),
                metadata={
                    "workflow_name": "finite_u8_generic_modulus",
                    "domain_family": "finite_ring_u8",
                    "phase_label": "large_generic_prime_modulus_probe",
                    "prime_or_composite": "prime",
                    "modulus_role": "generic_non_hot_prime",
                    "large_shape_role": "finite_generic_modulus_probe",
                    "promotion_scope": "exploratory_only",
                },
            ),
            ScenarioItem(
                "finite-generic-moduli",
                "ring-composite-253-2048",
                "finite-u8-ring",
                finite_generic_2048,
                "large finite-ring u8 generic composite modulus workload",
                "finite_u8_canonical_host_export",
                "tests large composite ring behavior without implying accelerator coverage for arbitrary moduli",
                backends=direct_oneshot_backends,
                finite_moduli=(253,),
                metadata={
                    "workflow_name": "finite_u8_generic_modulus",
                    "domain_family": "finite_ring_u8",
                    "phase_label": "large_generic_composite_modulus_probe",
                    "prime_or_composite": "composite",
                    "modulus_role": "generic_non_hot_composite",
                    "large_shape_role": "finite_generic_modulus_probe",
                    "promotion_scope": "exploratory_only",
                },
            ),
        ],
        "native-to-rns-bridge": [
            ScenarioItem(
                "native-to-rns-bridge",
                "bounded-i64-64",
                "bounded-i64",
                small_64,
                "AUTO requested plan forced through Direct-HIP native-to-RNS conversion before RNS GEMM",
                "host_export",
                "turns the native-device-buffer to RNS consumer bridge into a schema-visible executable capture",
                backends=("auto",),
                native_to_rns_bridge=True,
                metadata={
                    "workflow_name": "native_to_rns_bridge",
                    "bridge_role": "device_native_to_rns_consumer_path",
                    "conversion_event_required": "native_i64_to_rns_kernel",
                    "selected_backend_requirement": "hip-direct",
                    "promotion_scope": "execution_path_evidence",
                },
            ),
            ScenarioItem(
                "native-to-rns-bridge",
                "bounded-u64-64",
                "bounded-u64",
                small_64,
                "AUTO requested plan forced through Direct-HIP native-to-RNS conversion before RNS GEMM",
                "host_export",
                "keeps the strong native bounded-u64 family connected to an explicit RNS consumer path",
                backends=("auto",),
                native_to_rns_bridge=True,
                metadata={
                    "workflow_name": "native_to_rns_bridge",
                    "bridge_role": "device_native_to_rns_consumer_path",
                    "conversion_event_required": "native_u64_to_rns_kernel",
                    "selected_backend_requirement": "hip-direct",
                    "promotion_scope": "execution_path_evidence",
                },
            ),
            ScenarioItem(
                "native-to-rns-bridge",
                "bounded-i64-128",
                "bounded-i64",
                small_128,
                "AUTO requested plan forced through Direct-HIP native-to-RNS conversion before RNS GEMM",
                "host_export",
                "checks that bridge timing remains event-visible beyond the tiniest smoke shape",
                backends=("auto",),
                native_to_rns_bridge=True,
                metadata={
                    "workflow_name": "native_to_rns_bridge",
                    "bridge_role": "device_native_to_rns_consumer_path",
                    "conversion_event_required": "native_i64_to_rns_kernel",
                    "selected_backend_requirement": "hip-direct",
                    "promotion_scope": "execution_path_evidence",
                },
            ),
            ScenarioItem(
                "native-to-rns-bridge",
                "bounded-u64-128",
                "bounded-u64",
                small_128,
                "AUTO requested plan forced through Direct-HIP native-to-RNS conversion before RNS GEMM",
                "host_export",
                "checks the bounded-u64 bridge path at a shape large enough to expose conversion overhead",
                backends=("auto",),
                native_to_rns_bridge=True,
                metadata={
                    "workflow_name": "native_to_rns_bridge",
                    "bridge_role": "device_native_to_rns_consumer_path",
                    "conversion_event_required": "native_u64_to_rns_kernel",
                    "selected_backend_requirement": "hip-direct",
                    "promotion_scope": "execution_path_evidence",
                },
            ),
        ],
        "vector-to-rns-chain": [
            ScenarioItem(
                "vector-to-rns-chain",
                "bounded-i64-64",
                "bounded-i64",
                small_64,
                "vector-ALU native producer result materialized into a Direct-HIP RNS consumer GEMM",
                "host_export",
                "turns the strong vector bounded path into an executable producer for an RNS-resident downstream GEMM",
                backends=("auto",),
                vector_to_rns_chain=True,
                metadata={
                    "workflow_name": "vector_to_rns_chain",
                    "bridge_role": "vector_native_output_to_direct_rns_consumer",
                    "producer_backend_requirement": "hip-vector-alu-int64",
                    "consumer_backend_requirement": "hip-direct",
                    "conversion_event_required": "native_i64_to_rns_kernel",
                    "selected_backend_requirement": "hip-direct",
                    "promotion_scope": "execution_path_evidence",
                },
            ),
            ScenarioItem(
                "vector-to-rns-chain",
                "bounded-u64-64",
                "bounded-u64",
                small_64,
                "vector-ALU native producer result materialized into a Direct-HIP RNS consumer GEMM",
                "host_export",
                "checks that bounded-u64 vector output can feed a downstream Direct-HIP RNS workload without host export",
                backends=("auto",),
                vector_to_rns_chain=True,
                metadata={
                    "workflow_name": "vector_to_rns_chain",
                    "bridge_role": "vector_native_output_to_direct_rns_consumer",
                    "producer_backend_requirement": "hip-vector-alu-int64",
                    "consumer_backend_requirement": "hip-direct",
                    "conversion_event_required": "native_u64_to_rns_kernel",
                    "selected_backend_requirement": "hip-direct",
                    "promotion_scope": "execution_path_evidence",
                },
            ),
            ScenarioItem(
                "vector-to-rns-chain",
                "bounded-i64-128",
                "bounded-i64",
                small_128,
                "vector-ALU native producer result materialized into a Direct-HIP RNS consumer GEMM",
                "host_export",
                "exposes whether the native-to-RNS handoff remains visible beyond the smallest smoke shape",
                backends=("auto",),
                vector_to_rns_chain=True,
                metadata={
                    "workflow_name": "vector_to_rns_chain",
                    "bridge_role": "vector_native_output_to_direct_rns_consumer",
                    "producer_backend_requirement": "hip-vector-alu-int64",
                    "consumer_backend_requirement": "hip-direct",
                    "conversion_event_required": "native_i64_to_rns_kernel",
                    "selected_backend_requirement": "hip-direct",
                    "promotion_scope": "execution_path_evidence",
                },
            ),
            ScenarioItem(
                "vector-to-rns-chain",
                "bounded-u64-128",
                "bounded-u64",
                small_128,
                "vector-ALU native producer result materialized into a Direct-HIP RNS consumer GEMM",
                "host_export",
                "exposes bounded-u64 materialization overhead at a shape large enough to separate copy and GEMM costs",
                backends=("auto",),
                vector_to_rns_chain=True,
                metadata={
                    "workflow_name": "vector_to_rns_chain",
                    "bridge_role": "vector_native_output_to_direct_rns_consumer",
                    "producer_backend_requirement": "hip-vector-alu-int64",
                    "consumer_backend_requirement": "hip-direct",
                    "conversion_event_required": "native_u64_to_rns_kernel",
                    "selected_backend_requirement": "hip-direct",
                    "promotion_scope": "execution_path_evidence",
                },
            ),
            ScenarioItem(
                "vector-to-rns-chain",
                "bounded-i64-64-reuse-consumer-b",
                "bounded-i64",
                small_64,
                "vector-ALU native producer result feeds a Direct-HIP RNS consumer with reused consumer B",
                "host_export",
                "checks whether the chain path can remove per-repeat Direct-HIP consumer-B packing",
                backends=("auto",),
                pack_mode="prepacked_reuse_b",
                vector_to_rns_chain=True,
                metadata={
                    "workflow_name": "vector_to_rns_chain",
                    "bridge_role": "vector_native_output_to_direct_rns_consumer",
                    "producer_backend_requirement": "hip-vector-alu-int64",
                    "consumer_backend_requirement": "hip-direct",
                    "reuse_contract": "consumer_b_prepacked_before_warmups",
                    "conversion_event_required": "native_i64_to_rns_kernel",
                    "selected_backend_requirement": "hip-direct",
                    "promotion_scope": "execution_path_evidence",
                },
            ),
            ScenarioItem(
                "vector-to-rns-chain",
                "bounded-u64-64-reuse-consumer-b",
                "bounded-u64",
                small_64,
                "vector-ALU native producer result feeds a Direct-HIP RNS consumer with reused consumer B",
                "host_export",
                "checks whether bounded-u64 chain work benefits when downstream RNS B is source-stable",
                backends=("auto",),
                pack_mode="prepacked_reuse_b",
                vector_to_rns_chain=True,
                metadata={
                    "workflow_name": "vector_to_rns_chain",
                    "bridge_role": "vector_native_output_to_direct_rns_consumer",
                    "producer_backend_requirement": "hip-vector-alu-int64",
                    "consumer_backend_requirement": "hip-direct",
                    "reuse_contract": "consumer_b_prepacked_before_warmups",
                    "conversion_event_required": "native_u64_to_rns_kernel",
                    "selected_backend_requirement": "hip-direct",
                    "promotion_scope": "execution_path_evidence",
                },
            ),
            ScenarioItem(
                "vector-to-rns-chain",
                "bounded-i64-128-reuse-consumer-b",
                "bounded-i64",
                small_128,
                "vector-ALU native producer result feeds a Direct-HIP RNS consumer with reused consumer B",
                "host_export",
                "measures setup-inclusive consumer-B reuse at a shape where pack overhead is still event-visible",
                backends=("auto",),
                pack_mode="prepacked_reuse_b",
                vector_to_rns_chain=True,
                metadata={
                    "workflow_name": "vector_to_rns_chain",
                    "bridge_role": "vector_native_output_to_direct_rns_consumer",
                    "producer_backend_requirement": "hip-vector-alu-int64",
                    "consumer_backend_requirement": "hip-direct",
                    "reuse_contract": "consumer_b_prepacked_before_warmups",
                    "conversion_event_required": "native_i64_to_rns_kernel",
                    "selected_backend_requirement": "hip-direct",
                    "promotion_scope": "execution_path_evidence",
                },
            ),
            ScenarioItem(
                "vector-to-rns-chain",
                "bounded-u64-128-reuse-consumer-b",
                "bounded-u64",
                small_128,
                "vector-ALU native producer result feeds a Direct-HIP RNS consumer with reused consumer B",
                "host_export",
                "measures bounded-u64 setup-inclusive consumer-B reuse before larger chain workload promotion",
                backends=("auto",),
                pack_mode="prepacked_reuse_b",
                vector_to_rns_chain=True,
                metadata={
                    "workflow_name": "vector_to_rns_chain",
                    "bridge_role": "vector_native_output_to_direct_rns_consumer",
                    "producer_backend_requirement": "hip-vector-alu-int64",
                    "consumer_backend_requirement": "hip-direct",
                    "reuse_contract": "consumer_b_prepacked_before_warmups",
                    "conversion_event_required": "native_u64_to_rns_kernel",
                    "selected_backend_requirement": "hip-direct",
                    "promotion_scope": "execution_path_evidence",
                },
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
                metadata={
                    "workflow_name": "rns_gemm_chain",
                    "phase_label": "residue_current_chain",
                    "reuse_profile": "chain_current_residue_output",
                    "output_domain_requirement": "lazy_export",
                    "lowering_role": "native_rns_chain",
                },
            ),
            ScenarioItem(
                "rns-chain",
                "bounded-i64-chain3-reuse-b",
                "bounded-i64",
                chain_128,
                "three chained RNS GEMMs with stable RHS packed once before warmups",
                "residue_current_rns",
                "measures whether a residue-current chain benefits when the repeated B operand stays resident",
                backends=("hip-direct",),
                pack_mode="prepacked_reuse_b",
                residue_chain_length=3,
                metadata={
                    "workflow_name": "rns_gemm_chain",
                    "phase_label": "residue_current_chain_reuse_b",
                    "reuse_profile": "chain_current_residue_output_reused_b",
                    "reuse_contract": "stable_chain_rhs_prepacked_before_warmups",
                    "output_domain_requirement": "lazy_export",
                    "lowering_role": "native_rns_chain",
                    "setup_cost_policy": "prepack_setup_us_excluded_from_per_repeat_included_in_review",
                },
            ),
            ScenarioItem(
                "rns-chain",
                "bounded-i64-chain4-256",
                "bounded-i64",
                chain_256,
                "four chained bounded i64 RNS GEMMs with one final checksum export",
                "residue_current_rns",
                "separates longer residue-current chain scheduling from ordinary one-shot host export",
                backends=accelerator_backends,
                residue_chain_length=4,
                metadata={
                    "workflow_name": "rns_gemm_chain",
                    "phase_label": "residue_current_chain",
                    "reuse_profile": "chain_current_residue_output",
                    "output_domain_requirement": "lazy_export",
                    "lowering_role": "native_rns_chain",
                    "chain_depth": 4,
                },
            ),
            ScenarioItem(
                "rns-chain",
                "bounded-i64-chain4-256-reuse-b",
                "bounded-i64",
                chain_256,
                "four chained bounded i64 RNS GEMMs with stable RHS packed once before warmups",
                "residue_current_rns",
                "checks whether longer residue-current chains amortize downstream B residency at 256",
                backends=("hip-direct",),
                pack_mode="prepacked_reuse_b",
                residue_chain_length=4,
                metadata={
                    "workflow_name": "rns_gemm_chain",
                    "phase_label": "residue_current_chain_reuse_b",
                    "reuse_profile": "chain_current_residue_output_reused_b",
                    "reuse_contract": "stable_chain_rhs_prepacked_before_warmups",
                    "output_domain_requirement": "lazy_export",
                    "lowering_role": "native_rns_chain",
                    "chain_depth": 4,
                    "setup_cost_policy": "prepack_setup_us_excluded_from_per_repeat_included_in_review",
                },
            ),
            ScenarioItem(
                "rns-chain",
                "bounded-u64-chain3-256",
                "bounded-u64",
                chain_256,
                "three chained bounded u64 RNS GEMMs with one final checksum export",
                "residue_current_rns",
                "checks whether unsigned bounded chain behavior tracks signed chain behavior before a public chain planner exists",
                backends=accelerator_backends,
                residue_chain_length=3,
                metadata={
                    "workflow_name": "rns_gemm_chain",
                    "phase_label": "residue_current_chain",
                    "reuse_profile": "chain_current_residue_output",
                    "output_domain_requirement": "lazy_export",
                    "lowering_role": "native_rns_chain",
                    "chain_depth": 3,
                },
            ),
            ScenarioItem(
                "rns-chain",
                "bounded-u64-chain3-256-reuse-b",
                "bounded-u64",
                chain_256,
                "three chained bounded u64 RNS GEMMs with stable RHS packed once before warmups",
                "residue_current_rns",
                "checks whether unsigned bounded chains amortize B residency differently than signed chains",
                backends=("hip-direct",),
                pack_mode="prepacked_reuse_b",
                residue_chain_length=3,
                metadata={
                    "workflow_name": "rns_gemm_chain",
                    "phase_label": "residue_current_chain_reuse_b",
                    "reuse_profile": "chain_current_residue_output_reused_b",
                    "reuse_contract": "stable_chain_rhs_prepacked_before_warmups",
                    "output_domain_requirement": "lazy_export",
                    "lowering_role": "native_rns_chain",
                    "chain_depth": 3,
                    "setup_cost_policy": "prepack_setup_us_excluded_from_per_repeat_included_in_review",
                },
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
                metadata={
                    "workflow_name": "exact_wide_rns_chain",
                    "phase_label": "residue_current_chain",
                    "reuse_profile": "chain_current_residue_output",
                    "output_domain_requirement": "lazy_export",
                    "lowering_role": "native_rns_chain_before_limb_export",
                    "chain_depth": 3,
                },
            ),
            ScenarioItem(
                "rns-chain",
                "exact-wide-signed-chain3-reuse-b",
                "exact-wide-signed",
                chain_128,
                "three chained exact-wide signed RNS GEMMs with stable RHS packed once before warmups",
                "residue_current_rns",
                "measures whether exact-wide lazy-export chains amortize reusable B residency before limb export",
                backends=("hip-direct",),
                pack_mode="prepacked_reuse_b",
                residue_chain_length=3,
                exact_wide_limb_counts=(4,),
                metadata={
                    "workflow_name": "exact_wide_rns_chain",
                    "phase_label": "residue_current_chain_reuse_b",
                    "reuse_profile": "chain_current_residue_output_reused_b",
                    "reuse_contract": "stable_chain_rhs_prepacked_before_warmups",
                    "output_domain_requirement": "lazy_export",
                    "lowering_role": "native_rns_chain_before_limb_export",
                    "chain_depth": 3,
                    "setup_cost_policy": "prepack_setup_us_excluded_from_per_repeat_included_in_review",
                },
            ),
            ScenarioItem(
                "rns-chain",
                "exact-wide-unsigned-chain3-256",
                "exact-wide-unsigned",
                chain_256,
                "three chained exact-wide unsigned RNS GEMMs with one final checksum export",
                "residue_current_rns",
                "keeps unsigned exact-wide lazy-export evidence separate from signed limb-output evidence",
                backends=EXACT_WIDE_BACKENDS,
                residue_chain_length=3,
                exact_wide_limb_counts=(4,),
                metadata={
                    "workflow_name": "exact_wide_rns_chain",
                    "phase_label": "residue_current_chain",
                    "reuse_profile": "chain_current_residue_output",
                    "output_domain_requirement": "lazy_export",
                    "lowering_role": "native_rns_chain_before_limb_export",
                    "chain_depth": 3,
                },
            ),
            ScenarioItem(
                "rns-chain",
                "exact-wide-unsigned-chain3-256-reuse-b",
                "exact-wide-unsigned",
                chain_256,
                "three chained exact-wide unsigned RNS GEMMs with stable RHS packed once before warmups",
                "residue_current_rns",
                "keeps exact-wide unsigned reusable-B chain evidence separate from one-shot limb-output evidence",
                backends=("hip-direct",),
                pack_mode="prepacked_reuse_b",
                residue_chain_length=3,
                exact_wide_limb_counts=(4,),
                metadata={
                    "workflow_name": "exact_wide_rns_chain",
                    "phase_label": "residue_current_chain_reuse_b",
                    "reuse_profile": "chain_current_residue_output_reused_b",
                    "reuse_contract": "stable_chain_rhs_prepacked_before_warmups",
                    "output_domain_requirement": "lazy_export",
                    "lowering_role": "native_rns_chain_before_limb_export",
                    "chain_depth": 3,
                    "setup_cost_policy": "prepack_setup_us_excluded_from_per_repeat_included_in_review",
                },
            ),
        ],
        "rns-chain-final-output": [
            ScenarioItem(
                "rns-chain-final-output",
                "bounded-i64-chain3-final-export",
                "bounded-i64",
                chain_128,
                "three chained bounded i64 RNS GEMMs with final host export inside each measured repeat",
                "host_export",
                "compares lazy intermediate residency against the same final-output contract callers already request",
                backends=("cpu", "hip-direct", "ck", "rocwmma"),
                residue_chain_length=3,
                residue_chain_final_export=True,
                next_op_hint="final-export",
                metadata={
                    "workflow_name": "rns_chain_final_output",
                    "phase_label": "chain_final_export",
                    "output_domain_requirement": "same_final_output",
                    "chain_depth": 3,
                    "validation_contract": "independent_calls_vs_chain_plus_final_export",
                    "promotion_scope": "final_output_chain_evidence_only",
                },
            ),
            ScenarioItem(
                "rns-chain-final-output",
                "bounded-i64-chain3-final-export-reuse-b",
                "bounded-i64",
                chain_128,
                "three chained bounded i64 RNS GEMMs with stable RHS and final host export inside each measured repeat",
                "host_export",
                "checks whether reusable B survives the setup-inclusive same-final-output contract",
                backends=("hip-direct",),
                pack_mode="prepacked_reuse_b",
                residue_chain_length=3,
                residue_chain_final_export=True,
                next_op_hint="final-export",
                metadata={
                    "workflow_name": "rns_chain_final_output",
                    "phase_label": "chain_final_export_reuse_b",
                    "output_domain_requirement": "same_final_output",
                    "reuse_contract": "stable_chain_rhs_prepacked_before_warmups",
                    "chain_depth": 3,
                    "setup_cost_policy": "prepack_setup_us_excluded_from_per_repeat_included_in_review",
                    "validation_contract": "independent_calls_vs_chain_plus_final_export",
                    "promotion_scope": "final_output_chain_reuse_evidence_only",
                },
            ),
            ScenarioItem(
                "rns-chain-final-output",
                "bounded-u64-chain3-final-export-256",
                "bounded-u64",
                chain_256,
                "three chained bounded u64 RNS GEMMs with final host export inside each measured repeat",
                "host_export",
                "keeps unsigned same-final-output chain evidence separate from lazy residue-current evidence",
                backends=("cpu", "hip-direct", "ck", "rocwmma"),
                residue_chain_length=3,
                residue_chain_final_export=True,
                next_op_hint="final-export",
                metadata={
                    "workflow_name": "rns_chain_final_output",
                    "phase_label": "chain_final_export",
                    "output_domain_requirement": "same_final_output",
                    "chain_depth": 3,
                    "validation_contract": "independent_calls_vs_chain_plus_final_export",
                    "promotion_scope": "final_output_chain_evidence_only",
                },
            ),
            ScenarioItem(
                "rns-chain-final-output",
                "exact-wide-signed-chain3-final-export",
                "exact-wide-signed",
                chain_128,
                "three chained exact-wide signed RNS GEMMs with final fixed-limb export inside each measured repeat",
                "exact_wide_signed_limbs",
                "turns lazy exact-wide chaining into a same-final-output benchmark contract",
                backends=EXACT_WIDE_BACKENDS,
                residue_chain_length=3,
                residue_chain_final_export=True,
                next_op_hint="final-export",
                exact_wide_limb_counts=(4,),
                metadata={
                    "workflow_name": "exact_wide_rns_chain_final_output",
                    "phase_label": "chain_final_export",
                    "output_domain_requirement": "same_final_output",
                    "lowering_role": "native_rns_chain_before_final_limb_export",
                    "chain_depth": 3,
                    "validation_contract": "independent_calls_vs_chain_plus_final_export",
                    "promotion_scope": "final_output_chain_evidence_only",
                },
            ),
            ScenarioItem(
                "rns-chain-final-output",
                "exact-wide-signed-chain3-final-export-reuse-b",
                "exact-wide-signed",
                chain_128,
                "three chained exact-wide signed RNS GEMMs with stable RHS and final fixed-limb export inside each measured repeat",
                "exact_wide_signed_limbs",
                "measures whether reusable B still wins when the final exact-wide host output is part of the repeat",
                backends=("hip-direct",),
                pack_mode="prepacked_reuse_b",
                residue_chain_length=3,
                residue_chain_final_export=True,
                next_op_hint="final-export",
                exact_wide_limb_counts=(4,),
                metadata={
                    "workflow_name": "exact_wide_rns_chain_final_output",
                    "phase_label": "chain_final_export_reuse_b",
                    "output_domain_requirement": "same_final_output",
                    "reuse_contract": "stable_chain_rhs_prepacked_before_warmups",
                    "lowering_role": "native_rns_chain_before_final_limb_export",
                    "chain_depth": 3,
                    "setup_cost_policy": "prepack_setup_us_excluded_from_per_repeat_included_in_review",
                    "validation_contract": "independent_calls_vs_chain_plus_final_export",
                    "promotion_scope": "final_output_chain_reuse_evidence_only",
                },
            ),
            ScenarioItem(
                "rns-chain-final-output",
                "exact-wide-unsigned-chain3-final-export-256",
                "exact-wide-unsigned",
                chain_256,
                "three chained exact-wide unsigned RNS GEMMs with final fixed-limb export inside each measured repeat",
                "exact_wide_unsigned_limbs",
                "adds unsigned exact-wide same-final-output chain coverage before API-level lazy output work",
                backends=EXACT_WIDE_BACKENDS,
                residue_chain_length=3,
                residue_chain_final_export=True,
                next_op_hint="final-export",
                exact_wide_limb_counts=(4,),
                metadata={
                    "workflow_name": "exact_wide_rns_chain_final_output",
                    "phase_label": "chain_final_export",
                    "output_domain_requirement": "same_final_output",
                    "lowering_role": "native_rns_chain_before_final_limb_export",
                    "chain_depth": 3,
                    "validation_contract": "independent_calls_vs_chain_plus_final_export",
                    "promotion_scope": "final_output_chain_evidence_only",
                },
            ),
            ScenarioItem(
                "rns-chain-final-output",
                "exact-wide-signed-chain3-final-export-512",
                "exact-wide-signed",
                chain_512,
                "512-sized exact-wide signed RNS chain with final fixed-limb export inside each measured repeat",
                "exact_wide_signed_limbs",
                "seeds the larger same-final-output matrix without running the full 2048/4096 release sweep",
                backends=("hip-direct",),
                residue_chain_length=3,
                residue_chain_final_export=True,
                next_op_hint="final-export",
                exact_wide_limb_counts=(4,),
                metadata={
                    "workflow_name": "exact_wide_rns_chain_final_output",
                    "phase_label": "chain_final_export_512",
                    "output_domain_requirement": "same_final_output",
                    "lowering_role": "native_rns_chain_before_final_limb_export",
                    "chain_depth": 3,
                    "validation_contract": "independent_calls_vs_chain_plus_final_export",
                    "promotion_scope": "larger_final_output_chain_evidence_only",
                },
            ),
        ],
        "hip-graph-replay": [
            ScenarioItem(
                "hip-graph-replay",
                "bounded-i64-chain3-reuse-inputs-baseline",
                "bounded-i64",
                chain_128,
                "ordinary Direct-HIP residue-current chain with stable A and B packed once before warmups",
                "residue_current_rns",
                "same-contract baseline for HIP Graph replay; prepack setup and final checksum export stay visible",
                backends=("hip-direct",),
                pack_mode="prepacked_reuse",
                residue_chain_length=3,
                next_op_hint="rns-gemm",
                metadata={
                    "workflow_name": "hip_graph_replay",
                    "phase_label": "ordinary_residue_current_chain_reuse_inputs",
                    "graph_role": "same_contract_non_graph_baseline",
                    "reuse_contract": "stable_chain_a_b_prepacked_before_warmups",
                    "setup_cost_policy": "prepack_setup_us_excluded_from_per_repeat_included_in_review",
                },
            ),
            ScenarioItem(
                "hip-graph-replay",
                "bounded-i64-chain3-reuse-inputs-graph",
                "bounded-i64",
                chain_128,
                "Direct-HIP HIP Graph replay candidate for a stable-input residue-current chain",
                "residue_current_rns",
                "measures whether graph replay removes launch overhead for the same resident RNS chain contract",
                backends=("hip-direct",),
                pack_mode="prepacked_reuse",
                residue_chain_length=3,
                next_op_hint="rns-gemm",
                hip_graph_replay=True,
                metadata={
                    "workflow_name": "hip_graph_replay",
                    "phase_label": "hip_graph_residue_current_chain_reuse_inputs",
                    "graph_role": "graph_replay_candidate",
                    "reuse_contract": "stable_chain_a_b_prepacked_before_capture",
                    "setup_cost_policy": "prepack_setup_us_graph_capture_us_and_instantiate_us_included_in_review",
                },
            ),
            ScenarioItem(
                "hip-graph-replay",
                "bounded-u64-chain3-reuse-inputs-baseline",
                "bounded-u64",
                chain_128,
                "ordinary Direct-HIP bounded-u64 residue-current chain with stable A and B",
                "residue_current_rns",
                "keeps signed and unsigned bounded graph comparisons separate",
                backends=("hip-direct",),
                pack_mode="prepacked_reuse",
                residue_chain_length=3,
                next_op_hint="rns-gemm",
                metadata={
                    "workflow_name": "hip_graph_replay",
                    "phase_label": "ordinary_residue_current_chain_reuse_inputs",
                    "graph_role": "same_contract_non_graph_baseline",
                    "reuse_contract": "stable_chain_a_b_prepacked_before_warmups",
                    "setup_cost_policy": "prepack_setup_us_excluded_from_per_repeat_included_in_review",
                },
            ),
            ScenarioItem(
                "hip-graph-replay",
                "bounded-u64-chain3-reuse-inputs-graph",
                "bounded-u64",
                chain_128,
                "Direct-HIP HIP Graph replay candidate for bounded-u64 stable-input residue-current chains",
                "residue_current_rns",
                "checks whether unsigned bounded chains benefit differently from graph replay",
                backends=("hip-direct",),
                pack_mode="prepacked_reuse",
                residue_chain_length=3,
                next_op_hint="rns-gemm",
                hip_graph_replay=True,
                metadata={
                    "workflow_name": "hip_graph_replay",
                    "phase_label": "hip_graph_residue_current_chain_reuse_inputs",
                    "graph_role": "graph_replay_candidate",
                    "reuse_contract": "stable_chain_a_b_prepacked_before_capture",
                    "setup_cost_policy": "prepack_setup_us_graph_capture_us_and_instantiate_us_included_in_review",
                },
            ),
            ScenarioItem(
                "hip-graph-replay",
                "exact-wide-signed-chain3-reuse-inputs-baseline",
                "exact-wide-signed",
                chain_128,
                "ordinary Direct-HIP exact-wide signed residue-current chain with stable A and B",
                "residue_current_rns",
                "keeps graph replay compared against the same exact-wide lazy-export contract",
                backends=("hip-direct",),
                pack_mode="prepacked_reuse",
                residue_chain_length=3,
                next_op_hint="rns-gemm",
                exact_wide_limb_counts=(4,),
                metadata={
                    "workflow_name": "hip_graph_replay",
                    "phase_label": "ordinary_exact_wide_chain_reuse_inputs",
                    "graph_role": "same_contract_non_graph_baseline",
                    "reuse_contract": "stable_chain_a_b_prepacked_before_warmups",
                    "setup_cost_policy": "prepack_setup_us_excluded_from_per_repeat_included_in_review",
                },
            ),
            ScenarioItem(
                "hip-graph-replay",
                "exact-wide-signed-chain3-reuse-inputs-graph",
                "exact-wide-signed",
                chain_128,
                "Direct-HIP HIP Graph replay candidate for exact-wide signed residue-current chains",
                "residue_current_rns",
                "checks graph replay on an exact-wide lazy-export workflow before any public chain planner exists",
                backends=("hip-direct",),
                pack_mode="prepacked_reuse",
                residue_chain_length=3,
                next_op_hint="rns-gemm",
                hip_graph_replay=True,
                exact_wide_limb_counts=(4,),
                metadata={
                    "workflow_name": "hip_graph_replay",
                    "phase_label": "hip_graph_exact_wide_chain_reuse_inputs",
                    "graph_role": "graph_replay_candidate",
                    "reuse_contract": "stable_chain_a_b_prepacked_before_capture",
                    "setup_cost_policy": "prepack_setup_us_graph_capture_us_and_instantiate_us_included_in_review",
                },
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
        "many-small": [
            ScenarioItem(
                "many-small",
                "bounded-i64-32-proxy",
                "bounded-i64",
                many_small_32,
                "many-small proxy using repeated fixed-shape bounded i64 captures",
                "host_export",
                "ranks low-overhead CPU, vector, direct-HIP, and accelerator paths before a grouped independent-GEMM API exists",
                backends=BOUNDED_BACKENDS,
                metadata={
                    "evidence_role": "proxy_single_shape_repeat",
                    "workflow_name": "many_small_independent_gemms",
                    "phase_label": "batch_proxy",
                    "shape_signature": "tiny_square",
                    "batch_count_model": 64,
                    "reuse_profile": "independent_inputs",
                    "dense_kernel_extracted": True,
                },
            ),
            ScenarioItem(
                "many-small",
                "bounded-i64-32-oneshot-proxy",
                "bounded-i64",
                many_small_32,
                "many-small public one-shot bounded i64 proxy",
                "native_i64_host_output",
                "keeps one-shot setup cost visible for tiny independent tasks instead of folding it into persistent RNS evidence",
                backends=direct_oneshot_backends,
                oneshot=True,
                metadata={
                    "evidence_role": "proxy_single_shape_repeat",
                    "workflow_name": "many_small_independent_gemms",
                    "phase_label": "oneshot_batch_proxy",
                    "shape_signature": "tiny_square",
                    "batch_count_model": 64,
                    "reuse_profile": "independent_inputs",
                    "dense_kernel_extracted": True,
                },
            ),
            ScenarioItem(
                "many-small",
                "bounded-i64-32-host-batch64",
                "bounded-i64",
                many_small_32,
                "many-small host API batch bounded i64 candidate",
                "host_export",
                "measures an actual benchmark-owned host batch of independent resident RNS tasks against the pre-grouped baseline surface",
                backends=HOST_API_BATCH_BACKENDS,
                host_api_batch_size=64,
                metadata={
                    "evidence_role": "host_api_batch_candidate",
                    "workflow_name": "many_small_independent_gemms",
                    "phase_label": "host_api_batch",
                    "shape_signature": "tiny_square",
                    "batch_count_model": 64,
                    "reuse_profile": "independent_inputs",
                    "grouping_role": "host_api_batch_candidate",
                    "dense_kernel_extracted": True,
                },
            ),
            ScenarioItem(
                "many-small",
                "finite-ring-64-proxy",
                "finite-u8-ring",
                many_small_64,
                "many-small finite-ring u8 proxy with canonical host output",
                "finite_u8_canonical_host_export",
                "separates small finite setup overhead from medium-shape finite accelerator evidence",
                backends=FINITE_BACKENDS,
                finite_moduli=(251, 255),
                metadata={
                    "evidence_role": "proxy_single_shape_repeat",
                    "workflow_name": "many_small_finite_ring_gemms",
                    "domain_family": "finite_ring_u8",
                    "phase_label": "batch_proxy",
                    "shape_signature": "small_square",
                    "batch_count_model": 32,
                    "reuse_profile": "independent_inputs",
                    "dense_kernel_extracted": True,
                },
            ),
            ScenarioItem(
                "many-small",
                "finite-ring-64-host-batch32",
                "finite-u8-ring",
                many_small_64,
                "many-small host API batch finite-ring u8 candidate",
                "finite_u8_canonical_host_export",
                "tests whether explicit host batching changes the setup/launch balance for small finite-ring tasks",
                backends=FINITE_BACKENDS,
                finite_moduli=(251,),
                host_api_batch_size=32,
                metadata={
                    "evidence_role": "host_api_batch_candidate",
                    "workflow_name": "many_small_finite_ring_gemms",
                    "domain_family": "finite_ring_u8",
                    "phase_label": "host_api_batch",
                    "shape_signature": "small_square",
                    "batch_count_model": 32,
                    "reuse_profile": "independent_inputs",
                    "grouping_role": "host_api_batch_candidate",
                    "dense_kernel_extracted": True,
                },
            ),
            ScenarioItem(
                "many-small",
                "bounded-u64-64-proxy",
                "bounded-u64",
                many_small_64,
                "many-small bounded u64 proxy for native/vector-to-RNS routing",
                "host_export",
                "keeps the strong bounded-u64 native/vector family visible before grouped or chained consumers exist",
                backends=BOUNDED_BACKENDS,
                metadata={
                    "evidence_role": "proxy_single_shape_repeat",
                    "workflow_name": "many_small_independent_gemms",
                    "phase_label": "pre_grouped_batch_proxy",
                    "shape_signature": "small_square",
                    "batch_count_model": 128,
                    "reuse_profile": "independent_inputs",
                    "grouping_role": "pre_grouped_baseline",
                    "bridge_role": "native_vector_to_rns_candidate",
                    "dense_kernel_extracted": True,
                },
            ),
            ScenarioItem(
                "many-small",
                "bounded-i64-128-proxy",
                "bounded-i64",
                many_small_128,
                "many-small bounded i64 proxy at the largest small-batch square shape",
                "host_export",
                "tests whether a future grouped dispatcher should target 128-square jobs or stay focused on tinier launch-bound cases",
                backends=BOUNDED_BACKENDS,
                metadata={
                    "evidence_role": "proxy_single_shape_repeat",
                    "workflow_name": "many_small_independent_gemms",
                    "phase_label": "pre_grouped_batch_proxy",
                    "shape_signature": "medium_small_square",
                    "batch_count_model": 64,
                    "reuse_profile": "independent_inputs",
                    "grouping_role": "pre_grouped_baseline",
                    "dense_kernel_extracted": True,
                },
            ),
            ScenarioItem(
                "many-small",
                "bounded-u64-skinny-n1-proxy",
                "bounded-u64",
                many_small_skinny,
                "many-small skinny bounded u64 proxy for N=1 native/vector dispatch",
                "host_export",
                "keeps skinny exact jobs in the same grouped-workload planning surface as square many-small jobs",
                backends=BOUNDED_BACKENDS,
                metadata={
                    "evidence_role": "proxy_single_shape_repeat",
                    "workflow_name": "many_small_independent_gemms",
                    "phase_label": "pre_grouped_skinny_proxy",
                    "shape_signature": "skinny_n1_long_k",
                    "batch_count_model": 128,
                    "reuse_profile": "single_rhs_column",
                    "grouping_role": "pre_grouped_baseline",
                    "bridge_role": "native_vector_to_rns_candidate",
                    "dense_kernel_extracted": True,
                },
            ),
            ScenarioItem(
                "many-small",
                "bounded-u64-skinny-n1-host-batch128",
                "bounded-u64",
                many_small_skinny,
                "many-small host API batch skinny bounded u64 candidate",
                "host_export",
                "tests whether batching long-K N=1 exact jobs can beat independent resident calls before a device grouped scheduler exists",
                backends=HOST_API_BATCH_BACKENDS,
                host_api_batch_size=128,
                metadata={
                    "evidence_role": "host_api_batch_candidate",
                    "workflow_name": "many_small_independent_gemms",
                    "phase_label": "host_api_batch",
                    "shape_signature": "skinny_n1_long_k",
                    "batch_count_model": 128,
                    "reuse_profile": "single_rhs_column",
                    "grouping_role": "host_api_batch_candidate",
                    "bridge_role": "native_vector_to_rns_candidate",
                    "dense_kernel_extracted": True,
                },
            ),
            ScenarioItem(
                "many-small",
                "exact-wide-signed-64-proxy",
                "exact-wide-signed",
                many_small_64,
                "many-small exact-wide signed proxy with fixed limb export",
                "exact_wide_signed_limbs",
                "adds exact-wide jobs to the grouped-workload planning surface instead of leaving many-small evidence bounded-only",
                backends=EXACT_WIDE_BACKENDS,
                exact_wide_limb_counts=(4,),
                metadata={
                    "evidence_role": "proxy_single_shape_repeat",
                    "workflow_name": "many_small_exact_wide_gemms",
                    "phase_label": "pre_grouped_exact_wide_proxy",
                    "shape_signature": "small_square",
                    "batch_count_model": 32,
                    "reuse_profile": "independent_inputs",
                    "grouping_role": "pre_grouped_baseline",
                    "dense_kernel_extracted": True,
                },
            ),
            ScenarioItem(
                "many-small",
                "exact-wide-signed-64-host-batch32",
                "exact-wide-signed",
                many_small_64,
                "many-small host API batch exact-wide signed candidate",
                "exact_wide_signed_limbs",
                "measures explicit independent exact-wide task batching with fixed-limb export and per-task checksum coverage",
                backends=EXACT_WIDE_BACKENDS,
                exact_wide_limb_counts=(4,),
                host_api_batch_size=32,
                metadata={
                    "evidence_role": "host_api_batch_candidate",
                    "workflow_name": "many_small_exact_wide_gemms",
                    "phase_label": "host_api_batch",
                    "shape_signature": "small_square",
                    "batch_count_model": 32,
                    "reuse_profile": "independent_inputs",
                    "grouping_role": "host_api_batch_candidate",
                    "dense_kernel_extracted": True,
                },
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
            ScenarioItem(
                "skinny-gemv",
                "bounded-i64-n1-longk-256",
                "bounded-i64",
                skinny_longk,
                "long-K N=1 bounded i64 GEMV workload",
                "host_export",
                "exercises the vector-ALU N=1 reduction kernel on multiple output rows",
                backends=BOUNDED_BACKENDS,
                metadata={
                    "shape_signature": "tall_skinny_long_k",
                    "workflow_name": "gemv_n1_long_k",
                    "phase_label": "native_vector_reduction",
                    "reuse_profile": "single_rhs_column",
                },
            ),
        ],
        "computational-algebra-proxies": [
            ScenarioItem(
                "computational-algebra-proxies",
                "dense-finite-field-blas-512",
                "finite-u8-field",
                finite_512,
                "dense finite-field BLAS phase over GF(p <= 251)",
                "finite_u8_canonical_host_export",
                "isolates a dense modular GEMM phase without claiming rank, determinant, solve, or full CAS workflow coverage",
                backends=FINITE_BACKENDS,
                finite_moduli=(251,),
                metadata={
                    "source_role": "computational_algebra_proxy",
                    "algebra_family": "finite_field_linear_algebra",
                    "domain_family": "prime_field_u8",
                    "workflow_name": "dense_finite_field_blas",
                    "phase_label": "gemm",
                    "phase_id": "dense_gemm",
                    "prime_or_composite": "prime",
                    "extension_degree": 1,
                    "dense_kernel_extracted": True,
                    "oracle_role": "optional_cpu_cas_comparison",
                    "artifact_lineage": "rns8_synthetic_shape",
                },
            ),
            ScenarioItem(
                "computational-algebra-proxies",
                "rank-k-update-field-251",
                "finite-u8-field",
                algebra_rank_update,
                "rectangular finite-field rank-k update phase",
                "finite_u8_canonical_host_export",
                "models dense trailing-update work that can appear inside modular rank, PLUQ, CUP, PLE, and echelon pipelines",
                backends=FINITE_BACKENDS,
                finite_moduli=(251,),
                metadata={
                    "source_role": "computational_algebra_proxy",
                    "algebra_family": "finite_field_linear_algebra",
                    "domain_family": "prime_field_u8",
                    "workflow_name": "modular_rank_or_elimination",
                    "phase_label": "rank_k_trailing_update",
                    "phase_id": "dense_update",
                    "shape_signature": "rectangular_rank_k",
                    "symbolic_precompute": "outside_timed_region",
                    "dense_kernel_extracted": True,
                    "verification_mode": "cpu_reference_capture",
                },
            ),
            ScenarioItem(
                "computational-algebra-proxies",
                "f4-dense-field-251",
                "finite-u8-field",
                algebra_f4_dense,
                "F4 dense finite-field matrix phase proxy",
                "finite_u8_canonical_host_export",
                "labels dense F4 matrix arithmetic separately from sparse symbolic preprocessing and reduction-control work",
                backends=FINITE_BACKENDS,
                finite_moduli=(251,),
                metadata={
                    "source_role": "computational_algebra_proxy",
                    "algebra_family": "groebner_basis",
                    "domain_family": "prime_field_u8",
                    "workflow_name": "F4",
                    "phase_label": "dense_finite_field_matrix_phase",
                    "phase_id": "f4_dense_block",
                    "symbolic_precompute": "outside_timed_region",
                    "density": "dense_phase_only",
                    "dense_kernel_extracted": True,
                    "certificate_mode": "not_applicable_to_raw_gemm",
                },
            ),
            ScenarioItem(
                "computational-algebra-proxies",
                "fglm-multiplication-matrix-field-251",
                "finite-u8-field",
                algebra_fglm_mulmat,
                "FGLM multiplication-matrix conversion dense phase proxy",
                "finite_u8_canonical_host_export",
                "keeps multiplication-matrix dense arithmetic distinct from ordering conversion and basis-controller work",
                backends=FINITE_BACKENDS,
                finite_moduli=(251,),
                metadata={
                    "source_role": "computational_algebra_proxy",
                    "algebra_family": "groebner_basis",
                    "domain_family": "prime_field_u8",
                    "workflow_name": "FGLM",
                    "phase_label": "multiplication_matrix_dense_phase",
                    "phase_id": "fglm_mulmat",
                    "symbolic_precompute": "outside_timed_region",
                    "dense_kernel_extracted": True,
                    "controller_mode": "not_timed",
                },
            ),
            ScenarioItem(
                "computational-algebra-proxies",
                "crt-rational-reconstruction-export",
                "exact-wide-signed",
                algebra_crt_export,
                "CRT/Garner export-heavy exact-LA proxy",
                "exact_wide_signed_limbs",
                "profiles reconstruction/export pressure before mapping dense GEMM timings onto rational reconstruction or Dixon-style workflows",
                backends=EXACT_WIDE_BACKENDS,
                exact_wide_limb_counts=(4,),
                metadata={
                    "source_role": "computational_algebra_proxy",
                    "algebra_family": "exact_linear_algebra",
                    "domain_family": "integer_rns",
                    "workflow_name": "rational_reconstruction_or_dixon",
                    "phase_label": "crt_garner_export",
                    "phase_id": "reconstruction_export",
                    "reconstruction_mode": "fixed_width_limb_export",
                    "dense_kernel_extracted": True,
                    "controller_mode": "outside_timed_region",
                },
            ),
        ],
        "fhe-lattice-proxies": [
            ScenarioItem(
                "fhe-lattice-proxies",
                "ntt-log12-pressure-proxy",
                "finite-u8-ring",
                fhe_ntt_pressure,
                "NTT/INTT pressure proxy with power-of-two polynomial dimension",
                "finite_u8_canonical_host_export",
                "labels transform-pressure ranking work without claiming RNS8 currently implements an NTT backend",
                backends=("hip-direct",),
                finite_moduli=(251,),
                metadata={
                    "source_role": "fhe_lattice_proxy",
                    "evidence_role": "proxy_not_ntt_proof",
                    "workflow_name": "ntt_intt_pressure",
                    "phase_label": "transform_pressure_proxy",
                    "ring_dimension": 4096,
                    "log_n": 12,
                    "coefficient_modulus_count": 1,
                    "current_domain": "coefficient_or_ntt_domain_proxy",
                    "reuse_profile": "independent_polynomial_channels",
                    "lowering_role": "not_a_dense_gemm_claim",
                },
            ),
            ScenarioItem(
                "fhe-lattice-proxies",
                "key-switch-digit-reuse-b",
                "bounded-i64",
                fhe_key_switch,
                "key-switch digit aggregation proxy with reused key material",
                "host_export",
                "measures repeated read-only B-like operand reuse separately from ordinary one-shot dense GEMM evidence",
                backends=bounded_gpu_backends,
                pack_mode="prepacked_reuse_b",
                prefix_policy="fixed-requested",
                max_prefix=9,
                metadata={
                    "source_role": "fhe_lattice_proxy",
                    "workflow_name": "key_switch_digit_aggregation",
                    "phase_label": "external_product_like_dense_proxy",
                    "ring_dimension": 4096,
                    "log_n": 12,
                    "coefficient_modulus_count": 4,
                    "decomposition_digit_count": 4,
                    "ciphertext_component_count": 2,
                    "evaluation_key_count": 8,
                    "key_material_reuse": "B_operand_reused",
                    "reuse_profile": "large_read_only_key_material",
                    "lowering_role": "dense_gemm_adjacent_proxy",
                },
            ),
            ScenarioItem(
                "fhe-lattice-proxies",
                "ckks-rescale-chain4",
                "bounded-i64",
                fhe_chain,
                "CKKS rescale/mod-drop chain proxy with resident RNS output",
                "residue_current_rns",
                "keeps chained residue-domain work separate from final native host export timings",
                backends=accelerator_backends,
                residue_chain_length=4,
                metadata={
                    "source_role": "fhe_lattice_proxy",
                    "workflow_name": "ckks_rescale_mod_drop",
                    "phase_label": "residue_chain_proxy",
                    "ring_dimension": 8192,
                    "log_n": 13,
                    "slot_count": 4096,
                    "coefficient_modulus_count": 4,
                    "modulus_chain_bits": [60, 40, 40, 60],
                    "current_domain": "rns_residue_current",
                    "reuse_profile": "chain_current_residue_output",
                    "output_domain_requirement": "lazy_export",
                    "lowering_role": "not_a_public_fhe_backend",
                },
            ),
            ScenarioItem(
                "fhe-lattice-proxies",
                "encrypted-linear-layer-reuse-b",
                "bounded-i64",
                fhe_linear_layer,
                "encrypted-inference linear-layer proxy with repeated plaintext matrix",
                "host_export",
                "separates repeated plaintext-matrix reuse from diagonal/rotation and convolution lowerings",
                backends=bounded_gpu_backends,
                pack_mode="prepacked_reuse_b",
                prefix_policy="fixed-requested",
                max_prefix=9,
                metadata={
                    "source_role": "fhe_lattice_proxy",
                    "workflow_name": "encrypted_inference_linear_layer",
                    "phase_label": "dense_linear_layer_proxy",
                    "ring_dimension": 8192,
                    "log_n": 13,
                    "slot_count": 4096,
                    "coefficient_modulus_count": 4,
                    "plaintext_matrix_reuse": "B_operand_reused",
                    "reuse_profile": "many_encrypted_vectors_same_plaintext_matrix",
                    "lowering_role": "dense_gemm_proxy_not_rotation_method",
                },
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
        "large-release-validation": [
            ScenarioItem(
                "large-release-validation",
                "bounded-i64-2048-required-baselines",
                "bounded-i64",
                large_2048,
                "large bounded i64 release-validation workload",
                "host_export",
                "fills the required CPU, Direct-HIP, vector, and accelerator comparator set missing from exploratory 2048 captures",
                backends=bounded_release_backends,
                metadata={
                    "workflow_name": "large_shape_release_validation",
                    "large_shape_role": "required_baseline_matrix",
                    "promotion_scope": "cpu_backed_2048_release_review_candidate",
                    "validation_contract": "same_contract_cpu_direct_vector_accelerator_release_review",
                },
            ),
            ScenarioItem(
                "large-release-validation",
                "bounded-u64-2048-required-baselines",
                "bounded-u64",
                large_2048,
                "large bounded u64 release-validation workload",
                "host_export",
                "fills the required CPU, Direct-HIP, vector, and accelerator comparator set missing from exploratory 2048 captures",
                backends=bounded_release_backends,
                metadata={
                    "workflow_name": "large_shape_release_validation",
                    "large_shape_role": "required_baseline_matrix",
                    "promotion_scope": "cpu_backed_2048_release_review_candidate",
                    "validation_contract": "same_contract_cpu_direct_vector_accelerator_release_review",
                },
            ),
            ScenarioItem(
                "large-release-validation",
                "bounded-i64-2048-reuse-b-required-baselines",
                "bounded-i64",
                large_2048,
                "large bounded i64 release-validation workload with stable RHS packed once before warmups",
                "host_export",
                "checks whether reusable B survives setup-inclusive review when the required CPU, Direct-HIP, and vector comparators are present",
                backends=bounded_release_backends,
                pack_mode="prepacked_reuse_b",
                metadata={
                    "workflow_name": "large_shape_release_validation",
                    "large_shape_role": "required_baseline_reuse_b_matrix",
                    "promotion_scope": "explicit_reuse_contract_only",
                    "reuse_contract": "large_stable_rhs_prepacked_before_warmups",
                    "validation_contract": "same_contract_cpu_direct_vector_accelerator_release_review",
                    "setup_cost_policy": "prepack_setup_us_excluded_from_per_repeat_included_in_review",
                },
            ),
            ScenarioItem(
                "large-release-validation",
                "bounded-u64-2048-reuse-b-required-baselines",
                "bounded-u64",
                large_2048,
                "large bounded u64 release-validation workload with stable RHS packed once before warmups",
                "host_export",
                "checks whether reusable B survives setup-inclusive review when the required CPU, Direct-HIP, and vector comparators are present",
                backends=bounded_release_backends,
                pack_mode="prepacked_reuse_b",
                metadata={
                    "workflow_name": "large_shape_release_validation",
                    "large_shape_role": "required_baseline_reuse_b_matrix",
                    "promotion_scope": "explicit_reuse_contract_only",
                    "reuse_contract": "large_stable_rhs_prepacked_before_warmups",
                    "validation_contract": "same_contract_cpu_direct_vector_accelerator_release_review",
                    "setup_cost_policy": "prepack_setup_us_excluded_from_per_repeat_included_in_review",
                },
            ),
            ScenarioItem(
                "large-release-validation",
                "exact-wide-signed-2048-required-baselines",
                "exact-wide-signed",
                large_exact_2048,
                "large exact-wide signed release-validation workload",
                "exact_wide_signed_limbs",
                "adds CPU and Direct-HIP comparator coverage to the large exact-wide export-heavy matrix",
                backends=tuple(EXACT_WIDE_BACKENDS),
                exact_wide_limb_counts=(4,),
                metadata={
                    "workflow_name": "large_shape_release_validation",
                    "large_shape_role": "exact_wide_export_required_baselines",
                    "promotion_scope": "cpu_backed_2048_release_review_candidate",
                    "validation_contract": "same_contract_cpu_direct_accelerator_release_review",
                },
            ),
            ScenarioItem(
                "large-release-validation",
                "exact-wide-unsigned-2048-required-baselines",
                "exact-wide-unsigned",
                large_exact_2048,
                "large exact-wide unsigned release-validation workload",
                "exact_wide_unsigned_limbs",
                "adds CPU and Direct-HIP comparator coverage to the large exact-wide export-heavy matrix",
                backends=tuple(EXACT_WIDE_BACKENDS),
                exact_wide_limb_counts=(4,),
                metadata={
                    "workflow_name": "large_shape_release_validation",
                    "large_shape_role": "exact_wide_export_required_baselines",
                    "promotion_scope": "cpu_backed_2048_release_review_candidate",
                    "validation_contract": "same_contract_cpu_direct_accelerator_release_review",
                },
            ),
            ScenarioItem(
                "large-release-validation",
                "finite-ring-2048-hot-required-baselines",
                "finite-u8-ring",
                large_finite_2048,
                "large finite-ring u8 hot-modulus release-validation workload",
                "finite_u8_canonical_host_export",
                "adds CPU and Direct-HIP comparator coverage to the 2048 hot-modulus finite matrix",
                backends=tuple(FINITE_BACKENDS),
                finite_moduli=(251, 255, 256),
                metadata={
                    "workflow_name": "large_shape_release_validation",
                    "large_shape_role": "finite_hot_modulus_required_baselines",
                    "promotion_scope": "cpu_backed_2048_release_review_candidate",
                    "validation_contract": "same_contract_cpu_direct_accelerator_release_review",
                },
            ),
            ScenarioItem(
                "large-release-validation",
                "finite-field-2048-hot-required-baselines",
                "finite-u8-field",
                large_finite_2048,
                "large finite-field u8 hot-prime release-validation workload",
                "finite_u8_canonical_host_export",
                "adds CPU and Direct-HIP comparator coverage to the 2048 hot-prime finite matrix",
                backends=tuple(FINITE_BACKENDS),
                finite_moduli=(251,),
                metadata={
                    "workflow_name": "large_shape_release_validation",
                    "large_shape_role": "finite_hot_prime_required_baselines",
                    "promotion_scope": "cpu_backed_2048_release_review_candidate",
                    "validation_contract": "same_contract_cpu_direct_accelerator_release_review",
                },
            ),
            ScenarioItem(
                "large-release-validation",
                "wrap64-2048-required-baselines",
                "wrap-u64",
                large_wrap64_2048,
                "large strict wrap64 release-validation workload",
                "low64_wrap_u64_host_output",
                "adds the required byte-limb reference beside large direct-HIP wrap64 v4 evidence",
                backends=tuple(WRAP64_BACKENDS),
                metadata={
                    "workflow_name": "large_shape_release_validation",
                    "large_shape_role": "wrap64_required_baselines",
                    "promotion_scope": "cpu_backed_2048_release_review_candidate",
                    "validation_contract": "same_contract_byte_limb_direct_hip_release_review",
                },
            ),
        ],
        "large-release-validation-4096-budgeted": [
            ScenarioItem(
                "large-release-validation-4096-budgeted",
                "bounded-i64-4096-budgeted-baselines",
                "bounded-i64",
                large_4096,
                "budgeted 4096 bounded i64 release-validation dry-run/resume workload",
                "host_export",
                "keeps 4096 proof collection resumable and memory-capped before any release claim",
                backends=tuple(BOUNDED_BACKENDS),
                release_gate="large-release-validation-4096-budgeted",
                metadata={
                    "large_shape_role": "budgeted_release_validation_probe",
                    "promotion_scope": "non_promoting_budgeted_dry_run",
                    "resume_policy": "use --skip-existing with --max-new-captures",
                    "timeout_policy": "runner_enforced_timeout_required",
                    "cpu_chunk_metadata": "record chunking in reviewed summary if CPU baseline is split",
                    "memory_cap_metadata": "record host and device cap before accepting capture",
                },
            ),
            ScenarioItem(
                "large-release-validation-4096-budgeted",
                "bounded-u64-4096-budgeted-baselines",
                "bounded-u64",
                large_4096,
                "budgeted 4096 bounded u64 release-validation dry-run/resume workload",
                "host_export",
                "keeps 4096 unsigned proof collection resumable and memory-capped before any release claim",
                backends=tuple(BOUNDED_BACKENDS),
                release_gate="large-release-validation-4096-budgeted",
                metadata={
                    "large_shape_role": "budgeted_release_validation_probe",
                    "promotion_scope": "non_promoting_budgeted_dry_run",
                    "resume_policy": "use --skip-existing with --max-new-captures",
                    "timeout_policy": "runner_enforced_timeout_required",
                    "cpu_chunk_metadata": "record chunking in reviewed summary if CPU baseline is split",
                    "memory_cap_metadata": "record host and device cap before accepting capture",
                },
            ),
            ScenarioItem(
                "large-release-validation-4096-budgeted",
                "finite-ring-4096-hot-budgeted-baselines",
                "finite-u8-ring",
                large_finite_4096,
                "budgeted 4096 finite-ring u8 hot-modulus release-validation workload",
                "finite_u8_canonical_host_export",
                "adds CPU/direct baselines to the finite-ring 4096 hot-modulus throughput signals",
                backends=tuple(FINITE_BACKENDS),
                finite_moduli=(251, 255, 256),
                release_gate="large-release-validation-4096-budgeted",
                metadata={
                    "large_shape_role": "budgeted_finite_hot_modulus_probe",
                    "promotion_scope": "non_promoting_budgeted_dry_run",
                    "resume_policy": "use --skip-existing with --max-new-captures",
                    "timeout_policy": "runner_enforced_timeout_required",
                    "memory_cap_metadata": "record host output allocation bytes and device memory pressure",
                },
            ),
            ScenarioItem(
                "large-release-validation-4096-budgeted",
                "finite-field-4096-hot-budgeted-baselines",
                "finite-u8-field",
                large_finite_4096,
                "budgeted 4096 finite-field u8 hot-prime release-validation workload",
                "finite_u8_canonical_host_export",
                "adds CPU/direct baselines to the finite-field 4096 hot-prime throughput signal",
                backends=tuple(FINITE_BACKENDS),
                finite_moduli=(251,),
                release_gate="large-release-validation-4096-budgeted",
                metadata={
                    "large_shape_role": "budgeted_finite_hot_prime_probe",
                    "promotion_scope": "non_promoting_budgeted_dry_run",
                    "resume_policy": "use --skip-existing with --max-new-captures",
                    "timeout_policy": "runner_enforced_timeout_required",
                    "memory_cap_metadata": "record host output allocation bytes and device memory pressure",
                },
            ),
            ScenarioItem(
                "large-release-validation-4096-budgeted",
                "wrap64-4096-budgeted-baselines",
                "wrap-u64",
                large_wrap64_4096,
                "budgeted 4096 strict wrap64 byte-limb release-validation workload",
                "low64_wrap_u64_host_output",
                "adds CPU byte-limb and Direct-HIP comparator coverage to the strict wrap64 4096 throughput signal",
                backends=tuple(WRAP64_BACKENDS),
                release_gate="large-release-validation-4096-budgeted",
                metadata={
                    "large_shape_role": "budgeted_wrap64_direct_hip_probe",
                    "promotion_scope": "non_promoting_budgeted_dry_run",
                    "resume_policy": "use --skip-existing with --max-new-captures",
                    "timeout_policy": "runner_enforced_timeout_required",
                    "memory_cap_metadata": "record byte-limb output allocation bytes and device memory pressure",
                },
            ),
            ScenarioItem(
                "large-release-validation-4096-budgeted",
                "exact-wide-signed-4096-budgeted-export",
                "exact-wide-signed",
                large_exact_4096,
                "budgeted 4096 exact-wide signed export-heavy release-validation dry-run",
                "exact_wide_signed_limbs",
                "separates exact-wide export pressure from bounded throughput before optimization starts",
                backends=tuple(EXACT_WIDE_BACKENDS),
                exact_wide_limb_counts=(4,),
                release_gate="large-release-validation-4096-budgeted",
                metadata={
                    "large_shape_role": "budgeted_exact_wide_export_probe",
                    "promotion_scope": "non_promoting_budgeted_dry_run",
                    "resume_policy": "use --skip-existing with --max-new-captures",
                    "timeout_policy": "runner_enforced_timeout_required",
                    "memory_cap_metadata": "record output limb allocation bytes and device memory pressure",
                },
            ),
            ScenarioItem(
                "large-release-validation-4096-budgeted",
                "exact-wide-unsigned-4096-budgeted-export",
                "exact-wide-unsigned",
                large_exact_4096,
                "budgeted 4096 exact-wide unsigned export-heavy release-validation dry-run",
                "exact_wide_unsigned_limbs",
                "adds CPU and Direct-HIP comparator coverage to the unsigned exact-wide 4096 throughput signal",
                backends=tuple(EXACT_WIDE_BACKENDS),
                exact_wide_limb_counts=(4,),
                release_gate="large-release-validation-4096-budgeted",
                metadata={
                    "large_shape_role": "budgeted_exact_wide_export_probe",
                    "promotion_scope": "non_promoting_budgeted_dry_run",
                    "resume_policy": "use --skip-existing with --max-new-captures",
                    "timeout_policy": "runner_enforced_timeout_required",
                    "memory_cap_metadata": "record output limb allocation bytes and device memory pressure",
                },
            ),
        ],
        "hipblaslt-bounded-i64-1024-ab": [
            ScenarioItem(
                "hipblaslt-bounded-i64-1024-ab",
                "bounded-i64-1024-current-reducer",
                "bounded-i64",
                repeated_b_1024,
                "hipBLASLt bounded i64 1024 current reducer A/B workload",
                "host_export",
                "keeps hipBLASLt reducer evidence comparable against Direct-HIP and CPU at a narrow-margin shape",
                backends=("cpu", "hip-direct", "hipblaslt"),
                metadata={
                    "promotion_scope": "narrow_margin_review_only",
                    "comparison_role": "current_reducer_vs_direct_hip_baseline",
                    "margin_policy": "require reviewed same-contract margin before promotion",
                },
            ),
            ScenarioItem(
                "hipblaslt-bounded-i64-1024-ab",
                "bounded-i64-1024-reuse-b",
                "bounded-i64",
                repeated_b_1024,
                "hipBLASLt bounded i64 1024 stable-B reuse A/B workload",
                "host_export",
                "separates setup-inclusive reusable-B behavior from ordinary per-repeat packing",
                backends=("hip-direct", "hipblaslt"),
                pack_mode="prepacked_reuse_b",
                metadata={
                    "promotion_scope": "explicit_reuse_contract_only",
                    "comparison_role": "reuse_b_vs_repack",
                    "margin_policy": "narrow_margin_report_required",
                },
            ),
        ],
        "finite-modulus-map": [
            ScenarioItem(
                "finite-modulus-map",
                "finite-ring-map-512",
                "finite-u8-ring",
                finite_generic_512,
                "finite-ring u8 modulus map over hot and generic byte moduli",
                "finite_u8_canonical_host_export",
                "maps prime/composite and power-of-two behavior without assuming the hot-modulus kernels generalize",
                backends=("cpu", "hip-direct", "ck", "rocwmma"),
                finite_moduli=(127, 241, 243, 251, 253, 255, 256),
                metadata={
                    "modulus_role": "prime_composite_power_of_two_map",
                    "promotion_scope": "non_promoting_modulus_map",
                    "prime_or_composite": "mixed",
                },
            ),
            ScenarioItem(
                "finite-modulus-map",
                "finite-field-prime-map-512",
                "finite-u8-field",
                finite_generic_512,
                "finite-field u8 prime modulus map",
                "finite_u8_canonical_host_export",
                "keeps prime field behavior separate from composite finite-ring captures",
                backends=("cpu", "hip-direct", "ck", "rocwmma"),
                finite_moduli=(127, 241, 251),
                metadata={
                    "modulus_role": "prime_field_map",
                    "promotion_scope": "non_promoting_modulus_map",
                    "prime_or_composite": "prime",
                },
            ),
        ],
        "modulus-set-autotune": [
            ScenarioItem(
                "modulus-set-autotune",
                "bounded-i64-prefix5-experimental-ladder",
                "bounded-i64",
                layout_512,
                "experimental modulus-set and residue-count evidence for bounded i64",
                "host_export",
                "feeds offline ladder search and residue-count autotuning without cache promotion",
                backends=("cpu", "hip-direct"),
                prefix_policy="fixed-requested",
                max_prefix=5,
                modulus_set="experimental:prefix5-byte-ladder-search",
                metadata={
                    "promotion_scope": "non_promoting_modulus_set_experiment",
                    "cache_promotion_blocker": "experimental_modulus_set",
                    "reducer_cost_policy": "offline_search_report_required",
                },
            ),
            ScenarioItem(
                "modulus-set-autotune",
                "bounded-u64-prefix9-default-count",
                "bounded-u64",
                layout_512,
                "default ladder residue-count comparison for bounded u64",
                "host_export",
                "anchors experimental residue-count captures to the current default ladder",
                backends=("cpu", "hip-direct"),
                prefix_policy="fixed-requested",
                max_prefix=9,
                metadata={
                    "promotion_scope": "comparison_anchor_only",
                    "cache_promotion_blocker": "residue_count_policy_review_required",
                },
            ),
        ],
        "tile-shape-sweeps": [
            ScenarioItem(
                "tile-shape-sweeps",
                "bounded-i64-512-64x64",
                "bounded-i64",
                tile_512_64,
                "Direct-HIP bounded i64 512 tile-shape variant",
                "host_export",
                "groups tile-shape evidence with kernel/resource reports before selected-kernel changes",
                backends=("hip-direct",),
                tile_shape_variant="direct-hip-bounded-512-64x64",
                metadata={"promotion_scope": "tile_shape_evidence_only", "resource_report_required": "isa_or_counter"},
            ),
            ScenarioItem(
                "tile-shape-sweeps",
                "bounded-i64-512-256x128",
                "bounded-i64",
                tile_512_256,
                "Direct-HIP bounded i64 512 asymmetric tile-shape variant",
                "host_export",
                "tests occupancy and memory traffic sensitivity to tile M/N without changing public routing",
                backends=("hip-direct",),
                tile_shape_variant="direct-hip-bounded-512-256x128",
                metadata={"promotion_scope": "tile_shape_evidence_only", "resource_report_required": "isa_or_counter"},
            ),
            ScenarioItem(
                "tile-shape-sweeps",
                "bounded-i64-1024-64x128",
                "bounded-i64",
                tile_1024_64,
                "Direct-HIP bounded i64 1024 tile-shape variant",
                "host_export",
                "compares 1024-size tile shape choices against event/resource evidence",
                backends=("hip-direct",),
                tile_shape_variant="direct-hip-bounded-1024-64x128",
                metadata={"promotion_scope": "tile_shape_evidence_only", "resource_report_required": "isa_or_counter"},
            ),
            ScenarioItem(
                "tile-shape-sweeps",
                "bounded-i64-1024-256x256",
                "bounded-i64",
                tile_1024_256,
                "Direct-HIP bounded i64 1024 large tile-shape variant",
                "host_export",
                "detects stale-kernel/resource regressions before any tile-shape promotion",
                backends=("hip-direct",),
                tile_shape_variant="direct-hip-bounded-1024-256x256",
                metadata={"promotion_scope": "tile_shape_evidence_only", "resource_report_required": "isa_or_counter"},
            ),
            ScenarioItem(
                "tile-shape-sweeps",
                "finite-ring-2048-64x64",
                "finite-u8-ring",
                tile_finite_2048_64,
                "Direct-HIP finite-ring u8 2048 tile-shape variant",
                "finite_u8_canonical_host_export",
                "keeps finite-u8 tile-shape evidence separate from bounded RNS kernels",
                backends=("hip-direct",),
                finite_moduli=(251,),
                tile_shape_variant="direct-hip-finite-2048-64x64",
                metadata={"promotion_scope": "tile_shape_evidence_only", "resource_report_required": "isa_or_counter"},
            ),
            ScenarioItem(
                "tile-shape-sweeps",
                "finite-ring-2048-256x128",
                "finite-u8-ring",
                tile_finite_2048_256,
                "Direct-HIP finite-ring u8 2048 asymmetric tile-shape variant",
                "finite_u8_canonical_host_export",
                "checks finite-u8 occupancy and bandwidth sensitivity before finite-kernel selection changes",
                backends=("hip-direct",),
                finite_moduli=(251,),
                tile_shape_variant="direct-hip-finite-2048-256x128",
                metadata={"promotion_scope": "tile_shape_evidence_only", "resource_report_required": "isa_or_counter"},
            ),
        ],
        "exact-wide-output-chain": [
            ScenarioItem(
                "exact-wide-output-chain",
                "exact-wide-signed-chain3-final-export",
                "exact-wide-signed",
                chain_256,
                "exact-wide signed RNS chain with final checksum/export outside measured repeats",
                "residue_current_then_final_limb_export",
                "keeps lazy-output/reconstruction pressure separate from per-repeat GEMM timing",
                backends=("cpu", "hip-direct", "ck", "rocwmma"),
                residue_chain_length=3,
                next_op_hint="rns-gemm",
                exact_wide_limb_counts=(4,),
                metadata={"promotion_scope": "lazy_export_chain_evidence_only", "output_domain_requirement": "rns_then_final_export"},
            ),
            ScenarioItem(
                "exact-wide-output-chain",
                "exact-wide-unsigned-chain3-reuse-b",
                "exact-wide-unsigned",
                chain_256,
                "exact-wide unsigned RNS chain with stable RHS packed once before warmups",
                "residue_current_then_final_limb_export",
                "tests whether reusable B matters in exact-wide lazy-output chains",
                backends=("hip-direct", "ck", "rocwmma"),
                residue_chain_length=3,
                pack_mode="prepacked_reuse_b",
                next_op_hint="rns-gemm",
                exact_wide_limb_counts=(4,),
                metadata={"promotion_scope": "lazy_export_chain_reuse_evidence_only", "reuse_contract": "stable_rhs_exact_wide_chain"},
            ),
        ],
        "export-bound-limb-variants": [
            ScenarioItem(
                "export-bound-limb-variants",
                "exact-wide-signed-limb-zoo",
                "exact-wide-signed",
                exact_512,
                "exact-wide signed fixed-limb export variants",
                "exact_wide_signed_limbs",
                "reuses current fixed-limb export kernels while making export-bound behavior explicit",
                backends=("cpu", "hip-direct", "ck", "rocwmma"),
                exact_wide_limb_counts=(1, 2, 3, 4, 8, 16, 32),
                export_variant="fixed_limb_export_zoo",
                metadata={"promotion_scope": "export_variant_evidence_only", "cache_promotion_blocker": "fixed_limb_export_review_required"},
            ),
            ScenarioItem(
                "export-bound-limb-variants",
                "exact-wide-unsigned-limb-zoo",
                "exact-wide-unsigned",
                exact_512,
                "exact-wide unsigned fixed-limb export variants",
                "exact_wide_unsigned_limbs",
                "checks whether unsigned limb count changes export/status balance differently from signed",
                backends=("cpu", "hip-direct", "ck", "rocwmma"),
                exact_wide_limb_counts=(1, 2, 3, 4, 8, 16, 32),
                export_variant="fixed_limb_export_zoo",
                metadata={"promotion_scope": "export_variant_evidence_only", "cache_promotion_blocker": "fixed_limb_export_review_required"},
            ),
        ],
        "reconstruction-zoo": [
            ScenarioItem(
                "reconstruction-zoo",
                "bounded-i64-garner-precomputed",
                "bounded-i64",
                layout_512,
                "benchmark-only Garner reconstruction variant metadata",
                "host_export",
                "plumbs reconstruction variant evidence before any CRT fusion or kernel change",
                backends=("cpu", "hip-direct"),
                reconstruction_variant="experimental:garner_precomputed_constants",
                metadata={"promotion_scope": "reconstruction_variant_evidence_only", "cache_promotion_blocker": "experimental_reconstruction_variant"},
            ),
            ScenarioItem(
                "reconstruction-zoo",
                "exact-wide-signed-mixed-radix",
                "exact-wide-signed",
                exact_512,
                "benchmark-only exact-wide mixed-radix reconstruction variant metadata",
                "exact_wide_signed_limbs",
                "keeps reconstruction-zoo captures exact-checked against current output while non-promoting",
                backends=("cpu", "hip-direct"),
                exact_wide_limb_counts=(4,),
                reconstruction_variant="experimental:mixed_radix_fixed_prefix20",
                metadata={"promotion_scope": "reconstruction_variant_evidence_only", "cache_promotion_blocker": "experimental_reconstruction_variant"},
            ),
        ],
        "grouped-dispatch": [
            ScenarioItem(
                "grouped-dispatch",
                "bounded-i64-64-group32",
                "bounded-i64",
                many_small_64,
                "many-small bounded i64 grouped-dispatch metadata capture",
                "host_export",
                "compares independent and host-batch captures with grouped descriptor identity before public resident API work",
                backends=("hip-direct",),
                grouped_dispatch_tasks=32,
                metadata={"promotion_scope": "grouped_dispatch_evidence_only", "grouping_role": "same_shape_grouped_descriptor"},
            ),
            ScenarioItem(
                "grouped-dispatch",
                "finite-ring-64-group32",
                "finite-u8-ring",
                many_small_64,
                "many-small finite-ring grouped-dispatch metadata capture",
                "finite_u8_canonical_host_export",
                "keeps finite-u8 grouped-dispatch evidence separate from bounded exact paths",
                backends=("hip-direct",),
                finite_moduli=(251,),
                grouped_dispatch_tasks=32,
                metadata={"promotion_scope": "grouped_dispatch_evidence_only", "grouping_role": "same_shape_grouped_descriptor"},
            ),
            ScenarioItem(
                "grouped-dispatch",
                "exact-wide-signed-64-group32",
                "exact-wide-signed",
                many_small_64,
                "many-small exact-wide signed grouped-dispatch capture",
                "exact_wide_signed_limbs",
                "tests the grouped-dispatch path on the only host-batch workload that beat the fastest independent baseline",
                backends=("hip-direct",),
                exact_wide_limb_counts=(4,),
                grouped_dispatch_tasks=32,
                metadata={
                    "promotion_scope": "grouped_dispatch_evidence_only",
                    "grouping_role": "same_shape_grouped_descriptor",
                    "comparison_required": "fastest_independent_and_same_backend_host_batch",
                    "prior_host_batch_signal": "direct_hip_exact_wide_signed_64_hostbatch32",
                },
            ),
        ],
        "resident-lifetime-arena": [
            ScenarioItem(
                "resident-lifetime-arena",
                "bounded-i64-512-reuse-b-arena",
                "bounded-i64",
                layout_512,
                "Direct-HIP resident lifetime plus workspace arena reuse evidence",
                "host_export",
                "proves resident source-version and arena allocation-free repeat metadata before public lifetime APIs",
                backends=("hip-direct",),
                pack_mode="prepacked_reuse_b",
                resident_lifetime=True,
                workspace_arena=True,
                metadata={
                    "promotion_scope": "resident_lifetime_arena_evidence_only",
                    "stale_source_policy": "source_descriptor_semantic_prefix_target_workspace_mismatch_rejects",
                    "allocation_policy": "zero_measured_repeat_allocation_required_for_promotion",
                },
            ),
            ScenarioItem(
                "resident-lifetime-arena",
                "exact-wide-signed-chain3-arena",
                "exact-wide-signed",
                chain_256,
                "exact-wide residue-current chain resident lifetime evidence",
                "residue_current_then_final_limb_export",
                "ties lazy exact-wide output to explicit resident currentness and workspace identity",
                backends=("hip-direct",),
                residue_chain_length=3,
                next_op_hint="rns-gemm",
                exact_wide_limb_counts=(4,),
                resident_lifetime=True,
                workspace_arena=True,
                metadata={
                    "promotion_scope": "resident_lifetime_arena_evidence_only",
                    "output_domain_requirement": "rns_then_final_export",
                },
            ),
        ],
        "adaptive-grouped-scheduler": [
            ScenarioItem(
                "adaptive-grouped-scheduler",
                "bounded-u64-adaptive-bands-grouped",
                "bounded-u64",
                adaptive_1024,
                "Direct-HIP adaptive prefix grouped scheduler evidence",
                "host_export",
                "records prefix/tile/zero-mask grouping identity before routing grouped adaptive execution",
                backends=("hip-direct",),
                prefix_policy="minimum-proven",
                bound_source="input-scan",
                adaptive_grouped_scheduler=True,
                metadata={
                    "promotion_scope": "adaptive_grouped_scheduler_evidence_only",
                    "schedule_strategy": "prefix_tile_zero_mask_grouped_descriptors",
                },
            ),
        ],
        "streaming-overlap": [
            ScenarioItem(
                "streaming-overlap",
                "bounded-i64-512-reuse-b-overlap",
                "bounded-i64",
                layout_512,
                "Direct-HIP repeated-B pack/compute/export overlap evidence",
                "host_export",
                "declares double-buffered stream dependency contracts before overlap routing",
                backends=("hip-direct",),
                pack_mode="prepacked_reuse_b",
                resident_lifetime=True,
                workspace_arena=True,
                streaming_overlap=True,
                metadata={
                    "promotion_scope": "streaming_overlap_evidence_only",
                    "dependency_contract": "pack_before_gemm_gemm_before_export_status_before_host_read",
                },
            ),
            ScenarioItem(
                "streaming-overlap",
                "exact-wide-signed-chain3-overlap",
                "exact-wide-signed",
                chain_256,
                "exact-wide chain plus final export overlap evidence",
                "residue_current_then_final_limb_export",
                "tests whether final export can overlap with next pack without changing output contract",
                backends=("hip-direct",),
                residue_chain_length=3,
                next_op_hint="rns-gemm",
                exact_wide_limb_counts=(4,),
                resident_lifetime=True,
                workspace_arena=True,
                streaming_overlap=True,
                metadata={"promotion_scope": "streaming_overlap_evidence_only"},
            ),
        ],
        "release-gate-closeout": [
            ScenarioItem(
                "release-gate-closeout",
                "bounded-i64-4096-budgeted",
                "bounded-i64",
                large_4096,
                "budgeted 4096 release gate with chunk/resume metadata",
                "host_export",
                "keeps 4096 classification separate from installed-cache eligibility until reviewed",
                backends=("cpu", "hip-direct"),
                release_gate="large-release-validation-4096-budgeted",
                metadata={
                    "large_shape_role": "budgeted_4096_release_gate",
                    "promotion_scope": "release_gate_review_required",
                    "resume_policy": "use --skip-existing with --max-new-captures",
                },
            ),
            ScenarioItem(
                "release-gate-closeout",
                "finite-ring-4096-budgeted",
                "finite-u8-ring",
                large_4096,
                "budgeted finite-u8 4096 release gate with chunk/resume metadata",
                "finite_u8_canonical_host_export",
                "keeps finite 4096 target claims blocked until CPU/direct baselines exist",
                backends=("cpu", "hip-direct"),
                finite_moduli=(251,),
                release_gate="large-release-validation-4096-budgeted",
                metadata={"large_shape_role": "budgeted_4096_release_gate", "promotion_scope": "release_gate_review_required"},
            ),
        ],
        "fhe-lattice-proxy-starfoundry": [
            ScenarioItem(
                "fhe-lattice-proxy-starfoundry",
                "key-switch-reuse-b-output-rns",
                "bounded-i64",
                fhe_key_switch,
                "FHE/lattice key-switch proxy with stable RHS and RNS continuation intent",
                "residue_current_then_final_export",
                "groups tower/reuse/output-domain metadata without claiming compatibility with an FHE library",
                backends=("hip-direct", "ck", "rocwmma"),
                pack_mode="prepacked_reuse_b",
                next_op_hint="rns-gemm",
                workload_proxy="fhe:key_switch_digit_aggregation",
                verification_amortization="reuse_shape_seed_reference_inputs",
                metadata={
                    "source_role": "fhe_lattice_proxy",
                    "algebra_family": "fhe_lattice",
                    "workflow_name": "key_switch_digit_aggregation",
                    "reuse_profile": "large_read_only_key_material",
                    "output_domain_requirement": "rns_residue_current",
                    "promotion_scope": "proxy_evidence_only",
                },
            ),
            ScenarioItem(
                "fhe-lattice-proxy-starfoundry",
                "ckks-linear-layer-final-export",
                "bounded-i64",
                fhe_linear_layer,
                "FHE/lattice dense linear-layer proxy with final export",
                "host_export",
                "keeps proxy workload grouping visible without asserting frontend library readiness",
                backends=("cpu", "hip-direct", "ck", "rocwmma"),
                workload_proxy="fhe:ckks_linear_layer_dense_proxy",
                verification_amortization="reuse_shape_seed_reference_inputs",
                metadata={
                    "source_role": "fhe_lattice_proxy",
                    "algebra_family": "fhe_lattice",
                    "workflow_name": "ckks_linear_layer_dense_proxy",
                    "reuse_profile": "single_call_dense_layer",
                    "output_domain_requirement": "host_export",
                    "promotion_scope": "proxy_evidence_only",
                },
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
                "bounded-i64-2048-reuse-b",
                "bounded-i64",
                large_2048,
                "large bounded i64 exploratory workload with stable RHS packed once before warmups",
                "host_export",
                "checks whether pack-bound matrix-engine paths at 2048 benefit from reusable B residency",
                backends=accelerator_backends,
                pack_mode="prepacked_reuse_b",
                metadata={
                    "large_shape_role": "throughput_probe_reuse_b",
                    "promotion_scope": "exploratory_only",
                    "reuse_contract": "large_stable_rhs_prepacked_before_warmups",
                    "setup_cost_policy": "prepack_setup_us_excluded_from_per_repeat_included_in_review",
                },
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
            ScenarioItem(
                "large-exploratory",
                "bounded-u64-2048-reuse-b",
                "bounded-u64",
                large_2048,
                "large bounded u64 exploratory workload with stable RHS packed once before warmups",
                "host_export",
                "checks whether large unsigned bounded paths amortize reusable B residency differently from signed paths",
                backends=accelerator_backends,
                pack_mode="prepacked_reuse_b",
                metadata={
                    "large_shape_role": "throughput_probe_reuse_b",
                    "promotion_scope": "exploratory_only",
                    "reuse_contract": "large_stable_rhs_prepacked_before_warmups",
                    "setup_cost_policy": "prepack_setup_us_excluded_from_per_repeat_included_in_review",
                },
            ),
            ScenarioItem(
                "large-exploratory",
                "bounded-i64-4096",
                "bounded-i64",
                large_4096,
                "very large bounded i64 exploratory throughput workload",
                "host_export",
                "checks whether bounded i64 remains launch/export-bound or becomes compute/throughput-bound at 4096",
                backends=bounded_gpu_backends,
                metadata={"large_shape_role": "throughput_probe", "promotion_scope": "exploratory_only"},
            ),
            ScenarioItem(
                "large-exploratory",
                "bounded-i64-4096-reuse-b",
                "bounded-i64",
                large_4096,
                "very large bounded i64 throughput workload with stable RHS packed once before warmups",
                "host_export",
                "measures whether reusable B is still valuable after launch overhead is fully amortized",
                backends=accelerator_backends,
                pack_mode="prepacked_reuse_b",
                metadata={
                    "large_shape_role": "throughput_probe_reuse_b",
                    "promotion_scope": "exploratory_only",
                    "reuse_contract": "large_stable_rhs_prepacked_before_warmups",
                    "setup_cost_policy": "prepack_setup_us_excluded_from_per_repeat_included_in_review",
                },
            ),
            ScenarioItem(
                "large-exploratory",
                "bounded-u64-4096",
                "bounded-u64",
                large_4096,
                "very large bounded u64 exploratory throughput workload",
                "host_export",
                "checks whether native vector, Direct-HIP, and accelerator paths scale differently once launch overhead is amortized",
                backends=bounded_gpu_backends,
                metadata={"large_shape_role": "throughput_probe", "promotion_scope": "exploratory_only"},
            ),
            ScenarioItem(
                "large-exploratory",
                "bounded-u64-4096-reuse-b",
                "bounded-u64",
                large_4096,
                "very large bounded u64 throughput workload with stable RHS packed once before warmups",
                "host_export",
                "tests whether large unsigned matrix-engine paths should prefer explicit stable-B contracts",
                backends=accelerator_backends,
                pack_mode="prepacked_reuse_b",
                metadata={
                    "large_shape_role": "throughput_probe_reuse_b",
                    "promotion_scope": "exploratory_only",
                    "reuse_contract": "large_stable_rhs_prepacked_before_warmups",
                    "setup_cost_policy": "prepack_setup_us_excluded_from_per_repeat_included_in_review",
                },
            ),
            ScenarioItem(
                "large-exploratory",
                "exact-wide-signed-2048",
                "exact-wide-signed",
                large_exact_2048,
                "large exact-wide signed exploratory export-heavy workload",
                "exact_wide_signed_limbs",
                "extends exact-wide evidence beyond 512/1024 while keeping large-shape output-limb costs visible",
                backends=EXACT_WIDE_BACKENDS,
                exact_wide_limb_counts=(4,),
                metadata={"large_shape_role": "exact_wide_export_probe", "promotion_scope": "exploratory_only"},
            ),
            ScenarioItem(
                "large-exploratory",
                "exact-wide-unsigned-2048",
                "exact-wide-unsigned",
                large_exact_2048,
                "large exact-wide unsigned exploratory export-heavy workload",
                "exact_wide_unsigned_limbs",
                "checks whether unsigned exact-wide export behavior diverges from signed large-shape behavior",
                backends=EXACT_WIDE_BACKENDS,
                exact_wide_limb_counts=(4,),
                metadata={"large_shape_role": "exact_wide_export_probe", "promotion_scope": "exploratory_only"},
            ),
            ScenarioItem(
                "large-exploratory",
                "exact-wide-signed-4096",
                "exact-wide-signed",
                large_exact_4096,
                "very large exact-wide signed exploratory export-heavy workload",
                "exact_wide_signed_limbs",
                "pushes exact-wide prefix-20 export and GEMM balance into a throughput-dominated shape when runtime is tolerable",
                backends=EXACT_WIDE_BACKENDS,
                exact_wide_limb_counts=(4,),
                metadata={"large_shape_role": "exact_wide_throughput_probe", "promotion_scope": "exploratory_only"},
            ),
            ScenarioItem(
                "large-exploratory",
                "exact-wide-unsigned-4096",
                "exact-wide-unsigned",
                large_exact_4096,
                "very large exact-wide unsigned exploratory export-heavy workload",
                "exact_wide_unsigned_limbs",
                "keeps unsigned exact-wide 4096 evidence separate from signed and default bounded paths",
                backends=EXACT_WIDE_BACKENDS,
                exact_wide_limb_counts=(4,),
                metadata={"large_shape_role": "exact_wide_throughput_probe", "promotion_scope": "exploratory_only"},
            ),
            ScenarioItem(
                "large-exploratory",
                "finite-ring-2048",
                "finite-u8-ring",
                large_finite_2048,
                "large finite-ring u8 hot-modulus exploratory workload",
                "finite_u8_canonical_host_export",
                "tests whether 1024 finite-u8 accelerator wins generalize to larger hot-modulus shapes",
                backends=FINITE_BACKENDS,
                finite_moduli=(251, 255, 256),
                metadata={"large_shape_role": "finite_hot_modulus_probe", "promotion_scope": "exploratory_only"},
            ),
            ScenarioItem(
                "large-exploratory",
                "finite-field-2048",
                "finite-u8-field",
                large_finite_2048,
                "large finite-field u8 hot-prime exploratory workload",
                "finite_u8_canonical_host_export",
                "keeps field-251 large-shape evidence separate from ring-modulus behavior",
                backends=FINITE_BACKENDS,
                finite_moduli=(251,),
                metadata={"large_shape_role": "finite_hot_prime_probe", "promotion_scope": "exploratory_only"},
            ),
            ScenarioItem(
                "large-exploratory",
                "finite-ring-4096",
                "finite-u8-ring",
                large_finite_4096,
                "very large finite-ring u8 hot-modulus exploratory workload",
                "finite_u8_canonical_host_export",
                "shows whether finite hot-modulus paths remain throughput-effective at 4096 when runtime is tolerable",
                backends=FINITE_BACKENDS,
                finite_moduli=(251, 255, 256),
                metadata={"large_shape_role": "finite_hot_modulus_throughput_probe", "promotion_scope": "exploratory_only"},
            ),
            ScenarioItem(
                "large-exploratory",
                "finite-field-4096",
                "finite-u8-field",
                large_finite_4096,
                "very large finite-field u8 hot-prime exploratory workload",
                "finite_u8_canonical_host_export",
                "separates large field-251 throughput from finite-ring composite-modulus behavior",
                backends=FINITE_BACKENDS,
                finite_moduli=(251,),
                metadata={"large_shape_role": "finite_hot_prime_throughput_probe", "promotion_scope": "exploratory_only"},
            ),
            ScenarioItem(
                "large-exploratory",
                "wrap64-2048",
                "wrap-u64",
                large_wrap64_2048,
                "large strict wrap64 direct byte-limb exploratory workload",
                "low64_wrap_u64_host_output",
                "extends direct-HIP wrap64 v4 evidence beyond 1024 without claiming matrix-engine promotion",
                backends=tuple(WRAP64_BACKENDS),
                metadata={"large_shape_role": "wrap64_direct_hip_probe", "promotion_scope": "exploratory_only"},
            ),
            ScenarioItem(
                "large-exploratory",
                "wrap64-4096",
                "wrap-u64",
                large_wrap64_4096,
                "very large strict wrap64 direct byte-limb exploratory workload",
                "low64_wrap_u64_host_output",
                "tests the upper safe uint32-accumulation shape for direct-HIP wrap64 v4 when runtime is tolerable",
                backends=tuple(WRAP64_BACKENDS),
                metadata={"large_shape_role": "wrap64_direct_hip_throughput_probe", "promotion_scope": "exploratory_only"},
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
    scenario_args.residue_chain_final_export = item.residue_chain_final_export
    scenario_args.output_ld_padding = item.output_ld_padding
    scenario_args.host_api_batch_size = item.host_api_batch_size
    scenario_args.native_to_rns_bridge = item.native_to_rns_bridge
    scenario_args.vector_to_rns_chain = item.vector_to_rns_chain
    scenario_args.prefix_policy = item.prefix_policy or getattr(args, "prefix_policy", None)
    scenario_args.max_prefix = item.max_prefix if item.max_prefix is not None else getattr(args, "max_prefix", None)
    scenario_args.bound_source = item.bound_source or getattr(args, "bound_source", None)
    scenario_args.next_op_hint = item.next_op_hint or getattr(args, "next_op_hint", None)
    scenario_args.residue_channel_fusion = item.residue_channel_fusion
    scenario_args.modulus_set = item.modulus_set or getattr(args, "modulus_set", "default")
    scenario_args.tile_shape_variant = item.tile_shape_variant or getattr(args, "tile_shape_variant", "default")
    scenario_args.export_variant = item.export_variant or getattr(args, "export_variant", "default")
    scenario_args.reconstruction_variant = item.reconstruction_variant or getattr(args, "reconstruction_variant", "default_garner")
    scenario_args.grouped_dispatch_tasks = item.grouped_dispatch_tasks
    scenario_args.hip_graph_replay = item.hip_graph_replay
    scenario_args.workload_proxy = item.workload_proxy or getattr(args, "workload_proxy", "none")
    scenario_args.resident_lifetime = item.resident_lifetime or getattr(args, "resident_lifetime", False)
    scenario_args.workspace_arena = item.workspace_arena or getattr(args, "workspace_arena", False)
    scenario_args.adaptive_grouped_scheduler = item.adaptive_grouped_scheduler or getattr(
        args, "adaptive_grouped_scheduler", False
    )
    scenario_args.streaming_overlap = item.streaming_overlap or getattr(args, "streaming_overlap", False)
    scenario_args.release_gate = (
        item.release_gate if item.release_gate and item.release_gate != "none" else getattr(args, "release_gate", "none")
    )
    scenario_args.verification_amortization = item.verification_amortization or getattr(
        args, "verification_amortization", "none"
    )
    return scenario_args


def scenario_backends_for_item(args: argparse.Namespace, item: ScenarioItem) -> list[str]:
    backends = list(item.backends or default_backends_for(item.semantics, item.case))
    if item.host_api_batch_size > 1:
        backends = [backend for backend in backends if backend in HOST_API_BATCH_BACKENDS]
    if item.hip_graph_replay:
        backends = [backend for backend in backends if backend == "hip-direct"]
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
    metadata = {
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
        "residue_chain_final_export": item.residue_chain_final_export,
        "output_ld_padding": item.output_ld_padding,
        "host_api_batch_size": item.host_api_batch_size,
        "native_to_rns_bridge": item.native_to_rns_bridge,
        "vector_to_rns_chain": item.vector_to_rns_chain,
        "next_op_hint": item.next_op_hint,
        "residue_channel_fusion": item.residue_channel_fusion,
        "modulus_set": item.modulus_set,
        "tile_shape_variant": item.tile_shape_variant,
        "export_variant": item.export_variant,
        "reconstruction_variant": item.reconstruction_variant,
        "grouped_dispatch_tasks": item.grouped_dispatch_tasks,
        "hip_graph_replay": item.hip_graph_replay,
        "workload_proxy": item.workload_proxy,
        "resident_lifetime": item.resident_lifetime,
        "workspace_arena": item.workspace_arena,
        "adaptive_grouped_scheduler": item.adaptive_grouped_scheduler,
        "streaming_overlap": item.streaming_overlap,
        "release_gate": item.release_gate,
        "verification_amortization": item.verification_amortization,
        "oneshot": oneshot,
        "evidence_scope": item.evidence_scope,
        "output_domain": item.output_domain,
        "rationale": item.rationale,
    }
    if item.metadata:
        metadata["metadata"] = item.metadata
    return metadata


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
    residue_chain_final_export: bool = False,
    output_ld_padding: int = 0,
    host_api_batch_size: int = 1,
    hip_graph_replay: bool = False,
    oneshot: bool = False,
) -> str:
    parts = [semantics, case.name, f"{case.m}x{case.n}x{case.k}"]
    if modulus is not None:
        parts.append(f"mod{modulus}")
    if semantics in EXACT_WIDE_SEMANTICS and exact_wide_limb_count not in (None, DEFAULT_EXACT_WIDE_LIMB_COUNT):
        parts.append(f"limbs{exact_wide_limb_count}")
    if semantics in RNS_CHAIN_SEMANTICS and residue_chain_length > 1:
        parts.append(f"chain{residue_chain_length}")
        if residue_chain_final_export:
            parts.append("finalexport")
    if output_ld_padding > 0:
        parts.append(f"outpad{output_ld_padding}")
    if host_api_batch_size > 1:
        parts.append(f"hostbatch{host_api_batch_size}")
    if hip_graph_replay:
        parts.append("hipgraph")
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
    next_op_hint = getattr(args, "next_op_hint", None)
    if next_op_hint:
        command.extend(["--next-op-hint", next_op_hint])
    prefix_policy = getattr(args, "prefix_policy", None)
    max_prefix = getattr(args, "max_prefix", None)
    if prefix_policy:
        command.extend(["--prefix-policy", prefix_policy])
    if max_prefix is not None:
        command.extend(["--max-prefix", str(max_prefix)])
    if args.residue_chain_length > 1:
        command.extend(["--residue-chain-length", str(args.residue_chain_length)])
    if getattr(args, "residue_chain_final_export", False):
        command.append("--residue-chain-final-export")
    host_api_batch_size = int(getattr(args, "host_api_batch_size", 1) or 1)
    if host_api_batch_size > 1:
        command.extend(["--host-api-batch-size", str(host_api_batch_size)])
    output_ld_padding = int(getattr(args, "output_ld_padding", 0) or 0)
    if output_ld_padding > 0:
        command.extend(["--output-ld-padding", str(output_ld_padding)])
    if getattr(args, "residue_channel_fusion", False):
        command.append("--residue-channel-fusion")
    if oneshot:
        command.append("--oneshot")
    pack_mode = requested_pack_mode(args)
    if pack_mode == "prepacked_reuse":
        command.append("--reuse-packed-inputs")
    elif pack_mode == "prepacked_reuse_a":
        command.append("--reuse-packed-a")
    elif pack_mode == "prepacked_reuse_b":
        command.append("--reuse-packed-b")
    if getattr(args, "native_to_rns_bridge", False):
        command.append("--native-to-rns-bridge")
    if getattr(args, "vector_to_rns_chain", False):
        command.append("--vector-to-rns-chain")
    modulus_set = getattr(args, "modulus_set", "default")
    if modulus_set and modulus_set != "default":
        command.extend(["--modulus-set", modulus_set])
    tile_shape_variant = getattr(args, "tile_shape_variant", "default")
    if tile_shape_variant and tile_shape_variant != "default":
        command.extend(["--tile-shape-variant", tile_shape_variant])
    export_variant = getattr(args, "export_variant", "default")
    if export_variant and export_variant != "default":
        command.extend(["--export-variant", export_variant])
    reconstruction_variant = getattr(args, "reconstruction_variant", "default_garner")
    if reconstruction_variant and reconstruction_variant != "default_garner":
        command.extend(["--reconstruction-variant", reconstruction_variant])
    grouped_dispatch_tasks = int(getattr(args, "grouped_dispatch_tasks", 1) or 1)
    if grouped_dispatch_tasks > 1:
        command.extend(["--grouped-dispatch", str(grouped_dispatch_tasks)])
    if getattr(args, "hip_graph_replay", False):
        command.append("--hip-graph-replay")
    workload_proxy = getattr(args, "workload_proxy", "none")
    if workload_proxy and workload_proxy != "none":
        command.extend(["--workload-proxy", workload_proxy])
    if getattr(args, "resident_lifetime", False):
        command.append("--resident-lifetime")
    if getattr(args, "workspace_arena", False):
        command.append("--workspace-arena")
    if getattr(args, "adaptive_grouped_scheduler", False):
        command.append("--adaptive-grouped-scheduler")
    if getattr(args, "streaming_overlap", False):
        command.append("--streaming-overlap")
    release_gate = getattr(args, "release_gate", "none")
    if release_gate and release_gate != "none":
        command.extend(["--release-gate", release_gate])
    verification_amortization = getattr(args, "verification_amortization", "none")
    if verification_amortization and verification_amortization != "none":
        command.extend(["--verification-amortization", verification_amortization])
    return command


def default_sweep_command_entries(args: argparse.Namespace) -> list[SweepCommand]:
    backend_benches = parse_backend_bench(args.bench_for)
    commands: list[SweepCommand] = []
    host_api_batch_size = int(getattr(args, "host_api_batch_size", 1) or 1)
    if host_api_batch_size <= 0:
        raise SystemExit("--host-api-batch-size must be positive")
    if host_api_batch_size > 1:
        if any(
            [
                getattr(args, "include_oneshot", False),
                getattr(args, "oneshot_only", False),
                getattr(args, "reuse_packed_inputs", False),
                getattr(args, "reuse_packed_a", False),
                getattr(args, "reuse_packed_b", False),
                getattr(args, "native_to_rns_bridge", False),
                getattr(args, "vector_to_rns_chain", False),
                getattr(args, "residue_channel_fusion", False),
                int(getattr(args, "residue_chain_length", 1) or 1) != 1,
                getattr(args, "bound_source", None) == "input-scan",
            ]
        ):
            raise SystemExit(
                "--host-api-batch-size > 1 cannot be combined with one-shot, reuse, bridge, chain, "
                "residue-fusion, residue-chain, or input-scan modes"
            )
        if args.backends:
            args.backends = [backend for backend in args.backends if backend in HOST_API_BATCH_BACKENDS]
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
    if getattr(args, "residue_channel_fusion", False):
        if requested_pack_mode(args) != "per_repeat_repack":
            raise SystemExit("--residue-channel-fusion cannot be combined with packed-input reuse")
        if getattr(args, "prefix_policy", None) != "fixed-requested" or getattr(args, "max_prefix", None) not in {9, None}:
            raise SystemExit("--residue-channel-fusion requires --prefix-policy fixed-requested and --max-prefix 9")
        if any(semantics not in BOUNDED_SEMANTICS for semantics in semantics_values):
            raise SystemExit("--residue-channel-fusion is only valid for bounded semantics")
    if getattr(args, "hip_graph_replay", False):
        if getattr(args, "residue_chain_final_export", False):
            raise SystemExit("--hip-graph-replay cannot be combined with --residue-chain-final-export")
        if requested_pack_mode(args) != "prepacked_reuse":
            raise SystemExit("--hip-graph-replay requires --reuse-packed-inputs")
        if args.residue_chain_length <= 1:
            raise SystemExit("--hip-graph-replay requires --residue-chain-length > 1")
        if getattr(args, "next_op_hint", None) != "rns-gemm":
            raise SystemExit("--hip-graph-replay requires --next-op-hint rns-gemm")
        if host_api_batch_size > 1 or getattr(args, "include_oneshot", False) or getattr(args, "oneshot_only", False):
            raise SystemExit("--hip-graph-replay cannot be combined with host batching or one-shot sweeps")
        if getattr(args, "native_to_rns_bridge", False) or getattr(args, "vector_to_rns_chain", False):
            raise SystemExit("--hip-graph-replay cannot be combined with bridge or vector-to-RNS chain sweeps")
        if getattr(args, "residue_channel_fusion", False):
            raise SystemExit("--hip-graph-replay cannot be combined with residue-channel fusion")
        if getattr(args, "bound_source", None) == "input-scan":
            raise SystemExit("--hip-graph-replay currently requires static-profile bounds")
        if any(semantics not in RNS_CHAIN_SEMANTICS for semantics in semantics_values):
            raise SystemExit("--hip-graph-replay is only valid for bounded or exact-wide RNS sweeps")
    if args.residue_chain_length < 1:
        raise SystemExit("--residue-chain-length must be positive")
    if getattr(args, "residue_chain_final_export", False) and args.residue_chain_length <= 1:
        raise SystemExit("--residue-chain-final-export requires --residue-chain-length > 1")
    if getattr(args, "residue_chain_final_export", False) and getattr(args, "next_op_hint", None) == "rns-gemm":
        raise SystemExit("--residue-chain-final-export cannot use --next-op-hint rns-gemm")
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
            if host_api_batch_size > 1:
                backends = [backend for backend in backends if backend in HOST_API_BATCH_BACKENDS]
            if getattr(args, "residue_channel_fusion", False):
                backends = [backend for backend in backends if backend == "hip-direct"]
            if getattr(args, "hip_graph_replay", False):
                backends = [backend for backend in backends if backend == "hip-direct"]
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
                                bool(getattr(args, "residue_chain_final_export", False)),
                                int(getattr(args, "output_ld_padding", 0) or 0),
                                int(getattr(args, "host_api_batch_size", 1) or 1),
                                bool(getattr(args, "hip_graph_replay", False)),
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
                                bool(getattr(args, "residue_chain_final_export", False)),
                                int(getattr(args, "output_ld_padding", 0) or 0),
                                int(getattr(args, "host_api_batch_size", 1) or 1),
                                bool(getattr(args, "hip_graph_replay", False)),
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
            getattr(args, "next_op_hint", None),
            getattr(args, "residue_channel_fusion", False),
            getattr(args, "modulus_set", "default") != "default",
            getattr(args, "tile_shape_variant", "default") != "default",
            getattr(args, "export_variant", "default") != "default",
            getattr(args, "reconstruction_variant", "default_garner") != "default_garner",
            int(getattr(args, "grouped_dispatch_tasks", 1) or 1) != 1,
            getattr(args, "hip_graph_replay", False),
            getattr(args, "residue_chain_final_export", False),
            getattr(args, "workload_proxy", "none") != "none",
            int(getattr(args, "output_ld_padding", 0) or 0) != 0,
            int(getattr(args, "residue_chain_length", 1) or 1) != 1,
            int(getattr(args, "host_api_batch_size", 1) or 1) != 1,
        ]
    ):
        raise SystemExit("--scenario cannot be combined with manual sweep shape/reuse/include flags")

    backend_benches = parse_backend_bench(args.bench_for)
    semantics_filter = {normalize_semantics(item) for item in (args.semantics or [])}
    commands: list[SweepCommand] = []
    for item in selected_scenario_items(args):
        if semantics_filter and item.semantics not in semantics_filter:
            continue
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
                    if (
                        not item.native_to_rns_bridge
                        and not item.vector_to_rns_chain
                        and not backend_allowed_for(item.semantics, item.case, backend)
                    ):
                        continue
                    if (
                        item.residue_chain_length > 1
                        and item.semantics in BOUNDED_SEMANTICS
                        and backend in {"auto", "hip-vector-alu-int64"}
                    ):
                        continue
                    bench = backend_benches.get(backend)
                    if bench is None and (item.native_to_rns_bridge or item.vector_to_rns_chain) and backend == "auto":
                        bench = backend_benches.get("hip-direct")
                    if bench is None:
                        bench = args.bench
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
                        item.residue_chain_final_export,
                        item.output_ld_padding,
                        item.host_api_batch_size,
                        item.hip_graph_replay,
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
        "--skip-existing",
        action="store_true",
        help="reuse valid existing capture outputs instead of rerunning them",
    )
    parser.add_argument(
        "--max-new-captures",
        type=int,
        help="limit the number of newly executed captures while still reviewing existing/skipped captures",
    )
    parser.add_argument(
        "--capture-timeout-seconds",
        type=float,
        help="optional wall-clock timeout for each benchmark capture command; timed-out captures are recorded as .failed.json",
    )
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
        "--next-op-hint",
        choices=["final-export", "rns-gemm", "native-gemm", "native-to-rns", "reuse-b"],
        help="benchmark-only next-operation hint to pass through to rns8-bench",
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
        help="bounded/exact-wide RNS GEMM chain length; values above 1 leave output residue-current unless final export is requested",
    )
    parser.add_argument(
        "--residue-chain-final-export",
        action="store_true",
        help="measure one final host export inside each residue-chain repeat instead of leaving output residue-current",
    )
    parser.add_argument(
        "--host-api-batch-size",
        type=int,
        default=1,
        help="benchmark-owned host API batch size for persistent resident matrix captures",
    )
    parser.add_argument(
        "--output-ld-padding",
        type=int,
        default=0,
        help="add padding columns to the benchmark host output leading dimension",
    )
    parser.add_argument(
        "--residue-channel-fusion",
        action="store_true",
        help="benchmark-only direct-HIP residue-channel fusion experiment passthrough",
    )
    parser.add_argument(
        "--modulus-set",
        default="default",
        help="benchmark-only modulus set metadata, default or experimental:NAME",
    )
    parser.add_argument(
        "--tile-shape-variant",
        default="default",
        help="benchmark-only tile-shape variant label",
    )
    parser.add_argument(
        "--export-variant",
        default="default",
        help="benchmark-only export/reconstruction evidence label",
    )
    parser.add_argument(
        "--reconstruction-variant",
        default="default_garner",
        help="benchmark-only reconstruction variant label",
    )
    parser.add_argument(
        "--grouped-dispatch-tasks",
        type=int,
        default=1,
        help="benchmark-only grouped-dispatch task count metadata",
    )
    parser.add_argument(
        "--hip-graph-replay",
        action="store_true",
        help="benchmark-only Direct-HIP HIP Graph replay experiment passthrough",
    )
    parser.add_argument(
        "--workload-proxy",
        default="none",
        help="benchmark-only workload proxy label",
    )
    parser.add_argument(
        "--resident-lifetime",
        action="store_true",
        help="benchmark-only resident matrix lifetime evidence metadata",
    )
    parser.add_argument(
        "--workspace-arena",
        action="store_true",
        help="benchmark-only workspace arena evidence metadata",
    )
    parser.add_argument(
        "--adaptive-grouped-scheduler",
        action="store_true",
        help="benchmark-only adaptive prefix grouped scheduler evidence metadata",
    )
    parser.add_argument(
        "--streaming-overlap",
        action="store_true",
        help="benchmark-only streaming pack/compute/export overlap evidence metadata",
    )
    parser.add_argument(
        "--release-gate",
        default="none",
        help="benchmark-only release gate label such as large-release-validation-4096-budgeted",
    )
    parser.add_argument(
        "--verification-amortization",
        default="none",
        help="benchmark-only verification amortization policy label",
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
    args = parser.parse_args()
    if args.max_new_captures is not None and args.max_new_captures < 0:
        parser.error("--max-new-captures must be non-negative")
    if args.capture_timeout_seconds is not None and args.capture_timeout_seconds <= 0:
        parser.error("--capture-timeout-seconds must be positive")
    if args.grouped_dispatch_tasks <= 0:
        parser.error("--grouped-dispatch-tasks must be positive")
    if args.modulus_set != "default" and not args.modulus_set.startswith("experimental:"):
        parser.error("--modulus-set must be default or experimental:NAME")
    if not args.release_gate:
        parser.error("--release-gate must not be empty")
    if not args.verification_amortization:
        parser.error("--verification-amortization must not be empty")
    return args


def main() -> int:
    args = parse_args()
    args.out_root = Path(args.out_root)
    args.out_root.mkdir(parents=True, exist_ok=True)

    capture_paths = list(args.capture)
    entries: list[SweepCommand] = []
    execution_stats: dict[str, int] | None = None
    if not args.review_only:
        if args.bench is None:
            raise SystemExit("--bench is required unless --review-only is used")
        entries = sweep_command_entries(args)
        execution_stats = execute_sweep_entries(entries, args, capture_paths)

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
    if execution_stats is not None:
        output["sweep_execution"] = execution_stats
    if scenario_paths is not None:
        output.update(scenario_paths)
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
