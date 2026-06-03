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
EXACT_WIDE_BACKENDS = ["cpu", "hip-direct", "hipblaslt", "ck", "rocwmma"]
FINITE_BACKENDS = ["cpu", "hip-direct", "hipblaslt", "ck", "rocwmma"]
WRAP64_BACKENDS = ["wrap64-byte-limb", "hip-direct"]
WRAP64_WMMA_CANDIDATE_BACKEND = "rocwmma-wrap64-candidate"
EXACT_WIDE_SEMANTICS = ["exact-wide-signed", "exact-wide-unsigned"]
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
DEFAULT_FINITE_RING_MODULI = [251, 255]
DEFAULT_FINITE_FIELD_MODULI = [251]


@dataclass(frozen=True)
class SweepCase:
    name: str
    m: int
    n: int
    k: int
    tile_m: int = 128
    tile_n: int = 128
    bound_mode: str = "global"
    require_adaptive: bool = False
    promotable: bool = True


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
    expected = 5 if adaptive else 3
    if len(parts) != expected:
        shape = "NAME:M,N,K,TILE_M,TILE_N" if adaptive else "NAME:M,N,K"
        raise SystemExit(f"case must use {shape}, got {value!r}")
    m = parse_int(parts[0], "m")
    n = parse_int(parts[1], "n")
    k = parse_int(parts[2], "k")
    tile_m = parse_int(parts[3], "tile_m") if adaptive else 128
    tile_n = parse_int(parts[4], "tile_n") if adaptive else 128
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
        require_adaptive=adaptive,
        promotable=promotable,
    )


def capture_contract_key(capture: dict[str, Any]) -> str:
    tile_bounds = capture.get("tile_bounds_u64")
    tile_hash = tile_bounds.get("hash_u64") if isinstance(tile_bounds, dict) else None
    wrap64_contract = capture.get("semantics") == "wrap_u64_mod_2_64"
    tile_m = "wrap64_semantic_contract" if wrap64_contract else capture.get("tile_m")
    tile_n = "wrap64_semantic_contract" if wrap64_contract else capture.get("tile_n")
    parts = [
        f"semantics={capture.get('semantics')}",
        f"finite_modulus={capture.get('finite_modulus')}",
        f"bound_kind={capture.get('bound_kind')}",
        f"bound_mode={capture.get('bound_mode')}",
        f"bound={capture.get('bound')}",
        f"m={capture.get('m')}",
        f"n={capture.get('n')}",
        f"k={capture.get('k')}",
        f"prefix={capture.get('prefix')}",
        f"layout={capture.get('layout')}",
        f"tile_m={tile_m}",
        f"tile_n={tile_n}",
        f"k_block={capture.get('k_block_size')}",
        f"seed={capture.get('seed')}",
        f"input_distribution={capture.get('input_distribution')}",
        f"reuse_packed_inputs={capture.get('reuse_packed_inputs') is True}",
        f"pack_mode={capture_pack_mode(capture)}",
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
    if capture.get("backend_requested") == WRAP64_WMMA_CANDIDATE_BACKEND:
        return WRAP64_WMMA_CANDIDATE_BACKEND
    return str(capture.get("backend_selected"))


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


def capture_backend_metadata(capture: dict[str, Any]) -> dict[str, Any]:
    metadata = capture.get("backend_metadata")
    return metadata if isinstance(metadata, dict) else {}


def capture_timing_metadata(capture: dict[str, Any]) -> dict[str, Any]:
    metadata = capture.get("timing_metadata")
    return metadata if isinstance(metadata, dict) else {}


def capture_pack_mode(capture: dict[str, Any]) -> str:
    timing = capture_timing_metadata(capture)
    mode = timing.get("pack_mode")
    if mode is None:
        mode = capture.get("pack_mode")
    if isinstance(mode, str):
        return mode
    return "prepacked_reuse" if capture.get("reuse_packed_inputs") is True else "per_repeat_repack"


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
        "hip_sdk_or_rocm_version": toolchain.get("hip_sdk_or_rocm_version"),
        "accelerator_library": metadata.get("accelerator_library"),
        "accelerator_version": metadata.get("accelerator_version"),
        "compiler": {"id": compiler.get("id"), "version": compiler.get("version")},
        "configured_amdgpu_targets": capture.get("configured_amdgpu_targets"),
        "git_commit": capture.get("git_commit"),
        "seed": capture.get("seed"),
        "warmups": capture.get("warmups"),
        "repeats": capture.get("repeats"),
        "layout": capture.get("layout"),
        "reuse_packed_inputs": capture.get("reuse_packed_inputs") is True,
        "pack_mode": capture_pack_mode(capture),
        "prepack_setup_us": capture.get("prepack_setup_us"),
        "prefix": capture.get("prefix"),
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
        "compilers": compilers,
        "git_commits": sorted({str(item.get("git_commit")) for item in items if item.get("git_commit")}),
        "seeds": sorted({int(item.get("seed")) for item in items if isinstance(item.get("seed"), int)}),
        "warmups": sorted({int(item.get("warmups")) for item in items if isinstance(item.get("warmups"), int)}),
        "repeats": sorted({int(item.get("repeats")) for item in items if isinstance(item.get("repeats"), int)}),
        "reuse_packed_inputs": sorted({bool(item.get("reuse_packed_inputs") is True) for item in items}),
        "pack_modes": sorted({capture_pack_mode(item) for item in items}),
        "prepack_reuse_operands": sorted({"/".join(capture_prepack_reuse_operands(item)) or "none" for item in items}),
        "event_sources": sorted(
            {
                str(capture_timing_metadata(item).get("gpu_event_timing_source") or "unavailable")
                for item in items
            }
        ),
    }


def cli_backend(backend: str) -> str:
    return "rocwmma" if backend == "wmma" else backend


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
    gpu_target_compatible: bool,
    accelerator: bool,
    internal_candidate: bool,
    prepacked_reuse: bool,
    end_to_end: float | None,
    direct: float | None,
    vector: float | None,
) -> list[str]:
    blockers: list[str] = []
    if missing:
        blockers.append("missing_required_baselines")
    if not release_review_satisfied:
        blockers.append("not_release_review")
    if not gpu_target_compatible:
        blockers.append("gpu_target_mismatch")
    if not accelerator:
        blockers.append("not_accelerator_backend")
    if internal_candidate:
        blockers.append("internal_candidate_not_public_backend")
    if prepacked_reuse:
        blockers.append("prepacked_reuse_not_autotune_promotable")
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


def review_captures(captures: list[dict[str, Any]], *, review_mode: str = "smoke") -> dict[str, Any]:
    if review_mode not in {"smoke", "release"}:
        raise ValueError(f"unsupported review mode: {review_mode}")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for capture in captures:
        grouped[capture_contract_key(capture)].append(capture)

    groups = []
    promotable_entries = []
    for key, items in sorted(grouped.items()):
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
            if backend not in {"cpu-reference", "wrap64-byte-limb"}
        }
        gpu_target_values = {value for value in gpu_targets.values() if value}
        gpu_target_compatible = len(gpu_target_values) <= 1
        release_review_satisfied = review_mode == "release" and all(release_capture_satisfied(item) for item in items)
        candidates = []
        for item in items:
            backend = backend_id(item)
            metadata = capture_backend_metadata(item)
            accelerator = metadata.get("accelerator_backend") is True
            internal_candidate = backend == WRAP64_WMMA_CANDIDATE_BACKEND
            end_to_end = median_phase(item, "end_to_end")
            direct = median_phase(direct_capture, "end_to_end") if direct_capture else None
            vector = median_phase(vector_capture, "end_to_end") if vector_capture else None
            blockers = promotion_blockers(
                missing=missing,
                semantics=semantics,
                release_review_satisfied=release_review_satisfied,
                gpu_target_compatible=gpu_target_compatible,
                accelerator=accelerator,
                internal_candidate=internal_candidate,
                prepacked_reuse=capture_pack_mode(item) != "per_repeat_repack",
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
                "gpu_target_compatible": gpu_target_compatible,
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
    prefix_schedule_hash = (
        f"tile_rows={schedule.get('tile_rows')};tile_cols={schedule.get('tile_cols')};"
        f"groups={schedule.get('prefix_group_count')};"
        f"adaptive_prefix={int(bool(schedule.get('adaptive_prefix_active')))};"
        f"adaptive_skip={int(bool(schedule.get('adaptive_skip_active')))};"
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
    if args.adaptive_case:
        return [parse_case(value, adaptive=True) for value in args.adaptive_case]
    if args.include_default_adaptive:
        return [parse_case(value, adaptive=True) for value in DEFAULT_ADAPTIVE_CASES]
    return []


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
    if args.include_wrap64_wmma_candidate:
        backends.append(WRAP64_WMMA_CANDIDATE_BACKEND)
    return backends


def finite_moduli_for(semantics: str, args: argparse.Namespace) -> list[int | None]:
    if semantics == "finite-u8-ring":
        return args.modulus or DEFAULT_FINITE_RING_MODULI
    if semantics == "finite-u8-field":
        return args.modulus or DEFAULT_FINITE_FIELD_MODULI
    return [None]


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
        return backend in WRAP64_BACKENDS or backend == WRAP64_WMMA_CANDIDATE_BACKEND
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
) -> str:
    parts = [semantics, case.name, f"{case.m}x{case.n}x{case.k}"]
    if modulus is not None:
        parts.append(f"mod{modulus}")
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
    args: argparse.Namespace,
) -> list[str]:
    tile_m = 16 if semantics == "wrap-u64" and backend == WRAP64_WMMA_CANDIDATE_BACKEND else case.tile_m
    tile_n = 16 if semantics == "wrap-u64" and backend == WRAP64_WMMA_CANDIDATE_BACKEND else case.tile_n
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
    if case.require_adaptive:
        command.append("--require-adaptive-execution")
    if modulus is not None:
        command.extend(["--modulus", str(modulus)])
    pack_mode = requested_pack_mode(args)
    if pack_mode == "prepacked_reuse":
        command.append("--reuse-packed-inputs")
    elif pack_mode == "prepacked_reuse_a":
        command.append("--reuse-packed-a")
    elif pack_mode == "prepacked_reuse_b":
        command.append("--reuse-packed-b")
    return command


def sweep_commands(args: argparse.Namespace) -> list[tuple[str, list[str], Path]]:
    backend_benches = parse_backend_bench(args.bench_for)
    commands: list[tuple[str, list[str], Path]] = []
    semantics_values = [normalize_semantics(item) for item in (args.semantics or ["bounded-i64", "bounded-u64"])]
    cases = [*([] if args.adaptive_only else default_cases(args)), *adaptive_cases(args)]
    if args.adaptive_only and not cases:
        raise SystemExit("--adaptive-only requires --adaptive-case or --include-default-adaptive")
    if (args.include_wrap64 or args.include_wrap64_wmma_candidate) and "wrap-u64" not in semantics_values:
        semantics_values.append("wrap-u64")
    if args.include_exact_wide:
        for exact_semantics in EXACT_WIDE_SEMANTICS:
            if exact_semantics not in semantics_values:
                semantics_values.append(exact_semantics)
    for semantics in semantics_values:
        if semantics == "wrap-u64":
            if args.adaptive_only:
                continue
            active_cases = wrap64_cases(args)
        else:
            active_cases = cases
        for case in active_cases:
            backends = args.backends or (
                wrap64_backends_for(args) if semantics == "wrap-u64" else default_backends_for(semantics, case)
            )
            for modulus in finite_moduli_for(semantics, args):
                for backend in backends:
                    if not backend_allowed_for(semantics, case, backend):
                        continue
                    bench = backend_benches.get(backend, args.bench)
                    if bench is None:
                        raise SystemExit(f"no benchmark executable configured for backend {backend}")
                    name = capture_name(semantics, case, backend, modulus, requested_pack_mode(args))
                    command = command_for(bench, backend, semantics, case, modulus, args)
                    commands.append((name, command, args.out_root / name))
    return commands


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
        lines.append(f"- missing_required_baselines: `{','.join(missing) if missing else 'none'}`")
        lines.append(f"- release_review_satisfied: `{group.get('release_review_satisfied')}`")
        fastest = group.get("fastest_promotable")
        if fastest:
            lines.append(f"- fastest_promotable: `{fastest['backend']}/{fastest['selected_kernel']}`")
            lines.append(f"- winner_rationale: `{fastest.get('promotion_reason')}`")
        else:
            lines.append("- fastest_promotable: `none`")
        lines.append("")
        lines.append("| backend | kernel | target | e2e median us | promotable | cache | blockers | primary loss phase |")
        lines.append("|---|---|---|---:|---|---|---|---|")
        for candidate in group.get("candidates", []):
            blockers = ",".join(candidate.get("promotion_blockers") or [])
            source = candidate.get("source_metadata") if isinstance(candidate.get("source_metadata"), dict) else {}
            lines.append(
                "| {backend} | {kernel} | {target} | {median} | {promotable} | {cache} | {blockers} | {loss} |".format(
                    backend=candidate.get("backend"),
                    kernel=candidate.get("selected_kernel"),
                    target=source.get("target_id"),
                    median=candidate.get("median_end_to_end_us"),
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
    parser.add_argument("--backend", dest="backends", action="append", help="backend to sweep; repeatable")
    parser.add_argument("--semantics", action="append", help="benchmark semantics to sweep; repeatable")
    parser.add_argument("--case", action="append", help="global case NAME:M,N,K; repeatable")
    parser.add_argument("--adaptive-case", action="append", help="adaptive case NAME:M,N,K,TILE_M,TILE_N; repeatable")
    parser.add_argument("--shape", dest="shapes", action="append", help="legacy square shape to sweep; repeatable")
    parser.add_argument("--modulus", type=int, action="append", help="finite-u8 modulus; repeatable")
    parser.add_argument("--include-default-adaptive", action="store_true", help="include default adaptive bounded cases")
    parser.add_argument("--adaptive-only", action="store_true", help="run only adaptive cases, skipping global cases")
    parser.add_argument("--include-wrap64", action="store_true", help="include wrap64 CPU/direct-HIP captures")
    parser.add_argument(
        "--include-wrap64-wmma-candidate",
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
    if not args.review_only:
        if args.bench is None:
            raise SystemExit("--bench is required unless --review-only is used")
        for _name, command, output in sweep_commands(args):
            if run_command(command, output):
                capture_paths.append(output)

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
    print(
        json.dumps(
            {
                "review_report": str(report_path),
                "markdown_report": str(markdown_path),
                "captures": len(captures),
                "promoted_cache_entries": promoted,
                "autotune_cache": str(cache_path) if args.write_autotune_cache else None,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
