from __future__ import annotations

import json
import hashlib
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evidence_database_lib.isa import lookup_isa_resources

from .capture_metadata import (
    backend_id,
    backend_family_id,
    candidate_source_metadata,
    backend_requires_gpu_target,
    capture_backend_metadata,
    capture_bound_source,
    capture_compiler,
    capture_contract_key,
    capture_device,
    capture_execution_mode,
    capture_hip_toolchain,
    capture_pack_mode,
    capture_timing_metadata,
    group_source_metadata,
    median_phase,
    normalized_compiler_identity,
    normalized_identity_text,
    normalized_positive_int,
    normalized_target_id,
    selected_kernel,
)
from .config import (
    PHASES,
    RELEASE_MIN_REPEATS,
    RELEASE_MIN_WARMUPS,
    REVIEW_SCHEMA_VERSION,
    WRAP64_ROCWMMA_CANDIDATE_BACKEND,
)

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


REUSE_EVIDENCE_PROMOTION_SCOPES = {"explicit_reuse_contract_only", "reuse_contract_evidence_only"}
GRAPH_EVIDENCE_PROMOTION_SCOPES = {"hip_graph_replay_evidence_only"}
CORRECTNESS_ANCHOR_REFERENCE_BACKENDS = {"cpu-reference", "wrap64-byte-limb"}


def correctness_anchor_reference_capture(capture: dict[str, Any]) -> bool:
    if backend_family_id(backend_id(capture)) not in CORRECTNESS_ANCHOR_REFERENCE_BACKENDS:
        return False
    cpu_parallel = capture.get("cpu_parallel")
    if not isinstance(cpu_parallel, dict):
        return False
    return (
        cpu_parallel.get("correctness_anchor") is True
        or cpu_parallel.get("reference_mode") == "correctness-anchor"
    )


def autotune_promotable_scope(capture: dict[str, Any]) -> bool:
    scope = capture_scenario_promotion_scope(capture)
    return scope is None or scope == "release_review_candidate"


def reuse_evidence_group(items: list[dict[str, Any]]) -> bool:
    if not items:
        return False
    if not all(capture_pack_mode(item) != "per_repeat_repack" for item in items):
        return False
    scopes = {capture_scenario_promotion_scope(item) for item in items}
    return bool(scopes & REUSE_EVIDENCE_PROMOTION_SCOPES)


def required_baselines_for_group(semantics: Any, items: list[dict[str, Any]]) -> list[str]:
    if not any(autotune_promotable_scope(item) for item in items):
        return []
    if reuse_evidence_group(items):
        return []
    scopes = {capture_scenario_promotion_scope(item) for item in items}
    if scopes & GRAPH_EVIDENCE_PROMOTION_SCOPES:
        return []
    return required_baselines(semantics)


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


def capture_checksum(capture: dict[str, Any] | None) -> Any:
    if not capture:
        return None
    if capture.get("checksum_u64") is not None:
        return capture.get("checksum_u64")
    return capture.get("checksum")


def reference_checksum_for_group(by_backend: dict[str, dict[str, Any]]) -> tuple[str | None, Any]:
    for backend in ("cpu-reference", "wrap64-byte-limb", "hip-direct"):
        checksum = capture_checksum(by_backend.get(backend))
        if checksum is not None:
            return backend, checksum
    for backend, capture in sorted(by_backend.items()):
        checksum = capture_checksum(capture)
        if checksum is not None:
            return backend, checksum
    return None, None


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
    if accelerator:
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


def capture_scenario_promotion_scope(capture: dict[str, Any]) -> str | None:
    scenario = capture.get("scenario_metadata")
    if not isinstance(scenario, dict):
        return None
    eligibility = scenario.get("promotion_eligibility")
    if isinstance(eligibility, str) and eligibility:
        return eligibility
    metadata = scenario.get("metadata")
    if isinstance(metadata, dict):
        scope = metadata.get("promotion_scope")
        if isinstance(scope, str) and scope:
            return scope
    return None


def scenario_promotion_blockers(capture: dict[str, Any]) -> list[str]:
    scope = capture_scenario_promotion_scope(capture)
    if scope is None or scope == "release_review_candidate":
        return []
    return ["scenario_scope_not_autotune_promotable"]


def matrix_instruction_histogram(
    capture: dict[str, Any],
    isa_index: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    direct = capture.get("matrix_instruction_histogram")
    if isinstance(direct, dict):
        return direct
    for key in ("gpu_isa_report", "isa_report", "backend_isa_report", "compiled_isa"):
        report = capture.get(key)
        if not isinstance(report, dict):
            continue
        for nested_key in (
            "matrix_instruction_histogram",
            "matrix_instruction_counts",
            "mnemonic_histogram",
        ):
            histogram = report.get(nested_key)
            if isinstance(histogram, dict):
                return histogram
    if isa_index is not None:
        target = capture_device(capture).get("gcn_arch") or capture.get("configured_amdgpu_targets")
        resources = lookup_isa_resources(isa_index, backend_family_id(backend_id(capture)), target)
        histogram = resources.get("isa_matrix_instruction_histogram")
        if isinstance(histogram, dict):
            return histogram
    return {}


def review_route_candidate(candidate: dict[str, Any]) -> bool:
    scope = candidate.get("scenario_promotion_scope")
    blockers = {str(item) for item in candidate.get("promotion_blockers", [])}
    fatal = {
        "missing_required_baselines",
        "not_release_review",
        "missing_gpu_target_id",
        "gpu_target_mismatch",
        "missing_configured_gpu_target",
        "configured_gpu_target_mismatch",
        "missing_hip_toolchain_version",
        "hip_toolchain_version_mismatch",
        "missing_hip_runtime_version",
        "hip_runtime_version_mismatch",
        "missing_hip_driver_version",
        "hip_driver_version_mismatch",
        "missing_compiler_identity",
        "compiler_identity_mismatch",
        "missing_git_commit",
        "git_commit_mismatch",
        "missing_warmup_count",
        "warmup_count_mismatch",
        "missing_repeat_count",
        "repeat_count_mismatch",
        "duplicate_backend_capture",
        "internal_candidate_not_public_backend",
        "prepacked_reuse_not_autotune_promotable",
        "oneshot_api_capture_not_autotune_promotable",
        "host_api_batch_not_autotune_promotable",
        "hip_graph_replay_not_autotune_promotable",
        "missing_checksum",
        "missing_reference_checksum",
        "checksum_mismatch_vs_reference",
        "scenario_scope_not_autotune_promotable",
    }
    return (
        scope in {None, "release_review_candidate"}
        and candidate.get("release_review_capture") is True
        and candidate.get("checksum_matches_reference") is True
        and candidate.get("median_end_to_end_us") is not None
        and not (blockers & fatal)
    )


def fastest_route(candidates: list[dict[str, Any]], *, accelerator_only: bool) -> dict[str, Any] | None:
    route_candidates = [
        candidate
        for candidate in candidates
        if review_route_candidate(candidate)
        and (candidate.get("accelerator_backend") is True if accelerator_only else candidate.get("backend") != "cpu-reference")
    ]
    if not route_candidates:
        return None
    return min(route_candidates, key=lambda item: item["median_end_to_end_us"])


def review_captures(
    captures: list[dict[str, Any]],
    *,
    review_mode: str = "smoke",
    isa_index: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
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
        by_backend: dict[str, dict[str, Any]] = {}
        for item in items:
            by_backend.setdefault(backend_id(item), item)
        semantics = items[0].get("semantics")
        required = required_baselines_for_group(semantics, items)
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
        release_timing_by_backend = {
            backend: capture
            for backend, capture in by_backend.items()
            if not correctness_anchor_reference_capture(capture)
        }
        warmup_counts = {
            backend: normalized_positive_int(capture.get("warmups"))
            for backend, capture in release_timing_by_backend.items()
        }
        missing_warmup_counts = sorted(backend for backend, count in warmup_counts.items() if count is None)
        warmup_count_complete = not missing_warmup_counts
        warmup_count_values = {count for count in warmup_counts.values() if count}
        warmup_count_compatible = warmup_count_complete and len(warmup_count_values) <= 1
        repeat_counts = {
            backend: normalized_positive_int(capture.get("repeats"))
            for backend, capture in release_timing_by_backend.items()
        }
        missing_repeat_counts = sorted(backend for backend, count in repeat_counts.items() if count is None)
        repeat_count_complete = not missing_repeat_counts
        repeat_count_values = {count for count in repeat_counts.values() if count}
        repeat_count_compatible = repeat_count_complete and len(repeat_count_values) <= 1
        release_review_satisfied = review_mode == "release" and bool(release_timing_by_backend) and all(
            release_capture_satisfied(item) for item in release_timing_by_backend.values()
        )
        checksum_by_backend = {backend: capture_checksum(capture) for backend, capture in by_backend.items()}
        checksum_reference_backend, checksum_reference = reference_checksum_for_group(by_backend)
        missing_checksums = sorted(backend for backend, checksum in checksum_by_backend.items() if checksum is None)
        checksum_mismatches = sorted(
            backend
            for backend, checksum in checksum_by_backend.items()
            if checksum_reference is not None and checksum is not None and checksum != checksum_reference
        )
        checksum_consistent = checksum_reference is not None and not missing_checksums and not checksum_mismatches
        scenario_promotion_scopes = sorted(
            {
                scope
                for item in items
                if (scope := capture_scenario_promotion_scope(item)) is not None
            }
        )
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
            timed_cpu_capture = (
                cpu_capture
                if cpu_capture is not None and not correctness_anchor_reference_capture(cpu_capture)
                else None
            )
            cpu = median_phase(timed_cpu_capture, "end_to_end") if timed_cpu_capture else None
            direct = median_phase(direct_capture, "end_to_end") if direct_capture else None
            vector = median_phase(vector_capture, "end_to_end") if vector_capture else None
            if autotune_promotable_scope(item):
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
            else:
                blockers = []
            item_checksum = capture_checksum(item)
            if item_checksum is None:
                blockers.append("missing_checksum")
            elif checksum_reference is None:
                blockers.append("missing_reference_checksum")
            elif item_checksum != checksum_reference:
                blockers.append("checksum_mismatch_vs_reference")
            blockers.extend(scenario_promotion_blockers(item))
            promotable = not blockers
            candidate = {
                "backend": backend,
                "selected_kernel": selected_kernel(item),
                "capture": item.get("_path"),
                "source_metadata": candidate_source_metadata(item),
                "scenario_promotion_scope": capture_scenario_promotion_scope(item),
                "accelerator_backend": accelerator,
                "release_review_capture": release_capture_satisfied(item),
                "checksum": item_checksum,
                "checksum_reference_backend": checksum_reference_backend,
                "checksum_reference": checksum_reference,
                "checksum_matches_reference": (
                    item_checksum is not None
                    and checksum_reference is not None
                    and item_checksum == checksum_reference
                ),
                "median_end_to_end_us": end_to_end,
                "phase_diagnostics": phase_ratios(item, direct_capture, vector_capture),
                "speedup_vs_direct_hip": (direct / end_to_end) if direct and end_to_end else None,
                "speedup_vs_vector_alu": (vector / end_to_end) if vector and end_to_end else None,
                "matrix_instruction_histogram": matrix_instruction_histogram(item, isa_index),
                "promotable": promotable,
                "promotion_blockers": blockers,
                "promotion_reason": "beats_required_same_contract_gpu_baselines" if promotable else "blocked",
                "primary_loss_phase_vs_direct_hip": None if promotable else primary_loss_phase(item, direct_capture),
                "bottleneck": bottleneck_classification(item),
                "cache_write_status": "eligible_after_review" if promotable else "not_eligible",
            }
            candidates.append(candidate)

        fastest_production_route = fastest_route(candidates, accelerator_only=False)
        fastest_accelerator_route = fastest_route(candidates, accelerator_only=True)

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
                "checksum_reference_backend": checksum_reference_backend,
                "checksum_reference": checksum_reference,
                "checksum_by_backend": checksum_by_backend,
                "missing_checksums": missing_checksums,
                "checksum_mismatches": checksum_mismatches,
                "checksum_consistent": checksum_consistent,
                "scenario_promotion_scopes": scenario_promotion_scopes,
                "phase_medians_us": phase_medians,
                "fastest_promotable": fastest,
                "fastest_production_route": fastest_production_route,
                "fastest_accelerator_route": fastest_accelerator_route,
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
    export_variant = capture.get("export_variant") if isinstance(capture.get("export_variant"), dict) else {}
    reconstruction_variant = (
        capture.get("reconstruction_variant") if isinstance(capture.get("reconstruction_variant"), dict) else {}
    )
    export_variant_name = str(export_variant.get("name") or "default")
    reconstruction_variant_name = str(reconstruction_variant.get("name") or "default_garner")
    export_selector_key = export_variant.get("selector_key")
    export_selector_hash = (
        hashlib.sha256(export_selector_key.encode("utf-8")).hexdigest()[:16]
        if isinstance(export_selector_key, str) and export_selector_key
        else None
    )
    default_export_contract = export_variant_name == "default" and reconstruction_variant_name == "default_garner"
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
    key = metadata.get("autotune_key")
    if isinstance(key, str) and key and not default_export_contract and export_selector_hash:
        key = (
            f"{key};export_variant={export_variant_name};"
            f"reconstruction_variant={reconstruction_variant_name};"
            f"export_selector_hash={export_selector_hash}"
        )

    def median(phase: str) -> float:
        item = medians.get(phase) if isinstance(medians, dict) else None
        if isinstance(item, dict) and isinstance(item.get("median"), (int, float)):
            return float(item["median"])
        return 0.0

    return {
        "key": key,
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
        "export_variant": export_variant_name,
        "reconstruction_variant": reconstruction_variant_name,
        "export_selector_key": export_selector_key,
        "export_selector_hash": export_selector_hash,
        "export_selector_policy": export_variant.get("selector_policy"),
        "export_cache_visibility": export_variant.get("cache_visibility"),
        "export_stale_entry_reason": export_variant.get("stale_entry_reason"),
        "cache_scope": "runtime_exact_autotune" if default_export_contract else "selector_review_only_non_default",
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


def reviewed_release_status_for_target(target_id: str | None) -> str:
    target = str(target_id or "")
    if target == "gfx1100":
        return "reviewed_release_same_contract_fastest_windows_gfx1100"
    if target in {"gfx90a", "gfx942", "gfx950"}:
        return f"reviewed_release_same_contract_fastest_linux_{target}"
    if target.startswith("gfx"):
        return f"reviewed_release_same_contract_fastest_target_{target}"
    return "reviewed_release_unknown_target"


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
        device = capture.get("device") if isinstance(capture.get("device"), dict) else {}
        target_id = normalized_target_id(device.get("gcn_arch")) or "cpu"
        entry = cache_entry_from_capture(capture, reviewed_release_status_for_target(target_id))
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


