#!/usr/bin/env python3
"""Print compact failure and mismatch summaries for benchmark sweep outputs."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from evidence_database_lib.isa import load_isa_index, lookup_isa_resources

from benchmark_sweep_lib.capture_metadata import backend_family_id, backend_id, capture_contract_key
from benchmark_sweep_lib.review import capture_checksum


REFERENCE_BACKEND_FAMILIES = {"cpu-reference", "wrap64-byte-limb", "hip-direct"}
NON_ACTIONABLE_BLOCKERS = {"not_accelerator_backend", "scenario_scope_not_autotune_promotable"}
PROMOTABLE_SCOPES = {None, "release_review_candidate"}
DEFAULT_MAX_ROUTE_ROWS = 40
DEFAULT_MAX_DETAIL_ROWS = 80


def _latest_cdna_out(root: Path) -> Path:
    candidates = sorted(root.glob("cdna-*mi300x-*"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise SystemExit(f"{root}: no cdna-*mi300x-* sweep directories found")
    return candidates[-1]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _scenario_capture_paths(out: Path) -> list[Path]:
    return sorted(
        path
        for path in out.rglob("rank-scenarios/*/scenarios/**/*.json")
        if not path.name.endswith(".failed.json")
    )


def _reference_checksum(rows: list[tuple[str, Any, Path]]) -> tuple[str | None, Any]:
    for backend, checksum, _path in rows:
        if backend_family_id(backend) in REFERENCE_BACKEND_FAMILIES and checksum is not None:
            return backend, checksum
    for backend, checksum, _path in rows:
        if checksum is not None:
            return backend, checksum
    return None, None


def _relative_capture(out: Path, value: Any) -> str:
    if not value:
        return "unknown"
    path = Path(str(value))
    try:
        return str(path.relative_to(out))
    except ValueError:
        return str(path)


def _actionable_blockers(candidate: dict[str, Any]) -> list[str]:
    scope = candidate.get("scenario_promotion_scope")
    if scope not in PROMOTABLE_SCOPES:
        return []
    if candidate.get("accelerator_backend") is not True:
        return []
    blockers = candidate.get("promotion_blockers")
    if not isinstance(blockers, list):
        return []
    return [str(item) for item in blockers if str(item) not in NON_ACTIONABLE_BLOCKERS]


def _shape_text(group: dict[str, Any]) -> str:
    shape = group.get("shape")
    if isinstance(shape, dict):
        return f"{shape.get('m')}x{shape.get('n')}x{shape.get('k')}"
    return "unknown"


def _scenario_families(group: dict[str, Any]) -> list[str]:
    families = group.get("scenario_families")
    if isinstance(families, list):
        result = [str(item) for item in families if isinstance(item, str) and item]
        if result:
            return result
    return ["unknown"]


def _candidate_histogram(
    candidate: dict[str, Any],
    isa_index: dict[str, list[dict[str, Any]]] | None,
) -> dict[str, Any]:
    histogram = candidate.get("matrix_instruction_histogram")
    if isinstance(histogram, dict) and histogram:
        return histogram
    if isa_index is None:
        return {}
    source = candidate.get("source_metadata")
    target = source.get("target_id") if isinstance(source, dict) else None
    resources = lookup_isa_resources(isa_index, candidate.get("backend"), target)
    fallback = resources.get("isa_matrix_instruction_histogram")
    return fallback if isinstance(fallback, dict) else {}


def _histogram_text(
    candidate: dict[str, Any],
    isa_index: dict[str, list[dict[str, Any]]] | None = None,
) -> str:
    histogram = _candidate_histogram(candidate, isa_index)
    if not histogram:
        return "none"
    items = sorted((str(key), value) for key, value in histogram.items() if isinstance(value, int) and value > 0)
    if not items:
        return "none"
    return ",".join(f"{key}:{value}" for key, value in items[:8])


def _pack_split_text(candidate: dict[str, Any]) -> str:
    medians = candidate.get("phase_medians_us")
    if not isinstance(medians, dict):
        return "none"
    values: list[str] = []
    for phase in ("pack_a", "pack_b"):
        value = medians.get(phase)
        if isinstance(value, (int, float)):
            values.append(f"{phase}:{float(value)}")
    return ",".join(values) if values else "none"


def _matrix_metadata_text(candidate: dict[str, Any]) -> str:
    values = [
        candidate.get("matrix_instruction_family"),
        candidate.get("matrix_instruction_shape"),
        candidate.get("matrix_instruction_dtype"),
        candidate.get("matrix_instruction_sparsity"),
    ]
    extra_values = [
        candidate.get("matrix_operand_signedness"),
        candidate.get("matrix_a_value_contract"),
        candidate.get("matrix_b_value_contract"),
        candidate.get("matrix_sparse_contract"),
        candidate.get("matrix_sparse_dense_operand"),
        candidate.get("matrix_sparse_a_compression_index_layout"),
    ]
    if not any(isinstance(value, str) and value for value in values + extra_values):
        return "none"
    base = "/".join(str(value or "unknown") for value in values)
    extras: list[str] = []
    labels = (
        ("operand", candidate.get("matrix_operand_signedness")),
        ("a", candidate.get("matrix_a_value_contract")),
        ("b", candidate.get("matrix_b_value_contract")),
        ("sparse", candidate.get("matrix_sparse_contract")),
        ("dense", candidate.get("matrix_sparse_dense_operand")),
        ("index", candidate.get("matrix_sparse_a_compression_index_layout")),
    )
    for label, value in labels:
        if isinstance(value, str) and value:
            extras.append(f"{label}={value}")
    policy = candidate.get("matrix_rdna_integer_modifier_policy")
    if isinstance(policy, dict) and policy:
        extras.append("rdna_neg_policy=present")
    return base + (" " + " ".join(extras) if extras else "")


def _phase_ratio_text(candidate: dict[str, Any]) -> str:
    diagnostics = candidate.get("phase_diagnostics")
    if not isinstance(diagnostics, dict):
        return "none"
    parts: list[str] = []
    slowest = diagnostics.get("slowest_phase_vs_direct_hip")
    slowest_ratio = diagnostics.get("slowest_phase_candidate_over_direct")
    if isinstance(slowest, str) and slowest:
        parts.append(f"slowest={slowest}:{slowest_ratio}")
    speedups = diagnostics.get("phase_speedups_vs_direct_hip")
    if isinstance(speedups, dict):
        ordered: list[str] = []
        for phase in ("pack", "rns_gemm", "crt_export", "end_to_end"):
            value = speedups.get(phase)
            if isinstance(value, (int, float)):
                ordered.append(f"{phase}:{float(value)}")
        if ordered:
            parts.append("speedups=" + ",".join(ordered))
    return " ".join(parts) if parts else "none"


def _route_line(
    out: Path,
    label: str,
    group: dict[str, Any],
    candidate: dict[str, Any],
    isa_index: dict[str, list[dict[str, Any]]] | None = None,
) -> str:
    bottleneck = candidate.get("bottleneck")
    bottleneck_text = "unknown"
    if isinstance(bottleneck, dict):
        bottleneck_text = f"{bottleneck.get('class')}:{bottleneck.get('phase')}"
    return (
        "  "
        f"{label} "
        f"backend={candidate.get('backend')} "
        f"semantics={group.get('semantics')} "
        f"shape={_shape_text(group)} "
        f"kernel={candidate.get('selected_kernel')} "
        f"e2e={candidate.get('median_end_to_end_us')} "
        f"vs_direct={candidate.get('speedup_vs_direct_hip')} "
        f"primary_loss={candidate.get('primary_loss_phase_vs_direct_hip')} "
        f"bottleneck={bottleneck_text} "
        f"phase_ratios={_phase_ratio_text(candidate)} "
        f"pack_split={_pack_split_text(candidate)} "
        f"matrix_meta={_matrix_metadata_text(candidate)} "
        f"matrix_isa={_histogram_text(candidate, isa_index)} "
        f"capture={_relative_capture(out, candidate.get('capture'))}"
    )


def _group_backends(group: dict[str, Any]) -> list[str]:
    candidates = group.get("candidates")
    if not isinstance(candidates, list):
        return []
    return sorted(str(candidate.get("backend")) for candidate in candidates if isinstance(candidate, dict))


def _blocker_text(values: Any) -> str:
    return ",".join(str(item) for item in values) if isinstance(values, list) and values else "none"


def _review_detail_text(candidate: dict[str, Any]) -> str:
    details: list[str] = []
    phase_ratios = _phase_ratio_text(candidate)
    if phase_ratios != "none":
        details.append(f"phase_ratios={phase_ratios}")
    pack_split = _pack_split_text(candidate)
    if pack_split != "none":
        details.append(f"pack_split={pack_split}")
    prepack = candidate.get("prepacked_reuse_review")
    if isinstance(prepack, dict):
        details.extend(
            [
                f"reuse_setup_e2e={prepack.get('setup_inclusive_median_end_to_end_us')}",
                f"reuse_steady_e2e={prepack.get('candidate_median_end_to_end_us')}",
                f"prepack_setup={prepack.get('prepack_setup_us')}",
                f"reuse_repeats={prepack.get('declared_repeat_count')}",
                f"setup_amortized={prepack.get('setup_amortized_us')}",
                f"setup_share={prepack.get('setup_share_of_setup_inclusive')}",
                f"same_backend={prepack.get('same_backend_nonreuse_backend')}",
                f"same_e2e={prepack.get('same_backend_nonreuse_median_end_to_end_us')}",
                f"best_nonreuse={prepack.get('best_nonreuse_backend')}",
                f"best_e2e={prepack.get('best_nonreuse_median_end_to_end_us')}",
                f"reuse_vs_same={prepack.get('speedup_vs_same_backend_setup_inclusive')}",
                f"reuse_vs_best={prepack.get('speedup_vs_best_nonreuse_setup_inclusive')}",
            ]
        )
    graph = candidate.get("hip_graph_replay_review")
    if isinstance(graph, dict):
        declared_meets_break_even = _declared_repeats_meet_break_even(graph)
        details.extend(
            [
                f"graph_setup_e2e={graph.get('setup_inclusive_median_end_to_end_us')}",
                f"graph_steady_e2e={graph.get('candidate_median_end_to_end_us')}",
                f"graph_capture={graph.get('graph_capture_us')}",
                f"graph_instantiate={graph.get('graph_instantiate_us')}",
                f"graph_total_setup={graph.get('graph_total_setup_us')}",
                f"graph_setup_amortized={graph.get('graph_setup_amortized_us')}",
                f"graph_setup_share={graph.get('graph_setup_share_of_setup_inclusive')}",
                f"graph_baseline={graph.get('baseline_backend')}",
                f"baseline_steady_e2e={graph.get('baseline_median_end_to_end_us')}",
                f"baseline_total_setup={graph.get('baseline_total_setup_us')}",
                f"baseline_e2e={graph.get('baseline_setup_inclusive_median_end_to_end_us')}",
                f"baseline_setup_share={graph.get('baseline_setup_share_of_setup_inclusive')}",
                f"graph_break_even_repeats={graph.get('break_even_repeat_count')}",
                f"graph_declared_repeats={graph.get('declared_repeat_count')}",
                f"graph_declared_meets_break_even={declared_meets_break_even}",
                f"graph_vs_baseline={graph.get('speedup_vs_non_graph_setup_inclusive')}",
            ]
        )
    return " ".join(details) if details else "none"


SETUP_BLOCKERS = {
    "reuse_not_faster_than_same_backend_setup_inclusive",
    "reuse_not_faster_than_best_nonreuse_setup_inclusive",
    "graph_not_faster_than_non_graph_setup_inclusive",
    "missing_graph_setup_inclusive_timing",
    "missing_prepack_setup_inclusive_timing",
}


def _positive_number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
        return float(value)
    return None


def _candidate_underperformance_score(candidate: dict[str, Any]) -> float:
    score = 0.0
    for key in ("speedup_vs_direct_hip", "speedup_vs_vector_alu"):
        speedup = _positive_number(candidate.get(key))
        if speedup is not None and speedup < 1.0:
            score = max(score, 1.0 / speedup)

    prepack = candidate.get("prepacked_reuse_review")
    if isinstance(prepack, dict):
        for key in ("speedup_vs_same_backend_setup_inclusive", "speedup_vs_best_nonreuse_setup_inclusive"):
            speedup = _positive_number(prepack.get(key))
            if speedup is not None and speedup < 1.0:
                score = max(score, 1.0 / speedup)

    graph = candidate.get("hip_graph_replay_review")
    if isinstance(graph, dict):
        speedup = _positive_number(graph.get("speedup_vs_non_graph_setup_inclusive"))
        if speedup is not None and speedup < 1.0:
            score = max(score, 1.0 / speedup)

    diagnostics = candidate.get("phase_diagnostics")
    if isinstance(diagnostics, dict):
        ratio = _positive_number(diagnostics.get("slowest_phase_candidate_over_direct"))
        if ratio is not None and ratio > 1.0:
            score = max(score, ratio)
    return score


def _candidate_setup_blocked(candidate: dict[str, Any]) -> bool:
    blockers = _actionable_blockers(candidate)
    return any(blocker in SETUP_BLOCKERS for blocker in blockers)


def _candidate_sort_key(row: tuple[str, str, dict[str, Any], dict[str, Any]]) -> tuple[int, float, str, str]:
    _report_path, _contract_key, group, candidate = row
    setup_rank = 1 if _candidate_setup_blocked(candidate) else 0
    phase = str(candidate.get("primary_loss_phase_vs_direct_hip") or "")
    backend = str(candidate.get("backend") or "")
    semantics = str(group.get("semantics") or "")
    return (setup_rank, _candidate_underperformance_score(candidate), phase, f"{semantics}:{backend}")


def _top_actionable_line(
    out: Path,
    report_path: str,
    contract_key: str,
    group: dict[str, Any],
    candidate: dict[str, Any],
) -> str:
    blockers = ",".join(_actionable_blockers(candidate))
    bottleneck = candidate.get("bottleneck")
    bottleneck_text = "unknown"
    if isinstance(bottleneck, dict):
        bottleneck_text = f"{bottleneck.get('class')}:{bottleneck.get('phase')}"
    return (
        "  "
        f"review={report_path} "
        f"backend={candidate.get('backend')} "
        f"semantics={group.get('semantics')} "
        f"shape={_shape_text(group)} "
        f"kernel={candidate.get('selected_kernel')} "
        f"score={_candidate_underperformance_score(candidate)} "
        f"primary_loss={candidate.get('primary_loss_phase_vs_direct_hip')} "
        f"bottleneck={bottleneck_text} "
        f"e2e={candidate.get('median_end_to_end_us')} "
        f"vs_direct={candidate.get('speedup_vs_direct_hip')} "
        f"vs_vector={candidate.get('speedup_vs_vector_alu')} "
        f"phase_ratios={_phase_ratio_text(candidate)} "
        f"pack_split={_pack_split_text(candidate)} "
        f"blockers={blockers} "
        f"capture={_relative_capture(out, candidate.get('capture'))} "
        f"contract={contract_key}"
    )


def _declared_repeats_meet_break_even(graph: dict[str, Any]) -> Any:
    explicit = graph.get("declared_repeats_meet_break_even")
    if explicit is not None:
        return explicit
    declared = graph.get("declared_repeat_count")
    break_even = graph.get("break_even_repeat_count")
    if (
        isinstance(declared, int)
        and not isinstance(declared, bool)
        and isinstance(break_even, int)
        and not isinstance(break_even, bool)
    ):
        return declared >= break_even
    return None


def _prepack_reuse_line(
    out: Path,
    report_path: str,
    group: dict[str, Any],
    candidate: dict[str, Any],
    prepack: dict[str, Any],
) -> str:
    blockers = _blocker_text(prepack.get("blockers"))
    runtime_cache = candidate.get("runtime_prepack_cache")
    cache_text = "cache=none"
    if isinstance(runtime_cache, dict):
        cache_text = (
            f"cache_production={runtime_cache.get('production_prepack_cache_available')} "
            f"cache_scope={runtime_cache.get('cache_scope')} "
            f"cache_source_version={runtime_cache.get('source_version')} "
            f"cache_hash={runtime_cache.get('cache_key_hash')} "
            f"cache_device_bytes={runtime_cache.get('device_bytes')} "
            f"cache_operand_bytes={runtime_cache.get('operand_pack_bytes')}"
        )
    return (
        "  "
        f"review={report_path} "
        f"backend={candidate.get('backend')} "
        f"semantics={group.get('semantics')} "
        f"shape={_shape_text(group)} "
        f"kernel={candidate.get('selected_kernel')} "
        f"setup_e2e={prepack.get('setup_inclusive_median_end_to_end_us')} "
        f"steady_e2e={prepack.get('candidate_median_end_to_end_us')} "
        f"prepack_setup={prepack.get('prepack_setup_us')} "
        f"declared_repeats={prepack.get('declared_repeat_count')} "
        f"setup_amortized={prepack.get('setup_amortized_us')} "
        f"setup_share={prepack.get('setup_share_of_setup_inclusive')} "
        f"same_backend={prepack.get('same_backend_nonreuse_backend')} "
        f"same_e2e={prepack.get('same_backend_nonreuse_median_end_to_end_us')} "
        f"best_nonreuse={prepack.get('best_nonreuse_backend')} "
        f"best_e2e={prepack.get('best_nonreuse_median_end_to_end_us')} "
        f"reuse_vs_same={prepack.get('speedup_vs_same_backend_setup_inclusive')} "
        f"reuse_vs_best={prepack.get('speedup_vs_best_nonreuse_setup_inclusive')} "
        f"{cache_text} "
        f"blockers={blockers} "
        f"capture={_relative_capture(out, candidate.get('capture'))}"
    )


def _graph_replay_line(
    out: Path,
    report_path: str,
    group: dict[str, Any],
    candidate: dict[str, Any],
    graph: dict[str, Any],
) -> str:
    blockers = _blocker_text(graph.get("blockers"))
    declared_meets_break_even = _declared_repeats_meet_break_even(graph)
    return (
        "  "
        f"review={report_path} "
        f"backend={candidate.get('backend')} "
        f"semantics={group.get('semantics')} "
        f"shape={_shape_text(group)} "
        f"kernel={candidate.get('selected_kernel')} "
        f"setup_e2e={graph.get('setup_inclusive_median_end_to_end_us')} "
        f"steady_e2e={graph.get('candidate_median_end_to_end_us')} "
        f"graph_setup={graph.get('graph_total_setup_us')} "
        f"graph_setup_amortized={graph.get('graph_setup_amortized_us')} "
        f"graph_setup_share={graph.get('graph_setup_share_of_setup_inclusive')} "
        f"capture_us={graph.get('graph_capture_us')} "
        f"instantiate_us={graph.get('graph_instantiate_us')} "
        f"baseline={graph.get('baseline_backend')} "
        f"baseline_steady_e2e={graph.get('baseline_median_end_to_end_us')} "
        f"baseline_setup={graph.get('baseline_total_setup_us')} "
        f"baseline_e2e={graph.get('baseline_setup_inclusive_median_end_to_end_us')} "
        f"baseline_setup_share={graph.get('baseline_setup_share_of_setup_inclusive')} "
        f"break_even_repeats={graph.get('break_even_repeat_count')} "
        f"declared_repeats={graph.get('declared_repeat_count')} "
        f"declared_meets_break_even={declared_meets_break_even} "
        f"speedup_vs_baseline={graph.get('speedup_vs_non_graph_setup_inclusive')} "
        f"blockers={blockers} "
        f"capture={_relative_capture(out, candidate.get('capture'))}"
    )


def _export_route_active(candidate: dict[str, Any]) -> bool:
    if candidate.get("primary_loss_phase_vs_direct_hip") == "crt_export":
        return True
    export = candidate.get("export_variant")
    if isinstance(export, dict) and export.get("name") not in (None, "default"):
        return True
    reconstruction = candidate.get("reconstruction_variant")
    if isinstance(reconstruction, dict) and reconstruction.get("name") not in (None, "default_garner"):
        return True
    return False


def _pack_diagnostic_active(candidate: dict[str, Any]) -> bool:
    diagnostics = candidate.get("pack_diagnostics")
    if not isinstance(diagnostics, dict):
        return False
    share = diagnostics.get("pack_share_of_end_to_end")
    if isinstance(share, (int, float)) and share > 0:
        return True
    return diagnostics.get("split_available") is True


def _pack_diagnostic_line(
    out: Path,
    report_path: str,
    group: dict[str, Any],
    candidate: dict[str, Any],
) -> str:
    diagnostics = candidate.get("pack_diagnostics") if isinstance(candidate.get("pack_diagnostics"), dict) else {}
    return (
        "  "
        f"review={report_path} "
        f"backend={candidate.get('backend')} "
        f"semantics={group.get('semantics')} "
        f"shape={_shape_text(group)} "
        f"shape_family={group.get('shape_family') or 'unknown'} "
        f"kernel={candidate.get('selected_kernel')} "
        f"pack={diagnostics.get('pack_median_us')} "
        f"pack_a={diagnostics.get('pack_a_median_us')} "
        f"pack_b={diagnostics.get('pack_b_median_us')} "
        f"pack_share={diagnostics.get('pack_share_of_end_to_end')} "
        f"split={diagnostics.get('split_available')} "
        f"dominant={diagnostics.get('dominant_operand')} "
        f"pack_mode={diagnostics.get('pack_mode')} "
        f"pack_layout={diagnostics.get('pack_layout')} "
        f"source_versioned={diagnostics.get('source_versioned_inputs')} "
        f"same_version_elision={diagnostics.get('same_source_version_pack_elision_available')} "
        f"capture={_relative_capture(out, candidate.get('capture'))}"
    )


def _native_handoff_active(candidate: dict[str, Any]) -> bool:
    diagnostics = candidate.get("native_to_rns_handoff_diagnostics")
    return isinstance(diagnostics, dict) and bool(diagnostics.get("execution_mode"))


def _native_handoff_line(
    out: Path,
    report_path: str,
    group: dict[str, Any],
    candidate: dict[str, Any],
) -> str:
    diagnostics = (
        candidate.get("native_to_rns_handoff_diagnostics")
        if isinstance(candidate.get("native_to_rns_handoff_diagnostics"), dict)
        else {}
    )
    return (
        "  "
        f"review={report_path} "
        f"backend={candidate.get('backend')} "
        f"semantics={group.get('semantics')} "
        f"shape={_shape_text(group)} "
        f"kernel={candidate.get('selected_kernel')} "
        f"mode={diagnostics.get('execution_mode')} "
        f"control={diagnostics.get('control_mode')} "
        f"producer={diagnostics.get('producer_backend')} "
        f"consumer={diagnostics.get('consumer_backend')} "
        f"consumer_k={diagnostics.get('consumer_k')} "
        f"conversion_label={diagnostics.get('conversion_event_label')} "
        f"conversion={diagnostics.get('conversion_median_us')} "
        f"host_repack={diagnostics.get('host_repack_median_us')} "
        f"vector_output_d2h={diagnostics.get('vector_output_d2h_median_us')} "
        f"consumer_gemm={diagnostics.get('consumer_gemm_median_us')} "
        f"conversion_share={diagnostics.get('conversion_share_of_consumer_gemm')} "
        f"capture={_relative_capture(out, candidate.get('capture'))}"
    )


def _contract_without_scenario_identity(group: dict[str, Any]) -> str:
    contract = str(group.get("contract_key") or "")
    if not contract:
        return f"semantics={group.get('semantics')};shape={_shape_text(group)}"
    scenario_keys = {
        "scenario_identity",
        "name",
        "promotion",
        "output_domain",
        "workflow",
        "evidence_scope",
    }
    kept: list[str] = []
    for part in contract.split(";"):
        key = part.split("=", 1)[0]
        if key not in scenario_keys:
            kept.append(part)
    return ";".join(kept) if kept else contract


def _contract_value(contract: str, name: str) -> str | None:
    prefix = f"{name}="
    for part in contract.split(";"):
        if part.startswith(prefix):
            return part[len(prefix) :]
    return None


def _native_handoff_pair_key(report_path: str, group: dict[str, Any]) -> tuple[str, str]:
    return (report_path, _contract_without_scenario_identity(group))


def _native_handoff_pair_disposition(
    fused: dict[str, Any] | None,
    control: dict[str, Any] | None,
) -> tuple[str, list[str], float | None]:
    blockers: list[str] = []
    if fused is None:
        blockers.append("missing_fused_device_handoff")
    if control is None:
        blockers.append("missing_host_export_repack_control")
    if fused is None or control is None:
        return "keep_experimental", blockers, None

    fused_e2e = fused.get("median_end_to_end_us")
    control_e2e = control.get("median_end_to_end_us")
    if not isinstance(fused_e2e, (int, float)):
        blockers.append("missing_fused_device_end_to_end")
    if not isinstance(control_e2e, (int, float)):
        blockers.append("missing_host_control_end_to_end")
    if fused.get("checksum") != control.get("checksum"):
        blockers.append("checksum_mismatch")

    speedup = None
    if isinstance(fused_e2e, (int, float)) and fused_e2e > 0 and isinstance(control_e2e, (int, float)):
        speedup = float(control_e2e) / float(fused_e2e)

    if blockers:
        return "keep_experimental", blockers, speedup
    if speedup is not None and speedup >= 1.02:
        return "local_promote", blockers, speedup
    if speedup is not None and speedup < 1.0:
        return "drop_deprioritize", blockers, speedup
    return "keep_experimental", blockers, speedup


def _native_handoff_pair_line(
    out: Path,
    report_path: str,
    contract: str,
    fused_row: tuple[str, dict[str, Any], dict[str, Any]] | None,
    control_row: tuple[str, dict[str, Any], dict[str, Any]] | None,
) -> tuple[str, str, list[str]]:
    group = fused_row[1] if fused_row is not None else control_row[1] if control_row is not None else {}
    fused = fused_row[2] if fused_row is not None else None
    control = control_row[2] if control_row is not None else None
    disposition, blockers, speedup = _native_handoff_pair_disposition(fused, control)
    fused_diag = (
        fused.get("native_to_rns_handoff_diagnostics")
        if isinstance(fused, dict) and isinstance(fused.get("native_to_rns_handoff_diagnostics"), dict)
        else {}
    )
    control_diag = (
        control.get("native_to_rns_handoff_diagnostics")
        if isinstance(control, dict) and isinstance(control.get("native_to_rns_handoff_diagnostics"), dict)
        else {}
    )
    line = (
        "  "
        f"review={report_path} "
        f"semantics={group.get('semantics')} "
        f"shape={_shape_text(group)} "
        f"pack_mode={_contract_value(contract, 'pack_mode') or 'unknown'} "
        f"disposition={disposition} "
        f"speedup_vs_host_repack={speedup} "
        f"fused_e2e={fused.get('median_end_to_end_us') if fused else None} "
        f"control_e2e={control.get('median_end_to_end_us') if control else None} "
        f"fused_conversion={fused_diag.get('conversion_median_us')} "
        f"host_repack={control_diag.get('host_repack_median_us')} "
        f"vector_output_d2h={control_diag.get('vector_output_d2h_median_us')} "
        f"consumer_gemm={fused_diag.get('consumer_gemm_median_us')} "
        f"blockers={_blocker_text(blockers)} "
        f"fused_capture={_relative_capture(out, fused.get('capture') if fused else None)} "
        f"control_capture={_relative_capture(out, control.get('capture') if control else None)}"
    )
    return line, disposition, blockers


def _export_route_line(
    out: Path,
    report_path: str,
    group: dict[str, Any],
    candidate: dict[str, Any],
) -> str:
    medians = candidate.get("phase_medians_us") if isinstance(candidate.get("phase_medians_us"), dict) else {}
    export = candidate.get("export_variant") if isinstance(candidate.get("export_variant"), dict) else {}
    reconstruction = (
        candidate.get("reconstruction_variant") if isinstance(candidate.get("reconstruction_variant"), dict) else {}
    )
    exact_output = (
        candidate.get("exact_output_contract") if isinstance(candidate.get("exact_output_contract"), dict) else {}
    )
    blockers = _blocker_text(candidate.get("promotion_blockers"))
    return (
        "  "
        f"review={report_path} "
        f"backend={candidate.get('backend')} "
        f"semantics={group.get('semantics')} "
        f"shape={_shape_text(group)} "
        f"kernel={candidate.get('selected_kernel')} "
        f"export_kernel={export.get('selected_kernel') or exact_output.get('kernel_identity')} "
        f"export_variant={export.get('name')} "
        f"reconstruction={reconstruction.get('name')} "
        f"crt_export={medians.get('crt_export')} "
        f"e2e={candidate.get('median_end_to_end_us')} "
        f"primary_loss={candidate.get('primary_loss_phase_vs_direct_hip')} "
        f"status_policy={export.get('selector_status_policy') or exact_output.get('status_policy')} "
        f"d2h_policy={export.get('d2h_policy')} "
        f"final_mode={export.get('final_output_mode') or exact_output.get('output_domain_after_measured_repeats')} "
        f"blockers={blockers} "
        f"capture={_relative_capture(out, candidate.get('capture'))}"
    )


def _sparse_a_route_active(group: dict[str, Any], candidate: dict[str, Any]) -> bool:
    families = set(_scenario_families(group))
    backend = str(candidate.get("backend") or "")
    kernel = str(candidate.get("selected_kernel") or "")
    return (
        "sparse-a-4-to-2" in families
        or "sparse-a" in backend
        or "dense-sparse-a-input" in backend
        or "sparse_a" in kernel
    )


def _sparse_a_route_line(
    out: Path,
    report_path: str,
    group: dict[str, Any],
    candidate: dict[str, Any],
    isa_index: dict[str, list[dict[str, Any]]] | None = None,
) -> str:
    medians = candidate.get("phase_medians_us") if isinstance(candidate.get("phase_medians_us"), dict) else {}
    blockers = _blocker_text(candidate.get("promotion_blockers"))
    return (
        "  "
        f"review={report_path} "
        f"backend={candidate.get('backend')} "
        f"semantics={group.get('semantics')} "
        f"shape={_shape_text(group)} "
        f"kernel={candidate.get('selected_kernel')} "
        f"e2e={candidate.get('median_end_to_end_us')} "
        f"pack={medians.get('pack')} "
        f"rns_gemm={medians.get('rns_gemm')} "
        f"crt_export={medians.get('crt_export')} "
        f"matrix_meta={_matrix_metadata_text(candidate)} "
        f"matrix_isa={_histogram_text(candidate, isa_index)} "
        f"blockers={blockers} "
        f"capture={_relative_capture(out, candidate.get('capture'))}"
    )


MATRIX_CORE_FAMILIES = {"mfma", "smfmac", "wmma", "swmmac"}
MATRIX_CORE_MNEMONIC_PREFIXES = ("v_mfma", "v_smfmac", "v_wmma", "v_swmmac")


def _matrix_core_route_active(candidate: dict[str, Any]) -> bool:
    family = candidate.get("matrix_instruction_family")
    if isinstance(family, str) and family.lower() in MATRIX_CORE_FAMILIES:
        return True
    kernel = str(candidate.get("selected_kernel") or "")
    if any(f"_{family}_" in kernel for family in MATRIX_CORE_FAMILIES):
        return True
    histogram = candidate.get("matrix_instruction_histogram")
    if isinstance(histogram, dict):
        return any(str(key).startswith(MATRIX_CORE_MNEMONIC_PREFIXES) for key, value in histogram.items() if value)
    return False


def _matrix_core_route_line(
    out: Path,
    report_path: str,
    group: dict[str, Any],
    candidate: dict[str, Any],
    isa_index: dict[str, list[dict[str, Any]]] | None = None,
) -> str:
    medians = candidate.get("phase_medians_us") if isinstance(candidate.get("phase_medians_us"), dict) else {}
    blockers = _blocker_text(candidate.get("promotion_blockers"))
    return (
        "  "
        f"review={report_path} "
        f"backend={candidate.get('backend')} "
        f"semantics={group.get('semantics')} "
        f"shape={_shape_text(group)} "
        f"tile_shape_variant={candidate.get('tile_shape_variant') or 'default'} "
        f"kernel={candidate.get('selected_kernel')} "
        f"e2e={candidate.get('median_end_to_end_us')} "
        f"pack={medians.get('pack')} "
        f"rns_gemm={medians.get('rns_gemm')} "
        f"crt_export={medians.get('crt_export')} "
        f"primary_loss={candidate.get('primary_loss_phase_vs_direct_hip')} "
        f"matrix_meta={_matrix_metadata_text(candidate)} "
        f"matrix_isa={_histogram_text(candidate, isa_index)} "
        f"blockers={blockers} "
        f"capture={_relative_capture(out, candidate.get('capture'))}"
    )


def build_summary(
    out: Path,
    *,
    max_route_rows: int = DEFAULT_MAX_ROUTE_ROWS,
    max_detail_rows: int = DEFAULT_MAX_DETAIL_ROWS,
) -> list[str]:
    lines = [f"HEAD {_git_head()}", f"OUT {out}"]
    isa_index = load_isa_index([out])

    failed = sorted(out.rglob("*.failed.json"))
    capture_paths = _scenario_capture_paths(out)
    lines.append(f"FAILED_CAPTURES {len(failed)}")
    lines.append(f"CAPTURE_JSON_COUNT {len(capture_paths)}")
    for path in failed:
        payload = _load_json(path)
        stderr = str(payload.get("stderr", "")).strip().replace("\n", " | ")
        lines.append(f"{path.relative_to(out)}: {stderr}")

    groups: dict[str, list[tuple[str, Any, Path]]] = defaultdict(list)
    for path in capture_paths:
        payload = _load_json(path)
        groups[capture_contract_key(payload)].append((backend_id(payload), capture_checksum(payload), path))

    mismatches: list[tuple[str, str | None, Any, list[tuple[str, Any, Path]]]] = []
    for key, rows in groups.items():
        reference_backend, reference = _reference_checksum(rows)
        if reference is None:
            continue
        bad = [
            (backend, checksum, path)
            for backend, checksum, path in rows
            if checksum is not None and checksum != reference
        ]
        if bad:
            mismatches.append((key, reference_backend, reference, bad))

    lines.append(f"CHECKSUM_MISMATCH_GROUPS {len(mismatches)}")
    for key, reference_backend, reference, bad in sorted(mismatches):
        lines.append(f"GROUP {key}")
        lines.append(f"  ref_backend={reference_backend} ref={reference}")
        for backend, checksum, path in bad:
            lines.append(f"  {backend}\t{checksum}\t{path.relative_to(out)}")

    blocker_counts: Counter[str] = Counter()
    actionable_counts: Counter[str] = Counter()
    actionable_rows: list[tuple[str, str, dict[str, Any], dict[str, Any]]] = []
    promotable_entries: list[tuple[str, dict[str, Any]]] = []
    missing_baseline_rows: list[tuple[str, dict[str, Any]]] = []
    production_routes: list[tuple[dict[str, Any], dict[str, Any]]] = []
    accelerator_routes: list[tuple[dict[str, Any], dict[str, Any]]] = []
    production_route_counts: Counter[str] = Counter()
    accelerator_route_counts: Counter[str] = Counter()
    loss_phase_counts: Counter[str] = Counter()
    loss_phase_by_backend: Counter[tuple[str, str]] = Counter()
    loss_phase_by_semantics: Counter[tuple[str, str]] = Counter()
    loss_phase_by_shape_family: Counter[tuple[str, str]] = Counter()
    loss_phase_by_scenario_family: Counter[tuple[str, str]] = Counter()
    bottleneck_counts: Counter[str] = Counter()
    direct_hip_production_wins: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    next_work_rows: list[tuple[str, dict[str, Any]]] = []
    prepack_reuse_rows: list[tuple[str, dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    prepack_reuse_blockers: Counter[str] = Counter()
    graph_replay_rows: list[tuple[str, dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    graph_replay_blockers: Counter[str] = Counter()
    pack_diagnostic_rows: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    pack_split_counts: Counter[str] = Counter()
    pack_dominant_operand_counts: Counter[str] = Counter()
    native_handoff_rows: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    native_handoff_pair_rows: dict[
        tuple[str, str],
        dict[str, tuple[str, dict[str, Any], dict[str, Any]]],
    ] = defaultdict(dict)
    export_route_rows: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    export_kernel_counts: Counter[str] = Counter()
    sparse_a_route_rows: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    sparse_a_route_counts: Counter[str] = Counter()
    matrix_core_route_rows: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    matrix_core_family_counts: Counter[str] = Counter()
    matrix_core_backend_counts: Counter[str] = Counter()
    review_report_count = 0
    for path in out.rglob("review_report.json"):
        review_report_count += 1
        report = _load_json(path)
        summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
        if isinstance(summary.get("next_work"), list):
            for row in summary["next_work"]:
                if isinstance(row, dict):
                    next_work_rows.append((str(path.relative_to(out)), row))
        for entry in report.get("promotable_autotune_entries", []):
            if isinstance(entry, dict):
                promotable_entries.append((str(path.relative_to(out)), entry))
        for group in report.get("groups", []):
            if group.get("missing_required_baselines"):
                missing_baseline_rows.append((str(path.relative_to(out)), group))
            production = group.get("fastest_production_route")
            if isinstance(production, dict):
                production_routes.append((group, production))
                production_route_counts.update([str(production.get("backend"))])
                bottleneck = production.get("bottleneck") if isinstance(production.get("bottleneck"), dict) else {}
                bottleneck_counts.update([str(bottleneck.get("class") or "unknown")])
                if production.get("backend") == "hip-direct":
                    direct_hip_production_wins.append((str(path.relative_to(out)), group, production))
            accelerator = group.get("fastest_accelerator_route")
            if isinstance(accelerator, dict):
                accelerator_routes.append((group, accelerator))
                accelerator_route_counts.update([str(accelerator.get("backend"))])
            for candidate in group.get("candidates", []):
                blocker_counts.update(candidate.get("promotion_blockers", []))
                blockers = _actionable_blockers(candidate)
                prepack = candidate.get("prepacked_reuse_review")
                if isinstance(prepack, dict):
                    prepack_reuse_rows.append((str(path.relative_to(out)), group, candidate, prepack))
                    prepack_reuse_blockers.update(str(item) for item in prepack.get("blockers", []) if item)
                graph = candidate.get("hip_graph_replay_review")
                if isinstance(graph, dict):
                    graph_replay_rows.append((str(path.relative_to(out)), group, candidate, graph))
                    graph_replay_blockers.update(str(item) for item in graph.get("blockers", []) if item)
                diagnostics = candidate.get("pack_diagnostics")
                if isinstance(diagnostics, dict):
                    split_state = "split_available" if diagnostics.get("split_available") is True else "split_missing"
                    pack_split_counts.update([split_state])
                    dominant = diagnostics.get("dominant_operand")
                    if isinstance(dominant, str) and dominant:
                        pack_dominant_operand_counts.update([dominant])
                    if _pack_diagnostic_active(candidate):
                        pack_diagnostic_rows.append((str(path.relative_to(out)), group, candidate))
                if _native_handoff_active(candidate):
                    report_path = str(path.relative_to(out))
                    native_handoff_rows.append((report_path, group, candidate))
                    diagnostics = candidate.get("native_to_rns_handoff_diagnostics")
                    if isinstance(diagnostics, dict):
                        control_mode = diagnostics.get("control_mode")
                        if control_mode in {"fused_device_native_to_rns", "host_export_repack_control"}:
                            native_handoff_pair_rows[_native_handoff_pair_key(report_path, group)][
                                str(control_mode)
                            ] = (report_path, group, candidate)
                if _export_route_active(candidate):
                    export_route_rows.append((str(path.relative_to(out)), group, candidate))
                    export = candidate.get("export_variant") if isinstance(candidate.get("export_variant"), dict) else {}
                    exact = (
                        candidate.get("exact_output_contract")
                        if isinstance(candidate.get("exact_output_contract"), dict)
                        else {}
                    )
                    export_kernel_counts.update([str(export.get("selected_kernel") or exact.get("kernel_identity") or "unknown")])
                if _sparse_a_route_active(group, candidate):
                    sparse_a_route_rows.append((str(path.relative_to(out)), group, candidate))
                    sparse_a_route_counts.update([str(candidate.get("backend") or "unknown")])
                if _matrix_core_route_active(candidate):
                    matrix_core_route_rows.append((str(path.relative_to(out)), group, candidate))
                    matrix_core_family_counts.update([str(candidate.get("matrix_instruction_family") or "unknown")])
                    matrix_core_backend_counts.update([str(candidate.get("backend") or "unknown")])
                if blockers:
                    actionable_counts.update(blockers)
                    actionable_rows.append((str(path.relative_to(out)), str(group.get("contract_key", "")), group, candidate))
                    phase = candidate.get("primary_loss_phase_vs_direct_hip")
                    if isinstance(phase, str) and phase:
                        backend = str(candidate.get("backend") or "unknown")
                        semantics = str(group.get("semantics") or "unknown")
                        shape_family = str(group.get("shape_family") or "unknown")
                        loss_phase_counts.update([phase])
                        loss_phase_by_backend.update([(backend, phase)])
                        loss_phase_by_semantics.update([(semantics, phase)])
                        loss_phase_by_shape_family.update([(shape_family, phase)])
                        loss_phase_by_scenario_family.update((family, phase) for family in _scenario_families(group))
                    bottleneck = candidate.get("bottleneck") if isinstance(candidate.get("bottleneck"), dict) else {}
                    bottleneck_counts.update([str(bottleneck.get("class") or "unknown")])
    lines.append(f"REVIEW_REPORTS {review_report_count}")
    lines.append("REVIEW_BLOCKER_COUNTS")
    for blocker, count in blocker_counts.most_common():
        lines.append(f"{blocker} {count}")
    lines.append(f"PROMOTABLE_AUTOTUNE_ENTRIES {len(promotable_entries)}")
    for report_path, entry in promotable_entries[:max_detail_rows]:
        lines.append(
            "  "
            f"review={report_path} "
            f"backend={entry.get('selected_backend')} "
            f"kernel={entry.get('selected_kernel')} "
            f"e2e={entry.get('median_end_to_end_us')} "
            f"selection_e2e={entry.get('selection_end_to_end_us')} "
            f"source={_relative_capture(out, entry.get('source_capture'))}"
        )
    if len(promotable_entries) > max_detail_rows:
        lines.append(f"  ... {len(promotable_entries) - max_detail_rows} more")
    lines.append(f"MISSING_REQUIRED_BASELINE_GROUPS {len(missing_baseline_rows)}")
    for report_path, group in missing_baseline_rows[:max_detail_rows]:
        lines.append(
            "  "
            f"review={report_path} "
            f"semantics={group.get('semantics')} "
            f"shape={_shape_text(group)} "
            f"missing={_blocker_text(group.get('missing_required_baselines'))} "
            f"required={_blocker_text(group.get('required_baselines'))} "
            f"present={_blocker_text(_group_backends(group))} "
            f"scopes={_blocker_text(group.get('scenario_promotion_scopes'))}"
        )
    if len(missing_baseline_rows) > max_detail_rows:
        lines.append(f"  ... {len(missing_baseline_rows) - max_detail_rows} more")
    lines.append("ROUTE_SUMMARY")
    if production_route_counts:
        for backend, count in production_route_counts.most_common():
            lines.append(f"fastest_production {backend} {count}")
    else:
        lines.append("fastest_production none 0")
    if accelerator_route_counts:
        for backend, count in accelerator_route_counts.most_common():
            lines.append(f"fastest_accelerator {backend} {count}")
    else:
        lines.append("fastest_accelerator none 0")
    lines.append(f"DIRECT_HIP_PRODUCTION_WINS {len(direct_hip_production_wins)}")
    for report_path, group, candidate in direct_hip_production_wins[:max_detail_rows]:
        lines.append(
            "  "
            f"review={report_path} "
            f"semantics={group.get('semantics')} "
            f"shape={_shape_text(group)} "
            f"shape_family={group.get('shape_family') or 'unknown'} "
            f"e2e={candidate.get('median_end_to_end_us')} "
            f"bottleneck={candidate.get('bottleneck')} "
            f"capture={_relative_capture(out, candidate.get('capture'))}"
        )
    if len(direct_hip_production_wins) > max_detail_rows:
        lines.append(f"  ... {len(direct_hip_production_wins) - max_detail_rows} more")
    lines.append("LOSS_PHASE_COUNTS")
    if loss_phase_counts:
        for phase, count in loss_phase_counts.most_common():
            lines.append(f"{phase} {count}")
    else:
        lines.append("none 0")
    lines.append("LOSS_PHASE_BY_BACKEND")
    if loss_phase_by_backend:
        for (backend, phase), count in loss_phase_by_backend.most_common():
            lines.append(f"{backend} {phase} {count}")
    else:
        lines.append("none none 0")
    lines.append("LOSS_PHASE_BY_SEMANTICS")
    if loss_phase_by_semantics:
        for (semantics, phase), count in loss_phase_by_semantics.most_common():
            lines.append(f"{semantics} {phase} {count}")
    else:
        lines.append("none none 0")
    lines.append("LOSS_PHASE_BY_SHAPE_FAMILY")
    if loss_phase_by_shape_family:
        for (shape_family, phase), count in loss_phase_by_shape_family.most_common():
            lines.append(f"{shape_family} {phase} {count}")
    else:
        lines.append("none none 0")
    lines.append("LOSS_PHASE_BY_SCENARIO_FAMILY")
    if loss_phase_by_scenario_family:
        for (family, phase), count in loss_phase_by_scenario_family.most_common():
            lines.append(f"{family} {phase} {count}")
    else:
        lines.append("none none 0")
    lines.append("BOTTLENECK_COUNTS")
    if bottleneck_counts:
        for bottleneck, count in bottleneck_counts.most_common():
            lines.append(f"{bottleneck} {count}")
    else:
        lines.append("none 0")
    lines.append(f"NEXT_WORK {len(next_work_rows)}")
    for report_path, row in next_work_rows[:max_detail_rows]:
        lines.append(
            "  "
            f"review={report_path} "
            f"priority={row.get('priority')} "
            f"work={row.get('work')} "
            f"reason={row.get('reason')}"
        )
    if len(next_work_rows) > max_detail_rows:
        lines.append(f"  ... {len(next_work_rows) - max_detail_rows} more")
    lines.append(f"PREPACK_REUSE_REVIEWS {len(prepack_reuse_rows)}")
    lines.append("PREPACK_REUSE_BLOCKER_COUNTS")
    if prepack_reuse_blockers:
        for blocker, count in prepack_reuse_blockers.most_common():
            lines.append(f"{blocker} {count}")
    else:
        lines.append("none 0")
    for report_path, group, candidate, prepack in prepack_reuse_rows[:max_detail_rows]:
        lines.append(_prepack_reuse_line(out, report_path, group, candidate, prepack))
    if len(prepack_reuse_rows) > max_detail_rows:
        lines.append(f"  ... {len(prepack_reuse_rows) - max_detail_rows} more")
    lines.append(f"HIP_GRAPH_REPLAY_REVIEWS {len(graph_replay_rows)}")
    lines.append("HIP_GRAPH_REPLAY_BLOCKER_COUNTS")
    if graph_replay_blockers:
        for blocker, count in graph_replay_blockers.most_common():
            lines.append(f"{blocker} {count}")
    else:
        lines.append("none 0")
    for report_path, group, candidate, graph in graph_replay_rows[:max_detail_rows]:
        lines.append(_graph_replay_line(out, report_path, group, candidate, graph))
    if len(graph_replay_rows) > max_detail_rows:
        lines.append(f"  ... {len(graph_replay_rows) - max_detail_rows} more")
    lines.append(f"PACK_PHASE_DIAGNOSTICS {len(pack_diagnostic_rows)}")
    lines.append("PACK_PHASE_SPLIT_COUNTS")
    if pack_split_counts:
        for split_state, count in pack_split_counts.most_common():
            lines.append(f"{split_state} {count}")
    else:
        lines.append("none 0")
    lines.append("PACK_PHASE_DOMINANT_OPERAND_COUNTS")
    if pack_dominant_operand_counts:
        for operand, count in pack_dominant_operand_counts.most_common():
            lines.append(f"{operand} {count}")
    else:
        lines.append("none 0")
    pack_diagnostic_rows.sort(
        key=lambda row: (
            row[2].get("pack_diagnostics", {}).get("pack_share_of_end_to_end")
            if isinstance(row[2].get("pack_diagnostics", {}).get("pack_share_of_end_to_end"), (int, float))
            else 0.0
        ),
        reverse=True,
    )
    for report_path, group, candidate in pack_diagnostic_rows[:max_detail_rows]:
        lines.append(_pack_diagnostic_line(out, report_path, group, candidate))
    if len(pack_diagnostic_rows) > max_detail_rows:
        lines.append(f"  ... {len(pack_diagnostic_rows) - max_detail_rows} more")
    lines.append(f"NATIVE_TO_RNS_HANDOFF_DIAGNOSTICS {len(native_handoff_rows)}")
    for report_path, group, candidate in native_handoff_rows[:max_detail_rows]:
        lines.append(_native_handoff_line(out, report_path, group, candidate))
    if len(native_handoff_rows) > max_detail_rows:
        lines.append(f"  ... {len(native_handoff_rows) - max_detail_rows} more")
    native_handoff_pair_lines: list[tuple[str, str, list[str]]] = []
    for (_report_path, contract), pair in sorted(native_handoff_pair_rows.items()):
        fused_row = pair.get("fused_device_native_to_rns")
        control_row = pair.get("host_export_repack_control")
        report_path = (
            fused_row[0]
            if fused_row is not None
            else control_row[0]
            if control_row is not None
            else _report_path
        )
        native_handoff_pair_lines.append(
            _native_handoff_pair_line(out, report_path, contract, fused_row, control_row)
        )
    native_handoff_pair_dispositions = Counter(row[1] for row in native_handoff_pair_lines)
    native_handoff_pair_blockers: Counter[str] = Counter()
    for _line, _disposition, blockers in native_handoff_pair_lines:
        native_handoff_pair_blockers.update(blockers)
    lines.append(f"NATIVE_TO_RNS_CHAIN_PAIRS {len(native_handoff_pair_lines)}")
    lines.append("NATIVE_TO_RNS_CHAIN_PAIR_DISPOSITIONS")
    if native_handoff_pair_dispositions:
        for disposition, count in native_handoff_pair_dispositions.most_common():
            lines.append(f"{disposition} {count}")
    else:
        lines.append("none 0")
    lines.append("NATIVE_TO_RNS_CHAIN_PAIR_BLOCKERS")
    if native_handoff_pair_blockers:
        for blocker, count in native_handoff_pair_blockers.most_common():
            lines.append(f"{blocker} {count}")
    else:
        lines.append("none 0")
    for line, _disposition, _blockers in native_handoff_pair_lines[:max_detail_rows]:
        lines.append(line)
    if len(native_handoff_pair_lines) > max_detail_rows:
        lines.append(f"  ... {len(native_handoff_pair_lines) - max_detail_rows} more")
    lines.append(f"EXPORT_CRT_ROUTE_ROWS {len(export_route_rows)}")
    lines.append("EXPORT_CRT_KERNEL_COUNTS")
    if export_kernel_counts:
        for kernel, count in export_kernel_counts.most_common():
            lines.append(f"{kernel} {count}")
    else:
        lines.append("none 0")
    for report_path, group, candidate in export_route_rows[:max_detail_rows]:
        lines.append(_export_route_line(out, report_path, group, candidate))
    if len(export_route_rows) > max_detail_rows:
        lines.append(f"  ... {len(export_route_rows) - max_detail_rows} more")
    lines.append(f"SPARSE_A_ROUTE_ROWS {len(sparse_a_route_rows)}")
    lines.append("SPARSE_A_ROUTE_COUNTS")
    if sparse_a_route_counts:
        for backend, count in sparse_a_route_counts.most_common():
            lines.append(f"{backend} {count}")
    else:
        lines.append("none 0")
    for report_path, group, candidate in sparse_a_route_rows[:max_detail_rows]:
        lines.append(_sparse_a_route_line(out, report_path, group, candidate, isa_index))
    if len(sparse_a_route_rows) > max_detail_rows:
        lines.append(f"  ... {len(sparse_a_route_rows) - max_detail_rows} more")
    lines.append(f"MATRIX_CORE_ROUTE_ROWS {len(matrix_core_route_rows)}")
    lines.append("MATRIX_CORE_FAMILY_COUNTS")
    if matrix_core_family_counts:
        for family, count in matrix_core_family_counts.most_common():
            lines.append(f"{family} {count}")
    else:
        lines.append("none 0")
    lines.append("MATRIX_CORE_BACKEND_COUNTS")
    if matrix_core_backend_counts:
        for backend, count in matrix_core_backend_counts.most_common():
            lines.append(f"{backend} {count}")
    else:
        lines.append("none 0")
    for report_path, group, candidate in matrix_core_route_rows[:max_detail_rows]:
        lines.append(_matrix_core_route_line(out, report_path, group, candidate, isa_index))
    if len(matrix_core_route_rows) > max_detail_rows:
        lines.append(f"  ... {len(matrix_core_route_rows) - max_detail_rows} more")
    lines.append("ACTIONABLE_PROMOTION_BLOCKER_COUNTS")
    if actionable_counts:
        for blocker, count in actionable_counts.most_common():
            lines.append(f"{blocker} {count}")
    else:
        lines.append("none 0")
    lines.append(f"ACTIONABLE_PROMOTION_CANDIDATES {len(actionable_rows)}")
    for report_path, contract_key, group, candidate in actionable_rows[:max_detail_rows]:
        blockers = ",".join(_actionable_blockers(candidate))
        lines.append(
            "  "
            f"review={report_path} "
            f"{candidate.get('backend')} "
            f"semantics={group.get('semantics')} "
            f"shape={_shape_text(group)} "
            f"kernel={candidate.get('selected_kernel')} "
            f"e2e={candidate.get('median_end_to_end_us')} "
            f"vs_direct={candidate.get('speedup_vs_direct_hip')} "
            f"vs_vector={candidate.get('speedup_vs_vector_alu')} "
            f"blockers={blockers} "
            f"details={_review_detail_text(candidate)} "
            f"capture={_relative_capture(out, candidate.get('capture'))} "
            f"contract={contract_key}"
        )
    if len(actionable_rows) > max_detail_rows:
        lines.append(f"  ... {len(actionable_rows) - max_detail_rows} more")
    top_actionable_rows = sorted(actionable_rows, key=_candidate_sort_key, reverse=True)
    lines.append(f"TOP_ACTIONABLE_ACCELERATOR_BLOCKERS {len(top_actionable_rows)}")
    for report_path, contract_key, group, candidate in top_actionable_rows[:max_detail_rows]:
        lines.append(_top_actionable_line(out, report_path, contract_key, group, candidate))
    if len(top_actionable_rows) > max_detail_rows:
        lines.append(f"  ... {len(top_actionable_rows) - max_detail_rows} more")
    lines.append(f"FASTEST_PRODUCTION_ROUTES {len(production_routes)}")
    for group, candidate in production_routes[:max_route_rows]:
        lines.append(_route_line(out, "production", group, candidate, isa_index))
    if len(production_routes) > max_route_rows:
        lines.append(f"  ... {len(production_routes) - max_route_rows} more")
    lines.append(f"FASTEST_ACCELERATOR_ROUTES {len(accelerator_routes)}")
    for group, candidate in accelerator_routes[:max_route_rows]:
        lines.append(_route_line(out, "accelerator", group, candidate, isa_index))
    if len(accelerator_routes) > max_route_rows:
        lines.append(f"  ... {len(accelerator_routes) - max_route_rows} more")
    return lines


def _prefixed_int(lines: list[str], prefix: str) -> int | None:
    for line in lines:
        if not line.startswith(prefix):
            continue
        parts = line.split()
        if len(parts) < 2:
            return None
        try:
            return int(parts[1])
        except ValueError:
            return None
    return None


def clean_gate_failures(lines: list[str]) -> list[str]:
    checks = [
        ("FAILED_CAPTURES", "failed captures"),
        ("CHECKSUM_MISMATCH_GROUPS", "comparable checksum mismatch groups"),
        ("MISSING_REQUIRED_BASELINE_GROUPS", "missing required baseline groups"),
    ]
    failures: list[str] = []
    for prefix, label in checks:
        count = _prefixed_int(lines, prefix)
        if count is None:
            failures.append(f"{prefix} section missing or malformed")
        elif count != 0:
            failures.append(f"{label}={count}")
    review_count = _prefixed_int(lines, "REVIEW_REPORTS")
    if review_count is None:
        failures.append("REVIEW_REPORTS section missing or malformed")
    elif review_count <= 0:
        failures.append("review_reports=0")
    capture_count = _prefixed_int(lines, "CAPTURE_JSON_COUNT")
    if capture_count is None:
        failures.append("CAPTURE_JSON_COUNT section missing or malformed")
    elif capture_count <= 0:
        failures.append("capture_json_count=0")
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", nargs="?", type=Path, help="sweep output directory; defaults to latest temp/cdna-*mi300x-*")
    parser.add_argument("--temp-root", type=Path, default=Path("temp"))
    parser.add_argument("--max-route-rows", type=int, default=DEFAULT_MAX_ROUTE_ROWS)
    parser.add_argument("--max-detail-rows", type=int, default=DEFAULT_MAX_DETAIL_ROWS)
    parser.add_argument(
        "--require-clean",
        action="store_true",
        help=(
            "exit nonzero unless the sweep has at least one review report, zero failed captures, "
            "zero comparable checksum mismatch groups, and zero missing required baseline groups"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = args.out if args.out is not None else _latest_cdna_out(args.temp_root)
    lines = build_summary(out, max_route_rows=args.max_route_rows, max_detail_rows=args.max_detail_rows)
    for line in lines:
        print(line)
    if args.require_clean:
        failures = clean_gate_failures(lines)
        if failures:
            print("require-clean failed: " + "; ".join(failures), file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
