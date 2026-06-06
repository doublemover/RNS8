from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from .config import PHASES
from .isa import lookup_isa_resources, lookup_metadata
from .work_model import (
    build_roofline_priority,
    classify_bottleneck,
    estimate_work,
    gpu_evidence_rows,
    optimization_hint_for_target,
    roofline_target_for_row,
)

def build_row(
    capture: dict[str, Any],
    scenario: dict[str, Any],
    review: dict[str, Any],
    isa_resources: dict[str, Any],
) -> dict[str, Any]:
    metadata = capture.get("backend_metadata") if isinstance(capture.get("backend_metadata"), dict) else {}
    timing_metadata = capture.get("timing_metadata") if isinstance(capture.get("timing_metadata"), dict) else {}
    device = capture.get("device") if isinstance(capture.get("device"), dict) else {}
    work = estimate_work(capture)
    bottleneck = classify_bottleneck(capture)
    scenario_extra = scenario.get("metadata") if isinstance(scenario.get("metadata"), dict) else {}
    reuse_contract = capture.get("reuse_contract") if isinstance(capture.get("reuse_contract"), dict) else {}
    export_variant = capture.get("export_variant") if isinstance(capture.get("export_variant"), dict) else {}
    reconstruction_variant = (
        capture.get("reconstruction_variant") if isinstance(capture.get("reconstruction_variant"), dict) else {}
    )
    modulus_set = capture.get("modulus_set") if isinstance(capture.get("modulus_set"), dict) else {}
    residue_count_policy = (
        capture.get("residue_count_policy") if isinstance(capture.get("residue_count_policy"), dict) else {}
    )
    tile_shape_variant = capture.get("tile_shape_variant") if isinstance(capture.get("tile_shape_variant"), dict) else {}
    grouped_dispatch = capture.get("grouped_dispatch") if isinstance(capture.get("grouped_dispatch"), dict) else {}
    adaptive_grouped_scheduler = (
        capture.get("adaptive_grouped_scheduler")
        if isinstance(capture.get("adaptive_grouped_scheduler"), dict)
        else {}
    )
    resident_lifetime = capture.get("resident_lifetime") if isinstance(capture.get("resident_lifetime"), dict) else {}
    workspace_arena = capture.get("workspace_arena") if isinstance(capture.get("workspace_arena"), dict) else {}
    streaming_overlap = capture.get("streaming_overlap") if isinstance(capture.get("streaming_overlap"), dict) else {}
    hip_graph_replay = capture.get("hip_graph_replay") if isinstance(capture.get("hip_graph_replay"), dict) else {}
    release_gate = capture.get("release_gate") if isinstance(capture.get("release_gate"), dict) else {}
    verification_amortization = (
        capture.get("verification_amortization")
        if isinstance(capture.get("verification_amortization"), dict)
        else {}
    )
    workload_proxy = capture.get("workload_proxy") if isinstance(capture.get("workload_proxy"), dict) else {}
    row = {
        "capture_path": capture.get("_path"),
        "scenario_family": scenario.get("family", "unlabeled"),
        "scenario_name": scenario.get("name"),
        "scenario_evidence_scope": scenario.get("evidence_scope"),
        "scenario_rationale": scenario.get("rationale"),
        "scenario_source_role": scenario_extra.get("source_role"),
        "scenario_evidence_role": scenario_extra.get("evidence_role"),
        "scenario_domain_family": scenario_extra.get("domain_family"),
        "scenario_algebra_family": scenario_extra.get("algebra_family"),
        "scenario_workflow_name": scenario_extra.get("workflow_name"),
        "scenario_phase_label": scenario_extra.get("phase_label"),
        "scenario_reuse_profile": scenario_extra.get("reuse_profile"),
        "scenario_lowering_role": scenario_extra.get("lowering_role"),
        "scenario_output_domain_requirement": scenario_extra.get("output_domain_requirement"),
        "scenario_large_shape_role": scenario_extra.get("large_shape_role"),
        "scenario_promotion_scope": scenario_extra.get("promotion_scope"),
        "scenario_grouping_role": scenario_extra.get("grouping_role"),
        "scenario_bridge_role": scenario_extra.get("bridge_role"),
        "scenario_modulus_role": scenario_extra.get("modulus_role"),
        "scenario_prime_or_composite": scenario_extra.get("prime_or_composite"),
        "scenario_metadata": scenario_extra,
        "scenario_metadata_json": json.dumps(scenario_extra, sort_keys=True) if scenario_extra else None,
        "semantics": capture.get("semantics"),
        "backend": capture.get("backend_selected"),
        "backend_requested": capture.get("backend_requested"),
        "selected_kernel": capture.get("selected_kernel"),
        "m": capture.get("m"),
        "n": capture.get("n"),
        "k": capture.get("k"),
        "finite_modulus": capture.get("finite_modulus"),
        "prefix": capture.get("prefix"),
        "selected_prefix": capture.get("selected_prefix"),
        "exact_wide_limb_count": capture.get("exact_wide_limb_count"),
        "pack_mode": capture.get("pack_mode") or timing_metadata.get("pack_mode") or scenario.get("pack_mode"),
        "prepack_reuse_strategy": capture.get("prepack_reuse_strategy") or timing_metadata.get("prepack_reuse_strategy"),
        "reuse_contract_enabled": reuse_contract.get("enabled"),
        "reuse_contract_operand_role": reuse_contract.get("operand_role"),
        "reuse_contract_setup_scope": reuse_contract.get("setup_scope"),
        "export_variant": export_variant.get("name"),
        "reconstruction_variant": reconstruction_variant.get("name"),
        "modulus_set": modulus_set.get("name"),
        "residue_count_policy": residue_count_policy.get("policy"),
        "tile_shape_variant": tile_shape_variant.get("name"),
        "grouped_dispatch_task_count": grouped_dispatch.get("task_count"),
        "grouped_dispatch_status": grouped_dispatch.get("capture_status"),
        "adaptive_grouped_scheduler_requested": adaptive_grouped_scheduler.get("requested"),
        "adaptive_grouped_scheduler_status": adaptive_grouped_scheduler.get("capture_status"),
        "resident_lifetime_enabled": resident_lifetime.get("enabled"),
        "resident_lifetime_output_domain": resident_lifetime.get("output_domain"),
        "workspace_arena_enabled": workspace_arena.get("enabled"),
        "workspace_arena_repeat_allocation_free": workspace_arena.get("measured_repeat_allocation_free"),
        "streaming_overlap_requested": streaming_overlap.get("requested"),
        "streaming_overlap_status": streaming_overlap.get("capture_status"),
        "hip_graph_replay_requested": hip_graph_replay.get("requested"),
        "hip_graph_replay_status": hip_graph_replay.get("capture_status"),
        "release_gate": release_gate.get("name"),
        "release_gate_review_status": release_gate.get("review_status"),
        "verification_amortization_policy": verification_amortization.get("policy"),
        "workload_proxy_family": workload_proxy.get("family"),
        "workload_proxy_label": workload_proxy.get("label"),
        "output_domain": scenario.get("output_domain") or capture.get("residue_output_mode"),
        "target_id": device.get("gcn_arch"),
        "device_name": device.get("name"),
        "hip_runtime_version": device.get("hip_runtime_version"),
        "hip_driver_version": device.get("hip_driver_version"),
        "hip_sdk_or_rocm_version": (capture.get("hip_toolchain") or {}).get("hip_sdk_or_rocm_version")
        if isinstance(capture.get("hip_toolchain"), dict)
        else None,
        "accelerator_library": metadata.get("accelerator_library"),
        "workspace_bytes": metadata.get("workspace_required_bytes"),
        "isa_evidence": metadata.get("isa_evidence"),
        "autotune_key": metadata.get("autotune_key"),
        "checksum": capture.get("checksum"),
        "warmups": capture.get("warmups"),
        "repeats": capture.get("repeats"),
        "seed": capture.get("seed"),
        "review": review,
        **work,
        **bottleneck,
        **isa_resources,
    }
    for phase in PHASES:
        row[f"median_{phase}_us"] = row["phase_medians_us"].get(phase)
    row["roofline_target"] = roofline_target_for_row(row)
    row["optimization_hint"] = optimization_hint_for_target(str(row["roofline_target"]))
    row["promotable"] = review.get("promotable")
    row["promotion_blockers"] = review.get("promotion_blockers") or []
    return row


def build_database(
    captures: list[dict[str, Any]],
    *,
    scenario_index: dict[str, dict[str, Any]] | None = None,
    review_index: dict[str, dict[str, Any]] | None = None,
    isa_index: dict[str, list[dict[str, Any]]] | None = None,
    invalid_captures: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    scenario_index = scenario_index or {}
    review_index = review_index or {}
    isa_index = isa_index or {}
    invalid_captures = invalid_captures or []
    rows = []
    for capture in captures:
        path = str(capture.get("_path", ""))
        device = capture.get("device") if isinstance(capture.get("device"), dict) else {}
        rows.append(
            build_row(
                capture,
                lookup_metadata(scenario_index, path),
                lookup_metadata(review_index, path),
                lookup_isa_resources(isa_index, capture.get("backend_selected"), device.get("gcn_arch")),
            )
        )
    bottleneck_counts = Counter(str(row.get("bottleneck_class")) for row in rows)
    scenario_counts = Counter(str(row.get("scenario_family") or "unlabeled") for row in rows)
    backend_counts = Counter(str(row.get("backend")) for row in rows)
    isa_report_paths = sorted(
        {
            str(path)
            for row in rows
            for path in (row.get("isa_report_paths") or [])
            if path is not None
        }
    )
    roofline_priority = build_roofline_priority(rows)
    gpu_roofline_priority = build_roofline_priority(gpu_evidence_rows(rows))
    return {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "capture_count": len(rows),
        "summary": {
            "bottleneck_counts": dict(sorted(bottleneck_counts.items())),
            "scenario_counts": dict(sorted(scenario_counts.items())),
            "backend_counts": dict(sorted(backend_counts.items())),
            "isa_report_count": len(isa_report_paths),
            "captures_with_isa_resources": sum(1 for row in rows if row.get("isa_report_count", 0) > 0),
            "roofline_priority": roofline_priority,
            "gpu_roofline_priority": gpu_roofline_priority,
            "skipped_invalid_capture_count": len(invalid_captures),
        },
        "skipped_invalid_captures": invalid_captures,
        "rows": rows,
    }


