#!/usr/bin/env python3
"""Review tile-shape benchmark captures with baseline and resource gates."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from report_capture_inputs import load_report_captures
from diagnostic_companion import load_bindings
from evidence_database_lib.isa import isa_index_from_reports, lookup_isa_resources
from observation_cohorts import phase_median
from report_comparison import ComparisonPolicy, ReportCohorts
from target_qualification import EvidenceStore, file_digest


DEFAULT_OUT_DIR = Path("temp") / "tile-shape-reports"
DEFAULT_TILE_M = 128
DEFAULT_TILE_N = 128
TILE_POLICY = ComparisonPolicy(
    "physical_tile_and_k_block_completed_output",
    work_exclusions=(
        "tile_m",
        "tile_n",
        "k_block_size",
        "tile_shape_variant.name",
        "tile_shape_variant.tile_m",
        "tile_shape_variant.tile_n",
        "tile_shape_variant.tile_k",
        "tile_shape_variant.k_block_policy",
        "schedule_metadata.tile_rows",
        "schedule_metadata.tile_cols",
        "schedule_metadata.tile_count",
        "export_variant.selector_key",
    ),
    amortize_prepack_setup=True,
)


def _number(value: Any) -> float | None:
    return float(value) if type(value) in (int, float) and math.isfinite(value) else None


def _median(capture: dict[str, Any] | None, phase: str) -> float | None:
    return phase_median(capture, phase) if capture is not None else None


def _target_id(capture: dict[str, Any]) -> str | None:
    for source in (capture.get("target_variant"), capture.get("device"), capture):
        if isinstance(source, dict) and isinstance(source.get("target_id"), str):
            return source["target_id"]
        if isinstance(source, dict) and isinstance(source.get("target_arch"), str):
            return source["target_arch"]
        if isinstance(source, dict) and isinstance(source.get("gcn_arch"), str):
            return source["gcn_arch"]
    return None


def _backend(capture: dict[str, Any]) -> str:
    return str(capture.get("backend_selected") or capture.get("backend_requested") or "")


def _tile_variant(capture: dict[str, Any]) -> dict[str, Any]:
    value = capture.get("tile_shape_variant")
    return value if isinstance(value, dict) else {}


def _backend_metadata(capture: dict[str, Any]) -> dict[str, Any]:
    value = capture.get("backend_metadata")
    return value if isinstance(value, dict) else {}


def _schedule(capture: dict[str, Any]) -> dict[str, Any]:
    value = capture.get("schedule_metadata")
    return value if isinstance(value, dict) else {}


def _is_default_tile_anchor(capture: dict[str, Any]) -> bool:
    variant = _tile_variant(capture)
    return (
        _backend(capture) == "hip-direct"
        and capture.get("tile_m") == DEFAULT_TILE_M
        and capture.get("tile_n") == DEFAULT_TILE_N
        and variant.get("k_block_policy", "auto") == "auto"
        and variant.get("name", "default") in {"default", "direct-hip-default-128x128"}
    )


def _is_tile_candidate(capture: dict[str, Any]) -> bool:
    variant = _tile_variant(capture)
    k_block_policy = variant.get("k_block_policy", "auto")
    non_default_tile = (
        capture.get("tile_m") != DEFAULT_TILE_M or capture.get("tile_n") != DEFAULT_TILE_N
    )
    non_default_k_block = k_block_policy not in {"", "auto"}
    return (
        _backend(capture) == "hip-direct"
        and bool(variant)
        and variant.get("name") not in {None, "", "default", "direct-hip-default-128x128"}
        and (non_default_tile or non_default_k_block)
    )


def _resource_report_key(capture: dict[str, Any]) -> str:
    value = _tile_variant(capture).get("resource_report_key")
    return value if isinstance(value, str) else ""


def _autotune_key(capture: dict[str, Any]) -> str:
    value = _backend_metadata(capture).get("autotune_key")
    return value if isinstance(value, str) else ""


def _autotune_key_has_tile_identity(capture: dict[str, Any]) -> bool:
    key = f";{_autotune_key(capture)};"
    tile_k = _tile_variant(capture).get("tile_k", capture.get("k_block_size"))
    selected_kernel = capture.get("selected_kernel")
    required = [
        f";tile_m={capture.get('tile_m')};",
        f";tile_n={capture.get('tile_n')};",
        f";k_block_size={tile_k};",
    ]
    if isinstance(selected_kernel, str) and selected_kernel:
        required.append(f";kernel={selected_kernel};")
    return all(item in key for item in required)


def _selected_kernel_has_tile_report_identity(capture: dict[str, Any]) -> bool:
    variant = _tile_variant(capture)
    selected_kernel = capture.get("selected_kernel")
    resource_key = f";{_resource_report_key(capture)};"
    if not isinstance(selected_kernel, str) or not selected_kernel:
        return False
    return (
        variant.get("selected_kernel_identity") == selected_kernel
        and f";kernel={selected_kernel};" in resource_key
        and f";tile_m={capture.get('tile_m')};" in resource_key
        and f";tile_n={capture.get('tile_n')};" in resource_key
        and f";tile_k={variant.get('tile_k', capture.get('k_block_size'))};" in resource_key
    )


@dataclass
class TileResources:
    """Own resource inputs once; capture filenames are never qualification keys."""

    store: EvidenceStore = field(default_factory=EvidenceStore)
    reports: list[tuple[Path, dict]] = field(default_factory=list)
    inventory: list[dict] = field(default_factory=list)
    index: dict = field(default_factory=dict)

    def document(self, reference: str | dict, parent: Path) -> tuple[Path, dict]:
        if isinstance(reference, str):
            path = (parent / reference).resolve(strict=True)
            reference = dict(path=str(path), sha256=file_digest(path))
        return self.store.json(reference, parent)


def load_resource_manifest(path: Path | None) -> TileResources:
    resources = TileResources()
    if path is None:
        return resources
    manifest_path, data = resources.document(str(path.resolve()), path.parent)
    if isinstance(data.get("captures"), list):
        entries = data["captures"]
    else:
        entries = list(data.values())
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("tile resource entries must be objects")
        if isinstance(entry.get("resource_summary"), dict):
            resources.inventory.append(dict(source="unbound_inline_inventory", record=entry))
        for key in ("gpu_counter_report", "counter_report", "isa_summary", "resource_report"):
            reference = entry.get(key)
            if reference is None:
                continue
            report_path, report = resources.document(reference, manifest_path.parent)
            resources.inventory.append(
                dict(
                    path=str(report_path),
                    sha256=resources.store.hashes[report_path],
                    resource_summary=report.get("resource_summary"),
                )
            )
            if isinstance(report.get("symbols"), list):
                resources.reports.append((report_path, report))
            # Counter reports may point at original immutable ISA inputs. Their
            # flattened resource summaries and precomputed association verdicts
            # never substitute for rechecking the selected-symbol binding.
            for summary in report.get("isa_summaries") or []:
                if not isinstance(summary, dict):
                    raise ValueError("ISA input summaries must be objects")
                isa_path, isa_report = resources.document(
                    dict(path=summary.get("path"), sha256=summary.get("sha256")),
                    report_path.parent,
                )
                resources.reports.append((isa_path, isa_report))
    resources.index = isa_index_from_reports(resources.reports)
    return resources


def _resource_for_capture(capture: dict, resources: TileResources) -> dict:
    association = lookup_isa_resources(
        resources.index, _backend(capture), _target_id(capture), capture
    )
    selected = association["isa_resource_reports"]
    fields = {
        "vgpr": "isa_vgpr_count",
        "sgpr": "isa_sgpr_count",
        "lds_bytes": "isa_lds_bytes",
        "scratch_bytes": "isa_scratch_bytes",
        "occupancy": "isa_occupancy",
    }
    summary = {}
    for destination, source in fields.items():
        values = [_number(report.get(source)) for report in selected]
        # One missing symbol cannot borrow another symbol's resource metadata.
        if values and all(value is not None and value >= 0 for value in values):
            summary[destination] = min(values) if destination == "occupancy" else max(values)
        else:
            summary[destination] = None
    return dict(
        resource_summary=summary,
        selected_isa=association,
        evidence_status=dict(
            isa_resource_status="selected_static_binding"
            if association["isa_association"]["qualified"]
            else "unqualified",
            profiler_counter_status="not_used_as_static_resource_proof",
            occupancy_scope="static_reported_value_not_measured_wave_occupancy",
        ),
    )


def _resource_status(resource: dict) -> dict:
    summary = resource.get("resource_summary") or {}
    selected = resource.get("selected_isa") or {}
    association = selected.get("isa_association") or {}
    present = association.get("qualified") is True
    registers = all(
        _number(summary.get(key)) is not None and summary[key] > 0 for key in ("vgpr", "sgpr")
    )
    lds = _number(summary.get("lds_bytes")) is not None and summary["lds_bytes"] >= 0
    occupancy = _number(summary.get("occupancy")) is not None and summary["occupancy"] > 0
    return dict(
        resource_summary=summary,
        evidence_status=resource.get("evidence_status", {}),
        selected_isa=selected,
        resource_evidence_present=present,
        register_pressure_present=registers,
        lds_signal_present=lds,
        occupancy_present=occupancy,
        complete_for_promotion=present and registers and lds and occupancy,
    )


def row_for_candidate(
    candidate: dict, direct: dict | None, resources: TileResources, cohorts: ReportCohorts
) -> dict:
    variant = _tile_variant(candidate)
    raw_observations = cohorts.raw_observations(candidate)
    resource_observations = [
        dict(
            capture_path=raw.get("_path"),
            status=_resource_status(_resource_for_capture(raw, resources)),
        )
        for raw in raw_observations
    ]
    resource_status = (
        dict(resource_observations[0]["status"]) if resource_observations else _resource_status({})
    )
    comparison = cohorts.compare(candidate, direct, TILE_POLICY)
    numerical = cohorts.numerical(candidate)
    direct_numerical = cohorts.numerical(direct) if direct is not None else None
    core_blockers = list(comparison["blockers"])
    resource_blockers = []
    if not raw_observations:
        core_blockers.append("candidate_process_observations_missing")
    if direct is None:
        core_blockers.append("missing_default_direct_hip_tile_baseline")
    if not all(_autotune_key_has_tile_identity(raw) for raw in raw_observations):
        core_blockers.append("autotune_key_missing_tile_identity")
    if not all(_selected_kernel_has_tile_report_identity(raw) for raw in raw_observations):
        core_blockers.append("selected_kernel_missing_tile_report_identity")
    for raw in raw_observations:
        raw_variant = _tile_variant(raw)
        if raw_variant.get("k_block_policy", "auto") not in {"", "auto"}:
            if raw_variant.get("split_k_mode") != "single_gpu_no_split_k":
                core_blockers.append("k_block_split_mode_not_single_gpu")
            if not isinstance(
                raw_variant.get("accumulator_safety_key"), str
            ) or not raw_variant.get("accumulator_safety_key"):
                core_blockers.append("missing_accumulator_safety_key")
            if not isinstance(
                raw_variant.get("resource_report_required"), str
            ) or not raw_variant.get("resource_report_required"):
                core_blockers.append("missing_resource_report_requirement")
    for signal, blocker in (
        ("resource_evidence_present", "missing_selected_static_isa_resource_evidence"),
        ("register_pressure_present", "missing_vgpr_sgpr_signal"),
        ("lds_signal_present", "missing_lds_allocation_signal"),
        ("occupancy_present", "missing_occupancy_signal"),
    ):
        if not resource_observations or any(
            not item["status"][signal] for item in resource_observations
        ):
            resource_blockers.append(blocker)
    for signal in (
        "resource_evidence_present",
        "register_pressure_present",
        "lds_signal_present",
        "occupancy_present",
        "complete_for_promotion",
    ):
        resource_status[signal] = bool(resource_observations) and all(
            item["status"][signal] for item in resource_observations
        )
    speedup = comparison["speedup"]
    margin = comparison["required_speedup_margin"]
    blockers = core_blockers + resource_blockers
    if core_blockers:
        decision = "keep experimental"
    elif speedup is not None and margin is not None and speedup < 1 / margin:
        decision = "drop/deprioritize"
    elif resource_blockers:
        decision = "keep experimental"
    elif speedup is not None and margin is not None and speedup > margin:
        decision = "promote locally"
    else:
        decision = "keep experimental"
        blockers.append("no_setup_inclusive_default_tile_win_above_observed_noise")

    return {
        "capture_path": candidate.get("_path"),
        "semantics": candidate.get("semantics"),
        "backend": _backend(candidate),
        "target_id": _target_id(candidate),
        "shape": {"m": candidate.get("m"), "n": candidate.get("n"), "k": candidate.get("k")},
        "variant_name": variant.get("name", "legacy"),
        "tile_m": variant.get("tile_m", candidate.get("tile_m")),
        "tile_n": variant.get("tile_n", candidate.get("tile_n")),
        "tile_k": variant.get("tile_k", candidate.get("k_block_size")),
        "k_block_policy": variant.get("k_block_policy", "auto"),
        "split_k_mode": variant.get("split_k_mode", "single_gpu_no_split_k"),
        "shape_family_bucket": variant.get("shape_family_bucket"),
        "selected_kernel": candidate.get("selected_kernel"),
        "autotune_key": _autotune_key(candidate),
        "resource_report_key": _resource_report_key(candidate),
        "candidate_median_end_to_end_us": _median(candidate, "end_to_end"),
        "candidate_median_gemm_us": _median(candidate, "rns_gemm"),
        "candidate_median_pack_us": _median(candidate, "pack"),
        "candidate_median_export_us": _median(candidate, "crt_export"),
        "default_direct_hip_baseline_path": direct.get("_path") if direct else None,
        "default_direct_hip_median_end_to_end_us": _median(direct, "end_to_end")
        if direct
        else None,
        "speedup_vs_default_direct_hip_tile": speedup,
        "observation": cohorts.metadata(candidate, TILE_POLICY),
        "comparison": comparison,
        "resource_observations": resource_observations,
        "autotune_key_has_tile_identity": _autotune_key_has_tile_identity(candidate),
        "selected_kernel_has_tile_report_identity": _selected_kernel_has_tile_report_identity(
            candidate
        ),
        "resource_status": resource_status,
        "numerical_reference": numerical.to_json(),
        "direct_numerical_reference": direct_numerical.to_json() if direct_numerical else None,
        "promotion_eligible": decision == "promote locally",
        "promotion_blockers": blockers,
        "decision": decision,
    }


def build_report(
    paths: list[Path],
    resource_manifest: TileResources | None = None,
    *,
    diagnostic_bindings: dict | None = None,
) -> dict:
    return build_report_from_captures(
        load_report_captures(paths), resource_manifest, diagnostic_bindings=diagnostic_bindings
    )


def build_report_from_captures(
    captures: list[dict],
    resource_manifest: TileResources | None = None,
    *,
    diagnostic_bindings: dict | None = None,
) -> dict:
    resources = resource_manifest or TileResources()
    cohorts = ReportCohorts(captures, diagnostic_bindings)
    candidates = [view for view in cohorts.observations if _is_tile_candidate(view)]
    anchors = [view for view in cohorts.observations if _is_default_tile_anchor(view)]
    rows = [
        row_for_candidate(
            candidate, cohorts.best(anchors, candidate, TILE_POLICY), resources, cohorts
        )
        for candidate in candidates
    ]
    resources.store.finish()
    grouped_rows: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        shape = row["shape"]
        grouped_rows[
            (row["semantics"], row["target_id"], shape["m"], shape["n"], shape["k"])
        ].append(row)
    return {
        "schema": "rns8_tile_shape_report_v3",
        "capture_count": len(captures),
        "candidate_count": len(candidates),
        "observation_count": len(cohorts.observations),
        "observations": [cohorts.metadata(view, TILE_POLICY) for view in cohorts.observations],
        "resource_inventory": resources.inventory,
        "default_direct_hip_anchor_count": len(anchors),
        "cpu_anchor_count": sum(
            _backend(view) in {"cpu", "cpu-reference", "wrap64-byte-limb"}
            for view in cohorts.observations
        ),
        "policy": (
            "Tile review requires independent process cohorts, full CPU references, hooks-disabled timings "
            "with per-process diagnostic companions, same-source/native-device default Direct-HIP anchors, "
            "setup-inclusive wins above observed noise, and artifact-bound selected static ISA resource "
            "records for every process. Resource inventory alone cannot qualify a tile; static occupancy "
            "does not prove dynamic occupancy. Physical tile differences do not relax input/range/proof contracts."
        ),
        "decision_counts": {
            decision: sum(1 for row in rows if row["decision"] == decision)
            for decision in ["promote locally", "keep experimental", "drop/deprioritize"]
        },
        "groups": [
            {
                "key": key,
                "rows": sorted(
                    value, key=lambda row: row.get("candidate_median_end_to_end_us") or float("inf")
                ),
            }
            for key, value in sorted(grouped_rows.items(), key=lambda item: str(item[0]))
        ],
        "rows": rows,
    }


def write_outputs(report: dict[str, Any], out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "tile-shape-report.json"
    md_path = out_dir / "tile-shape-report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Tile Shape Report",
        "",
        f"- capture_count: `{report['capture_count']}`",
        f"- candidate_count: `{report['candidate_count']}`",
        f"- policy: {report['policy']}",
        "",
        "| Capture | Shape | Tile | Kernel | Direct Speedup | Resource | Decision | Blockers |",
        "|---|---:|---:|---|---:|---|---|---|",
    ]
    for row in report["rows"]:
        speedup = row.get("speedup_vs_default_direct_hip_tile")
        status = row.get("resource_status") or {}
        resource = "complete" if status.get("complete_for_promotion") else "incomplete"
        lines.append(
            "| {capture} | {shape} | {tile} | `{kernel}` | {speedup} | {resource} | {decision} | {blockers} |".format(
                capture=Path(str(row.get("capture_path") or "unknown")).name,
                shape="{m}x{n}x{k}".format(**row["shape"]),
                tile=f"{row.get('tile_m')}x{row.get('tile_n')}x{row.get('tile_k')}",
                kernel=row.get("selected_kernel"),
                speedup=f"{float(speedup):.3f}x" if isinstance(speedup, (int, float)) else "n/a",
                resource=resource,
                decision=row["decision"],
                blockers=", ".join(row.get("promotion_blockers") or []) or "none",
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", type=Path, nargs="+")
    parser.add_argument("--resource-manifest", type=Path)
    parser.add_argument("--diagnostic-binding", type=Path, action="append", default=[])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    report = build_report(
        args.captures,
        load_resource_manifest(args.resource_manifest),
        diagnostic_bindings=load_bindings(args.diagnostic_binding),
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        for label, path in write_outputs(report, args.out_dir).items():
            print(f"{label}: {path}")
    if args.require_complete and (
        report["candidate_count"] == 0
        or any(row["decision"] == "keep experimental" for row in report["rows"])
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
