#!/usr/bin/env python3
"""Close out FHE/lattice proxy workload captures without compatibility claims."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from benchmark_sweep_lib.capture_metadata import backend_id, median_phase
from report_capture_inputs import load_report_captures


DEFAULT_OUT_DIR = Path("temp") / "fhe-workload-reports"
PROMOTION_MARGIN = 1.10
REQUIRED_PROXY_OPERATIONS = {
    "key_switch_digit_aggregation",
    "rotation_batch",
    "modup_base_extension",
    "moddown_rescale",
    "ckks_rescale",
    "relinearization",
    "bootstrapping_stage",
    "tower_reuse",
    "dense_linear_transform",
}
FINAL_COMPARISON_STATUSES = {
    "checksum_recorded_reference_required",
    "reference_required",
    "exact_cpu_reference_compared",
    "passed",
}


def _scenario(capture: dict[str, Any]) -> dict[str, Any]:
    value = capture.get("scenario_metadata")
    return value if isinstance(value, dict) else {}


def _scenario_extra(capture: dict[str, Any]) -> dict[str, Any]:
    value = _scenario(capture).get("metadata")
    return value if isinstance(value, dict) else {}


def _proxy(capture: dict[str, Any]) -> dict[str, Any]:
    value = capture.get("workload_proxy")
    return value if isinstance(value, dict) else {}


def _verification(capture: dict[str, Any]) -> dict[str, Any]:
    value = capture.get("verification_amortization")
    return value if isinstance(value, dict) else {}


def _path(capture: dict[str, Any]) -> str:
    value = capture.get("_path")
    return str(value) if isinstance(value, str) else "<in-memory>"


def _operation(capture: dict[str, Any]) -> str:
    scenario = _scenario(capture)
    extra = _scenario_extra(capture)
    proxy = _proxy(capture)
    for value in (
        extra.get("proxy_operation"),
        extra.get("workflow_name"),
        proxy.get("label"),
        scenario.get("name"),
    ):
        if isinstance(value, str) and value:
            return value
    return "unlabeled"


def _group_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row["proxy_operation"]),
        str(row["tower_basis"]),
        str(row["reuse_mode"]),
        str(row["output_domain_requirement"]),
    )


def row_for_capture(capture: dict[str, Any]) -> dict[str, Any]:
    proxy = _proxy(capture)
    scenario = _scenario(capture)
    extra = _scenario_extra(capture)
    verification = _verification(capture)
    backend = backend_id(capture)
    blockers: list[str] = []
    family = proxy.get("family") or extra.get("algebra_family") or scenario.get("family")
    if family != "fhe_lattice_proxy":
        blockers.append("not_fhe_lattice_proxy")
    if proxy.get("compatibility_claim") is not False:
        blockers.append("fhe_compatibility_claim_forbidden")
    if scenario.get("promotion_eligibility") != "proxy_evidence_only":
        blockers.append("scenario_not_proxy_evidence_only")
    if extra.get("compatibility_claim") is True:
        blockers.append("scenario_metadata_compatibility_claim_forbidden")
    if verification.get("enabled") is not True:
        blockers.append("verification_amortization_not_enabled")
    if verification.get("final_exact_comparison_required") is not True:
        blockers.append("final_exact_comparison_not_required")
    if verification.get("final_exact_comparison_status") not in FINAL_COMPARISON_STATUSES:
        blockers.append("final_exact_comparison_status_missing")
    if verification.get("promotion_eligible") is not False:
        blockers.append("verification_amortization_promotable")
    release_gate = capture.get("release_gate") if isinstance(capture.get("release_gate"), dict) else {}
    if release_gate.get("cache_eligible") is True:
        blockers.append("release_gate_cache_eligible_for_proxy")
    operation = _operation(capture)
    row = {
        "path": _path(capture),
        "proxy_operation": operation,
        "family": family,
        "tower_basis": extra.get("tower_basis") or extra.get("tower_role") or proxy.get("tower_role") or "unspecified",
        "reuse_mode": extra.get("reuse_mode") or proxy.get("reuse_profile") or "none",
        "output_domain_requirement": proxy.get("output_domain_requirement")
        or scenario.get("output_domain")
        or "unknown",
        "verification_policy": verification.get("policy"),
        "verification_status": verification.get("final_exact_comparison_status"),
        "compatibility_claim": proxy.get("compatibility_claim"),
        "scenario_promotion_eligibility": scenario.get("promotion_eligibility"),
        "backend": backend,
        "target_id": capture.get("target_id"),
        "semantics": capture.get("semantics"),
        "shape": {"m": capture.get("m"), "n": capture.get("n"), "k": capture.get("k")},
        "median_end_to_end_us": median_phase(capture, "end_to_end"),
        "blockers": sorted(set(blockers)),
    }
    row["status"] = "blocked" if row["blockers"] else "proxy-ready"
    return row


def build_report(paths: list[Path]) -> dict[str, Any]:
    captures = load_report_captures(paths)
    rows = [row_for_capture(capture) for capture in captures]
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_group_key(row)].append(row)

    group_rows: list[dict[str, Any]] = []
    for key, grouped_rows in sorted(groups.items(), key=lambda item: str(item[0])):
        backends = sorted({str(row["backend"]) for row in grouped_rows})
        blockers = sorted({blocker for row in grouped_rows for blocker in row["blockers"]})
        if "cpu-reference" not in backends and "cpu" not in backends:
            blockers.append("cpu_reference_baseline_missing")
        cpu_rows = [row for row in grouped_rows if row["backend"] in {"cpu-reference", "cpu"}]
        gpu_rows = [row for row in grouped_rows if row["backend"] not in {"cpu-reference", "cpu"}]
        cpu_median = min(
            (
                float(row["median_end_to_end_us"])
                for row in cpu_rows
                if isinstance(row["median_end_to_end_us"], (int, float))
            ),
            default=None,
        )
        fastest_gpu = min(
            (row for row in gpu_rows if isinstance(row["median_end_to_end_us"], (int, float))),
            key=lambda row: float(row["median_end_to_end_us"]),
            default=None,
        )
        fastest_gpu_median = (
            float(fastest_gpu["median_end_to_end_us"])
            if fastest_gpu and isinstance(fastest_gpu.get("median_end_to_end_us"), (int, float))
            else None
        )
        speedup_vs_cpu = (
            cpu_median / fastest_gpu_median
            if cpu_median is not None and fastest_gpu_median is not None and fastest_gpu_median > 0
            else None
        )
        if blockers:
            promotion_decision = "blocked"
        elif speedup_vs_cpu is not None and speedup_vs_cpu >= PROMOTION_MARGIN:
            promotion_decision = "promote_local_dense_rns_workload_profile"
        else:
            promotion_decision = "keep_experimental_no_gpu_speedup"
        group_rows.append(
            {
                "proxy_operation": key[0],
                "tower_basis": key[1],
                "reuse_mode": key[2],
                "output_domain_requirement": key[3],
                "backends": backends,
                "row_count": len(grouped_rows),
                "fastest_gpu_backend": fastest_gpu.get("backend") if fastest_gpu else None,
                "fastest_gpu_median_end_to_end_us": fastest_gpu_median,
                "cpu_median_end_to_end_us": cpu_median,
                "speedup_vs_cpu": speedup_vs_cpu,
                "promotion_decision": promotion_decision,
                "promotion_margin_required": PROMOTION_MARGIN,
                "promotion_scope": "local_dense_rns_workload_profile_not_fhe_compatibility",
                "status": "blocked" if blockers else "proxy-ready",
                "blockers": sorted(set(blockers)),
                "rows": grouped_rows,
            }
        )

    seen_operations = {group["proxy_operation"] for group in group_rows}
    missing_operations = sorted(REQUIRED_PROXY_OPERATIONS - seen_operations)
    blocker_counts = Counter()
    for group in group_rows:
        for blocker in group["blockers"]:
            blocker_counts[blocker] += 1
    for operation in missing_operations:
        blocker_counts[f"missing_proxy_operation:{operation}"] += 1
    complete = bool(group_rows) and not missing_operations and not any(group["blockers"] for group in group_rows)
    promotable_profiles = [
        {
            "proxy_operation": group["proxy_operation"],
            "tower_basis": group["tower_basis"],
            "reuse_mode": group["reuse_mode"],
            "output_domain_requirement": group["output_domain_requirement"],
            "promoted_backend": group["fastest_gpu_backend"],
            "speedup_vs_cpu": group["speedup_vs_cpu"],
            "promotion_scope": group["promotion_scope"],
        }
        for group in group_rows
        if group.get("promotion_decision") == "promote_local_dense_rns_workload_profile"
    ]
    return {
        "schema": "rns8_fhe_lattice_workload_report_v2",
        "policy": "proxy_workloads_may_promote_local_dense_rns_profiles_no_fhe_library_compatibility_claim",
        "rank63_gate_complete": complete,
        "promotable_workload_profile_count": len(promotable_profiles),
        "promotable_workload_profiles": promotable_profiles,
        "required_proxy_operations": sorted(REQUIRED_PROXY_OPERATIONS),
        "missing_required_proxy_operations": missing_operations,
        "capture_count": len(rows),
        "group_count": len(group_rows),
        "blocker_counts": dict(sorted(blocker_counts.items())),
        "groups": group_rows,
    }


def write_report(report: dict[str, Any], out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "fhe-workload-report.json"
    md_path = out_dir / "fhe-workload-report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# FHE/Lattice Proxy Workload Report",
        "",
        f"- Policy: `{report['policy']}`",
        f"- Rank 63 gate complete: `{report['rank63_gate_complete']}`",
        f"- Captures: `{report['capture_count']}`",
        f"- Promotable workload profiles: `{report['promotable_workload_profile_count']}`",
        f"- Missing operations: `{', '.join(report['missing_required_proxy_operations']) or 'none'}`",
        "",
        "| operation | reuse | output domain | promoted backend | speedup vs CPU | status | blockers |",
        "| --- | --- | --- | --- | ---: | --- | --- |",
    ]
    for group in report["groups"]:
        speedup = group.get("speedup_vs_cpu")
        lines.append(
            "| `{operation}` | `{reuse}` | `{output}` | `{backend}` | `{speedup}` | `{status}` | `{blockers}` |".format(
                operation=group["proxy_operation"],
                reuse=group["reuse_mode"],
                output=group["output_domain_requirement"],
                backend=group.get("fastest_gpu_backend") or "none",
                speedup=f"{speedup:.2f}x" if isinstance(speedup, (int, float)) else "n/a",
                status=group["status"],
                blockers=", ".join(group["blockers"]) or "none",
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", type=Path, nargs="+")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    report = build_report(args.captures)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        for label, path in write_report(report, args.out_dir).items():
            print(f"{label}: {path}")
    if args.require_complete and not report["rank63_gate_complete"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
