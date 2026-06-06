#!/usr/bin/env python3
"""Self-test GPU CRT/export reconstruction variant report classification."""

from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path

import reconstruction_export_report
from test_export_selector_report import reviewable_fixed_limb_capture


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def set_medians(capture: dict, end_to_end: float, export: float) -> dict:
    end_value = int(end_to_end)
    export_value = int(export)
    capture["timing_summary_us"]["end_to_end"] = {
        "avg": float(end_value),
        "median": end_value,
        "p95": end_value,
    }
    capture["timing_summary_us"]["crt_export"] = {
        "avg": float(export_value),
        "median": export_value,
        "p95": export_value,
    }
    capture["avg_end_to_end_us"] = end_to_end
    capture["avg_crt_export_us"] = export
    capture["raw_timings_us"]["end_to_end"] = [end_value for _ in range(capture["repeats"])]
    capture["raw_timings_us"]["crt_export"] = [export_value for _ in range(capture["repeats"])]
    return capture


def as_default(capture: dict) -> dict:
    capture = copy.deepcopy(capture)
    export = capture["export_variant"]
    export["name"] = "default"
    export["source"] = "current_backend_export_path"
    export["d2h_policy"] = "host_ld_padded"
    export["promotion_eligible"] = True
    export["promotion_blocker"] = None
    export["selector_key"] = export["selector_key"].replace(
        "d2h_policy=compact_contiguous",
        "d2h_policy=host_ld_padded",
    )
    return capture


def as_candidate(capture: dict, name: str, *, reconstruction: str = "default_garner") -> dict:
    capture = copy.deepcopy(capture)
    export = capture["export_variant"]
    export["name"] = name
    export["promotion_eligible"] = False
    export["promotion_blocker"] = "experimental_export_variant"
    if name == "compact-d2h-export-candidate":
        export["d2h_policy"] = "compact_contiguous"
        export["selector_key"] = export["selector_key"].replace(
            "d2h_policy=host_ld_padded",
            "d2h_policy=compact_contiguous",
        )
    if name == "status-elided-exact-proof-export-candidate":
        export["selector_status_policy"] = "none"
        export["status_elision_reason"] = "exact_wide_requested_limb_count_covers_range_status"
    if name == "prefix20-fixed-export-candidate":
        export["limb_count"] = 8
        export["prefix_contract"] = "prefix=20;min_selected=20;max_selected=20;groups=1"
        capture["reconstruction_variant"]["prefix_count"] = 20
    if reconstruction == "tree_crt_candidate":
        capture["reconstruction_variant"] = {
            "name": "tree_crt_candidate",
            "family": "tree_pair_crt_fixed_prefix",
            "prefix_count": capture.get("selected_prefix", capture["prefix"]),
            "kernel_identity": export["selected_kernel"],
            "controller": "benchmark_metadata_only",
            "promotion_eligible": False,
            "promotion_blocker": "experimental_reconstruction_variant",
        }
        export["selector_key"] += ";reconstruction_variant=tree_crt_candidate"
    return capture


def write_review_report(path: Path, capture_paths: list[Path]) -> None:
    write_json(
        path,
        {
            "review_mode": "release",
            "groups": [
                {
                    "review_mode": "release",
                    "release_review_satisfied": True,
                    "missing_required_baselines": [],
                    "duplicate_backends": [],
                    "contract_key": "fixture",
                    "candidates": [
                        {
                            "capture": str(capture_path),
                            "promotable": True,
                            "promotion_reason": "fixture_release_review",
                            "promotion_blockers": [],
                            "cache_write_status": "not_requested",
                        }
                        for capture_path in capture_paths
                    ],
                }
            ],
        },
    )


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        base = as_default(reviewable_fixed_limb_capture())

        captures: list[tuple[str, dict]] = [
            ("baseline.json", set_medians(base, 100.0, 40.0)),
            (
                "compact.json",
                set_medians(as_candidate(base, "compact-d2h-export-candidate"), 110.0, 38.0),
            ),
            (
                "status.json",
                set_medians(as_candidate(base, "status-elided-exact-proof-export-candidate"), 105.0, 42.0),
            ),
            (
                "tree.json",
                set_medians(as_candidate(base, "tree-crt-export-candidate", reconstruction="tree_crt_candidate"), 90.0, 20.0),
            ),
        ]
        prefix_base = as_default(base)
        prefix_base["export_variant"]["limb_count"] = 8
        prefix_base["reconstruction_variant"]["prefix_count"] = 20
        captures.extend(
            [
                ("prefix-baseline.json", set_medians(prefix_base, 100.0, 50.0)),
                (
                    "prefix-candidate.json",
                    set_medians(as_candidate(prefix_base, "prefix20-fixed-export-candidate"), 98.0, 45.0),
                ),
            ]
        )

        capture_paths: list[Path] = []
        for name, capture in captures:
            path = tmp / name
            write_json(path, capture)
            capture_paths.append(path)
        write_review_report(tmp / "review_report.json", capture_paths)

        report = reconstruction_export_report.build_report([tmp])
        assert report["schema"] == "rns8_reconstruction_export_report_v1"
        assert report["rank33_classification_complete"] is True, json.dumps(
            {
                "summary": report["variant_class_summary"],
                "skipped": report["skipped_json"],
                "comparison_count": report["comparison_count"],
            },
            indent=2,
        )
        assert report["variant_class_summary"]["compact_d2h_host_scatter"]["ready"] is True
        assert report["variant_class_summary"]["status_elided_exact_proof"]["ready"] is True
        assert report["variant_class_summary"]["prefix20_fixed_export"]["ready"] is True
        assert report["variant_class_summary"]["tree_crt_reconstruction"]["ready"] is True
        dispositions = {item["variant_class"]: item["disposition"] for item in report["comparisons"]}
        assert dispositions["compact_d2h_host_scatter"] == "drop/deprioritize"
        assert dispositions["status_elided_exact_proof"] == "drop/deprioritize"
        assert dispositions["prefix20_fixed_export"] == "keep experimental"
        assert dispositions["tree_crt_reconstruction"] == "keep experimental"

        no_review_report = reconstruction_export_report.build_report([capture_paths[0], capture_paths[2]])
        assert no_review_report["rank33_classification_complete"] is False
        blocked = no_review_report["comparisons"][0]
        assert blocked["disposition"] == "blocked"
        assert "candidate_release_review_not_ready" in blocked["blockers"]

    print("reconstruction export report self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
