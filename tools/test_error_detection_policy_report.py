#!/usr/bin/env python3
"""Self-test research-only error-detection policy report gates."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import error_detection_policy_report
from test_starfoundry_reports import starfoundry_capture


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def error_detecting_capture() -> dict:
    capture = starfoundry_capture()
    capture["scenario_metadata"] = {
        "family": "error-detecting-fast-path",
        "name": "freivalds-product-check-research",
        "promotion_eligibility": "proxy_evidence_only",
        "metadata": {
            "promotion_scope": "proxy_evidence_only",
            "verification_role": "research_only_error_detection",
        },
    }
    capture["error_detection_policy"] = {
        "enabled": True,
        "policy": "freivalds_two_round_product_check_research",
        "mode": "probabilistic_product_check",
        "verification_basis": "same_shape_same_seed_fixture_cpu_reference_final_compare",
        "false_negative_policy": "bounded_by_recorded_rounds_seed_and_reference_field_not_default_exact_api",
        "verification_rounds": 2,
        "rng_seed_recorded": True,
        "final_exact_comparison_required": True,
        "final_exact_comparison_status": "checksum_recorded_reference_required",
        "research_only": True,
        "default_exact_api_unchanged": True,
        "runtime_routing_allowed": False,
        "cache_eligible": False,
        "promotion_eligible": False,
    }
    return capture


def write_review_report(
    path: Path,
    capture_path: Path,
    *,
    include_cpu: bool = True,
    candidate_promotable: bool = False,
    cache_status: str = "not_eligible",
) -> None:
    candidates = []
    if include_cpu:
        candidates.append(
            {
                "backend": "cpu-reference",
                "capture": str(path.parent / "cpu-reference.json"),
                "release_review_capture": True,
                "promotable": False,
                "promotion_blockers": ["not_accelerator_backend"],
                "cache_write_status": "not_eligible",
                "scenario_promotion_scope": "proxy_evidence_only",
            }
        )
    candidates.extend(
        [
            {
                "backend": "hip-direct",
                "capture": str(path.parent / "hip-direct.json"),
                "release_review_capture": True,
                "promotable": False,
                "promotion_blockers": ["not_accelerator_backend"],
                "cache_write_status": "not_eligible",
                "scenario_promotion_scope": "proxy_evidence_only",
            },
            {
                "backend": "ck",
                "capture": str(capture_path),
                "release_review_capture": True,
                "promotable": candidate_promotable,
                "promotion_blockers": [] if candidate_promotable else ["scenario_scope_not_autotune_promotable"],
                "cache_write_status": cache_status,
                "scenario_promotion_scope": "proxy_evidence_only",
            },
        ]
    )
    write_json(
        path,
        {
            "schema_version": 1,
            "review_mode": "release",
            "groups": [
                {
                    "review_mode": "release",
                    "release_review_satisfied": True,
                    "required_baselines": ["cpu-reference", "hip-direct"],
                    "missing_required_baselines": [] if include_cpu else ["cpu-reference"],
                    "duplicate_backends": [],
                    "candidates": candidates,
                }
            ],
        },
    )


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        capture_path = tmp / "error-detecting.json"
        write_json(capture_path, error_detecting_capture())
        write_review_report(tmp / "review_report.json", capture_path)
        report = error_detection_policy_report.build_report([tmp])
        assert report["schema"] == "rns8_error_detection_policy_report_v1"
        assert report["rank39_gate_complete"] is True, json.dumps(report["blocker_counts"], indent=2)
        row = report["rows"][0]
        assert row["ready"] is True
        assert row["mode"] == "probabilistic_product_check"
        assert row["review_cpu_reference_present"] is True

        no_cpu = tmp / "no-cpu"
        no_cpu.mkdir()
        no_cpu_capture = no_cpu / "error-detecting.json"
        write_json(no_cpu_capture, error_detecting_capture())
        write_review_report(no_cpu / "review_report.json", no_cpu_capture, include_cpu=False)
        no_cpu_report = error_detection_policy_report.build_report([no_cpu])
        assert no_cpu_report["rank39_gate_complete"] is False
        assert "cpu_reference_baseline_missing" in no_cpu_report["rows"][0]["blockers"]
        assert "cpu_reference_marked_missing_required_baseline" in no_cpu_report["rows"][0]["blockers"]

        cache_eligible = tmp / "cache-eligible"
        cache_eligible.mkdir()
        cache_capture = cache_eligible / "error-detecting.json"
        write_json(cache_capture, error_detecting_capture())
        write_review_report(
            cache_eligible / "review_report.json",
            cache_capture,
            candidate_promotable=True,
            cache_status="eligible_after_review",
        )
        cache_report = error_detection_policy_report.build_report([cache_eligible])
        assert cache_report["rank39_gate_complete"] is False
        assert "error_detection_capture_promotable_in_review" in cache_report["rows"][0]["blockers"]
        assert "error_detection_capture_cache_write_eligible" in cache_report["rows"][0]["blockers"]

        public_scope = tmp / "public-scope"
        public_scope.mkdir()
        public_capture = error_detecting_capture()
        public_capture["scenario_metadata"]["promotion_eligibility"] = "release_review_candidate"
        public_capture["scenario_metadata"]["metadata"]["promotion_scope"] = "release_review_candidate"
        public_capture_path = public_scope / "error-detecting.json"
        write_json(public_capture_path, public_capture)
        write_review_report(public_scope / "review_report.json", public_capture_path)
        public_report = error_detection_policy_report.build_report([public_scope])
        assert public_report["rank39_gate_complete"] is False
        assert "error_detection_scenario_not_research_only" in public_report["rows"][0]["blockers"]

    print("error detection policy report self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
