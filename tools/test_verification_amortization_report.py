#!/usr/bin/env python3
"""Self-test verification amortization report gates."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import verification_amortization_report
from test_starfoundry_reports import starfoundry_capture


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def amortized_capture() -> dict:
    capture = starfoundry_capture()
    capture["scenario_metadata"] = {
        "family": "fhe-lattice-proxy-starfoundry",
        "name": "ckks-linear-layer-final-export",
        "promotion_eligibility": "proxy_evidence_only",
        "metadata": {"promotion_scope": "proxy_evidence_only"},
    }
    capture["verification_amortization"] = {
        "enabled": True,
        "policy": "reuse_shape_seed_reference_inputs",
        "reused_reference_structure": "shape_seed_semantic_reference_inputs",
        "final_exact_comparison_required": True,
        "final_exact_comparison_status": "checksum_recorded_reference_required",
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
                "backend": "hipblaslt",
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
        capture_path = tmp / "amortized.json"
        write_json(capture_path, amortized_capture())
        write_review_report(tmp / "review_report.json", capture_path)
        report = verification_amortization_report.build_report([tmp])
        assert report["schema"] == "rns8_verification_amortization_report_v1"
        assert report["rank38_gate_complete"] is True, json.dumps(report["blocker_counts"], indent=2)
        row = report["rows"][0]
        assert row["ready"] is True
        assert row["review_cpu_reference_present"] is True
        assert row["review_candidate_cache_write_status"] == "not_eligible"

        no_cpu = tmp / "no-cpu"
        no_cpu.mkdir()
        no_cpu_capture = no_cpu / "amortized.json"
        write_json(no_cpu_capture, amortized_capture())
        write_review_report(no_cpu / "review_report.json", no_cpu_capture, include_cpu=False)
        no_cpu_report = verification_amortization_report.build_report([no_cpu])
        assert no_cpu_report["rank38_gate_complete"] is False
        assert "cpu_reference_baseline_missing" in no_cpu_report["rows"][0]["blockers"]
        assert "cpu_reference_marked_missing_required_baseline" in no_cpu_report["rows"][0]["blockers"]

        cache_eligible = tmp / "cache-eligible"
        cache_eligible.mkdir()
        cache_capture = cache_eligible / "amortized.json"
        write_json(cache_capture, amortized_capture())
        write_review_report(
            cache_eligible / "review_report.json",
            cache_capture,
            candidate_promotable=True,
            cache_status="eligible_after_review",
        )
        cache_report = verification_amortization_report.build_report([cache_eligible])
        assert cache_report["rank38_gate_complete"] is False
        assert "amortized_capture_promotable_in_review" in cache_report["rows"][0]["blockers"]
        assert "amortized_capture_cache_write_eligible" in cache_report["rows"][0]["blockers"]

        public_scope = tmp / "public-scope"
        public_scope.mkdir()
        public_capture = amortized_capture()
        public_capture["scenario_metadata"]["promotion_eligibility"] = "release_review_candidate"
        public_capture["scenario_metadata"]["metadata"]["promotion_scope"] = "release_review_candidate"
        public_capture_path = public_scope / "amortized.json"
        write_json(public_capture_path, public_capture)
        write_review_report(public_scope / "review_report.json", public_capture_path)
        public_report = verification_amortization_report.build_report([public_scope])
        assert public_report["rank38_gate_complete"] is False
        assert "verification_amortization_scenario_not_tooling_only" in public_report["rows"][0]["blockers"]

    print("verification amortization report self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
