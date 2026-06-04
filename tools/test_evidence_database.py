#!/usr/bin/env python3
"""Self-test the RNS8 evidence database builder."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import evidence_database


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "benchmark_schema"


def main() -> int:
    hip_capture = FIXTURE_DIR / "v4_bounded_i64_hipblaslt.json"
    ck_capture = FIXTURE_DIR / "v4_bounded_i64_ck.json"
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        scenario_manifest = tmp / "scenario_manifest.json"
        scenario_manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "entries": [
                        {
                            "family": "repeated-b",
                            "name": "bounded-i64-512",
                            "capture_path": str(hip_capture),
                            "capture_name": hip_capture.name,
                            "output_domain": "host_export",
                            "evidence_scope": "same B operand reused across measured repeats",
                            "rationale": "test scenario join",
                            "pack_mode": "prepacked_reuse_b",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        review_report = tmp / "review_report.json"
        review_report.write_text(
            json.dumps(
                {
                    "groups": [
                        {
                            "review_mode": "release",
                            "release_review_satisfied": False,
                            "missing_required_baselines": ["cpu-reference"],
                            "candidates": [
                                {
                                    "capture": str(hip_capture),
                                    "promotable": False,
                                    "promotion_blockers": ["missing_required_baseline:cpu-reference"],
                                    "primary_loss_phase_vs_direct_hip": "rns_gemm",
                                    "speedup_vs_direct_hip": None,
                                    "speedup_vs_vector_alu": None,
                                }
                            ],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

        captures = evidence_database.load_validated_captures([hip_capture, ck_capture])
        database = evidence_database.build_database(
            captures,
            scenario_index=evidence_database.load_scenario_index([scenario_manifest]),
            review_index=evidence_database.load_review_index([review_report]),
        )

        assert database["schema_version"] == 1
        assert database["capture_count"] == 2
        assert database["summary"]["scenario_counts"]["repeated-b"] == 1
        assert database["summary"]["scenario_counts"]["unlabeled"] == 1
        hip_row = next(row for row in database["rows"] if row["capture_path"] == str(hip_capture))
        assert hip_row["scenario_family"] == "repeated-b"
        assert hip_row["output_domain"] == "host_export"
        assert hip_row["promotion_blockers"] == ["missing_required_baseline:cpu-reference"]
        assert hip_row["estimated_ops"] > 0
        assert hip_row["estimated_bytes"] > 0
        assert hip_row["arithmetic_intensity_ops_per_byte"] > 0
        assert hip_row["bottleneck_class"] in {
            "compute_bound",
            "export_bound",
            "launch_or_api_bound",
            "mixed_bound",
            "pack_bound",
            "unknown",
        }

        outputs = evidence_database.write_outputs(database, tmp / "evidence")
        for path in outputs.values():
            assert Path(path).exists()
        written = json.loads(Path(outputs["evidence_database"]).read_text(encoding="utf-8"))
        assert written["capture_count"] == 2
        csv_text = Path(outputs["evidence_rows_csv"]).read_text(encoding="utf-8")
        assert "scenario_family" in csv_text
        markdown = Path(outputs["evidence_summary"]).read_text(encoding="utf-8")
        assert "RNS8 Evidence Database Summary" in markdown

    print("evidence database self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
