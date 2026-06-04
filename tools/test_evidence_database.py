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
                            "metadata": {
                                "source_role": "fhe_lattice_proxy",
                                "evidence_role": "dense_gemm_adjacent_proxy",
                                "domain_family": "integer_rns",
                                "algebra_family": "fhe_lattice",
                                "workflow_name": "key_switch_digit_aggregation",
                                "phase_label": "external_product_like_dense_proxy",
                                "reuse_profile": "large_read_only_key_material",
                                "ring_dimension": 4096,
                            },
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
        isa_report = tmp / "ck_backend_kernels-gfx1100-ck-isa-summary.json"
        isa_report.write_text(
            json.dumps(
                {
                    "object": "build/windows-msvc-ck-release/ck/ck_backend_kernels.obj",
                    "target": "gfx1100",
                    "backend": "ck",
                    "tools": {"rga_status": "not_run_optional", "rga": None},
                    "code_object_note": None,
                    "device_symbol_count": 5,
                    "reported_symbol_count": 2,
                    "instruction_totals": {
                        "wmma": 7,
                        "mfma": 0,
                        "global_store": 3,
                        "lds_mentions": 11,
                        "wait_instructions": 13,
                        "instruction_lines": 2400,
                        "vgpr_count": 64,
                        "sgpr_count": 48,
                        "occupancy": 8,
                    },
                }
            ),
            encoding="utf-8",
        )

        captures = evidence_database.load_validated_captures([hip_capture, ck_capture])
        database = evidence_database.build_database(
            captures,
            scenario_index=evidence_database.load_scenario_index([scenario_manifest]),
            review_index=evidence_database.load_review_index([review_report]),
            isa_index=evidence_database.load_isa_index([isa_report]),
        )

        assert database["schema_version"] == 1
        assert database["capture_count"] == 2
        assert database["summary"]["scenario_counts"]["repeated-b"] == 1
        assert database["summary"]["scenario_counts"]["unlabeled"] == 1
        assert database["summary"]["isa_report_count"] == 1
        assert database["summary"]["captures_with_isa_resources"] == 1
        hip_row = next(row for row in database["rows"] if row["capture_path"] == str(hip_capture))
        assert hip_row["scenario_family"] == "repeated-b"
        assert hip_row["output_domain"] == "host_export"
        assert hip_row["scenario_source_role"] == "fhe_lattice_proxy"
        assert hip_row["scenario_evidence_role"] == "dense_gemm_adjacent_proxy"
        assert hip_row["scenario_domain_family"] == "integer_rns"
        assert hip_row["scenario_algebra_family"] == "fhe_lattice"
        assert hip_row["scenario_workflow_name"] == "key_switch_digit_aggregation"
        assert hip_row["scenario_phase_label"] == "external_product_like_dense_proxy"
        assert hip_row["scenario_reuse_profile"] == "large_read_only_key_material"
        assert hip_row["scenario_metadata"]["ring_dimension"] == 4096
        assert "key_switch_digit_aggregation" in hip_row["scenario_metadata_json"]
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
        ck_row = next(row for row in database["rows"] if row["capture_path"] == str(ck_capture))
        assert ck_row["isa_report_count"] == 1
        assert ck_row["isa_report_backends"] == ["ck"]
        assert ck_row["isa_report_targets"] == ["gfx1100"]
        assert ck_row["isa_wmma_count"] == 7
        assert ck_row["isa_global_store_count"] == 3
        assert ck_row["isa_vgpr_count"] == 64
        assert ck_row["isa_occupancy"] == 8

        outputs = evidence_database.write_outputs(database, tmp / "evidence")
        for path in outputs.values():
            assert Path(path).exists()
        written = json.loads(Path(outputs["evidence_database"]).read_text(encoding="utf-8"))
        assert written["capture_count"] == 2
        csv_text = Path(outputs["evidence_rows_csv"]).read_text(encoding="utf-8")
        assert "scenario_family" in csv_text
        assert "isa_wmma_count" in csv_text
        assert "scenario_source_role" in csv_text
        assert "key_switch_digit_aggregation" in csv_text
        markdown = Path(outputs["evidence_summary"]).read_text(encoding="utf-8")
        assert "RNS8 Evidence Database Summary" in markdown
        assert "Scenario Metadata" in markdown
        assert "ISA Resources" in markdown

    print("evidence database self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
