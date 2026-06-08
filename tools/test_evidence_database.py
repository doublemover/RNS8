#!/usr/bin/env python3
"""Self-test the RNS8 evidence database builder."""

from __future__ import annotations

import contextlib
import io
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
                                "lowering_role": "dense_gemm_adjacent_proxy",
                                "output_domain_requirement": "host_export",
                                "promotion_scope": "release_candidate",
                                "ring_dimension": 4096,
                            },
                        },
                        {
                            "family": "large-exploratory",
                            "name": "bounded-i64-large-fixture",
                            "capture_path": str(ck_capture),
                            "capture_name": ck_capture.name,
                            "output_domain": "host_export",
                            "evidence_scope": "large shape metadata fixture",
                            "rationale": "test queue-control scenario metadata join",
                            "metadata": {
                                "large_shape_role": "throughput_probe",
                                "promotion_scope": "exploratory_only",
                                "output_domain_requirement": "host_export",
                                "lowering_role": "large_shape_throughput_probe",
                                "grouping_role": "pre_grouped_baseline",
                                "bridge_role": "native_vector_to_rns_candidate",
                                "modulus_role": "generic_non_hot_prime",
                                "prime_or_composite": "prime",
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
                        "matrix_instruction_count": 7,
                        "dense_integer_matrix_instruction_count": 7,
                        "sparse_integer_matrix_instruction_count": 0,
                        "wmma": 7,
                        "mfma": 0,
                        "smfmac": 0,
                        "swmmac": 0,
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
        invalid_capture = tmp / "invalid_capture.json"
        invalid_capture.write_text(json.dumps({"schema_version": 4}), encoding="utf-8")
        candidate_cache = tmp / "candidate-autotune-cache.json"
        candidate_cache.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
        discovered_names = {path.name for path in evidence_database.discover_capture_paths([tmp])}
        assert "invalid_capture.json" in discovered_names
        assert "review_report.json" not in discovered_names
        assert "scenario_manifest.json" not in discovered_names
        assert "candidate-autotune-cache.json" not in discovered_names
        assert "ck_backend_kernels-gfx1100-ck-isa-summary.json" not in discovered_names
        try:
            evidence_database.load_validated_captures([invalid_capture])
        except Exception:
            pass
        else:
            raise AssertionError("strict evidence database loading should reject invalid captures")
        skipped_invalid: list[dict[str, str]] = []
        with contextlib.redirect_stderr(io.StringIO()):
            skip_loaded = evidence_database.load_validated_captures(
                [hip_capture, invalid_capture],
                skip_invalid=True,
                invalid_captures=skipped_invalid,
            )
        assert len(skip_loaded) == 1
        assert len(skipped_invalid) == 1
        assert skipped_invalid[0]["capture_path"] == str(invalid_capture)

        captures = evidence_database.load_validated_captures([hip_capture, ck_capture])
        captures[0]["timing_summary_us"]["pack_a"] = {"avg": 24.0, "median": 24.0, "p95": 24.0}
        captures[0]["timing_summary_us"]["pack_b"] = {"avg": 16.0, "median": 16.0, "p95": 16.0}
        database = evidence_database.build_database(
            captures,
            scenario_index=evidence_database.load_scenario_index([scenario_manifest]),
            review_index=evidence_database.load_review_index([review_report]),
            isa_index=evidence_database.load_isa_index([isa_report]),
            invalid_captures=skipped_invalid,
        )

        assert database["schema_version"] == 1
        assert database["capture_count"] == 2
        assert database["summary"]["scenario_counts"]["repeated-b"] == 1
        assert database["summary"]["scenario_counts"]["large-exploratory"] == 1
        assert database["summary"]["isa_report_count"] == 1
        assert database["summary"]["captures_with_isa_resources"] == 1
        assert database["summary"]["skipped_invalid_capture_count"] == 1
        assert database["summary"]["pack_split_dominant_counts"] == {"A": 1}
        assert database["skipped_invalid_captures"] == skipped_invalid
        assert database["summary"]["roofline_priority"]
        assert database["summary"]["gpu_roofline_priority"]
        assert all(
            priority["target_id"] != "none"
            for priority in database["summary"]["gpu_roofline_priority"]
        )
        first_priority = database["summary"]["roofline_priority"][0]
        assert first_priority["rank"] == 1
        assert first_priority["roofline_target"] in {
            "compute_throughput",
            "export_bandwidth",
            "launch_api_overhead",
            "mixed_phase_balance",
            "pack_bandwidth",
            "status_overhead",
            "transfer_bandwidth",
            "unclassified",
        }
        assert first_priority["total_bottleneck_us"] > 0
        assert first_priority["optimization_hint"]
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
        assert hip_row["scenario_lowering_role"] == "dense_gemm_adjacent_proxy"
        assert hip_row["scenario_output_domain_requirement"] == "host_export"
        assert hip_row["scenario_promotion_scope"] == "release_candidate"
        assert hip_row["scenario_metadata"]["ring_dimension"] == 4096
        assert "key_switch_digit_aggregation" in hip_row["scenario_metadata_json"]
        assert hip_row["promotion_blockers"] == ["missing_required_baseline:cpu-reference"]
        assert hip_row["estimated_ops"] > 0
        assert hip_row["estimated_input_a_bytes"] == 2048
        assert hip_row["estimated_input_b_bytes"] == 2048
        assert hip_row["estimated_bytes"] > 0
        assert hip_row["arithmetic_intensity_ops_per_byte"] > 0
        assert hip_row["median_pack_a_us"] == 24.0
        assert hip_row["median_pack_b_us"] == 16.0
        assert hip_row["pack_split_dominant_operand"] == "A"
        assert hip_row["pack_a_share_of_split"] == 0.6
        assert hip_row["pack_b_share_of_split"] == 0.4
        assert hip_row["pack_a_bandwidth_gbs"] == 2048 / (24.0 * 1000.0)
        assert hip_row["pack_b_bandwidth_gbs"] == 2048 / (16.0 * 1000.0)
        assert hip_row["bottleneck_class"] in {
            "compute_bound",
            "export_bound",
            "launch_or_api_bound",
            "mixed_bound",
            "pack_bound",
            "unknown",
        }
        assert hip_row["roofline_target"]
        assert hip_row["optimization_hint"]
        ck_row = next(row for row in database["rows"] if row["capture_path"] == str(ck_capture))
        assert ck_row["scenario_family"] == "large-exploratory"
        assert ck_row["scenario_large_shape_role"] == "throughput_probe"
        assert ck_row["scenario_promotion_scope"] == "exploratory_only"
        assert ck_row["scenario_lowering_role"] == "large_shape_throughput_probe"
        assert ck_row["scenario_grouping_role"] == "pre_grouped_baseline"
        assert ck_row["scenario_bridge_role"] == "native_vector_to_rns_candidate"
        assert ck_row["scenario_modulus_role"] == "generic_non_hot_prime"
        assert ck_row["scenario_prime_or_composite"] == "prime"
        assert ck_row["isa_report_count"] == 1
        assert ck_row["isa_report_backends"] == ["ck"]
        assert ck_row["isa_report_targets"] == ["gfx1100"]
        assert ck_row["isa_matrix_instruction_count"] == 7
        assert ck_row["isa_dense_integer_matrix_instruction_count"] == 7
        assert ck_row["isa_sparse_integer_matrix_instruction_count"] == 0
        assert ck_row["isa_wmma_count"] == 7
        assert ck_row["isa_smfmac_count"] == 0
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
        assert "isa_matrix_instruction_count" in csv_text
        assert "median_pack_a_us" in csv_text
        assert "pack_split_dominant_operand" in csv_text
        assert "pack_a_bandwidth_gbs" in csv_text
        assert "isa_dense_integer_matrix_instruction_count" in csv_text
        assert "isa_wmma_count" in csv_text
        assert "scenario_source_role" in csv_text
        assert "roofline_target" in csv_text
        assert "scenario_large_shape_role" in csv_text
        assert "scenario_grouping_role" in csv_text
        assert "native_vector_to_rns_candidate" in csv_text
        md_text = Path(outputs["evidence_summary"]).read_text(encoding="utf-8")
        assert "## Pack Split Dominance" in md_text
        assert "| A | 1 |" in md_text
        assert "exploratory_only" in csv_text
        assert "key_switch_digit_aggregation" in csv_text
        markdown = Path(outputs["evidence_summary"]).read_text(encoding="utf-8")
        assert "RNS8 Evidence Database Summary" in markdown
        assert "skipped_invalid_capture_count" in markdown
        assert "Skipped Invalid Captures" in markdown
        assert "GPU Roofline Priority" in markdown
        assert "Roofline Priority" in markdown
        assert "bottleneck us" in markdown
        assert "median GOP/s" in markdown
        assert "Scenario Metadata" in markdown
        assert "large_shape_role" in markdown
        assert "throughput_probe" in markdown
        assert "promotion_scope" in markdown
        assert "bridge_role" in markdown
        assert "prime_or_composite" in markdown
        assert "ISA Resources" in markdown
        assert "dense int" in markdown

    print("evidence database self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
