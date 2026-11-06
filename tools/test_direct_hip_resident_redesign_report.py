#!/usr/bin/env python3
"""Self-test Direct-HIP resident redesign reporting."""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path

import direct_hip_resident_redesign_report as report


def capture(
    *,
    kernel: str,
    median_us: float,
    rns_gemm_us: float,
    crt_export_us: float = 300.0,
    semantics: str = "bounded_i64",
    path: str = "capture.json",
    checksum: int = 1234,
    redesign: dict | None = None,
    schema_status: str = "valid",
    events: bool = True,
) -> dict:
    return {
        "_path": path,
        "_schema_status": schema_status,
        "_schema_errors": ["schema error"] if schema_status != "valid" else [],
        "schema_version": 4,
        "benchmark": "rns8_bounded_gemm_persistent_rns",
        "benchmark_execution_mode": "persistent_resident_matrices",
        "backend_selected": "hip-direct",
        "backend_requested": "hip-direct",
        "selected_kernel": kernel,
        "backend_metadata": {"selected_kernel": kernel},
        "semantics": semantics,
        "m": 512,
        "n": 512,
        "k": 512,
        "prefix": 9,
        "selected_prefix": 9,
        "requested_max_prefix": 9,
        "contract_prefix_policy": "fixed_requested",
        "input_distribution": "signed_uniform_-16_16",
        "seed": 20260606,
        "warmups": 3,
        "repeats": 9,
        "checksum_u64": checksum,
        "timing_summary_us": {
            "end_to_end": {"median": median_us},
            "rns_gemm": {"median": rns_gemm_us},
            "crt_export": {"median": crt_export_us},
        },
        "timing_metadata": {
            "gpu_event_timing": events,
            "gpu_event_timing_status": "available" if events else "not_available",
            "gpu_event_phase_order": ["pack", "rns_gemm_kernel_group", "crt_export"]
            if events
            else None,
            "gpu_event_timing_source": "hipEventElapsedTime" if events else None,
        },
        "gpu_event_timings_us": {"rns_gemm_kernel_group": [rns_gemm_us] * 9} if events else None,
        "resident_redesign": redesign or {},
    }


def redesign_metadata(candidate: str = "layout_tile_export_prototype") -> dict:
    return {
        "candidate": candidate,
        "dimensions": [
            "data_layout",
            "tile_shape",
            "export_interaction",
            "schedule_upload",
            "workspace_reuse",
        ],
        "resource_summary": {
            "vgpr": 48,
            "sgpr": 64,
            "lds_bytes": 4096,
            "occupancy": 0.75,
            "bottleneck_class": "memory_pressure",
        },
    }


def production_controls() -> None:
    import json
    from diagnostic_companion import build_binding
    from target_qualification import file_digest
    from test_direct_hip_report_support import collect_resident_processes
    from test_capture_identity_support import synthetic_static_isa_report
    from selected_kernel_resources import load_static_counter_report
    from unittest.mock import patch

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)

        def annotate(fixture, mode, run):
            if mode not in ("cpu", "default"):
                fixture["resident_redesign"] = redesign_metadata(mode)

        captures, paths, bindings = collect_resident_processes(
            root,
            "matched",
            dict(
                default=(report.DEFAULT_KERNEL, 100),
                candidate=(report.GROUPED_ACTIVE_SCHEDULE_KERNEL, 80),
                colpair=(report.REJECTED_COLPAIR_KERNEL, 130),
            ),
            annotate,
        )
        isa_paths, summaries = [], []
        for i, fixture in enumerate((captures[4], captures[6])):
            isa = synthetic_static_isa_report(
                fixture, dict(vgpr_count=48, sgpr_count=64, lds_bytes=4096, occupancy=4)
            )
            path = root / f"selected-{i}.json"
            path.write_text(json.dumps(isa))
            isa_paths.append(path)
            summaries.append(dict(path=str(path), sha256=file_digest(path)))
        counter_path = root / "counter-inventory.json"
        counter_path.write_text(
            json.dumps(
                dict(
                    reports=[
                        dict(
                            capture=dict(path="unrelated-filename-is-not-a-binding.json"),
                            resource_summary=dict(vgpr=999, occupancy=999),
                            isa_summaries=summaries,
                        )
                    ]
                )
            )
        )

        def resources():
            return load_static_counter_report(counter_path)

        def build(values, receipts=bindings, resource_inputs=None):
            return report.build_direct_hip_resident_redesign_report(
                values, diagnostic_bindings=receipts, resources=resource_inputs or resources()
            )

        result = build(captures)
        rows = {row["candidate"]: row for row in result["comparisons"]}
        winner, loser = rows["candidate"], rows["colpair"]
        assert winner["decision"] == "route_candidate", winner["blockers"]
        assert winner["speedup_vs_resident_default"] == 1.25
        assert winner["promotion_eligible"] is True
        assert winner["resource_observations"][0]["status"]["resource_summary"]["vgpr"] == 48
        assert loser["decision"] == "drop/deprioritize"
        assert result["summary"]["rank51_gate_complete"] is True
        for order in (list(reversed(captures)), captures[4:] + captures[:4]):
            row = next(
                row for row in build(order)["comparisons"] if row["candidate"] == "candidate"
            )
            assert row["speedup_vs_resident_default"] == 1.25
        copied = build(captures[:5] + [copy.deepcopy(captures[4])])
        assert copied["comparisons"][0]["speedup_vs_resident_default"] is None
        incomplete = dict(bindings)
        incomplete.pop(str(paths[5].resolve()))
        row = next(
            row
            for row in build(captures, incomplete)["comparisons"]
            if row["candidate"] == "candidate"
        )
        assert row["speedup_vs_resident_default"] is None
        no_resources = report.build_direct_hip_resident_redesign_report(
            captures, diagnostic_bindings=bindings
        )
        assert all(row["promotion_eligible"] is False for row in no_resources["comparisons"])
        assert all(not row["resource_explanation_available"] for row in no_resources["comparisons"])

        # Remove one process's declared dimension and rebuild its real receipt.
        changed = copy.deepcopy(captures)
        changed[5]["resident_redesign"]["dimensions"].remove("workspace_reuse")
        source = paths[5]
        raw = {key: value for key, value in changed[5].items() if not key.startswith("_")}
        source.write_text(json.dumps(raw))
        instrumented = source.parent / "instrumented.json"
        diagnostic = json.loads(instrumented.read_text())
        diagnostic["resident_redesign"]["dimensions"].remove("workspace_reuse")
        instrumented.write_text(json.dumps(diagnostic))
        receipts = dict(bindings)
        receipts[str(source.resolve())] = build_binding(source, instrumented)
        row = next(
            row
            for row in build(changed, receipts)["comparisons"]
            if row["candidate"] == "candidate"
        )
        assert row["promotion_eligible"] is False
        assert "missing_redesign_dimension:workspace_reuse" in row["blockers"]
        loaded = report.build_report(
            paths, counter_report=counter_path, diagnostic_bindings=receipts
        )
        assert loaded["summary"]["route_candidate_count"] == 0
        for margin in (True, 0, 1.02, float("nan"), float("inf")):
            try:
                report.build_direct_hip_resident_redesign_report([], margin=margin)
            except ValueError:
                pass
            else:
                raise AssertionError("invalid margin accepted")

        owned = resources()
        original = report.compare_candidate

        def mutate_after_row(*args, **kwargs):
            result = original(*args, **kwargs)
            isa_paths[0].write_text(isa_paths[0].read_text() + " ")
            return result

        with patch.object(report, "compare_candidate", side_effect=mutate_after_row):
            try:
                build(changed, receipts, owned)
            except ValueError as error:
                assert "changed during review" in str(error)
            else:
                raise AssertionError("changed ISA input was published")


def main() -> int:
    baseline = capture(
        kernel=report.DEFAULT_KERNEL,
        median_us=2000.0,
        rns_gemm_us=1000.0,
        path="baseline.json",
    )
    colpair_loss = capture(
        kernel=report.REJECTED_COLPAIR_KERNEL,
        median_us=2500.0,
        rns_gemm_us=800.0,
        path="colpair-loss.json",
        redesign=redesign_metadata("selected_prefix_colpair_rejected_route"),
    )
    route_candidate = capture(
        kernel=report.GROUPED_ACTIVE_SCHEDULE_KERNEL,
        median_us=1800.0,
        rns_gemm_us=900.0,
        path="layout-prototype.json",
        redesign=redesign_metadata(),
    )
    result = report.build_direct_hip_resident_redesign_report(
        [baseline, colpair_loss, route_candidate]
    )
    summary = result["summary"]
    assert summary["legacy_capture_count"] == 3
    assert summary["candidate_count"] == 0
    assert summary["rank51_gate_complete"] is False
    assert summary["route_candidate_count"] == 0
    assert result["comparisons"] == []
    assert len(result["legacy_inventory"]) == 3
    production_controls()

    with tempfile.TemporaryDirectory() as tmp_name:
        outputs = report.write_outputs(result, Path(tmp_name))
        assert Path(outputs["json"]).exists()
        assert Path(outputs["markdown"]).exists()

    print("direct-HIP resident redesign report self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
