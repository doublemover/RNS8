#!/usr/bin/env python3
"""Self-test many-small grouped-dispatch comparison reporting."""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path

import many_small_grouped_report
from benchmark_schema import load_capture
from metadata_registry_constants import (
    GROUPED_DISPATCH_STRATEGY_DEVICE_GROUPED_PACK_GEMM_HOST_EXPORTS,
)
from test_benchmark_schema_support_core import (
    as_host_api_batch_capture,
)
from test_benchmark_schema_support_grouped import (
    as_grouped_dispatch_capture,
)


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "benchmark_schema"


def with_path(capture: dict, path: str) -> dict:
    capture = copy.deepcopy(capture)
    capture["_path"] = path
    return capture


def mark_release(capture: dict) -> dict:
    capture = copy.deepcopy(capture)
    capture["warmups"] = 3
    capture["repeats"] = 9
    return capture


def set_backend(capture: dict, backend: str) -> dict:
    capture = copy.deepcopy(capture)
    capture["backend_requested"] = backend
    capture["backend_selected"] = backend
    return capture


def set_e2e_median(capture: dict, aggregate_median_us: float) -> dict:
    capture = copy.deepcopy(capture)
    summary = capture.setdefault("timing_summary_us", {}).setdefault("end_to_end", {})
    summary["median"] = aggregate_median_us
    summary["avg"] = aggregate_median_us
    summary["p95"] = aggregate_median_us
    return capture


def set_checksum(capture: dict, checksum: int) -> dict:
    capture = copy.deepcopy(capture)
    capture["checksum_u64"] = checksum
    return capture


def align_current_contract(capture: dict) -> dict:
    capture = copy.deepcopy(capture)
    capture["bound_source"] = "static_profile"
    capture["output_policy"] = {
        "destination_layout": "contiguous_row_major",
        "logical_ld": capture["n"],
        "ld_padding": 0,
        "per_repeat_logical_export": True,
        "final_checksum_export_after_repeats": False,
        "status_handling": "required",
        "status_event_policy": "status_memset_and_status_d2h_labels_required_when_gpu_events_available",
    }
    return capture


def build_grouped_case(
    grouped_per_task_us: float,
    *,
    include_host_batch: bool = True,
    mismatch_host_batch_checksum: bool = False,
    mismatch_host_batch_task_count: bool = False,
) -> list[dict]:
    base = align_current_contract(
        mark_release(load_capture(FIXTURE_DIR / "v4_bounded_i64_ck.json"))
    )
    best_independent = with_path(
        set_e2e_median(set_backend(base, "cpu-reference"), 80.0), "independent-cpu.json"
    )
    same_backend = with_path(
        set_e2e_median(set_backend(base, "hip-direct"), 100.0), "independent-hip-direct.json"
    )
    grouped_task_count = 32

    captures = [best_independent, same_backend]
    if include_host_batch:
        host_batch = as_host_api_batch_capture(same_backend)
        host_batch = set_backend(host_batch, "hip-direct")
        if mismatch_host_batch_task_count:
            host_batch["host_api_batch"]["batch_size"] = 16
        else:
            host_batch["host_api_batch"]["batch_size"] = grouped_task_count
        host_batch = set_e2e_median(host_batch, 70.0 * host_batch["host_api_batch"]["batch_size"])
        if mismatch_host_batch_checksum:
            host_batch = set_checksum(host_batch, int(host_batch["checksum_u64"]) + 1)
        captures.append(with_path(host_batch, "hostbatch-hip-direct.json"))

    grouped = as_grouped_dispatch_capture(same_backend)
    grouped = set_backend(grouped, "hip-direct")
    grouped["grouped_dispatch"]["task_count"] = grouped_task_count
    grouped["grouped_dispatch"]["task_descriptor_contract"]["task_count"] = grouped_task_count
    grouped = set_e2e_median(
        grouped, grouped_per_task_us * grouped["grouped_dispatch"]["task_count"]
    )
    captures.append(with_path(grouped, "grouped-hip-direct.json"))
    return captures


def grouped_row(report):
    return next(
        row
        for group in report["groups"]
        for row in group["rows"]
        if row["mode"] == "grouped_dispatch"
    )


def production_controls():
    from itertools import permutations
    from diagnostic_companion import build_binding
    from test_benchmark_sweep_support import bounded_capture, set_phase
    from test_diagnostic_companion import authored_capture_pair
    from test_review_reference_support import add_synthetic_full_reference

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        captures, bindings, blocks = [], {}, []
        for backend, mode in (
            ("cpu-reference", "independent"),
            ("cpu-reference", "batch"),
            ("hip-direct", "independent"),
            ("hip-direct", "batch"),
            ("hip-direct", "grouped"),
        ):
            block = []
            for run in range(2):
                fixture = bounded_capture(backend, 400)
                fixture.update(
                    benchmark_execution_mode="persistent_resident",
                    pack_mode="per_repeat_repack",
                    reuse_packed_inputs=False,
                    prepack_reuse_operands=[],
                    prepack_reuse_strategy="none",
                    bound_source="static_profile",
                )
                fixture.setdefault("output_policy", {})["completion_boundary"] = (
                    "per_task_host_destinations"
                )
                tasks = 1
                if mode == "batch":
                    fixture = as_host_api_batch_capture(fixture)
                    tasks = 8
                    fixture["host_api_batch"].update(
                        batch_size=tasks,
                        tasks_per_measured_repeat=tasks,
                        total_measured_tasks=tasks * fixture["repeats"],
                    )
                    fixture["timing_metadata"].update(
                        host_api_batch_size=tasks, inner_repetitions=tasks
                    )
                elif mode == "grouped":
                    fixture = as_grouped_dispatch_capture(fixture)
                    tasks = 8
                    strategy = GROUPED_DISPATCH_STRATEGY_DEVICE_GROUPED_PACK_GEMM_HOST_EXPORTS
                    fixture["grouped_dispatch"]["execution_strategy"] = strategy
                    fixture["timing_metadata"]["grouped_dispatch_execution_strategy"] = strategy
                    fixture["grouped_dispatch"]["task_descriptor_contract"][
                        "device_descriptor_policy"
                    ] = "device_pointer_tables_and_compact_slabs"
                total = 1600 if mode == "grouped" else 2400 if mode == "batch" else 400
                set_phase(fixture, total)
                if tasks > 1:
                    for phase in ("pack", "rns_gemm", "crt_export", "end_to_end"):
                        fixture[f"avg_{phase}_per_task_us"] = fixture[f"avg_{phase}_us"] / tasks
                fixture = add_synthetic_full_reference(fixture)
                lane = root / f"{backend}-{mode}-{run}"
                lane.mkdir()
                _, _, paths = authored_capture_pair(lane, fixture)
                capture = many_small_grouped_report.load_validated_capture(paths[0])
                block.append(capture)
                if backend == "hip-direct":
                    bindings[str(paths[0].resolve())] = build_binding(*paths)
            captures.extend(block)
            blocks.append(block)

        def build(items, receipts=bindings):
            return many_small_grouped_report.build_report_from_captures(
                items, diagnostic_bindings=receipts
            )

        expected = grouped_row(build(captures))
        assert expected["decision"] == "candidate_win", expected["blockers"]
        assert expected["speedup_vs_best_independent"] == 2.0
        assert expected["speedup_vs_same_backend_host_batch"] == 1.5
        assert expected["measurement_cohort"]["independent_runs"] == 2
        assert expected["same_backend_host_batch"]["measurement_cohort"]["independent_runs"] == 2
        assert expected["normalization"]["tasks"] == 8
        assert expected["median_per_task_end_to_end_us"] == 200
        assert expected["numerical_reference"]["passed"]
        assert expected["checksum_matches_same_backend_host_batch"] is True
        assert expected["same_backend_host_batch_task_count_matches"] is True
        for ordering in permutations(blocks[1:]):
            row = grouped_row(build(blocks[0] + [item for block in ordering for item in block]))
            assert row["decision"] == expected["decision"]
            assert row["speedup_vs_best_independent"] == expected["speedup_vs_best_independent"]
        copied = grouped_row(build(captures + [copy.deepcopy(captures[-1])]))
        assert copied["decision"] == "candidate_win"
        assert copied["measurement_cohort"]["independent_runs"] == 2
        renamed_path = root / "renamed-production.json"
        renamed_path.write_text(Path(captures[-1]["_path"]).read_text(), encoding="utf-8")
        renamed = many_small_grouped_report.load_validated_capture(renamed_path)
        renamed_bindings = dict(bindings)
        original_binding = bindings[str(Path(captures[-1]["_path"]).resolve())]
        renamed_bindings[str(renamed_path.resolve())] = build_binding(
            renamed_path, Path(original_binding["diagnostic_capture"])
        )
        row = grouped_row(build(captures[:-1] + [renamed], renamed_bindings))
        assert row["decision"] == "candidate_win"
        assert row["speedup_vs_best_independent"] == expected["speedup_vs_best_independent"]
        # Re-author companions for an actual spread change so admission cannot
        # fail merely because a stale receipt was retained after fixture editing.
        noisy = captures[:-2]
        noisy_bindings = dict(bindings)
        import json

        for run, total in enumerate((1600, 2400)):
            source = bindings[str(Path(captures[-2 + run]["_path"]).resolve())]
            fixture = json.loads(Path(source["diagnostic_capture"]).read_text())
            set_phase(fixture, total)
            for phase in ("pack", "rns_gemm", "crt_export", "end_to_end"):
                fixture[f"avg_{phase}_per_task_us"] = fixture[f"avg_{phase}_us"] / 8
            lane = root / f"noisy-{run}"
            lane.mkdir()
            _, _, paths = authored_capture_pair(lane, fixture)
            noisy.append(many_small_grouped_report.load_validated_capture(paths[0]))
            noisy_bindings[str(paths[0].resolve())] = build_binding(*paths)
        row = grouped_row(build(noisy, noisy_bindings))
        assert row["decision"] != "candidate_win"
        assert row["measurement_cohort"]["process_spread_us"]["end_to_end"]["relative_range"] > 0.1
        assert row["speedup_vs_best_independent"] is None
        singleton = grouped_row(build(captures[:-1] + [copy.deepcopy(captures[-2])]))
        assert singleton["decision"] != "candidate_win"
        assert singleton["speedup_vs_best_independent"] is None
        missing = dict(bindings)
        missing.pop(str(Path(captures[-1]["_path"]).resolve()))
        no_companion = grouped_row(build(captures, missing))
        assert "matched_diagnostic_companion_missing" in no_companion["blockers"]
        assert no_companion["speedup_vs_same_backend_host_batch"] is None
        for field, value in (
            ("device", dict(captures[-1]["device"], name="Other physical GPU")),
            ("seed", captures[-1]["seed"] + 1),
            ("input_distribution", "other-authored-input-recipe"),
        ):
            changed = copy.deepcopy(captures)
            for item in changed[-2:]:
                item[field] = value
            row = grouped_row(build(changed))
            assert row["decision"] == "missing_baseline", (field, row["blockers"])
            assert row["speedup_vs_best_independent"] is None
        compact = copy.deepcopy(captures)
        for item in compact[-2:]:
            item["output_policy"]["completion_boundary"] = "compact_task_major_host_slab"
        row = grouped_row(build(compact))
        assert row["decision"] != "candidate_win"
        assert row["speedup_vs_best_independent"] is None
        zero = copy.deepcopy(captures)
        for item in zero[-2:]:
            item["warmups"] = 0
        row = grouped_row(build(zero))
        assert "not_release_review" in row["blockers"]
        assert row["speedup_vs_best_independent"] is None
        conflict = copy.deepcopy(captures)
        for item in conflict[-2:]:
            item["checksum_u64"] += 1
        row = grouped_row(build(conflict))
        assert row["decision"] != "candidate_win"
        assert "checksum_mismatch_same_backend_host_batch" in row["blockers"]
        assert row["speedup_vs_same_backend_host_batch"] is None
        # Every original file goes through the public report loader as well.
        loaded = many_small_grouped_report.build_report(
            [Path(item["_path"]) for item in captures], diagnostic_bindings=bindings
        )
        assert grouped_row(loaded)["decision"] == "candidate_win"


def main() -> int:
    # Historical, single-file diagnostic fixtures stay descriptive, including
    # descriptor failures; none supplies independent production observations.
    for options in (
        {},
        {"include_host_batch": False},
        {"mismatch_host_batch_checksum": True},
        {"mismatch_host_batch_task_count": True},
    ):
        report = many_small_grouped_report.build_report_from_captures(
            build_grouped_case(60.0, **options)
        )
        row = grouped_row(report)
        assert report["summary"]["candidate_wins"] == 0
        assert row["decision"] != "candidate_win"
        assert row["speedup_vs_best_independent"] is None
        assert row["speedup_vs_same_backend_host_batch"] is None

    invalid = build_grouped_case(60.0)
    invalid[-1]["grouped_dispatch"]["task_descriptor_contract"]["task_count"] += 1
    row = grouped_row(many_small_grouped_report.build_report_from_captures(invalid))
    assert "invalid_grouped_task_descriptor_contract" in row["blockers"]
    assert row["speedup_vs_best_independent"] is None
    # A bucketed task set cannot borrow uniform host-batch or scalar timings.
    heterogeneous = build_grouped_case(60.0)
    descriptor = heterogeneous[-1]["grouped_dispatch"]["task_descriptor_contract"]
    descriptor.update(
        descriptor_layout="same_contract_bucketed_resident_task_triplets_v1",
        bucket_policy="same_contract_shape_buckets",
        bucket_count=2,
        same_shape_required=False,
        shape_key="multiple_shape_buckets",
        buckets=[
            dict(
                bucket_index=index,
                task_offset=index * 16,
                task_count=16,
                shape_key=f"m={64 << index};n=64;k=64;tile_m=128;tile_n=128;prefix=9",
                semantics=heterogeneous[-1]["semantics"],
                output_domain="native_i64_u64_host",
            )
            for index in range(2)
        ],
    )
    row = grouped_row(many_small_grouped_report.build_report_from_captures(heterogeneous))
    assert row["grouped_task_descriptor_valid"]
    assert row["decision"] != "candidate_win"
    assert row["speedup_vs_best_independent"] is None

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for name in (
            "many-small-grouped-report.json",
            "review_report.json",
            "scenario_manifest.json",
            "capture.json",
        ):
            (root / name).write_text("{}", encoding="utf-8")
        assert many_small_grouped_report.expand_inputs([root]) == [root / "capture.json"]
    production_controls()
    print("many-small grouped report self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
