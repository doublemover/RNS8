"""Authored normal-fixture event/work accounting and explicit production contracts."""

from copy import deepcopy
import json
from pathlib import Path
from benchmark_schema import validate_capture
from benchmark_schema.public_timing import validate_sustained
from collect_public_calibration import commands
from evidence_database_lib.event_cover import disjoint_event_cover
from evidence_database_lib.work_model import (
    estimate_work,
    classify_bottleneck,
    build_roofline_priority,
)
from measurement_contract import host_instrumentation, output_completion_boundary
from result_compare import contract_value


def main():
    root = Path(__file__).resolve().parents[1]
    finite = json.loads(
        (root / "tests/fixtures/benchmark_schema/v4_finite_ring_u8_ck.json").read_text()
    )
    roc = json.loads(
        (root / "tests/fixtures/benchmark_schema/v4_bounded_i64_rocwmma.json").read_text()
    )
    for capture in (finite, roc):
        validate_capture(capture)
    cover = disjoint_event_cover(finite)
    assert sum(cover["category_medians_us"].values()) == 190
    assert "pack" not in cover["selected_events"] and "rns_gemm" not in cover["selected_events"]
    assert classify_bottleneck(roc)["event_bottleneck_class"] == "transfer_event_bound"
    original = deepcopy(roc)
    disjoint_event_cover(roc)
    assert original == roc
    aggregate_only = deepcopy(finite)
    for key in ("gpu_event_timings_us", "gpu_event_timing_summary_us"):
        aggregate_only[key] = {
            name: value
            for name, value in aggregate_only[key].items()
            if name in ("pack", "rns_gemm", "crt_export")
        }
    assert set(disjoint_event_cover(aggregate_only)["selected_events"]) == {
        "pack",
        "rns_gemm",
        "crt_export",
    }
    incomplete = deepcopy(finite)
    del incomplete["gpu_event_timings_us"]["finite_pack_kernel"]
    assert "pack" in disjoint_event_cover(incomplete)["selected_events"]
    single = estimate_work(roc)
    for tasks, field, count_field in (
        (4, "host_api_batch", "tasks_per_measured_repeat"),
        (8, "grouped_dispatch", "task_count"),
    ):
        batch = deepcopy(roc)
        batch[field] = {count_field: tasks}
        result = estimate_work(batch)
        assert result["estimated_ops"] == single["estimated_ops"] * tasks
        assert result["estimated_output_bytes"] == single["estimated_output_bytes"] * tasks
        assert result["measured_gops"] == single["measured_gops"] * tasks
    reused = deepcopy(roc)
    reused["pack_mode"] = "prepacked_reuse_b"
    assert estimate_work(reused)["estimated_input_b_bytes"] == 0
    chain = deepcopy(roc)
    chain["residue_chain_length"] = 2
    assert estimate_work(chain)["estimated_ops"] is None
    for domain in ("i64", "u64"):
        native = json.loads(
            (
                root / f"tests/fixtures/benchmark_schema/v4_bounded_{domain}_vector_alu.json"
            ).read_text()
        )
        validate_capture(native)
        expected = 2 * native["m"] * native["n"] * native["k"]
        rates = []
        for prefix in (1, 9, 20):
            # An unused plan-prefix annotation cannot manufacture native work.
            value = deepcopy(native)
            value["prefix"] = prefix
            model = estimate_work(value)
            assert model["estimated_ops"] == model["useful_integer_gemm_ops"] == expected
            assert model["estimated_residue_or_byte_limb_ops"] is None
            assert model["estimated_residue_capacity_bytes"] == 0
            assert model["estimated_native_result_bytes"] == native["m"] * native["n"] * 8
            assert model["representation_gops"] is None
            rates.append(model["useful_gops"])
        assert len(set(rates)) == 1
    adaptive = json.loads(
        (root / "tests/fixtures/benchmark_schema/v4_bounded_i64_adaptive_hip.json").read_text()
    )
    validate_capture(adaptive)
    missing_weights = estimate_work(adaptive)
    assert not missing_weights["work_estimate_available"]
    assert missing_weights["estimated_ops"] is None
    assert missing_weights["representation_gops"] is None
    assert (
        missing_weights["representation_ops_upper_bound"]
        == missing_weights["useful_integer_gemm_ops"] * 4
    )
    assert missing_weights["estimated_residue_capacity_bytes"] == adaptive["m"] * adaptive["n"] * 9
    mixture = build_roofline_priority(
        [
            dict(estimate_work(native), semantics="bounded_i64"),
            dict(estimate_work(roc), semantics="bounded_i64"),
        ]
    )[0]
    assert mixture["median_measured_gops"] is None
    assert mixture["median_representation_gops"] is None
    assert mixture["median_useful_gops"] is not None
    assert host_instrumentation(roc) == "diagnostic_gpu_events"
    production = deepcopy(roc)
    production["timing_metadata"]["gpu_event_timing"] = False
    production["timing_metadata"]["host_instrumentation_state"] = "hooks_disabled"
    assert contract_value(production, "host_instrumentation_state") != contract_value(
        roc, "host_instrumentation_state"
    )
    production["sustained_production"] = dict(
        requested_batches=1,
        min_calls=2,
        max_calls=4,
        min_duration_us=100,
        boundary="same_completed_output_as_latency_lane",
        instrumentation="hooks_disabled",
        observations=[
            dict(calls=4, tasks=4, duration_us=80, per_call_us=20, termination="hard_call_cap")
        ],
    )
    assert not validate_sustained(production)
    production["sustained_production"]["observations"][0]["calls"] = 3
    assert validate_sustained(production)
    assert (
        output_completion_boundary(
            {"timing_metadata": {"grouped_dispatch_batched_export_enabled": True}}
        )
        == "compact_task_major_host_slab"
    )
    assert (
        output_completion_boundary(
            {"output_policy": {"completion_boundary": "per_task_host_destinations"}}
        )
        == "per_task_host_destinations"
    )
    plan = commands(
        Path("benchmark"),
        [dict(name="case", arguments=["--backend", "hip-direct"])],
        2,
        7,
        Path("out"),
    )
    assert plan[0]["mode"] != plan[2]["mode"]
    assert {entry["mode"] for entry in plan} == {"instrumented", "production"}
    assert all(
        ("--sustained-batches" in entry["command"]) == (entry["mode"] == "production")
        for entry in plan
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
