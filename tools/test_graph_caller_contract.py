"""Metadata-only graph boundary controls; no graph or device is executed."""

from __future__ import annotations

import unittest

from graph_caller_contract import (
    caller_transaction_blockers,
    graph_setup_us,
    graph_phase_diagnostic,
)
from hip_graph_replay_report import compare_graph_capture, setup_inclusive_per_repeat
from test_hip_graph_replay_report import capture
from measurement_contract import output_completion_boundary


def caller_capture(semantics="bounded_i64", detailed=False):
    value = capture(
        graph=True, median_us=720, setup_us=None, name="synthetic-caller", full_path=True
    )
    value["semantics"] = semantics
    width = 1 if semantics in ("finite_ring_u8", "finite_field_u8") else 8
    limbs = 4 if semantics in ("exact_wide_signed", "exact_wide_unsigned") else 1
    if limbs != 1:
        value["exact_wide_limb_count"] = limbs
    scopes = {
        "finite_ring_u8": "finite_u8",
        "finite_field_u8": "finite_u8",
        "exact_wide_signed": "exact_wide",
        "exact_wide_unsigned": "exact_wide",
        "wrap_u64_mod_2_64": "wrap64",
    }
    graph = value["hip_graph_replay"]
    graph["scope"] = "direct_hip_" + scopes.get(semantics, "bounded") + "_pack_gemm_export"
    observation = dict(
        total_us=720,
        input_staging_us=10 if detailed else None,
        replay_us=700 if detailed else None,
        caller_delivery_us=10 if detailed else None,
    )
    graph["caller_transaction"] = dict(
        schema="rns8_graph_caller_transaction_v1",
        completion_boundary="per_task_host_destinations",
        input_staging="every_completed_call",
        caller_output_delivery="every_completed_call",
        common_setup_excluded=["planning_workspace", "matrix_allocation"],
        phase_clocks=detailed,
        additional_preparation_us=100,
        stream_preparation_us=20,
        teardown_us=30,
        input_a_bytes_per_call=value["m"] * value["k"] * width,
        input_b_bytes_per_call=value["k"] * value["n"] * width,
        output_bytes_per_call=value["m"] * value["n"] * width * limbs,
        first_replay=dict(observation),
        observations=[dict(observation) for _ in range(value["repeats"])],
    )
    value["raw_timings_us"] = dict(end_to_end=[720] * value["repeats"])
    if not detailed:
        value["timing_metadata"].update(
            host_instrumentation_state="hooks_disabled",
            gpu_event_timing=False,
            gpu_event_timing_reason="explicit_production_hooks_disabled",
            gpu_event_timing_status="not_requested_production_timing",
        )
    else:
        value["timing_metadata"]["host_instrumentation_state"] = "diagnostic_gpu_events"
    return value


class GraphCallerContractTest(unittest.TestCase):
    def test_all_full_path_semantics_account_for_preparation_and_payloads(self):
        for semantics in (
            "bounded_i64",
            "bounded_u64",
            "finite_ring_u8",
            "finite_field_u8",
            "exact_wide_signed",
            "exact_wide_unsigned",
            "wrap_u64_mod_2_64",
        ):
            for detailed in (False, True):
                with self.subTest(semantics=semantics, detailed=detailed):
                    value = caller_capture(semantics, detailed)
                    self.assertEqual(caller_transaction_blockers(value), [])
                    self.assertEqual(graph_setup_us(value), 309)
                    self.assertEqual(graph_phase_diagnostic(value), detailed)
                    self.assertAlmostEqual(
                        setup_inclusive_per_repeat(value, graph=True), 720 + 309 / 9
                    )
                    value["hip_graph_replay"]["caller_transaction"][
                        "additional_preparation_us"
                    ] += 90
                    self.assertAlmostEqual(
                        setup_inclusive_per_repeat(value, graph=True), 720 + 399 / 9
                    )

    def test_stale_or_incomplete_delivery_and_clock_contracts_fail_closed(self):
        for mutate in (
            lambda v: v["hip_graph_replay"].pop("caller_transaction"),
            lambda v: v["hip_graph_replay"]["caller_transaction"].update(
                caller_output_delivery="after_all_repeats"
            ),
            lambda v: v["hip_graph_replay"]["caller_transaction"].update(
                input_staging="before_capture_only"
            ),
            lambda v: v["hip_graph_replay"]["caller_transaction"].pop("stream_preparation_us"),
            lambda v: v["hip_graph_replay"]["caller_transaction"]["observations"].pop(),
            lambda v: v["hip_graph_replay"]["caller_transaction"].update(output_bytes_per_call=1),
            lambda v: v["hip_graph_replay"]["caller_transaction"].update(phase_clocks=True),
            lambda v: v["raw_timings_us"]["end_to_end"].__setitem__(0, 721),
        ):
            value = caller_capture()
            mutate(value)
            self.assertTrue(caller_transaction_blockers(value))
            self.assertIsNone(graph_setup_us(value))

    def test_legacy_pinned_output_cannot_share_the_caller_host_identity(self):
        value = caller_capture()
        self.assertEqual(output_completion_boundary(value), "per_task_host_destinations")
        value["hip_graph_replay"].pop("caller_transaction")
        self.assertEqual(output_completion_boundary(value), "legacy_prepared_pinned_graph_output")

    def test_legacy_instrumented_ordinary_control_cannot_be_called_a_workload_win(self):
        value = caller_capture()
        baseline = capture(graph=False, median_us=800, setup_us=None, name="legacy", full_path=True)
        row = compare_graph_capture(value, baseline)
        self.assertEqual(row["decision"], "keep_experimental")
        self.assertIn("host_instrumentation_state_mismatch", row["blockers"])
        self.assertTrue(any("companion" in blocker for blocker in row["blockers"]))


if __name__ == "__main__":
    unittest.main()
