"""Authored synthetic companion receipts; no benchmark executable is invoked."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
import uuid
from benchmark_schema import load_capture, validate_capture, validation_errors
from benchmark_sweep_lib.review import review_captures, write_promoted_cache_entries
from promotion_ledger import capture_entry
from collect_diagnostic_pairs import paired_plan
from diagnostic_companion import (
    build_binding,
    common_identity,
    digest,
    load_bindings,
    production_evidence,
    validate_binding,
)
from test_capture_identity_support import synthetic_identity, synthetic_host_measurements
from test_benchmark_schema_support_chains_wrap64 import (
    as_reused_pack_capture,
    as_reused_a_capture,
    as_hip_graph_full_bounded_capture,
    as_hip_graph_full_finite_capture,
    as_hip_graph_full_exact_wide_capture,
    as_hip_graph_full_wrap64_capture,
)
from test_benchmark_schema_support_metadata import (
    add_helper_lane_fields,
)

FIXTURES = Path(__file__).parents[1] / "tests/fixtures/benchmark_schema"


def authored_pair(root, name="v4_bounded_i64_rocwmma.json"):
    return authored_capture_pair(root, load_capture(FIXTURES / name))


def authored_capture_pair(root, diagnostic):
    diagnostic = copy.deepcopy(diagnostic)
    synthetic_identity(diagnostic)
    synthetic_host_measurements(diagnostic)
    diagnostic["timing_metadata"]["host_instrumentation_state"] = "diagnostic_gpu_events"
    production = copy.deepcopy(diagnostic)
    production["execution_id"] = str(uuid.uuid4())
    p = production["timing_metadata"]
    p.update(
        host_instrumentation_state="hooks_disabled",
        gpu_event_timing=False,
        gpu_event_timing_status="not_requested_production_timing",
        gpu_event_timing_reason="explicit_production_hooks_disabled",
    )
    for key in (
        "gpu_event_phase_order",
        "gpu_event_timing_source",
        "gpu_event_timing_source_scope",
    ):
        p[key] = None
    production["gpu_event_timings_us"] = production["gpu_event_timing_summary_us"] = None
    paths = []
    for capture, mode in ((production, "production"), (diagnostic, "instrumented")):
        request = dict(
            schema="rns8_benchmark_request_v1",
            arguments=["--backend", capture["backend_selected"], "--host-timing", mode],
            executable_sha256="b" * 64,
            environment={"HIP_VISIBLE_DEVICES": "0"},
            overrides_sha256="e" * 64,
        )
        request["identity_sha256"] = digest(request)
        capture["request_identity"] = request
        path = root / (mode + ".json")
        path.write_text(json.dumps(capture))
        paths.append(path)
    return production, diagnostic, paths


def authored_reuse_pair(root, operands):
    production, diagnostic, paths = authored_pair(root, "v4_bounded_i64_ck.json")
    result = []
    for capture, path, cost in zip((production, diagnostic), paths, (123, 147)):
        capture = (
            as_reused_pack_capture(capture)
            if operands == ("A", "B")
            else as_reused_a_capture(capture)
        )
        if operands == ("B",):
            capture["pack_mode"] = "prepacked_reuse_b"
            capture["prepack_reuse_operands"] = ["B"]
            capture["timing_metadata"]["pack_mode"] = "prepacked_reuse_b"
            capture["timing_metadata"]["prepack_reuse_operands"] = ["B"]
            capture["timing_metadata"]["phase_notes"]["pack"] = (
                "per-repeat packing of A; B is retained"
            )
            capture["timing_metadata"]["phase_availability"]["prepack_setup"]["reason"] = (
                "B packed once before warmups"
            )
        add_helper_lane_fields(capture)
        breakdown = dict(
            pack_a=cost if "A" in operands else 0,
            pack_b=0 if "A" in operands else cost,
            runtime_cache=0,
            unclassified=0,
        )
        capture["prepack_setup_us"] = capture["avg_prepack_setup_us"] = cost
        capture["prepack_setup_breakdown_us"] = breakdown
        repeats = capture["repeats"]
        median = capture["timing_summary_us"]["end_to_end"]["median"]
        capture["reuse_contract"] = dict(
            enabled=True,
            operand_role="A+B" if len(operands) == 2 else operands[0],
            source_version_inputs="monotonic_source_version_per_repeat_when_packing_runs",
            setup_scope="one_time_before_warmups",
            setup_cost_us=cost,
            setup_breakdown_us=dict(breakdown),
            setup_amortized_us=cost / repeats,
            repeat_median_end_to_end_us=median,
            setup_inclusive_median_end_to_end_us=median + cost / repeats,
            setup_inclusive_policy="one_time_setup_amortized_over_measured_repeats",
            measured_repeat_count=repeats,
            break_even_repeat_count=None,
            output_domain="native_i64_u64_host",
            next_op="final-export",
            target_fingerprint=capture["device"]["gcn_arch"],
            backend_fingerprint="ck",
            kernel_fingerprint=capture["selected_kernel"],
            workspace_fingerprint="synthetic-retained",
            runtime_prepack_cache=None,
            production_runtime_prepack_cache_available=False,
            setup_inclusive_cache_promotion_candidate=False,
            promotion_eligible=False,
            invalidation_reasons=["source_version_changed"],
        )
        request = capture["request_identity"]
        request["arguments"] += (
            ["--reuse-packed-inputs"]
            if len(operands) == 2
            else ["--reuse-packed-" + operands[0].lower()]
        )
        request["identity_sha256"] = digest(
            {k: v for k, v in request.items() if k != "identity_sha256"}
        )
        synthetic_host_measurements(capture)
        validate_capture(capture, path)
        path.write_text(json.dumps(capture), encoding="utf-8")
        result.append(capture)
    return *result, paths


def authored_graph_pair(root, transform):
    fixture = (
        "v4_wrap64_hip.json"
        if transform is as_hip_graph_full_wrap64_capture
        else ("v4_bounded_i64_adaptive_hip.json")
    )
    production, diagnostic, paths = authored_pair(root, fixture)
    result = []
    for index, (capture, path) in enumerate(zip((production, diagnostic), paths)):
        capture = transform(capture)
        detailed = index == 1
        metadata = capture["timing_metadata"]
        metadata["host_instrumentation_state"] = (
            "diagnostic_gpu_events" if detailed else "hooks_disabled"
        )
        if not detailed:
            metadata.update(
                gpu_event_timing_status="not_requested_production_timing",
                gpu_event_timing_reason="explicit_production_hooks_disabled",
            )
        graph = capture["hip_graph_replay"]
        finite = capture["semantics"] in ("finite_ring_u8", "finite_field_u8")
        width = 1 if finite else 8
        wide = capture["semantics"] in ("exact_wide_signed", "exact_wide_unsigned")
        limbs = capture["exact_wide_limb_count"] if wide else 1
        records = [
            dict(
                total_us=value,
                input_staging_us=1 if detailed else None,
                replay_us=value - 2 if detailed else None,
                caller_delivery_us=1 if detailed else None,
            )
            for value in capture["raw_timings_us"]["end_to_end"]
        ]
        graph["caller_transaction"] = dict(
            schema="rns8_graph_caller_transaction_v1",
            completion_boundary="per_task_host_destinations",
            input_staging="every_completed_call",
            caller_output_delivery="every_completed_call",
            common_setup_excluded=["planning_workspace", "matrix_allocation"],
            phase_clocks=detailed,
            additional_preparation_us=100 + index,
            stream_preparation_us=20 + index,
            teardown_us=30 + index,
            input_a_bytes_per_call=capture["m"] * capture["k"] * width,
            input_b_bytes_per_call=capture["k"] * capture["n"] * width,
            output_bytes_per_call=capture["m"] * capture["n"] * width * limbs,
            first_replay=dict(records[0]),
            observations=records,
        )
        request = capture["request_identity"]
        mode = "instrumented" if detailed else "production"
        semantic = {"wrap_u64_mod_2_64": "wrap-u64"}.get(
            capture["semantics"], capture["semantics"].replace("_", "-")
        )
        request["arguments"] = [
            "--backend",
            "hip-direct",
            "--semantics",
            semantic,
            "--hip-graph-replay",
            "--host-timing",
            mode,
        ]
        request["identity_sha256"] = digest(
            {key: value for key, value in request.items() if key != "identity_sha256"}
        )
        synthetic_host_measurements(capture)
        validate_capture(capture, path)
        path.write_text(json.dumps(capture), encoding="utf-8")
        result.append(capture)
    return *result, paths


class DiagnosticCompanion(unittest.TestCase):
    def test_independent_reuse_observations_bind_without_weakening_immutable_receipts(self):
        for operands in (("A",), ("B",), ("A", "B")):
            with self.subTest(operands=operands), tempfile.TemporaryDirectory() as temp:
                production, diagnostic, paths = authored_reuse_pair(Path(temp), operands)
                self.assertNotEqual(
                    production["reuse_contract"]["setup_cost_us"],
                    diagnostic["reuse_contract"]["setup_cost_us"],
                )
                binding = build_binding(*paths)
                self.assertEqual(
                    binding["common_identity_schema"], "rns8_companion_common_contract_v2"
                )
                self.assertEqual(validate_binding(binding, production), binding)
                production["_path"] = str(paths[0])
                report = review_captures(
                    [production],
                    review_mode="release",
                    diagnostic_bindings={str(paths[0].resolve()): binding},
                )
                candidate = report["groups"][0]["candidates"][0]
                self.assertTrue(candidate["production_timing_evidence"]["eligible"])
                self.assertFalse(candidate["promotable"])
                ledger = capture_entry(paths[0], diagnostic_binding=binding)
                self.assertTrue(ledger["production_timing_evidence"]["eligible"])
                diagnostic["reuse_contract"]["setup_cost_us"] += 1
                paths[1].write_text(json.dumps(diagnostic), encoding="utf-8")
                with self.assertRaises(ValueError):
                    validate_binding(binding)

    def test_reuse_controls_and_unclassified_observations_still_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            production, diagnostic, _ = authored_reuse_pair(Path(temp), ("A", "B"))
            for field, value in (
                ("setup_inclusive_policy", "different"),
                ("measured_repeat_count", diagnostic["repeats"] + 1),
                ("source_version_inputs", "different"),
                ("output_domain", "residue_current"),
                ("kernel_fingerprint", "different"),
                ("operand_role", "A"),
            ):
                bad = copy.deepcopy(diagnostic)
                bad["reuse_contract"][field] = value
                self.assertNotEqual(common_identity(production), common_identity(bad), field)
            for mutate in (
                lambda c: c.update(future_budget_us=123),
                lambda c: c["reuse_contract"].update(future_elapsed_us=123),
                lambda c: c["reuse_contract"]["setup_breakdown_us"].update(future_stage_us=123),
                lambda c: c["timing_metadata"].update(gpu_event_future_control=True),
                lambda c: c.update(sustained_production={"min_duration_us": 123}),
            ):
                bad = copy.deepcopy(diagnostic)
                mutate(bad)
                self.assertNotEqual(common_identity(production), common_identity(bad))

    def test_full_path_graph_host_phase_companions_do_not_claim_hip_event_or_isa_evidence(self):
        for transform in (
            as_hip_graph_full_bounded_capture,
            as_hip_graph_full_finite_capture,
            as_hip_graph_full_exact_wide_capture,
            as_hip_graph_full_wrap64_capture,
        ):
            with self.subTest(transform=transform.__name__), tempfile.TemporaryDirectory() as temp:
                production, diagnostic, paths = authored_graph_pair(Path(temp), transform)
                binding = build_binding(*paths)
                self.assertEqual(binding["diagnostic_kind"], "graph_caller_host_phases")
                self.assertEqual(validate_binding(binding, production), binding)
                self.assertTrue(production_evidence(production, binding)["eligible"])
                diagnostic["hip_graph_replay"]["caller_transaction"]["caller_output_delivery"] = (
                    "after_all_repeats"
                )
                paths[1].write_text(json.dumps(diagnostic), encoding="utf-8")
                with self.assertRaises(ValueError):
                    validate_binding(binding)

    def test_eventless_accelerator_requires_explicit_production_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            production, diagnostic, paths = authored_pair(Path(temp))
            validate_capture(production)
            validate_capture(diagnostic)
            bad = copy.deepcopy(production)
            bad["timing_metadata"].pop("host_instrumentation_state")
            self.assertTrue(any("HIP event" in error for error in validation_errors(bad)))
            bad = copy.deepcopy(production)
            bad["gpu_event_timings_us"] = diagnostic["gpu_event_timings_us"]
            self.assertTrue(any("must be null" in error for error in validation_errors(bad)))

    def test_pack_event_observations_do_not_erase_retention_policy_or_counters(self):
        with tempfile.TemporaryDirectory() as temp:
            production, diagnostic, _ = authored_pair(Path(temp))
            for value in (production, diagnostic):
                value["hipblaslt_operand_prepack"] = dict(
                    policy="both", after_warmups=dict(a_retained_planes_packed=9), event_work=None
                )
            diagnostic["hipblaslt_operand_prepack"]["event_work"] = dict(
                schema="rns8_hipblaslt_pack_event_work_v1", complete=True, per_gemm=[]
            )
            self.assertEqual(common_identity(production), common_identity(diagnostic))
            for key, value in (
                ("policy", "none"),
                ("after_warmups", dict(a_retained_planes_packed=0)),
                ("future_control", 1),
            ):
                changed = copy.deepcopy(diagnostic)
                changed["hipblaslt_operand_prepack"][key] = value
                self.assertNotEqual(common_identity(production), common_identity(changed))

    def test_raw_files_and_all_bound_contracts_are_immutable(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            production, diagnostic, paths = authored_pair(root)
            before = [p.read_bytes() for p in paths]
            binding = build_binding(*paths)
            self.assertEqual(validate_binding(binding, production), binding)
            self.assertTrue(production_evidence(production, binding)["eligible"])
            self.assertEqual(before, [p.read_bytes() for p in paths])
            self.assertEqual(common_identity(production), common_identity(diagnostic))
            changed = copy.deepcopy(production)
            changed["selected_kernel"] += "_other"
            self.assertFalse(production_evidence(changed, binding)["eligible"])
            for field, value in [
                ("selected_kernel", "other"),
                ("seed", 99),
                ("warmups", 0),
                ("checksum_u64", 0),
                ("packed_layout_version", 99),
            ]:
                bad = copy.deepcopy(diagnostic)
                bad[field] = value
                self.assertNotEqual(common_identity(production), common_identity(bad), field)
            paths[1].write_text(
                paths[1].read_text() + "\n"
            )  # even byte-only post-binding replacement is detectable
            with self.assertRaises(ValueError):
                validate_binding(binding)

    def test_independent_process_artifact_environment_and_request_are_required(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            production, diagnostic, paths = authored_pair(root)
            mutations = [
                lambda c: c.update(execution_id=production["execution_id"]),
                lambda c: c["artifact_capture"].update(executable_sha256="0" * 64),
                lambda c: c["request_identity"]["environment"].update(HIP_VISIBLE_DEVICES="1"),
                lambda c: c["request_identity"].update(identity_sha256="0" * 64),
                lambda c: c["build_manifest"].update(source_digest="f" * 64),
            ]
            for mutate in mutations:
                bad = copy.deepcopy(diagnostic)
                mutate(bad)
                paths[1].write_text(json.dumps(bad))
                with self.assertRaises(ValueError):
                    build_binding(*paths)
            paths[1].write_text(json.dumps(diagnostic))
            binding = build_binding(*paths)
            sidecar = root / "binding.json"
            sidecar.write_text(json.dumps(binding))
            index = load_bindings([sidecar])
            self.assertIn(str(paths[0].resolve()), index)
            with self.assertRaises(ValueError):
                load_bindings([sidecar, sidecar])

    def test_binding_satisfies_only_diagnostic_gate_not_other_release_gates(self):
        with tempfile.TemporaryDirectory() as temp:
            production, diagnostic, paths = authored_pair(Path(temp))
            binding = build_binding(*paths)
            production["_path"] = str(paths[0])
            missing = review_captures([production], review_mode="release")
            candidate = missing["groups"][0]["candidates"][0]
            self.assertIn("missing_required_gpu_events", candidate["promotion_blockers"])
            report = review_captures(
                [production],
                review_mode="release",
                diagnostic_bindings={str(paths[0].resolve()): binding},
            )
            candidate = report["groups"][0]["candidates"][0]
            self.assertNotIn("missing_required_gpu_events", candidate["promotion_blockers"])
            self.assertTrue(candidate["production_timing_evidence"]["eligible"])
            self.assertIn("missing_required_baselines", candidate["promotion_blockers"])
            self.assertFalse(candidate["promotable"])
            ledger = capture_entry(paths[0], diagnostic_binding=binding)
            self.assertTrue(ledger["production_timing_evidence"]["eligible"])
            self.assertNotIn("matched_diagnostic_companion_missing", ledger["promotion_blockers"])
            cache = Path(temp) / "not-written.json"
            self.assertEqual(write_promoted_cache_entries(report, [production], cache), 0)
            self.assertFalse(cache.exists())
            # A mutated/stale report cannot turn instrumented timings into a cache write.
            forged = dict(
                promotable_autotune_entries=[
                    dict(
                        source_capture=str(paths[0]),
                        source_capture_sha256="0" * 64,
                        diagnostic_companion=binding,
                    )
                ]
            )
            self.assertEqual(write_promoted_cache_entries(forged, [production], cache), 0)
            self.assertFalse(cache.exists())

    def test_collection_counterbalances_modes_without_running_binary(self):
        plan = paired_plan(
            Path("not-executed"),
            ["--m", "17", "--host-timing", "production"],
            Path("out"),
            3,
            "fixture",
        )
        self.assertEqual(
            [r["captures"][0]["mode"] for r in plan], ["production", "instrumented", "production"]
        )
        self.assertEqual(len({c["output"] for r in plan for c in r["captures"]}), 6)
        for row in plan:
            self.assertEqual(len(row["captures"]), 2)
        with self.assertRaises(ValueError):
            paired_plan(
                Path("not-executed"), ["--sustained-batches", "3"], Path("out"), 2, "fixture"
            )


if __name__ == "__main__":
    unittest.main()
