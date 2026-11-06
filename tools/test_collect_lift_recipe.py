import copy
import unittest
from unittest.mock import patch
from collect_lift_recipe import (
    validate_case,
    validate_capture,
    summarize,
    bounds,
    native_payload,
    payload_bytes,
    OWNER_MODES,
    exact_proof_policy,
    useful_gemm_ops,
    graph,
)
from host_control_contract import HOST_CONTROL_FIELDS, PHASE_TIME_CONTRACT


def replay_receipt(c, publish=0):
    from product_replay import policy, estimate

    p = policy(c["case"], c.get("stage_price_training_charge_ns", 0), publish)
    plan = c["materialization_plan"]
    backend = plan["estimated_ns"] - p["orchestration_ns"] - p["publication_ns"]
    plan.update(
        replay_policy=p,
        replay_costs=estimate(p, backend, backend),
        completed_replays=1
        + c["case"]["warmups"]
        + c["case"]["repeats"]
        + sum(s["calls"] for s in c["sustained"]),
        reused_external_packing_calls=0,
        retained_external_packed_bytes=0,
    )


class LiftRecipeContracts(unittest.TestCase):
    def test_observation_boundaries_are_strict_and_keep_historical_scopes_distinct(self):
        from call_observation_contract import CURRENT, HISTORICAL, KNOWN, observation_boundary

        self.assertEqual(observation_boundary({}), HISTORICAL)
        for value in KNOWN:
            self.assertEqual(observation_boundary(dict(hot_loop_serialization=value)), value)
        self.assertNotEqual(CURRENT, HISTORICAL)
        for value in (None, True, False, 0, 1, {}, [], CURRENT + " ", HISTORICAL, "json_per_call"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    observation_boundary(dict(hot_loop_serialization=value))

    def case(self):
        return dict(
            schema="lift_recipe_case_v1",
            side=32,
            depth=3,
            prefix=20,
            source_prefix=1,
            repeats=3,
            warmups=1,
            model_launch_ns=5000,
            max_payload_bytes=1 << 30,
            unsigned_inputs=False,
            mode="recipe-minimum",
            topology="chain",
        )

    def capture(self, mode=None, **changes):
        c = self.case()
        c.update(changes)
        if mode is not None:
            c["mode"] = mode
        cells = c["side"] ** 2
        p = c["prefix"]
        nodes = c["depth"]
        phase = dict(producer_us=1.0, execute_lift_us=2.0, final_export_us=3.0, total_us=6.0)
        h2d, d2h, host = native_payload(c)
        native = c["mode"] == "native-roundtrip"
        result = dict(
            schema="rns8_lift_recipe_microcase_v3",
            case=c,
            run_id="fixture",
            output_boundary="complete_dag_final_three_limb_export",
            extra_intermediate_observations=c["mode"] == "eager-observe",
            private_residue_transfer_control=False,
            native_roundtrip_control=native,
            fixed_plan_exact_proof_policy=exact_proof_policy(c),
            logical_payload_estimate_bytes=payload_bytes(c),
            host_payload_bytes_before_final_export=h2d + d2h,
            final_export_payload_bytes=cells * 24,
            exact_reference_passed=True,
            promotion_eligible=False,
            native_roundtrip_h2d_payload_bytes_per_replay=h2d,
            native_roundtrip_d2h_payload_bytes_per_replay=d2h,
            native_roundtrip_retained_host_bytes=host,
            replay_hip_counter_samples=c["repeats"],
            replay_hip_h2d_bytes=h2d * p // 8 * c["repeats"],
            replay_hip_d2h_bytes=(d2h + cells * 24) * c["repeats"],
            replay_hip_h2d_calls=2 * nodes * c["repeats"] if native else 0,
            replay_hip_d2h_calls=nodes * c["repeats"] if native else c["repeats"],
            build_manifest={},
            execution_id="process1",
            target="gfx1100",
            device="fixture",
            artifact_capture=dict(unchanged_during_run=True, executable_sha256="a" * 64),
            ragged_retained_bytes=0,
            ragged_materialized_input_bytes=0,
            ragged_binding_peak_bytes=0,
            recipe_arena_bytes=25600,
            recipe_retained_lift_bytes=17412,
            auto_retained_device_bytes=0,
            replay_hip_allocations=0,
            replay_hip_frees=0,
            selected_compute_prefixes=[p] * nodes
            if c["mode"]
            in (
                *OWNER_MODES,
                "manual-full",
                "manual-scalar-first",
                "native-roundtrip",
                "eager-observe",
            )
            else [1, 2, 3],
            cost_model=None,
            materialization_plan=None,
            stage_price_identity=None,
            stage_price_training_charge_ns=0,
            stage_price_graph_identity=None,
            timing_source="host_monotonic_completed_public_call",
            nominal_clock_tick_ns=1.0,
            setup_us=10.0,
            first_call=phase,
            samples=[copy.deepcopy(phase) for _ in range(3)],
            sustained=[dict(calls=16, total_us=250000.0, per_call_us=15625.0) for _ in range(3)],
        )
        result.update(
            fixed_plan_selected_kernel="synthetic-fixed-kernel",
            fixed_plan_target_id="synthetic-target",
            kernel_identity_scope="declared_fixed_plan_not_launched_symbol_or_all_recipe_nodes_proof",
            useful_integer_gemm_ops=useful_gemm_ops(c),
            host_control_diagnostics_compiled=c.get("host_diagnostics", False),
        )
        if c["mode"] in OWNER_MODES:
            self.add_owner_fixture(result)
        return result

    def add_owner_fixture(self, capture):
        """Synthetic protocol records only; these are not hardware observations."""
        c = capture["case"]
        nodes = graph(c)
        prepared = c["mode"] == "prepared-nodes"
        refreshes = sum(a >= 2 or b >= 2 for a, b in nodes) if prepared else 0
        completed = 1 + c["warmups"] + c["repeats"] + sum(s["calls"] for s in capture["sustained"])
        initial = dict(
            schema="rns8_exact_proof_owner_observation_v1",
            kind="prepared_nodes" if prepared else "resident_node_dag",
            owner_count=len(nodes) if prepared else 1,
            builds=len(nodes) if prepared else 1,
            reuses=0,
            retained_object_bytes=4096,
            logical_limb_bytes=64,
            memory_scope="retained_objects_and_vector_capacity_separate_logical_limb_bytes_not_allocator_peak",
        )
        before = dict(initial, builds=initial["builds"] + refreshes * completed, reuses=completed)
        final = copy.deepcopy(before)
        capture.update(
            proof_owner_setup=initial,
            proof_execution=dict(
                schema="rns8_exact_proof_execution_v1",
                mode=c["mode"],
                node_count=len(nodes),
                initialization_products_charged_to_setup=len(
                    {v for pair in nodes for v in pair if v >= 2}
                )
                if prepared
                else 0,
                input_refreshes_per_call=refreshes,
                native_graph_prepared=c["mode"] in ("resident-stream", "resident-graph"),
                native_graph_owned_capture=c["mode"] == "resident-graph",
                retained_device_control_bytes=4
                if c["mode"] in ("resident-stream", "resident-graph")
                else 0,
                retained_pinned_control_bytes=4
                if c["mode"] in ("resident-stream", "resident-graph")
                else 0,
                terminal_contract="all_products_complete_then_padded_final_three_limb_host_export",
            ),
        )
        if c.get("host_diagnostics", False):
            final.update(builds=before["builds"] + refreshes, reuses=before["reuses"] + 1)
            cold = dict.fromkeys(HOST_CONTROL_FIELDS, 0)
            warm = dict(cold, range_proof_calls=refreshes, proof_builds=refreshes, proof_reuses=1)
            if c["mode"] in ("resident-stream", "resident-graph"):
                warm.update(
                    submission_proofs=1,
                    stream_device_queries=1,
                    stream_capture_queries=1,
                    event_records=1,
                    event_waits=1,
                )
            warm["graph_launches"] = int(c["mode"] == "resident-graph")
            capture["proof_host_diagnostic"] = dict(
                schema="rns8_exact_proof_host_diagnostic_v1",
                scope="cold_setup_and_one_completed_call_after_all_timing",
                phase_time_contract=PHASE_TIME_CONTRACT,
                cold_setup=cold,
                warm_observation=warm,
                owner_before=before,
                owner_after=final,
                exact_reference_passed=True,
                latency_evidence=False,
            )
        capture["proof_owner_final"] = final

    def ordered_capture(self, policy=1):
        c = self.capture(execution_policy=policy)
        c["recipe_retained_lift_bytes"] += 4 if policy else 0
        c["hot_loop_serialization"] = "typed_observation_no_phase_clocks_in_sustained_v2"
        c["recipe_execution"] = dict(
            schema="rns8_product_recipe_execution_v1",
            policy=policy,
            completion_contract="ordered_nondefault_single_terminal_v1"
            if policy
            else "synchronous_wrappers_v1",
            stream_prepared=bool(policy),
            guarded_layout_products=bool(policy),
            retained_control_device_bytes=4 if policy else 0,
            retained_control_pinned_bytes=4 if policy else 0,
            retained_layout_host_bytes=4096,
            stream_wait_calls=policy,
            ordered_gemm_calls=3 if policy else 0,
            ordered_conversion_calls=0,
            ordered_lift_calls=3 if policy else 0,
            ordered_copy_calls=policy,
            ordered_status_readback_bytes=4 if policy else 0,
            planning_price_scope="synchronous_stage_prices_before_explicit_execution_policy",
        )
        return c

    @patch("collect_lift_recipe.manifest_provenance_complete", return_value=True)
    def test_prepared_and_dag_routes_bind_real_lifecycle_and_diagnostic_fields(self, _manifest):
        for mode in OWNER_MODES:
            for diagnostic in (False, True):
                c = self.capture(mode, host_diagnostics=diagnostic)
                entry = dict(case=c["case"], run_id="fixture")
                self.assertEqual(validate_capture(c, entry), c)
                self.assertEqual(
                    c["proof_execution"]["initialization_products_charged_to_setup"],
                    2 if mode == "prepared-nodes" else 0,
                )
                for key, value in (
                    ("node_count", True),
                    ("input_refreshes_per_call", 3),
                    ("native_graph_prepared", 1),
                    ("native_graph_owned_capture", mode != "resident-graph"),
                    ("terminal_contract", "enqueue_only"),
                    ("initialization_products_charged_to_setup", 3),
                    ("retained_device_control_bytes", 2**64),
                ):
                    with self.subTest(mode=mode, diagnostic=diagnostic, key=key):
                        bad = copy.deepcopy(c)
                        bad["proof_execution"][key] = value
                        with self.assertRaises(ValueError):
                            validate_capture(bad, entry)
                for key, value in (
                    ("builds", c["proof_owner_final"]["builds"] + 1),
                    ("reuses", 0),
                    ("owner_count", True),
                    ("retained_object_bytes", 0),
                    ("logical_limb_bytes", 0),
                    ("memory_scope", "peak_heap_bytes"),
                ):
                    bad = copy.deepcopy(c)
                    bad["proof_owner_final"][key] = value
                    with self.assertRaises(ValueError):
                        validate_capture(bad, entry)
                for key in (
                    "proof_execution",
                    "proof_owner_setup",
                    "proof_owner_final",
                    "fixed_plan_selected_kernel",
                ):
                    bad = copy.deepcopy(c)
                    bad.pop(key)
                    with self.assertRaises(ValueError):
                        validate_capture(bad, entry)
                bad = copy.deepcopy(c)
                bad["useful_integer_gemm_ops"] += 1
                with self.assertRaises(ValueError):
                    validate_capture(bad, entry)
                if diagnostic:
                    for field, value in (
                        ("range_proof_host_ns", True),
                        ("proof_identity_host_ns", -1),
                        ("proof_metadata_host_ns", 2**64),
                        ("range_proof_calls", 99),
                        ("graph_launches", 2),
                    ):
                        bad = copy.deepcopy(c)
                        bad["proof_host_diagnostic"]["warm_observation"][field] = value
                        with self.assertRaises(ValueError):
                            validate_capture(bad, entry)
                    for field, value in (
                        ("phase_time_contract", "additive"),
                        ("latency_evidence", True),
                        ("owner_before", None),
                        ("exact_reference_passed", False),
                    ):
                        bad = copy.deepcopy(c)
                        bad["proof_host_diagnostic"][field] = value
                        with self.assertRaises(ValueError):
                            validate_capture(bad, entry)

    @patch("collect_lift_recipe.manifest_provenance_complete", return_value=True)
    def test_bounded_short_k_and_compute_controls_are_explicit(self, _manifest):
        for mode in OWNER_MODES:
            for side, reduction in ((1024, 16), (512, 512), (16, 65537)):
                c = self.capture(mode, side=side, reduction_k=reduction, depth=1)
                self.assertEqual(validate_capture(c, dict(case=c["case"], run_id="fixture")), c)
                self.assertEqual(c["useful_integer_gemm_ops"], 2 * side * side * reduction)
                self.assertEqual(
                    c["proof_execution"]["initialization_products_charged_to_setup"], 0
                )
        for changes in (
            dict(mode="prepared-nodes", reduction_k=16),
            dict(mode="prepared-nodes", reduction_k=True),
            dict(mode="prepared-nodes", host_diagnostics=1),
            dict(mode="recipe-minimum", side=512),
            dict(mode="recipe-full", reduction_k=32),
            dict(mode="prepared-nodes", side=1024, reduction_k=1048576, depth=1),
            dict(mode="prepared-nodes", exact_proof_policy="approximate"),
            dict(mode="manual-scalar-first", exact_proof_policy="eager_axis"),
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_case(dict(self.case(), **changes))

    @patch("collect_lift_recipe.manifest_provenance_complete", return_value=True)
    def test_policy_variants_never_mix_repeats_and_fastest_manual_is_baseline(self, _manifest):
        observations = []
        # Omitted eager and explicit eager are the same route; scalar is separate.
        for mode, policy, latency in (
            ("manual-full", None, 10000),
            ("manual-scalar-first", None, 12000),
            ("resident-dag", None, 8000),
            ("resident-dag", "scalar_first", 7000),
        ):
            for run in range(2):
                changes = dict(exact_proof_policy=policy) if policy else {}
                if mode == "resident-dag" and policy is None and run:
                    changes["exact_proof_policy"] = "eager_axis"
                    changes["reduction_k"] = 32
                    changes["host_diagnostics"] = False
                c = self.capture(mode, **changes)
                for row in c["sustained"]:
                    row.update(total_us=latency * row["calls"], per_call_us=float(latency))
                # Keep every interval above the collector's minimum burst duration.
                for row in c["sustained"]:
                    row.update(calls=64, total_us=latency * 64)
                if mode in OWNER_MODES:
                    self.add_owner_fixture(c)
                c["execution_id"] = f"{mode}-{policy}-{run}"
                entry = dict(
                    case=c["case"],
                    run_id="fixture",
                    independent_run=run,
                    output=c["execution_id"] + ".json",
                )
                observations.append((entry, c))
        result = summarize(observations)
        self.assertEqual(len(result), 1)
        row = result[0]
        self.assertEqual(row["baseline"], "manual-full")
        self.assertEqual(
            set(row["median_process_metrics"]),
            {
                "manual-full",
                "manual-scalar-first",
                "resident-dag",
                "resident-dag@proof-scalar_first",
            },
        )
        self.assertFalse(row["matched_comparisons"]["resident-dag"]["changed_exact_proof_policy"])
        self.assertTrue(
            row["matched_comparisons"]["resident-dag@proof-scalar_first"][
                "changed_exact_proof_policy"
            ]
        )
        self.assertEqual(row["matched_comparisons"]["resident-dag"]["speedup"], 1.25)
        single_scalar = [
            (e, c)
            for e, c in observations
            if not (
                c["case"].get("exact_proof_policy") == "scalar_first" and e["independent_run"] == 1
            )
        ]
        with self.assertRaises(ValueError):
            summarize(single_scalar)
        diagnostic = self.capture("resident-graph", host_diagnostics=True)
        diagnostics = []
        for run in range(2):
            c = copy.deepcopy(diagnostic)
            c["execution_id"] = "diagnostic" + str(run)
            diagnostics.append(
                (dict(case=c["case"], run_id="fixture", independent_run=run, output=str(run)), c)
            )
        self.assertFalse(summarize(diagnostics)[0]["production_latency_eligible"])

    @patch("collect_lift_recipe.manifest_provenance_complete", return_value=True)
    def test_ordered_recipe_requires_actual_completion_and_proof_receipts(self, _manifest):
        for policy in (0, 1):
            c = self.ordered_capture(policy)
            entry = dict(case=c["case"], run_id="fixture")
            self.assertEqual(validate_capture(c, entry), c)
            for key, value in [
                ("policy", 1 - policy),
                ("stream_wait_calls", 2),
                ("ordered_gemm_calls", True),
                ("ordered_lift_calls", 7),
                ("ordered_copy_calls", 2),
                ("ordered_conversion_calls", 1),
                ("ordered_status_readback_bytes", 0 if policy else 4),
                ("retained_layout_host_bytes", 0),
                ("retained_control_device_bytes", 0 if policy else 4),
                ("retained_control_pinned_bytes", 8),
                ("completion_contract", "device_wide"),
                ("planning_price_scope", "ordered_optimal"),
            ]:
                with self.subTest(policy=policy, key=key):
                    bad = copy.deepcopy(c)
                    bad["recipe_execution"][key] = value
                    with self.assertRaises(ValueError):
                        validate_capture(bad, entry)
            for key in ("recipe_execution", "hot_loop_serialization"):
                bad = copy.deepcopy(c)
                bad.pop(key)
                with self.assertRaises(ValueError):
                    validate_capture(bad, entry)
        for value in (True, -1, 2, "1"):
            with self.assertRaises(ValueError):
                validate_case(dict(self.case(), execution_policy=value))
        with self.assertRaises(ValueError):
            validate_case(dict(self.case(), mode="manual-full", execution_policy=0))

    @patch("collect_lift_recipe.manifest_provenance_complete", return_value=True)
    def test_ordered_full_prefix_has_no_proof_flag_and_matched_policy_comparison(self, _manifest):
        c = self.ordered_capture()
        c["case"]["mode"] = "recipe-full"
        c["selected_compute_prefixes"] = [20] * 3
        receipt = c["recipe_execution"]
        receipt.update(
            guarded_layout_products=False,
            retained_control_device_bytes=0,
            retained_control_pinned_bytes=0,
            ordered_lift_calls=0,
            ordered_status_readback_bytes=0,
        )
        self.assertEqual(validate_capture(c, dict(case=c["case"], run_id="fixture")), c)
        observations = []
        for policy in (0, 1):
            for run in range(2):
                capture = self.ordered_capture(policy)
                capture["execution_id"] = f"policy{policy}-run{run}"
                for sample in capture["sustained"]:
                    sample["total_us"] *= 2 if not policy else 1
                    sample["per_call_us"] = sample["total_us"] / sample["calls"]
                entry = dict(
                    case=capture["case"],
                    run_id="fixture",
                    independent_run=run,
                    output=f"{policy}-{run}.json",
                )
                observations.append((entry, capture))
        cohorts = summarize(observations)
        self.assertEqual(len(cohorts), 1)
        pair = cohorts[0]["execution_policy_comparisons"]["recipe-minimum"]
        self.assertEqual(pair["speedup"], 2)
        self.assertEqual(pair["cost_ratio"], 0.5)
        # Historical JSON-bearing intervals cannot be pooled with corrected captures.
        for run in range(2):
            legacy = self.capture()
            legacy["execution_id"] = f"legacy{run}"
            observations.append(
                (
                    dict(
                        case=legacy["case"],
                        run_id="fixture",
                        independent_run=run,
                        output=f"old{run}.json",
                    ),
                    legacy,
                )
            )
        self.assertEqual(len(summarize(observations)), 2)

    def test_chain_diamond_bounds_and_noop_lift(self):
        c = self.case()
        self.assertEqual(bounds(c), [1, 1, 32, 1024, 32768])
        self.assertEqual(validate_case(c), c)
        c.update(topology="diamond", depth=4)
        self.assertEqual(bounds(c)[-1], 32**5)
        self.assertEqual(validate_case(c), c)
        c = self.case()
        c.update(mode="lift-graph", source_prefix=20)
        self.assertEqual(validate_case(c), c)
        for key, value in [
            ("side", True),
            ("prefix", 1),
            ("source_prefix", 21),
            ("max_payload_bytes", 64),
        ]:
            bad = self.case()
            bad[key] = value
            with self.assertRaises(ValueError):
                validate_case(bad)

    @patch("collect_lift_recipe.manifest_provenance_complete", return_value=True)
    def test_real_retention_and_independent_prefix_proof(self, _manifest):
        e = dict(case=self.case(), run_id="fixture")
        c = self.capture()
        self.assertEqual(validate_capture(c, e), c)
        for key, value in [
            ("replay_hip_allocations", 6),
            ("replay_hip_frees", 6),
            ("selected_compute_prefixes", [1, 1, 1]),
            ("extra_intermediate_observations", True),
            ("promotion_eligible", True),
            ("cost_model", {"qualified": True}),
        ]:
            bad = copy.deepcopy(c)
            bad[key] = value
            with self.assertRaises(ValueError):
                validate_capture(bad, e)
        bad = copy.deepcopy(c)
        bad["samples"][0]["total_us"] = 1
        with self.assertRaises(ValueError):
            validate_capture(bad, e)

    @patch("collect_lift_recipe.manifest_provenance_complete", return_value=True)
    def test_packed_trace_replays_growth_lifetimes_costs_and_budget(self, _manifest):
        c = self.capture()
        c["case"]["mode"] = "recipe-packed-minimum"
        c["recipe_arena_bytes"] = 65536
        c["cost_model"] = dict(
            kind="explicit-unqualified-layout-work-model-v3-blocked-k65536",
            launch_ns=5000,
            gemm_units="one_ns_per_scalar_integer_multiply",
            reduction_units="one_ns_per_safe_k_slab",
            arithmetic_policy="centered-byte-blocked-k65536-v2",
            packed_gemm_units="one_ns_per_four_term_group",
            pack_units="one_ns_per_logical_byte_read_or_write",
            cost_schema_version=3,
            packed_cost_capacity="physical_shared_b_prefix",
            qualified=False,
        )
        view = dict(
            value_id=1, capacity_prefix=3, first_reader=0, last_reader=2, bytes=3072, arena_offset=0
        )
        c["materialization_plan"] = dict(
            packed_b=[1, 1, 1],
            packed_values=[view],
            evaluated=4096,
            exhaustive=False,
            chain_dynamic_programming=True,
            cost_optimal=False,
            estimated_ns=206072,
            completed_packing_calls=3,
        )
        replay_receipt(c)
        e = dict(case=c["case"], run_id="fixture")
        self.assertEqual(validate_capture(c, e), c)
        for key, value in [
            ("completed_packing_calls", 1),
            ("estimated_ns", 206071),
            ("packed_b", [1, 0, 1]),
            ("evaluated", True),
        ]:
            bad = copy.deepcopy(c)
            bad["materialization_plan"][key] = value
            with self.assertRaises(ValueError):
                validate_capture(bad, e)
        for key, value in [
            ("last_reader", 1),
            ("bytes", 3071),
            ("arena_offset", 1),
            ("capacity_prefix", 2),
        ]:
            bad = copy.deepcopy(c)
            bad["materialization_plan"]["packed_values"][0][key] = value
            with self.assertRaises(ValueError):
                validate_capture(bad, e)
        bad = copy.deepcopy(c)
        bad["materialization_plan"]["packed_values"].append(copy.deepcopy(view))
        with self.assertRaises(ValueError):
            validate_capture(bad, e)
        full = copy.deepcopy(c)
        full["case"]["mode"] = "recipe-packed-full"
        full["selected_compute_prefixes"] = [20] * 3
        full["materialization_plan"].update(estimated_ns=613920, completed_packing_calls=1)
        full["materialization_plan"]["packed_values"][0].update(capacity_prefix=20, bytes=20480)
        replay_receipt(full)
        self.assertEqual(validate_capture(full, dict(case=full["case"], run_id="fixture")), full)
        bad = copy.deepcopy(c)
        del bad["materialization_plan"]
        with self.assertRaises(ValueError):
            validate_capture(bad, e)

    def test_native64_intermediate_admission_keeps_wide_final(self):
        for unsigned, side, allowed_depth, rejected_depth in [
            (False, 128, 9, 10),
            (True, 256, 8, 9),
        ]:
            c = self.case()
            c.update(
                mode="native-roundtrip", side=side, depth=allowed_depth, unsigned_inputs=unsigned
            )
            self.assertGreaterEqual(bounds(c)[-1], 2 ** (64 if unsigned else 63))
            self.assertEqual(validate_case(c), c)
            c["depth"] = rejected_depth
            with self.assertRaisesRegex(ValueError, "intermediate exceeds native64"):
                validate_case(c)
        c = self.case()
        c.update(mode="native-roundtrip", depth=1)
        self.assertEqual(native_payload(validate_case(c)), (2 * 32**2 * 8, 0, 2 * 32**2 * 8))
        with patch("collect_lift_recipe.graph", return_value=[(0, 0), (2, 1)]):
            self.assertEqual(native_payload(c), (3 * 32**2 * 8, 32**2 * 8, 3 * 32**2 * 8))

    @patch("collect_lift_recipe.manifest_provenance_complete", return_value=True)
    def test_native_roundtrip_logical_and_actual_transfer_receipts(self, _manifest):
        c = self.capture("native-roundtrip")
        e = dict(case=c["case"], run_id="fixture")
        self.assertEqual(validate_capture(c, e), c)
        self.assertNotEqual(
            c["replay_hip_h2d_bytes"], c["native_roundtrip_h2d_payload_bytes_per_replay"]
        )
        for key, value in [
            ("native_roundtrip_control", False),
            ("extra_intermediate_observations", True),
            ("native_roundtrip_h2d_payload_bytes_per_replay", 0),
            ("native_roundtrip_d2h_payload_bytes_per_replay", 0),
            ("native_roundtrip_retained_host_bytes", 0),
            ("host_payload_bytes_before_final_export", 0),
            ("replay_hip_counter_samples", 1),
            ("replay_hip_h2d_calls", 0),
            ("replay_hip_d2h_calls", 0),
            ("replay_hip_h2d_bytes", False),
            ("replay_hip_d2h_bytes", -1),
            ("selected_compute_prefixes", [1, 2, 3]),
            ("schema", "rns8_lift_recipe_microcase_v2"),
        ]:
            bad = copy.deepcopy(c)
            bad[key] = value
            with self.assertRaises(ValueError):
                validate_capture(bad, e)
        for key in (
            "native_roundtrip_retained_host_bytes",
            "replay_hip_h2d_bytes",
            "replay_hip_counter_samples",
        ):
            bad = copy.deepcopy(c)
            del bad[key]
            with self.assertRaises(ValueError):
                validate_capture(bad, e)
        one = self.capture("native-roundtrip", depth=1)
        self.assertEqual(validate_capture(one, dict(case=one["case"], run_id="fixture")), one)
        diamond = self.capture(
            "native-roundtrip", depth=4, topology="diamond", unsigned_inputs=True
        )
        self.assertEqual(
            validate_capture(diamond, dict(case=diamond["case"], run_id="fixture")), diamond
        )

    @patch("collect_lift_recipe.manifest_provenance_complete", return_value=True)
    def test_scalar_control_and_native_internal_exports_share_final_output_cohort(self, _manifest):
        observations = []
        for mode in ("manual-full", "manual-scalar-first", "native-roundtrip", "eager-observe"):
            for run in range(2):
                c = self.capture(mode)
                c["execution_id"] = mode + str(run)
                e = dict(
                    case=c["case"],
                    run_id="fixture",
                    independent_run=run,
                    output=mode + str(run) + ".json",
                )
                observations.append((e, c))
        cohorts = summarize(observations)
        self.assertEqual(len(cohorts), 2)
        same = next(c for c in cohorts if c["baseline"] == "manual-scalar-first")
        self.assertEqual(
            set(same["median_process_metrics"]),
            {"manual-full", "manual-scalar-first", "native-roundtrip"},
        )
        self.assertIn("native-roundtrip", same["matched_comparisons"])
        eager = next(c for c in cohorts if c["identity"]["extra_intermediate_observations"])
        self.assertEqual(set(eager["median_process_metrics"]), {"eager-observe"})
        c = self.capture("manual-scalar-first")
        c["fixed_plan_exact_proof_policy"] = "eager_axis"
        with self.assertRaises(ValueError):
            validate_capture(c, dict(case=c["case"], run_id="fixture"))

    @patch("collect_lift_recipe.manifest_provenance_complete", return_value=True)
    def test_no_unmatched_or_single_process_speedup(self, _manifest):
        e = dict(case=self.case(), run_id="fixture", independent_run=0, output="a.json")
        c = self.capture()
        with self.assertRaises(ValueError):
            summarize([(e, c)])
        d = copy.deepcopy(c)
        d["execution_id"] = "process2"
        r = summarize([(e, c), (dict(e, independent_run=1), d)])
        self.assertEqual(r[0]["matched_comparisons"], {})
        self.assertIsNone(r[0]["baseline"])


class ReplayPolicyContracts(unittest.TestCase):
    def test_dataset_signedness_identity_and_propagation(self):
        from product_dataset import inputs, identity

        base = LiftRecipeContracts().case()
        self.assertEqual(inputs(base), [1, 1])
        changed = dict(base, input_values=[-2, 3])
        self.assertEqual(bounds(changed), [2, 3, 192, 18432, 1769472])
        self.assertNotEqual(identity(base), identity(changed))
        self.assertEqual(validate_case(changed), changed)
        for values in ([True, 1], [2**63, 1], [1], [-1, 1]):
            bad = dict(base, input_values=values, unsigned_inputs=True)
            if values == [2**63, 1]:
                bad["unsigned_inputs"] = False
            with self.assertRaises(ValueError):
                validate_case(bad)

    def test_cold_steady_training_strict_payback_and_unknown(self):
        from product_replay import policy, estimate, LIMIT

        p = policy(
            dict(
                replay_policy=dict(
                    expected_replays=100,
                    retain_external_b=True,
                    preparation_ns=200,
                    calibration_ns=1,
                    orchestration_ns=7,
                    publication_ns=11,
                    baseline_replay_ns=300,
                )
            ),
            training_ns=500,
        )
        r = estimate(p, 1030, 30)
        self.assertEqual(
            (r["first_ns"], r["steady_ns"], r["once_ns"], r["lifetime_ns"], r["amortized_ns"]),
            (1048, 48, 700, 6500, 65),
        )
        self.assertEqual(r["break_even_replays"], 7)
        self.assertEqual(r["amortization_status"], 1)
        p["expected_replays"] = 6
        self.assertEqual(estimate(p, 1030, 30)["amortization_status"], 2)
        p["baseline_replay_ns"] = LIMIT
        self.assertEqual(estimate(p, 1030, 30)["amortization_status"], 0)
        for extra in (
            dict(expected_replays=0),
            dict(expected_replays=True),
            dict(retain_external_b=1),
            dict(unknown=1),
        ):
            with self.assertRaises(ValueError):
                policy(dict(replay_policy=extra))
        p["calibration_ns"] = LIMIT
        with self.assertRaises(ValueError):
            estimate(p, 1030, 30)


if __name__ == "__main__":
    unittest.main()
