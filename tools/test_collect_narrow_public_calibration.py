from benchmark_cases import case_root
import copy
import json
import unittest
from collect_narrow_public_calibration import validate_case, validate_capture, selected_prefix
from test_capture_identity_support import synthetic_identity


def synthetic_narrow_summary(c):
    case = c["case"]
    narrow = case["mode"].startswith(("scalar-", "wmma-"))
    cold = case["residency"] == "cold"
    c["separate_native_observation"] = None
    if narrow:
        fused = case["mode"].endswith("-fused")
        wmma = case["mode"].startswith("wmma-")
        product = (
            ("native_byte_wmma_fused_rns" if fused else "native_byte_wmma_gemm")
            if wmma
            else (
                "narrow_native_i64_fused_rns_kernel" if fused else "narrow_native_i64_gemm_kernel"
            )
        )
        split_k = case["mode"].startswith("scalar-split-k-")
        labels = (
            {"narrow_native_split_k_partials": 1, "narrow_native_split_k_encode": 1}
            if split_k
            else {product: 1}
        )
        if not fused:
            labels["narrow_native_encode_rns"] = 1
        if cold and wmma:
            labels["native_wmma_prepare_capability"] = 1
        c["separate_native_observation"] = dict(
            scope="one_separate_complete_public_call",
            inside_timed_samples=False,
            inside_sustained_interval=False,
            public_calls=1,
            exact_reference_passed=True,
            native_event_receipts_complete=True,
            copy_scope="tracked_completed_synchronous_HIP_copy_wrappers",
            allocation_epoch=1,
            device_allocation_calls=0,
            device_free_calls=0,
            h2d_calls=0,
            h2d_bytes=0,
            d2h_calls=0,
            d2h_bytes=0,
            event_labels=labels,
            event_microseconds={key: 1.5 for key in labels},
        )
    c["narrow_summary_scope"] = (
        "not_applicable"
        if not narrow
        else "cold_workspace_not_observed"
        if cold
        else "completed_retained_workspace_snapshots_outside_timing"
    )
    c["narrow_summary_first"] = c["narrow_summary_final"] = None
    if not narrow or cold:
        return
    calls = 1 + case["warmups"] + case["repeats"] + sum(b["calls"] for b in c["sustained"])
    for name, count in (("narrow_summary_first", 1), ("narrow_summary_final", calls)):
        repack_a = case["residency"] in ("repack", "repack-a")
        repack_b = case["residency"] in ("repack", "repack-b")
        wmma = int(case["mode"].startswith("wmma-"))
        c[name] = dict(
            retained_scan_bytes=0,
            retained_product_bytes=case["m"] * case["n"] * ((case["k"] + 4095) // 4096) * 8
            if case["mode"].startswith("scalar-split-k-")
            else 0
            if case["mode"].endswith("-fused")
            else case["m"] * case["n"] * 8,
            wmma_capability_observed=wmma,
            wmma_capability_supported=wmma,
            a_instance=11,
            b_instance=12,
            a_generation=1 + count if repack_a else 1,
            b_generation=1 + count if repack_b else 1,
            a_device_scans=0,
            b_device_scans=0,
            a_reuses=0 if repack_a else count - 1,
            b_reuses=0 if repack_b else count - 1,
            a_host_proofs=count if repack_a else 1,
            b_host_proofs=count if repack_b else 1,
        )


class NarrowPublicCollection(unittest.TestCase):
    def test_checked_in_adaptive_cases_are_individual_matched_collector_inputs(self):
        folder = case_root() / "narrow-public/adaptive-summaries"
        cases = [
            validate_case(json.loads(path.read_text())) for path in sorted(folder.glob("*.json"))
        ]
        self.assertEqual(len(cases), 8)
        groups = {}
        for case in cases:
            key = json.dumps({k: v for k, v in case.items() if k != "mode"}, sort_keys=True)
            groups.setdefault(key, set()).add(case["mode"])
        self.assertEqual(len(groups), 4)
        self.assertTrue(
            all(modes == {"adaptive-fresh", "adaptive-cached"} for modes in groups.values())
        )

    def case(self):
        return dict(
            schema="narrow_public_case_v1",
            m=17,
            n=19,
            k=31,
            prefix=0,
            unsigned_inputs=False,
            mode="scalar-fused",
            residency="resident",
            repeats=3,
            warmups=1,
            seed=5353,
        )

    def test_resident_shortlist_has_best_same_output_controls(self):
        folder = case_root() / "narrow-public/adaptive-summaries-resident"
        cases = [
            validate_case(json.loads(path.read_text())) for path in sorted(folder.glob("*.json"))
        ]
        self.assertEqual(len(cases), 19)
        groups = {}
        for case in cases:
            key = (case["m"], case["n"], case["k"], case["unsigned_inputs"])
            groups.setdefault(key, set()).add(case["mode"])
        self.assertEqual(len(groups), 4)
        for key, modes in groups.items():
            expected = {
                "adaptive-resident-fresh",
                "adaptive-resident-cached",
                "wmma-resident-fused",
                "rns-resident-public",
            }
            if not key[3]:
                expected.add("scalar-resident-fused")
            self.assertEqual(modes, expected)

    def capture(self):
        case = self.case()
        entry = dict(case=case, run_id="synthetic-not-hardware-evidence")
        sample = dict(prepare_us=1, execute_us=2, export_us=3, cleanup_us=0, total_us=6)
        c = dict(
            schema="rns8_public_narrow_microcase_v2",
            case=case,
            run_id=entry["run_id"],
            dataset="affine_native_small_v1",
            selected_prefix=2,
            narrow_producer="scalar-fused",
            narrow_input_admission_policy="generation_bound_host_pack_certificate",
            narrow_product_storage="fused_no_native_product",
            known_narrow_scratch_bytes=16,
            output_boundary="complete_public_product_and_padded_three_limb_integer_adapter",
            vector_export_adapter="not_applicable",
            host_instrumentation_state="hooks_disabled",
            exact_reference_passed=True,
            all_residue_planes_audited=True,
            residue_audit_scope="post_timing_public_product",
            oracle_inside_production_interval=False,
            promotion_eligible=False,
            native_useful_products=17 * 19 * 31,
            ordinary_plan_kernel="direct_hip_selected_prefix_grouped_rns_gemm_v1",
            setup_us=1,
            first_call=sample,
            samples=[copy.deepcopy(sample) for _ in range(3)],
            sustained=[
                dict(calls=16, total_us=250000, per_call_us=15625, minimum_duration_reached=True)
                for _ in range(3)
            ],
        )
        synthetic_identity(c)
        synthetic_narrow_summary(c)
        return c, entry

    def test_complete_public_boundary_not_private_native_output(self):
        c, entry = self.capture()
        self.assertEqual(validate_capture(c, entry), c)
        for key, value in [
            ("schema", "rns8_public_narrow_microcase_v1"),
            ("narrow_input_admission_policy", "device_scan_every_call"),
            ("narrow_product_storage", "unobserved_transient_product"),
            ("selected_prefix", 9),
            ("known_narrow_scratch_bytes", 0),
            ("all_residue_planes_audited", False),
            ("output_boundary", "native_kernel_only"),
            ("promotion_eligible", True),
            ("vector_export_adapter", "hidden"),
        ]:
            bad = copy.deepcopy(c)
            bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_capture(bad, entry)

    def test_prefix_and_explicit_control_contracts(self):
        c = self.case()
        self.assertEqual(selected_prefix(c), 2)
        c["prefix"] = 1
        with self.assertRaises(ValueError):
            validate_case(c)

        c = self.case()
        c["unsigned_inputs"] = True
        with self.assertRaises(ValueError):
            validate_case(c)
        c = self.case()
        c.update(mode="vector-exact-width", n=3, k=512)
        self.assertEqual(validate_case(c), c)
        c["k"] = 511
        with self.assertRaises(ValueError):
            validate_case(c)
        c = self.case()
        c["mode"] = "rns-parallel"
        with self.assertRaises(ValueError):
            validate_case(c)

    def test_staged_owner_storage_is_part_of_capture_identity(self):
        for mode in ("scalar-staged", "wmma-staged"):
            c, entry = self.capture()
            entry["case"]["mode"] = mode
            c.update(
                narrow_producer=mode,
                narrow_product_storage="workspace_retained_native_product",
                known_narrow_scratch_bytes=(32 if mode.startswith("wmma") else 16) + 17 * 19 * 8,
            )
            synthetic_narrow_summary(c)
            self.assertEqual(validate_capture(c, entry), c)
            c["narrow_product_storage"] = "fused_no_native_product"
            with self.assertRaises(ValueError):
                validate_capture(c, entry)

    def test_split_k_partials_are_distinct_from_zero_scratch_fusion(self):
        for k in (4095, 4096, 4097, 65535, 65536, 65537):
            c, entry = self.capture()
            entry["case"].update(mode="scalar-split-k-fused", k=k)
            parts = (k + 4095) // 4096
            c.update(
                narrow_producer="scalar-split-k-fused",
                narrow_product_storage="workspace_retained_split_k_partials",
                known_narrow_scratch_bytes=16 + 17 * 19 * parts * 8,
                native_useful_products=17 * 19 * k,
                selected_prefix=selected_prefix(entry["case"]),
            )
            synthetic_narrow_summary(c)
            self.assertEqual(validate_capture(c, entry), c)
            bad = copy.deepcopy(c)
            bad["narrow_summary_final"]["retained_product_bytes"] = 0
            with self.assertRaises(ValueError):
                validate_capture(bad, entry)
            bad = copy.deepcopy(c)
            observation = bad["separate_native_observation"]
            observation["event_labels"].pop("narrow_native_split_k_partials")
            observation["event_microseconds"].pop("narrow_native_split_k_partials")
            with self.assertRaises(ValueError):
                validate_capture(bad, entry)

    def test_resident_output_is_separate_from_host_adapter_boundary(self):
        for mode in ["rns-resident-public", "scalar-resident-fused", "wmma-resident-fused"]:
            c, entry = self.capture()
            entry["case"]["mode"] = mode
            c.update(
                narrow_producer=mode if mode != "rns-resident-public" else "not_applicable",
                narrow_input_admission_policy="not_applicable"
                if mode == "rns-resident-public"
                else "generation_bound_host_pack_certificate",
                narrow_product_storage="not_applicable"
                if mode == "rns-resident-public"
                else "fused_no_native_product",
                known_narrow_scratch_bytes=32
                if mode.startswith("wmma-")
                else 16
                if mode.startswith("scalar-")
                else 0,
                output_boundary="complete_public_product_resident_rns",
                timed_host_export=False,
                oracle_export_outside_production_interval=True,
                exact_proof_policy="scalar_first",
            )
            for sample in [c["first_call"], *c["samples"]]:
                sample.update(export_us=0, total_us=3)
            synthetic_narrow_summary(c)
            self.assertEqual(validate_capture(c, entry), c)
            for key, value in [
                ("timed_host_export", True),
                ("oracle_export_outside_production_interval", False),
                (
                    "output_boundary",
                    "complete_public_product_and_padded_three_limb_integer_adapter",
                ),
                ("exact_proof_policy", "default_eager"),
            ]:
                bad = copy.deepcopy(c)
                bad[key] = value
                with self.subTest(mode=mode, key=key), self.assertRaises(ValueError):
                    validate_capture(bad, entry)
            bad = copy.deepcopy(c)
            bad["first_call"].update(export_us=1, total_us=4)
            with self.assertRaises(ValueError):
                validate_capture(bad, entry)

    def test_generation_bound_summary_receipts_charge_each_completed_call(self):
        for mode in (
            "adaptive-fresh",
            "adaptive-cached",
            "adaptive-resident-fresh",
            "adaptive-resident-cached",
        ):
            cached = mode.endswith("-cached")
            resident = "-resident-" in mode
            c, entry = self.capture()
            entry["case"]["mode"] = mode
            c.update(
                narrow_producer="not_applicable",
                narrow_input_admission_policy="not_applicable",
                narrow_product_storage="not_applicable",
                known_narrow_scratch_bytes=0,
                adaptive_summary_policy="generation_bound" if cached else "fresh_scan",
                adaptive_method="axis_l1",
                adaptive_group=2,
                adaptive_workers=32,
                exact_proof_policy="scalar_first",
            )
            first = dict(
                retained_bytes=96 if cached else 0,
                a_instance=1 if cached else 0,
                a_generation=int(cached),
                b_instance=2 if cached else 0,
                b_generation=int(cached),
                a_scans=int(cached),
                b_scans=int(cached),
                a_reuses=0,
                b_reuses=0,
            )
            last = dict(first, a_reuses=52 if cached else 0, b_reuses=52 if cached else 0)
            c.update(adaptive_summary_first=first, adaptive_summary_final=last)
            synthetic_narrow_summary(c)
            if resident:
                c.update(
                    output_boundary="complete_public_product_resident_rns",
                    timed_host_export=False,
                    oracle_export_outside_production_interval=True,
                )
                for sample in [c["first_call"], *c["samples"]]:
                    sample.update(export_us=0, total_us=3)
            self.assertEqual(validate_capture(c, entry), c)
            for key, value in [
                ("a_scans", 2),
                ("b_reuses", 0 if cached else 1),
                ("retained_bytes", 0 if cached else 96),
                ("a_generation", 3),
            ]:
                bad = copy.deepcopy(c)
                bad["adaptive_summary_final"][key] = value
                with self.subTest(cached=cached, key=key), self.assertRaises(ValueError):
                    validate_capture(bad, entry)
            bad = copy.deepcopy(entry["case"])
            bad["residency"] = "repack"
            with self.assertRaises(ValueError):
                validate_case(bad)

    def test_separate_native_observations_do_not_inherit_production_timing(self):
        c, entry = self.capture()
        for key, value in (
            ("inside_timed_samples", True),
            ("public_calls", True),
            ("native_event_receipts_complete", False),
            ("allocation_epoch", 0),
            ("h2d_calls", -1),
            ("h2d_bytes", 1),
            ("event_microseconds", {}),
            ("event_labels", {"narrow_native_domain_scan": 1}),
        ):
            bad = copy.deepcopy(c)
            bad["separate_native_observation"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_capture(bad, entry)
        bad = copy.deepcopy(c)
        bad["separate_native_observation"]["event_microseconds"][
            "narrow_native_i64_fused_rns_kernel"
        ] = True
        with self.assertRaises(ValueError):
            validate_capture(bad, entry)

    def test_narrow_proof_snapshots_cover_retained_repack_and_unobserved_cold(self):
        for residency in ("resident", "repack", "repack-a", "repack-b", "cold"):
            c, entry = self.capture()
            entry["case"]["residency"] = residency
            synthetic_narrow_summary(c)
            self.assertEqual(validate_capture(c, entry), c)
            if residency == "cold":
                c["narrow_summary_first"] = {}
                with self.assertRaises(ValueError):
                    validate_capture(c, entry)
                continue
            for key, value in (
                ("a_device_scans", 1),
                ("b_host_proofs", 0),
                ("a_reuses", True),
                ("a_generation", 0),
                ("retained_scan_bytes", 16),
                ("retained_product_bytes", 8),
                ("wmma_capability_observed", 1),
                ("wmma_capability_supported", True),
                ("b_instance", 11),
                ("a_instance", 13),
                ("b_reuses", 1 << 64),
            ):
                bad = copy.deepcopy(c)
                bad["narrow_summary_final"][key] = value
                with self.subTest(residency=residency, key=key), self.assertRaises(ValueError):
                    validate_capture(bad, entry)
            c["narrow_summary_scope"] = "inside_production"
            with self.assertRaises(ValueError):
                validate_capture(c, entry)


if __name__ == "__main__":
    unittest.main()
