import copy
import json
from pathlib import Path
import unittest
from collect_narrow_public_calibration import validate_case, validate_capture, selected_prefix
from test_capture_identity_support import synthetic_identity


class NarrowPublicCollection(unittest.TestCase):
    def test_checked_in_adaptive_cases_are_individual_matched_collector_inputs(self):
        folder=Path(__file__).resolve().parents[1]/'benchmarks/narrow-public/adaptive-summaries'
        cases=[validate_case(json.loads(path.read_text())) for path in sorted(folder.glob('*.json'))]
        self.assertEqual(len(cases),8)
        groups={}
        for case in cases:
            key=json.dumps({k:v for k,v in case.items() if k!='mode'},sort_keys=True)
            groups.setdefault(key,set()).add(case['mode'])
        self.assertEqual(len(groups),4)
        self.assertTrue(all(modes=={'adaptive-fresh','adaptive-cached'} for modes in groups.values()))

    def case(self):
        return dict(schema='narrow_public_case_v1', m=17, n=19, k=31, prefix=0, unsigned_inputs=False, mode='scalar-fused', residency='resident', repeats=3, warmups=1, seed=5353)

    def capture(self):
        case = self.case(); entry = dict(case=case, run_id='synthetic-not-hardware-evidence'); sample = dict(prepare_us=1, execute_us=2, export_us=3, cleanup_us=0, total_us=6)
        c = dict(schema='rns8_public_narrow_microcase_v1', case=case, run_id=entry['run_id'], dataset='affine_native_small_v1', selected_prefix=2,
                 narrow_producer='scalar-fused', known_narrow_scratch_bytes=16, output_boundary='complete_public_product_and_padded_three_limb_integer_adapter',
                 vector_export_adapter='not_applicable', host_instrumentation_state='hooks_disabled', exact_reference_passed=True,
                 all_residue_planes_audited=True, residue_audit_scope='post_timing_public_product', oracle_inside_production_interval=False,
                 promotion_eligible=False, native_useful_products=17*19*31, ordinary_plan_kernel='direct_hip_selected_prefix_grouped_rns_gemm_v1',
                 setup_us=1, first_call=sample, samples=[copy.deepcopy(sample) for _ in range(3)],
                 sustained=[dict(calls=16, total_us=250000, per_call_us=15625, minimum_duration_reached=True) for _ in range(3)])
        synthetic_identity(c); return c, entry

    def test_complete_public_boundary_not_private_native_output(self):
        c, entry = self.capture(); self.assertEqual(validate_capture(c, entry), c)
        for key, value in [('selected_prefix', 9), ('known_narrow_scratch_bytes', 0), ('all_residue_planes_audited', False),
                           ('output_boundary', 'native_kernel_only'), ('promotion_eligible', True), ('vector_export_adapter', 'hidden')]:
            bad = copy.deepcopy(c); bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_capture(bad, entry)

    def test_prefix_and_explicit_control_contracts(self):
        c = self.case(); self.assertEqual(selected_prefix(c), 2)
        c['prefix'] = 1
        with self.assertRaises(ValueError): validate_case(c)

        c = self.case(); c['unsigned_inputs'] = True
        with self.assertRaises(ValueError): validate_case(c)
        c = self.case(); c.update(mode='vector-exact-width', n=3, k=512); self.assertEqual(validate_case(c), c)
        c['k'] = 511
        with self.assertRaises(ValueError): validate_case(c)
        c = self.case(); c['mode'] = 'rns-parallel'
        with self.assertRaises(ValueError): validate_case(c)

    def test_generation_bound_summary_receipts_charge_each_completed_call(self):
        for cached in (False, True):
            c, entry = self.capture()
            entry['case']['mode'] = 'adaptive-cached' if cached else 'adaptive-fresh'
            c.update(narrow_producer='not_applicable', known_narrow_scratch_bytes=0,
                     adaptive_summary_policy='generation_bound' if cached else 'fresh_scan',
                     adaptive_method='axis_l1', adaptive_group=2, adaptive_workers=32, exact_proof_policy='scalar_first')
            first = dict(retained_bytes=96 if cached else 0, a_instance=1 if cached else 0,
                         a_generation=int(cached), b_instance=2 if cached else 0, b_generation=int(cached),
                         a_scans=int(cached), b_scans=int(cached), a_reuses=0, b_reuses=0)
            last = dict(first, a_reuses=52 if cached else 0, b_reuses=52 if cached else 0)
            c.update(adaptive_summary_first=first, adaptive_summary_final=last)
            self.assertEqual(validate_capture(c, entry), c)
            for key, value in [('a_scans', 2), ('b_reuses', 0 if cached else 1), ('retained_bytes', 0 if cached else 96), ('a_generation', 3)]:
                bad=copy.deepcopy(c); bad['adaptive_summary_final'][key]=value
                with self.subTest(cached=cached, key=key), self.assertRaises(ValueError):
                    validate_capture(bad,entry)
            bad=copy.deepcopy(entry['case']); bad['residency']='repack'
            with self.assertRaises(ValueError): validate_case(bad)



if __name__ == '__main__':
    unittest.main()
