import copy
import json
from pathlib import Path
import tempfile
import unittest
from collect_int4_calibration import collection_plan, validate_case, validate_capture
from test_capture_identity_support import synthetic_identity


class Int4Collection(unittest.TestCase):
    def case(self):
        return dict(schema='finite_int4_public_case_v1', m=17, n=19, k=65537, modulus=16,
                    mode='packed-b', residency='repeated-b', repeats=3, warmups=1, seed=4646)

    def capture(self):
        case = self.case(); entry = dict(case=case, run_id='synthetic-not-evidence')
        sample = dict(prepare_us=1, execute_us=2, export_us=3, cleanup_us=0, total_us=6)
        c = dict(schema='rns8_public_int4_microcase_v1', case=case, run_id=entry['run_id'], dataset='affine_finite_bytes_v1',
                 host_instrumentation_state='hooks_disabled', output_boundary='complete_public_finite_product_and_padded_export',
                 exact_reference_passed=True, oracle_inside_production_interval=False, promotion_eligible=False,
                 native_useful_products=case['m']*case['n']*case['k'], signed_nibble_operands=True, layout_version=1,
                 retained_packed_b_device_bytes=((case['k']+15)//16)*case['n']*8, certificate_sha256='a'*64,
                 target='gfx1100', setup_us=12, first_call=sample, samples=[copy.deepcopy(sample) for _ in range(3)],
                 sustained=[dict(calls=16, total_us=250000, per_call_us=15625, minimum_duration_reached=True) for _ in range(3)])
        synthetic_identity(c)
        return c, entry

    def test_complete_receipt_and_forged_amortization_rejected(self):
        c, entry = self.capture(); self.assertEqual(validate_capture(c, entry), c)
        for key, value in [('retained_packed_b_device_bytes', 0), ('signed_nibble_operands', False),
                           ('target', 'gfx942'), ('promotion_eligible', True), ('oracle_inside_production_interval', True)]:
            bad = copy.deepcopy(c); bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_capture(bad, entry)
        for key in ('prepare_us', 'export_us'):
            bad = copy.deepcopy(c); bad['samples'][0][key] = 0
            with self.assertRaises(ValueError):
                validate_capture(bad, entry)
        bad = copy.deepcopy(c); bad['sustained'][0]['calls'] = 32
        with self.assertRaises(ValueError):
            validate_capture(bad, entry)

    def test_cases_and_counterbalanced_independent_processes(self):
        for key, value in [('modulus', 17), ('mode', 'unsigned'), ('residency', 'hidden-repack'), ('m', True)]:
            c = self.case(); c[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_case(c)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); paths = []
            for mode in ('int8', 'onthefly', 'packed-b'):
                c = self.case(); c['mode'] = mode; path = root/(mode+'.json'); path.write_text(json.dumps(c)); paths.append(path)
            plan = collection_plan(Path('binary'), paths, root/'output', 4, 1, 'synthetic')
            self.assertEqual(len(plan), 12); self.assertEqual(len({p['run_id'] for p in plan}), 12)
            self.assertNotEqual([p['case']['mode'] for p in plan[:3]], [p['case']['mode'] for p in plan[3:6]])


if __name__ == '__main__':
    unittest.main()
