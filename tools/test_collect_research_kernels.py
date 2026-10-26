import copy
from pathlib import Path
import unittest
from collect_research_kernels import validate_case, validate_capture
from collect_native_calibration import strict_json
from test_capture_identity_support import synthetic_identity


class ResearchCapture(unittest.TestCase):
    def fixture(self, fp8):
        case = dict(schema='finite_fp8_public_case_v1' if fp8 else 'native_tdm_case_v1', m=17, n=19, k=33,
                    mode='fp8' if fp8 else 'multicast4', repeats=3, warmups=1, seed=52645)
        case.update(dict(modulus=251, residency='repeated-ab') if fp8 else dict(prefix=9, group=4, signed_input=True, compute='wmma', cluster_axis='moduli'))
        c = dict(schema='rns8_public_fp8_microcase_v1' if fp8 else 'rns8_native_tdm_microcase_v1', case=case,
                 run_id='synthetic-not-evidence', exact_reference_passed=True, oracle_inside_production_interval=False,
                 target_qualified=False, performance_qualified=False, promotion_eligible=False, target='gfx942' if fp8 else 'gfx1250',
                 device='synthetic', runtime_version=1, setup_us=10,
                 sustained=[dict(calls=16, total_us=250000, per_call_us=15625, minimum_duration_reached=True) for _ in range(3)])
        if fp8:
            sample = dict(prepare_us=1, execute_us=2, export_us=3, cleanup_us=0, total_us=6)
            c.update(format='amd_e4m3_fnuz', conditional_instruction_model=True, retained_recipe_device_bytes=4, method_actual=0, dataset='affine_finite_bytes_v1',
                     host_instrumentation_state='hooks_disabled', native_useful_products=17*19*33, output_boundary='complete_public_finite_product_and_padded_export',
                     backend_actual='hip-direct', a_checksum='a'*64, b_checksum='b'*64, output_checksum='c'*64, certificate_sha256='a'*64, first_call=sample, samples=[copy.deepcopy(sample) for _ in range(3)])
        else:
            c.update(mode_actual='multicast4', cluster_actual=4, backend_actual='hip-direct', source_representation='native64_signed', signed_input=True,
                     compute_primitive='wmma_i32_16x16x64_iu8_wave32', cluster_axis_actual='moduli', stage_count=2, lds_bytes=32768, status_bytes=4, capability_flags=7,
                     dataset='fullwidth_native_xor_affine_v1', semantic_contract='residues_only_no_native_range_claim',
                     output_boundary='complete_resident_native64_to_centered_rns_checked_call', production_timing_hooks_enabled=False,
                     modeled_no_intergroup_reuse_read_bytes=57024, modeled_ideal_multicast_source_read_bytes=19008, modeled_ideal_source_read_amplification=2.0,
                     actual_hbm_read_bytes=None, tdm_stall_cycles=None, barrier_stall_cycles=None, achieved_occupancy=None, compiler_private_scratch_bytes=None,
                     grid_blocks=16, logical_payload_envelope_bytes=1000, payload_budget_bytes=2000,
                     a_checksum='a'*64, b_checksum='b'*64, output_checksum='c'*64, first_call_us=5, samples=[5, 5, 5])
        synthetic_identity(c)
        return c, dict(case=case, run_id=c['run_id'])

    def test_complete_and_unqualified(self):
        for fp8 in (True, False):
            c, entry = self.fixture(fp8)
            self.assertEqual(validate_capture(c, entry), c)
            for field in ('target_qualified', 'performance_qualified', 'promotion_eligible', 'oracle_inside_production_interval'):
                bad = copy.deepcopy(c); bad[field] = True
                with self.subTest(fp8=fp8, field=field), self.assertRaises(ValueError):
                    validate_capture(bad, entry)

    def test_target_opcode_and_real_work_required(self):
        for fp8 in (True, False):
            c, entry = self.fixture(fp8)
            mutations = [('target', 'gfx1100'), ('samples', []), ('setup_us', 0)]
            mutations += [('format', 'ocp_e4m3'), ('retained_recipe_device_bytes', 0)] if fp8 else [('capability_flags', 1), ('cluster_actual', 1), ('grid_blocks', 4), ('compute_primitive', 'mfma')]
            for field, value in mutations:
                bad = copy.deepcopy(c); bad[field] = value
                with self.subTest(fp8=fp8, field=field), self.assertRaises(ValueError):
                    validate_capture(bad, entry)
            bad = copy.deepcopy(c); bad['sustained'][0]['total_us'] = 1
            with self.assertRaises(ValueError):
                validate_capture(bad, entry)

    def test_all_committed_cases_and_closed_domain(self):
        root = Path(__file__).resolve().parents[1]
        for family in ('fp8', 'tdm'):
            for path in (root/'benchmarks'/family).glob('*.json'):
                case = validate_case(strict_json(path.read_text()))
                bad = dict(case, k=True)
                with self.assertRaises(ValueError):
                    validate_case(bad)
                bad = dict(case, undeclared=1)
                with self.assertRaises(ValueError):
                    validate_case(bad)


if __name__ == '__main__':
    unittest.main()
