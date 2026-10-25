import copy
import unittest
from unittest.mock import patch
from collect_native_integrity import validate_case, validate_capture

class NativeIntegrityContracts(unittest.TestCase):
    def case(self):
        return dict(schema='native_integrity_case_v1',m=3,n=5,k=7,prefix=3,unsigned_inputs=False,mode='compute-sentinel',residency='resident',repeats=3,warmups=1,seed=49,max_payload_bytes=1<<28,dataset='small')
    def test_shapes_and_proof_are_not_inferred_from_success(self):
        self.assertEqual(validate_case(self.case()),self.case())
        for key,value in [('m',True),('prefix',1),('dataset','fp8'),('max_payload_bytes',8),('mode','freivalds')]:
            bad=self.case();bad[key]=value
            with self.assertRaises(ValueError):validate_case(bad)
    def test_full_width_needs_complete_strict_range(self):
        c=self.case();c.update(dataset='fullwidth',prefix=20,unsigned_inputs=True)
        self.assertEqual(validate_case(c),c)
        c['prefix']=16
        with self.assertRaises(ValueError):validate_case(c)
    def test_standalone_not_silently_compute(self):
        c=self.case();c.update(mode='sentinel-only',residency='repack')
        with self.assertRaises(ValueError):validate_case(c)
    def capture(self):
        case=self.case();m,n,k,p=(case[x] for x in ('m','n','k','prefix'))
        phase=dict(prepare_us=1.,execute_us=2.,export_reference_us=3.,cleanup_us=0.,total_us=6.)
        return dict(schema='rns8_native_integrity_microcase_v1',case=case,run_id='fixture',
            host_instrumentation_state='hooks_disabled',output_boundary='public_prepared_product_and_padded_three_limb_export',standalone_control=False,
            standalone_control_scope='private_read_only_checker_per_call_flag_allocation_device_sync_and_export',known_payload_envelope_bytes=(2*p+24)*(m*k+k*n+m*n)+48*m*(n+3)+8,
            memory_scope='logical_payload_excludes_cpp_int_heap_capacity_driver_allocator_and_opaque_runtime',guarantee='fixed_redundant_residue_consistency_not_arbitrary_correctness',
            ideal_failure_probability=None,check_moduli_in_computational_range=False,cpu_reference_inputs='original_host_dataset_no_input_D2H',exact_reference_passed=True,
            full_cpu_reference_inside_interval=False,promotion_eligible=False,build_manifest={},execution_id='process1',target='gfx1100',device='fixture',
            artifact_capture=dict(unchanged_during_run=True,executable_sha256='a'*64),verification=dict(outcome=1,failure_mask=0,basis_version=1,prefix=p,check_moduli=[257,263],device_flag_bytes=4,pinned_flag_bytes=4,operation=1,epoch=2,a_generation=1,b_generation=1),
            setup_us=10.,first_call=phase,samples=[copy.deepcopy(phase) for _ in range(3)],sustained=[dict(calls=16,total_us=250000.,per_call_us=15625.) for _ in range(3)])
    @patch('collect_native_integrity.manifest_provenance_complete',return_value=True)
    def test_complete_capture_and_specific_mutations(self, _manifest):
        entry=dict(case=self.case(),run_id='fixture');capture=self.capture()
        self.assertEqual(validate_capture(capture,entry),capture)
        for key,value in [('ideal_failure_probability',0.0),('check_moduli_in_computational_range',True),('promotion_eligible',True),('standalone_control',True)]:
            bad=copy.deepcopy(capture);bad[key]=value
            with self.assertRaises(ValueError):validate_capture(bad,entry)
        for key,value in [('epoch',0),('check_moduli',[256,257]),('outcome',0),('failure_mask',4),('prefix',20)]:
            bad=copy.deepcopy(capture);bad['verification'][key]=value
            with self.assertRaises(ValueError):validate_capture(bad,entry)
        bad=copy.deepcopy(capture);bad['samples'][0]['total_us']=3
        with self.assertRaises(ValueError):validate_capture(bad,entry)
        bad=copy.deepcopy(capture);bad['sustained'][0]['calls']=15
        with self.assertRaises(ValueError):validate_capture(bad,entry)
    def test_forged_probability_or_computational_check_channel_rejected(self):
        entry=dict(case=self.case(),run_id='fixture')
        for capture in [dict(ideal_failure_probability=0),dict(check_moduli_in_computational_range=True),dict(exact_reference_passed=True,promotion_eligible=True)]:
            with self.assertRaises(ValueError):validate_capture(copy.deepcopy(capture),entry)

if __name__=='__main__':unittest.main()
