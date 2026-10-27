import copy
import unittest
from collect_bilinear_calibration import validate_case,validate_capture,resources,summarize
from test_capture_identity_support import synthetic_identity


class BilinearCollection(unittest.TestCase):
    def case(self):
        return dict(schema='bilinear_case_v1',m=17,n=19,k=31,prefix=3,unsigned_inputs=False,mode='october55-retained',subproduct_backend='direct',residency='resident',instrumented=False,repeats=3,warmups=1,seed=4855,max_scratch_bytes=2**32,max_host_bytes=2**32,matrix_variant=0)

    def capture(self):
        case=self.case();entry=dict(case=case,run_id='synthetic-not-hardware-evidence');scratch,residue,host=resources(case)
        sample=dict(prepare_us=1,execute_us=2,export_us=3,cleanup_us=0,total_us=6)
        c=dict(schema='rns8_bilinear_microcase_v1',case=case,run_id=entry['run_id'],dataset='period15_native_small_v1',selected_prefix=3,algorithm='october55-retained',subproduct_backend='direct',matrix_family_override='not_applicable',subproduct_launch_schedule='serial',
            required_bilinear_scratch_bytes=scratch,resident_residue_payload_bytes=residue,ordinary_workspace_bytes=0,known_peak_device_payload_bytes=residue+scratch+17*19*24,host_payload_envelope_bytes=host,
            memory_scope='logical_live_payload_excludes_driver_allocator_and_opaque_runtime_overhead',transform_binary_nodes=55,input_transform_evaluations_per_cell=1,
            output_boundary='complete_public_product_and_padded_three_limb_export',host_instrumentation_state='hooks_disabled',diagnostic_events=[],setup_us=1,first_call=sample,samples=[copy.deepcopy(sample) for _ in range(3)],
            sustained=[dict(calls=16,total_us=250000,per_call_us=15625) for _ in range(3)],exact_reference_passed=True,all_residue_planes_audited=True,residue_audit_scope='post_timing_public_product',oracle_inside_production_interval=False,promotion_eligible=False,native_useful_products=17*19*31)
        synthetic_identity(c);return c,entry

    def test_actual_DAG_and_memory_boundary(self):
        c,e=self.capture();self.assertEqual(validate_capture(c,e),c)
        for key,value in [('transform_binary_nodes',51),('required_bilinear_scratch_bytes',0),('known_peak_device_payload_bytes',0),('all_residue_planes_audited',False),('promotion_eligible',True),('diagnostic_events',[dict(label='hidden',microseconds=1)])]:
            bad=copy.deepcopy(c);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_capture(bad,e)

    def test_recompute_budget_and_matrix_cap(self):
        c=self.case();retained=resources(c)[0];c['mode']='october55-recompute';small=resources(c)[0]
        self.assertLess(small,retained);c['max_scratch_bytes']=small;self.assertEqual(validate_case(c),c)
        c['max_scratch_bytes']-=1
        with self.assertRaises(ValueError):validate_case(c)
        c=self.case();c.update(mode='classical',subproduct_backend='matrix',k=65537,prefix=9)
        with self.assertRaises(ValueError):validate_case(c)

    def test_rank48_uses_four_block_resources_and_explicit_mixed_ring_algorithm(self):
        c=self.case();c.update(mode='rank48-dyadic',m=17,n=19,k=33)
        scratch,residue,host=resources(c)
        self.assertEqual(scratch,3*(5*9+9*5+5*5+17*19))
        self.assertEqual(validate_case(c),c)
        c['max_scratch_bytes']=scratch-1
        with self.assertRaises(ValueError):validate_case(c)

    def test_diagnostics_never_enter_latency_medians(self):
        rows=[]
        for mode in ('classical','october55-retained'):
            for run in range(2):
                for instrumented in (False,True):
                    rows.append(dict(mode=mode,instrumented=instrumented,execution_id=f'{mode}-{run}-{instrumented}',independent_run=run,path='synthetic',median_us=0.001 if instrumented else 10 if mode=='classical' else 20,peak_bytes=100))
        report=summarize(rows);self.assertEqual(report['comparisons'][0]['complete_route_speedup'],0.5);self.assertFalse(report['promotion_eligible'])
        rows.append(rows[0])
        with self.assertRaises(ValueError):summarize(rows)


if __name__=='__main__':unittest.main()
