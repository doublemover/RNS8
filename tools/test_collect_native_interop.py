import copy
import unittest
from unittest.mock import patch
from collect_native_interop import validate_case, validate_capture, summarize, quantile

class NativeInteropContracts(unittest.TestCase):
    def case(self):
        return dict(schema='native_interop_case_v1',m=3,n=5,k=7,prefix=3,unsigned_inputs=False,mode='device-graph',dataset='small',repeats=3,warmups=1,seed=67,max_payload_bytes=1<<28)
    def capture(self):
        c=self.case();m,n,k,p=(c[x] for x in ('m','n','k','prefix'));ac,bc,cc=m*k,k*n,m*(n+3)*3
        phase=dict(producer_enqueue_us=1.,input_compute_us=2.,export_observation_us=3.,consumer_us=4.,total_us=10.)
        return dict(schema='rns8_native_interop_microcase_v1',case=c,run_id='fixture',output_boundary='external_producer_to_external_consumer_padded_three_limbs',dataset_replayed_unchanged=True,external_generations_advanced=True,
            resident_control_excludes_input_binding=False,host_payload_bytes_per_call=0,compact_status_bytes_per_call=12,consumer_validation_bytes=8,validation_payload_bytes=cc*8,
            retained_device_payload_bytes=(ac+bc+m*n)*p+(ac+bc)*8+cc*8+8+(ac+bc)*p+32+m*n*24+4,retained_host_payload_bytes=(ac+bc)*8+2*cc*8+12,
            memory_scope='logical_payload_estimate_excludes_boost_host_matrix_capacity_driver_opaque_events_graphs_workspace_separate',launch_count=None,launch_count_scope='not_inferred_from_api_calls',
            timing_source='steady_clock_complete_host_observed_transaction',phase_scope='enqueue_and_observation_not_disjoint_device_kernel_durations',exact_reference_passed=True,owner_callbacks_exactly_once=True,promotion_eligible=False,
            plan_workspace_bytes=4096,build_manifest={},target='gfx1100',device='fixture',execution_id='process1',artifact_capture=dict(unchanged_during_run=True,executable_sha256='a'*64),setup_us=20.,
            first_call=phase,samples=[copy.deepcopy(phase) for _ in range(3)],sustained=[dict(calls=16,total_us=250000.,per_call_us=15625.) for _ in range(3)])
    def test_strict_shapes_budgets_and_ranges(self):
        self.assertEqual(validate_case(self.case()),self.case())
        for key,value in [('m',True),('prefix',1),('mode','zero-copy'),('max_payload_bytes',8),('dataset','approximate')]:
            bad=self.case();bad[key]=value
            with self.assertRaises(ValueError):validate_case(bad)
        c=self.case();c.update(dataset='fullwidth',prefix=20,unsigned_inputs=True);self.assertEqual(validate_case(c),c)
        c['prefix']=16
        with self.assertRaises(ValueError):validate_case(c)
    @patch('collect_native_interop.manifest_provenance_complete',return_value=True)
    def test_payload_and_full_validation_cannot_be_omitted(self,_manifest):
        entry=dict(case=self.case(),run_id='fixture');c=self.capture();self.assertEqual(validate_capture(c,entry),c)
        for key,value in [('host_payload_bytes_per_call',8),('compact_status_bytes_per_call',0),('exact_reference_passed',False),('owner_callbacks_exactly_once',False),('promotion_eligible',True),('resident_control_excludes_input_binding',True),('external_generations_advanced',False),('launch_count',1)]:
            bad=copy.deepcopy(c);bad[key]=value
            with self.assertRaises(ValueError):validate_capture(bad,entry)
        bad=copy.deepcopy(c);bad['samples'][0]['total_us']=2
        with self.assertRaises(ValueError):validate_capture(bad,entry)
    @patch('collect_native_interop.manifest_provenance_complete',return_value=True)
    def test_process_reuse_and_single_run_are_not_evidence(self,_manifest):
        entry=dict(case=self.case(),run_id='fixture',independent_run=0,output='a.json');c=self.capture()
        with self.assertRaises(ValueError):summarize([(entry,c)])
        with self.assertRaises(ValueError):summarize([(entry,c),(dict(entry,independent_run=1),c)])
        second=copy.deepcopy(c);second['execution_id']='process2'
        result=summarize([(entry,c),(dict(entry,independent_run=1),second)])
        self.assertEqual(result[0]['device_to_host_roundtrip_cost_ratios'],{})
        self.assertFalse(result[0]['promotion_eligible'])
    def test_quantiles_use_observations_not_minimum(self):
        self.assertEqual(quantile([10,1,2],.5),2)
        self.assertAlmostEqual(quantile([10,1,2],.95),9.2)

if __name__=='__main__':unittest.main()
