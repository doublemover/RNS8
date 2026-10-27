from benchmark_cases import case_root
import copy
import unittest
from pathlib import Path
from unittest.mock import patch
from collect_native_interop import validate_case, validate_capture, summarize, quantile

class NativeInteropContracts(unittest.TestCase):
    def case(self):
        return dict(schema='native_interop_case_v1',m=3,n=5,k=7,prefix=3,unsigned_inputs=False,mode='device-graph',dataset='small',repeats=3,warmups=1,seed=67,max_payload_bytes=1<<28)
    def capture(self):
        c=self.case();m,n,k,p=(c[x] for x in ('m','n','k','prefix'));ac,bc,cc=m*k,k*n,m*(n+3)*3
        phase=dict(producer_enqueue_us=1.,input_compute_us=2.,export_observation_us=3.,consumer_us=4.,total_us=10.)
        empty=dict(live_bytes=0,peak_bytes=0,live_allocations=0,peak_allocations=0,allocation_calls=0,free_calls=0,allocated_bytes=0)
        live=dict(live_bytes=4096,peak_bytes=4096,live_allocations=1,peak_allocations=1,allocation_calls=1,free_calls=0,allocated_bytes=4096)
        retired=dict(live,live_bytes=0,live_allocations=0,free_calls=1)
        return dict(schema='rns8_native_interop_microcase_v4',case=c,run_id='fixture',exact_proof_policy='scalar_first',output_boundary='external_producer_to_external_consumer_padded_three_limbs',dataset_replayed_unchanged=True,external_generations_advanced=True,
            compute_provider='public_prepared_direct_HIP_grouped_prefix',compute_execution_mode='owned_graph',
            compute_provider_scope='source_bound_enqueue_prepared_to_hip_direct_enqueue_rns_no_publish_to_launch_rns_grouped_prefix_gemm',dynamic_compute_symbol=None,
            resident_control_excludes_input_binding=False,host_payload_bytes_per_call=0,compact_status_bytes_per_call=12,consumer_validation_bytes=8,validation_payload_bytes=cc*8,
            preflight_device_payload_estimate_bytes=(ac+bc+m*n)*p+(ac+bc)*8+cc*8+8+(ac+bc)*p+32+m*n*24+4,preflight_host_payload_estimate_bytes=(ac+bc)*8+2*cc*8+12,
            memory_scope='conservative_all_route_logical_preflight_excludes_Boost_library_heap_driver_opaque_events_graphs_workspace_separate',launch_count=None,launch_count_scope='not_inferred_from_api_calls',
            participating_external_owners=3,owner_acquire_callbacks=3,owner_release_callbacks=3,
            timing_source='steady_clock_complete_host_observed_transaction',phase_scope='enqueue_and_observation_not_disjoint_device_kernel_durations',exact_reference_passed=True,owner_callbacks_exactly_once=True,promotion_eligible=False,
            plan_workspace_bytes=4096,plan_default_compute_kernel='fixture-plan',build_manifest={},target='gfx1100',device='fixture',execution_id='process1',artifact_capture=dict(unchanged_during_run=True,executable_sha256='a'*64),setup_us=20.,
            first_call=phase,samples=[copy.deepcopy(phase) for _ in range(3)],sustained=[dict(calls=16,total_us=250000.,per_call_us=15625.) for _ in range(3)],
            observed_allocation_scope='RNS8_owned_requested_device_and_pinned_payload_excludes_app_allocations_host_heap_allocator_and_driver',
            all_owned_payload_retired=True,external_application_device_requested_bytes=(ac+bc)*8+cc*8+8,
            ordinary_host_native_vector_capacity_bytes=(ac+bc+2*cc)*8,
            ordinary_host_capacity_scope='four_native_vectors_only_excludes_Boost_objects_and_dynamic_limbs_library_heap_and_RSS',
            owned_before=dict(epoch=1,device=copy.deepcopy(empty),pinned_host=copy.deepcopy(empty)),
            owned_after_setup=dict(epoch=1,device=copy.deepcopy(live),pinned_host=copy.deepcopy(empty)),
            owned_after_calls=dict(epoch=1,device=copy.deepcopy(live),pinned_host=copy.deepcopy(empty)),
            owned_after_cleanup=dict(epoch=1,device=copy.deepcopy(retired),pinned_host=copy.deepcopy(empty)))
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

    @patch('collect_native_interop.manifest_provenance_complete',return_value=True)
    def test_observed_peaks_epochs_and_retirement_are_checked(self,_manifest):
        entry=dict(case=self.case(),run_id='fixture');c=self.capture()
        for snapshot,key,value in [('owned_after_calls','peak_bytes',1),('owned_after_cleanup','live_bytes',1),
                                   ('owned_after_cleanup','free_calls',0),('owned_after_calls','allocation_calls',True)]:
            bad=copy.deepcopy(c);bad[snapshot]['device'][key]=value
            with self.subTest(snapshot=snapshot,key=key),self.assertRaises(ValueError):validate_capture(bad,entry)
        bad=copy.deepcopy(c);bad['owned_after_calls']['epoch']=2
        with self.assertRaises(ValueError):validate_capture(bad,entry)
        bad=copy.deepcopy(c);bad.pop('owned_after_calls')
        with self.assertRaises(ValueError):validate_capture(bad,entry)
        bad=copy.deepcopy(c);bad['ordinary_host_native_vector_capacity_bytes']=8
        with self.assertRaises(ValueError):validate_capture(bad,entry)

    @patch('collect_native_interop.manifest_provenance_complete',return_value=True)
    def test_only_participating_owners_are_charged(self,_manifest):
        for mode,count in [('device-stream',3),('device-graph',3),('resident-control',1),('host-roundtrip',0),('host-scalar-first',0)]:
            c=self.capture();c['case']['mode']=mode;c['resident_control_excludes_input_binding']=mode=='resident-control'
            c['exact_proof_policy']='default_eager' if mode=='host-roundtrip' else 'scalar_first'
            c['compute_execution_mode']='owned_graph' if mode=='device-graph' else 'borrowed_stream'
            c['compact_status_bytes_per_call']=12 if mode.startswith('device-') else 4
            m,n,k=(c['case'][x] for x in ('m','n','k'))
            c['host_payload_bytes_per_call']=2*(m*k+k*n)*8+m*n*24+m*(n+3)*24 if mode.startswith('host-') else 0
            for key in ('participating_external_owners','owner_acquire_callbacks','owner_release_callbacks'):c[key]=count
            entry=dict(case=c['case'],run_id='fixture');self.assertEqual(validate_capture(c,entry),c)
            bad=copy.deepcopy(c);bad['owner_release_callbacks']+=1
            with self.assertRaises(ValueError):validate_capture(bad,entry)

    def test_checked_in_cases_are_complete_matched_routes(self):
        import json
        groups={}
        for path in (case_root() / 'native-interop').glob('*.json'):
            c=validate_case(json.loads(path.read_text()));mode=c.pop('mode')
            groups.setdefault(json.dumps(c,sort_keys=True),set()).add(mode)
        self.assertTrue(groups)
        for modes in groups.values():self.assertEqual(modes,{'device-stream','device-graph','host-roundtrip','host-scalar-first','resident-control'})

    @patch('collect_native_interop.manifest_provenance_complete',return_value=True)
    def test_sustained_means_weight_durations_by_calls(self,_manifest):
        entry=dict(case=self.case(),run_id='fixture',independent_run=0,output='a.json');c=self.capture()
        c['sustained']=[dict(calls=count,total_us=duration,per_call_us=duration/count) for count,duration in [(16,250000.),(32,250000.),(64,500000.)]]
        second=copy.deepcopy(c);second['execution_id']='process2'
        result=summarize([(entry,c),(dict(entry,independent_run=1),second)])
        self.assertAlmostEqual(result[0]['median_process_metrics']['device-graph']['sustained_us'],1000000/112)

    @patch('collect_native_interop.manifest_provenance_complete',return_value=True)
    def test_stronger_host_control_is_not_hidden_by_eager_proofs(self,_manifest):
        observations=[]
        for mode,factor in [('host-roundtrip',100),('host-scalar-first',2),('device-graph',1)]:
            for run in range(2):
                c=self.capture();c['case']['mode']=mode;c['execution_id']=f'{mode}-{run}'
                c['exact_proof_policy']='default_eager' if mode=='host-roundtrip' else 'scalar_first'
                c['compute_execution_mode']='owned_graph' if mode=='device-graph' else 'borrowed_stream'
                if mode.startswith('host-'):
                    m,n,k=(c['case'][x] for x in ('m','n','k'))
                    c.update(compact_status_bytes_per_call=4,host_payload_bytes_per_call=2*(m*k+k*n)*8+m*n*24+m*(n+3)*24,
                        participating_external_owners=0,owner_acquire_callbacks=0,owner_release_callbacks=0)
                for sample in c['samples']:sample.update({key:value*factor for key,value in sample.items()})
                for burst in c['sustained']:burst['total_us']*=factor;burst['per_call_us']*=factor
                observations.append((dict(case=c['case'],run_id='fixture',independent_run=run,output=c['execution_id']),c))
        row=summarize(observations)[0]
        self.assertEqual(row['strongest_matched_host_control'],'host-scalar-first')
        self.assertEqual(row['device_to_host_roundtrip_cost_ratios']['device-graph']['sustained_us'],.01)
        self.assertEqual(row['device_to_strongest_host_cost_ratios']['device-graph']['sustained_us'],.5)

if __name__=='__main__':unittest.main()
