import copy
import unittest
from unittest.mock import patch
from collect_lift_recipe import validate_case,validate_capture,summarize,bounds,native_payload,payload_bytes

def replay_receipt(c,publish=0):
    from product_replay import policy,estimate
    p=policy(c['case'],c.get('stage_price_training_charge_ns',0),publish)
    plan=c['materialization_plan'];backend=plan['estimated_ns']-p['orchestration_ns']-p['publication_ns']
    plan.update(replay_policy=p,replay_costs=estimate(p,backend,backend),
        completed_replays=1+c['case']['warmups']+c['case']['repeats']+sum(s['calls'] for s in c['sustained']),
        reused_external_packing_calls=0,retained_external_packed_bytes=0)

class LiftRecipeContracts(unittest.TestCase):
    def case(self):return dict(schema='lift_recipe_case_v1',side=32,depth=3,prefix=20,source_prefix=1,repeats=3,warmups=1,model_launch_ns=5000,max_payload_bytes=1<<30,unsigned_inputs=False,mode='recipe-minimum',topology='chain')
    def capture(self,mode=None,**changes):
        c=self.case();c.update(changes)
        if mode is not None:c['mode']=mode
        cells=c['side']**2;p=c['prefix'];nodes=c['depth'];phase=dict(producer_us=1.,execute_lift_us=2.,final_export_us=3.,total_us=6.)
        h2d,d2h,host=native_payload(c);native=c['mode']=='native-roundtrip'
        return dict(schema='rns8_lift_recipe_microcase_v3',case=c,run_id='fixture',output_boundary='complete_dag_final_three_limb_export',extra_intermediate_observations=c['mode']=='eager-observe',private_residue_transfer_control=False,
            native_roundtrip_control=native,fixed_plan_exact_proof_policy='scalar_first' if c['mode']=='manual-scalar-first' else 'eager_axis',
            logical_payload_estimate_bytes=payload_bytes(c),host_payload_bytes_before_final_export=h2d+d2h,final_export_payload_bytes=cells*24,exact_reference_passed=True,promotion_eligible=False,
            native_roundtrip_h2d_payload_bytes_per_replay=h2d,native_roundtrip_d2h_payload_bytes_per_replay=d2h,native_roundtrip_retained_host_bytes=host,
            replay_hip_counter_samples=c['repeats'],replay_hip_h2d_bytes=h2d*p//8*c['repeats'],replay_hip_d2h_bytes=(d2h+cells*24)*c['repeats'],
            replay_hip_h2d_calls=2*nodes*c['repeats'] if native else 0,replay_hip_d2h_calls=nodes*c['repeats'] if native else c['repeats'],
            build_manifest={},execution_id='process1',target='gfx1100',device='fixture',artifact_capture=dict(unchanged_during_run=True,executable_sha256='a'*64),
            ragged_retained_bytes=0,ragged_materialized_input_bytes=0,ragged_binding_peak_bytes=0,recipe_arena_bytes=25600,recipe_retained_lift_bytes=17412,auto_retained_device_bytes=0,replay_hip_allocations=0,replay_hip_frees=0,selected_compute_prefixes=[p]*nodes if c['mode'] in ('manual-full','manual-scalar-first','native-roundtrip','eager-observe') else [1,2,3],cost_model=None,materialization_plan=None,
            stage_price_identity=None,stage_price_training_charge_ns=0,stage_price_graph_identity=None,timing_source='host_monotonic_completed_public_call',nominal_clock_tick_ns=1.,setup_us=10.,first_call=phase,samples=[copy.deepcopy(phase) for _ in range(3)],sustained=[dict(calls=16,total_us=250000.,per_call_us=15625.) for _ in range(3)])
    def test_chain_diamond_bounds_and_noop_lift(self):
        c=self.case();self.assertEqual(bounds(c),[1,1,32,1024,32768]);self.assertEqual(validate_case(c),c)
        c.update(topology='diamond',depth=4);self.assertEqual(bounds(c)[-1],32**5);self.assertEqual(validate_case(c),c)
        c=self.case();c.update(mode='lift-graph',source_prefix=20);self.assertEqual(validate_case(c),c)
        for key,value in [('side',True),('prefix',1),('source_prefix',21),('max_payload_bytes',64)]:
            bad=self.case();bad[key]=value
            with self.assertRaises(ValueError):validate_case(bad)
    @patch('collect_lift_recipe.manifest_provenance_complete',return_value=True)
    def test_real_retention_and_independent_prefix_proof(self,_manifest):
        e=dict(case=self.case(),run_id='fixture');c=self.capture();self.assertEqual(validate_capture(c,e),c)
        for key,value in [('replay_hip_allocations',6),('replay_hip_frees',6),('selected_compute_prefixes',[1,1,1]),('extra_intermediate_observations',True),('promotion_eligible',True),('cost_model',{'qualified':True})]:
            bad=copy.deepcopy(c);bad[key]=value
            with self.assertRaises(ValueError):validate_capture(bad,e)
        bad=copy.deepcopy(c);bad['samples'][0]['total_us']=1
        with self.assertRaises(ValueError):validate_capture(bad,e)
    @patch('collect_lift_recipe.manifest_provenance_complete',return_value=True)
    def test_packed_trace_replays_growth_lifetimes_costs_and_budget(self,_manifest):
        c=self.capture();c['case']['mode']='recipe-packed-minimum';c['recipe_arena_bytes']=65536
        c['cost_model']=dict(kind='explicit-unqualified-layout-work-model-v2',launch_ns=5000,
            gemm_units='one_ns_per_scalar_modular_multiply',packed_gemm_units='one_ns_per_four_term_group',
            pack_units='one_ns_per_logical_byte_read_or_write',cost_schema_version=3,packed_cost_capacity='physical_shared_b_prefix',qualified=False)
        view=dict(value_id=1,capacity_prefix=3,first_reader=0,last_reader=2,bytes=3072,arena_offset=0)
        c['materialization_plan']=dict(packed_b=[1,1,1],packed_values=[view],evaluated=4096,exhaustive=False,
            chain_dynamic_programming=True,cost_optimal=False,estimated_ns=199928,completed_packing_calls=3)
        replay_receipt(c)
        e=dict(case=c['case'],run_id='fixture');self.assertEqual(validate_capture(c,e),c)
        for key,value in [('completed_packing_calls',1),('estimated_ns',199927),('packed_b',[1,0,1]),('evaluated',True)]:
            bad=copy.deepcopy(c);bad['materialization_plan'][key]=value
            with self.assertRaises(ValueError):validate_capture(bad,e)
        for key,value in [('last_reader',1),('bytes',3071),('arena_offset',1),('capacity_prefix',2)]:
            bad=copy.deepcopy(c);bad['materialization_plan']['packed_values'][0][key]=value
            with self.assertRaises(ValueError):validate_capture(bad,e)
        bad=copy.deepcopy(c);bad['materialization_plan']['packed_values'].append(copy.deepcopy(view))
        with self.assertRaises(ValueError):validate_capture(bad,e)
        full=copy.deepcopy(c);full['case']['mode']='recipe-packed-full';full['selected_compute_prefixes']=[20]*3
        full['materialization_plan'].update(estimated_ns=552480,completed_packing_calls=1)
        full['materialization_plan']['packed_values'][0].update(capacity_prefix=20,bytes=20480)
        replay_receipt(full)
        self.assertEqual(validate_capture(full,dict(case=full['case'],run_id='fixture')),full)
        bad=copy.deepcopy(c);del bad['materialization_plan']
        with self.assertRaises(ValueError):validate_capture(bad,e)

    def test_native64_intermediate_admission_keeps_wide_final(self):
        for unsigned,side,allowed_depth,rejected_depth in [(False,128,9,10),(True,256,8,9)]:
            c=self.case();c.update(mode='native-roundtrip',side=side,depth=allowed_depth,unsigned_inputs=unsigned)
            self.assertGreaterEqual(bounds(c)[-1],2**(64 if unsigned else 63))
            self.assertEqual(validate_case(c),c)
            c['depth']=rejected_depth
            with self.assertRaisesRegex(ValueError,'intermediate exceeds native64'):validate_case(c)
        c=self.case();c.update(mode='native-roundtrip',depth=1)
        self.assertEqual(native_payload(validate_case(c)),(2*32**2*8,0,2*32**2*8))
        with patch('collect_lift_recipe.graph',return_value=[(0,0),(2,1)]):
            self.assertEqual(native_payload(c),(3*32**2*8,32**2*8,3*32**2*8))

    @patch('collect_lift_recipe.manifest_provenance_complete',return_value=True)
    def test_native_roundtrip_logical_and_actual_transfer_receipts(self,_manifest):
        c=self.capture('native-roundtrip');e=dict(case=c['case'],run_id='fixture')
        replay_receipt(c)
        self.assertEqual(validate_capture(c,e),c)
        self.assertNotEqual(c['replay_hip_h2d_bytes'],c['native_roundtrip_h2d_payload_bytes_per_replay'])
        for key,value in [('native_roundtrip_control',False),('extra_intermediate_observations',True),
                ('native_roundtrip_h2d_payload_bytes_per_replay',0),('native_roundtrip_d2h_payload_bytes_per_replay',0),
                ('native_roundtrip_retained_host_bytes',0),('host_payload_bytes_before_final_export',0),
                ('replay_hip_counter_samples',1),('replay_hip_h2d_calls',0),('replay_hip_d2h_calls',0),
                ('replay_hip_h2d_bytes',False),('replay_hip_d2h_bytes',-1),('selected_compute_prefixes',[1,2,3]),
                ('schema','rns8_lift_recipe_microcase_v2')]:
            bad=copy.deepcopy(c);bad[key]=value
            with self.assertRaises(ValueError):validate_capture(bad,e)
        for key in ('native_roundtrip_retained_host_bytes','replay_hip_h2d_bytes','replay_hip_counter_samples'):
            bad=copy.deepcopy(c);del bad[key]
            with self.assertRaises(ValueError):validate_capture(bad,e)
        one=self.capture('native-roundtrip',depth=1)
        self.assertEqual(validate_capture(one,dict(case=one['case'],run_id='fixture')),one)
        diamond=self.capture('native-roundtrip',depth=4,topology='diamond',unsigned_inputs=True)
        self.assertEqual(validate_capture(diamond,dict(case=diamond['case'],run_id='fixture')),diamond)

    @patch('collect_lift_recipe.manifest_provenance_complete',return_value=True)
    def test_scalar_control_and_native_internal_exports_share_final_output_cohort(self,_manifest):
        observations=[]
        for mode in ('manual-full','manual-scalar-first','native-roundtrip','eager-observe'):
            for run in range(2):
                c=self.capture(mode);c['execution_id']=mode+str(run)
                e=dict(case=c['case'],run_id='fixture',independent_run=run,output=mode+str(run)+'.json')
                observations.append((e,c))
        cohorts=summarize(observations);self.assertEqual(len(cohorts),2)
        same=next(c for c in cohorts if c['baseline']=='manual-scalar-first')
        self.assertEqual(set(same['median_process_metrics']),{'manual-full','manual-scalar-first','native-roundtrip'})
        self.assertIn('native-roundtrip',same['matched_comparisons'])
        eager=next(c for c in cohorts if c['identity']['extra_intermediate_observations'])
        self.assertEqual(set(eager['median_process_metrics']),{'eager-observe'})
        c=self.capture('manual-scalar-first');c['fixed_plan_exact_proof_policy']='eager_axis'
        with self.assertRaises(ValueError):validate_capture(c,dict(case=c['case'],run_id='fixture'))

    @patch('collect_lift_recipe.manifest_provenance_complete',return_value=True)
    def test_no_unmatched_or_single_process_speedup(self,_manifest):
        e=dict(case=self.case(),run_id='fixture',independent_run=0,output='a.json');c=self.capture()
        with self.assertRaises(ValueError):summarize([(e,c)])
        d=copy.deepcopy(c);d['execution_id']='process2';r=summarize([(e,c),(dict(e,independent_run=1),d)])
        self.assertEqual(r[0]['matched_comparisons'],{});self.assertIsNone(r[0]['baseline'])

class ReplayPolicyContracts(unittest.TestCase):
    def test_dataset_signedness_identity_and_propagation(self):
        from product_dataset import inputs,identity
        base=LiftRecipeContracts().case()
        self.assertEqual(inputs(base),[1,1])
        changed=dict(base,input_values=[-2,3])
        self.assertEqual(bounds(changed),[2,3,192,18432,1769472])
        self.assertNotEqual(identity(base),identity(changed))
        self.assertEqual(validate_case(changed),changed)
        for values in ([True,1],[2**63,1],[1],[-1,1]):
            bad=dict(base,input_values=values,unsigned_inputs=True)
            if values==[2**63,1]:bad['unsigned_inputs']=False
            with self.assertRaises(ValueError):validate_case(bad)
    def test_cold_steady_training_strict_payback_and_unknown(self):
        from product_replay import policy,estimate,LIMIT
        p=policy(dict(replay_policy=dict(expected_replays=100,retain_external_b=True,
            preparation_ns=200,calibration_ns=1,orchestration_ns=7,publication_ns=11,
            baseline_replay_ns=300)),training_ns=500)
        r=estimate(p,1030,30)
        self.assertEqual((r['first_ns'],r['steady_ns'],r['once_ns'],r['lifetime_ns'],r['amortized_ns']),(1048,48,700,6500,65))
        self.assertEqual(r['break_even_replays'],7);self.assertEqual(r['amortization_status'],1)
        p['expected_replays']=6;self.assertEqual(estimate(p,1030,30)['amortization_status'],2)
        p['baseline_replay_ns']=LIMIT;self.assertEqual(estimate(p,1030,30)['amortization_status'],0)
        for extra in (dict(expected_replays=0),dict(expected_replays=True),dict(retain_external_b=1),dict(unknown=1)):
            with self.assertRaises(ValueError):policy(dict(replay_policy=extra))
        p['calibration_ns']=LIMIT
        with self.assertRaises(ValueError):estimate(p,1030,30)

if __name__=='__main__':unittest.main()
