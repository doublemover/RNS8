import copy
import unittest
from unittest.mock import patch
from collect_lift_recipe import validate_case,validate_capture,summarize,bounds

class LiftRecipeContracts(unittest.TestCase):
    def case(self):return dict(schema='lift_recipe_case_v1',side=32,depth=3,prefix=20,source_prefix=1,repeats=3,warmups=1,model_launch_ns=5000,max_payload_bytes=1<<30,unsigned_inputs=False,mode='recipe-minimum',topology='chain')
    def capture(self):
        c=self.case();cells=32**2;p=20;nodes=3;phase=dict(producer_us=1.,execute_lift_us=2.,final_export_us=3.,total_us=6.)
        return dict(schema='rns8_lift_recipe_microcase_v1',case=c,run_id='fixture',output_boundary='complete_dag_final_three_limb_export',extra_intermediate_observations=False,private_residue_transfer_control=False,
            logical_payload_estimate_bytes=(nodes+3)*cells*p+cells*48*(nodes+1)+cells*16,host_payload_bytes_before_final_export=0,final_export_payload_bytes=cells*24,exact_reference_passed=True,promotion_eligible=False,
            build_manifest={},execution_id='process1',target='gfx1100',device='fixture',artifact_capture=dict(unchanged_during_run=True,executable_sha256='a'*64),
            ragged_retained_bytes=0,ragged_materialized_input_bytes=0,ragged_binding_peak_bytes=0,recipe_arena_bytes=25600,recipe_retained_lift_bytes=17412,auto_retained_device_bytes=0,replay_hip_allocations=0,replay_hip_frees=0,selected_compute_prefixes=[1,2,3],cost_model=None,
            setup_us=10.,first_call=phase,samples=[copy.deepcopy(phase) for _ in range(3)],sustained=[dict(calls=16,total_us=250000.,per_call_us=15625.) for _ in range(3)])
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
    def test_no_unmatched_or_single_process_speedup(self,_manifest):
        e=dict(case=self.case(),run_id='fixture',independent_run=0,output='a.json');c=self.capture()
        with self.assertRaises(ValueError):summarize([(e,c)])
        d=copy.deepcopy(c);d['execution_id']='process2';r=summarize([(e,c),(dict(e,independent_run=1),d)])
        self.assertEqual(r[0]['matched_comparisons'],{});self.assertIsNone(r[0]['baseline'])

if __name__=='__main__':unittest.main()
