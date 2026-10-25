"""Synthetic contract fixtures; no GPU calibration or speedup evidence."""
import copy
import unittest
from unittest.mock import patch
from collect_product_stages import validate_case,validate_capture,stage_keys,minimum_prefixes,price_bundle,held_out_report
from product_stage_contract import price_cells,price_identity


class ProductStageContracts(unittest.TestCase):
    def case(self):
        return dict(schema='product_stage_case_v1',side=2,depth=3,prefix=3,topology='chain',unsigned_inputs=False,
            prefixes=[1,2,3],capacities=[1,2,3],warmups=2,repeats=3,max_payload_bytes=1<<20)

    def observation(self,index):
        case=self.case();cells=case['side']**2;p=case['prefix']
        identity=dict(graph_sha256='a'*64,build_sha256='b'*64,target='gfx1100',node_count=3,minimum_prefixes=minimum_prefixes(case))
        stages=[dict(zip(('node','kind','capacity','old_prefix','prefix'),key)) | dict(elapsed_ns=[100.+index,102.+index,101.+index],
            sample_h2d_bytes=0,sample_d2h_bytes=0,sample_d2d_bytes=0,exact_reference_passed=True) for key in sorted(stage_keys(case))]
        capture=dict(schema='rns8_product_stage_capture_v1',case=case,run_id='train'+str(index),execution_id='train-process'+str(index),
            cost_schema_version=2,identity=identity,build_manifest={},device='synthetic',runtime_version=7,
            artifact_capture=dict(unchanged_during_run=True,executable_sha256='c'*64),dataset='all_one_inputs_exact_integer_dag_v1',
            cache_state='hot_repeated_stage',timing_source='host_monotonic_completed_backend_call',host_instrumentation_state='hooks_disabled',
            exact_reference_passed=True,promotion_eligible=False,setup_us=100.,calibration_total_us=200.,nominal_clock_tick_ns=1.,
            logical_device_payload_bytes=5*cells*p+256*case['capacities'][-1]+4+cells*(p-1),stages=stages)
        return dict(case=case,run_id=capture['run_id']),capture

    def test_domains_range_and_capacity_axes(self):
        self.assertEqual(validate_case(self.case()),self.case())
        keys=stage_keys(self.case())
        self.assertIn((0,'packed_gemm',3,0,1),keys);self.assertIn((0,'pack',3,1,2),keys)
        self.assertNotIn((0,'pack',1,1,2),keys);self.assertIn((2,'publish',0,0,3),keys)
        for key,value in [('prefixes',[2,1]),('capacities',[1,1]),('side',True),('prefix',1),('max_payload_bytes',64)]:
            bad=self.case();bad[key]=value
            with self.assertRaises(ValueError):validate_case(bad)

    @patch('collect_product_stages.manifest_provenance_complete',return_value=True)
    def test_complete_stage_receipts_and_frozen_process_prices(self,_manifest):
        observations=[self.observation(i) for i in range(5)]
        entry,c=observations[0];self.assertEqual(validate_capture(c,entry),c)
        for mutate in [lambda x:x['stages'].pop(),lambda x:x['stages'].append(copy.deepcopy(x['stages'][0])),
                lambda x:x['stages'][0].update(capacity=True),lambda x:x['stages'][0].update(elapsed_ns=[0,1,2]),
                lambda x:x['identity'].update(minimum_prefixes=[2,2,2]),lambda x:x.update(logical_device_payload_bytes=0)]:
            bad=copy.deepcopy(c);mutate(bad)
            with self.assertRaises(ValueError):validate_capture(bad,entry)
        bundle,report=price_bundle(observations);self.assertFalse(bundle['qualified']);self.assertFalse(report['isa_qualified'])
        self.assertEqual(set(price_cells(bundle)),stage_keys(self.case()))
        self.assertEqual(len(bundle['training_execution_ids']),5);self.assertEqual(len(price_identity(bundle)),64)
        with self.assertRaises(ValueError):price_bundle(observations[:4])
        bad=copy.deepcopy(observations);bad[-1][1]['execution_id']=bad[0][1]['execution_id']
        with self.assertRaises(ValueError):price_bundle(bad)
        bad=copy.deepcopy(observations);bad[-1][1]['artifact_capture']['executable_sha256']='d'*64
        with self.assertRaises(ValueError):price_bundle(bad)
        bad=copy.deepcopy(observations);bad[-1][1]['case']['warmups']=3
        with self.assertRaises(ValueError):price_bundle(bad)

    @patch('collect_product_stages.manifest_provenance_complete',return_value=True)
    def test_price_contract_rejects_duplicates_bad_strides_and_fake_qualification(self,_manifest):
        bundle,_=price_bundle([self.observation(i) for i in range(5)])
        for mutate in [lambda x:x.update(cost_schema_version=1),lambda x:x.update(qualified=True),
                lambda x:x['cells'].append(copy.deepcopy(x['cells'][0])),lambda x:x['cells'][0].update(cost_ns=True),
                lambda x:x['cells'][0].update(cost_ns=2**64-1),lambda x:x['cells'][0].update(node=3),
                lambda x:x['cells'][0].update(interval_ns=[200.,100.]),lambda x:x['training_execution_ids'].pop(),
                lambda x:x['source_artifacts'][0].update(capture_sha256='invalid')]:
            bad=copy.deepcopy(bundle);mutate(bad)
            with self.assertRaises(ValueError):price_cells(bad)

    def held_out_fixture(self):
        from test_collect_lift_recipe import LiftRecipeContracts
        capture=LiftRecipeContracts().capture();case=capture['case'];case['mode']='recipe-measured'
        identity=dict(graph_sha256='a'*64,build_sha256='b'*64,target='gfx1100',node_count=3,minimum_prefixes=[1,2,3])
        cells=[dict(node=i,kind='uniform_gemm',capacity=0,old_prefix=0,prefix=20,cost_ns=100,interval_ns=[99.,101.]) for i in range(3)]
        cells.append(dict(node=2,kind='publish',capacity=0,old_prefix=0,prefix=20,cost_ns=50,interval_ns=[49.,51.]))
        bundle=dict(schema='rns8_product_stage_prices_v1',cost_schema_version=2,identity=identity,
            training_run_ids=['train'+str(i) for i in range(5)],training_execution_ids=['train-process'+str(i) for i in range(5)],
            source_artifacts=[dict(executable_sha256='c'*64,capture_sha256='d'*64) for _ in range(5)],
            dataset='all_one_inputs_exact_integer_dag_v1',cache_state='hot_repeated_stage',timing_source='host_monotonic_completed_backend_call',device='fixture',runtime_version=7,cells=cells,qualified=False,promotion_eligible=False)
        capture['runtime_version']=7
        capture.update(stage_price_identity=price_identity(bundle),selected_compute_prefixes=[20]*3,recipe_retained_lift_bytes=0,
            cost_model=dict(kind='measured-stage-prices-unqualified',cost_schema_version=2,price_sha256=price_identity(bundle),qualified=False),
            materialization_plan=dict(packed_b=[0,0,0],packed_values=[],evaluated=4,exhaustive=False,chain_dynamic_programming=True,cost_optimal=False,estimated_ns=300,completed_packing_calls=0))
        phase=dict(producer_us=1.,execute_lift_us=.35,final_export_us=3.,total_us=4.35)
        capture.update(first_call=phase,samples=[copy.deepcopy(phase) for _ in range(3)])
        observations=[]
        for i in range(5):
            c=copy.deepcopy(capture);c.update(run_id='held'+str(i),execution_id='held-process'+str(i))
            observations.append((dict(case=c['case'],run_id=c['run_id'],stage_price_document=bundle),c))
        return bundle,observations

    @patch('collect_lift_recipe.manifest_provenance_complete',return_value=True)
    def test_held_out_process_separation_and_discrepancy_failure(self,_manifest):
        bundle,observations=self.held_out_fixture();report=held_out_report(bundle,observations,.1)
        self.assertTrue(report['held_out_passed']);self.assertFalse(report['promotion_eligible'])
        bad=copy.deepcopy(observations);bad[0][1]['execution_id']='train-process0'
        with self.assertRaises(ValueError):held_out_report(bundle,bad,.1)
        bad=copy.deepcopy(observations);bad[0][1]['stage_price_identity']='e'*64
        with self.assertRaises(ValueError):held_out_report(bundle,bad,.1)
        bad=copy.deepcopy(observations)
        for sample in bad[0][1]['samples']:sample.update(execute_lift_us=100.,total_us=104.)
        self.assertFalse(held_out_report(bundle,bad,.1)['held_out_passed'])


if __name__=='__main__':unittest.main()
