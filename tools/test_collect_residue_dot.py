from benchmark_cases import case_root
import copy
import unittest
from unittest.mock import patch
from pathlib import Path
import json
from collect_residue_dot import validate_case,validate_capture,summarize,paired_manifest
from gpu_isa_report import scan_disassembly

class ResidueDotContracts(unittest.TestCase):
    def case(self):return dict(schema='residue_dot_case_v1',m=17,n=19,k=65,prefix=9,modulus=0,tasks=3,repeats=3,warmups=1,mode='independent',domain='bounded',dataset='centered')
    def capture(self,arithmetic='scalar',mode='independent'):
        c=self.case();c['mode']=mode;region=mode=='region';cells=(17-int(region))*(19-int(region))*3*9;phase=dict(compute_us=2.,export_us=1.,total_us=3.)
        return dict(schema='rns8_residue_dot_microcase_v3',case=c,run_id='fixture',arithmetic=arithmetic,output_boundary='public_incremental_nonzero_rectangle_with_full_padded_export' if region else 'public_complete_products_with_full_padded_export',dataset_seed=7171,residency='unchanged_packed_inputs',hooks_enabled=False,diagnostic_events=None,diagnostic_event_scope='separate_final_hook_enabled_invocation_group_spans_not_dynamic_launch_or_instruction_counts',oracle_inside_interval=False,all_residue_planes_checked=True,exact_reference_passed=True,promotion_eligible=False,
            source_logical_scalar_updates=cells*65 if arithmetic=='scalar' else 0,source_logical_dot4_updates=cells*17 if arithmetic=='dot4' else 0,source_count_scope='active_output_accumulator_updates_not_issued_instructions_inactive_lanes_or_total_gather_traffic',selected_kernel_scope='plan_identity_plus_explicit_public_scheduler_not_dynamic_profiler_evidence',actual_launches=None,isa_and_resources=None,sustained_body='compute_export_only_no_oracle_or_json',plan_contract=dict(global_bound=0 if mode=='scheduled' else 1,per_tile_bound_count=int(mode=='scheduled'),schedule_entries=int(mode=='scheduled'),min_selected_prefix=9,max_selected_prefix=9,min_required_prefix=1,max_required_prefix=1,effective_dispatch=mode),
            build_manifest=dict(configuration='residue_dot4='+('ON' if arithmetic=='dot4' else 'OFF')+'\nresidue_scalar_control='+('OFF' if arithmetic=='dot4' else 'ON')+'\nflags=-O3\n'),execution_id='process1',target='gfx1100',device='fixture',selected_kernel=mode,artifact_capture=dict(unchanged_during_run=True,executable_sha256=('a' if arithmetic=='scalar' else 'b')*64),setup_us=10.,first_call=phase,samples=[copy.deepcopy(phase) for _ in range(3)],sustained=[dict(calls=16,total_us=250000.,per_call_us=15625.) for _ in range(3)])
    def test_all_authored_cases_and_scope_rejections(self):
        for path in (case_root() / 'residue-dot').glob('*.json'):validate_case(json.loads(path.read_text()))
        for key,value in [('tasks',True),('modulus',251),('dataset','fullwidth'),('m',1)]:
            bad=self.case();bad[key]=value
            with self.assertRaises(ValueError):validate_case(bad)
    @patch('collect_residue_dot.manifest_provenance_complete',return_value=True)
    def test_legacy_off_is_not_scalar_and_other_flags_cannot_drift(self,_):
        a=self.capture();b=self.capture('dot4');self.assertEqual(paired_manifest(a['build_manifest'],'scalar'),paired_manifest(b['build_manifest'],'dot4'))
        b['build_manifest']['configuration']+='different_runtime=1\n';self.assertNotEqual(paired_manifest(a['build_manifest'],'scalar'),paired_manifest(b['build_manifest'],'dot4'))
        a['build_manifest']['configuration']=a['build_manifest']['configuration'].replace('scalar_control=ON','scalar_control=OFF')
        with self.assertRaises(ValueError):paired_manifest(a['build_manifest'],'scalar')
    @patch('collect_residue_dot.manifest_provenance_complete',return_value=True)
    def test_full_phases_and_no_inferred_instructions(self,_):
        c=self.capture();e=dict(case=c['case'],run_id='fixture',arithmetic='scalar');validate_capture(c,e)
        for key,value in [('source_logical_scalar_updates',1),('actual_launches',7),('hooks_enabled',True),('all_residue_planes_checked',False),('promotion_eligible',True)]:
            bad=copy.deepcopy(c);bad[key]=value
            with self.assertRaises(ValueError):validate_capture(bad,e)
        bad=copy.deepcopy(c);bad['samples'][0]['total_us']=1
        with self.assertRaises(ValueError):validate_capture(bad,e)
        bad=copy.deepcopy(c);bad.update(hooks_enabled=True,diagnostic_events=[dict(label='group_span',microseconds=1.)]);validate_capture(bad,e,True)
        with self.assertRaises(ValueError):validate_capture(bad,e)
    @patch('collect_residue_dot.manifest_provenance_complete',return_value=True)
    def test_comparison_axes_and_region_boundary(self,_):
        observations=[]
        for arithmetic in ('scalar','dot4'):
            for mode in ('independent','grouped','region'):
                for run in (0,1):
                    c=self.capture(arithmetic,mode);c['execution_id']=f'{arithmetic}-{mode}-{run}';e=dict(case=c['case'],run_id='fixture',arithmetic=arithmetic,independent_run=run,output=c['execution_id']+'.json');observations.append((e,c))
        result=summarize(observations);self.assertEqual(len(result),1);self.assertEqual(set(result[0]['same_scheduler_arithmetic_cost_ratios']),{'independent','grouped','region'});self.assertEqual(set(result[0]['same_arithmetic_scheduler_cost_ratios']),{'scalar:grouped','dot4:grouped'})
        with self.assertRaises(ValueError):summarize(observations+[observations[0]])
        with self.assertRaises(ValueError):summarize(observations[:1])
    @patch('collect_residue_dot.manifest_provenance_complete',return_value=True)
    def test_scheduled_bound_ownership_and_allocation_free_timing_contract(self,_):
        c=self.capture(mode='scheduled');e=dict(case=c['case'],run_id='fixture',arithmetic='scalar');validate_capture(c,e)
        for key,value in [('global_bound',1),('per_tile_bound_count',0),('schedule_entries',0),('min_selected_prefix',1),('max_selected_prefix',20),('schedule_entries',True)]:
            bad=copy.deepcopy(c);bad['plan_contract'][key]=value
            with self.assertRaises(ValueError):validate_capture(bad,e)
        for key,value in [('schema','rns8_residue_dot_microcase_v1'),('schema','rns8_residue_dot_microcase_v2'),('sustained_body','json_sample_allocation'),('plan_contract',None)]:
            bad=copy.deepcopy(c);bad[key]=value
            with self.assertRaises(ValueError):validate_capture(bad,e)
    @patch('collect_residue_dot.manifest_provenance_complete',return_value=True)
    def test_folded_uniform_request_retains_its_actual_owner(self,_):
        observations=[]
        for arithmetic in ('scalar','dot4'):
            for run in (0,1):
                c=self.capture(arithmetic,'scheduled');c['plan_contract'].update(min_required_prefix=9,max_required_prefix=9,schedule_entries=0,effective_dispatch='folded_uniform');c['execution_id']=f'folded-{arithmetic}-{run}'
                e=dict(case=c['case'],run_id='fixture',arithmetic=arithmetic,independent_run=run,output=c['execution_id']+'.json');validate_capture(c,e);observations.append((e,c))
                bad=copy.deepcopy(c);bad['plan_contract']['effective_dispatch']='scheduled'
                with self.assertRaises(ValueError):validate_capture(bad,e)
                bad=copy.deepcopy(c);bad['plan_contract']['schedule_entries']=1
                with self.assertRaises(ValueError):validate_capture(bad,e)
        result=summarize(observations)
        self.assertEqual(set(result[0]['same_scheduler_arithmetic_cost_ratios']),{'scheduled:folded_uniform'})
        self.assertFalse(result[0]['same_arithmetic_scheduler_cost_ratios'])
    def test_static_dot_gather_counts_not_matrix_or_dynamic_work(self):
        counts=scan_disassembly('v_dot4_i32_iu8 v0, v1, v2, v0 neg_lo:[1,1,0]\nv_bfe_i32 v3, v4, 8, 8\nv_mul_lo_u32 v5, v6, v7\nds_read_b32 v1, v2\nds_write_b8 v1, v2\n')
        self.assertEqual(counts['packed_dot4'],1);self.assertEqual(counts['byte_extract_permute'],1);self.assertEqual(counts['scalar_integer_multiply_add'],1);self.assertEqual(counts['lds_read'],1);self.assertEqual(counts['lds_write'],1);self.assertEqual(counts['matrix_instruction_count'],0)

if __name__=='__main__':unittest.main()
