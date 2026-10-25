import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from collect_cpu_amx import validate_case,validate_capture,summarize
from collect_int4_calibration import collection_plan

class CpuAmxReceipts(unittest.TestCase):
    def case(self):
        return dict(schema='cpu_amx_case_v1',m=17,n=19,k=65,prefix=9,threads=4,modulus=0,warmups=1,repeats=3,seed=54,mode='cpu-amx-int8',residency='resident',semantics='bounded-signed')
    def fixture(self):
        case=self.case();calls=1+1+3+3*16+3;sample=dict(prepare_us=1.,gemm_us=2.,crt_export_us=3.,cleanup_us=.5,total_us=6.5)
        c=dict(schema='rns8_cpu_amx_microcase_v1',case=case,run_id='fixture',build_manifest={'configuration':'cpu_amx=ON'},execution_id='process1',
            artifact_capture=dict(unchanged_during_run=True,executable_sha256='a'*64),selected_kernel='cpu_amx_int8_16x16x64_centered_v1',
            environment=dict(cpu_model='synthetic fixture only',os='Linux fixture',process_id=1,requested_threads=4,maximum_threads=4,hardware_threads=8,
                physical_core_count=4,topology_scope='proc_cpuinfo_visible_hardware_not_affinity_filtered',parallel_runtime='openmp',affinity_policy='inherited_no_pinning',page_placement_observed=False,allowed_cpus='0-3',allowed_memory_nodes='0'),
            admission=dict(compiled=1,cpuid_tile=1,cpuid_int8=1,osxsave=1,palette_supported=1,os_supported=1,permission_enabled=1,xcr0_enabled=1,admitted=1,permission_errno=0),
            calls=calls,pack_a_calls=1,pack_b_calls=1,setup_us=1.,first_call=sample,samples=[sample]*3,
            sustained=[dict(calls=16,wall_us=250000.,operation_us=104.,per_call_us=15625.,energy=None)]*3,
            output_boundary='public_host_pack_dense_gemm_complete_padded_export',gemm_scope='public_range_dispatch_compute_reduction_publication_including_AMX_pack_when_selected',
            sustained_scope='complete_calls_cleanup_and_harness_clock_json_overhead_no_oracle',sample_scope='disjoint_operation_and_cleanup_phase_sum_excludes_between_phase_oracle',
            oracle_inside_interval=False,all_residue_planes_checked=True,exact_reference_passed=True,promotion_eligible=False,
            last_workspace=dict(prepared_threads=4,last_execution_threads=4,owned_payload_bytes=100000,packed_b_bytes=9*2*2*1024,output_staging_bytes=9*17*19,
                b_matrix_instance=2,b_generation=1,b_pack_count=1,b_reuse_count=calls-1,layout_version=1,safe_k=65536))
        return c,dict(case=case,run_id='fixture',independent_run=0,output='synthetic.json')
    def test_public_range_and_modes(self):
        self.assertEqual(validate_case(self.case()),self.case())
        for key,value in [('m',True),('prefix',1),('mode','cpu-reference-disguised-amx'),('semantics','fp8'),('threads',0)]:
            c=self.case();c[key]=value
            with self.assertRaises(ValueError):validate_case(c)
        c=self.case();c.update(semantics='wide-signed',prefix=9)
        with self.assertRaises(ValueError):validate_case(c)
        c['prefix']=20;validate_case(c)
    @patch('collect_cpu_amx.manifest_provenance_complete',return_value=True)
    def test_strict_receipt_and_negative_mutations(self,_):
        valid,entry=self.fixture();validate_capture(valid,entry)
        mutations=[lambda c:c['admission'].update(permission_enabled=0),lambda c:c['admission'].update(cpuid_int8=0),
            lambda c:c['environment'].update(page_placement_observed=True),lambda c:c['environment'].update(allowed_memory_nodes='unavailable'),
            lambda c:c['last_workspace'].update(b_pack_count=2),lambda c:c['last_workspace'].update(b_reuse_count=0),
            lambda c:c['last_workspace'].update(packed_b_bytes=1024),lambda c:c['last_workspace'].update(safe_k=131072),
            lambda c:c['last_workspace'].update(last_execution_threads=5),lambda c:c.update(calls=1),lambda c:c.update(pack_b_calls=2),
            lambda c:c.update(oracle_inside_interval=True),lambda c:c.update(promotion_eligible=True),
            lambda c:c['sustained'][0].update(energy=dict(delta_uj_mod_counter=4,range_uj=100,scope='process_exact_energy',counter_wrap_count_known=True)),
            lambda c:c.update(selected_kernel='cpu_reference_scalar_rns_gemm_v1')]
        for mutate in mutations:
            c=copy.deepcopy(valid);mutate(c)
            with self.assertRaises(ValueError):validate_capture(c,entry)
    @patch('collect_cpu_amx.manifest_provenance_complete',return_value=True)
    def test_cold_and_generation_counts(self,_):
        for residency in ('cold','resident','repack-a','repack-b','repack-both'):
            c,entry=self.fixture();c['case']['residency']=residency;entry['case']=c['case'];calls=c['calls']
            for side in ('a','b'):c['pack_'+side+'_calls']=calls if residency=='cold' else 1+calls if residency in ('repack-both','repack-'+side) else 1
            c['last_workspace']['b_pack_count']=1 if residency in ('cold','resident','repack-a') else calls
            c['last_workspace']['b_reuse_count']=calls-1 if residency in ('resident','repack-a') else 0
            validate_capture(c,entry)
    @patch('collect_cpu_amx.manifest_provenance_complete',return_value=True)
    def test_complete_matched_independent_pairs_and_negative_break_even(self,_):
        observations=[]
        for run in range(2):
            for mode in ('cpu-reference','cpu-amx-int8'):
                c,e=self.fixture();c['case']['mode']=mode;c['execution_id']=mode+str(run);e['independent_run']=run;e['case']=c['case']
                if mode=='cpu-reference':c['selected_kernel']='cpu_reference_scalar_rns_gemm_v1';c['last_workspace']=None
                observations.append((e,c))
        result=summarize(observations);self.assertEqual(len(result),1);self.assertTrue(result[0]['slower_amx_preserved'])
        with self.assertRaises(ValueError):summarize(observations[:-1])
        observations[-1][1]['execution_id']=observations[0][1]['execution_id']
        with self.assertRaises(ValueError):summarize(observations)
    def test_corpus_and_immutable_counterbalance_plan(self):
        root=Path(__file__).resolve().parents[1]/'benchmarks/cpu-amx'
        files=sorted(root.glob('*.json'));self.assertGreater(len(files),100)
        for path in files:validate_case(json.loads(path.read_text()))
        with tempfile.TemporaryDirectory() as directory:
            plan=collection_plan(Path('/not-executed'),files[:2],Path(directory),2,54,'synthetic',case_validator=validate_case)
            self.assertEqual(len(plan),4);self.assertEqual(len({e['run_id'] for e in plan}),4)

if __name__=='__main__':unittest.main()
