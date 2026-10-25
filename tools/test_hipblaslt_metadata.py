import copy
import json
import unittest
from benchmark_schema.hipblaslt_metadata import FIELDS,PREPACK_FIELDS,validate_hipblaslt_metadata,validate_hipblaslt_prepack,validate_hipblaslt_algorithms,hipblaslt_algorithm_signature
from collect_public_calibration import commands
from pathlib import Path

class MetadataContracts(unittest.TestCase):
    def prepack_fixture(self,policy):
        def counts(calls):
            result=dict.fromkeys(PREPACK_FIELDS,0)
            for operand in ('a','b'):
                if policy in (operand,'both'):
                    result[operand+'_retained_builds']=1
                    result[operand+'_retained_hits']=calls-1
                    result[operand+'_retained_planes_packed']=9
                else:result[operand+'_transient_planes_packed']=9*calls
            return result
        return dict(backend_selected='hipblaslt',hipblaslt_operand_prepack=dict(abi_version=1,policy=policy,
            scope='calling_thread_successful_pack_submissions_and_cache_matches_not_bytes_or_time',
            after_warmups=counts(3),after_measured_repeats=counts(12)))

    def test_independent_operand_receipts_and_rejection(self):
        for policy in ('none','a','b','both'):
            capture=self.prepack_fixture(policy);self.assertEqual(validate_hipblaslt_prepack(capture),[])
            for key in PREPACK_FIELDS:
                bad=copy.deepcopy(capture);bad['hipblaslt_operand_prepack']['after_measured_repeats'][key]=-1
                self.assertTrue(validate_hipblaslt_prepack(bad))
            for operand in ('a','b'):
                if policy not in (operand,'both'):
                    bad=copy.deepcopy(capture);bad['hipblaslt_operand_prepack']['after_measured_repeats'][operand+'_retained_hits']=1
                    self.assertTrue(validate_hipblaslt_prepack(bad))
            bad=copy.deepcopy(capture);bad['backend_selected']='cpu-reference'
            self.assertTrue(validate_hipblaslt_prepack(bad))
        for key,value in (('abi_version',True),('policy','auto'),('scope','bytes')):
            bad=self.prepack_fixture('both');bad['hipblaslt_operand_prepack'][key]=value
            self.assertTrue(validate_hipblaslt_prepack(bad))
        bad=self.prepack_fixture('both');bad['hipblaslt_operand_prepack']['after_measured_repeats']['a_retained_planes_packed']=0
        self.assertTrue(validate_hipblaslt_prepack(bad))

    def fixture(self):
        def counts(calls):
            d=dict.fromkeys(FIELDS,0)
            d.update(descriptor_create_calls=10*calls,descriptor_destroy_calls=10*calls,
                attribute_set_calls=4*calls,descriptor_sets_created=2*calls,
                descriptor_set_hits=38*calls,matmul_block_calls=40*calls,
                algorithm_cache_hits=40*calls,maximum_live_sets=2)
            return d
        return dict(hipblaslt_metadata=dict(abi_version=1,policy='per_call_distinct_geometry',
            scope='calling_thread_library_API_attempt_counts_not_allocator_bytes_or_device_work',
            cross_call_raw_descriptor_retention=False,after_warmups=counts(3),after_measured_repeats=counts(12)))
    def algorithm_fixture(self):
        result=self.fixture();result['backend_selected']='hipblaslt'
        def observation(calls):
            return dict(available=True,complete=True,index_queries=2,entries=[dict(device_id=0,m=16,n=32,k=k,scratch_ld=32,workspace_bytes=4096,attribute_abi=1,algorithm_index=index,matmul_calls=20*calls) for k,index in ((65536,7),(16,8))])
        result['hipblaslt_algorithms']=dict(abi_version=1,scope='library_build_device_padded_geometry_algorithm_index',query_policy='opt_in_once_per_cached_algorithm',after_warmups=observation(3),after_measured_repeats=observation(12))
        return result
    def test_selected_algorithm_identity_and_unknown_are_distinct(self):
        result=self.algorithm_fixture();self.assertEqual(validate_hipblaslt_algorithms(result),[])
        for phase in ('after_warmups','after_measured_repeats'):
            snapshot=result['hipblaslt_algorithms'][phase];snapshot.update(available=False,complete=False,index_queries=0)
            for entry in snapshot['entries']:entry['algorithm_index']=None
        self.assertEqual(validate_hipblaslt_algorithms(result),[])
    def test_algorithm_comparison_rejects_unknown_and_separates_physical_choices(self):
        result=self.algorithm_fixture()
        result.update(build_manifest=dict(source_digest='a'*64,dependency_digest='b'*64,configuration='one-build'),backend_metadata=dict(accelerator_version='one-library'),device=dict(device_id=0,hip_runtime_version=7,hip_driver_version=8,target_cache_key='one-target',target_instance_id='one-device'))
        expected=hipblaslt_algorithm_signature(result);self.assertIsNotNone(expected)
        changed=copy.deepcopy(result);changed['hipblaslt_algorithms']['after_measured_repeats']['entries'][0]['algorithm_index']=99
        self.assertNotEqual(hipblaslt_algorithm_signature(changed),expected)
        for section,field in (('build_manifest','source_digest'),('backend_metadata','accelerator_version'),('device','target_cache_key')):
            changed=copy.deepcopy(result);changed[section][field]='different'
            self.assertNotEqual(hipblaslt_algorithm_signature(changed),expected)
        result['hipblaslt_algorithms']['after_measured_repeats']['complete']=False
        self.assertIsNone(hipblaslt_algorithm_signature(result))
    def test_selected_algorithm_mutations_fail(self):
        for field,value in (('algorithm_index',True),('algorithm_index',None),('k',0),('matmul_calls',1),('attribute_abi',2)):
            result=self.algorithm_fixture();result['hipblaslt_algorithms']['after_measured_repeats']['entries'][0][field]=value
            with self.subTest(field=field):self.assertTrue(validate_hipblaslt_algorithms(result))
        result=self.algorithm_fixture();result['hipblaslt_algorithms']['after_measured_repeats']['entries']*=2
        self.assertTrue(validate_hipblaslt_algorithms(result))
        result=self.algorithm_fixture();result['hipblaslt_algorithms']['after_measured_repeats']['index_queries']=481
        self.assertTrue(validate_hipblaslt_algorithms(result))
    def test_successful_full_tail_lifecycle(self):
        self.assertEqual(validate_hipblaslt_metadata(self.fixture()),[])
    def test_each_counter_mutation_rejected(self):
        for key in FIELDS:
            f=self.fixture();f['hipblaslt_metadata']['after_measured_repeats'][key]+=1
            with self.subTest(key=key):self.assertTrue(validate_hipblaslt_metadata(f))
    def test_scope_and_ownership_mutations(self):
        for key,value in [('abi_version',2),('scope','device_time'),('cross_call_raw_descriptor_retention',True),('policy','auto')]:
            f=self.fixture();f['hipblaslt_metadata'][key]=value;self.assertTrue(validate_hipblaslt_metadata(f))
    def test_physical_controls_interleaved_and_reversed(self):
        cases=[dict(name=n,arguments=['--backend','hipblaslt','--hipblaslt-metadata-policy',n]) for n in ('per-call','per-block')]
        plan=commands(Path('binary'),cases,2,1,Path('out'))
        self.assertEqual([e['case'] for e in plan if e['position']==0],['per-call','per-block','per-block','per-call'])
    def test_operand_retention_manifest_composes_all_physical_controls(self):
        path=Path(__file__).resolve().parents[1]/'benchmarks/public-calibration/hipblaslt-prepack.json'
        cases=json.loads(path.read_text())['cases'];groups={}
        for case in cases:
            args=case['arguments'];values=dict(zip(args[::2],args[1::2]))
            key=tuple(values[k] for k in ('--semantics','--m','--n','--k'))
            groups.setdefault(key,set()).add((values['--hipblaslt-prepack-policy'],values['--hipblaslt-metadata-policy']))
        self.assertTrue(groups)
        expected={(retained,metadata) for retained in ('none','a','b','both') for metadata in ('per-call','per-block')}
        self.assertTrue(all(values==expected for values in groups.values()))
        self.assertTrue(any(key[-1]=='65537' for key in groups))

if __name__=='__main__':unittest.main()
