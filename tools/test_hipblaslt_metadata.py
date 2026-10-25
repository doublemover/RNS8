import copy
import json
import unittest
from benchmark_schema.hipblaslt_metadata import FIELDS,PREPACK_FIELDS,validate_hipblaslt_metadata,validate_hipblaslt_prepack
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
