import copy
import unittest
from benchmark_schema.hipblaslt_metadata import FIELDS,validate_hipblaslt_metadata
from collect_public_calibration import commands
from pathlib import Path

class MetadataContracts(unittest.TestCase):
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

if __name__=='__main__':unittest.main()
