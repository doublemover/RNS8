import copy
import hashlib
import json
import unittest
from collect_system_roof import validate_case,validate_capture,resources,summarize,checksums,floor_sum
from test_capture_identity_support import synthetic_identity

def case():
    return dict(schema='rns8_system_roof_case_v1',device=0,mode='lds-read-write',words=384,blocks=3,
        iterations=17,stride=32,displacement_words=0,budget_bytes=2**30,warmups=2,repeats=3)

class SystemRoofContracts(unittest.TestCase):
    def fixture(self,run=0):
        request=case();entry=dict(case=request,run_id=f'synthetic-{run}',independent_run=run,
            command=['synthetic-binary','synthetic-case',f'synthetic-{run}'])
        c=dict(schema='rns8_system_roof_microcase_v1',case=request,run_id=entry['run_id'],device=0,
            target='synthetic-gfx1100',device_name='synthetic',runtime_version=701,driver_version=701,setup_ns=1,first_ns=1000,
            host_platform='other',device_uuid='0'*32,device_bdf='synthetic-bdf',
            capture_instantiate_ns=None,actual_hbm_bytes=None,actual_lds_transactions=None,
            physical_cache_eviction_proven=False,clock_power_policy='inherited_unobserved',
            lds_bank_mapping='unknown_until_named_target_instruction_review',
            timer='host_steady_clock_nanoseconds_and_same_stream_HIP_events',
            host_scope='complete_borrowed_probe_on_owned_stream_including_requested_displacement_and_stream_completion',
            event_scope='probe_only_excludes_displacement_includes_recorded_operation_and_required_LDS_barriers',
            arithmetic='uint32_mod_2_32_probe_only',oracle_inside_interval=False,exact_reference_passed=True,
            production_timing_hooks_enabled=False,promotion_eligible=False,allocator_overhead_known=False,
            input_seed=2891336453,input_multiplier=747796405,descriptor_buffer_kind='portable_pinned_owned',isa_verified=False,isa_sha256=None,
            samples=[dict(instrumented_host_ns=1000,production_host_ns=900,device_event_ns=None,exact_reference_passed=True) for _ in range(3)],
            sustained=[dict(calls=16,elapsed_ns=250000000,duration_target_met=True,call_cap_reached=False,exact_reference_passed=True) for _ in range(3)],
            calls=57,displacement_calls=0,**resources(request))
        c['source_u64_sum'],c['output_u64_sum']=checksums(request['mode'],request['words'],request['blocks'],request['iterations'])
        c['checksum_contract']='sum_of_all_canonical_uint32_values_mod_2_64_not_a_correctness_substitute'
        synthetic_identity(c);identity=c['request_identity'];identity['arguments']=entry['command'][1:];identity.pop('identity_sha256')
        identity['identity_sha256']=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        return entry,c
    def test_case_admission_and_live_payload(self):
        c=case();self.assertEqual(validate_case(c),c);q=resources(c)
        self.assertEqual(q['modeled_lds_read_bytes'],384*4*17)
        self.assertEqual(q['modeled_lds_write_bytes'],384*4*18)
        for key,value in [('words',383),('iterations',True),('stride',3),('blocks',0),('budget_bytes',q['admitted_payload_bytes']-1)]:
            bad=case();bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_case(bad)
        c.update(mode='global-copy',words=257,blocks=4,iterations=1,stride=1,displacement_words=1024)
        validate_case(c);self.assertEqual(resources(c)['modeled_displacement_bytes'],8192)
    def test_exact_large_domain_checksum_without_large_allocation(self):
        for n in (1,2,127,128,257,1024):
            actual=sum((747796405*i+2891336453)%2**32 for i in range(n))
            self.assertEqual(checksums('global-copy',n,1,1),(actual,actual))
        self.assertEqual(floor_sum(7,11,13,17),sum((13*i+17)//11 for i in range(7)))
    def test_censored_events_remain_missing_and_all_work_is_accounted(self):
        entry,c=self.fixture();validate_capture(c,entry)
        for key,value in [('actual_hbm_bytes',0),('calls',56),('exact_reference_passed',False),('promotion_eligible',True),
                ('pinned_payload_bytes',0),('modeled_lds_write_bytes',0),('physical_cache_eviction_proven',True)]:
            bad=copy.deepcopy(c);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_capture(bad,entry)
        bad=copy.deepcopy(c);bad['samples'][0]['device_event_ns']=0
        with self.assertRaises(ValueError):validate_capture(bad,entry)
        bad=copy.deepcopy(c);bad['sustained'][0]['elapsed_ns']=100
        with self.assertRaises(ValueError):validate_capture(bad,entry)
    def test_independent_inventory_and_no_automatic_rates(self):
        observations=[self.fixture(i) for i in range(5)]
        rows=summarize(observations);self.assertEqual(len(rows),1);self.assertFalse(rows[0]['planner_rate_admitted'])
        with self.assertRaises(ValueError):summarize([observations[0],observations[0]])
        bad=copy.deepcopy(observations);bad[1][1]['request_identity']['arguments']=['different']
        with self.assertRaises(ValueError):summarize(bad)

if __name__=='__main__':unittest.main()
