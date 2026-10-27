import copy
import hashlib
import json
import unittest
from collect_distributed_calibration import validate_case, validate_capture, summarize
from test_capture_identity_support import synthetic_identity

def case():
    return dict(schema='rns8_distributed_case_v1',m=17,n=19,k=65,prefix=3,tile=16,slots=2,
        devices=[0],transport='local',scope='resident-device-limbs',host_budget=2**30,
        pinned_budget=2**20,device_budget=2**30,warmups=3,repeats=5)

class DistributedCases(unittest.TestCase):
    def test_explicit_axes_and_fragment_membership(self):
        for axis in ('modulus','rows','columns','2d','contraction'):
            c=case();c.update(schema='rns8_distributed_case_v2',axis=axis,
                k_fragments=3 if axis=='contraction' else 0,
                rank_grid_rows=1 if axis=='2d' else 0,rank_grid_cols=1 if axis=='2d' else 0)
            self.assertEqual(validate_case(c),c)
            bad=dict(c,k_fragments=66)
            with self.assertRaises(ValueError):validate_case(bad)
        for grid in ((0,1),(2,2)):
            c.update(axis='2d',k_fragments=0,rank_grid_rows=grid[0],rank_grid_cols=grid[1])
            with self.assertRaises(ValueError):validate_case(c)
    def fixture(self,run=0,multi=False):
        request=case();request.update(devices=[0,1] if multi else [0],transport='pinned' if multi else 'local')
        entry=dict(case=request,run_id=f'synthetic-{run}-{multi}',independent_run=run,command=['synthetic-binary','synthetic-case',f'synthetic-{run}-{multi}'])
        cells=request['m']*request['n'];total=cells*request['prefix'];local=[total-total//2,total//2] if multi else [total]
        ranks=[dict(rank=i,ordinal=i,physical_id=f'synthetic-visible-{i}',target='gfx1100',runtime_version=701,driver_version=701,
            requested_device_reservation=2**20,native_inputs=8*(request['m']*request['k']+request['k']*request['n']),
            packed_inputs=4096,local_output=local[i],gather=total if i==0 else 0) for i in range(len(local))]
        capture=dict(schema='rns8_distributed_microcase_v1',case=request,run_id=entry['run_id'],
            dataset='signed_separable_7x3x5_v1',timer='single_host_steady_clock_nanoseconds',completion_inside_interval=True,
            oracle_inside_interval=False,exact_reference_passed=True,promotion_eligible=False,physical_card_count_known=False,
            allocator_overhead_known=False,published=True,ranks=ranks,arithmetic_certificates=['a'*64]*len(local),
            plan_reused=True,plan_compilation_ns=1,production_timing_hooks_enabled=False,affinity_policy='inherited_not_modified',
            numa_placement_observed=False,host_platform='other',host_hardware_threads=8,
            expected_work=12,local_completed=12,transfers_queued=12,plan_sha256='a'*64,
            caller_output_bytes=request['m']*(request['n']+2)*24,host_owner_reservation=2**20,pinned_reservation=256,
            transfer_bytes_per_call=sum(local[1:]),setup_ns=1,first_ns=1000,samples_ns=[1000]*request['repeats'],
            sustained=[dict(calls=16,elapsed_ns=250000000,mean_call_ns=250000000/16,duration_reached=True,stop_reason='duration') for _ in range(3)],
            calls=1+request['warmups']+request['repeats']+48,setup_scope='retained_preparation')
        capture.update(integrity_requested=False,integrity_completed=False,integrity_failure_mask=0,checking_moduli=[257,263])
        synthetic_identity(capture)
        identity=capture['request_identity'];identity['arguments']=entry['command'][1:];identity.pop('identity_sha256')
        identity['identity_sha256']=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        return entry,capture
    def test_explicit_rank_transport_contract(self):
        c=case();self.assertEqual(validate_case(c),c)
        for devices,transport in [([], 'local'),([0,0],'pinned'),([0,1],'local'),([False],'local'),([0],'peer')]:
            bad=copy.deepcopy(c);bad.update(devices=devices,transport=transport)
            with self.assertRaises(ValueError):validate_case(bad)
        c.update(devices=[2,5,7],transport='pinned');self.assertEqual(validate_case(c),c)
    def test_strict_range_and_typed_resource_caps(self):
        for key,value in [('prefix',1),('slots',0),('slots',4),('m',True),('host_budget',0),('pinned_budget',3),('scope','partial-shard')]:
            bad=case();bad[key]=value
            with self.assertRaises(ValueError):validate_case(bad)
    def test_raw_output_without_wrapper_provenance_cannot_qualify(self):
        with self.assertRaises(ValueError):validate_capture({},dict(case=case(),run_id='unqualified',command=['binary','case','unqualified']))
    def test_integrity_policy_requires_actual_completion_and_own_cohort(self):
        entry,capture=self.fixture();entry['case']['native_integrity']=True
        capture['integrity_requested']=True;capture['integrity_completed']=True
        validate_capture(capture,entry)
        for field,value in [('integrity_completed',False),('integrity_failure_mask',4),('checking_moduli',[257])]:
            bad=copy.deepcopy(capture);bad[field]=value
            with self.assertRaises(ValueError):validate_capture(bad,entry)
    def test_complete_fixture_and_missing_or_false_completion(self):
        entry,capture=self.fixture();validate_capture(capture,entry)
        for key,value in [('published',False),('local_completed',11),('transfers_queued',True),('physical_card_count_known',True),
                ('caller_output_bytes',0),('pinned_reservation',0),('transfer_bytes_per_call',1),('plan_sha256','unknown')]:
            bad=copy.deepcopy(capture);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_capture(bad,entry)
        bad=copy.deepcopy(capture);bad['ranks']=[]
        with self.assertRaises(ValueError):validate_capture(bad,entry)
        bad=copy.deepcopy(capture);bad['sustained'][0]['elapsed_ns']=1
        with self.assertRaises(ValueError):validate_capture(bad,entry)
    def test_incomplete_or_duplicate_independent_inventory_rejected(self):
        rows=[self.fixture(run,multi) for run in range(2) for multi in (False,True)]
        self.assertEqual(len(summarize(rows)[0]['observations']),2)
        with self.assertRaises(ValueError):summarize(rows[:-1])
        with self.assertRaises(ValueError):summarize(rows+[rows[0]])

if __name__=='__main__':unittest.main()
