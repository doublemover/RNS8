from benchmark_cases import case_root
from copy import deepcopy
import json
from pathlib import Path
import unittest
from benchmark_schema.rocwmma_packing import FIELDS,POLICIES,packing_policy,validate_rocwmma_packing
from observation_cohorts import aggregate_captures,evidence
from test_capture_identity_support import synthetic_independent_runs
from test_benchmark_sweep_support import bounded_capture


class RocwmmaPacking(unittest.TestCase):
    def fixture(self,policy):
        capture=bounded_capture('rocwmma',100)
        capture['rocwmma_packing']=dict(abi_version=1,policy=policy,
            scope='calling_thread_successful_submissions_not_bytes_or_time',
            after_warmups=dict.fromkeys(FIELDS,0),after_measured_repeats=dict.fromkeys(FIELDS,0))
        return capture

    def test_all_policies_and_counter_rejection(self):
        for policy in POLICIES:
            capture=self.fixture(policy);self.assertEqual(validate_rocwmma_packing(capture),[])
            for key in FIELDS:
                bad=deepcopy(capture);bad['rocwmma_packing']['after_measured_repeats'][key]=-1
                self.assertTrue(validate_rocwmma_packing(bad))
            for key,value in [('abi_version',True),('policy','auto'),('scope','time')]:
                bad=deepcopy(capture);bad['rocwmma_packing'][key]=value
                self.assertTrue(validate_rocwmma_packing(bad))
        for policy,key in [('packed','direct_a_blocks'),('direct-a','a_panel_hits'),('reuse','direct_a_blocks')]:
            bad=self.fixture(policy);bad['rocwmma_packing']['after_measured_repeats'][key]=1
            self.assertTrue(validate_rocwmma_packing(bad))
        bad=self.fixture('packed');bad['rocwmma_packing']['after_warmups']['a_pack_launches']=2
        self.assertTrue(validate_rocwmma_packing(bad))

    def test_policies_partition_independent_process_observations(self):
        captures=[]
        for policy in POLICIES:captures.extend(synthetic_independent_runs([self.fixture(policy)],2))
        cohorts=aggregate_captures(captures)
        self.assertEqual(len(cohorts),4)
        self.assertEqual({packing_policy(c) for c in cohorts},set(POLICIES))
        self.assertTrue(all(evidence(c)['independent_runs']==2 for c in cohorts))
        self.assertEqual(packing_policy({'backend_selected':'rocwmma'}),'packed')

    def test_matched_manifest_contains_complete_controls(self):
        cases=json.loads((case_root() / 'public-calibration/rocwmma-packing.json').read_text())['cases']
        groups={}
        for case in cases:
            a=case['arguments'];v=dict(zip(a[::2],a[1::2]))
            key=tuple(v[k] for k in ('--semantics','--m','--n','--k','--bound-mode'))
            groups.setdefault(key,set()).add(v['--rocwmma-packing-policy'])
        self.assertTrue(groups);self.assertTrue(all(p==set(POLICIES) for p in groups.values()))


if __name__=='__main__':unittest.main()
