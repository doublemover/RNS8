import copy
import json
import tempfile
import unittest
from pathlib import Path
from collect_epilogue_calibration import PROFILES,collection_plan,validate_case,validate_capture
from test_capture_identity_support import synthetic_identity
from collect_native_calibration import epilogue_contract,validate_case as validate_native

class EpilogueCollection(unittest.TestCase):
    def case(self):return dict(schema='epilogue_public_case_v1',m=17,n=19,k=257,repeats=9,warmups=3,backend='hip-direct',coefficient_profile='copy',residency='resident')
    def test_every_profile_has_explicit_original_coefficients(self):
        for profile,(alpha,beta) in PROFILES.items():
            case=self.case();case['coefficient_profile']=profile;self.assertEqual(validate_case(case),case)
            for candidate in ('epilogue_direct','epilogue_rocwmma','epilogue_unfused'):
                n=validate_native(dict(candidate=candidate,m=17,n=19,k=257,prefix=20,group=1,unsigned_inputs=False,coefficient_profile=profile))
                actual=epilogue_contract(n)['expression'];self.assertEqual((actual['alpha'],actual['beta']),(alpha,beta))
    def fixture(self):
        case=self.case();entry=dict(case=case,run_id='authored-process')
        c=dict(schema='rns8_public_epilogue_microcase_v1',case=case,run_id=entry['run_id'],dataset='signed_small_mt19937_64_seed4747_v1',seed=4747,
            alpha=0,beta=1,prefix=20,limbs=3,output_ld=21,host_instrumentation_state='hooks_disabled',output_boundary='complete_public_expression_and_padded_signed_limb_export',
            promotion_eligible=False,exact_reference_passed=True,oracle_inside_production_interval=False,source_counts_are_measured_instructions=False,
            active_dot_channels=0,active_addend_channels=20,source_scalar_dot_terms_per_call=0,setup_us=10,proof_diagnostic_us=1,
            first_call=dict(pack_us=0,expression_us=10,export_us=20,total_us=30),
            samples=[dict(pack_us=0,expression_us=10,export_us=20,total_us=30) for _ in range(9)],
            sustained=[dict(calls=16,duration_us=250000,first_input_generation=1,last_input_generation=1,exact_reference_passed=True) for _ in range(3)])
        synthetic_identity(c);return c,entry
    def test_completed_boundary_and_mutation_receipts(self):
        c,e=self.fixture();self.assertEqual(validate_capture(c,e),c)
        for key,value in [('alpha',255),('active_dot_channels',1),('output_boundary','residues_only'),('source_counts_are_measured_instructions',True),('exact_reference_passed',False)]:
            bad=copy.deepcopy(c);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_capture(bad,e)
        for field,value in [('calls',True),('duration_us',100),('last_input_generation',2)]:
            bad=copy.deepcopy(c);bad['sustained'][0][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):validate_capture(bad,e)
        bad=copy.deepcopy(c);bad['samples'][0]['total_us']=20
        with self.assertRaises(ValueError):validate_capture(bad,e)
    def test_bad_cases_rejected(self):
        for key,value in [('m',True),('k',65538),('repeats',2),('warmups',0),('residency','unknown'),('coefficient_profile','modular-only')]:
            case=self.case();case[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_case(case)
        with self.assertRaises(ValueError):validate_native(dict(candidate='wmma',m=17,n=19,k=257,prefix=20,group=1,unsigned_inputs=False,coefficient_profile='zero'))
    def test_each_process_retains_unique_run_receipt_and_complete_case(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);paths=[]
            for profile in ('copy','unit','extreme'):
                c=self.case();c['coefficient_profile']=profile;p=root/(profile+'.json');p.write_text(json.dumps(c));paths.append(p)
            plan=collection_plan(Path('binary'),paths,root/'out',3,1,'collection')
            self.assertEqual(len(plan),9);self.assertEqual(len({e['run_id'] for e in plan}),9)
            self.assertEqual({e['independent_run'] for e in plan},{0,1,2})
            self.assertNotEqual([e['case'] for e in plan[:3]],[e['case'] for e in plan[3:6]])

if __name__=='__main__':unittest.main()
