import copy
import unittest
from native_dataset import inputs,model,mt19937_64,PROFILES
from collect_native_calibration import validate_case,validate_capture
from test_collect_native_calibration import case_for,capture_for

class NativeDatasetTests(unittest.TestCase):
    def test_standard_mt19937_64_reference(self):
        self.assertEqual(next(mt19937_64(5489)),14514284786278117030)
    def test_distributions_and_group_accounting(self):
        for unsigned in (False,True):
            for profile in PROFILES:
                for group in (1,2,4):
                    value=model(17,19,17,unsigned,group,profile,3135,True)
                    hist=value['required_prefix_histogram']
                    self.assertEqual(sum(hist),4)
                    self.assertEqual(sum(p*x for p,x in enumerate(hist)),value['required_channels'])
                    self.assertEqual(sum(((p+group-1)//group)*x for p,x in enumerate(hist)),value['adaptive_group_work_items'])
                    if profile=='zero-tiles-v2':self.assertEqual(hist[0],3)
                    if profile!='full-width-v1':self.assertLess(value['adaptive_group_work_items'],value['fixed_capacity_group_work_items'])
    def test_strict_capture_binding(self):
        for candidate in ('wmma','pack_gemm','adaptive_grid','adaptive_compact','global_minimum'):
            case=case_for(candidate,prefix=20,dataset_profile='mixed-tiles-v2',dataset_seed=3135)
            capture=capture_for(case);validate_capture(capture,case,'fixture-run',3,1)
            for field in ('required_channels','adaptive_group_work_items','input_hash_fnv1a64','zero_tiles'):
                bad=copy.deepcopy(capture);bad['dataset_model'][field]=None
                with self.assertRaises(ValueError):validate_capture(bad,case,'fixture-run',3,1)
            changed=dict(case,dataset_seed=3136)
            with self.assertRaises(ValueError):validate_capture(capture,changed,'fixture-run',3,1)
    def test_degenerate_and_irrelevant_controls(self):
        with self.assertRaises(ValueError):model(16,16,17,False,1,'mixed-tiles-v2',3135,True)
        for changes in ({'dataset_seed':True},{'dataset_profile':'unversioned'},{'prefix':9},{'m':16}):
            with self.assertRaises(ValueError):case_for('adaptive_grid',**dict({'dataset_profile':'mixed-tiles-v2','dataset_seed':3135},**changes))
        with self.assertRaises(ValueError):case_for('queue',dataset_profile='uniform-small-v2',dataset_seed=3135)

    def test_global_uniform_prefix_uses_same_certificate_and_counts_uniform_work(self):
        for unsigned in (False,True):
            for group in (1,2,4):
                adaptive=model(17,19,17,unsigned,group,'zero-tiles-v2',3135,True)
                uniform=model(17,19,17,unsigned,group,'zero-tiles-v2',3135,False,True)
                self.assertEqual(uniform['required_prefix_histogram'],adaptive['required_prefix_histogram'])
                self.assertEqual(uniform['globally_required_prefix'],adaptive['globally_required_prefix'])
                self.assertEqual(uniform['actual_candidate_group_work_items'],4*((max(1,uniform['globally_required_prefix'])+group-1)//group))
                self.assertGreater(uniform['actual_candidate_group_work_items'],adaptive['actual_candidate_group_work_items'])
                self.assertEqual(len(uniform['global_bound_magnitude_limbs']),3)
