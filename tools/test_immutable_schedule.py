import copy
import unittest
from benchmark_schema.immutable_schedule import FIELDS, validate_immutable_schedule

class ScheduleCounterTests(unittest.TestCase):
    def fixture(self):
        start=dict.fromkeys(FIELDS,0)
        start.update(backend_schedule_seals=1,backend_sealed_tile_visits=16384)
        end=dict(start,backend_fast_submissions=9)
        return {'immutable_schedule_work':{'abi_version':1,'scope':'calling_thread_source_work_counters_not_latency',
                'after_warmups':start,'after_measured_repeats':end}}
    def test_fixed_preparation_and_fast_submissions(self):
        self.assertEqual(validate_immutable_schedule(self.fixture()),[])
    def test_malformed_and_decreasing_counters(self):
        for field in FIELDS:
            for invalid in (-1,True,1.5,None):
                capture=self.fixture();capture['immutable_schedule_work']['after_measured_repeats'][field]=invalid
                self.assertTrue(validate_immutable_schedule(capture))
        capture=self.fixture();capture['immutable_schedule_work']['after_measured_repeats']['backend_sealed_tile_visits']=0
        self.assertTrue(validate_immutable_schedule(capture))
    def test_historical_absence_not_fake_evidence(self):
        self.assertEqual(validate_immutable_schedule({}),[])
        capture=self.fixture();capture['immutable_schedule_work']['scope']='measured_speedup'
        self.assertTrue(validate_immutable_schedule(capture))
