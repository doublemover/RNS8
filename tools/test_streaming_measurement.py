"""Authored observations, never collected hardware evidence."""
from copy import deepcopy
from pathlib import Path
import unittest
from benchmark_schema.streaming_timing import POLICY, UNIT, validate_streaming_observations, raw_host_sample_count, gpu_stage_sample_count
from collect_public_calibration import commands
from measurement_contract import host_observation_contract, host_observation_count
from perf_variance_report import _timing_sample_count, _within_capture_rel_spread
from timing_eligibility import phase_blockers, pair_blockers


def fixture():
    batches=[dict(calls=9,duration_us=90+i*9,setup_us=12,first_source_version=28+i*9,
                  device_allocate_calls_inside_batch=0,device_allocated_bytes_inside_batch=0) for i in range(9)]
    return dict(synthetic_fixture_only=True,repeats=9,warmups=3,
        streaming_overlap=dict(requested=True,capture_status="executed",observation_contract_version=2,batches=batches,
            measured_repeat_count=81,warmup_batch_count=3,warmup_calls_per_batch=9,fill_drain_policy=POLICY,
            schedule="overlapped_three_stream",per_call_latency_available=False,batch_wall_us=162,final_exact_reference_passed=True),
        timing_metadata=dict(host_observation_unit=UNIT,host_result_kind="pipeline_batch_throughput",host_timing_contract_version=2,
            censor_threshold_us=0.2,phase_states={"pack":"unsupported","rns_gemm":"unsupported","crt_export":"unsupported","end_to_end":"measured"}),
        raw_timings_us=dict(pack=[],rns_gemm=[],crt_export=[],end_to_end=[10+i for i in range(9)]),
        timing_summary_us={"end_to_end":dict(median=14,p95=18)})


class StreamingMeasurementTests(unittest.TestCase):
    def test_true_batches_and_independent_stage_observations(self):
        capture=fixture();self.assertEqual(validate_streaming_observations(capture),[])
        self.assertEqual(host_observation_count(capture),9);self.assertEqual(_timing_sample_count(capture),9)
        self.assertEqual(raw_host_sample_count(capture,"end_to_end",9),9)
        self.assertEqual(raw_host_sample_count(capture,"pack",9),0)
        self.assertEqual(gpu_stage_sample_count(capture,9),81)
        self.assertEqual(phase_blockers(capture),[])
        self.assertAlmostEqual(_within_capture_rel_spread(capture),4/14)
    def test_old_repeated_batch_mean_is_one_observation_not_zero_variance(self):
        capture=fixture();capture["streaming_overlap"].pop("observation_contract_version")
        capture["raw_timings_us"]["end_to_end"]=[18]*9
        capture["timing_summary_us"]["end_to_end"]=dict(median=18,p95=18)
        original=deepcopy(capture)
        self.assertEqual(_timing_sample_count(capture),1)
        self.assertIsNone(_within_capture_rel_spread(capture))
        self.assertIn("legacy_repeated_batch_quotients_not_repeat_observations",phase_blockers(capture))
        self.assertEqual(capture,original)
    def test_corrupt_or_cold_receipts_fail(self):
        for field,value in (("calls",8),("duration_us",float("nan")),("first_source_version",27),
                            ("device_allocate_calls_inside_batch",3),("device_allocated_bytes_inside_batch",4194308)):
            capture=fixture();capture["streaming_overlap"]["batches"][0][field]=value
            with self.subTest(field=field):self.assertTrue(validate_streaming_observations(capture))
        for mutation in range(5):
            capture=fixture()
            if mutation==0:capture["raw_timings_us"]["end_to_end"]*=9
            if mutation==1:capture["raw_timings_us"]["pack"]=[1]*81
            if mutation==2:capture["streaming_overlap"]["warmup_calls_per_batch"]=1
            if mutation==3:capture["streaming_overlap"]["per_call_latency_available"]=True
            if mutation==4:capture["streaming_overlap"]["measured_repeat_count"]=9
            self.assertTrue(validate_streaming_observations(capture))
    def test_timer_resolution_applies_to_batch_not_divided_mean(self):
        capture=fixture();capture["timing_metadata"]["censor_threshold_us"]=20
        self.assertEqual(phase_blockers(capture),[])
        capture["streaming_overlap"]["batches"][0]["duration_us"]=10
        self.assertIn("end_to_end_samples_below_resolution",phase_blockers(capture))
    def test_serial_control_has_same_boundary_but_not_call_latency_identity(self):
        capture=fixture();serial=deepcopy(capture);serial["streaming_overlap"]["schedule"]="serial_dependency_chain"
        self.assertEqual(host_observation_contract(capture),host_observation_contract(serial))
        ordinary=deepcopy(capture);ordinary.pop("streaming_overlap");ordinary["timing_metadata"]["host_result_kind"]="single_operation"
        self.assertIn("host_observation_contract_mismatch",pair_blockers(capture,ordinary))
    def test_collector_does_not_request_incompatible_sustained_lane(self):
        plan=commands(Path("binary"),[dict(name="stream",arguments=["--streaming-overlap"])],2,31,Path("output"))
        self.assertEqual({e["mode"] for e in plan},{"instrumented","production"})
        self.assertTrue(all("--sustained-batches" not in e["command"] for e in plan))

if __name__=="__main__":unittest.main()
