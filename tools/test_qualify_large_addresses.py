"""Physical qualification admission must never turn missing evidence into success."""
import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from qualify_large_addresses import main, strict_json, validate_receipt


def fixture():
    return dict(schema="rns8_large_address_qualification_v1", backend="hip-direct",
        semantic_contract="finite_ring_mod253_centered_i8", m=32769, n=1, k=65536,
        ldc=3, window_rows=128, prefix_count=1, a_allocation_bytes=2147549184,
        b_allocation_bytes=65536, c_allocation_bytes=98307,
        first_a_offset_above_int32=2147483648, sparse_existing_u64_control_modulus=256,
        sparse_existing_u64_high_offset=2147483649, compute_host_pointer_rebasing=False,
        all_rows_and_padding_exact=True, all_owned_device_bytes_retired=True,
        device_allocation_calls=3, device_free_calls=3, matched_m=32767,
        matched_first_row=32639, matched_warmups=2, matched_repeats=9,
        matched_timing_hooks_enabled=False, promotion_eligible=False,
        timing_source="host_monotonic_completed_backend_window", device_budget_bytes=3<<30,
        available_device_bytes_before=4<<30, total_device_bytes=24<<30, reserve_bytes=512<<20,
        observed_requested_device_peak_delta=2147713027, narrow_rejected_before_write=[32768,32769],
        build_manifest=dict(version=2, source_dirty=False, source_commit="a"*40,
            source_digest="b"*64, dependency_digest="c"*64, linkage="static-rns8",
            identity_scope="source-config-and-external-headers", configuration="hip=ON", external_dependencies={}),
        compiled_source_commit="a"*40, target="gfx1100", device="physical-device",
        hip_runtime_version=70100000, hip_driver_version=70100000,
        centered_output_weighted_checksum="-123", matched_i32_host_ns=[100]*9,
        matched_u64_host_ns=[101]*9, nominal_clock_tick_ns=100)


class LargeAddressContracts(unittest.TestCase):
    def test_complete_physical_receipt_and_each_required_field(self):
        good=fixture()
        validate_receipt(good)
        for key in good:
            with self.subTest(missing=key):
                bad=copy.deepcopy(good);del bad[key]
                with self.assertRaises((ValueError, TypeError)):
                    validate_receipt(bad)

    def test_no_boolean_bytes_simulated_spans_dirty_builds_or_incomplete_retirement(self):
        mutations=[
            ("a_allocation_bytes", True), ("a_allocation_bytes", 1),
            ("observed_requested_device_peak_delta", 2147713026),
            ("device_budget_bytes", 1), ("available_device_bytes_before", 2147713027),
            ("total_device_bytes", 1), ("reserve_bytes", 0),
            ("all_owned_device_bytes_retired", False), ("all_rows_and_padding_exact", False),
            ("compute_host_pointer_rebasing", True), ("device_free_calls", 2),
            ("compiled_source_commit", "d"*40), ("target", "unknown"), ("device", ""),
            ("hip_runtime_version", 0), ("narrow_rejected_before_write", [32768]),
            ("centered_output_weighted_checksum", "01"), ("matched_u64_host_ns", [1]*8),
            ("matched_i32_host_ns", [float("nan")]*9), ("nominal_clock_tick_ns", 0)]
        for key,value in mutations:
            with self.subTest(field=key, value=value):
                bad=fixture();bad[key]=value
                with self.assertRaises(ValueError):validate_receipt(bad)
        bad=fixture();bad["build_manifest"]["source_dirty"]=True
        with self.assertRaises(ValueError):validate_receipt(bad)

    def test_json_duplicate_and_nonfinite_evidence_rejected(self):
        for text in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}'):
            with self.assertRaises(ValueError):strict_json(text)

    def test_process_skip_timeout_and_malformed_output_persist_without_retry(self):
        for mode,expected_code,expected_status in (("skip",77,"unrun_or_skipped"),
                ("timeout",1,"failed"),("malformed",1,"failed")):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root=Path(directory);binary=root/"test-binary";binary.write_bytes(b"pinned")
                output=root/"captures"
                def run(command, **kwargs):
                    if mode=="timeout":raise subprocess.TimeoutExpired(command,1)
                    if mode=="malformed":kwargs["stdout"].write('RNS8_LARGE_ADDRESS_RECEIPT {"schema":"wrong"}\n')
                    return subprocess.CompletedProcess(command,0)
                argv=["qualify_large_addresses.py","--binary",str(binary),"--output",str(output),
                    "--device-budget-bytes",str(3<<30),"--runs","5"]
                with patch("sys.argv",argv),patch("qualify_large_addresses.subprocess.run",side_effect=run) as launch:
                    self.assertEqual(main(),expected_code)
                    self.assertEqual(launch.call_count,1)
                receipt=json.loads((output/"receipt.json").read_text())
                self.assertEqual(receipt["status"],expected_status)
                self.assertEqual(len(receipt["records"]),1)
                self.assertTrue(receipt["records"][0]["executable_unchanged"])


if __name__ == "__main__":
    unittest.main()
