"""Authored mapping/ELF custody controls; no native decoder or recorder runs."""

from __future__ import annotations

import copy
from pathlib import Path
import struct
import tempfile
import unittest

from perf_mapping_contract import MappingIndex, frame_custody
from perf_recording_contract import read_recording
from test_elf_module_identity import image
from test_perf_recording_contract import recording_bytes, sample_record


def mmap_record(
    path,
    *,
    pid=41,
    start=4096,
    length=4096,
    offset=0,
    time_ns=100,
    build_id=b"\x01\x23\xab\xcd",
    kind=10,
    clock_trailer=True,
):
    payload = struct.pack("<IIQQQ", pid, pid, start, length, offset)
    if kind == 10:
        payload += struct.pack("<BBH20sII", len(build_id), 0, 0, build_id, 5, 2)
    name = str(path).encode() + b"\0"
    payload += name + bytes((-len(name)) % 8)
    if clock_trailer:
        # Primary fixture has TID, TIME and CPU in sample_id_all; no ID/PERIOD.
        payload += struct.pack("<IIQQ", pid, pid, time_ns, 0)
    return (
        struct.pack("<IHH", kind, 2 | ((1 << 14) if kind == 10 else 0), len(payload) + 8) + payload
    )


class MappingCustody(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.root = Path(scratch.name)
        self.module = self.root / "inert.so"
        self.module.write_bytes(image(executable=True))
        self.recording = self.root / "perf.data"

    def read(self, *maps, records=(), flags=None):
        self.recording.write_bytes(
            recording_bytes(
                records=[*maps, *records, sample_record(dict(ip=4275))],
                flags=flags if flags is not None else (1 << 25) | (1 << 5) | (1 << 10) | (1 << 18),
            )
        )
        return read_recording(self.recording)

    def lookup(self, value, **changes):
        return MappingIndex(value).lookup(
            **(dict(pid=41, ip=4275, time_ns=123456789, clock_id=1) | changes)
        )

    def test_real_mmap2_bytes_bind_native_leaf_and_caller_to_exact_current_ELF(self):
        value = self.read(mmap_record(self.module))
        row, reason = self.lookup(value)
        self.assertIsNone(reason)
        self.assertEqual(row["native_build_id"], "0123abcd")
        sample = dict(
            pid=41,
            time_ns=123456789,
            ip=4275,
            dso_path=str(self.module),
            dso_build_id="0123abcd",
            symbol="leaf",
            callchain=[dict(ip=4276, dso=self.module.name, symbol="caller")],
        )
        report = frame_custody([sample], value, machine=62)
        self.assertEqual(report["selected_frame_observations"], 2)
        self.assertEqual(
            sum(
                row["observations"]
                for row in report["counts"]
                if row["state"] == "recorded_build_id_and_current_executable_extent"
            ),
            2,
        )
        self.assertFalse(report["complete_transient_loaded_image_history_verified"])
        self.module.write_bytes(image(value=b"\x11\x22", executable=True))
        changed = frame_custody([sample], value, machine=62)
        self.assertTrue(
            all(
                row["state"] == "current_ELF_identity_or_executable_extent_unresolved"
                for row in changed["observations"]
            )
        )

    def test_overlap_wrong_pid_foreign_clock_future_and_exec_never_borrow_identity(self):
        value = self.read(mmap_record(self.module))
        for changes in (dict(pid=99), dict(clock_id=4), dict(time_ns=100), dict(ip=8192)):
            self.assertIsNone(self.lookup(value, **changes)[0])
        overlap = self.read(
            mmap_record(self.module), mmap_record(self.module, start=4200, time_ns=200)
        )
        self.assertIsNone(self.lookup(overlap)[0])
        boundary = copy.deepcopy(value)
        boundary["process_mapping_boundaries"] = [dict(pid=41, time_ns=200, clock_id=1)]
        self.assertIsNone(self.lookup(boundary)[0])
        boundary["process_mapping_boundaries"][0]["time_ns"] = 123456790
        self.assertIsNotNone(self.lookup(boundary)[0])
        boundary["process_mapping_boundaries"][0]["time_ns"] = None
        self.assertIsNone(self.lookup(boundary)[0])

    def test_legacy_untimed_malformed_and_overflow_mappings_are_explicit(self):
        legacy = self.read(mmap_record(self.module, kind=1))
        self.assertEqual(self.lookup(legacy)[1], "recorded_executable_protection_unavailable")
        untimed = self.read(
            mmap_record(self.module, clock_trailer=False), flags=(1 << 25) | (1 << 5) | (1 << 10)
        )
        self.assertEqual(
            self.lookup(untimed)[1], "mapping_timestamp_missing_foreign_or_not_before_sample"
        )
        raw = bytearray(mmap_record(self.module))
        raw[8 + 32] = 21
        with self.assertRaisesRegex(ValueError, "build-ID union"):
            self.read(raw)
        with self.assertRaisesRegex(ValueError, "overflows"):
            self.read(mmap_record(self.module, start=(1 << 64) - 1))


if __name__ == "__main__":
    unittest.main()
