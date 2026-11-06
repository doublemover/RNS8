"""Authored perf-file ABI controls; no recorder/native module is executed."""

from __future__ import annotations

import copy
from pathlib import Path
import struct
import tempfile
import unittest

from perf_recording_contract import (
    REQUIRED_SAMPLE_FIELDS,
    SAMPLE_IDENTIFIER,
    SAMPLE_READ,
    read_recording,
    validate_decoded_samples,
)


def record(kind, payload=b""):
    return struct.pack("<IHH", kind, 0, len(payload) + 8) + payload


def read_value(lost=0, *, read_format=20, identity=1, group=False):
    words = [1] if group else [100]
    words.extend(1000 for bit in (1, 2) if read_format & bit)
    if group:
        words.append(100)
    if read_format & 4:
        words.append(identity)
    if read_format & 16:
        words.append(lost)
    return struct.pack("<" + "Q" * len(words), *words)


def sample_record(row=None, *, sample_type=REQUIRED_SAMPLE_FIELDS, read_format=0):
    row = dict(pid=41, tid=42, time_ns=123456789, period=1000000, ip=4099, cpu=3) | (row or {})
    words = [row.get("identifier", 1)] if sample_type & SAMPLE_IDENTIFIER else []
    fields = (
        (1, row["ip"]),
        (2, row["pid"] | (row["tid"] << 32)),
        (4, row["time_ns"]),
        (8, 0),
        (64, row.get("identifier", 1)),
        (512, 1),
        (128, row["cpu"]),
        (256, row["period"]),
    )
    words.extend(value for bit, value in fields if sample_type & bit)
    payload = struct.pack("<" + "Q" * len(words), *words)
    if sample_type & SAMPLE_READ:
        payload += read_value(
            row.get("read_lost", 0), read_format=read_format, group=bool(read_format & 8)
        )
    return record(9, payload)


def recording_bytes(
    samples=None,
    *,
    records=None,
    clock_id=1,
    sample_type=REQUIRED_SAMPLE_FIELDS,
    read_format=0,
    flags=None,
    features=None,
    ids=(1,),
):
    """Real header/attribute/fixed-prefix encoding; this is not a measured file."""
    if records is None:
        records = [
            sample_record(row, sample_type=sample_type, read_format=read_format)
            for row in (samples if samples is not None else [{}])
        ]
    data = b"".join(records)
    attr = bytearray(128)
    if flags is None:
        flags = (1 << 25) | (1 << 5) | (1 << 10)
    struct.pack_into("<II5Q", attr, 0, 1, 128, 0, 99, sample_type, read_format, flags)
    struct.pack_into("<i", attr, 92, clock_id)
    attr_size = len(attr) + 16
    ids_bytes = struct.pack("<" + "Q" * len(ids), *ids)
    ids_offset = 104 + attr_size
    attr += struct.pack("<QQ", ids_offset, len(ids_bytes))
    data_offset = ids_offset + len(ids_bytes)
    features = dict(sorted((features or {}).items()))
    position = data_offset + len(data) + 16 * len(features)
    sections, payloads = [], []
    for payload in features.values():
        sections.append(struct.pack("<QQ", position, len(payload)))
        payloads.append(payload)
        position += len(payload)
    feature_bits = sum(1 << value for value in features)
    header = struct.pack(
        "<8s12Q",
        b"PERFILE2",
        104,
        attr_size,
        104,
        attr_size,
        data_offset,
        len(data),
        0,
        0,
        *(feature_bits >> (64 * index) & ((1 << 64) - 1) for index in range(4)),
    )
    return header + attr + ids_bytes + data + b"".join(sections) + b"".join(payloads)


class PerfRecordingContract(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.path = self.root / "perf.data"

    def read(self, data=None, **limits):
        self.path.write_bytes(recording_bytes() if data is None else data)
        return read_recording(self.path, **limits)

    def test_clock_features_are_observations_and_one_anchor_is_not_a_join(self):
        value = self.read(
            recording_bytes(
                features={
                    23: struct.pack("<Q", 1),
                    29: struct.pack("<IiQQ", 1, 1, 9876543210, 123456789),
                    33: struct.pack("<II", 62, 0),
                }
            )
        )
        self.assertTrue(value["complete_record_walk"])
        self.assertEqual(value["sample_records"], 1)
        self.assertEqual(value["clock"]["attribute_clock_id"], 1)
        self.assertEqual(value["clock"]["resolution_ns"], 1)
        self.assertEqual(value["recorded_machine"], 62)
        self.assertFalse(value["clock"]["wall_clock_anchor"]["calibration_or_error_bound_proven"])
        self.assertEqual(value["clock"]["cross_clock_join"], "unbound")
        self.assertFalse(value["promotion_allowed"])
        self.assertEqual(self.read(recording_bytes(clock_id=4))["clock"]["attribute_clock_id"], 4)
        for feature in (struct.pack("<IiQQ", 1, 4, 1, 1), struct.pack("<IiQQ", 2, 1, 1, 1)):
            with self.assertRaises(ValueError):
                self.read(recording_bytes(features={29: feature}))

    def test_loss_events_and_cumulative_reads_are_never_added(self):
        for grouped in (False, True):
            fmt = 20 | (8 if grouped else 0) | 3
            records = [
                sample_record(),
                record(2, struct.pack("<QQ", 1, 7)),
                record(13, struct.pack("<Q", 3)),
            ]
            for lost in (4, 9, 2):
                records.append(
                    record(
                        8,
                        struct.pack("<II", 41, 42)
                        + read_value(lost, read_format=fmt, group=grouped),
                    )
                )
            value = self.read(recording_bytes(records=records, read_format=fmt))
            self.assertEqual(value["loss"]["ring_lost_events"], 7)
            self.assertEqual(value["loss"]["lost_sample_events"], 3)
            self.assertEqual(value["loss"]["cumulative_read_maxima"][0]["maximum"], 9)
            self.assertFalse(value["loss"]["counts_are_additive"])
            self.assertIn("recorded_loss", value["blockers"])
            self.assertFalse(value["loss"]["complete_loss_free_acquisition_proven"])

    def test_embedded_reads_and_sample_id_trailers_are_bounded(self):
        fmt = 20
        value = self.read(
            recording_bytes(
                samples=[dict(read_lost=8)],
                sample_type=REQUIRED_SAMPLE_FIELDS | SAMPLE_READ,
                read_format=fmt,
            )
        )
        self.assertEqual(value["loss"]["cumulative_read_maxima"][0]["maximum"], 8)
        trailer = struct.pack("<IIQII", 41, 42, 123456789, 3, 0)
        flags = (1 << 25) | (1 << 5) | (1 << 18)
        records = [sample_record(), record(2, struct.pack("<QQ", 1, 5) + trailer)]
        self.assertEqual(
            self.read(recording_bytes(records=records, flags=flags))["loss"]["ring_lost_events"], 5
        )
        records[-1] = record(2, struct.pack("<QQ", 1, 5) + trailer[:-1])
        with self.assertRaises(ValueError):
            self.read(recording_bytes(records=records, flags=flags))

    def test_sample_multisets_allow_decoder_reordering_but_preserve_every_field(self):
        samples = [
            dict(pid=41, tid=42, time_ns=11, period=7, ip=4096, cpu=3),
            dict(pid=41, tid=42, time_ns=10, period=9, ip=8192, cpu=4),
        ]
        value = self.read(recording_bytes(samples=samples))
        validate_decoded_samples(value, list(reversed(samples)), pid=41, observed=2, excluded=0)
        for field in samples[0]:
            rows = copy.deepcopy(samples)
            rows[0][field] += 1
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_decoded_samples(value, rows, pid=41, observed=2, excluded=0)
        with self.assertRaises(ValueError):
            validate_decoded_samples(value, samples[:1], pid=41, observed=2, excluded=1)

    def test_incomplete_unsupported_and_overlapping_input_never_gets_a_walk_receipt(self):
        valid = recording_bytes()
        mutations = [
            valid[:100],
            valid[:-1],
            b"2ELIFREP" + valid[8:],
            recording_bytes(flags=1 << 5),
            recording_bytes(sample_type=1),
            recording_bytes(features={27: b"compression"}),
            recording_bytes(records=[record(81)]),
            recording_bytes(records=[record(999)]),
            recording_bytes(ids=(1, 1)),
            recording_bytes(ids=(0,)),
            recording_bytes(records=[record(2, struct.pack("<Q", 1))]),
        ]
        bad = bytearray(valid)
        struct.pack_into("<Q", bad, 40, 104)  # Data may not overlap attributes.
        mutations.append(bytes(bad))
        for index, data in enumerate(mutations):
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.read(data)
        with self.assertRaises(ValueError):
            self.read(valid, max_input_bytes=1)
        with self.assertRaises(ValueError):
            self.read(recording_bytes(samples=[{}, {}]), max_samples=1)


if __name__ == "__main__":
    unittest.main()
