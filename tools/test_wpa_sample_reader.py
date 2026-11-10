"""Authored WPA CSV contracts; numeric rows are synthetic, not measurements."""

from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import wpa_sample_reader as wpa


def sample_fields(pid=222, *, image="app.exe"):
    return [
        f"{image} ({pid})",
        "223",
        "0",
        "0.000000001",
        "app!rns8_gemm",
        "app.exe",
        "rns8_gemm",
        "1.25",
        "1",
    ]


class WpaSamples(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.path = Path(scratch.name) / "native.csv"
        self.field_limit = csv.field_size_limit()

    def write(self, rows, headers=wpa.HEADERS):
        with self.path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(headers)
            writer.writerows(rows)

    def read(self):
        return wpa.read_samples(self.path, pid=222, executable="app.exe")

    def test_exact_units_selected_pid_threads_and_foreign_idle_are_preserved(self):
        idle = sample_fields(0, image="Idle")
        idle[-1] = "0"
        self.write([sample_fields(), sample_fields(999), idle])
        value = self.read()
        self.assertEqual(value["observed_rows"], 3)
        self.assertEqual(value["excluded_other_process_rows"], 2)
        self.assertEqual(value["samples"][0]["time_ns"], 1)
        self.assertEqual(value["total_sampled_cpu_weight_ns"], 1250000)
        self.assertFalse(value["promotion_allowed"])
        self.assertFalse(value["loaded_module_or_PDB_identity_verified"])

    def test_aggregated_wrong_image_missing_units_and_locale_are_not_guessed(self):
        for index, value in ((8, "2"), (7, "1,25"), (3, "1e-3"), (1, "-1"), (0, "other.exe (222)")):
            row = sample_fields()
            row[index] = value
            self.write([row])
            with self.assertRaises(ValueError):
                self.read()
        self.write(
            [sample_fields()],
            headers=tuple(
                "TimeStamp" if value == "TimeStamp (s)" else value for value in wpa.HEADERS
            ),
        )
        with self.assertRaisesRegex(ValueError, "columns/locale/units"):
            self.read()

    def test_empty_foreign_only_partial_overlong_and_byte_limits_are_failures(self):
        for rows in ([], [sample_fields(999)], [sample_fields()[:-1]]):
            self.write(rows)
            with self.assertRaises(ValueError):
                self.read()
        row = sample_fields()
        row[4] = "x" * 16385
        self.write([row])
        with self.assertRaisesRegex(ValueError, "oversized"):
            self.read()
        self.write([sample_fields()])
        with patch.object(wpa, "MAX_BYTES", 32), self.assertRaises(ValueError):
            self.read()
        self.assertEqual(csv.field_size_limit(), self.field_limit)


if __name__ == "__main__":
    unittest.main()
