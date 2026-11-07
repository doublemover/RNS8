"""Authored native raw allocation controls; these are never performance evidence."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

import heaptrack_native as native


def raw_fixture(events="", *, executable="/tmp/rns8 exact é"):
    return (
        f"v 10500 3\nx {len(executable.encode()):x} {executable}\n"
        "X /tmp/rns8 --synthetic\nI 1000 100000\nm 1 -\nm 1 x 1000 0 100\n"
        "t 1010 0\nc 0\n" + events + "c a\nR 2\n"
    )


class NativeHeaptrack(unittest.TestCase):
    def read(self, source):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.heaptrack"
            path.write_text(source, encoding="utf-8", newline="\n")
            return native.read_allocations(
                path, pid=1234, execution_id="authored-run", executable="/tmp/rns8 exact é"
            )

    def test_real_raw_generations_free_size_recovery_lifetime_bounds_and_rss_are_separate(self):
        result = self.read(raw_fixture("+ 20 1 aa\nc 2\n- aa\n+ 40 1 aa\nc 3\n- aa\n- bb\n"))
        first, second = result["allocations"]
        self.assertEqual([first["generation"], second["generation"]], [1, 2])
        self.assertEqual(first["bytes"], 32)
        self.assertEqual(first["lifetime_bounds_ns"], dict(minimum=0, maximum=3_000_000))
        self.assertEqual(second["lifetime_bounds_ns"], dict(minimum=0, maximum=9_000_000))
        self.assertEqual(result["summary"]["requested_bytes"], 96)
        self.assertEqual(result["summary"]["observed_live_peak_bytes"], 64)
        self.assertEqual(result["summary"]["observed_retained_bytes"], 0)
        self.assertEqual(result["summary"]["by_trace"]["1"]["freed_bytes"], 96)
        self.assertEqual(result["summary"]["unmatched_frees"], 1)
        self.assertEqual(result["sampled_rss_max_bytes"], 8192)
        for field in (
            "pre_capture_live_bytes",
            "exact_process_peak_bytes",
            "device_or_pinned_bytes",
        ):
            self.assertIsNone(result[field])
        self.assertFalse(result["production_latency"])
        self.assertFalse(result["complete_interception_proven"])

    def test_retained_payload_and_final_unbounded_time_do_not_become_zero_lifetime(self):
        result = self.read(raw_fixture("+ 0 1 aa\n+ 30 1 bb\n") + "+ 1 1 cc\n- cc\n")
        self.assertEqual(result["summary"]["observed_retained_bytes"], 48)
        self.assertIsNone(result["allocations"][0]["lifetime_bounds_ns"])
        self.assertIsNone(result["allocations"][2]["lifetime_bounds_ns"])
        self.assertIsNone(result["time_buckets"][-1]["upper_ns"])

    def test_native_trace_tree_and_module_generations_do_not_guess_symbol_ownership(self):
        result = self.read(raw_fixture("t 1020 1\n+ 10 2 aa\nm 1 -\nt 2020 0\n+ 10 3 bb\n"))
        self.assertEqual([row["parent"] for row in result["trace_nodes"]], [0, 1, 0])
        self.assertEqual([row["module_generation"] for row in result["trace_nodes"]], [1, 1, 2])
        self.assertEqual(
            result["symbol_attribution"], "raw_addresses_only_no_loaded_symbol_identity"
        )
        self.assertEqual(result["memory_domain"], "intercepted_host_heap_requested_payload")

    def test_unknown_version_child_process_bad_fields_and_truncated_or_reversed_data_fail(self):
        original = raw_fixture("+ 20 1 aa\n- aa\n")
        for source in (
            original.replace("10500 3", "10600 3"),
            original.replace("10500 3", "10500 4"),
            original + "v 10500 3\n",
            original + "x 6 /other\n",
            original.replace("t 1010 0", "t 1010 1"),
            original.replace("+ 20 1 aa", "+ 20 2 aa"),
            original.replace("+ 20 1 aa", "+ 20 1 aa\n+ 20 1 aa"),
            original.replace("+ 20 1 aa", "+ 20 1 0"),
            original.replace("c a", "c 3\nc 2"),
            original.replace("I 1000", "I 1001"),
            original.replace("R 2", "R ffffffffffffffff"),
            original.replace("m 1 x 1000 0 100", "m 1 x ffffffffffffffff 1 100"),
            original + "A\n",
            original[:-1],
            original.replace("- aa", "- aa unexpected"),
        ):
            with self.subTest(source=source[-70:]), self.assertRaises(ValueError):
                self.read(source)

    def test_record_trace_and_module_budgets_and_hash_bookends_fail_closed(self):
        for name, limit in (("MAX_RECORDS", 3), ("MAX_TRACES", 0), ("MAX_MODULES", 0)):
            with mock.patch.object(native, name, limit), self.assertRaises(ValueError):
                self.read(raw_fixture("+ 1 1 aa\n"))
        reader = native.bounded_read
        calls = 0

        def changed(path):
            nonlocal calls
            calls += 1
            value, reference = reader(path)
            if calls == 2:
                reference["sha256"] = "f" * 64
            return value, reference

        with (
            mock.patch.object(native, "bounded_read", side_effect=changed),
            self.assertRaisesRegex(ValueError, "changed during decoding"),
        ):
            self.read(raw_fixture("+ 1 1 aa\n"))


if __name__ == "__main__":
    unittest.main()
