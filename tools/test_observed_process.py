"""Native child resource custody, unit conversion, failure and timeout controls."""

from __future__ import annotations

import copy
from pathlib import Path
import platform
import sys
import tempfile
import unittest

from observed_process import observed_capture, resource_record
from static_analysis import memory_summary


class OwnedResourceCapture(unittest.TestCase):
    def capture(self, code, timeout=15):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root / "temp") as directory:
            return observed_capture(
                # Windows venv redirectors spawn another process; the native
                # observer deliberately excludes descendants on that platform.
                [sys._base_executable, "-c", code],
                resource_path=Path(directory) / "resources.json",
                cwd=root,
                label="native-resource-control",
                timeout_seconds=timeout,
            )

    def test_native_peak_is_measured_and_bound_to_the_actual_child(self):
        value = self.capture(
            "x = bytearray(32 * 1024**2); x[::4096] = b'x' * (len(x) // 4096); "
            "print('complete child output')"
        )
        self.assertEqual(value["returncode"], 0)
        self.assertEqual(value["stdout"].strip(), "complete child output")
        document = value["child_resources"]
        self.assertIsNotNone(document, value["resource_observation_error"])
        self.assertEqual(document["status"], "observed", document.get("resource_error"))
        self.assertEqual(document["os"], platform.system())
        self.assertGreaterEqual(document["peak_memory_bytes"], 32 * 1024**2)
        self.assertEqual(document["process_id"], value["process_id"])
        self.assertNotEqual(document["process_id"], value["resource_supervisor"]["process_id"])
        for key, replacement in (
            ("peak_memory_bytes", document["peak_memory_bytes"] + 1),
            ("process_id", value["resource_supervisor"]["process_id"]),
            ("returncode", 3),
            ("command", ["different-tool"]),
            ("elapsed_seconds", value["elapsed_seconds"] + 1),
            ("scope", "whole-machine-peak"),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                changed = copy.deepcopy(document)
                changed[key] = replacement
                resource_record(changed, value["command"], value["resource_supervisor"])
        summary = memory_summary([value, {"child_resources": None}])
        self.assertEqual(summary["peak_process_memory_bytes"], document["peak_memory_bytes"])
        self.assertEqual(summary["memory_observed_processes"], 1)
        self.assertEqual(summary["memory_unobserved_processes"], 1)

    def test_native_failure_preserves_exit_and_diagnostics(self):
        value = self.capture("import sys; print('native rejection', file=sys.stderr); sys.exit(7)")
        self.assertEqual(value["returncode"], 7)
        self.assertIn("native rejection", value["stderr"])
        self.assertEqual(value["child_resources"]["returncode"], 7)

    def test_interrupted_resource_acquisition_cannot_supply_a_peak(self):
        value = self.capture("import time; time.sleep(15)", timeout=0.5)
        self.assertTrue(value["timed_out"])
        self.assertIsNone(value["child_resources"])
        self.assertTrue(value["resource_observation_error"])
        self.assertIsNone(memory_summary([value])["peak_process_memory_bytes"])


if __name__ == "__main__":
    unittest.main()
