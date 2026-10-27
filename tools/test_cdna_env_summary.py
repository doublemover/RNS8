#!/usr/bin/env python3
"""Provided-log normalization never infers runtime/SMI device correlation."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cdna_env_summary as summary


class SummaryTests(unittest.TestCase):
    def build(self, root, devices="4,7", **options):
        return summary.build_summary(root, devices_option=devices, environment={}, **options)

    def test_missing_logs_do_not_fabricate_devices_or_readiness(self):
        with tempfile.TemporaryDirectory() as directory:
            data = self.build(Path(directory), dry_run=True)
            self.assertEqual(data["physical_devices"], [])
            self.assertEqual(data["requested_devices"], ["4", "7"])
            self.assertEqual(data["requested_gpu_count"], 2)
            self.assertIsNone(data["node_gpu_count"])
            self.assertIsNone(data["visible_gpu_count"])
            self.assertEqual(data["physical_device_mapping_source"], "unknown")
            self.assertFalse(data["runtime_to_physical_mapping_verified"])
            self.assertFalse(data["rocprofv3_ready"])
            self.assertFalse(data["rccl_ready"])
            self.assertFalse(data["promotion_allowed"])

    def test_indexed_smi_preserves_evidence_without_correlating_ordinals(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "rocm_smi_showallinfo.log").write_text(
                "GPU[7] : Card series: MI300X-seven\nGPU[7] : PCI Bus: 0000:10:00.0\n"
                "GPU[7] : NUMA Node: 1\nGPU[4] : PCI Bus: 0000:80:00.0\nGPU[4] : ASIC: gfx942\n")
            (root / "rocminfo.log").write_text("Name: gfx942\nName: gfx1100\n")
            data = self.build(root)
            self.assertEqual(data["observed_smi_device_count"], 2)
            records = {item["physical_device_id"]: item for item in data["physical_devices"]}
            self.assertEqual(set(records), {4, 7})
            self.assertEqual(records[7]["bdf"], "0000:10:00.0")
            self.assertEqual(records[7]["numa_node"], 1)
            self.assertIsNone(records[7]["target_arch"])
            self.assertEqual(records[4]["target_arch"], "gfx942")
            self.assertIsNone(records[4]["visible"])
            self.assertIsNone(records[7]["visibility_index"])
            self.assertEqual(data["rocminfo_gfx_targets"], ["gfx1100", "gfx942"])

    def test_single_unindexed_target_is_not_copied_to_every_device(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "rocminfo.log").write_text("Name: gfx942\n")
            (root / "rocm_smi_showallinfo.log").write_text("Card series: MI300X\n0000:01:00.0\n")
            data = self.build(root, devices="4,5,6,7")
            self.assertEqual(data["physical_devices"], [])
            self.assertEqual(data["requested_gpu_count"], 4)
            self.assertIsNone(data["node_gpu_count"])

    def test_conflicting_indexed_evidence_stays_unknown(self):
        records = summary.parse_indexed_smi_records("GPU[4]: ASIC: gfx942\nGPU[4]: ASIC: gfx1100\n"
                                                    "GPU[4]: PCI Bus: 0000:01:00.0\nGPU[4]: PCI Bus: 0000:02:00.0\n")
        self.assertIsNone(records[4]["target_arch"])
        self.assertIsNone(records[4]["bdf"])
        self.assertIn("target_arch", records[4]["conflicting_fields"])

    def test_masks_are_recorded_but_never_select_devices(self):
        with tempfile.TemporaryDirectory() as directory:
            data = summary.build_summary(Path(directory), environment={"HIP_VISIBLE_DEVICES": "4,5"})
            self.assertEqual(data["requested_devices"], [])
            self.assertEqual(data["physical_devices"], [])
            self.assertEqual(data["runtime_environment"]["HIP_VISIBLE_DEVICES"], "4,5")

    def test_secret_allowlist(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "env.log").write_text("AWS_SECRET_ACCESS_KEY=top-secret\nGITHUB_TOKEN=top-secret\n"
                                         "HIP_PATH=/sdk with spaces\nLD_LIBRARY_PATH=https://user:top-secret@example/a\n")
            data = self.build(root)
            self.assertNotIn("top-secret", json.dumps(data))
            self.assertEqual(data["runtime_environment"]["HIP_PATH"], "/sdk with spaces")
            self.assertIn("redacted", data["runtime_environment"]["LD_LIBRARY_PATH"])

    def test_invalid_requested_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            for value in ("all", "0,0", "-1", "a,2"):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    self.build(Path(directory), devices=value)

    def test_tool_readiness_requires_captured_logs(self):
        self.assertFalse(summary.captured_tool_ready("", fallback_commands=["python"]))
        self.assertFalse(summary.captured_tool_ready("dry-run: rocprofv3 --version", "agents"))
        self.assertFalse(summary.captured_tool_ready("command not found: rocprofv3", "agents"))
        self.assertFalse(summary.captured_tool_ready("version", ""))
        self.assertTrue(summary.captured_tool_ready("version", "agents", "pmcs"))


if __name__ == "__main__":
    unittest.main()
