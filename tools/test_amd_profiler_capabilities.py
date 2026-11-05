"""Authored preflight controls; synthetic tools are never executed."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from amd_profiler_capabilities import (
    probe_plan,
    preflight,
    support_matrix,
    _version,
    availability_inventory,
)
from compiled_binary_identity import canonical_digest


class ProfilerCapabilities(unittest.TestCase):
    def test_target_feature_policy_does_not_extrapolate(self):
        rdna = support_matrix("Linux", "gfx1100")
        self.assertEqual(rdna["timeline"].official_support, "supported")
        self.assertEqual(rdna["att"].official_support, "trace_only")
        for name in (
            "compute_profile",
            "att_counters",
            "pc_host_trap",
            "pc_stochastic",
        ):
            self.assertEqual(rdna[name].official_support, "unsupported")
        self.assertEqual(
            support_matrix("Linux", "gfx90a")["pc_stochastic"].official_support,
            "unsupported",
        )
        self.assertEqual(
            support_matrix("Linux", "gfx942")["pc_stochastic"].official_support,
            "supported",
        )
        self.assertEqual(
            support_matrix("Windows", "gfx950")["timeline"].official_support,
            "unsupported",
        )
        self.assertTrue(all(value.capture == "not_run" for value in rdna.values()))

    def test_missing_tools_are_not_advertised_features(self):
        plan = probe_plan("Linux", "gfx942", resolver=lambda _: None)
        self.assertEqual(plan["commands"], [])
        self.assertFalse(any(value["located"] for value in plan["tools"].values()))
        self.assertEqual(plan["features"]["counters"]["advertised"], "unknown")

    def test_explicit_agent_namespace_and_bounded_counter_set(self):
        plan = probe_plan(
            "Linux",
            "gfx942",
            resolver=lambda name: "/authored/" + name,
            agent=7,
            counters=("SQ_WAVES", "GRBM_GUI_ACTIVE"),
        )
        command = next(item for item in plan["commands"] if item["name"] == "pmc-check")
        self.assertEqual(
            command["argv"][1:], ["pmc-check", "-d", "7", "SQ_WAVES", "GRBM_GUI_ACTIVE"]
        )
        self.assertIn("not_HIP_ordinal", plan["agent_namespace"])
        for counters in (
            ("all counters",),
            ("SQ_WAVES",) * 2,
            tuple("C" + str(i) for i in range(9)),
        ):
            with self.assertRaises(ValueError):
                probe_plan("Linux", "gfx942", counters=counters, agent=0)
        with self.assertRaises(ValueError):
            probe_plan("Linux", "gfx942", counters=("SQ_WAVES",))

    def test_windows_plan_never_contains_a_gui_or_service_start(self):
        plan = probe_plan(
            "Windows", "gfx1100", resolver=lambda name: "/authored/" + name + ".exe"
        )
        self.assertEqual(
            [row["argv"][1:] for row in plan["commands"]],
            [["--version"], ["--help=profiling"]],
        )
        self.assertTrue(
            all(row["tool"] == "RadeonDeveloperPanelCLI" for row in plan["commands"])
        )

    def test_versions_do_not_parse_unrelated_numbers(self):
        self.assertEqual(
            _version("ROCm rocprofiler-sdk version: 1.4.1", "rocprofv3"), "1.4.1"
        )
        self.assertEqual(
            _version("Radeon Developer Panel CLI v3.5.1", "RadeonDeveloperPanelCLI"),
            "3.5.1",
        )
        self.assertIsNone(_version("driver 26.10.2 date 2026.11.5", "rocprofv3"))

    def test_vendor_agent_text_preserves_its_namespace(self):
        agents = availability_inventory(
            "GPU:7\n name : gfx942\n logical_node_id : 9\n uuid : GPU-a\n"
        )
        self.assertEqual(agents[0]["selector"], 7)
        self.assertEqual(agents[0]["properties"]["logical_node_id"], "9")
        self.assertIn("not_HIP_ordinal", agents[0]["namespace"])
        for malformed in (
            "",
            "GPU:7\nname:gfx942\nname:gfx942",
            "GPU:7\nname:gfx942\nGPU:7\nname:gfx942",
            "GPU:7\nname:unknown",
        ):
            with self.assertRaises(ValueError):
                availability_inventory(malformed)

    def test_read_only_runner_keeps_raw_failures_unknown_and_rejects_resealed_commands(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tool = root / "rocprofv3"
            tool.write_bytes(b"AUTHORED TOOL, NOT EXECUTABLE")
            plan = probe_plan(
                "Linux",
                "gfx942",
                resolver=lambda name: str(tool) if name == "rocprofv3" else None,
            )
            calls = []

            def runner(command, **kwargs):
                calls.append(command)
                return dict(
                    returncode=1,
                    timed_out=False,
                    process_id=12,
                    stdout="",
                    stderr="permission denied",
                    elapsed_seconds=0.01,
                )

            with patch(
                "amd_profiler_capabilities.platform.system", return_value="Linux"
            ):
                result = preflight(
                    plan,
                    root / "probe",
                    environment={},
                    runner=runner,
                    minimum_free_bytes=1,
                )
                self.assertEqual(len(calls), 2)
                self.assertTrue(
                    all(not row["completed"] for row in result["observations"])
                )
                self.assertEqual(
                    result["features"]["counters"]["permission"], "unknown"
                )
                self.assertEqual(
                    result["tools"]["rocprofv3"]["version_state"], "unknown"
                )
                altered = copy.deepcopy(plan)
                altered["commands"][0]["argv"] = [
                    "unrelated",
                    "--enable-developer-mode",
                ]
                altered["identity_sha256"] = canonical_digest(
                    {
                        key: value
                        for key, value in altered.items()
                        if key != "identity_sha256"
                    }
                )
                with self.assertRaises(ValueError):
                    preflight(altered, root / "rejected", environment={}, runner=runner)
                self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
