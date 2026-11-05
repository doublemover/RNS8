"""Focused mutation controls use a tiny authored catalog; no workload executes."""

from __future__ import annotations

import copy
from dataclasses import replace
import unittest
from unittest.mock import patch

from benchmark_sweep_lib import commands
from benchmark_sweep_lib.config import ScenarioItem, SweepCase
from test_benchmark_sweep_support import scenario_command_args
from test_sweep_contract_support import (
    assert_scenario_commands,
    assert_unique_destinations,
    assert_visibility,
)


class SweepCommandContracts(unittest.TestCase):
    def entries(self, *, sharded=False):
        args = scenario_command_args()
        args.scenario = ["authored-contract"]
        args.bench_for = ["hip-direct=explicit benchmark with spaces"]
        args.hip_visible_devices = "7"
        args.rocr_visible_devices = None
        args.gpu_device_ordinal = None
        args.gpu_shards = "0,1,2,3" if sharded else None
        item = ScenarioItem(
            family="authored-contract",
            name="reuse-b",
            semantics="bounded-i64",
            case=SweepCase("small", 16, 16, 16),
            evidence_scope="authored_test_only",
            output_domain="host_export",
            rationale="contract mutation control",
            review_mode_expectation="smoke",
            promotion_eligibility="diagnostic_only",
            backends=("hip-direct",),
            pack_mode="prepacked_reuse_b",
        )
        with patch(
            "benchmark_sweep_lib.commands.scenario_catalog",
            return_value={"authored-contract": [item]},
        ):
            return commands.sweep_command_entries(args)

    def test_real_planner_forwards_masks_unique_outputs_and_reuse(self):
        entries = self.entries()
        self.assertEqual(len(entries), 1)
        assert_visibility(entries, {"HIP_VISIBLE_DEVICES": "7"})
        assert_scenario_commands(entries)
        self.assertEqual(entries[0].command[0], "explicit benchmark with spaces")
        shards = self.entries(sharded=True)
        self.assertEqual(len(shards), 4)
        assert_scenario_commands(shards)
        self.assertEqual({entry.scenario["gpu_shard"] for entry in shards}, {"0", "1", "2", "3"})
        for entry in shards:
            self.assertEqual(
                entry.env,
                {
                    "HIP_VISIBLE_DEVICES": entry.scenario["gpu_shard"],
                    "ROCR_VISIBLE_DEVICES": entry.scenario["gpu_shard"],
                },
            )
            self.assertEqual(entry.output.parent.name, "gpu" + entry.scenario["gpu_shard"])

    def test_removing_visibility_is_detected(self):
        entries = [replace(entry, env={}) for entry in self.entries()]
        with self.assertRaises(AssertionError):
            assert_visibility(entries, {"HIP_VISIBLE_DEVICES": "7"})

    def test_aliasing_every_shard_output_is_detected(self):
        entries = self.entries(sharded=True)
        mutated = [replace(entry, output=entries[0].output) for entry in entries]
        with self.assertRaises(AssertionError):
            assert_unique_destinations(mutated)

    def test_removing_reuse_flags_while_retaining_labels_is_detected(self):
        entries = self.entries()
        mutated = [
            replace(entry, command=[arg for arg in entry.command if arg != "--reuse-packed-b"])
            for entry in entries
        ]
        self.assertEqual(mutated[0].name, entries[0].name)
        with self.assertRaises(AssertionError):
            assert_scenario_commands(mutated)

    def test_bridge_flag_and_declared_backend_substitutions_are_detected(self):
        entry = self.entries()[0]
        for field in ("native_to_rns_bridge", "vector_to_rns_chain"):
            metadata = copy.deepcopy(entry.scenario)
            metadata[field] = True
            with self.assertRaises(AssertionError):
                assert_scenario_commands([replace(entry, scenario=metadata)])
        command = list(entry.command)
        command[command.index("--backend") + 1] = "cpu-reference"
        with self.assertRaises(AssertionError):
            assert_scenario_commands([replace(entry, command=command)])


if __name__ == "__main__":
    unittest.main()
