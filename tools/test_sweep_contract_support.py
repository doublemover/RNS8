"""Independent assertions on executable sweep contracts, not descriptive labels."""

from __future__ import annotations

from collections.abc import Sequence
from benchmark_sweep_lib.config import SweepCommand

REUSE_FLAGS = {"--reuse-packed-inputs", "--reuse-packed-a", "--reuse-packed-b"}


def option(command: Sequence[str], flag: str) -> str:
    assert command.count(flag) == 1, (flag, command)
    position = command.index(flag)
    assert position + 1 < len(command) and not command[position + 1].startswith("--"), (
        flag,
        command,
    )
    return command[position + 1]


def assert_reuse(command: Sequence[str], mode: str) -> None:
    actual = {flag for flag in REUSE_FLAGS if flag in command}
    expected = {
        "per_repeat_repack": [set()],
        "prepacked_reuse": [{"--reuse-packed-inputs"}, {"--reuse-packed-a", "--reuse-packed-b"}],
        "prepacked_reuse_a": [{"--reuse-packed-a"}],
        "prepacked_reuse_b": [{"--reuse-packed-b"}],
    }[mode]
    assert actual in expected, (mode, actual)
    assert all(command.count(flag) == 1 for flag in actual), command


def assert_visibility(entries: Sequence[SweepCommand], expected: dict[str, str]) -> None:
    assert entries
    assert all(entry.env == expected for entry in entries), [
        (entry.name, entry.env) for entry in entries
    ]


def assert_unique_destinations(entries: Sequence[SweepCommand]) -> None:
    assert entries
    assert len({entry.name for entry in entries}) == len(entries)
    assert len({entry.output for entry in entries}) == len(entries)


def assert_scenario_commands(entries: Sequence[SweepCommand]) -> None:
    assert_unique_destinations(entries)
    for entry in entries:
        scenario, command = entry.scenario, entry.command
        assert scenario is not None
        for dimension in ("m", "n", "k"):
            assert option(command, "--" + dimension) == str(scenario["shape"][dimension])
        assert option(command, "--semantics") == scenario["semantics"]
        cli_backend = {
            "hip-vector-alu-int64": "hip-vector-alu-int64-runtime",
        }.get(scenario["backend"], scenario["backend"])
        assert option(command, "--backend") == cli_backend
        assert_reuse(command, scenario["pack_mode"])
        assert ("--native-to-rns-bridge" in command) == scenario["native_to_rns_bridge"]
        host_control = scenario["vector_to_rns_chain_host_repack_control"]
        assert ("--vector-to-rns-chain-host-repack-control" in command) == host_control
        assert ("--vector-to-rns-chain" in command) == (
            scenario["vector_to_rns_chain"] and not host_control
        )
        for field, flag in (
            ("transient_uniform_small_inputs", "--transient-uniform-small-inputs"),
            ("residue_channel_fusion", "--residue-channel-fusion"),
            ("oneshot", "--oneshot"),
            ("hip_graph_replay", "--hip-graph-replay"),
        ):
            assert (flag in command) == scenario[field]
        for field, flag, default in (
            ("residue_chain_length", "--residue-chain-length", 1),
            ("output_ld_padding", "--output-ld-padding", 0),
            ("host_api_batch_size", "--host-api-batch-size", 1),
        ):
            if scenario[field] > default:
                assert option(command, flag) == str(scenario[field])
            else:
                assert flag not in command
        for field, flag in (
            ("modulus", "--modulus"),
            ("exact_wide_limb_count", "--exact-wide-limbs"),
        ):
            if scenario[field] is not None:
                assert option(command, flag) == str(scenario[field])
        if scenario.get("gpu_shard") is not None:
            shard = scenario["gpu_shard"]
            assert entry.env is not None
            assert entry.env["HIP_VISIBLE_DEVICES"] == shard
            assert entry.env["ROCR_VISIBLE_DEVICES"] == shard
            assert scenario["visibility_environment"] == entry.env
