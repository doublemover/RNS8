"""Disjoint setup accounting for a fresh caller-input transaction."""

from __future__ import annotations

import math
from typing import Any

SCHEMA = "rns8_first_call_setup_inventory_v1"
BOUNDARY = "caller_buffers_ready_after_context_creation"
EXCLUDED = (
    "context_creation",
    "caller_buffer_allocation",
    "input_generation",
    "independent_oracle",
)
BASE_COMPONENTS = {
    "planning_workspace": "raw_timings_us.planning",
    "matrix_allocation": "raw_timings_us.matrix_alloc",
    "completed_call": "raw_timings_us.end_to_end",
}


def inventory_errors(capture: dict[str, Any]) -> list[str]:
    inventory = capture.get("first_call_setup_inventory")
    if inventory is None:
        return []
    if not isinstance(inventory, dict):
        return ["first_call_setup_inventory must be an object"]
    expected = dict(
        BASE_COMPONENTS,
        prepack="prepack_setup_us" if capture.get("reuse_packed_inputs") is True else None,
    )
    if (
        inventory.get("schema") != SCHEMA
        or inventory.get("boundary") != BOUNDARY
        or type(inventory.get("supported")) is not bool
        or inventory.get("components") != expected
        or inventory.get("excluded") != list(EXCLUDED)
    ):
        return [
            "first_call_setup_inventory has an unknown or incomplete boundary/component contract"
        ]
    return []


def first_call_setup(capture: dict[str, Any]) -> dict[str, Any]:
    """Sum parent phases once; never sum diagnostic events or prepack subphases."""
    timing = capture.get("timing_metadata")
    raw = capture.get("raw_timings_us")
    if (
        type(capture.get("warmups")) is not int
        or capture["warmups"] != 0
        or type(capture.get("repeats")) is not int
        or capture["repeats"] != 1
        or not isinstance(timing, dict)
        or timing.get("host_instrumentation_state") != "hooks_disabled"
    ):
        raise ValueError(
            "first-call evidence requires explicit hooks-disabled zero-warmup one-call capture"
        )
    errors = inventory_errors(capture)
    inventory = capture.get("first_call_setup_inventory")
    if errors or not isinstance(inventory, dict) or inventory.get("supported") is not True:
        raise ValueError(
            "complete supported first-call setup inventory required: " + "; ".join(errors)
        )
    values = {}
    for role, reference in inventory["components"].items():
        if reference is None:
            values[role] = 0
            continue
        if reference.startswith("raw_timings_us."):
            samples = raw.get(reference.split(".", 1)[1]) if isinstance(raw, dict) else None
            if not isinstance(samples, list) or len(samples) != 1:
                raise ValueError("first-call setup requires one disjoint sample for " + role)
            value = samples[0]
        else:
            value = capture.get(reference)
        try:
            finite = type(value) in (int, float) and math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite or value < 0 or (role == "completed_call" and value == 0):
            raise ValueError(
                "first-call setup requires a finite nonnegative observation for " + role
            )
        values[role] = value
    narrow = values["planning_workspace"] + values["matrix_allocation"] + values["completed_call"]
    total = narrow + values["prepack"]
    try:
        finite_total = math.isfinite(narrow) and math.isfinite(total)
    except OverflowError:
        finite_total = False
    if not finite_total:
        raise ValueError("first-call setup total is not finite")
    return dict(
        schema=SCHEMA,
        scope=BOUNDARY,
        components_us=values,
        planning_us=values["planning_workspace"],
        matrix_alloc_us=values["matrix_allocation"],
        end_to_end_us=values["completed_call"],
        prepack_setup_us=values["prepack"],
        planning_allocation_and_prepared_input_invocation_us=narrow,
        setup_plus_first_completed_call_us=total,
        prepack_included=capture.get("reuse_packed_inputs") is True,
        excluded=list(EXCLUDED),
        context_creation_and_input_generation_included=False,
        independent_oracle_included=False,
    )


def validate_first_call_arguments(arguments: list[str]) -> None:
    """Reject known incomplete owners before scheduling any fresh child process."""
    forbidden = (
        "--hip-graph-replay",
        "--streaming-overlap",
        "--oneshot",
        "--one-shot",
        "--transient-uniform-small-inputs",
        "--transient-uniform-small-i8-inputs",
        "--native-to-rns-bridge",
        "--force-native-to-rns-bridge",
        "--vector-to-rns-chain",
        "--vector-native-to-rns-chain",
        "--vector-to-rns-chain-host-repack-control",
        "--vector-native-to-rns-chain-host-repack-control",
        "--vector-alu-baseline",
        "--sparse-a-4-to-2",
        "--residue-channel-fusion",
        "--resident-redesign-candidate",
        "--resident-lifetime",
        "--workspace-arena",
        "--sparse-a-4to2",
        "--sparse-a-4-to-2-dense-baseline",
        "--sparse-a-4to2-dense-baseline",
    )
    if any(arg.split("=", 1)[0] in forbidden for arg in arguments):
        raise ValueError(
            "first-call boundary lacks a complete inventory for this specialized owner"
        )
    for option in ("--residue-chain-length", "--host-api-batch-size", "--grouped-dispatch"):
        for index, arg in enumerate(arguments):
            if arg == option and (index + 1 == len(arguments) or arguments[index + 1] != "1"):
                raise ValueError("first-call boundary supports one ordinary product")
            if arg.startswith(option + "=") and arg != option + "=1":
                raise ValueError("first-call boundary supports one ordinary product")


def review_first_call(capture: dict[str, Any]) -> dict[str, Any] | None:
    if capture.get("warmups") != 0 or capture.get("repeats") != 1:
        return None
    try:
        return dict(available=True, measurements=first_call_setup(capture))
    except (ValueError, TypeError) as error:
        return dict(available=False, blocker=str(error))
