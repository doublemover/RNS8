"""Recorded heaptrack stack/module joins; no current-file or debug-symbol inference."""

from __future__ import annotations

from bisect import bisect_right
from collections import Counter, defaultdict

MAX_FRAME_BINDINGS = 262_144
MAX_SEGMENT_VISITS = 1_000_000
TRACE_PRODUCER = "https://github.com/KDE/heaptrack/blob/v1.5.0/src/track/tracetree.h"


def recorded_stacks(
    allocations: list[dict], traces: list[dict], modules: list[dict], executable: str
):
    """Normalize shared stack nodes by the allocation's actual module epoch.

    The native trace tree survives module resets, so a trace ID can be reused
    after dlclose/dlopen. Its first-definition epoch is not an allocation epoch.
    Parent links retain the producer's leaf-to-caller direction without copying
    whole stacks for every allocation or inventing physical unwind coverage.
    """
    segments = defaultdict(list)
    for module_id, module in enumerate(modules, 1):
        for offset, size in zip(module["segments"][::2], module["segments"][1::2]):
            if size:
                start = module["base"] + offset
                segments[module["generation"]].append((start, start + size, module_id))
    indexes = {}
    for epoch, rows in segments.items():
        rows.sort()
        maximum = 0
        prefix = []
        for _, end, _ in rows:
            maximum = max(maximum, end)
            prefix.append(maximum)
        indexes[epoch] = ([row[0] for row in rows], prefix, rows)

    visits = 0

    def module_at(epoch, address):
        nonlocal visits
        starts, prefix, rows = indexes.get(epoch, ([], [], []))
        position = bisect_right(starts, address) - 1
        owners = set()
        while position >= 0 and prefix[position] > address:
            visits += 1
            if visits > MAX_SEGMENT_VISITS:
                raise ValueError("native heaptrack module join exceeds segment-visit budget")
            start, end, module_id = rows[position]
            if start <= address < end:
                owners.add(module_id)
            position -= 1
        if len(owners) != 1:
            return dict(
                state="unresolved",
                reason="overlapping_recorded_modules" if owners else "recorded_module_missing",
                module_record_id=None,
                module_path=None,
                module_virtual_address=None,
            )
        module_id = owners.pop()
        module = modules[module_id - 1]
        return dict(
            state="unique_recorded_module",
            reason=None,
            module_record_id=module_id,
            module_path=executable if module["name"] == "x" else module["name"],
            module_virtual_address=address - module["base"],
        )

    roots = {}
    for allocation in allocations:
        key = (allocation["allocated_module_generation"], allocation["trace"])
        root = roots.setdefault(
            key, dict(allocations=0, requested_bytes=0, freed_bytes=0, retained_bytes=0)
        )
        root["allocations"] += 1
        root["requested_bytes"] += allocation["bytes"]
        root["freed_bytes" if allocation["free_record"] is not None else "retained_bytes"] += (
            allocation["bytes"]
        )

    def frame_id(epoch, trace_id):
        return f"{epoch}:{trace_id}" if trace_id else None

    frames = {}
    for epoch, root_id in sorted(roots):
        trace_id = root_id
        while trace_id and (epoch, trace_id) not in frames:
            if len(frames) >= MAX_FRAME_BINDINGS:
                raise ValueError("native heaptrack stack join exceeds frame-binding budget")
            node = traces[trace_id - 1]
            frames[(epoch, trace_id)] = dict(
                frame_id=frame_id(epoch, trace_id),
                trace_node_id=trace_id,
                allocation_module_generation=epoch,
                trace_definition_module_generation=node["module_generation"],
                recorded_address=node["address"],
                parent_frame_id=frame_id(epoch, node["parent"]),
                recorded_module=module_at(epoch, node["address"]),
            )
            trace_id = node["parent"]
    return dict(
        schema="rns8_recorded_heaptrack_stacks_v1",
        trace_producer=TRACE_PRODUCER,
        frames=[frames[key] for key in sorted(frames)],
        roots=[
            dict(
                trace_node_id=trace_id,
                allocation_module_generation=epoch,
                frame_id=frame_id(epoch, trace_id),
                **roots[(epoch, trace_id)],
            )
            for epoch, trace_id in sorted(roots)
        ],
        frame_mapping_states=dict(
            sorted(Counter(row["recorded_module"]["state"] for row in frames.values()).items())
        ),
        segment_visits=visits,
        address_policy="native_producer_already_subtracted_one_do_not_adjust_again",
        symbol_attribution="unresolved_no_recorded_build_ID_or_bound_debug_symbol_lookup",
        phase_attribution="unbound_native_trace_has_no_operation_or_thread_markers",
        loaded_module_bytes_verified=False,
        complete_unwind_proven=False,
        production_latency=False,
        promotion_allowed=False,
    )
