"""Disjoint native marker wall intervals and bounded exact launch-phase attribution."""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
import heapq

from host_range_contract import PHASES
from interval_observations import interval_summary


def exclusive_segments(identities, index, markers, allowed):
    """Partition one thread's scopes; foreign/non-ancestral overlaps stay masked."""
    events = []
    for identity in identities:
        marker = markers[identity]
        if marker["start_ns"] != marker["end_ns"]:
            events.extend(((marker["start_ns"], 1, identity), (marker["end_ns"], 0, identity)))
    active, heap, result, excluded_active = set(), [], [], 0
    prior = None
    for timestamp, action, identity in sorted(events):
        while heap and heap[0][1] not in active:
            heapq.heappop(heap)
        if prior is not None and prior < timestamp and active:
            leaf = heap[0][1]
            chain_count = 1 + sum(parent in active for parent in index.parents[leaf])
            # Siblings may overlap due to incomplete/concurrent observations.
            # A depth tie or any non-ancestral active range stays ambiguous.
            result.append(
                (
                    prior,
                    timestamp,
                    leaf if chain_count == len(active) and not excluded_active else None,
                )
            )
        if action:
            active.add(identity)
            heapq.heappush(heap, (-index.depth[identity], identity))
            excluded_active += identity not in allowed
        else:
            active.remove(identity)
            excluded_active -= identity not in allowed
        prior = timestamp
    return result


def attribute(index, markers, roots, selected):
    owners, groups, excluded = {}, defaultdict(list), []
    for identity, scope in index.records.items():
        owner = next((item for item in (identity, *index.parents[identity]) if item in roots), None)
        if owner is None:
            continue
        root, marker, enclosing = roots[owner], markers[identity], markers[owner]
        reason = None
        if (
            scope["role"] != root["role"]
            or scope["operation_namespace"] != root["operation_namespace"]
            or scope["operation_id"] != root["operation_id"]
            or scope["iteration_id"] != root["iteration_id"]
        ):
            reason = "scope_contract_differs_from_measured_operation"
        elif scope["stop_state"] != 1 or scope["start_thread_id"] != scope["stop_thread_id"]:
            reason = "async_or_cross_thread_scope_not_exclusive_host_work"
        elif (
            not enclosing["start_ns"]
            <= marker["start_ns"]
            <= marker["end_ns"]
            <= enclosing["end_ns"]
        ):
            reason = "phase_outside_completed_measured_native_window"
        if reason:
            excluded.append(
                dict(host_range_id=identity, native_marker_id=marker["id"], reason=reason)
            )
            if (
                scope["stop_state"] == 1
                and scope["start_thread_id"] == scope["stop_thread_id"]
                and enclosing["start_ns"]
                <= marker["start_ns"]
                <= marker["end_ns"]
                <= enclosing["end_ns"]
            ):
                # A foreign synchronous scope masks its interval rather than
                # crediting the enclosing phase with that unclassified work.
                groups[(owner, scope["start_thread_id"])].append(identity)
            continue
        owners[identity] = owner
        groups[(owner, scope["start_thread_id"])].append(identity)
    segments, starts, summaries = {}, {}, []
    for (owner, thread), identities in sorted(groups.items()):
        spans = exclusive_segments(identities, index, markers, owners)
        segments[(owner, thread)] = spans
        starts[(owner, thread)] = [row[0] for row in spans]
        inclusive = defaultdict(list)
        for identity in identities:
            if identity not in owners:
                continue
            marker = markers[identity]
            inclusive[PHASES[index.records[identity]["phase"]]].append(
                (marker["start_ns"], marker["end_ns"])
            )
        exclusive, ambiguous = defaultdict(int), 0
        for first, last, leaf in spans:
            if leaf is None:
                ambiguous += last - first
            else:
                exclusive[PHASES[index.records[leaf]["phase"]]] += last - first
        summaries.append(
            dict(
                host_range_id=owner,
                operation_id=roots[owner]["operation_id"],
                iteration_id=roots[owner]["iteration_id"],
                thread_id=thread,
                inclusive_phases={
                    phase: interval_summary(intervals)
                    for phase, intervals in sorted(inclusive.items())
                },
                exclusive_phase_wall_ns=dict(sorted(exclusive.items())),
                ambiguous_wall_ns=ambiguous,
                observed_scope_union_ns=sum(last - first for first, last, _ in spans),
            )
        )
    attributed = []
    for row in selected:
        owner = row["host_range_id"]
        thread = row["launch_thread_id"]
        key = (owner, thread)
        spans = segments.get(key, [])
        position = bisect_right(starts.get(key, []), row["api_start_ns"]) - 1
        leaf = (
            spans[position][2]
            if position >= 0 and spans[position][0] <= row["api_start_ns"] < spans[position][1]
            else None
        )
        phase = None
        if leaf is not None:
            phase = next(
                (
                    item
                    for item in (leaf, *index.parents[leaf])
                    if owners.get(item) == owner
                    and markers[item]["start_ns"]
                    <= row["api_start_ns"]
                    <= row["api_end_ns"]
                    <= markers[item]["end_ns"]
                    and index.records[item]["start_thread_id"] == thread
                ),
                None,
            )
        attributed.append(
            dict(
                {
                    key: row[key]
                    for key in (
                        "host_range_id",
                        "operation_id",
                        "iteration_id",
                        "native_table_id",
                        "native_api_id",
                        "launch_thread_id",
                        "kind",
                        "dispatch_id",
                        "agent_id",
                        "queue_id",
                    )
                    if key in row
                },
                launch_phase=None if phase is None else PHASES[index.records[phase]["phase"]],
                launch_phase_range_id=phase,
                phase_attribution="unresolved_or_ambiguous"
                if phase is None
                else "exact_launch_API_inside_native_marker_ancestry",
            )
        )
    return dict(
        schema="rns8_native_phase_attribution_v1",
        threads=summaries,
        records=attributed,
        excluded_scopes=excluded,
        clock="native_ROCprofiler_SDK_timestamp_ns_in_one_process",
        measure="observed_marker_wall_nanoseconds_not_CPU_time_or_exclusive_device_service",
        thread_totals_add_to_process_wall=False,
        complete_run_coverage=False,
        production_latency=False,
    )
