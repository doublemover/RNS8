"""Observed native launch/graph relationships; missing causality stays explicit."""

from __future__ import annotations

from collections import defaultdict

from interval_observations import interval_summary


def build(timeline, selected, roots, markers):
    native = {(row["kind"], row["id"]): row for row in timeline["records"]}
    joined = {(row.get("kind", "kernel_dispatch"), row["native_table_id"]): row for row in selected}
    launches, graphs, operations = [], defaultdict(list), defaultdict(list)
    for key, row in joined.items():
        record = native[key]
        launches.append(
            dict(
                source=dict(kind="host_region", id=row["native_api_id"], endpoint="start"),
                target=dict(kind=key[0], id=key[1], endpoint="start"),
                relation="exact_SDK_internal_launch_correlation",
                elapsed_ns=row["start_ns"] - row["api_start_ns"],
                elapsed_cause="unresolved_submission_driver_queue_or_other_work",
                API_return_to_device_start_edge=False,
            )
        )
        operations[row["host_range_id"]].append((key, record, row))
        if key[0] == "graph_launch" and record["graph_exec_id"]:
            graphs[(record["agent_id"], record["graph_exec_id"], row["native_api_id"])].append(
                (key, record)
            )
    memberships, unresolved = [], []
    for key, row in joined.items():
        record = native[key]
        if key[0] != "kernel_dispatch" or not record["graph_exec_id"]:
            continue
        candidates = graphs.get(
            (record["agent_id"], record["graph_exec_id"], row["native_api_id"]), []
        )
        if len(candidates) != 1:
            unresolved.append(
                dict(
                    kind=key[0],
                    id=key[1],
                    reason="missing_or_ambiguous_exact_graph_launch_membership",
                )
            )
            continue
        graph_key, graph = candidates[0]
        if not graph["start_ns"] <= record["start_ns"] <= record["end_ns"] <= graph["end_ns"]:
            unresolved.append(
                dict(kind=key[0], id=key[1], reason="graph_member_escapes_native_graph_interval")
            )
            continue
        memberships.append(
            dict(
                graph_native_id=graph_key[1],
                kernel_native_id=key[1],
                graph_exec_id=record["graph_exec_id"],
                graph_node_id=record["graph_node_id"],
                native_api_id=row["native_api_id"],
                graph_declared_dispatch_count=graph["kernel_dispatch_count"],
                relation="same_native_graph_instance_and_exact_launch_API_not_dependency_order",
                complete_graph_coverage=False,
            )
        )

    event_records = defaultdict(list)
    waits = []
    for key, record in native.items():
        if key[0] != "hip_event":
            continue
        if record["name"] == "HIP_EVENT_RECORD":
            event_records[(record["agent_id"], record["hip_event_handle"])].append(record)
        elif record["name"] == "HIP_EVENT_WAIT":
            waits.append(record)
        else:
            unresolved.append(
                dict(
                    kind="hip_event",
                    id=record["id"],
                    reason="unsupported_native_HIP_event_operation",
                )
            )
    event_candidates = []
    for wait in waits:
        candidates = event_records.get((wait["agent_id"], wait["hip_event_handle"]), [])
        reason = None
        if (
            wait["agent_id"] is None
            or not wait["hip_event_handle"]
            or wait["source_queue_id"] is None
        ):
            reason = "event_handle_agent_or_source_queue_unavailable"
        elif len(candidates) != 1:
            reason = "missing_or_multiple_observed_event_record_generations"
        else:
            record = candidates[0]
            before, after = (
                joined.get(("hip_event", record["id"])),
                joined.get(("hip_event", wait["id"])),
            )
            if record["queue_id"] != wait["source_queue_id"]:
                reason = "event_record_source_queue_conflicts"
            elif record["end_ns"] > wait["end_ns"]:
                reason = "observed_record_completion_after_wait"
            elif before is None or after is None or before["api_end_ns"] > after["api_start_ns"]:
                reason = "record_wait_API_ownership_or_order_unresolved"
            else:
                # The raw schema has no event-generation token or complete
                # create/destroy/loss witness. Handle equality is a candidate.
                event_candidates.append(
                    dict(
                        record_native_id=record["id"],
                        wait_native_id=wait["id"],
                        agent_id=wait["agent_id"],
                        hip_event_handle=wait["hip_event_handle"],
                        source_queue_id=wait["source_queue_id"],
                        destination_queue_id=wait["queue_id"],
                        state="observed_candidate_generation_unverified",
                        usable_as_causal_dependency=False,
                    )
                )
        if reason:
            unresolved.append(dict(kind="hip_event", id=wait["id"], reason=reason))
    summaries = []
    for owner, root in roots.items():
        rows = operations.get(owner, [])
        services = [
            (record["start_ns"], record["end_ns"])
            for key, record, _ in rows
            if key[0] in {"kernel_dispatch", "memory_copy"}
        ]
        apis = {
            row["native_api_id"]: (row["api_start_ns"], row["api_end_ns"]) for _, _, row in rows
        }
        service = interval_summary(services)
        marker = markers[owner]
        wall = marker["end_ns"] - marker["start_ns"]
        summaries.append(
            dict(
                host_range_id=owner,
                operation_id=root["operation_id"],
                iteration_id=root["iteration_id"],
                native_marker_wall_ns=wall,
                selected_kernel_copy_intervals=service,
                selected_API_intervals=interval_summary(list(apis.values())),
                native_window_outside_selected_service_ns=wall - service["interval_union_ns"],
                outside_service_cause="unresolved_host_work_wait_unselected_device_work_or_missing_coverage",
                host_and_device_times_additive=False,
            )
        )
    return dict(
        schema="rns8_native_dependency_report_v1",
        launch_relationships=launches,
        graph_memberships=memberships,
        event_generation_candidates=event_candidates,
        unresolved_relationships=unresolved,
        operations=summaries,
        critical_path=dict(
            state="unresolved",
            longest_path_ns=None,
            blockers=[
                "scheduler_running_runnable_blocked_and_spin_intervals_unavailable",
                "complete_device_dependency_and_loss_coverage_unestablished",
                "event_generation_and_GPU_clock_normalization_unqualified",
            ],
        ),
        conclusion="no_resolved_host_bottleneck_from_this_native_projection",
        next_experiment="collect same-contract process-scoped CPU/scheduler evidence and verify event-generation/clock coverage before attributing uncovered intervals",
        temporal_adjacency_dependency_inferred=False,
        production_latency=False,
        performance_promotion_allowed=False,
    )
