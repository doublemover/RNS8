"""Observed ROCpd allocation generations and uncertain requested-payload lifetimes."""

from __future__ import annotations

from collections import defaultdict

from interval_observations import weighted_peak


def unsigned(value, label):
    if type(value) is not int or not 0 <= value < 1 << 64:
        raise ValueError("native allocation uint64 required: " + label)
    return value


def summarize(timeline: dict) -> dict:
    if (
        not isinstance(timeline, dict)
        or timeline.get("schema") != "rns8_native_rocpd_timeline_v1"
        or timeline.get("clock") != "native_ROCprofiler_SDK_timestamp_ns_in_one_process"
    ):
        raise ValueError("actual typed native allocation timeline required")
    pid = unsigned(timeline.get("process_id"), "PID")
    if not pid:
        raise ValueError("native allocation process PID is absent")
    window, rows, agents = (
        timeline.get(key) for key in ("process_interval_ns", "records", "agents")
    )
    if (
        not isinstance(window, list)
        or len(window) != 2
        or not isinstance(rows, list)
        or len(rows) > 100000
        or not isinstance(agents, list)
    ):
        raise ValueError("bounded native allocation process/record/agent inventory required")
    begin, end = (unsigned(value, "process clock") for value in window)
    if begin >= end:
        raise ValueError("completed allocation process interval required")
    agent_index = {}
    for agent in agents:
        if not isinstance(agent, dict):
            raise ValueError("native allocation agent must be an object")
        key = unsigned(agent.get("id"), "agent ID")
        if key in agent_index or agent.get("type") not in ("CPU", "GPU", "NIC"):
            raise ValueError("native allocation agent identity/type is ambiguous")
        agent_index[key] = agent
    groups, unresolved, seen = defaultdict(list), [], set()
    allocation_calls = free_calls = other_calls = requested = 0
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("native allocation timeline record must be an object")
        if row.get("kind") != "memory_lifecycle":
            continue
        identity = unsigned(row.get("id"), "record ID")
        if identity in seen:
            raise ValueError("native allocation record identity is duplicated")
        seen.add(identity)
        first, last, size = (unsigned(row.get(key), key) for key in ("start_ns", "end_ns", "size"))
        if not begin <= first <= last <= end:
            raise ValueError("native allocation record escapes its process interval")
        kind, level = row.get("type"), row.get("level")
        if kind not in ("ALLOC", "FREE", "REALLOC", "RECLAIM") or level not in (
            "REAL",
            "VIRTUAL",
            "SCRATCH",
        ):
            raise ValueError("unknown native allocation operation/level")
        allocation_calls += kind == "ALLOC"
        free_calls += kind == "FREE"
        other_calls += kind not in ("ALLOC", "FREE")
        if kind == "ALLOC":
            requested += size
        address, agent = row.get("address"), row.get("agent_id")
        if address is not None:
            unsigned(address, "address")
        if agent is not None:
            unsigned(agent, "allocation agent")
        if address is None or agent is None or address == 0:
            unresolved.append(
                dict(record_id=identity, reason="missing_address_or_agent_generation_identity")
            )
            continue
        if agent not in agent_index:
            raise ValueError("native allocation belongs to an unknown agent")
        groups[(agent, level, address)].append(row)
    lifetimes, domains = [], {}
    for (agent, level, address), entries in sorted(groups.items()):
        domain = domains.setdefault(
            (agent, level),
            dict(
                agent_id=agent,
                agent_type=agent_index[agent]["type"],
                level=level,
                requested_payload_bytes=0,
                matched_free_bytes=0,
                observed_retained_bytes=0,
                resolved_generation_ids=[],
                unresolved_record_ids=[],
            ),
        )
        ordered = sorted(entries, key=lambda row: (row["start_ns"], row["end_ns"], row["id"]))
        domain["requested_payload_bytes"] += sum(
            row["size"] for row in entries if row["type"] == "ALLOC"
        )
        clusters = []
        for row in ordered:
            if not clusters or row["start_ns"] > clusters[-1][1]:
                clusters.append([row["start_ns"], row["end_ns"], [row]])
            else:
                clusters[-1][1] = max(clusters[-1][1], row["end_ns"])
                clusters[-1][2].append(row)
        live, generation = None, 0
        for _, _, cluster in clusters:
            if len(cluster) > 1:
                if live is not None:
                    live["state"] = "interrupted_by_ambiguous_transition"
                    live = None
                for row in cluster:
                    unresolved.append(
                        dict(
                            record_id=row["id"], reason="overlapping_same_pointer_effect_intervals"
                        )
                    )
                    domain["unresolved_record_ids"].append(row["id"])
                continue
            (row,) = cluster
            if row["type"] == "ALLOC":
                if live is not None:
                    live["state"] = "free_not_observed_before_pointer_reuse"
                    unresolved.append(
                        dict(
                            record_id=row["id"],
                            reason="live_pointer_reallocated_without_observed_free",
                        )
                    )
                generation += 1
                live = dict(
                    generation_id=row["id"],
                    resolved_pointer_ordinal=generation,
                    address=address,
                    agent_id=agent,
                    level=level,
                    bytes=row["size"],
                    allocation_record_id=row["id"],
                    allocation_interval_ns=[row["start_ns"], row["end_ns"]],
                    free_record_id=None,
                    free_interval_ns=None,
                    state="retained_under_observed_records",
                )
                lifetimes.append(live)
            elif row["type"] == "FREE":
                if live is None:
                    unresolved.append(
                        dict(record_id=row["id"], reason="free_without_observed_live_generation")
                    )
                    domain["unresolved_record_ids"].append(row["id"])
                    continue
                if row["size"] not in (0, live["bytes"]):
                    live["state"] = "free_size_conflicts_with_allocation"
                    unresolved.append(
                        dict(record_id=row["id"], reason="free_size_conflicts_with_allocation")
                    )
                    live = None
                    continue
                live.update(
                    free_record_id=row["id"],
                    free_interval_ns=[row["start_ns"], row["end_ns"]],
                    state="matched_observed_free",
                )
                domain["matched_free_bytes"] += live["bytes"]
                live = None
            else:
                if live is not None:
                    live["state"] = "interrupted_by_unresolved_realloc_or_reclaim"
                    live = None
                unresolved.append(
                    dict(
                        record_id=row["id"],
                        reason="realloc_or_reclaim_has_no_complete_generation_relation",
                    )
                )
                domain["unresolved_record_ids"].append(row["id"])
    by_domain = defaultdict(list)
    for life in lifetimes:
        if life["state"] not in ("matched_observed_free", "retained_under_observed_records"):
            life["lifetime_bounds_ns"] = None
            continue
        first, last = life["allocation_interval_ns"], life["free_interval_ns"]
        life["lifetime_bounds_ns"] = (
            dict(minimum=last[0] - first[1], maximum=last[1] - first[0])
            if last is not None
            else None
        )
        by_domain[(life["agent_id"], life["level"])].append(life)
    for key, domain in domains.items():
        lives = by_domain[key]
        certain, possible = [], []
        for life in lives:
            first, last, size = (
                life["allocation_interval_ns"],
                life["free_interval_ns"],
                life["bytes"],
            )
            certain.append((first[1], last[0] if last else end, size))
            possible.append((first[0], last[1] if last else end, size))
            domain["resolved_generation_ids"].append(life["generation_id"])
            if last is None:
                domain["observed_retained_bytes"] += size
        domain["resolved_payload_peak_minimum_bytes"] = max(
            weighted_peak(certain), max((life["bytes"] for life in lives), default=0)
        )
        domain["resolved_payload_peak_maximum_bytes"] = weighted_peak(
            possible, include_endpoints=True
        )
        domain["pinned_host_classification"] = "unresolved_not_inferred_from_agent_or_REAL_level"
        domain["whole_domain_peak_bytes"] = None
    return dict(
        schema="rns8_native_allocation_lifetimes_v1",
        process_id=pid,
        process_guid=timeline.get("process_guid"),
        native_node_id=timeline.get("native_node_id"),
        clock=timeline["clock"],
        allocation_calls=allocation_calls,
        free_calls=free_calls,
        realloc_or_reclaim_calls=other_calls,
        requested_payload_bytes=requested,
        unattributed_requested_payload_bytes=requested
        - sum(row["requested_payload_bytes"] for row in domains.values()),
        generations=lifetimes,
        domains=list(domains.values()),
        unresolved_records=unresolved,
        bounds_scope="resolved_observed_record_sequence_not_complete_interception_or_physical_commit",
        peak_assumption="matched_and_retained_resolved_events_only_unobserved_or_ambiguous_events_may_change_actual_peak",
        sampled_RSS_included=False,
        device_and_pinned_domains_combined=False,
        complete_generation_history_proven=False,
        loss_status=timeline.get("loss_status"),
        production_latency=False,
        promotion_allowed=False,
    )
