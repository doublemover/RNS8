"""Conditional native address/module joins and normalized local PDB source receipts."""

from __future__ import annotations

from bisect import bisect_right
from collections import Counter
import ntpath
import uuid

from etl_process_contract import integer


def module_points(cpu, images, modules):
    if cpu.get("schema") != "rns8_etl_cpu_observations_v1":
        raise ValueError("native CPU observations required")
    return _module_points(
        cpu["samples"],
        cpu.get("selected_pid"),
        images,
        modules,
        record_field="record_index",
        result_field="native_sample_record",
        schema="rns8_etl_CPU_module_candidates_v1",
    )


def address_module_candidates(addresses, pid, images, modules):
    return _module_points(
        addresses,
        pid,
        images,
        modules,
        record_field="source_address_index",
        result_field="source_address_index",
        schema="rns8_etl_address_module_candidates_v1",
    )


def frame_address_catalog(stacks, images):
    if (
        stacks.get("schema") != "rns8_etl_native_stack_observations_v1"
        or images.get("schema") != "rns8_etl_image_catalog_v1"
        or type(stacks.get("selected_pid")) is not int
        or type(images.get("selected_pid")) is not int
        or stacks["selected_pid"] != images.get("selected_pid")
    ):
        raise ValueError("same-process native stack/image observations required")
    # Every predicate in the module join is constant between these boundaries.
    # Repeated frames can share lookup inside one image-map epoch; an unload or
    # reuse boundary always starts a distinct candidate identity.
    boundaries = sorted(
        {
            row[key]["time_filetime_100ns"]
            for row in images["generations"]
            for key in ("load", "unload")
            if row[key] is not None
        }
    )
    native_fragments = {r["record_index"]: r for r in stacks["fragments"]}
    if len(native_fragments) != len(stacks["fragments"]) or len(native_fragments) > 8192:
        raise ValueError("bounded unique native stack fragments required")
    addresses, fragments, indices, seen = [], [], {}, set()
    total = 0
    for target in sorted(stacks["targets"], key=lambda r: r["source_target_id"]):
        stamp = integer(
            target.get("time_filetime_100ns"), "original stack event time", maximum=2**63 - 1
        )
        epoch = bisect_right(boundaries, stamp)
        for identity in target["stack_records"]:
            if identity in seen:
                raise ValueError("native stack fragment assigned to multiple event targets")
            seen.add(identity)
            native = native_fragments[identity]
            row = dict(
                stack_record=identity,
                source_target_id=target["source_target_id"],
                source_address_indices=[],
            )
            for ip in native["frames"]:
                integer(ip, "native recorded frame address")
                total += 1
                if total > 262144:
                    raise ValueError("native frame address budget exhausted")
                key = ip, epoch
                if key not in indices:
                    indices[key] = len(addresses) + 1
                    addresses.append(
                        dict(
                            source_address_index=indices[key],
                            instruction_pointer=ip,
                            time_filetime_100ns=stamp,
                            image_map_epoch=epoch,
                        )
                    )
                row["source_address_indices"].append(indices[key])
            fragments.append(row)
    return dict(
        schema="rns8_etl_native_frame_addresses_v1",
        selected_pid=stacks["selected_pid"],
        addresses=addresses,
        fragments=fragments,
        targets=stacks["targets"],
        native_bound_frame_observations=total,
        unique_address_epoch_candidates=len(addresses),
        address_policy="native_recorded_IP_no_return_address_adjustment",
        epoch_policy="all_observed_image_generation_load_unload_boundaries",
        unresolved_native_stack_fragments=stacks["unresolved"],
        complete_callchain_proven=False,
        promotion_allowed=False,
    )


def _module_points(source_points, pid, images, modules, *, record_field, result_field, schema):
    if (
        images.get("schema") != "rns8_etl_image_catalog_v1"
        or type(pid) is not int
        or type(images.get("selected_pid")) is not int
        or pid != images["selected_pid"]
        or not isinstance(source_points, list)
        or len(source_points) > 262144
        or len(modules) > 32
    ):
        raise ValueError("same-process native CPU/image inventories required")
    pairs = {}
    for index, module in enumerate(modules):
        name = module["recorded_image_path"]
        if type(name) is not str or not name or len(name) > 4096 or any(ord(c) < 32 for c in name):
            raise ValueError("bounded explicit recorded image path required")
        name = ntpath.normcase(ntpath.normpath(name))
        if name in pairs:
            raise ValueError("ambiguous local symbols for one recorded image path")
        pairs[name] = index
    # Index spatial spans; prefix maxima let ordinary disjoint modules stop the
    # backward search immediately. A separate visit budget bounds hostile overlap.
    generations = sorted(images["generations"], key=lambda row: row["image_base"])
    bases, prefix = [], []
    for row in generations:
        bases.append(row["image_base"])
        end = row["image_base"] + row["load"]["image_size"]
        prefix.append(max(end, prefix[-1] if prefix else 0))
    points, unresolved, visits, seen = [], [], 0, set()
    for point in source_points:
        ordinal = integer(
            point.get(record_field), "source point identity", minimum=1, maximum=1000000
        )
        if ordinal in seen:
            raise ValueError("duplicate source address identity")
        seen.add(ordinal)
        ip = integer(point.get("instruction_pointer"), "native sample IP")
        stamp = integer(point.get("time_filetime_100ns"), "native sample time")
        position, matching = bisect_right(bases, ip) - 1, []
        while position >= 0 and prefix[position] > ip:
            visits += 1
            if visits > 1_000_000:
                raise ValueError("native address/module overlap work budget exhausted")
            row = generations[position]
            load, unload = row["load"], row["unload"]
            if (
                load["time_filetime_100ns"] <= stamp
                and (unload is None or stamp < unload["time_filetime_100ns"])
                and row["image_base"] <= ip < row["image_base"] + load["image_size"]
            ):
                matching.append(row)
            position -= 1
        reason = None
        if images["undecoded_image_events"]:
            reason = "undecoded_image_boundaries"
        elif len(matching) != 1 or matching[0]["state"] != "observed_load_unload_pair":
            reason = "no_unique_observed_closed_image_generation"
        else:
            image = matching[0]
            load = image["load"]
            module_id = pairs.get(ntpath.normcase(ntpath.normpath(load["image_path_utf8"])))
            if module_id is None:
                reason = "no_explicit_local_PE_PDB_pair"
            else:
                local = modules[module_id]["pair"]["executable"]
                if any(
                    local[key] != load[key]
                    for key in ("image_size", "image_timestamp", "image_checksum")
                ):
                    reason = "current_PE_fingerprint_differs_from_observed_image"
        if reason:
            unresolved.append({result_field: ordinal, "reason": reason})
        else:
            points.append(
                {
                    result_field: ordinal,
                    "module_id": module_id,
                    "image_generation_id": image["generation_id"],
                    "rva": ip - image["image_base"],
                }
            )
    return dict(
        schema=schema,
        selected_pid=pid,
        points=points,
        unresolved=unresolved,
        join_condition="observed_image_lifetime_and_path_size_timestamp_checksum_not_loaded_bytes_attestation",
        path_alias_scope="operator_supplied_recorded_path_not_an_OS_device_path_mapping",
        loaded_image_bytes_verified=False,
        loaded_PDB_identity_verified=False,
        CPU_time_inferred=False,
        promotion_allowed=False,
    )


def decode_lookup(document, *, request, pair, process_id, source_sha256):
    if (
        not isinstance(document, dict)
        or document.get("schema") != "rns8_native_pdb_symbols_v1"
        or type(document.get("process_id")) is not int
        or document["process_id"] != process_id
        or document.get("reader_source_sha256") != source_sha256
        or document.get("lookup") != "explicit_local_PDB_RVAs_not_a_live_process"
        or document.get("inlined_callchain_resolved") is not False
        or document.get("source_contents_verified") is not False
        or type(document.get("pdb_path")) is not str
        or type(document.get("dbghelp_path")) is not str
        or type(document.get("image_size")) is not int
        or document["image_size"] != request["image_size"]
        or ntpath.normcase(document.get("pdb_path", "")) != ntpath.normcase(request["pdb"])
        or ntpath.normcase(document.get("dbghelp_path", "")) != ntpath.normcase(request["dbghelp"])
        or type(document.get("pdb_age")) is not int
        or document["pdb_age"] != pair["symbols"]["age"]
    ):
        raise ValueError("native local PDB lookup identity differs from its owned request")
    guid = document.get("pdb_guid_bytes_le")
    if (
        not isinstance(guid, list)
        or len(guid) != 16
        or any(type(byte) is not int or not 0 <= byte <= 255 for byte in guid)
        or str(uuid.UUID(bytes_le=bytes(guid))) != pair["symbols"]["guid"]
    ):
        raise ValueError("native opened PDB GUID differs from the PE/PDB pair")
    rows = document.get("symbols")
    if not isinstance(rows, list) or len(rows) != len(request["rvas"]) or len(rows) > 512:
        raise ValueError("complete bounded native RVA lookup projection required")
    result = {}
    for row, rva in zip(rows, request["rvas"]):
        if (
            not isinstance(row, dict)
            or not {"rva", "symbol", "source"}.issubset(row)
            or type(row.get("rva")) is not int
            or row["rva"] != rva
            or rva in result
        ):
            raise ValueError("native RVA lookup order/identity differs")
        for field in ("symbol", "source"):
            value = row.get(field)
            if value is None:
                integer(row.get(field + "_error"), "native lookup error", maximum=2**32 - 1)
                continue
            if not isinstance(value, dict):
                raise ValueError("typed native symbol/source observation required")
            start = integer(value.get("start_rva"), "native debug start RVA", maximum=rva)
            displacement = integer(
                value.get("displacement"), "native debug displacement", maximum=rva
            )
            if start + displacement != rva:
                raise ValueError("native debug displacement contradicts requested RVA")
            name = value.get("name" if field == "symbol" else "path")
            if (
                type(name) is not str
                or not name
                or len(name.encode("utf-8")) > 16384
                or any(ord(c) < 32 for c in name)
            ):
                raise ValueError("bounded nonempty native debug text required")
            if field == "source":
                integer(value.get("line"), "native source line", minimum=1, maximum=2**32 - 1)
                if value.get("text") is not None:
                    raise ValueError("PDB line location does not attest current source text")
        result[rva] = row
    return result


def source_report(cpu, candidates, lookups):
    if set(lookups) != {(p["module_id"], p["rva"]) for p in candidates["points"]}:
        raise ValueError("local PDB lookup inventory differs from native module candidates")
    phases = {
        r["native_sample_record"]: r
        for r in (cpu.get("selected_operation") or {}).get("samples", [])
    }
    rows, sites = [], Counter()
    for point in candidates["points"]:
        lookup = lookups[point["module_id"], point["rva"]]
        phase = phases.get(point["native_sample_record"])
        row = dict(
            point,
            symbol=lookup["symbol"],
            source=lookup["source"],
            selected_operation=phase,
            callchain=[],
        )
        rows.append(row)
        if phase:
            source = row["source"] or {}
            symbol = row["symbol"] or {}
            sites[
                (
                    phase["phase"],
                    point["module_id"],
                    symbol.get("name"),
                    source.get("path"),
                    source.get("line"),
                )
            ] += 1
    return dict(
        schema="rns8_windows_CPU_source_observations_v1",
        samples=rows,
        selected_sites=[
            dict(phase=k[0], module_id=k[1], symbol=k[2], path=k[3], line=k[4], samples=v)
            for k, v in sorted(sites.items(), key=lambda kv: str(kv[0]))
        ],
        unresolved=candidates["unresolved"],
        CPU_duration_ns=None,
        local_PDB_lookup_observed=bool(lookups),
        loaded_symbols_verified=False,
        inlined_or_unwound_stacks_resolved=False,
        image_join_condition=candidates["join_condition"],
        source_contents_verified=False,
        production_latency=False,
        promotion_allowed=False,
    )


def symbolized_report(cpu, candidates, frame_catalog, frame_candidates, lookups):
    leaf_keys = {(p["module_id"], p["rva"]) for p in candidates["points"]}
    frame_keys = {(p["module_id"], p["rva"]) for p in frame_candidates["points"]}
    if set(lookups) != leaf_keys | frame_keys:
        raise ValueError("complete native leaf/frame lookup inventory required")
    report = source_report(cpu, candidates, {key: lookups[key] for key in leaf_keys})
    lookup_indices = {key: i + 1 for i, key in enumerate(sorted(lookups))}
    table = [
        dict(lookup_index=lookup_indices[key], module_id=key[0], **lookups[key])
        for key in sorted(lookups)
    ]
    resolved = {p["source_address_index"]: p for p in frame_candidates["points"]}
    unresolved = {p["source_address_index"]: p["reason"] for p in frame_candidates["unresolved"]}
    expected_addresses = {p["source_address_index"] for p in frame_catalog["addresses"]}
    if (
        len(resolved) != len(frame_candidates["points"])
        or len(unresolved) != len(frame_candidates["unresolved"])
        or set(resolved) & set(unresolved)
        or set(resolved) | set(unresolved) != expected_addresses
    ):
        raise ValueError("complete nonoverlapping native frame candidate inventory required")
    addresses = []
    for point in frame_catalog["addresses"]:
        identity = point["source_address_index"]
        candidate = resolved.get(identity)
        if candidate is not None:
            addresses.append(
                dict(
                    point,
                    **{k: v for k, v in candidate.items() if k != "source_address_index"},
                    lookup_index=lookup_indices[candidate["module_id"], candidate["rva"]],
                )
            )
        else:
            addresses.append(dict(point, lookup_index=None, unresolved_reason=unresolved[identity]))
    address_map = {r["source_address_index"]: r for r in addresses}
    fragments = {r["stack_record"]: r for r in frame_catalog["fragments"]}
    lookup_map = {r["lookup_index"]: r for r in table}
    cpu_rows = {r["native_sample_record"]: r for r in report["samples"]}
    cpu_phases = {
        r["native_sample_record"]: r
        for r in (cpu.get("selected_operation") or {}).get("samples", [])
    }
    for point in cpu["samples"]:
        identity = point["record_index"]
        if identity not in cpu_rows:
            cpu_rows[identity] = dict(
                native_sample_record=identity,
                symbol=None,
                source=None,
                selected_operation=cpu_phases.get(identity),
                callchain=[],
            )
        cpu_rows[identity].update({k: v for k, v in point.items() if k != "record_index"})
        cpu_rows[identity]["stack_fragment_records"] = []
    frame_counts, event_counts, fragment_counts = Counter(), Counter(), Counter()
    unresolved_slots = 0
    for target in frame_catalog["targets"]:
        if target["source_kind"] == "cpu_sample":
            row = cpu_rows[target["native_point_record"]]
            if (
                row["tid"] != target["tid"]
                or row["thread_generation_id"] != target["thread_generation_id"]
                or row["time_filetime_100ns"] != target["time_filetime_100ns"]
            ):
                raise ValueError("native CPU point and fragment original-event identities differ")
            row["stack_fragment_records"] = target["stack_records"]
            first, second = row["selected_operation"], target.get("operation_phase")
            if (
                (first is None) != (second is None)
                or first is not None
                and any(
                    first[key] != second[key]
                    for key in ("measured_root_id", "phase_range_id", "phase", "tid")
                )
            ):
                raise ValueError("native CPU and stack event measured phases differ")
        phase = (target.get("operation_phase") or {}).get("phase")
        observed = set()
        for identity in target["stack_records"]:
            fragment = fragments[identity]
            if fragment["source_target_id"] != target["source_target_id"]:
                raise ValueError("normalized fragment belongs to another native event")
            within_fragment = set()
            for address_id in fragment["source_address_indices"]:
                lookup_id = address_map[address_id]["lookup_index"]
                if lookup_id is None:
                    unresolved_slots += 1
                    continue
                value = lookup_map[lookup_id]
                source, symbol = value["source"] or {}, value["symbol"] or {}
                key = (
                    target["source_kind"],
                    phase,
                    value["module_id"],
                    symbol.get("name"),
                    source.get("path"),
                    source.get("line"),
                )
                frame_counts[key] += 1
                within_fragment.add(key)
                observed.add(key)
            fragment_counts.update(within_fragment)
        event_counts.update(observed)
    report["samples"] = [cpu_rows[p["record_index"]] for p in cpu["samples"]]
    report.update(
        schema="rns8_windows_native_source_observations_v2",
        native_stack_fragment_sources=dict(
            schema="rns8_windows_native_fragment_sources_v1",
            addresses=addresses,
            fragments=frame_catalog["fragments"],
            targets=frame_catalog["targets"],
            lookup_table=table,
            address_policy=frame_catalog["address_policy"],
            native_bound_frame_observations=frame_catalog["native_bound_frame_observations"],
            unique_address_epoch_candidates=frame_catalog["unique_address_epoch_candidates"],
            unresolved_module_frame_slots=unresolved_slots,
            unresolved_native_stack_fragments=frame_catalog["unresolved_native_stack_fragments"],
            native_fragment_concatenation_performed=False,
            loaded_image_bytes_verified=False,
        ),
        stack_source_sites=[
            dict(
                source_kind=k[0],
                phase=k[1],
                scope="native_measured_phase" if k[1] else "process_total_outside_or_unbound_phase",
                module_id=k[2],
                symbol=k[3],
                path=k[4],
                line=k[5],
                native_event_count=event_counts[k],
                native_fragment_count=fragment_counts[k],
                native_frame_slot_observations=v,
            )
            for k, v in sorted(frame_counts.items(), key=lambda kv: str(kv[0]))
        ],
        native_stack_fragment_symbolization_observed=any(
            p["symbol"] is not None or p["source"] is not None
            for p in table
            if (p["module_id"], p["rva"]) in frame_keys
        ),
        local_PDB_lookup_observed=bool(lookups),
        current_source_custody_scope="conditional_native_leaf_and_fragment_debug_locations_not_additive_CPU_samples",
    )
    return report


def source_observations(report):
    """Stream normalized native debug locations into the shared source custodian."""
    catalog = report["native_stack_fragment_sources"]
    addresses = {r["source_address_index"]: r for r in catalog["addresses"]}
    fragments = {r["stack_record"]: r for r in catalog["fragments"]}
    lookups = {r["lookup_index"]: r for r in catalog["lookup_table"]}

    def frames(identities):
        for identity in identities:
            for address_id in fragments[identity]["source_address_indices"]:
                index = addresses[address_id]["lookup_index"]
                yield dict(source=lookups[index]["source"] if index is not None else None)

    for row in report["samples"]:
        yield dict(source=row["source"], callchain=frames(row["stack_fragment_records"]))
    for target in catalog["targets"]:
        if target["source_kind"] != "cpu_sample":
            yield dict(source=None, callchain=frames(target["stack_records"]))
