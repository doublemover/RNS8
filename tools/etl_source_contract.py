"""Conditional native CPU IP/module joins and strict local PDB lookup receipts."""

from __future__ import annotations

from bisect import bisect_right
from collections import Counter
import ntpath
import uuid

from etl_process_contract import integer


def module_points(cpu, images, modules):
    if (
        cpu.get("schema") != "rns8_etl_cpu_observations_v1"
        or images.get("schema") != "rns8_etl_image_catalog_v1"
        or type(cpu.get("selected_pid")) is not int
        or cpu["selected_pid"] != images.get("selected_pid")
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
    for point in cpu["samples"]:
        ordinal = integer(
            point.get("record_index"), "native sample record", minimum=1, maximum=1000000
        )
        if ordinal in seen:
            raise ValueError("duplicate native CPU source sample")
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
            unresolved.append(dict(native_sample_record=ordinal, reason=reason))
        else:
            points.append(
                dict(
                    native_sample_record=ordinal,
                    module_id=module_id,
                    image_generation_id=image["generation_id"],
                    rva=ip - image["image_base"],
                )
            )
    return dict(
        schema="rns8_etl_CPU_module_candidates_v1",
        selected_pid=cpu["selected_pid"],
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
