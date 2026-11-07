"""Recorded user mappings and conservative native-frame attribution.

MMAP2 build IDs are observations from the recording, never borrowed from a
similarly named current file. Overlapping/reused address ranges stay unresolved
instead of inferring a mapping lifetime from file order across CPU buffers.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import Counter, defaultdict
from pathlib import Path
import struct

from elf_module_identity import module_inventory

SCHEMA = "rns8_perf_recorded_mapping_custody_v1"
MAX_MAPPINGS = 8192
MAX_MAPPING_BYTES = 4 * 1024**2
MAX_FRAME_OBSERVATIONS = 500_000
SOURCE = "https://github.com/torvalds/linux/blob/47f05f71ad988c3180bdba807151e6e86ed75aa3/tools/lib/perf/include/perf/event.h"


def mapping_record(payload, *, kind, misc, trailer_size, time_ns, clock_id, ordinal):
    fixed = 32 if kind == 1 else 64
    body = payload[: len(payload) - trailer_size] if trailer_size else payload
    if len(body) <= fixed:
        raise ValueError("truncated perf MMAP/MMAP2 record")
    pid, tid, start, length, offset = struct.unpack_from("<IIQQQ", body)
    if not length or start + length > 1 << 64 or offset + length > 1 << 64:
        raise ValueError("perf mapping address/file extent overflows uint64")
    filename = body[fixed:]
    end = filename.find(b"\0")
    if not 0 < end <= 4096 or len(filename) - end > 8 or any(filename[end:]):
        raise ValueError("perf mapping filename is unterminated or has invalid padding")
    name = filename[:end].decode("utf-8", errors="strict")
    build_id, protection = None, None
    if kind == 10:
        protection, _ = struct.unpack_from("<II", body, 56)
        if misc & (1 << 14):
            size, reserved1, reserved2 = struct.unpack_from("<BBH", body, 32)
            if not 0 < size <= 20 or reserved1 or reserved2:
                raise ValueError("invalid recorded MMAP2 build-ID union")
            build_id = body[36 : 36 + size].hex()
    return dict(
        record_ordinal=ordinal,
        kind="MMAP" if kind == 1 else "MMAP2",
        cpu_mode=misc & 7,
        pid=pid,
        tid=tid,
        start=start,
        end=start + length,
        file_offset=offset,
        native_path=name,
        native_build_id=build_id,
        protection=protection,
        time_ns=time_ns,
        clock_id=clock_id,
    )


class MappingIndex:
    """O(M log M) construction and O(log M) lookup; ambiguous unions never pass."""

    def __init__(self, recording):
        self.rows = recording["recorded_mappings"]
        changes = defaultdict(list)
        self.lifecycle = defaultdict(list)
        self.unordered_lifecycle = set()
        for row in recording["process_mapping_boundaries"]:
            if (
                row["time_ns"] is None
                or row["clock_id"] != recording["clock"]["attribute_clock_id"]
            ):
                self.unordered_lifecycle.add(row["pid"])
            else:
                self.lifecycle[row["pid"]].append(row["time_ns"])
        for values in self.lifecycle.values():
            values.sort()
        for index, row in enumerate(self.rows):
            if row["cpu_mode"] == 2:
                changes[row["pid"]].extend(((row["start"], 1, index), (row["end"], -1, index)))
        self.intervals = {}
        for pid, events in changes.items():
            active, starts, entries = set(), [], []
            events.sort()
            cursor = 0
            while cursor < len(events):
                position = events[cursor][0]
                while cursor < len(events) and events[cursor][0] == position:
                    _, delta, index = events[cursor]
                    if delta == 1:
                        active.add(index)
                    else:
                        active.remove(index)
                    cursor += 1
                starts.append(position)
                entries.append(next(iter(active)) if len(active) == 1 else -1)
            self.intervals[pid] = starts, entries

    def lookup(self, *, pid, ip, time_ns, clock_id):
        starts, entries = self.intervals.get(pid, ([], []))
        index = bisect_right(starts, ip) - 1
        if index < 0 or entries[index] == -1:
            return None, "missing_or_overlapping_recorded_mapping"
        row = self.rows[entries[index]]
        if row["time_ns"] is None or row["clock_id"] != clock_id or not row["time_ns"] < time_ns:
            return None, "mapping_timestamp_missing_foreign_or_not_before_sample"
        if pid in self.unordered_lifecycle:
            return None, "unordered_process_address_space_boundary"
        boundaries = self.lifecycle.get(pid, ())
        boundary_index = bisect_right(boundaries, time_ns) - 1
        if boundary_index >= 0 and row["time_ns"] <= boundaries[boundary_index]:
            return None, "mapping_precedes_process_address_space_boundary"
        if row["protection"] is None or not row["protection"] & 4:
            return None, "recorded_executable_protection_unavailable"
        if row["native_build_id"] is None:
            return None, "recorded_mapping_has_no_build_id"
        if not Path(row["native_path"]).is_absolute():
            return None, "recorded_mapping_has_no_absolute_module_path"
        return row, None


def frame_custody(samples, recording, *, machine):
    """Join exact PID/IP/time to recorded maps, then inspect the actual ELF files."""
    index = MappingIndex(recording)
    counts, candidates, queries = Counter(), {}, {}
    selected = 0
    clock = recording["clock"]["attribute_clock_id"]
    for sample in samples:
        if type(sample) is not dict:
            raise ValueError("typed native sample required for frame custody")
        frames = sample.get("callchain")
        if type(frames) is not list or len(frames) > 128:
            raise ValueError("bounded native callchain required for mapping custody")
        # Include the leaf in exactly the same raw-record authority as callers.
        for role, frame in [("leaf", sample), *(("callchain", frame) for frame in frames)]:
            if type(frame) is not dict:
                raise ValueError("typed native frame required for mapping custody")
            selected += 1
            if selected > MAX_FRAME_OBSERVATIONS:
                raise ValueError("native frame attribution budget exhausted")
            ip = frame.get("ip")
            if type(ip) is not int or not 0 <= ip < 1 << 64:
                raise ValueError("invalid native frame IP")
            key = sample["pid"], sample["time_ns"], ip
            if key not in queries:
                queries[key] = index.lookup(pid=key[0], time_ns=key[1], ip=ip, clock_id=clock)
            mapping, reason = queries[key]
            if mapping is None:
                counts[(role, reason)] += 1
                continue
            name = frame.get("dso_path") if role == "leaf" else frame.get("dso")
            if name not in {mapping["native_path"], Path(mapping["native_path"]).name}:
                counts[(role, "native_frame_name_disagrees_with_recorded_mapping")] += 1
                continue
            if (
                role == "leaf"
                and (frame.get("dso_build_id") or "").lower() != mapping["native_build_id"]
            ):
                counts[(role, "native_leaf_build_id_disagrees_with_recorded_mapping")] += 1
                continue
            key = role, mapping["record_ordinal"], ip, frame.get("symbol")
            if key not in candidates:
                candidates[key] = dict(
                    role=role,
                    count=0,
                    record_ordinal=mapping["record_ordinal"],
                    observation=dict(
                        ip=ip,
                        dso_path=mapping["native_path"],
                        dso_build_id=mapping["native_build_id"],
                        symbol=frame.get("symbol"),
                        mapping=dict(
                            map_start=mapping["start"],
                            map_end=mapping["end"],
                            map_pgoff=mapping["file_offset"],
                        ),
                    ),
                )
            candidates[key]["count"] += 1
    # Decode each distinct module once, not once per sampled frame. Observation
    # multiplicities are retained separately rather than expanding the rows.
    modules = module_inventory([row["observation"] for row in candidates.values()], machine=machine)
    files = {(row["native_path"], row["native_build_id"]): row for row in modules["modules"]}
    observations = []
    for row in candidates.values():
        value = row["observation"]
        module = files[value["dso_path"], value["dso_build_id"]]
        offset = value["ip"] - value["mapping"]["map_start"] + value["mapping"]["map_pgoff"]
        executable = module["state"] == "matched" and any(
            segment["file_offset"] <= offset < segment["file_offset"] + segment["file_bytes"]
            for segment in module["file"]["executable_load_segments"]
        )
        state = (
            "recorded_build_id_and_current_executable_extent"
            if executable
            else "current_ELF_identity_or_executable_extent_unresolved"
        )
        counts[(row["role"], state)] += row["count"]
        observations.append(
            dict(
                role=row["role"],
                observations=row["count"],
                record_ordinal=row["record_ordinal"],
                ip=value["ip"],
                native_path=value["dso_path"],
                native_build_id=value["dso_build_id"],
                state=state,
            )
        )
    return dict(
        schema=SCHEMA,
        selected_frame_observations=selected,
        counts=[
            dict(role=role, state=state, observations=count)
            for (role, state), count in sorted(counts.items())
        ],
        observations=observations,
        current_modules=modules,
        recorded_mapping_records=len(recording["recorded_mappings"]),
        complete_transient_loaded_image_history_verified=False,
        independently_replayed_raw_unwind_frames=False,
        overlap_policy="any_address_reuse_in_recording_remains_unresolved",
        missing_inherited_mappings_policy="no_parent_or_basename_only_identity_inference",
        source=SOURCE,
    )
