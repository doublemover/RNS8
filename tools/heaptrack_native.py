"""Bounded heaptrack1.5.0 raw-format3 allocation generations, not latency evidence."""

from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path
import re

from execution_inventory import bounded_read

SCHEMA = "rns8_native_heaptrack_allocations_v1"
PRODUCER = "https://github.com/KDE/heaptrack/blob/v1.5.0/src/track/libheaptrack.cpp"
MAX_RECORDS = 100_000
MAX_TRACES = 32_768
MAX_MODULES = 16_384
MAX_SEGMENTS = 65_536
MS = 1_000_000


def hexadecimal(value: str) -> int:
    if not re.fullmatch(r"[0-9a-fA-F]{1,16}", value):
        raise ValueError("native heaptrack uint64 hexadecimal field required")
    return int(value, 16)


def numbers(payload: str, count: int) -> list[int]:
    parts = payload.split(" ")
    if len(parts) != count:
        raise ValueError("native heaptrack field count differs")
    return [hexadecimal(part) for part in parts]


def sized_string(payload: str) -> tuple[str, str]:
    # Native lengths are bytes, including for non-ASCII paths containing spaces.
    length, separator, remainder = payload.partition(" ")
    size = hexadecimal(length)
    raw = remainder.encode("utf-8")
    if not separator or not 0 < size <= 4096 or len(raw) < size:
        raise ValueError("native heaptrack sized string is truncated or over budget")
    text = raw[:size].decode("utf-8")
    tail = raw[size:].decode("utf-8")
    if "\0" in text or any(ord(character) < 32 for character in text):
        raise ValueError("native heaptrack path has unsupported control characters")
    if tail and not tail.startswith(" "):
        raise ValueError("native heaptrack sized string has trailing bytes")
    return text, tail[1:] if tail else ""


def read_allocations(path: Path, *, pid: int, execution_id: str, executable: str) -> dict:
    if (
        type(pid) is not int
        or not 0 < pid < 1 << 32
        or type(execution_id) is not str
        or not execution_id
        or len(execution_id) > 256
        or type(executable) is not str
        or not executable.startswith("/")
    ):
        raise ValueError("owned native process/run and absolute executable identity required")
    path = Path(path)
    source, before = bounded_read(path)
    if not source.endswith("\n"):
        raise ValueError("native heaptrack final record is truncated")
    lines = source.split("\n")[:-1]
    if not 1 <= len(lines) <= MAX_RECORDS:
        raise ValueError("native heaptrack record count exceeds budget")
    traces, modules, allocations, rss = [], [], [], []
    live, buckets, generations = {}, [dict(lower_ns=0, upper_ns=None)], Counter()
    sizes, by_trace, unknown = Counter(), {}, []
    version = image = system = command_sha256 = None
    current_bytes = peak_bytes = total_bytes = frees = temporary = suppressions = 0
    last_pointer = None
    module_generation = 0
    module_segments = 0
    for line_number, line in enumerate(lines, 1):
        if not 2 <= len(line.encode("utf-8")) <= 16_384 or line[1:2] != " ":
            raise ValueError("unsupported or over-budget native heaptrack record")
        mode, payload = line[0], line[2:]
        if line_number == 1 and mode != "v":
            raise ValueError("native heaptrack version must precede all records")
        if mode == "v":
            if version is not None or numbers(payload, 2) != [0x10500, 3]:
                raise ValueError("unsupported or repeated heaptrack producer/file version")
            version = "1.5.0"
        elif mode == "x":
            image, tail = sized_string(payload) if image is None else (None, "duplicate")
            if tail or image != executable:
                raise ValueError("heaptrack executable differs or contains another process")
        elif mode == "X":
            if command_sha256 is not None:
                raise ValueError("duplicate native heaptrack command record")
            command_sha256 = hashlib.sha256(payload.encode()).hexdigest()
        elif mode == "I":
            if system is not None:
                raise ValueError("duplicate native heaptrack system record")
            page_bytes, physical_pages = numbers(payload, 2)
            if not page_bytes or page_bytes & (page_bytes - 1) or page_bytes > 1 << 20:
                raise ValueError("unsupported native heaptrack page size")
            system = dict(page_bytes=page_bytes, physical_pages=physical_pages)
        elif mode == "c":
            stamp = numbers(payload, 1)[0]
            if stamp >= (1 << 64) // MS - 1 or stamp * MS < buckets[-1]["lower_ns"]:
                raise ValueError("native heaptrack clock is reversed or exceeds domain")
            # Timestamp records are floored milliseconds around groups of events,
            # not per-allocation timestamps. Keep the entire uncertainty interval.
            buckets[-1]["upper_ns"] = (stamp + 1) * MS
            buckets.append(dict(lower_ns=stamp * MS, upper_ns=None))
        elif mode == "R":
            if system is None:
                raise ValueError("native RSS pages have no recorded page size")
            pages = numbers(payload, 1)[0]
            if pages * system["page_bytes"] >= 1 << 64:
                raise ValueError("native RSS observation exceeds uint64 bytes")
            rss.append(dict(bucket=len(buckets) - 1, bytes=pages * system["page_bytes"]))
        elif mode == "S":
            suppressions += 1
        elif mode == "m":
            name, tail = sized_string(payload)
            if name == "-":
                if tail:
                    raise ValueError("native heaptrack module reset has extra fields")
                module_generation += 1
            else:
                values = [hexadecimal(part) for part in tail.split(" ")]
                if len(values) < 3 or len(values) % 2 != 1 or len(modules) >= MAX_MODULES:
                    raise ValueError("native heaptrack module extent/budget differs")
                base, *segments = values
                module_segments += len(segments) // 2
                if module_segments > MAX_SEGMENTS:
                    raise ValueError("native heaptrack total module-segment budget exhausted")
                if any(
                    base + first + size >= 1 << 64
                    for first, size in zip(segments[::2], segments[1::2])
                ):
                    raise ValueError("native heaptrack module extent wraps")
                modules.append(
                    dict(generation=module_generation, name=name, base=base, segments=segments)
                )
        elif mode == "t":
            address, parent = numbers(payload, 2)
            if len(traces) >= MAX_TRACES or parent > len(traces):
                raise ValueError("native heaptrack trace has a future parent or exceeds budget")
            traces.append(dict(address=address, parent=parent, module_generation=module_generation))
        elif mode == "+":
            size, trace, pointer = numbers(payload, 3)
            if image is None or not pointer or not 0 < trace <= len(traces) or pointer in live:
                raise ValueError(
                    "native heaptrack allocation identity/trace/live generation differs"
                )
            generations[pointer] += 1
            record = dict(
                allocation_id=len(allocations) + 1,
                pointer=pointer,
                generation=generations[pointer],
                bytes=size,
                trace=trace,
                allocated_module_generation=module_generation,
                allocated_bucket=len(buckets) - 1,
                freed_bucket=None,
                allocation_record=line_number,
                free_record=None,
            )
            live[pointer] = record
            allocations.append(record)
            current_bytes += size
            peak_bytes = max(peak_bytes, current_bytes)
            total_bytes += size
            sizes[0 if size == 0 else (size - 1).bit_length()] += 1
            costs = by_trace.setdefault(
                trace, dict(allocations=0, requested_bytes=0, freed_bytes=0)
            )
            costs["allocations"] += 1
            costs["requested_bytes"] += size
            last_pointer = pointer
        elif mode == "-":
            pointer = numbers(payload, 1)[0]
            if not pointer:
                raise ValueError("native heaptrack null free is not a recorded allocation")
            record = live.pop(pointer, None)
            if record is None:
                unknown.append(dict(pointer=pointer, record=line_number, bucket=len(buckets) - 1))
            else:
                record.update(freed_bucket=len(buckets) - 1, free_record=line_number)
                current_bytes -= record["bytes"]
                by_trace[record["trace"]]["freed_bytes"] += record["bytes"]
                frees += 1
                temporary += last_pointer == pointer
            last_pointer = None
        else:
            raise ValueError("unsupported native heaptrack record kind: " + mode)
    if image is None or system is None or command_sha256 is None or len(buckets) < 2:
        raise ValueError("native heaptrack process/system/clock records are incomplete")
    for record in allocations:
        first = buckets[record["allocated_bucket"]]
        last = buckets[record["freed_bucket"]] if record["freed_bucket"] is not None else None
        record["lifetime_bounds_ns"] = (
            dict(
                minimum=max(0, last["lower_ns"] - first["upper_ns"]),
                maximum=last["upper_ns"] - first["lower_ns"]
                if last["upper_ns"] is not None
                else None,
            )
            if last is not None and first["upper_ns"] is not None
            else None
        )
    _, after = bounded_read(path)
    if before != after:
        raise ValueError("native heaptrack data changed during decoding")
    partial = []
    if unknown:
        partial.append("frees_without_observed_allocation_generation")
    if any(
        buckets[row["allocated_bucket"]]["upper_ns"] is None
        or (row["freed_bucket"] is not None and buckets[row["freed_bucket"]]["upper_ns"] is None)
        for row in allocations
    ):
        partial.append("allocation_or_free_after_last_native_clock_bracket")
    return dict(
        schema=SCHEMA,
        input=before,
        producer=PRODUCER,
        producer_version=version,
        process_id=pid,
        execution_id=execution_id,
        process_identity_source="owned_direct_child_receipt_not_a_native_trace_PID",
        executable=image,
        native_command_text_sha256=command_sha256,
        memory_domain="intercepted_host_heap_requested_payload",
        clock="unbound_process_relative_steady_clock_floored_milliseconds",
        time_buckets=buckets,
        allocations=allocations,
        trace_nodes=traces,
        module_records=modules,
        unknown_frees=unknown,
        summary=dict(
            allocations=len(allocations),
            matched_frees=frees,
            unmatched_frees=len(unknown),
            requested_bytes=total_bytes,
            observed_live_peak_bytes=peak_bytes,
            observed_retained_bytes=current_bytes,
            adjacent_allocation_free_pairs=temporary,
            reallocation_count=None,
            size_buckets_log2_ceiling=dict(sorted(sizes.items())),
            by_trace={str(key): value for key, value in sorted(by_trace.items())},
        ),
        rss_observations=rss,
        sampled_rss_max_bytes=max((row["bytes"] for row in rss), default=None),
        suppression_records=suppressions,
        partial_observations=partial,
        phase_attribution="unbound_native_trace_has_no_operation_or_thread_markers",
        symbol_attribution="raw_addresses_only_no_loaded_symbol_identity",
        pre_capture_live_bytes=None,
        exact_process_peak_bytes=None,
        device_or_pinned_bytes=None,
        complete_interception_proven=False,
        native_complete_finalization_marker_available=False,
        production_latency=False,
        promotion_allowed=False,
    )
