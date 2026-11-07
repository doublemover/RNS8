"""Read an explicit unaggregated English WPA CPU-sample CSV contract."""

from __future__ import annotations

from collections import Counter
import csv
from decimal import Decimal
import hashlib
from pathlib import Path, PureWindowsPath
import re

MAX_BYTES = 64 * 1024**2
MAX_ROWS = 100_000
HEADERS = (
    "Process",
    "Thread ID",
    "CPU",
    "TimeStamp (s)",
    "Stack",
    "Module",
    "Function",
    "Weight (ms)",
    "Count",
)


def bounded_lines(stream):
    total, first = 0, True
    while raw := stream.readline(MAX_BYTES - total + 1):
        total += len(raw)
        if total > MAX_BYTES:
            raise ValueError("native WPA CSV exceeds byte budget during decoding")
        yield raw.decode("utf-8-sig" if first else "utf-8", errors="strict")
        first = False


def content_digest(path):
    total, digest = 0, hashlib.sha256()
    with path.open("rb") as stream:
        while raw := stream.read(min(1024**2, MAX_BYTES - total + 1)):
            total += len(raw)
            if total > MAX_BYTES:
                raise ValueError("native WPA CSV exceeds hash byte budget")
            digest.update(raw)
    return digest.hexdigest()


def unsigned(text, name, maximum=2**32 - 1):
    if type(text) is not str or not re.fullmatch(r"[0-9]{1,20}", text):
        raise ValueError("explicit WPA integer required: " + name)
    value = int(text)
    if value > maximum:
        raise ValueError("WPA integer outside native field domain: " + name)
    return value


def nanoseconds(text, scale):
    if type(text) is not str or not re.fullmatch(r"[0-9]{1,12}(?:\.[0-9]{1,9})?", text):
        raise ValueError("explicit nonnegative WPA time with reviewed units required")
    value = Decimal(text) * scale
    if value != value.to_integral_value() or value >= 1 << 64:
        raise ValueError("WPA time cannot be represented as exact uint64 nanoseconds")
    return int(value)


def read_samples(path, *, pid, executable):
    if type(pid) is not int or not 0 < pid < 1 << 32:
        raise ValueError("actual benchmark PID required for WPA selection")
    path = Path(path).resolve(strict=True)
    if not path.is_file() or not 0 < path.stat().st_size <= MAX_BYTES:
        raise ValueError("bounded nonempty native WPA CSV required")
    before = content_digest(path)
    previous = csv.field_size_limit(16384)
    rows, excluded, observed, threads = [], 0, 0, Counter()
    try:
        with path.open("rb") as stream:
            reader = csv.reader(bounded_lines(stream), strict=True)
            if tuple(next(reader, ())) != HEADERS:
                raise ValueError(
                    "unsupported WPA CSV columns/locale/units; review native export/profile"
                )
            for fields in reader:
                observed += 1
                if observed > MAX_ROWS or len(fields) != len(HEADERS):
                    raise ValueError("WPA row count/shape exceeds unaggregated sample contract")
                process = re.fullmatch(r"(.+) \(([0-9]{1,10})\)", fields[0])
                if process is None:
                    raise ValueError("native WPA Process column has no explicit image/PID")
                process_id = unsigned(process[2], "PID")
                # Validate the complete export before omitting foreign processes.
                thread = unsigned(fields[1], "TID")
                cpu = unsigned(fields[2], "CPU", 1048575)
                timestamp, weight = nanoseconds(fields[3], 10**9), nanoseconds(fields[7], 10**6)
                count = unsigned(fields[8], "Count")
                if count != 1 and not (count == 0 and process_id == 0):
                    raise ValueError("aggregated WPA rows are not individual CPU samples")
                if process_id != pid:
                    excluded += 1
                    continue
                if (
                    PureWindowsPath(process[1]).name.casefold()
                    != PureWindowsPath(str(executable)).name.casefold()
                    or not thread
                ):
                    raise ValueError(
                        "WPA selected PID/image/thread contradicts the benchmark process"
                    )
                rows.append(
                    dict(
                        pid=pid,
                        tid=thread,
                        cpu=cpu,
                        time_ns=timestamp,
                        sampled_cpu_weight_ns=weight,
                        stack=fields[4] or None,
                        module=fields[5] or None,
                        function=fields[6] or None,
                    )
                )
                threads[thread] += weight
    except csv.Error as error:
        raise ValueError("malformed or oversized native WPA CSV field") from error
    finally:
        csv.field_size_limit(previous)
    if not rows:
        raise ValueError("WPA export contains no selected benchmark samples")
    if path.stat().st_size > MAX_BYTES or content_digest(path) != before:
        raise ValueError("native WPA export changed during decoding")
    return dict(
        schema="rns8_wpa_cpu_samples_v1",
        input=dict(path=str(path), sha256=before),
        selected_pid=pid,
        observed_rows=observed,
        excluded_other_process_rows=excluded,
        samples=rows,
        threads=[
            dict(tid=tid, sampled_cpu_weight_ns=value) for tid, value in sorted(threads.items())
        ],
        clock="native_ETL_relative_time_not_aligned_to_RNS8_or_GPU_clock",
        scope="selected_process_total_not_measured_operation",
        total_sampled_cpu_weight_ns=sum(threads.values()),
        loss_status="not_observed_from_CPU_sample_table",
        loaded_module_or_PDB_identity_verified=False,
        source_contents_verified=False,
        production_latency=False,
        promotion_allowed=False,
    )
