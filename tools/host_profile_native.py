#!/usr/bin/env python3
"""Decode bounded process CPU samples through perf's native standalone API."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
from typing import Any

from benchmark_request import file_digest
from compiled_binary_identity import canonical_digest

SCHEMA = "rns8_perf_native_samples_v1"
PRODUCER_CONTRACT = "linux_perf_standalone_session_sample_api_v1"
PRODUCER_SOURCE = (
    "https://github.com/torvalds/linux/blob/master/tools/perf/Documentation/perf-script-python.txt"
)
MAX_SAMPLES = 100_000
MAX_FRAMES = 128
MAX_DECODED_BYTES = 64 * 1024**2


def integer(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or not minimum <= value <= 2**64 - 1:
        raise ValueError("invalid native perf " + name)
    return value


def text(value: Any, name: str) -> str | None:
    if value is None:
        return None
    if type(value) is not str or len(value.encode("utf-8")) > 4096 or "\0" in value:
        raise ValueError("invalid native perf " + name)
    return value


def optional_integer(value: Any, name: str) -> int | None:
    return None if value is None else integer(value, name)


def sample_record(sample: Any) -> dict:
    """Use producer attributes directly; legacy callback dictionaries are unsupported."""
    frames = []
    chain = sample.callchain
    if chain is not None:
        for node in chain:
            if len(frames) >= MAX_FRAMES:
                raise ValueError("native perf callchain exceeds frame budget")
            frames.append(
                dict(
                    ip=integer(node.ip, "callchain IP"),
                    symbol=text(node.symbol, "callchain symbol"),
                    dso=text(node.dso, "callchain DSO"),
                )
            )
    source = sample.srccode()
    if source is not None:
        if type(source) is not tuple or len(source) != 3:
            raise ValueError("unexpected native perf source tuple")
        source = dict(
            path=text(source[0], "source path"),
            line=integer(source[1], "source line"),
            text=text(source[2], "source text"),
        )
    event = str(sample.evsel)
    # The producer's evsel repr encloses the original event selector.
    if event.startswith("evsel(") and event.endswith(")"):
        event = event[6:-1]
    return dict(
        pid=integer(sample.sample_pid, "PID", 1),
        tid=integer(sample.sample_tid, "TID", 1),
        time_ns=integer(sample.sample_time, "timestamp"),
        cpu=integer(sample.sample_cpu, "CPU"),
        period=integer(sample.sample_period, "period", 1),
        event=text(event, "event"),
        ip=integer(sample.sample_ip, "IP"),
        machine_pid=integer(sample.machine_pid, "machine PID"),
        symbol=text(sample.symbol, "symbol"),
        symbol_start=optional_integer(sample.sym_start, "symbol start"),
        symbol_end=optional_integer(sample.sym_end, "symbol end"),
        symbol_offset=optional_integer(sample.sym_offset, "symbol offset"),
        dso=text(sample.dso, "DSO"),
        dso_path=text(sample.dso_long_name, "DSO path"),
        dso_build_id=text(sample.dso_bid, "DSO build ID"),
        source=source,
        callchain=frames,
        callchain_module_identity="native_path_only_not_independently_verified_build_ids",
    )


def decode(
    data: Path,
    pid: int,
    *,
    perf_module=None,
    max_samples: int = MAX_SAMPLES,
    max_decoded_bytes: int = MAX_DECODED_BYTES,
    max_input_bytes: int = 256 * 1024**2,
) -> dict:
    integer(pid, "selected PID", 1)
    if (
        type(max_samples) is not int
        or not 1 <= max_samples <= MAX_SAMPLES
        or type(max_decoded_bytes) is not int
        or not 1 <= max_decoded_bytes <= MAX_DECODED_BYTES
        or type(max_input_bytes) is not int
        or not 1 <= max_input_bytes <= 8 * 1024**3
    ):
        raise ValueError("invalid native perf input/sample/decoded-byte budget")
    data = data.resolve(strict=True)
    if not data.is_file() or not 1 <= data.stat().st_size <= max_input_bytes:
        raise ValueError("native perf data is empty, nonregular or outside input budget")
    digest = file_digest(data)
    if perf_module is None:
        perf_module = importlib.import_module("perf")
    if not callable(getattr(perf_module, "session", None)) or not callable(
        getattr(perf_module, "data", None)
    ):
        raise ValueError("perf module lacks the maintained standalone session/data interface")
    module_path = Path(perf_module.__file__).resolve(strict=True)
    module_digest = file_digest(module_path)
    rows, bytes_used, observed, excluded = [], 0, 0, 0

    def receive(sample):
        nonlocal bytes_used, observed, excluded
        observed += 1
        if observed > max_samples:
            raise ValueError("native perf observed-sample budget exhausted")
        if integer(sample.sample_pid, "PID", 1) != pid:
            excluded += 1
            return
        row = sample_record(sample)
        if row["machine_pid"] != 0:
            raise ValueError("guest samples require a separate VM clock/process contract")
        if row["event"] not in {"cpu-clock", "cpu-clock:u", "cpu-clock:up"}:
            raise ValueError(
                "native CPU sample decoder accepts only the declared software CPU clock event"
            )
        # A CPU-clock sample period estimates CPU exposure, never wall latency.
        row["period_unit"] = "cpu_clock_nanoseconds"
        bytes_used += len(json.dumps(row, ensure_ascii=True, allow_nan=False).encode("utf-8"))
        if bytes_used > max_decoded_bytes:
            raise ValueError("native perf decoded-byte budget exhausted")
        rows.append(row)

    session = perf_module.session(perf_module.data(str(data)), sample=receive)
    try:
        session.process_events()
        machine, is_64_bit = session.e_machine, session.is_64_bit
    finally:
        session = None  # Release the native session/callback cycle after errors too.
    if file_digest(data) != digest or file_digest(module_path) != module_digest:
        raise ValueError("native perf data or decoder module changed during decoding")
    if not rows:
        raise ValueError("native perf trace contains no selected-process CPU samples")
    result = dict(
        schema=SCHEMA,
        producer_contract=PRODUCER_CONTRACT,
        producer_source=PRODUCER_SOURCE,
        input=dict(path=str(data), sha256=digest),
        recording_sha256=digest,
        native_module=dict(path=str(module_path), sha256=module_digest),
        decoder_source_sha256=file_digest(Path(__file__)),
        pid=pid,
        machine=machine,
        is_64_bit=is_64_bit,
        clock="recorded_perf_clock_requires_acquisition_clockid_receipt",
        observed_samples=observed,
        excluded_other_process_samples=excluded,
        selected_process_samples=len(rows),
        decoded_sample_bytes=bytes_used,
        budgets=dict(
            max_samples=max_samples,
            max_frames=MAX_FRAMES,
            max_decoded_bytes=max_decoded_bytes,
            max_input_bytes=max_input_bytes,
        ),
        samples=rows,
        scope="selected_process_total_not_selected_operation",
        loss_status="unknown_native_session_api_does_not_expose_loss_counts",
        symbolized_samples=sum(bool(row["symbol"]) for row in rows),
        source_attributed_samples=sum(
            row["source"] is not None and bool(row["source"]["path"]) and row["source"]["line"] > 0
            for row in rows
        ),
        acquisition_verified=False,
        production_latency=False,
        promotion_allowed=False,
    )
    result["identity_sha256"] = canonical_digest(result)
    if (
        len(
            json.dumps(result, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode(
                "utf-8"
            )
        )
        > max_decoded_bytes
    ):
        raise ValueError("native perf complete document exceeds decoded-byte budget")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--pid", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-samples", type=int, default=MAX_SAMPLES)
    parser.add_argument("--max-decoded-bytes", type=int, default=MAX_DECODED_BYTES)
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise ValueError("native profile output must be absent; overlays are forbidden")
        result = decode(
            args.input,
            args.pid,
            max_samples=args.max_samples,
            max_decoded_bytes=args.max_decoded_bytes,
        )
        with args.output.open("x", encoding="utf-8") as output:
            output.write(
                json.dumps(result, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
            )
        return 0
    except (OSError, ValueError, ImportError, AttributeError, TypeError, RuntimeError) as error:
        print("native host-profile decode: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
