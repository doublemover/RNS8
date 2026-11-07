#!/usr/bin/env python3
"""Decode bounded process CPU samples through perf's native standalone API."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import platform
import sys
from typing import Any

from benchmark_request import file_digest
from compiled_binary_identity import canonical_digest
from execution_observation import execution_observation
from elf_module_identity import module_inventory
from perf_mapping_contract import frame_custody
from host_source_attribution import source_inventory
from json_evidence import load_evidence_json
from perf_recording_contract import (
    SCHEMA as RECORDING_CONTRACT_SCHEMA,
    read_recording,
    validate_decoded_samples,
)

SCHEMA = "rns8_perf_native_samples_v4"
PRODUCER_CONTRACT = "linux_perf_standalone_session_sample_api_v1"
PRODUCER_SOURCE = (
    "https://github.com/torvalds/linux/blob/master/tools/perf/Documentation/perf-script-python.txt"
)
MAX_SAMPLES = 100_000
MAX_FRAMES = 128
MAX_DECODED_BYTES = 64 * 1024**2
SAMPLE_ATTRIBUTES = (
    "sample_pid",
    "sample_tid",
    "sample_time",
    "sample_cpu",
    "sample_period",
    "sample_ip",
    "machine_pid",
    "evsel",
    "symbol",
    "sym_start",
    "sym_end",
    "sym_offset",
    "dso",
    "dso_long_name",
    "dso_bid",
    "callchain",
    "srccode",
    "map_start",
    "map_end",
    "map_pgoff",
)


def sample_interface(module) -> list[str]:
    sample_type = getattr(module, "sample_event", None)
    return [
        name for name in SAMPLE_ATTRIBUTES if sample_type is None or not hasattr(sample_type, name)
    ]


def capabilities() -> dict:
    result = dict(
        schema="rns8_perf_native_capabilities_v1",
        system=platform.system(),
        machine=platform.machine(),
        kernel=platform.release(),
        cpu_model_observation=platform.processor() or None,
        python_version=platform.python_version(),
        python_cache_tag=sys.implementation.cache_tag,
        producer_contract=PRODUCER_CONTRACT,
        workload_executed=False,
        sample_contract_qualified=False,
        loss_counts_exposed=False,
        loss_counts_scope="native_session_api_only",
        recording_file_audit_contract=RECORDING_CONTRACT_SCHEMA,
        recording_file_audit_qualified=False,
    )
    if result["system"] != "Linux":
        return dict(result, state="unsupported", reason="native Linux perf module required")
    selected = (Path(__file__).resolve().parents[1] / ".python-version").read_text().strip()
    if result["python_version"] != selected:
        return dict(
            result,
            state="unsupported",
            reason="selected interpreter version required for native module acquisition",
        )
    try:
        module = importlib.import_module("perf")
        if not callable(getattr(module, "session", None)) or not callable(
            getattr(module, "data", None)
        ):
            return dict(
                result, state="unsupported", reason="standalone session/data interface absent"
            )
        missing = sample_interface(module)
        if missing:
            return dict(
                result,
                state="unsupported",
                reason="native sample interface incomplete",
                missing_sample_attributes=missing,
            )
        path = Path(module.__file__).resolve(strict=True)
        return dict(
            result, state="available", native_module=dict(path=str(path), sha256=file_digest(path))
        )
    except (ImportError, OSError, AttributeError, TypeError) as error:
        return dict(result, state="unavailable", reason=str(error))


def seal(result: dict, byte_limit: int) -> dict:
    result.pop("identity_sha256", None)
    result["identity_sha256"] = canonical_digest(result)
    if (
        len(
            json.dumps(result, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode(
                "utf-8"
            )
        )
        > byte_limit
    ):
        raise ValueError("native perf complete document exceeds decoded-byte budget")
    return result


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


def source_record(source):
    if source is None:
        return None
    if type(source) is not tuple or len(source) != 3:
        raise ValueError("unexpected native perf source tuple")
    return dict(
        path=text(source[0], "source path"),
        line=integer(source[1], "source line"),
        text=text(source[2], "source text"),
    )


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
                    source=source_record(sample.srccode(node.ip)),
                )
            )
    source = source_record(sample.srccode())
    event = str(sample.evsel)
    # The producer's evsel repr encloses the original event selector.
    if event.startswith("evsel(") and event.endswith(")"):
        event = event[6:-1]
    mapping = {
        name: optional_integer(getattr(sample, name), name)
        for name in ("map_start", "map_end", "map_pgoff")
    }
    values = list(mapping.values())
    if any(value is not None and value >= 1 << 64 for value in values):
        raise ValueError("native perf mapping exceeds the uint64 address ABI")
    ip = integer(sample.sample_ip, "IP")
    start, end, offset = values
    if start is not None and end is not None and offset is not None:
        if not start <= ip < end:
            raise ValueError("native perf leaf IP is outside its resolved mapping")
    return dict(
        pid=integer(sample.sample_pid, "PID", 1),
        tid=integer(sample.sample_tid, "TID", 1),
        time_ns=integer(sample.sample_time, "timestamp"),
        cpu=integer(sample.sample_cpu, "CPU"),
        period=integer(sample.sample_period, "period", 1),
        event=text(event, "event"),
        ip=ip,
        mapping=mapping,
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
        callchain_module_identity="native_paths_with_separate_recorded_mapping_custody",
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
    if pid >= 1 << 32:
        raise ValueError("selected PID exceeds the perf 32-bit process ABI")
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
    recording = read_recording(data, max_input_bytes=max_input_bytes, max_samples=max_samples)
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
    validate_decoded_samples(recording, rows, pid=pid, observed=observed, excluded=excluded)
    if recording["recorded_machine"] is not None and recording["recorded_machine"] != machine:
        raise ValueError("native perf machine differs from the recorded header")
    modules = module_inventory(rows, machine=machine)
    frames = frame_custody(rows, recording, machine=machine)
    sources = source_inventory(rows)
    if file_digest(data) != digest or file_digest(module_path) != module_digest:
        raise ValueError("native perf data or decoder module changed during module inspection")
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
        clock=recording["clock"],
        recording_contract=recording,
        recording_contract_source_sha256=file_digest(
            Path(__file__).with_name("perf_recording_contract.py")
        ),
        leaf_module_custody=modules,
        recorded_frame_custody=frames,
        current_source_custody=sources,
        current_source_custody_source_sha256=file_digest(
            Path(__file__).with_name("host_source_attribution.py")
        ),
        recorded_frame_custody_source_sha256=file_digest(
            Path(__file__).with_name("perf_mapping_contract.py")
        ),
        leaf_module_custody_source_sha256=file_digest(
            Path(__file__).with_name("elf_module_identity.py")
        ),
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
        loss_status=recording["loss"]["status"],
        symbolized_samples=sum(bool(row["symbol"]) for row in rows),
        source_attributed_samples=sum(
            row["source"] is not None and bool(row["source"]["path"]) and row["source"]["line"] > 0
            for row in rows
        ),
        acquisition_verified=False,
        production_latency=False,
        promotion_allowed=False,
    )
    return seal(result, max_decoded_bytes)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capabilities", action="store_true")
    parser.add_argument("--input", type=Path)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--pid", type=int)
    source.add_argument(
        "--capture",
        type=Path,
        help="Use the actual PID and native argv witness from benchmark JSON",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-samples", type=int, default=MAX_SAMPLES)
    parser.add_argument("--max-decoded-bytes", type=int, default=MAX_DECODED_BYTES)
    args = parser.parse_args(argv)
    try:
        if args.capabilities:
            if any(
                value is not None for value in (args.input, args.pid, args.capture, args.output)
            ):
                raise ValueError("capability query does not accept decode inputs/outputs")
            value = capabilities()
            print(json.dumps(value, indent=2))
            return 0 if value["state"] == "available" else 1
        if args.input is None or args.output is None or (args.pid is None and args.capture is None):
            raise ValueError("decode requires input, output and either PID or benchmark capture")
        if args.output.exists():
            raise ValueError("native profile output must be absent; overlays are forbidden")
        capture = None
        if args.capture is not None:
            if not 1 <= args.capture.stat().st_size <= MAX_DECODED_BYTES:
                raise ValueError("benchmark capture exceeds decode budget")
            digest = file_digest(args.capture)
            captured = load_evidence_json(args.capture)
            if not isinstance(captured, dict):
                raise ValueError("native PID binding requires a benchmark capture object")
            execution_observation(captured, required=True)
            args.pid = integer(captured.get("process_id"), "captured benchmark PID", 1)
            capture = dict(path=str(args.capture.resolve()), sha256=digest)
        result = decode(
            args.input,
            args.pid,
            max_samples=args.max_samples,
            max_decoded_bytes=args.max_decoded_bytes,
        )
        if capture is not None:
            if file_digest(args.capture) != capture["sha256"]:
                raise ValueError("benchmark PID binding changed during decode")
            result["profiled_capture"] = capture
            seal(result, args.max_decoded_bytes)
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
