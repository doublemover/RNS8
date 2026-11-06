"""Bounded PERFILE2 clock/loss/sample custody beside the native perf decoder.

This reads Linux's file/event ABI, not symbols or DWARF. Native perf remains the
symbol decoder. Loss records and cumulative counters are separate observations;
their counts must never be added into an invented total.
"""

from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path
import struct

SCHEMA = "rns8_perf_recording_contract_v1"
LINUX_REVISION = "47f05f71ad988c3180bdba807151e6e86ed75aa3"
SOURCES = [
    f"https://github.com/torvalds/linux/blob/{LINUX_REVISION}/" + path
    for path in (
        "include/uapi/linux/perf_event.h",
        "tools/perf/util/header.h",
        "tools/perf/util/header.c",
        "tools/lib/perf/include/perf/event.h",
    )
]
MAX_RECORDS = 1_000_000
MAX_COUNTERS = 100_000
SAMPLE_IP, SAMPLE_TID, SAMPLE_TIME, SAMPLE_READ = 1, 2, 4, 16
SAMPLE_CPU, SAMPLE_PERIOD, SAMPLE_IDENTIFIER = 128, 256, 65536
REQUIRED_SAMPLE_FIELDS = SAMPLE_IP | SAMPLE_TID | SAMPLE_TIME | SAMPLE_CPU | SAMPLE_PERIOD
KNOWN_RECORDS = set(range(1, 23)) | {67, 68, 69, 73, 74, 78, 79, 82}


class SampleWitness:
    """Bounded sample multisets; native perf may time-sort separate ring buffers."""

    def __init__(self):
        self.threads = {}
        self.count = 0

    def add(self, pid: int, tid: int, time_ns: int, period: int, ip: int, cpu: int):
        for value, bits in ((pid, 32), (tid, 32), (time_ns, 64), (period, 64), (ip, 64), (cpu, 32)):
            if type(value) is not int or not 0 <= value < 1 << bits:
                raise ValueError("invalid raw/decoded perf sample field")
        if not period:
            raise ValueError("perf sample period is zero")
        if not pid or not tid:
            raise ValueError("perf user sample PID/TID is zero")
        self.count += 1
        if self.count > 100_000:
            raise ValueError("perf sample witness budget exhausted")
        key = (pid, tid)
        if key not in self.threads:
            if len(self.threads) >= MAX_COUNTERS:
                raise ValueError("perf sample thread budget exhausted")
            self.threads[key] = dict(count=0, first=time_ns, last=time_ns, periods=0, payloads=[])
        row = self.threads[key]
        row["count"] += 1
        row["first"] = min(row["first"], time_ns)
        row["last"] = max(row["last"], time_ns)
        row["periods"] += period
        row["payloads"].append(struct.pack("<IIQQQQ", pid, tid, time_ns, period, ip, cpu))

    def public(self, pid: int | None = None) -> list[dict]:
        return [
            dict(
                pid=process,
                tid=thread,
                samples=row["count"],
                first_time_ns=row["first"],
                last_time_ns=row["last"],
                cpu_clock_period_ns=row["periods"],
                payload_sha256=hashlib.sha256(b"".join(sorted(row["payloads"]))).hexdigest(),
            )
            for (process, thread), row in sorted(self.threads.items())
            if pid is None or process == pid
        ]


def decoded_sample_witness(rows: list[dict]) -> list[dict]:
    if not isinstance(rows, list) or len(rows) > 100_000:
        raise ValueError("bounded decoded perf sample rows required")
    witness = SampleWitness()
    for row in rows:
        witness.add(row["pid"], row["tid"], row["time_ns"], row["period"], row["ip"], row["cpu"])
    return witness.public()


def validate_decoded_samples(
    recording: dict, rows: list[dict], *, pid: int, observed: int, excluded: int
) -> None:
    if (
        type(pid) is not int
        or not 0 < pid < 1 << 32
        or type(observed) is not int
        or type(excluded) is not int
        or excluded < 0
        or not isinstance(rows, list)
        or not rows
        or len(rows) > 100_000
    ):
        raise ValueError("bounded selected-process sample coverage required")
    expected = [row for row in recording["sample_witness"] if row["pid"] == pid]
    if (
        observed != recording["sample_records"]
        or excluded != observed - len(rows)
        or any(not isinstance(row, dict) or row.get("pid") != pid for row in rows)
        or decoded_sample_witness(rows) != expected
    ):
        raise ValueError("decoded perf sample fields/coverage differ from the raw recording")


def read_recording(
    path: Path, *, max_input_bytes: int = 256 * 1024**2, max_samples: int = 100_000
) -> dict:
    if (
        type(max_input_bytes) is not int
        or not 1 <= max_input_bytes <= 8 * 1024**3
        or type(max_samples) is not int
        or not 1 <= max_samples <= 100_000
    ):
        raise ValueError("invalid perf recording audit budget")
    path = Path(path).resolve(strict=True)
    size = path.stat().st_size
    if not path.is_file() or not 104 <= size <= max_input_bytes:
        raise ValueError("perf recording is outside the bounded regular-file envelope")
    intervals = []
    with path.open("rb") as source:

        def read_at(offset: int, length: int) -> bytes:
            if offset < 0 or length < 0 or offset > size or length > size - offset:
                raise ValueError("perf section or record extends beyond the file")
            source.seek(offset)
            value = source.read(length)
            if len(value) != length:
                raise ValueError("perf recording changed or contains a truncated section")
            return value

        def section(offset: int, length: int, name: str):
            if length:
                if offset < 104 or offset > size or length > size - offset:
                    raise ValueError("invalid perf " + name + " section")
                intervals.append((offset, offset + length, name))

        header = read_at(0, 104)
        if header[:8] != b"PERFILE2":
            raise ValueError("only little-endian PERFILE2 regular files are supported")
        values = struct.unpack("<12Q", header[8:])
        header_size, attr_size, attrs_offset, attrs_size, data_offset, data_size = values[:6]
        if header_size != 104 or not 112 <= attr_size <= 1040 or attrs_size != attr_size:
            raise ValueError("one complete explicit perf event attribute is required")
        section(attrs_offset, attrs_size, "attributes")
        section(data_offset, data_size, "data")
        if not data_size:
            raise ValueError("empty perf recording data section")
        section(values[6], values[7], "legacy_event_types")
        features = sum(word << (64 * index) for index, word in enumerate(values[8:]))
        if features & ((1 << 18) | (1 << 19) | (1 << 24) | (1 << 27)):
            raise ValueError(
                "AUX/stat/directory/compressed perf recording needs a separate adapter"
            )
        if features & 1 or features >> 35:
            raise ValueError("unsupported perf header feature bits")
        feature_ids = [bit for bit in range(35) if features & (1 << bit)]
        feature_table = data_offset + data_size
        section(feature_table, len(feature_ids) * 16, "feature_table")
        feature_sections = {}
        for index, feature in enumerate(feature_ids):
            offset, length = struct.unpack("<QQ", read_at(feature_table + index * 16, 16))
            section(offset, length, "feature_" + str(feature))
            feature_sections[feature] = (offset, length)

        attr = read_at(attrs_offset, attr_size)
        kind, declared_size = struct.unpack_from("<II", attr)
        config, setting, sample_type, read_format, flags = struct.unpack_from("<5Q", attr, 8)
        if (
            kind != 1
            or config != 0
            or not setting
            or not 96 <= declared_size <= attr_size - 16
            or not flags & (1 << 25)
            or not flags & (1 << 5)
            or sample_type & REQUIRED_SAMPLE_FIELDS != REQUIRED_SAMPLE_FIELDS
            or sample_type >> 25
            or read_format & ~31
        ):
            raise ValueError("unsupported CPU-clock/user-sample/clockid perf event contract")
        clock_id = struct.unpack_from("<i", attr, 92)[0]
        ids_offset, ids_size = struct.unpack_from("<QQ", attr, attr_size - 16)
        if not ids_size or ids_size % 8 or ids_size > MAX_COUNTERS * 8:
            raise ValueError("bounded nonempty perf event ID inventory required")
        section(ids_offset, ids_size, "event_ids")
        ids_bytes = read_at(ids_offset, ids_size)
        ids = {value[0] for value in struct.iter_unpack("<Q", ids_bytes)}
        if 0 in ids or len(ids) != ids_size // 8:
            raise ValueError("duplicate or zero perf event ID")
        ordered = sorted(intervals)
        if any(left[1] > right[0] for left, right in zip(ordered, ordered[1:])):
            raise ValueError("overlapping perf file sections")
        clock = dict(
            attribute_clock_id=clock_id,
            use_clockid=True,
            source="recorded_perf_event_attr_not_requested_cli_clock",
            resolution_ns=None,
            wall_clock_anchor=None,
            cross_clock_join="unbound",
        )
        if 23 in feature_sections:
            offset, length = feature_sections[23]
            if length != 8:
                raise ValueError("unsupported perf clock-resolution feature")
            clock["resolution_ns"] = struct.unpack("<Q", read_at(offset, length))[0]
        if 29 in feature_sections:
            offset, length = feature_sections[29]
            if length != 24:
                raise ValueError("unsupported perf clock-data feature")
            version, recorded_id, wall, monotonic = struct.unpack("<IiQQ", read_at(offset, length))
            if version != 1 or recorded_id != clock_id:
                raise ValueError("perf feature and event clock identities disagree")
            clock["wall_clock_anchor"] = dict(
                version=version,
                clock_id=recorded_id,
                realtime_ns=wall,
                clock_ns=monotonic,
                calibration_or_error_bound_proven=False,
            )
        machine = None
        if 33 in feature_sections:
            offset, length = feature_sections[33]
            if length != 8:
                raise ValueError("unsupported perf machine feature")
            machine, _ = struct.unpack("<II", read_at(offset, length))

        witness, counts, read_counters = SampleWitness(), Counter(), {}
        ring_lost = lost_samples = sample_count = record_count = 0
        trailer_size = (
            8 * (sample_type & (2 | 4 | 64 | 512 | 128 | SAMPLE_IDENTIFIER)).bit_count()
            if flags & (1 << 18)
            else 0
        )

        def read_values(payload: bytes, start: int, pid: int, tid: int):
            cursor = start

            def word():
                nonlocal cursor
                if cursor + 8 > len(payload):
                    raise ValueError("truncated perf read-format payload")
                value = struct.unpack_from("<Q", payload, cursor)[0]
                cursor += 8
                return value

            group = bool(read_format & 8)
            nr = word() if group else 1
            if nr != 1:
                raise ValueError("single-event perf read group requires exactly one value")
            if not group:
                word()  # Counter value; not a sampled duration or a loss count.
            for bit in (1, 2):
                if read_format & bit:
                    word()
            if group:
                word()
            identity = word() if read_format & 4 else None
            if identity is not None and identity not in ids:
                raise ValueError("perf counter uses an undeclared event ID")
            if read_format & 16:
                lost = word()
                key = (pid, tid, identity)
                if key not in read_counters and len(read_counters) >= MAX_COUNTERS:
                    raise ValueError("perf read-counter budget exhausted")
                read_counters[key] = max(read_counters.get(key, 0), lost)
            return cursor

        offset, end = data_offset, data_offset + data_size
        while offset < end:
            if record_count >= MAX_RECORDS:
                raise ValueError("perf recording record budget exhausted")
            if end - offset < 8:
                raise ValueError("truncated perf record header")
            record_type, _, record_size = struct.unpack("<IHH", read_at(offset, 8))
            if record_size < 8 or record_size > end - offset:
                raise ValueError("invalid or truncated perf record extent")
            if record_type in {11, 12, 21, 64, 66, 70, 71, 72, 75, 76, 77, 80, 81, 83, 84, 85, 86}:
                raise ValueError("unsupported perf dynamic/AUX/stat/compressed record")
            if record_type not in KNOWN_RECORDS:
                raise ValueError("unsupported unknown perf record type")
            record_count += 1
            counts[record_type] += 1
            payload = read_at(offset + 8, record_size - 8) if record_type in {2, 8, 9, 13} else b""
            if record_type == 2:
                if len(payload) != 16 + trailer_size:
                    raise ValueError("truncated perf lost-event record")
                identity, lost = struct.unpack_from("<QQ", payload)
                if identity not in ids:
                    raise ValueError("perf loss record uses an undeclared event ID")
                ring_lost += lost
            elif record_type == 13:
                if len(payload) != 8 + trailer_size:
                    raise ValueError("truncated perf lost-samples record")
                lost_samples += struct.unpack_from("<Q", payload)[0]
            elif record_type == 8:
                if len(payload) < 8:
                    raise ValueError("truncated perf read PID/TID")
                pid, tid = struct.unpack_from("<II", payload)
                consumed = read_values(payload, 8, pid, tid)
                if len(payload) != consumed + trailer_size:
                    raise ValueError("perf read record has an invalid sample-ID trailer")
            elif record_type == 9:
                sample_count += 1
                if sample_count > max_samples:
                    raise ValueError("perf recording sample budget exhausted")
                cursor = 8 if sample_type & SAMPLE_IDENTIFIER else 0
                fields = {}
                for bit, name in (
                    (1, "ip"),
                    (2, "tid"),
                    (4, "time"),
                    (8, "addr"),
                    (64, "id"),
                    (512, "stream"),
                    (128, "cpu"),
                    (256, "period"),
                ):
                    if sample_type & bit:
                        if cursor + 8 > len(payload):
                            raise ValueError("truncated perf sample fixed fields")
                        fields[name] = struct.unpack_from("<Q", payload, cursor)[0]
                        cursor += 8
                pid, tid = fields["tid"] & 0xFFFFFFFF, fields["tid"] >> 32
                cpu = fields["cpu"] & 0xFFFFFFFF
                if fields["cpu"] >> 32:
                    raise ValueError("perf CPU sample reserved bits are nonzero")
                if (
                    sample_type & SAMPLE_IDENTIFIER
                    and struct.unpack_from("<Q", payload)[0] not in ids
                ):
                    raise ValueError("perf sample identifier is undeclared")
                if "id" in fields and fields["id"] not in ids:
                    raise ValueError("perf sample event ID is undeclared")
                witness.add(pid, tid, fields["time"], fields["period"], fields["ip"], cpu)
                if sample_type & SAMPLE_READ:
                    read_values(payload, cursor, pid, tid)
            offset += record_size
    if path.stat().st_size != size:
        raise ValueError("perf recording size changed during record audit")
    loss_present = bool(ring_lost or lost_samples or any(read_counters.values()))
    blockers = ["recorded_loss"] if loss_present else []
    if counts[5]:
        blockers.append("recorded_throttling")
    status = (
        "recorded_loss_not_complete_samples"
        if loss_present
        else "recorded_throttling_not_complete_samples"
        if counts[5]
        else "no_known_loss_observed_complete_record_walk"
    )
    return dict(
        schema=SCHEMA,
        abi_sources=SOURCES,
        file_bytes=size,
        data_bytes=data_size,
        event=dict(
            type=kind,
            config=config,
            frequency_mode=bool(flags & (1 << 10)),
            sample_setting=setting,
            kernel_excluded=True,
        ),
        complete_record_walk=True,
        record_count=record_count,
        record_types={str(key): value for key, value in sorted(counts.items())},
        sample_records=sample_count,
        sample_witness_contract="per_thread_sorted_le_pid32_tid32_time64_period64_ip64_cpu64_v1",
        sample_witness=witness.public(),
        clock=clock,
        recorded_machine=machine,
        event_ids=dict(count=len(ids), payload_sha256=hashlib.sha256(ids_bytes).hexdigest()),
        read_format=read_format,
        sample_type=sample_type,
        loss=dict(
            status=status,
            scope="recording_wide_not_selected_pid",
            ring_lost_events=ring_lost,
            lost_sample_events=lost_samples,
            cumulative_read_maxima=[
                dict(pid=pid, tid=tid, event_id=identity, maximum=value)
                for (pid, tid, identity), value in sorted(read_counters.items())
            ],
            counts_are_additive=False,
            complete_loss_free_acquisition_proven=False,
        ),
        blockers=blockers,
        operation_range="unbound",
        production_latency=False,
        promotion_allowed=False,
    )
