"""Bound current first-party source files behind native debug line observations."""

from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path

from benchmark_request import file_digest

ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_BYTES = 8 * 1024**2
MAX_TOTAL_BYTES = 64 * 1024**2
MAX_FILES = 256


def source_inventory(samples, *, root=ROOT):
    root = root.resolve(strict=True)
    requested, counts, aliases, paths = {}, Counter(), {}, {}
    selected = 0
    for sample in samples:
        for role, observation in [
            ("leaf", sample),
            *(("callchain", frame) for frame in sample["callchain"]),
        ]:
            selected += 1
            if selected > 500_000:
                raise ValueError("native source attribution exceeds frame budget")
            source = observation.get("source")
            if source is not None and type(source) is not dict:
                raise ValueError("typed native source observation required")
            if source is None or source.get("path") is None or source.get("line") == 0:
                counts[(role, "native_source_location_missing")] += 1
                continue
            name, line, text = source.get("path"), source.get("line"), source.get("text")
            if (
                type(name) is not str
                or len(name) > 4096
                or "\0" in name
                or type(line) is not int
                or not 1 <= line < 1 << 32
            ):
                raise ValueError("invalid native debug source location")
            if text is not None and (type(text) is not str or len(text.encode("utf-8")) > 4096):
                raise ValueError("invalid native source text")
            if name not in paths:
                if len(paths) >= 4096:
                    raise ValueError("native debug source path inventory exceeds budget")
                path, reason = Path(name), None
                if not path.is_absolute():
                    reason = "relative_debug_path_has_no_verified_compilation_directory"
                elif not path.is_relative_to(root):
                    reason = "outside_first_party_source_root"
                else:
                    try:
                        path = path.resolve(strict=True)
                        if not path.is_relative_to(root):
                            reason = "outside_first_party_source_root"
                        elif not path.is_file():
                            reason = "current_source_file_unavailable"
                    except OSError:
                        reason = "current_source_file_unavailable"
                paths[name] = path, reason
            path, reason = paths[name]
            if reason is not None:
                counts[(role, reason)] += 1
                continue
            aliases[name] = path
            if path not in requested:
                if len(requested) >= MAX_FILES:
                    raise ValueError("current source inventory exceeds file budget")
                requested[path] = Counter()
            requested[path][role, line, text] += 1
    total, files = 0, []
    for path, locations in sorted(requested.items()):
        with path.open("rb") as stream:
            data = stream.read(min(MAX_FILE_BYTES, MAX_TOTAL_BYTES - total) + 1)
        if len(data) > MAX_FILE_BYTES or len(data) > MAX_TOTAL_BYTES - total:
            raise ValueError("current source inventory exceeds byte budget")
        total += len(data)
        digest = hashlib.sha256(data).hexdigest()
        wanted, lines = {line for _, line, _ in locations}, {}
        line, cursor = 1, 0
        while cursor < len(data):
            end = data.find(b"\n", cursor)
            if end < 0:
                end = len(data)
            if line in wanted:
                lines[line] = data[cursor:end]
            cursor, line = end + 1, line + 1
        observations = []
        for (role, line, text), count in sorted(
            locations.items(), key=lambda item: (item[0][0], item[0][1], item[0][2] or "")
        ):
            state = (
                "source_line_outside_current_file"
                if line not in lines
                else "native_source_text_missing"
                if text is None
                else "native_text_matches_current_source_line"
                if text.encode("utf-8") == lines[line]
                else "native_text_differs_from_current_source_line"
            )
            counts[(role, state)] += count
            observations.append(dict(role=role, line=line, state=state, observations=count))
        files.append(dict(path=str(path), sha256=digest, bytes=len(data), locations=observations))
    for name, path in aliases.items():
        if Path(name).resolve(strict=True) != path:
            raise ValueError("native source path changed its resolved target")
    for file in files:
        if file_digest(file["path"]) != file["sha256"]:
            raise ValueError("current source changed during attribution")
    return dict(
        schema="rns8_native_current_source_custody_v1",
        root=str(root),
        files=files,
        current_source_bytes=total,
        selected_frame_observations=selected,
        counts=[
            dict(role=role, state=state, observations=count)
            for (role, state), count in sorted(counts.items())
        ],
        scope="native_debug_line_and_current_source_snapshot_not_compilation_source_checksum",
        compiled_source_contents_verified=False,
        source="https://github.com/torvalds/linux/blob/473f6c8f437b049f8ec015d57cd59bb983b1d85c/tools/perf/util/python.c",
    )
