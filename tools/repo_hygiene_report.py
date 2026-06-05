#!/usr/bin/env python3
"""Report RNS8 cleanup hotspots without mutating the repository."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from metadata_registry import load_registry, validate_registry


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LARGE_FILE_BYTES = 100_000
CODE_SUFFIXES = {".c", ".cc", ".cpp", ".h", ".hpp", ".hip", ".py", ".cmake", ".txt"}
SOURCE_PREFIXES = ("src/", "include/", "benchmarks/", "tools/", "tests/", "cmake/")
IGNORED_TRACKED_PATHS = {
    "tools/metadata_registry_constants.py",
}
CURRENTNESS_HELPER_FUNCTIONS = {
    "clear_native_current",
    "clear_residue_current",
    "clear_byte_limb_current",
    "mark_host_residues_current",
    "mark_device_residues_current",
    "mark_host_byte_limbs_current",
    "mark_device_byte_limbs_current",
    "mark_output_device_native_current",
}
RAW_HIP_RESOURCE_WRAPPER_PATHS = {
    "src/core/hip_resources.hpp",
    "tools/repo_hygiene_report.py",
}
CURRENTNESS_WRITE_RE = re.compile(
    r"(?:->|\.)"
    r"(host_residues_current|device_residues_current|host_byte_limbs_current|device_byte_limbs_current|"
    r"host_native_current|device_native_current)\s*="
)
RAW_HIP_RESOURCE_RE = re.compile(
    r"\b(hipMalloc|hipFree|hipHostMalloc|hipHostFree|hipEventCreate|hipEventDestroy|"
    r"hipStreamCreate|hipStreamCreateWithFlags|hipStreamDestroy)\b"
)


def tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return [REPO_ROOT / line.strip() for line in result.stdout.splitlines() if line.strip()]


def repo_relative(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def source_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")


def large_files(files: list[Path], threshold: int) -> list[dict[str, Any]]:
    rows = []
    for path in files:
        size = path.stat().st_size
        if size >= threshold:
            rows.append({"path": repo_relative(path), "bytes": size})
    return sorted(rows, key=lambda item: (-item["bytes"], item["path"]))


def registry_metadata_strings() -> set[str]:
    registry = load_registry()
    validate_registry(registry)
    strings: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, str) and value:
            strings.add(value)
        elif isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)

    for document in registry.files.values():
        visit(document)
    return strings


def duplicate_metadata_strings(files: list[Path], limit: int) -> list[dict[str, Any]]:
    metadata = registry_metadata_strings()
    counts: dict[str, Counter[str]] = {value: Counter() for value in metadata}
    for path in files:
        rel = repo_relative(path)
        if rel in IGNORED_TRACKED_PATHS or not rel.startswith(SOURCE_PREFIXES) or path.suffix not in CODE_SUFFIXES:
            continue
        text = source_text(path)
        for value in metadata:
            hits = text.count(value)
            if hits:
                counts[value][rel] += hits
    rows = []
    for value, per_file in counts.items():
        if len(per_file) >= 2 or sum(per_file.values()) >= 3:
            rows.append(
                {
                    "value": value,
                    "occurrences": sum(per_file.values()),
                    "files": sorted(per_file),
                }
            )
    rows.sort(key=lambda item: (-item["occurrences"], item["value"]))
    return rows[:limit]


def currentness_helper_write(rel: str, lines: list[str], index: int) -> bool:
    if rel != "src/core/api_matrix_workspace.cpp":
        return False
    for line in reversed(lines[max(0, index - 16):index]):
        match = re.match(r"\s*void\s+([A-Za-z0-9_]+)\s*\(", line)
        if match:
            return match.group(1) in CURRENTNESS_HELPER_FUNCTIONS
    return False


def raw_hip_wrapper_call(rel: str, _lines: list[str], _index: int) -> bool:
    return rel in RAW_HIP_RESOURCE_WRAPPER_PATHS


def line_findings(
    files: list[Path],
    pattern: re.Pattern[str],
    limit: int,
    skip_match: Any | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for path in files:
        rel = repo_relative(path)
        if not rel.startswith(SOURCE_PREFIXES) or path.suffix not in CODE_SUFFIXES:
            continue
        lines = source_text(path).splitlines()
        for index, line in enumerate(lines, start=1):
            match = pattern.search(line)
            if match:
                if pattern is CURRENTNESS_WRITE_RE and line.strip().startswith("out->"):
                    continue
                if skip_match is not None and skip_match(rel, lines, index - 1):
                    continue
                rows.append({"path": rel, "line": index, "match": match.group(1), "text": line.strip()})
                if len(rows) >= limit:
                    return rows
    return rows


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    files = tracked_files()
    large = large_files(files, args.large_file_bytes)
    duplicated = duplicate_metadata_strings(files, args.limit)
    currentness = line_findings(files, CURRENTNESS_WRITE_RE, args.limit, skip_match=currentness_helper_write)
    raw_hip = line_findings(files, RAW_HIP_RESOURCE_RE, args.limit, skip_match=raw_hip_wrapper_call)
    return {
        "large_file_threshold_bytes": args.large_file_bytes,
        "large_files": large,
        "duplicated_metadata_strings": duplicated,
        "direct_currentness_flag_writes": currentness,
        "raw_hip_resource_calls": raw_hip,
        "finding_counts": {
            "large_files": len(large),
            "duplicated_metadata_strings": len(duplicated),
            "direct_currentness_flag_writes_reported": len(currentness),
            "raw_hip_resource_calls_reported": len(raw_hip),
        },
    }


def print_markdown(report: dict[str, Any]) -> None:
    print("# RNS8 Repo Hygiene Report")
    print()
    print("| Finding | Count |")
    print("|---|---:|")
    for key, value in report["finding_counts"].items():
        print(f"| {key} | {value} |")
    print()
    print("## Large Files")
    for item in report["large_files"][:20]:
        print(f"- {item['path']}: {item['bytes']} bytes")
    print()
    print("## Duplicated Metadata Strings")
    for item in report["duplicated_metadata_strings"][:20]:
        print(f"- `{item['value']}`: {item['occurrences']} occurrences across {len(item['files'])} files")
    print()
    print("## Direct Currentness Flag Writes")
    for item in report["direct_currentness_flag_writes"][:20]:
        print(f"- {item['path']}:{item['line']} `{item['match']}`")
    print()
    print("## Raw HIP Resource Calls")
    for item in report["raw_hip_resource_calls"][:20]:
        print(f"- {item['path']}:{item['line']} `{item['match']}`")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--large-file-bytes", type=int, default=DEFAULT_LARGE_FILE_BYTES)
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--fail-on-findings", action="store_true")
    args = parser.parse_args()
    report = build_report(args)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print_markdown(report)
    if args.fail_on_findings and any(report["finding_counts"].values()):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
