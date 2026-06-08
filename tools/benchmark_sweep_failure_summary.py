#!/usr/bin/env python3
"""Print compact failure and mismatch summaries for benchmark sweep outputs."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import subprocess
from typing import Any

from benchmark_sweep_lib.capture_metadata import backend_family_id, backend_id, capture_contract_key
from benchmark_sweep_lib.review import capture_checksum


REFERENCE_BACKEND_FAMILIES = {"cpu-reference", "wrap64-byte-limb", "hip-direct"}


def _latest_cdna_out(root: Path) -> Path:
    candidates = sorted(root.glob("cdna-*mi300x-*"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise SystemExit(f"{root}: no cdna-*mi300x-* sweep directories found")
    return candidates[-1]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _scenario_capture_paths(out: Path) -> list[Path]:
    return sorted(
        path
        for path in out.rglob("rank-scenarios/*/scenarios/**/*.json")
        if not path.name.endswith(".failed.json")
    )


def _reference_checksum(rows: list[tuple[str, Any, Path]]) -> tuple[str | None, Any]:
    for backend, checksum, _path in rows:
        if backend_family_id(backend) in REFERENCE_BACKEND_FAMILIES and checksum is not None:
            return backend, checksum
    for backend, checksum, _path in rows:
        if checksum is not None:
            return backend, checksum
    return None, None


def build_summary(out: Path) -> list[str]:
    lines = [f"HEAD {_git_head()}", f"OUT {out}"]

    failed = sorted(out.rglob("*.failed.json"))
    lines.append(f"FAILED_CAPTURES {len(failed)}")
    for path in failed:
        payload = _load_json(path)
        stderr = str(payload.get("stderr", "")).strip().replace("\n", " | ")
        lines.append(f"{path.relative_to(out)}: {stderr}")

    groups: dict[str, list[tuple[str, Any, Path]]] = defaultdict(list)
    for path in _scenario_capture_paths(out):
        payload = _load_json(path)
        groups[capture_contract_key(payload)].append((backend_id(payload), capture_checksum(payload), path))

    mismatches: list[tuple[str, str | None, Any, list[tuple[str, Any, Path]]]] = []
    for key, rows in groups.items():
        reference_backend, reference = _reference_checksum(rows)
        if reference is None:
            continue
        bad = [
            (backend, checksum, path)
            for backend, checksum, path in rows
            if checksum is not None and checksum != reference
        ]
        if bad:
            mismatches.append((key, reference_backend, reference, bad))

    lines.append(f"CHECKSUM_MISMATCH_GROUPS {len(mismatches)}")
    for key, reference_backend, reference, bad in sorted(mismatches):
        lines.append(f"GROUP {key}")
        lines.append(f"  ref_backend={reference_backend} ref={reference}")
        for backend, checksum, path in bad:
            lines.append(f"  {backend}\t{checksum}\t{path.relative_to(out)}")

    blocker_counts: Counter[str] = Counter()
    for path in out.rglob("review_report.json"):
        report = _load_json(path)
        for group in report.get("groups", []):
            for candidate in group.get("candidates", []):
                blocker_counts.update(candidate.get("promotion_blockers", []))
    lines.append("REVIEW_BLOCKER_COUNTS")
    for blocker, count in blocker_counts.most_common():
        lines.append(f"{blocker} {count}")
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", nargs="?", type=Path, help="sweep output directory; defaults to latest temp/cdna-*mi300x-*")
    parser.add_argument("--temp-root", type=Path, default=Path("temp"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = args.out if args.out is not None else _latest_cdna_out(args.temp_root)
    for line in build_summary(out):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
