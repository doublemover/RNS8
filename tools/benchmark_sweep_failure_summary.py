#!/usr/bin/env python3
"""Print compact failure and mismatch summaries for benchmark sweep outputs."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import subprocess
from typing import Any

from evidence_database_lib.isa import load_isa_index, lookup_isa_resources

from benchmark_sweep_lib.capture_metadata import backend_family_id, backend_id, capture_contract_key
from benchmark_sweep_lib.review import capture_checksum


REFERENCE_BACKEND_FAMILIES = {"cpu-reference", "wrap64-byte-limb", "hip-direct"}
NON_ACTIONABLE_BLOCKERS = {"not_accelerator_backend", "scenario_scope_not_autotune_promotable"}
PROMOTABLE_SCOPES = {None, "release_review_candidate"}
DEFAULT_MAX_ROUTE_ROWS = 40


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


def _relative_capture(out: Path, value: Any) -> str:
    if not value:
        return "unknown"
    path = Path(str(value))
    try:
        return str(path.relative_to(out))
    except ValueError:
        return str(path)


def _actionable_blockers(candidate: dict[str, Any]) -> list[str]:
    scope = candidate.get("scenario_promotion_scope")
    if scope not in PROMOTABLE_SCOPES:
        return []
    if candidate.get("accelerator_backend") is not True:
        return []
    blockers = candidate.get("promotion_blockers")
    if not isinstance(blockers, list):
        return []
    return [str(item) for item in blockers if str(item) not in NON_ACTIONABLE_BLOCKERS]


def _shape_text(group: dict[str, Any]) -> str:
    shape = group.get("shape")
    if isinstance(shape, dict):
        return f"{shape.get('m')}x{shape.get('n')}x{shape.get('k')}"
    return "unknown"


def _candidate_histogram(
    candidate: dict[str, Any],
    isa_index: dict[str, list[dict[str, Any]]] | None,
) -> dict[str, Any]:
    histogram = candidate.get("matrix_instruction_histogram")
    if isinstance(histogram, dict) and histogram:
        return histogram
    if isa_index is None:
        return {}
    source = candidate.get("source_metadata")
    target = source.get("target_id") if isinstance(source, dict) else None
    resources = lookup_isa_resources(isa_index, candidate.get("backend"), target)
    fallback = resources.get("isa_matrix_instruction_histogram")
    return fallback if isinstance(fallback, dict) else {}


def _histogram_text(
    candidate: dict[str, Any],
    isa_index: dict[str, list[dict[str, Any]]] | None = None,
) -> str:
    histogram = _candidate_histogram(candidate, isa_index)
    if not histogram:
        return "none"
    items = sorted((str(key), value) for key, value in histogram.items() if isinstance(value, int) and value > 0)
    if not items:
        return "none"
    return ",".join(f"{key}:{value}" for key, value in items[:8])


def _route_line(
    out: Path,
    label: str,
    group: dict[str, Any],
    candidate: dict[str, Any],
    isa_index: dict[str, list[dict[str, Any]]] | None = None,
) -> str:
    bottleneck = candidate.get("bottleneck")
    bottleneck_text = "unknown"
    if isinstance(bottleneck, dict):
        bottleneck_text = f"{bottleneck.get('class')}:{bottleneck.get('phase')}"
    return (
        "  "
        f"{label} "
        f"backend={candidate.get('backend')} "
        f"semantics={group.get('semantics')} "
        f"shape={_shape_text(group)} "
        f"kernel={candidate.get('selected_kernel')} "
        f"e2e={candidate.get('median_end_to_end_us')} "
        f"vs_direct={candidate.get('speedup_vs_direct_hip')} "
        f"primary_loss={candidate.get('primary_loss_phase_vs_direct_hip')} "
        f"bottleneck={bottleneck_text} "
        f"matrix_isa={_histogram_text(candidate, isa_index)} "
        f"capture={_relative_capture(out, candidate.get('capture'))}"
    )


def build_summary(out: Path, *, max_route_rows: int = DEFAULT_MAX_ROUTE_ROWS) -> list[str]:
    lines = [f"HEAD {_git_head()}", f"OUT {out}"]
    isa_index = load_isa_index([out])

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
    actionable_counts: Counter[str] = Counter()
    actionable_rows: list[tuple[str, str, dict[str, Any], dict[str, Any]]] = []
    production_routes: list[tuple[dict[str, Any], dict[str, Any]]] = []
    accelerator_routes: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for path in out.rglob("review_report.json"):
        report = _load_json(path)
        for group in report.get("groups", []):
            production = group.get("fastest_production_route")
            if isinstance(production, dict):
                production_routes.append((group, production))
            accelerator = group.get("fastest_accelerator_route")
            if isinstance(accelerator, dict):
                accelerator_routes.append((group, accelerator))
            for candidate in group.get("candidates", []):
                blocker_counts.update(candidate.get("promotion_blockers", []))
                blockers = _actionable_blockers(candidate)
                if blockers:
                    actionable_counts.update(blockers)
                    actionable_rows.append((str(path.relative_to(out)), str(group.get("contract_key", "")), group, candidate))
    lines.append("REVIEW_BLOCKER_COUNTS")
    for blocker, count in blocker_counts.most_common():
        lines.append(f"{blocker} {count}")
    lines.append("ACTIONABLE_PROMOTION_BLOCKER_COUNTS")
    if actionable_counts:
        for blocker, count in actionable_counts.most_common():
            lines.append(f"{blocker} {count}")
    else:
        lines.append("none 0")
    lines.append(f"ACTIONABLE_PROMOTION_CANDIDATES {len(actionable_rows)}")
    for _report_path, _contract_key, group, candidate in actionable_rows:
        blockers = ",".join(_actionable_blockers(candidate))
        lines.append(
            "  "
            f"{candidate.get('backend')} "
            f"semantics={group.get('semantics')} "
            f"shape={_shape_text(group)} "
            f"kernel={candidate.get('selected_kernel')} "
            f"e2e={candidate.get('median_end_to_end_us')} "
            f"vs_direct={candidate.get('speedup_vs_direct_hip')} "
            f"vs_vector={candidate.get('speedup_vs_vector_alu')} "
            f"blockers={blockers} "
            f"capture={_relative_capture(out, candidate.get('capture'))}"
        )
    lines.append(f"FASTEST_PRODUCTION_ROUTES {len(production_routes)}")
    for group, candidate in production_routes[:max_route_rows]:
        lines.append(_route_line(out, "production", group, candidate, isa_index))
    if len(production_routes) > max_route_rows:
        lines.append(f"  ... {len(production_routes) - max_route_rows} more")
    lines.append(f"FASTEST_ACCELERATOR_ROUTES {len(accelerator_routes)}")
    for group, candidate in accelerator_routes[:max_route_rows]:
        lines.append(_route_line(out, "accelerator", group, candidate, isa_index))
    if len(accelerator_routes) > max_route_rows:
        lines.append(f"  ... {len(accelerator_routes) - max_route_rows} more")
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", nargs="?", type=Path, help="sweep output directory; defaults to latest temp/cdna-*mi300x-*")
    parser.add_argument("--temp-root", type=Path, default=Path("temp"))
    parser.add_argument("--max-route-rows", type=int, default=DEFAULT_MAX_ROUTE_ROWS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = args.out if args.out is not None else _latest_cdna_out(args.temp_root)
    for line in build_summary(out, max_route_rows=args.max_route_rows):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
