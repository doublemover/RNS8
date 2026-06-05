#!/usr/bin/env python3
"""Summarize release-gate and verification-amortization benchmark evidence."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from benchmark_schema import load_capture, validate_capture


DEFAULT_OUT_DIR = Path("temp") / "release-gate-reports"
RELEASE_MIN_WARMUPS = 3
RELEASE_MIN_REPEATS = 9


def backend_id(capture: dict[str, Any]) -> str:
    return str(capture.get("backend_selected") or capture.get("selected_backend") or capture.get("backend") or "unknown")


def required_baselines(semantics: Any) -> list[str]:
    if semantics in {"bounded_i64", "bounded_u64"}:
        return ["cpu-reference", "hip-direct", "hip-vector-alu-int64"]
    if semantics in {"finite_ring_u8", "finite_field_u8"}:
        return ["cpu-reference", "hip-direct"]
    if semantics in {"exact_wide_signed", "exact_wide_unsigned"}:
        return ["cpu-reference", "hip-direct"]
    if semantics == "wrap_u64_mod_2_64":
        return ["wrap64-byte-limb", "hip-direct"]
    return []


def release_review_capture(capture: dict[str, Any]) -> bool:
    return (
        isinstance(capture.get("warmups"), int)
        and isinstance(capture.get("repeats"), int)
        and capture["warmups"] >= RELEASE_MIN_WARMUPS
        and capture["repeats"] >= RELEASE_MIN_REPEATS
    )


def gate_name(capture: dict[str, Any]) -> str:
    gate = capture.get("release_gate") if isinstance(capture.get("release_gate"), dict) else {}
    return str(gate.get("name") or "ungated")


def group_key(capture: dict[str, Any]) -> tuple[Any, ...]:
    return (
        gate_name(capture),
        capture.get("semantics"),
        capture.get("finite_modulus"),
        capture.get("m"),
        capture.get("n"),
        capture.get("k"),
    )


def _load(path: Path) -> dict[str, Any]:
    capture = load_capture(path)
    validate_capture(capture, path)
    capture["_path"] = str(path)
    return capture


def build_report(paths: list[Path]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    blockers: dict[str, int] = {}
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for path in paths:
        capture = _load(path)
        gate = capture.get("release_gate") if isinstance(capture.get("release_gate"), dict) else {}
        amortization = (
            capture.get("verification_amortization")
            if isinstance(capture.get("verification_amortization"), dict)
            else {}
        )
        for blocker in gate.get("blockers", []) if isinstance(gate.get("blockers"), list) else []:
            blockers[blocker] = blockers.get(blocker, 0) + 1
        rows.append(
            {
                "capture_path": str(path),
                "backend": backend_id(capture),
                "semantics": capture.get("semantics"),
                "shape": [capture.get("m"), capture.get("n"), capture.get("k")],
                "gate": gate.get("name"),
                "review_status": gate.get("review_status"),
                "cache_eligible": gate.get("cache_eligible", False),
                "blockers": gate.get("blockers", []),
                "release_review_capture": release_review_capture(capture),
                "verification_policy": amortization.get("policy"),
                "final_exact_comparison_required": amortization.get("final_exact_comparison_required", True),
            }
        )
        grouped[group_key(capture)].append(capture)

    groups = []
    for key, captures in sorted(grouped.items(), key=lambda item: tuple("" if value is None else str(value) for value in item[0])):
        gate, semantics, finite_modulus, m, n, k = key
        backends = sorted({backend_id(capture) for capture in captures})
        required = required_baselines(semantics)
        missing = [backend for backend in required if backend not in backends]
        review_statuses = sorted(
            {
                str(capture.get("release_gate", {}).get("review_status"))
                for capture in captures
                if isinstance(capture.get("release_gate"), dict)
                and capture.get("release_gate", {}).get("review_status") is not None
            }
        )
        gate_blockers = sorted(
            {
                str(blocker)
                for capture in captures
                if isinstance(capture.get("release_gate"), dict)
                for blocker in capture.get("release_gate", {}).get("blockers", [])
                if isinstance(blocker, str)
            }
        )
        if missing:
            blockers["missing_required_baselines"] = blockers.get("missing_required_baselines", 0) + 1
        groups.append(
            {
                "gate": gate,
                "semantics": semantics,
                "finite_modulus": finite_modulus,
                "shape": [m, n, k],
                "capture_count": len(captures),
                "backends": backends,
                "required_baselines": required,
                "missing_required_baselines": missing,
                "required_baselines_complete": not missing,
                "release_review_captures_complete": all(release_review_capture(capture) for capture in captures),
                "review_statuses": review_statuses,
                "cache_eligible_rows": sum(
                    1
                    for capture in captures
                    if isinstance(capture.get("release_gate"), dict)
                    and capture.get("release_gate", {}).get("cache_eligible") is True
                ),
                "blockers": gate_blockers + ([] if not missing else ["missing_required_baselines"]),
            }
        )
    return {
        "schema": "rns8_release_gate_report_v2",
        "capture_count": len(rows),
        "group_count": len(groups),
        "rows": rows,
        "groups": groups,
        "blocker_counts": dict(sorted(blockers.items())),
        "policy": "release_gate_reports_are_review_inputs_not_raw_performance_claims",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", nargs="+", type=Path)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    report = build_report(args.captures)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "release-gate-report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
