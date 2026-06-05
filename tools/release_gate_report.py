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


def _command_arg(command: list[str], name: str) -> str | None:
    for index, value in enumerate(command[:-1]):
        if value == name:
            return command[index + 1]
    return None


def _command_int_arg(command: list[str], name: str) -> int | None:
    value = _command_arg(command, name)
    if value is None:
        return None
    return int(value)


def _normal_failure_backend(value: str | None) -> str:
    aliases = {
        "cpu": "cpu-reference",
        "hip-vector-alu-int64-runtime": "hip-vector-alu-int64",
        "vector-alu-int64": "hip-vector-alu-int64",
        "vector-alu-int64-runtime": "hip-vector-alu-int64",
    }
    if value is None:
        return "unknown"
    return aliases.get(value, value)


def _normal_failure_semantics(value: str | None) -> str:
    aliases = {
        "bounded-i64": "bounded_i64",
        "bounded-u64": "bounded_u64",
        "bounded_i64": "bounded_i64",
        "bounded_u64": "bounded_u64",
        "exact-wide-signed": "exact_wide_signed",
        "exact-wide-i64": "exact_wide_signed",
        "exact_wide_signed": "exact_wide_signed",
        "exact-wide-unsigned": "exact_wide_unsigned",
        "exact-wide-u64": "exact_wide_unsigned",
        "exact_wide_unsigned": "exact_wide_unsigned",
        "wrap-u64": "wrap_u64_mod_2_64",
        "wrap_u64_mod_2_64": "wrap_u64_mod_2_64",
        "finite-u8-ring": "finite_ring_u8",
        "finite-ring-u8": "finite_ring_u8",
        "finite_ring_u8": "finite_ring_u8",
        "finite-u8-field": "finite_field_u8",
        "finite-field-u8": "finite_field_u8",
        "finite_field_u8": "finite_field_u8",
    }
    if value is None:
        return "unknown"
    return aliases.get(value, value)


def _failure_kind(failure: dict[str, Any]) -> str:
    if failure.get("timed_out") is True:
        return "timeout"
    if failure.get("returncode") not in {None, 0}:
        return "nonzero_exit"
    return "failed_capture"


def _load_failure(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("command"), list):
        raise ValueError(f"{path} is not a benchmark failure record")
    command = [str(item) for item in data["command"]]
    semantics = _normal_failure_semantics(_command_arg(command, "--semantics"))
    backend = _normal_failure_backend(_command_arg(command, "--backend"))
    row = {
        "failure_path": str(path),
        "backend": backend,
        "semantics": semantics,
        "finite_modulus": _command_int_arg(command, "--modulus"),
        "shape": [
            _command_int_arg(command, "--m"),
            _command_int_arg(command, "--n"),
            _command_int_arg(command, "--k"),
        ],
        "gate": _command_arg(command, "--release-gate") or "ungated",
        "timed_out": data.get("timed_out") is True,
        "timeout_seconds": data.get("timeout_seconds"),
        "returncode": data.get("returncode"),
        "warmups": _command_int_arg(command, "--warmups"),
        "repeats": _command_int_arg(command, "--repeats"),
        "seed": _command_int_arg(command, "--seed"),
        "failure_kind": _failure_kind(data),
        "stdout_present": bool(data.get("stdout")),
        "stderr_present": bool(data.get("stderr")),
    }
    return row


def _failure_group_key(failure: dict[str, Any]) -> tuple[Any, ...]:
    shape = failure.get("shape") if isinstance(failure.get("shape"), list) else [None, None, None]
    return (
        failure.get("gate"),
        failure.get("semantics"),
        failure.get("finite_modulus"),
        shape[0] if len(shape) > 0 else None,
        shape[1] if len(shape) > 1 else None,
        shape[2] if len(shape) > 2 else None,
    )


def _failure_summary(failure: dict[str, Any]) -> dict[str, Any]:
    return {
        "failure_path": failure.get("failure_path"),
        "backend": failure.get("backend"),
        "failure_kind": failure.get("failure_kind"),
        "timed_out": failure.get("timed_out"),
        "timeout_seconds": failure.get("timeout_seconds"),
        "returncode": failure.get("returncode"),
    }


def build_report(paths: list[Path]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []
    blockers: dict[str, int] = {}
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    failed_grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for path in paths:
        if path.name.endswith(".failed.json"):
            failure = _load_failure(path)
            failed_rows.append(failure)
            failed_grouped[_failure_group_key(failure)].append(failure)
            blockers["failed_captures"] = blockers.get("failed_captures", 0) + 1
            if failure.get("timed_out") is True:
                blockers["timeout_captures"] = blockers.get("timeout_captures", 0) + 1
            continue
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
    all_group_keys = sorted(
        set(grouped.keys()) | set(failed_grouped.keys()),
        key=lambda item: tuple("" if value is None else str(value) for value in item),
    )
    for key in all_group_keys:
        captures = grouped.get(key, [])
        failures = failed_grouped.get(key, [])
        gate, semantics, finite_modulus, m, n, k = key
        backends = sorted({backend_id(capture) for capture in captures})
        failed_backends = sorted({str(failure.get("backend")) for failure in failures})
        timeout_backends = sorted(
            {str(failure.get("backend")) for failure in failures if failure.get("timed_out") is True}
        )
        required = required_baselines(semantics)
        missing = [backend for backend in required if backend not in backends]
        failed_required_backends = sorted(
            {str(failure.get("backend")) for failure in failures if failure.get("backend") in required}
        )
        timed_out_required_backends = sorted(
            {
                str(failure.get("backend"))
                for failure in failures
                if failure.get("backend") in required and failure.get("timed_out") is True
            }
        )
        unattempted_required = [
            backend for backend in required if backend not in backends and backend not in failed_required_backends
        ]
        required_attempted = sorted(
            {backend for backend in required if backend in backends or backend in failed_required_backends}
        )
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
        if failed_required_backends:
            blockers["failed_required_baselines"] = blockers.get("failed_required_baselines", 0) + 1
        if timed_out_required_backends:
            blockers["required_baseline_timeout"] = blockers.get("required_baseline_timeout", 0) + 1
        if unattempted_required:
            blockers["unattempted_required_baselines"] = blockers.get("unattempted_required_baselines", 0) + 1
        group_blockers = list(gate_blockers)
        if missing:
            group_blockers.append("missing_required_baselines")
        if failed_required_backends:
            group_blockers.append("failed_required_baselines")
        if timed_out_required_backends:
            group_blockers.append("required_baseline_timeout")
        if unattempted_required:
            group_blockers.append("unattempted_required_baselines")
        groups.append(
            {
                "gate": gate,
                "semantics": semantics,
                "finite_modulus": finite_modulus,
                "shape": [m, n, k],
                "capture_count": len(captures),
                "failed_capture_count": len(failures),
                "backends": backends,
                "failed_backends": failed_backends,
                "timeout_backends": timeout_backends,
                "required_baselines": required,
                "required_baselines_attempted": required_attempted,
                "missing_required_baselines": missing,
                "failed_required_baselines": failed_required_backends,
                "timed_out_required_baselines": timed_out_required_backends,
                "unattempted_required_baselines": unattempted_required,
                "required_baseline_failures": [
                    _failure_summary(failure) for failure in failures if failure.get("backend") in required
                ],
                "required_baselines_complete": not missing,
                "required_baseline_attempts_complete": not unattempted_required,
                "release_review_captures_complete": bool(captures)
                and all(release_review_capture(capture) for capture in captures),
                "review_statuses": review_statuses,
                "cache_eligible_rows": sum(
                    1
                    for capture in captures
                    if isinstance(capture.get("release_gate"), dict)
                    and capture.get("release_gate", {}).get("cache_eligible") is True
                ),
                "blockers": group_blockers,
            }
        )
    return {
        "schema": "rns8_release_gate_report_v2",
        "capture_count": len(rows),
        "failed_capture_count": len(failed_rows),
        "input_count": len(rows) + len(failed_rows),
        "group_count": len(groups),
        "rows": rows,
        "failed_rows": failed_rows,
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
