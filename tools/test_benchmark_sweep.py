#!/usr/bin/env python3
"""Self-test benchmark sweep review and promotion helpers."""

from __future__ import annotations

from pathlib import Path

from test_benchmark_sweep_support import *  # noqa: F401,F403 - compatibility re-export

_CASE_FRAGMENTS = [
    "test_benchmark_sweep_case_scenarios.py",
    "test_benchmark_sweep_case_command_matrix.py",
    "test_benchmark_sweep_case_large_resume.py",
    "test_benchmark_sweep_case_chain_adaptive.py",
    "test_benchmark_sweep_case_review_metadata.py",
    "test_benchmark_sweep_case_cache_wrap_exact.py",
]


def _run_case_fragment(name: str, state: dict) -> None:
    source = (Path(__file__).resolve().with_name(name)).read_text(encoding="utf-8")
    exec(compile(source, name, "exec"), state, state)


def main() -> int:
    state = dict(globals())
    for fragment in _CASE_FRAGMENTS:
        _run_case_fragment(fragment, state)
    print("benchmark sweep self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
