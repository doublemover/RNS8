#!/usr/bin/env python3
"""Self-test benchmark schema validation fixtures."""

from __future__ import annotations

from pathlib import Path

from test_benchmark_schema_support import *  # noqa: F401,F403 - compatibility re-export

_CASE_FRAGMENTS = [
    "test_benchmark_schema_case_vector_grouped.py",
    "test_benchmark_schema_case_reuse_chain.py",
    "test_benchmark_schema_case_oneshot_accelerators.py",
    "test_benchmark_schema_case_events_schedule.py",
]


def _run_case_fragment(name: str, state: dict) -> None:
    source = (Path(__file__).resolve().with_name(name)).read_text(encoding="utf-8")
    exec(compile(source, name, "exec"), state, state)


def main() -> int:
    state = dict(globals())
    for fragment in _CASE_FRAGMENTS:
        _run_case_fragment(fragment, state)
    print("benchmark schema self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
