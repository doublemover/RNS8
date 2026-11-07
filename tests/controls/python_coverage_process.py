"""Bounded parent/child tracing sensitivity; outside application PASS counts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


def parent_path(value):
    if value:
        return -1
    return 23


def child_path(value):
    if value:
        return 37
    return -1


def main(argv=None):
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "tools"))
    from process_progress import capture_succeeded, run_capture

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("parent", "child"))
    parser.add_argument("--child-cwd", type=Path)
    args = parser.parse_args(argv)
    if sys.flags.optimize:
        return 2
    if args.role == "child":
        print(json.dumps(dict(role="child", process_id=os.getpid(), value=child_path(True))))
        return 0
    if args.child_cwd is None or not args.child_cwd.is_dir():
        return 2
    result = run_capture(
        [sys.executable, str(Path(__file__).resolve()), "child"],
        cwd=args.child_cwd,
        label="coverage-child-control",
        env=dict(os.environ),
        timeout_seconds=30,
        max_output_bytes=4096,
    )
    if not capture_succeeded(result):
        return 1
    child = json.loads(result["stdout"])
    if (
        child.get("process_id") != result.get("process_id")
        or child.get("value") != 37
        or child.get("role") != "child"
    ):
        return 1
    print(
        json.dumps(
            dict(
                schema="rns8_python_coverage_process_control_v1",
                process_id=os.getpid(),
                value=parent_path(False),
                child=child,
                child_process=result,
            )
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
