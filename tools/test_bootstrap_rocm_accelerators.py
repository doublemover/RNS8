#!/usr/bin/env python3
"""Self-test dry-run ROCm accelerator bootstrap behavior."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    script = Path(__file__).with_name("bootstrap_rocm_accelerators.py")
    repo_root = script.parent.parent

    with tempfile.TemporaryDirectory() as temp:
        probe_root = Path(temp) / "planned-probes"
        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "--dry-run",
                "--init",
                "--probe",
                "--target",
                "gfx1100",
                "--probe-root",
                str(probe_root),
                "--json",
            ],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        expect(result.returncode == 0, result.stdout)
        report = json.loads(result.stdout)

        expect(report["dry_run"] is True, "dry-run flag should be recorded")
        expect(report["record_written"] is False, "dry-run must not claim a written record")
        expect(report["record_path"] == str(probe_root / "bootstrap_rocm_accelerators.json"), "record path mismatch")
        expect(not probe_root.exists(), "dry-run must not create the probe root")

        planned = report["planned_actions"]
        expect(planned["submodule_update"] is True, "submodule update should be planned")
        expect(planned["compile_probes"] is True, "compile probes should be planned")
        expect(planned["write_report"] is False, "dry-run should not plan a report write")

        submodules = report["submodules"]["items"]
        for name in ("ck", "rocwmma"):
            expect(submodules[name]["planned_command"], f"{name} submodule command should be recorded")

        probes = report["probes"]["items"]
        for name in ("ck", "rocwmma"):
            expect(probes[name]["status"] == "dry_run_probe_planned", f"{name} probe status mismatch")
            expect(probes[name]["primitive_probe_status"] == "DRY_RUN_PLANNED", f"{name} primitive status mismatch")
            expect(probes[name]["source"].startswith(str(probe_root)), f"{name} source should stay under probe root")

    print("bootstrap ROCm accelerator dry-run self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
