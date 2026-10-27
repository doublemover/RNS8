#!/usr/bin/env python3
"""CDNA entrypoints only dispatch; no probe/install/default-device side effects."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KINDS = {"cdna_prepare.sh": "prepare", "cdna_env_probe.sh": "environment", "cdna_smoke.sh": "smoke",
         "cdna_first_pass.sh": "first-pass", "cdna_multigpu_smoke.sh": "multigpu"}


class ShellTests(unittest.TestCase):
    def test_one_runner_no_hidden_bootstrap_or_device_discovery(self):
        common = (ROOT / "scripts/cdna_common.sh").read_text()
        self.assertIn("tools/gpu_qualification.py", common)
        self.assertIn('"$@"', common)
        for forbidden in ("rocm-smi", "rocminfo", "pip install", "git clone", "gfx942", "printenv", "cdna_discover_devices"):
            self.assertNotIn(forbidden, common)
        for name, kind in KINDS.items():
            source = (ROOT / "scripts" / name).read_text()
            self.assertIn('source "${SCRIPT_DIR}/cdna_common.sh"', source)
            self.assertIn('cdna_session ' + kind + ' "$@"', source)
            self.assertNotIn("HIP_VISIBLE_DEVICES=", source)
            self.assertNotIn("env >", source)

    @unittest.skipIf(os.name == "nt" or shutil.which("bash") is None, "requires native Linux Bash; no WSL assumption")
    def test_native_bash_help_and_required_selection(self):
        environment = dict(os.environ, PYTHON=sys.executable)
        for name in (*KINDS, "cdna_accelerators.sh"):
            script = ROOT / "scripts" / name
            syntax = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True, timeout=10)
            self.assertEqual(syntax.returncode, 0, syntax.stderr)
            help_result = subprocess.run(["bash", str(script), "--help"], cwd=ROOT, env=environment,
                                         capture_output=True, text=True, timeout=30)
            self.assertEqual(help_result.returncode, 0, help_result.stderr)
            self.assertIn("--devices", help_result.stdout)
            self.assertIn("--execute", help_result.stdout)
            missing = subprocess.run(["bash", str(script), "--dry-run"], cwd=ROOT, env=environment,
                                     capture_output=True, text=True, timeout=30)
            self.assertEqual(missing.returncode, 2)
            self.assertIn("--target", missing.stderr)
            self.assertIn("--devices", missing.stderr)


if __name__ == "__main__":
    unittest.main()
