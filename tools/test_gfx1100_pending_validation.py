#!/usr/bin/env python3
"""Self-test gfx1100 pending validation command planning."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import gfx1100_pending_validation


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        args = SimpleNamespace(
            out_dir=tmp,
            skip_build=True,
            warmups=3,
            repeats=9,
            seed=20260606,
            max_new_captures=2,
            capture_timeout_seconds=300,
        )
        commands = gfx1100_pending_validation.command_plan(args)
        assert len(commands) == 4
        names = [item.name for item in commands]
        assert names == [
            "sweep_exact-wide-export",
            "sweep_export-bound-limb-variants",
            "sweep_resident-lifetime-arena",
            "sweep_k-block-tile-variants",
        ]
        for item in commands:
            text = " ".join(item.command)
            assert "--review-mode release" in text
            assert "--warmups 3" in text
            assert "--repeats 9" in text
            assert "--bench-for hip-direct=" in text
            assert "--bench-for hipblaslt=" in text
            assert "--bench-for ck=" in text
            assert "--bench-for rocwmma=" in text
            assert "--max-new-captures 2" in text
            assert "--capture-timeout-seconds 300" in text
        outputs = gfx1100_pending_validation.write_command_plan(commands, tmp)
        payload = json.loads(Path(outputs["json"]).read_text(encoding="utf-8"))
        assert payload["target"] == "gfx1100"
        assert len(payload["commands"]) == 4

    print("gfx1100 pending validation planner self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
