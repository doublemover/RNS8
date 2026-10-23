"""Collector contract and process-record tests with an explicitly mocked child."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

from collect_native_calibration import main, validate_case


def run():
    case = dict(candidate="wmma", m=17, n=19, k=257, prefix=3, group=2, unsigned_inputs=False)
    assert validate_case(case) == case
    for key, value in (("m", True), ("k", 0), ("prefix", 21), ("group", 3),
                       ("unsigned_inputs", 0), ("candidate", "unknown")):
        try:
            validate_case({**case, key: value})
        except ValueError:
            pass
        else:
            raise AssertionError("malformed native case accepted")
    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch)
        exe, cases, output = root / "mock.exe", root / "cases.json", root / "records"
        exe.write_bytes(b"mocked subprocess only; never executed")
        cases.write_text(json.dumps([case]), encoding="utf-8")
        argv = ["collector", "--exe", str(exe), "--cases", str(cases), "--output", str(output),
                "--runs", "2", "--repeats", "3", "--warmups", "1"]
        def child(command, **kwargs):
            assert command[1] == "wmma" and command[-4] == "0"
            capture = {**case, "schema": "rns8_native_microcase_v1", "run_id": command[-1],
                       "repeats": 3, "warmups": 1, "samples": [{"exact_reference_passed": True}] * 3}
            return subprocess.CompletedProcess(command, 0, json.dumps(capture), "")
        with patch.object(sys, "argv", argv), patch("collect_native_calibration.subprocess.run", side_effect=child):
            assert main() == 0
        records = [json.loads(p.read_text()) for p in output.glob("*.json")]
        assert len(records) == 2 and len({r["run_id"] for r in records}) == 2
        assert {r["independent_repeat"] for r in records} == {0, 1}
        assert all(r["promotion_eligible"] is False for r in records)
        argv[argv.index(str(output))] = str(root / "timeout")
        with patch.object(sys, "argv", argv), patch("collect_native_calibration.subprocess.run",
                side_effect=subprocess.TimeoutExpired("mock", 1)):
            try:
                main()
            except RuntimeError:
                pass
            else:
                raise AssertionError("timeout accepted")
        failed = list((root / "timeout").glob("*.json"))
        assert len(failed) == 1 and json.loads(failed[0].read_text())["reason"] == "timeout"
    print("native calibration collector self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
