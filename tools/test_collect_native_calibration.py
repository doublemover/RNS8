"""Collector contract and process-record tests with an explicitly mocked child."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

from collect_native_calibration import main, validate_case, epilogue_contract, narrow_contract


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
    for name in ("epilogue_direct", "epilogue_rocwmma", "epilogue_unfused"):
        expression_case = {**case, "candidate": name, "group": 1}
        assert validate_case(expression_case) == expression_case
        contract = epilogue_contract(expression_case)
        assert contract["expression"]["alpha"] == -(1 << 63)
        assert contract["reserved_intermediate_bytes"] == 17 * 19 * 3
        assert bool(contract["materialized_intermediate_bytes"]) == (name == "epilogue_unfused")
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
        epilogue = {**case, "candidate": "epilogue_direct", "group": 1}
        cases.write_text(json.dumps([epilogue]), encoding="utf-8")
        argv[argv.index(str(root / "timeout"))] = str(root / "wrong-expression")
        def wrong_expression(command, **kwargs):
            capture = {**epilogue, **epilogue_contract(epilogue), "schema": "rns8_native_microcase_v1",
                       "run_id": command[-1], "repeats": 3, "warmups": 1,
                       "samples": [{"exact_reference_passed": True}] * 3}
            capture["expression"]["beta"] = 0
            return subprocess.CompletedProcess(command, 0, json.dumps(capture), "")
        with patch.object(sys, "argv", argv), patch("collect_native_calibration.subprocess.run", side_effect=wrong_expression):
            try:
                main()
            except RuntimeError as error:
                assert "contract/reference mismatch" in str(error)
            else:
                raise AssertionError("different epilogue expression accepted")
        assert len(list((root / "wrong-expression").glob("*.json"))) == 1
        for name in ("narrow_wmma", "narrow_scalar", "narrow_vector"):
            narrow = {**case, "candidate": name, "prefix": 1, "group": 1}
            assert validate_case(narrow) == narrow
            assert narrow_contract(narrow)["abs_partial_sum_bound"] == 257 * 128 ** 2
            for changes in ({"prefix": 2}, {"k": 32769}, {"group": 2}):
                try:
                    validate_case({**narrow, **changes})
                except ValueError:
                    pass
                else:
                    raise AssertionError("unmatched narrow contract accepted")
        narrow = {**case, "candidate": "narrow_wmma", "prefix": 1, "group": 1, "unsigned_inputs": True}
        assert narrow_contract(narrow)["abs_partial_sum_bound"] == 257 * 255 ** 2
        try:
            validate_case({**narrow, "candidate": "narrow_scalar"})
        except ValueError:
            pass
        else:
            raise AssertionError("unsigned scalar accepted")
        cases.write_text(json.dumps([narrow]), encoding="utf-8")
        argv[argv.index(str(root / "wrong-expression"))] = str(root / "wrong-native-output")
        def wrong_native_output(command, **kwargs):
            capture = {**narrow, **narrow_contract(narrow), "schema": "rns8_native_microcase_v1",
                       "run_id": command[-1], "repeats": 3, "warmups": 1,
                       "samples": [{"exact_reference_passed": True}] * 3}
            capture["output_contract"] = "product_residues_default_basis_no_integer_export"
            return subprocess.CompletedProcess(command, 0, json.dumps(capture), "")
        with patch.object(sys, "argv", argv), patch("collect_native_calibration.subprocess.run", side_effect=wrong_native_output):
            try:
                main()
            except RuntimeError as error:
                assert "contract/reference mismatch" in str(error)
            else:
                raise AssertionError("residue output admitted as native integer comparison")
    print("native calibration collector self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
