"""CLI parse-only subprocess checks: --help follows tokens; no contexts/workloads."""

import argparse
from pathlib import Path
import subprocess


def run(binary: Path, arguments: list[str], expected: int) -> None:
    result = subprocess.run(
        [str(binary), *arguments], capture_output=True, text=True, timeout=20, check=False
    )
    if result.returncode != expected:
        raise AssertionError(
            f"{binary.name} {arguments!r}: expected {expected}, got {result.returncode}\n{result.stdout}\n{result.stderr}"
        )
    if expected == 2 and (
        "usage:" not in result.stderr
        or "rns8_create_context(" in result.stderr
        or "rns8_get_backend_capability_info(" in result.stderr
    ):
        raise AssertionError(
            "invalid token must fail with usage before backend/context access: " + result.stderr
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inspect", type=Path)
    parser.add_argument("benchmark", type=Path)
    args = parser.parse_args()
    for binary in (args.inspect, args.benchmark):
        run(binary, ["--help"], 0)
        run(binary, ["--device"], 2)
        for value in (
            "",
            "not-a-number",
            "1junk",
            " 1",
            "1 ",
            "+1",
            "-2",
            "-2147483648",
            "2147483648",
            "4294967296",
            "9223372036854775808",
            "-9223372036854775809",
        ):
            run(binary, ["--backend", "hip-direct", "--device", value, "--help"], 2)
        for value in ("-1", "0", "1", "2147483647"):
            run(binary, ["--device", value, "--help"], 0)
    for option in ("--m", "--output-ld-padding"):
        for value in ("", "1junk", "9223372036854775808", "-9223372036854775809"):
            run(args.benchmark, ["--backend", "hip-direct", option, value, "--help"], 2)
    run(args.benchmark, ["--json-output"], 2)
    run(args.benchmark, ["--json-output", "--help"], 2)
    run(args.benchmark, ["--json-output", "", "--help"], 2)
    run(args.benchmark, ["--json-output", "temp/native-json-output.json", "--help"], 0)
    run(
        args.benchmark, ["--json-output", "first.json", "--json-output", "second.json", "--help"], 2
    )
    run(args.benchmark, ["--record-execution-environment", "--help"], 0)
    run(
        args.benchmark,
        ["--record-execution-environment", "--record-execution-environment", "--help"],
        2,
    )


if __name__ == "__main__":
    main()
