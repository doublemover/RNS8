"""Tiny real-checker controls; never compile or execute fixture workloads."""

from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile

from process_progress import capture_succeeded, run_capture
from quality_inventory import ROOT, SourceFile


def run_controls(tools: dict[str, str], *, plan, runner=run_capture) -> dict:
    scratch = ROOT / "temp"
    scratch.mkdir(exist_ok=True)
    records = []
    with tempfile.TemporaryDirectory(prefix="quality controls λ ", dir=scratch) as temporary:
        directory = Path(temporary)
        cases = {
            "space λ.cpp": ("cpp", "int identity(int value) { return value; }\n"),
            "kernel µ.hip": ("cpp", "__global__ void kernel(int* out) { out[0] = 1; }\n"),
            "tile λ.cuh": ("cpp", "template<int N> struct Tile { int values[N]; };\n"),
            "case µ.inc": ("cpp", "case 1: { return 2; }\n"),
            "capture λ.py": ("python", 'print("authored checker control")\n'),
            "wrapper µ.sh": ("bash", "#!/usr/bin/env bash\nprintf '%s\\n' \"$1\"\n"),
            "workflow λ.yml": (
                "workflow",
                "on: push\njobs:\n  control:\n    runs-on: ubuntu-24.04\n"
                "    steps:\n      - run: echo authored\n",
            ),
        }
        paths = {}
        rows = []

        def invoke(argv, label):
            result = runner(
                list(argv), cwd=ROOT, label=label, timeout_seconds=30, max_output_bytes=256 * 1024
            )
            if (
                result.get("timed_out") is not False
                or result.get("limit_exceeded") is not None
                or result.get("capture_error") is not None
                or result.get("interrupted") not in (None, False)
                or type(result.get("returncode")) is not int
            ):
                raise ValueError("quality control could not complete: " + label)
            return result

        for name, (role, text) in cases.items():
            path = directory / name
            path.write_text(text, encoding="utf-8", newline="\n")
            paths[name] = path
            rows.append(SourceFile(path.relative_to(ROOT).as_posix(), role))
            if role == "cpp":
                # Canonicalize only owned scratch inputs. The actual gate below
                # is check-only and rejects incomplete parsing in every role.
                result = invoke(
                    [
                        tools["clang-format"],
                        "--style=file:" + str(ROOT / ".clang-format"),
                        "--fallback-style=none",
                        "--fail-on-incomplete-format",
                        str(path),
                    ],
                    "quality-prepare-" + path.suffix[1:],
                )
                if not capture_succeeded(result) or not result["stdout"].strip():
                    raise ValueError("quality control preparation failed: " + name)
                path.write_text(result["stdout"], encoding="utf-8", newline="\n")

        commands = {}
        for command in plan(tuple(rows), tools):
            argv = list(command.argv)
            if command.role.startswith("python-"):
                # Only these generated controls bypass temp/ignore exclusions.
                # The repository config and all lint/format rules still apply.
                argv[2:2] = ["--no-force-exclude", "--no-respect-gitignore"]
            if "-i" in argv or "--fix" in argv:
                raise ValueError("quality controls require non-mutating gate commands")
            commands[command.role] = argv
        if set(commands) != {"cpp-format", "python-format", "python-lint", "bash", "workflow"}:
            raise ValueError("quality controls omit a maintained checker")

        def hashes():
            return {
                name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()
            }

        for role, argv in commands.items():
            before = hashes()
            result = invoke(argv, "quality-positive-" + role)
            if not capture_succeeded(result) or hashes() != before:
                raise ValueError(
                    "quality positive control failed or mutated input: "
                    + role
                    + "\n"
                    + result["stdout"]
                    + result["stderr"]
                )
            records.append(dict(role=role, expected="accept", returncode=0))

        negatives = (
            (
                "cpp-format",
                "space λ.cpp",
                "int  identity( int value){return value;}\n",
                "clang-format-violations",
            ),
            ("python-lint", "capture λ.py", "print(authored_missing_name)\n", "F821"),
            ("bash", "wrapper µ.sh", "#!/usr/bin/env bash\nprintf '%s\\n' $1\n", "SC2086"),
            (
                "workflow",
                "workflow λ.yml",
                "on: push\njobs:\n  control:\n    runs-on: ubuntu-24.04\n"
                "    steps:\n      - run: echo ${{ authored_missing_context.value }}\n",
                "authored_missing_context",
            ),
        )
        for role, name, text, marker in negatives:
            path = paths[name]
            original = path.read_bytes()
            try:
                path.write_text(text, encoding="utf-8", newline="\n")
                before = hashes()
                result = invoke(commands[role], "quality-negative-" + role)
                if (
                    result["returncode"] == 0
                    or marker not in result["stdout"] + result["stderr"]
                    or hashes() != before
                ):
                    raise ValueError("quality negative control escaped or mutated input: " + role)
                records.append(
                    dict(
                        role=role, expected="reject", marker=marker, returncode=result["returncode"]
                    )
                )
            finally:
                path.write_bytes(original)
    return dict(
        schema="rns8_quality_controls_v1",
        records=records,
        scope="static_checker_controls_no_compilation_or_workload_execution",
    )
