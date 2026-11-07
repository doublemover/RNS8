#!/usr/bin/env python3
"""Check tracked first-party source; fix mode changes formatting only."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from quality_inventory import ROOT, ROLES, SourceFile, batches, inventory
from quality_policy import load_quality_lock
from evidence_records import Command
from process_progress import capture_succeeded, run_capture


def command_plan(
    sources: tuple[SourceFile, ...],
    tools: dict[str, str],
    *,
    fix=False,
    root: Path = ROOT,
) -> tuple[Command, ...]:
    if type(fix) is not bool:
        raise ValueError("formatting mutation must be explicitly boolean")
    commands = []
    for role in ROLES:
        paths = [row.path for row in sources if row.role == role]
        if not paths:
            raise ValueError("quality inventory has no " + role + " files")
        for group in batches(paths):
            if role == "cpp":
                # The explicit style file also prevents home-directory fallback.
                flags = ("-i",) if fix else ("--dry-run", "--Werror")
                commands.append(
                    Command(
                        "cpp-format",
                        (
                            tools["clang-format"],
                            "--style=file:" + str(root / ".clang-format"),
                            "--fallback-style=none",
                            "--fail-on-incomplete-format",
                            *flags,
                            *group,
                        ),
                    )
                )
            elif role == "python":
                commands.append(
                    Command(
                        "python-format",
                        (
                            tools["ruff"],
                            "format",
                            "--config",
                            str(root / "ruff.toml"),
                            *(() if fix else ("--check",)),
                            "--",
                            *group,
                        ),
                    )
                )
                # Even explicit fix mode never deletes imports or rewrites predicates.
                commands.append(
                    Command(
                        "python-lint",
                        (
                            tools["ruff"],
                            "check",
                            "--config",
                            str(root / "ruff.toml"),
                            "--",
                            *group,
                        ),
                    )
                )
            elif role == "workflow":
                commands.append(
                    Command(
                        "workflow",
                        (
                            tools["actionlint"],
                            "-color=false",
                            "-config-file",
                            str(root / ".github/actionlint.yaml"),
                            "-shellcheck",
                            tools["shellcheck"],
                            *group,
                        ),
                    )
                )
            elif role == "bash":
                commands.append(
                    Command(
                        "bash",
                        (
                            tools["shellcheck"],
                            "--format=json",
                            "--shell=bash",
                            "--external-sources",
                            "--source-path=SCRIPTDIR",
                            "--",
                            *group,
                        ),
                    )
                )
    return tuple(commands)


def selected_tools(lock: dict, *, tool_bin: Path | None = None) -> dict[str, str]:
    if lock.get("schema") != "rns8_quality_tools_v1":
        raise ValueError("unsupported quality tool lock")
    tools = {}
    for tool in lock["tools"]:
        name = tool["name"]
        suffix = ".exe" if os.name == "nt" else ""
        path = (tool_bin / (name + suffix)) if tool_bin is not None else shutil.which(name)
        if path is None or not Path(path).is_file():
            raise ValueError("missing locked developer tool: " + name)
        binary = Path(path).resolve(strict=True)
        result = subprocess.run(
            [str(binary), "--version" if name != "actionlint" else "-version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
            timeout=15,
            check=True,
        )
        output = result.stdout + "\n" + result.stderr
        if re.search(r"(?<![\w.])" + re.escape(tool["version"]) + r"(?![\w.+-])", output) is None:
            raise ValueError(name + " must match locked version " + tool["version"])
        tools[name] = str(binary)
        print(
            json.dumps(
                dict(
                    tool=name,
                    version=tool["version"],
                    path=str(binary),
                    executable_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                )
            ),
            flush=True,
        )
    if set(tools) != {"clang-format", "ruff", "actionlint", "shellcheck"}:
        raise ValueError("quality lock must declare exactly the maintained four tools")
    return tools


def main(argv=None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "analyze":
        from static_analysis import main as analysis_main

        return analysis_main(arguments[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("check", "fix", "inventory", "types", "analyze", "self-test")
    )
    parser.add_argument("--tool-bin", type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=300)
    args = parser.parse_args(arguments)
    try:
        if args.mode == "types":
            from typecheck import main as typecheck_main

            return typecheck_main(
                ["--self-test", *(["--tool-bin", str(args.tool_bin)] if args.tool_bin else [])]
            )
        if not 0 < args.timeout_seconds < float("inf"):
            raise ValueError("positive finite per-command timeout required")
        sources = inventory()
        if args.mode == "inventory":
            print(json.dumps([asdict(row) for row in sources], indent=2))
            return 0
        lock = load_quality_lock(ROOT)
        tools = selected_tools(lock, tool_bin=args.tool_bin)
        if args.mode == "self-test":
            from quality_controls import run_controls

            print(json.dumps(run_controls(tools, plan=command_plan), indent=2))
            return 0
        for name in (
            ".clang-format",
            "ruff.toml",
            ".github/actionlint.yaml",
            "metadata/first-party-inventory.json",
        ):
            print(
                json.dumps(
                    dict(
                        config=name,
                        sha256=hashlib.sha256((ROOT / name).read_bytes()).hexdigest(),
                    )
                ),
                flush=True,
            )
        failed = False
        for command in command_plan(sources, tools, fix=args.mode == "fix"):
            print(json.dumps(asdict(command)), flush=True)
            result = run_capture(
                list(command.argv),
                cwd=ROOT,
                label="quality-" + command.role,
                timeout_seconds=args.timeout_seconds,
                max_output_bytes=8 * 1024**2,
            )
            print(result["stdout"], end="", flush=True)
            print(result["stderr"], end="", file=sys.stderr, flush=True)
            print(
                json.dumps(
                    {
                        key: result.get(key)
                        for key in (
                            "returncode",
                            "elapsed_seconds",
                            "timed_out",
                            "limit_exceeded",
                            "capture_error",
                        )
                    }
                ),
                flush=True,
            )
            failed |= not capture_succeeded(result)
        return int(failed)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
