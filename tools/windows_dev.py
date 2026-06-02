#!/usr/bin/env python3
"""Run a command inside the Visual Studio developer environment when needed."""

from __future__ import annotations

import argparse
import subprocess
import sys

from msvc_env import command_in_msvc_environment


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a command with MSVC developer environment variables loaded. "
            "If the current shell already has VCINSTALLDIR/INCLUDE/LIB, the "
            "command runs directly."
        )
    )
    parser.add_argument("--arch", default="x64", help="target architecture passed to VsDevCmd.bat")
    parser.add_argument("--host-arch", default="x64", help="host architecture passed to VsDevCmd.bat")
    parser.add_argument("--dry-run", action="store_true", help="print the command that would run")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="command and arguments to run")
    args = parser.parse_args()

    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("missing command to run")

    try:
        wrapped, used_vsdevcmd = command_in_msvc_environment(
            command,
            arch=args.arch,
            host_arch=args.host_arch,
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.dry_run:
        print(wrapped if isinstance(wrapped, str) else subprocess.list2cmdline(wrapped))
        return 0

    if used_vsdevcmd:
        print("Loading Visual Studio developer environment automatically...", file=sys.stderr)
    completed = subprocess.run(wrapped, check=False, shell=isinstance(wrapped, str))
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
