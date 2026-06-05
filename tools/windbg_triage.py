#!/usr/bin/env python3
"""Locate and run non-GUI WinDbg triage through cdb.exe."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def candidate_cdb_paths() -> list[Path]:
    candidates: list[Path] = []
    from_path = shutil.which("cdb.exe") or shutil.which("cdb")
    if from_path:
      candidates.append(Path(from_path))
    roots = [
        Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")),
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")),
    ]
    for root in roots:
        kits = root / "Windows Kits" / "10" / "Debuggers"
        candidates.extend([kits / "x64" / "cdb.exe", kits / "x86" / "cdb.exe"])
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in candidates:
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def locate_cdb() -> Path | None:
    for path in candidate_cdb_paths():
        if path.exists() and path.is_file():
            return path
    return None


def run_cdb(cdb: Path, args: list[str]) -> int:
    command = [str(cdb), *args]
    completed = subprocess.run(command, text=True)
    return completed.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locate-only", action="store_true", help="Only print the cdb.exe path.")
    parser.add_argument("--dump", type=Path, help="Crash dump to analyze with !analyze -v.")
    parser.add_argument("--exe", type=Path, help="Executable to run under cdb.exe.")
    parser.add_argument("exe_args", nargs=argparse.REMAINDER, help="Arguments passed after --exe.")
    args = parser.parse_args()

    cdb = locate_cdb()
    if not cdb:
        print("cdb.exe not found on PATH or under Windows Kits Debuggers.", file=sys.stderr)
        return 1
    if args.locate_only:
        print(cdb)
        return 0
    if args.dump:
        if not args.dump.exists():
            print(f"dump not found: {args.dump}", file=sys.stderr)
            return 1
        return run_cdb(cdb, ["-z", str(args.dump), "-c", ".symfix; .reload; !analyze -v; kb; q"])
    if args.exe:
        if not args.exe.exists():
            print(f"executable not found: {args.exe}", file=sys.stderr)
            return 1
        exe_args = args.exe_args[1:] if args.exe_args[:1] == ["--"] else args.exe_args
        return run_cdb(
            cdb,
            [
                "-o",
                "-G",
                "-g",
                "-c",
                ".symfix; .reload; g; !analyze -v; kb; q",
                str(args.exe),
                *exe_args,
            ],
        )
    parser.error("provide --locate-only, --dump, or --exe")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
