#!/usr/bin/env python3
"""Initialize with the repository's selected Python and automatic SDK paths."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def selected_python(root: Path = ROOT) -> Path:
    version = (root / ".python-version").read_text().strip()
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ValueError(".python-version must select an exact interpreter release")
    if ".".join(map(str, sys.version_info[:3])) == version:
        return Path(sys.executable)
    windows = sys.platform == "win32"
    candidates = []
    if os.environ.get("VIRTUAL_ENV"):
        candidates.append(Path(os.environ["VIRTUAL_ENV"]) / ("Scripts/python.exe" if windows else "bin/python"))
    candidates.append(root / (".venv/Scripts/python.exe" if windows else ".venv/bin/python"))
    if windows and os.environ.get("LOCALAPPDATA"):
        local = Path(os.environ["LOCALAPPDATA"])
        candidates.extend((local / f"RNS8/dependencies/python.{version}/tools/python.exe",
                           local / f"Programs/Python/Python{''.join(version.split('.')[:2])}/python.exe"))
    for name in ("python" + ".".join(version.split(".")[:2]), "python3", "python"):
        if executable := shutil.which(name):
            candidates.append(Path(executable))
    for candidate in dict.fromkeys(candidates):
        if not candidate.is_file() or "windowsapps" in str(candidate).lower():
            continue
        try:
            result = subprocess.run([str(candidate), "--version"], capture_output=True, text=True, timeout=15, check=True)
            if (result.stdout + result.stderr).strip() == "Python " + version:
                return candidate
        except (OSError, subprocess.SubprocessError):
            continue
    raise ValueError(f"Python {version} is not installed in the active/managed/PATH locations. Install the selected interpreter from the dependency guide; no older interpreter was substituted.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--no-build", action="store_true")
    args, forwarded = parser.parse_known_args(argv)
    try:
        python = selected_python()
    except (OSError, ValueError) as error:
        print("RNS8 init: " + str(error), file=sys.stderr)
        return 1
    if not any(value.split("=", 1)[0] in ("--sdk-auto", "--sdk-install", "--sdk-root") for value in forwarded):
        forwarded.insert(0, "--sdk-auto")
    if not args.no_build and "--sdk-build" not in forwarded:
        forwarded.append("--sdk-build")
    print(f"RNS8 init: selected Python {python}", file=sys.stderr)
    return subprocess.run([str(python), str(ROOT / "tools/bootstrap_rocm_accelerators.py"), *forwarded], cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
