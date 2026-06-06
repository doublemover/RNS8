from __future__ import annotations

import argparse
import glob
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

from msvc_env import command_in_msvc_environment, find_visual_studio_installation

from .human import print_human
from .report import build_report

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument(
        "--accelerator-probes",
        "--probe-accelerators",
        action="store_true",
        help=(
            "run optional compile/run probes plus CK/rocWMMA int8 primitive object probes "
            "for discovered accelerator components under temp/"
        ),
    )
    parser.add_argument(
        "--accelerator-probe-dir",
        type=Path,
        default=None,
        help="directory for optional accelerator probe source and binaries; defaults to temp/accelerator-deps/check-dependencies",
    )
    args = parser.parse_args()

    report, ok = build_report(args.accelerator_probes, args.accelerator_probe_dir)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_human(report)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
