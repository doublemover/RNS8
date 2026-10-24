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
from .resolution import FEATURE_KEYS, REQUEST_KEYS, Request, read_cache, resolve, truth

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--resolve-only", action="store_true", help="write-free shared SDK/toolchain selection; no command/runtime/probe execution")
    parser.add_argument("--resolution-request", type=Path, help="typed JSON request used by early CMake; no writes to the request")
    parser.add_argument("--cmake-cache", type=Path, help="read configured feature/root choices from a CMakeCache.txt")
    parser.add_argument("--set", dest="definitions", action="append", default=[], metavar="NAME=VALUE",
                        help="explicit root/toolchain or feature override; same names as CMake")
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

    try:
        if args.resolution_request and (args.cmake_cache or args.definitions):
            parser.error("--resolution-request cannot be combined with cache/definition overrides")
        if args.resolution_request:
            request = Request.from_mapping(json.loads(args.resolution_request.read_text(encoding="utf-8")))
        else:
            overrides, features = read_cache(args.cmake_cache) if args.cmake_cache else ({}, {})
            seen = set()
            for definition in args.definitions:
                if "=" not in definition: raise ValueError("--set requires NAME=VALUE")
                key, value = definition.split("=", 1)
                if key in seen: raise ValueError("duplicate --set: " + key)
                seen.add(key)
                if key in FEATURE_KEYS: features[key] = truth(value)
                elif key in REQUEST_KEYS: overrides[key] = value
                else: raise ValueError("unsupported --set variable: " + key)
            host = platform.system()
            hip_requested = features.get("RNS8_ENABLE_HIP", not bool(args.cmake_cache))
            request = Request(host, overrides, features, str(Path(__file__).resolve().parents[2]), hip_requested,
                              host == "Windows")
        if args.resolve_only:
            if args.accelerator_probes: parser.error("resolve-only cannot execute accelerator probes")
            report = resolve(request); ok = not report["errors"]
            if args.json:
                print(json.dumps(report, indent=2, sort_keys=True))
            else:
                print("Shared dependency resolution: " + report["status"])
                for name in ("hip", "vcpkg"):
                    item = report[name]; selected = item.get("selected")
                    print(name + ": " + item["status"] + " — " + (selected["canonical_root"] if selected else item["selection_reason"]))
                for error in report["errors"]: print("ERROR: " + error)
                print(report["validation_scope"])
            return 0 if ok else 1
        report, ok = build_report(args.accelerator_probes, args.accelerator_probe_dir, request)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_human(report)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
