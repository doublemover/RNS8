#!/usr/bin/env python3
"""Build RNS8 accelerator backends (CK, rocWMMA, hipBLASLt, AMDGPU builtins).

This wrapper loads the Visual Studio 2022 developer environment and builds each
accelerator preset (debug or release). Use it after the base HIP preset is built.

Usage:
  python tools/build_accelerators.py --debug     # build all debug accelerator presets
  python tools/build_accelerators.py --release    # build all release accelerator presets
  python tools/build_accelerators.py --debug --backend ck   # build only CK debug
  python tools/build_accelerators.py --list       # list available accelerator presets
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from msvc_env import command_in_msvc_environment


ACCELERATOR_PRESETS = {
    "ck": {
        "debug": "windows-msvc-ck-debug",
        "release": "windows-msvc-ck-release",
    },
    "rocwmma": {
        "debug": "windows-msvc-rocwmma-debug",
        "release": "windows-msvc-rocwmma-release",
    },
    "hipblaslt": {
        "debug": "windows-msvc-hipblaslt-debug",
        "release": "windows-msvc-hipblaslt-release",
    },
    "amdgpu-builtins": {
        "debug": "windows-msvc-amdgpu-builtins-debug",
        "release": "windows-msvc-amdgpu-builtins-release",
    },
}

BUILD_PRESETS = {
    "windows-msvc-ck-debug": "windows-ck-debug",
    "windows-msvc-ck-release": "windows-ck-release",
    "windows-msvc-rocwmma-debug": "windows-rocwmma-debug",
    "windows-msvc-rocwmma-release": "windows-rocwmma-release",
    "windows-msvc-hipblaslt-debug": "windows-hipblaslt-debug",
    "windows-msvc-hipblaslt-release": "windows-hipblaslt-release",
    "windows-msvc-amdgpu-builtins-debug": "windows-amdgpu-builtins-debug",
    "windows-msvc-amdgpu-builtins-release": "windows-amdgpu-builtins-release",
}


def build_preset(configure_preset: str, build_preset: str, repo_root: Path) -> bool:
    """Configure and build one accelerator preset."""
    print(f"\n{'='*60}")
    print(f"Building: {configure_preset}")
    print(f"{'='*60}")

    # Configure
    print(f"Configuring {configure_preset}...")
    config_args = ["cmake", "-S", str(repo_root), "--preset", configure_preset]
    cmd, _wrapped = command_in_msvc_environment(config_args)
    config_result = subprocess.run(cmd, check=False, cwd=str(repo_root),
                                      capture_output=True, text=True)
    if config_result.returncode != 0:
        print(f"ERROR: configure failed for {configure_preset}")
        if config_result.stdout:
            print(f"  STDOUT: {config_result.stdout[-500:]}")
        if config_result.stderr:
            print(f"  STDERR: {config_result.stderr[-500:]}")
        return False

    # Build
    print(f"Building {build_preset}...")
    build_args = ["cmake", "--build", "--preset", build_preset]
    cmd, _wrapped = command_in_msvc_environment(build_args)
    build_result = subprocess.run(cmd, check=False, cwd=str(repo_root),
                                    capture_output=True, text=True)
    if build_result.returncode != 0:
        print(f"ERROR: build failed for {build_preset}")
        # Print the last relevant lines from the build output
        error_lines = []
        for line in (build_result.stdout + build_result.stderr).split("\n"):
            if "error" in line.lower() or "ninja: build stopped" in line.lower():
                error_lines.append(line)
        if error_lines:
            print("  Build errors:")
            for line in error_lines[-15:]:
                print(f"    {line}")
        return False

    print(f"SUCCESS: {configure_preset}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--debug", action="store_true", help="build debug accelerator presets")
    parser.add_argument("--release", action="store_true", help="build release accelerator presets")
    parser.add_argument("--backend", choices=list(ACCELERATOR_PRESETS), help="build only this backend")
    parser.add_argument("--list", action="store_true", help="list available accelerator presets")
    parser.add_argument("--configure-only", action="store_true", help="only configure, don't build")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    if args.list:
        print("Accelerator presets available on Windows:")
        for backend, presets in ACCELERATOR_PRESETS.items():
            print(f"  {backend}: debug={presets['debug']}, release={presets['release']}")
        print("\nBuild presets:")
        for config, build in BUILD_PRESETS.items():
            print(f"  {config} -> {build}")
        return 0

    if not args.debug and not args.release:
        print("ERROR: specify --debug or --release")
        return 1

    configs: list[str] = []
    backends = [args.backend] if args.backend else list(ACCELERATOR_PRESETS)
    for backend in backends:
        if args.debug:
            configs.append(ACCELERATOR_PRESETS[backend]["debug"])
        if args.release:
            configs.append(ACCELERATOR_PRESETS[backend]["release"])

    print(f"Building {len(configs)} accelerator preset(s)...")

    failures = 0
    for config in configs:
        build = BUILD_PRESETS[config]
        result = build_preset(config, build, repo_root)
        if not result:
            failures += 1
        if args.configure_only:
            break  # Only configure first one for --configure-only

    if failures:
        print(f"\n{failures}/{len(configs)} preset(s) failed")
        return 1

    print(f"\nAll {len(configs)} accelerator preset(s) built successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
