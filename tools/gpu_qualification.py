#!/usr/bin/env python3
"""Plan or run bounded GPU qualification. Default is plan-only; never promotes a cache."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

from process_progress import run_capture

ROOT = Path(__file__).resolve().parents[1]
BACKENDS = ("hip-direct", "amdgpu-builtins", "hipblaslt", "ck", "rocwmma")
FLAGS = {"amdgpu-builtins": "AMDGPU_BUILTINS", "hipblaslt": "HIPBLASLT", "ck": "CK", "rocwmma": "ROCWMMA"}
TARGETS = {"mi300x": ("gfx942", "Linux", "linux-cdna-release"),
           "7900xtx": ("gfx1100", "Windows", "windows-msvc-hip-release")}


def kernel_inventory(root: Path = ROOT) -> list[dict]:
    kernels = []
    for path in sorted((root / "src").rglob("*")):
        if path.suffix not in {".hip", ".cuh", ".inc"}:
            continue
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r"__global__\s+(?:__launch_bounds__\([^\n]*?\)\s+)?void\s+(\w+)\s*\(", text):
            kernels.append({"name": match[1], "source": path.relative_to(root).as_posix(),
                            "line": text.count("\n", 0, match.start()) + 1,
                            "evidence": "source_only_not_hardware_qualified"})
    return kernels


def command_plan(target: str, backends: list[str], out: Path, jobs: int) -> list[dict]:
    arch, system, preset = TARGETS[target]
    extension = ".exe" if system == "Windows" else ""
    python = "python" if system == "Windows" else "python3"
    wrapper = [python, "tools/windows_dev.py"] if system == "Windows" else []
    commands = []

    def add(name, argv, **checks):
        commands.append({"name": name, "argv": argv, **checks})

    for backend in backends:
        build = Path("build") / "qualification" / arch / backend
        binary = lambda name: str(build / (name + extension))
        options = [f"-DRNS8_ENABLE_{flag}={'ON' if name == backend else 'OFF'}" for name, flag in FLAGS.items()]
        add(f"{backend}-configure", wrapper + ["cmake", "--preset", preset, "-B", str(build),
            f"-DRNS8_AMDGPU_TARGETS={arch}", "-DRNS8_ENABLE_HIP=ON", *options])
        add(f"{backend}-build", wrapper + ["cmake", "--build", str(build), "--parallel", str(jobs)])
        add(f"{backend}-inspect", [binary("rns8-inspect"), "--backend", backend, "--device", "0", "--json"],
            inspect_target=arch, inspect_backend=backend)
        # Unlike an all-skipped CTest run, this fails if there is no HIP device.
        add(f"{backend}-smoke", [binary("rns8-verify"), "--hip-smoke"])
        junit = out / f"{backend}-qualification.xml"
        add(f"{backend}-regressions", [binary("rns8_tests"), "[qualification]", "--reporter", "junit", "--out", str(junit)],
            junit=str(junit), require_test="qualification pack covers dispatch thresholds and plane tails")
        add(f"{backend}-ctest", wrapper + ["ctest", "--test-dir", str(build), "--output-on-failure",
            "--parallel", str(jobs), "--timeout", "600", "--no-tests=error"])
        # Small captures are smoke evidence only, with bounded runtime per command.
        cases = [("bounded-i64", []), ("bounded-u64", []),
                 ("exact-wide-signed", ["--exact-wide-limbs", "3"]),
                 ("exact-wide-unsigned", ["--exact-wide-limbs", "3"]),
                 ("finite-u8-ring", ["--modulus", "255"]),
                 ("finite-u8-field", ["--modulus", "251"])]
        if backend == "hip-direct":
            cases += [("wrap-u64", []), ("finite-u8-ring", ["--modulus", "256"])]
        for index, (semantics, extra) in enumerate(cases):
            name = f"{backend}-{index}-{semantics}"
            capture = out / f"{name}.json"
            add(name, [binary("rns8-bench"), "--backend", backend, "--semantics", semantics,
                "--m", "32", "--n", "32", "--k", "64", "--warmups", "1", "--repeats", "3",
                "--seed", "20261006", "--cpu-threads", "1", "--progress", *extra], capture=str(capture))
            add(name + "-schema", [python, "tools/benchmark_schema.py", str(capture)])
        if backend == "hip-direct":
            # Existing graph capture, not the removed dead global graph helpers.
            name = "hip-direct-graph"
            capture = out / f"{name}.json"
            add(name, [binary("rns8-bench"), "--backend", "hip-direct", "--semantics", "bounded-i64",
                "--m", "32", "--n", "32", "--k", "64", "--hip-graph-replay",
                "--warmups", "1", "--repeats", "3", "--seed", "20261006", "--progress"], capture=str(capture))
            add(name + "-schema", [python, "tools/benchmark_schema.py", str(capture)])
    return commands


def validate_result(item: dict, result: dict) -> list[str]:
    errors = []
    if result["returncode"] != 0 or result["timed_out"]:
        return ["command failed or timed out"]
    if "inspect_target" in item:
        try:
            info = json.loads(result["stdout"])
            if info.get("gcn_arch", "").split(":")[0] != item["inspect_target"]:
                errors.append("runtime GPU target does not match the manifest")
            if info.get("backend") != item["inspect_backend"]:
                errors.append("runtime backend does not match the manifest")
        except (ValueError, AttributeError):
            errors.append("invalid device inspection JSON")
    if "junit" in item:
        try:
            root = ET.parse(ROOT / item["junit"]).getroot()
            cases = list(root.iter("testcase"))
            if not any(case.get("name") == item["require_test"] for case in cases):
                errors.append("required hardware regression missing (possibly a CPU-only build)")
            if any(list(case.iter("skipped")) or list(case.iter("failure")) or list(case.iter("error")) for case in cases):
                errors.append("hardware qualification tests failed or skipped")
        except (OSError, ET.ParseError):
            errors.append("missing or invalid qualification JUnit report")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, choices=TARGETS)
    parser.add_argument("--backend", action="append", choices=BACKENDS)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--execute", action="store_true", help="actually build and use an already available GPU")
    parser.add_argument("--timeout-seconds", type=float, default=1800)
    args = parser.parse_args()
    if args.jobs < 1 or args.timeout_seconds <= 0:
        parser.error("jobs and timeout must be positive")
    if args.execute and platform.system() != TARGETS[args.target][1]:
        parser.error("execution requires the target operating system; plan-only works anywhere")
    out = (args.out_dir or ROOT / "temp" / "qualification" / args.target).resolve()
    out.mkdir(parents=True, exist_ok=True)
    # Refuse to overwrite evidence. Start a new directory for every execution.
    if args.execute and (out / "results.json").exists():
        parser.error("results.json already exists; choose a fresh --out-dir")
    backends = list(dict.fromkeys(args.backend or BACKENDS))
    command_out = Path(os.path.relpath(out, ROOT))
    commands = command_plan(args.target, backends, command_out, args.jobs)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()
    diff = subprocess.run(["git", "diff", "HEAD"], cwd=ROOT, capture_output=True, check=True).stdout
    manifest = {"schema_version": 1, "generated_utc": datetime.now(timezone.utc).isoformat(),
                "target": args.target, "architecture": TARGETS[args.target][0], "git_revision": revision,
                "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(), "backends": backends,
                "status": "planned_not_executed", "promotion_allowed": False,
                "command_working_directory": "repository_root",
                "environment": {"RNS8_AUTOTUNE_CACHE": str(command_out / "isolated-empty-cache.json")},
                "commands": commands, "kernels": kernel_inventory(),
                "remaining_gates": ["HIP compile and hardware execution for every selected backend",
                    "full correctness and backend-specific ISA checks", "large/padded/K-boundary and sparse target coverage",
                    "release benchmarks, timing variance, profiler/counter evidence and explicit cache review"]}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"{len(commands)} commands, {len(manifest['kernels'])} source kernel definitions: {out / 'manifest.json'}", flush=True)
    if not args.execute:
        print("Plan only. No build, GPU work, benchmark, or cache promotion was executed.", flush=True)
        return 0
    cache = ROOT / manifest["environment"]["RNS8_AUTOTUNE_CACHE"]
    if cache.exists():
        parser.error("isolated cache path exists; choose a fresh output directory")
    environment = dict(os.environ, **manifest["environment"])
    results = []
    for index, item in enumerate(commands, 1):
        label = f"{index}/{len(commands)} {item['name']}"
        result = run_capture(item["argv"], cwd=ROOT, label=label, timeout_seconds=args.timeout_seconds, env=environment)
        (out / f"{item['name']}.stdout.log").write_text(result["stdout"], encoding="utf-8")
        (out / f"{item['name']}.stderr.log").write_text(result["stderr"], encoding="utf-8")
        errors = validate_result(item, result)
        if "capture" in item and not errors:
            (ROOT / item["capture"]).write_text(result["stdout"], encoding="utf-8")
        results.append({"name": item["name"], **result, "validation_errors": errors})
        report = {"status": "blocked" if errors else "in_progress", "promotion_allowed": False,
                  "completed": len(results), "planned": len(commands), "results": results}
        if not errors and len(results) == len(commands):
            report["status"] = "smoke_passed_release_qualification_pending"
        (out / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if errors:
            print(f"STOP: {item['name']}: {'; '.join(errors)}", file=sys.stderr, flush=True)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
