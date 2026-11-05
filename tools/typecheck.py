#!/usr/bin/env python3
"""Run the explicitly migrated Python evidence boundary and checker controls."""

from __future__ import annotations

import argparse
import configparser
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys

from process_progress import run_capture, capture_succeeded
from quality_inventory import ROOT, inventory
from quality_policy import load_quality_lock


def checked_scope(root: Path = ROOT) -> dict:
    scope = json.loads((root / "metadata/python-typecheck-scope.json").read_text(encoding="utf-8"))
    if scope.get("schema") != "rns8_python_typecheck_scope_v1":
        raise ValueError("unsupported staged Python type-check scope")
    modules = scope.get("strict_modules")
    if (not isinstance(modules, list) or not modules or len(set(modules)) != len(modules)
            or any(not isinstance(name, str) or not name.startswith("tools/") for name in modules)):
        raise ValueError("type-check scope needs a nonempty explicit module inventory")
    config = configparser.ConfigParser(interpolation=None)
    if config.read(str(root / "mypy.ini"), encoding="utf-8") != [str(root / "mypy.ini")]:
        raise ValueError("repository mypy configuration is missing")
    configured = [name.strip() for name in config["mypy"]["files"].split(",")]
    if configured != modules or config["mypy"]["python_version"] != scope["source_target"]:
        raise ValueError("mypy config and migration inventory disagree")
    names = {Path(name).stem for name in modules}
    sections = [name for name in config.sections() if name.startswith("mypy-")]
    strict_names = set()
    for section in sections:
        if not config.getboolean(section, "strict") or not config.getboolean(section, "disallow_untyped_defs"):
            raise ValueError("migrated boundary must have strict complete signatures")
        strict_names.update(section.removeprefix("mypy-").split(","))
    if strict_names != names:
        raise ValueError("strict mypy module discovery differs from migrated sources")
    for section in config.sections():
        if config.get(section, "follow_imports", fallback="normal") == "skip" or config.getboolean(section, "ignore_missing_imports", fallback=False):
            raise ValueError("blanket import suppression cannot qualify a type gate")
    negatives = scope.get("negative_inputs")
    if not isinstance(negatives, dict) or len(negatives) != 3:
        raise ValueError("three maintained negative checker inputs required")
    policy = json.loads((root / "metadata/first-party-inventory.json").read_text(encoding="utf-8"))
    if set(negatives) != set(policy.get("checker_negative_inputs", {})):
        raise ValueError("negative checker inputs differ from explicit passing-source exclusions")
    for name in [*modules, *negatives]:
        path = root / name
        if path.suffix != ".py" or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("type-check source is absent or outside the repository")
    return scope


def run_checks(*, root: Path = ROOT, tool_bin: Path | None = None,
               self_test=False, runner=run_capture) -> dict:
    lock, scope = load_quality_lock(root), checked_scope(root)
    checker = next(row for row in lock["type_checking"]["wheels"] if row["name"] == "mypy")
    filename = "mypy.exe" if os.name == "nt" else "mypy"
    path = tool_bin / filename if tool_bin is not None else shutil.which("mypy")
    if path is None or not Path(path).is_file():
        raise ValueError("missing hash-locked developer type checker")
    executable = Path(path).resolve(strict=True)
    before = hashlib.sha256(executable.read_bytes()).hexdigest()
    environment = dict(os.environ)
    environment.pop("MYPYPATH", None)
    version = runner([str(executable), "--version"], cwd=root, env=environment,
        label="mypy-version", timeout_seconds=15, max_output_bytes=65536)
    if (not capture_succeeded(version) or re.search(
            r"(?<![\w.])" + re.escape(checker["version"]) + r"(?![\w.+-])",
            version["stdout"] + "\n" + version["stderr"]) is None):
        raise ValueError("type checker does not match pinned mypy " + checker["version"])
    base = [str(executable), "--config-file", str(root / "mypy.ini"), "--no-incremental",
            "--cache-dir", str(root / "temp/mypy-cache")]
    observed = runner([*base, *scope["strict_modules"]], cwd=root, env=environment,
        label="mypy-strict", timeout_seconds=120, max_output_bytes=4 * 1024**2)
    if not capture_succeeded(observed):
        raise ValueError("strict migrated Python boundary failed:\n" + observed["stdout"] + observed["stderr"])
    negatives = []
    if self_test:
        for name, code in scope["negative_inputs"].items():
            result = runner([*base, name], cwd=root, env=environment,
                label="mypy-negative-" + Path(name).stem, timeout_seconds=120,
                max_output_bytes=4 * 1024**2)
            text = result["stdout"] + "\n" + result["stderr"]
            errors = [line for line in text.splitlines()
                if Path(name).name in line and "error:" in line and "[" + code + "]" in line]
            if (type(result.get("returncode")) is not int or result["returncode"] != 1
                    or result.get("timed_out") is not False
                    or result.get("limit_exceeded") is not None
                    or result.get("capture_error") is not None or not errors):
                raise ValueError("checker did not reject its deliberate " + code + " input: " + name)
            negatives.append(dict(source=name, expected_error=code, diagnostics=errors))
    if hashlib.sha256(executable.read_bytes()).hexdigest() != before:
        raise ValueError("type-checker executable changed during invocation")
    python_sources = [row.path for row in inventory(root) if row.role == "python"]
    return dict(schema="rns8_python_typecheck_result_v1", checker="mypy",
        version=checker["version"], executable=str(executable), executable_sha256=before,
        source_target=scope["source_target"], strict_modules=scope["strict_modules"],
        explicit_strict_module_count=len(scope["strict_modules"]),
        outside_explicit_strict_scope_count=len(set(python_sources) - set(scope["strict_modules"])),
        remaining_migration=scope["remaining_migration"], negative_controls=negatives,
        scope="staged_static_boundary_not_all_python_or_runtime_validation")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool-bin", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(run_checks(tool_bin=args.tool_bin, self_test=args.self_test), indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError, configparser.Error) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
