#!/usr/bin/env python3
"""Check that the tracked release tree has no obvious scratch artifacts."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path, PurePosixPath


IGNORED_DIRS = {
    ".git",
    ".vs",
    ".cache",
    ".pytest_cache",
    ".ruff_cache",
    "build",
    "out",
    "temp",
    "vcpkg_installed",
}
FORBIDDEN_DIRS = {"__pycache__"}
FORBIDDEN_SUFFIXES = {
    ".a",
    ".bc",
    ".co",
    ".dll",
    ".dylib",
    ".exe",
    ".exp",
    ".hsaco",
    ".ilk",
    ".isa",
    ".lib",
    ".ll",
    ".map",
    ".o",
    ".obj",
    ".pdb",
    ".pyc",
    ".pyo",
    ".so",
    ".spv",
}
STALE_FIRST_PARTY_LICENSE_MARKERS = ("Apa" "che License", "Apa" "che-2.0")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def normalize_path(path: str | Path) -> PurePosixPath:
    return PurePosixPath(str(path).replace("\\", "/"))


def ignored_work_area(path: PurePosixPath) -> bool:
    parts = path.parts
    return bool(parts) and parts[0] in IGNORED_DIRS


def git_tree_paths(root: Path) -> list[PurePosixPath]:
    command = ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"]
    completed = subprocess.run(
        command,
        cwd=root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.decode("utf-8", errors="replace").strip())
    paths: list[PurePosixPath] = []
    for raw in completed.stdout.split(b"\0"):
        if raw:
            paths.append(normalize_path(raw.decode("utf-8", errors="replace")))
    return paths


def path_violations(paths: list[PurePosixPath]) -> list[str]:
    violations: list[str] = []
    for path in paths:
        if ignored_work_area(path):
            continue
        if any(part in FORBIDDEN_DIRS for part in path.parts):
            violations.append(f"{path}: forbidden generated-cache directory")
            continue
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            violations.append(f"{path}: forbidden native/generated artifact suffix")
    return violations


def metadata_violations(root: Path) -> list[str]:
    violations: list[str] = []
    license_text = (root / "LICENSE").read_text(encoding="utf-8", errors="replace")
    if "MIT License" not in license_text:
        violations.append("LICENSE: expected first-party MIT license text")

    notice_text = (root / "NOTICE").read_text(encoding="utf-8", errors="replace")
    if "MIT License" not in notice_text:
        violations.append("NOTICE: expected MIT license reference")
    if any(marker in notice_text for marker in STALE_FIRST_PARTY_LICENSE_MARKERS):
        violations.append("NOTICE: stale non-MIT first-party license reference")

    with (root / "vcpkg.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("license") != "MIT":
        violations.append("vcpkg.json: expected license MIT")
    dependencies = manifest.get("dependencies", [])
    if not isinstance(dependencies, list):
        violations.append("vcpkg.json: dependencies must be a list")
    else:
        dependency_names = {
            item if isinstance(item, str) else item.get("name")
            for item in dependencies
            if isinstance(item, (str, dict))
        }
        for stale in ("benchmark", "cli11", "fmt", "spdlog", "gmp", "flint"):
            if stale in dependency_names:
                violations.append(f"vcpkg.json: {stale} must not be a default dependency")
        for required in ("boost-multiprecision", "catch2", "nlohmann-json"):
            if required not in dependency_names:
                violations.append(f"vcpkg.json: missing default dependency {required}")
    features = manifest.get("features", {})
    optional = features.get("optional-exact-libs") if isinstance(features, dict) else None
    if not isinstance(optional, dict):
        violations.append("vcpkg.json: missing optional-exact-libs feature")
    else:
        feature_deps = optional.get("dependencies", [])
        feature_names = {
            item if isinstance(item, str) else item.get("name")
            for item in feature_deps
            if isinstance(item, (str, dict))
        }
        for required in ("gmp", "flint"):
            if required not in feature_names:
                violations.append(f"vcpkg.json: optional-exact-libs missing {required}")
    return violations


def release_tree_violations(root: Path, paths: list[PurePosixPath] | None = None) -> list[str]:
    checked_paths = paths if paths is not None else git_tree_paths(root)
    return path_violations(checked_paths) + metadata_violations(root)


def main() -> int:
    root = repo_root()
    violations = release_tree_violations(root)
    if violations:
        print("release tree check: FAIL")
        for violation in violations:
            print(f"- {violation}")
        return 1
    print("release tree check: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
