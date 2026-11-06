"""Integrity and version checks for the optional, shared-policy developer tools."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re


def _check_requirements(root: Path, path: str, expected: dict[str, set[str]], label: str) -> None:
    actual: dict[str, set[str]] = {}
    for line in (root / path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split()
        if parts[0] in actual or not all(part.startswith("--hash=sha256:") for part in parts[1:]):
            raise ValueError(
                label + " requirements contain duplicate packages or unsupported options"
            )
        actual[parts[0]] = {part.removeprefix("--hash=sha256:") for part in parts[1:]}
    if actual != expected:
        raise ValueError(label + " requirements differ from reviewed versions/hashes")


def _wheel_closure(
    root: Path,
    lock: dict,
    components: dict,
    *,
    key: str,
    checker: str,
    requirements: str,
    names: set[str],
) -> None:
    closure = lock.get(key)
    if (
        not isinstance(closure, dict)
        or closure.get("checker") != checker
        or closure.get("requirements") != requirements
        or not isinstance(closure.get("wheels"), list)
    ):
        raise ValueError("quality lock has no maintained " + key + " wheel closure")
    seen, expected = set(), {}
    for wheel in closure["wheels"]:
        if not isinstance(wheel, dict):
            raise ValueError("invalid " + key + " wheel identity")
        name, version = wheel.get("name"), wheel.get("version")
        component = components.get(name) if type(name) is str else None
        selected = component.get("selected") if isinstance(component, dict) else None
        if (
            type(name) is not str
            or name in seen
            or name not in names
            or type(version) is not str
            or not isinstance(selected, dict)
            or selected.get("version") != version
        ):
            raise ValueError(key + " wheel version differs from shared dependency policy")
        seen.add(name)
        artifacts = wheel.get("artifacts")
        if (
            not isinstance(artifacts, list)
            or len(artifacts) != 2
            or any(not isinstance(row, dict) for row in artifacts)
            or any(type(row.get("platform")) is not str for row in artifacts)
            or {row.get("platform") for row in artifacts} != {"windows-amd64", "linux-x86_64"}
        ):
            raise ValueError(key + " wheel platform inventory is incomplete")
        for artifact in artifacts:
            digest, url, filename = (artifact.get(field) for field in ("sha256", "url", "filename"))
            if (
                type(digest) is not str
                or not re.fullmatch(r"[0-9a-f]{64}", digest)
                or type(url) is not str
                or not url.startswith("https://")
                or type(filename) is not str
                or Path(filename).name != filename
                or not filename.endswith(".whl")
            ):
                raise ValueError(key + " wheel lacks immutable HTTPS identity")
        expected[name + "==" + version] = {row["sha256"] for row in artifacts}
    if seen != names:
        raise ValueError("unexpected " + key + " dependency closure")
    _check_requirements(root, requirements, expected, key)


def load_quality_lock(root: Path, *, policy: dict | None = None) -> dict:
    if policy is None:
        policy = json.loads((root / "metadata/dependencies.json").read_text(encoding="utf-8"))
    selection = policy.get("quality_tools")
    if not isinstance(selection, dict) or selection.get("lock") != "metadata/quality-tools.json":
        raise ValueError("dependency policy has no maintained quality-tool lock")
    raw = (root / selection["lock"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != selection.get("sha256"):
        raise ValueError("quality-tool lock differs from the shared dependency policy")
    lock = json.loads(raw)
    if lock.get("schema") != "rns8_quality_tools_v1" or lock.get("owner_issue") != 60:
        raise ValueError("unsupported quality-tool policy owner/schema")
    components = {row["name"]: row for row in policy["components"]}
    tools = lock.get("tools")
    if (
        not isinstance(tools, list)
        or {row.get("name") for row in tools}
        != {"ruff", "clang-format", "actionlint", "shellcheck"}
        or len(tools) != 4
    ):
        raise ValueError("maintained quality-tool inventory must contain four unique tools")
    expected = {}
    for tool in tools:
        name, version = tool["name"], tool["version"]
        if components.get(name, {}).get("selected", {}).get("version") != version:
            raise ValueError("quality-tool version differs from shared dependency policy: " + name)
        artifacts = tool["artifacts"]
        if len(artifacts) != 2 or {row["platform"] for row in artifacts} != {
            "windows-amd64",
            "linux-x86_64",
        }:
            raise ValueError("quality tool must lock Windows AMD64 and Linux x86_64 artifacts")
        for artifact in artifacts:
            if (
                not re.fullmatch(r"[0-9a-f]{64}", artifact["sha256"])
                or not artifact["url"].startswith("https://")
                or Path(artifact["filename"]).name != artifact["filename"]
            ):
                raise ValueError("quality tool lacks a valid immutable HTTPS artifact")
        if tool["distribution"] == "wheel":
            expected[name + "==" + version] = {row["sha256"] for row in artifacts}
        elif tool["distribution"] != "native_archive" or any(
            not row.get("member") for row in artifacts
        ):
            raise ValueError("unsupported quality artifact distribution")
    if lock.get("requirements") != "scripts/requirements-quality-tools.txt":
        raise ValueError("unsupported quality requirements file")
    _check_requirements(root, lock["requirements"], expected, "quality")
    _wheel_closure(
        root,
        lock,
        components,
        key="type_checking",
        checker="mypy",
        requirements="scripts/requirements-type-tools.txt",
        names={
            "mypy",
            "typing_extensions",
            "mypy_extensions",
            "pathspec",
            "librt",
            "ast-serialize",
        },
    )
    _wheel_closure(
        root,
        lock,
        components,
        key="analysis",
        checker="clang-tidy",
        requirements="scripts/requirements-analysis-tools.txt",
        names={"clang-tidy", "ruamel.yaml"},
    )
    _wheel_closure(
        root,
        lock,
        components,
        key="coverage",
        checker="coverage",
        requirements="scripts/requirements-coverage-tools.txt",
        names={"coverage"},
    )
    return lock
