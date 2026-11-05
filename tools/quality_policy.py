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
    typing = lock.get("type_checking")
    if (
        not isinstance(typing, dict)
        or typing.get("checker") != "mypy"
        or typing.get("requirements") != "scripts/requirements-type-tools.txt"
    ):
        raise ValueError("quality lock has no maintained Python type-checker closure")
    expected = {}
    names = set()
    for wheel in typing["wheels"]:
        name, version = wheel["name"], wheel["version"]
        if name in names or components.get(name, {}).get("selected", {}).get("version") != version:
            raise ValueError("type-tool version differs from shared dependency policy")
        names.add(name)
        if len(wheel["artifacts"]) != 2 or {row["platform"] for row in wheel["artifacts"]} != {
            "windows-amd64",
            "linux-x86_64",
        }:
            raise ValueError("type-tool wheel platform inventory is incomplete")
        for artifact in wheel["artifacts"]:
            if (
                not re.fullmatch(r"[0-9a-f]{64}", artifact["sha256"])
                or not artifact["url"].startswith("https://")
                or not artifact["filename"].endswith(".whl")
            ):
                raise ValueError("type-tool wheel lacks immutable HTTPS identity")
        expected[name + "==" + version] = {row["sha256"] for row in wheel["artifacts"]}
    if names != {
        "mypy",
        "typing_extensions",
        "mypy_extensions",
        "pathspec",
        "librt",
        "ast-serialize",
    }:
        raise ValueError("unexpected type-checker dependency closure")
    _check_requirements(root, typing["requirements"], expected, "type-tool")
    analysis = lock.get("analysis")
    if (
        not isinstance(analysis, dict)
        or analysis.get("checker") != "clang-tidy"
        or analysis.get("requirements") != "scripts/requirements-analysis-tools.txt"
    ):
        raise ValueError("quality lock has no maintained native analyzer closure")
    names, expected = set(), {}
    for wheel in analysis["wheels"]:
        name, version = wheel["name"], wheel["version"]
        if name in names or components.get(name, {}).get("selected", {}).get("version") != version:
            raise ValueError("analyzer-tool version differs from shared dependency policy")
        names.add(name)
        if len(wheel["artifacts"]) != 2 or {row["platform"] for row in wheel["artifacts"]} != {
            "windows-amd64",
            "linux-x86_64",
        }:
            raise ValueError("analyzer-tool platform inventory is incomplete")
        for artifact in wheel["artifacts"]:
            if (
                not re.fullmatch(r"[0-9a-f]{64}", artifact["sha256"])
                or not artifact["url"].startswith("https://")
                or not artifact["filename"].endswith(".whl")
            ):
                raise ValueError("analyzer wheel lacks immutable HTTPS identity")
        expected[name + "==" + version] = {row["sha256"] for row in wheel["artifacts"]}
    if names != {"clang-tidy", "ruamel.yaml"}:
        raise ValueError("unexpected analyzer dependency closure")
    _check_requirements(root, analysis["requirements"], expected, "analyzer")
    return lock
