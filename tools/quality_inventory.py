"""Deterministic tracked first-party roles shared by quality/analysis gates."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
import subprocess
from typing import Iterable, Literal

Role = Literal["cpp", "python", "bash", "workflow"]
ROLES: tuple[Role, ...] = ("cpp", "python", "bash", "workflow")
ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class SourceFile:
    path: str
    role: Role


def classify(paths: Iterable[str], policy: dict) -> tuple[SourceFile, ...]:
    if policy.get("schema") != "rns8_first_party_inventory_v1":
        raise ValueError("unsupported first-party inventory policy")
    result = []
    seen = set()
    for name in sorted(paths):
        path = PurePosixPath(name)
        if (
            not name
            or path.is_absolute()
            or ".." in path.parts
            or "\\" in name
            or name in seen
        ):
            raise ValueError("invalid or duplicate tracked inventory path: " + name)
        seen.add(name)
        if (path.parts[0] in policy["excluded_roots"] or name in policy["generated"]
                or name in policy.get("checker_negative_inputs", {})):
            continue
        if name.startswith(".github/workflows/") and path.suffix in {".yml", ".yaml"}:
            result.append(SourceFile(name, "workflow"))
            continue
        if path.parts[0] not in policy["roots"]:
            continue
        role: Role | None = None
        if path.suffix in policy["cpp_suffixes"]:
            role = "cpp"
        elif path.suffix == ".inc":
            if path.parts[0] not in policy["cpp_fragment_roots"]:
                raise ValueError("unclassified textual fragment: " + name)
            role = "cpp"
        elif path.suffix == ".py":
            role = "python"
        elif path.suffix == ".sh":
            role = "bash"
        if role is not None:
            result.append(SourceFile(name, role))
    return tuple(result)


def inventory(root: Path = ROOT) -> tuple[SourceFile, ...]:
    policy = json.loads(
        (root / "metadata/first-party-inventory.json").read_text(encoding="utf-8")
    )
    output = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    ).stdout
    if not output or output[-1:] != b"\0":
        raise ValueError("tracked file inventory is empty or not NUL-delimited")
    records = classify(output[:-1].decode("utf-8", errors="strict").split("\0"), policy)
    if any(not any(row.role == role for row in records) for role in ROLES):
        raise ValueError(
            "first-party inventory unexpectedly omits a required source role"
        )
    for row in records:
        path = root / row.path
        if (
            path.is_symlink()
            or not path.is_file()
            or not path.resolve().is_relative_to(root.resolve())
        ):
            raise ValueError(
                "first-party source is missing, linked or outside repository: "
                + row.path
            )
    return records


def batches(
    paths: Iterable[str], *, maximum_bytes=16000
) -> tuple[tuple[str, ...], ...]:
    """Conservative UTF-16 Windows command limit, also safe for POSIX argv."""
    result: list[tuple[str, ...]] = []
    current: list[str] = []
    size = 0
    if type(maximum_bytes) is not int or maximum_bytes < 1:
        raise ValueError("positive argument batch budget required")
    for path in paths:
        # Quoting/escaping allowance, including non-BMP surrogate pairs.
        cost = 2 * len(path.encode("utf-16-le")) + 8
        if cost > maximum_bytes:
            raise ValueError("one tracked path exceeds argument budget")
        if current and size + cost > maximum_bytes:
            result.append(tuple(current))
            current, size = [], 0
        current.append(path)
        size += cost
    if current:
        result.append(tuple(current))
    return tuple(result)
