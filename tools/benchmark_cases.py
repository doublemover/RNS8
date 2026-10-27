#!/usr/bin/env python3
"""Expand compact benchmark catalogs into byte-checked, untracked case files."""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "rns8_benchmark_catalog_v1"


def safe_relative(name: str) -> Path:
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_./-]+", name):
        raise ValueError("invalid case path")
    parts = name.split("/")
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
    if PurePosixPath(name).is_absolute() or any(p in ("", ".", "..") or p.endswith(".") or p.split(".")[0].upper() in reserved for p in parts):
        raise ValueError("unsafe case path")
    return Path(*parts)


def expand_value(value, depth=0):
    if depth > 64:
        raise ValueError("catalog nesting limit")
    if isinstance(value, list):
        return [expand_value(item, depth + 1) for item in value]
    if not isinstance(value, dict):
        return value
    if set(value) == {"$rns8_records"}:
        table = value["$rns8_records"]
        layouts, defaults, rows = (table[k] for k in ("layouts", "defaults", "rows"))
        if len(rows) > 100000 or not layouts or any(len(set(keys)) != len(keys) or not set(defaults).issubset(keys) for keys in layouts):
            raise ValueError("invalid record layouts")
        output = []
        for row in rows:
            if not row or type(row[0]) is not int or not 0 <= row[0] < len(layouts):
                raise ValueError("invalid record layout index")
            keys = layouts[row[0]]
            columns = [key for key in keys if key not in defaults]
            if len(row) != len(columns) + 1:
                raise ValueError("invalid table row")
            fields = {**defaults, **dict(zip(columns, row[1:]))}
            output.append({key: expand_value(fields[key], depth + 1) for key in keys})
        return output
    return {key: expand_value(item, depth + 1) for key, item in value.items()}


def catalog_documents(catalog: dict):
    if catalog.get("schema") != SCHEMA:
        raise ValueError("unsupported benchmark catalog")
    family = safe_relative(catalog["family"])
    if len(family.parts) != 1:
        raise ValueError("catalog family must be one directory")
    for group in catalog["groups"]:
        keys, columns, defaults = (group[k] for k in ("keys", "columns", "defaults"))
        if len(set(keys)) != len(keys) or len(set(columns)) != len(columns) or set(defaults) & set(columns) or set(keys) != set(defaults) | set(columns):
            raise ValueError("invalid document columns")
        for row in group["rows"]:
            if len(row) != len(columns) + 2:
                raise ValueError("invalid document row")
            name, expected, *values = row
            relative = family / safe_relative(name)
            if relative.suffix != ".json" or not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
                raise ValueError("invalid document identity")
            fields = {**defaults, **dict(zip(columns, values))}
            value = {key: expand_value(fields[key]) for key in keys}
            if group.get("root", "object") == "array" and keys == ["items"]:
                value = value["items"]
                if not isinstance(value, list):
                    raise ValueError("array document must expand to a list")
            elif group.get("root", "object") != "object":
                raise ValueError("invalid document root")
            data = (json.dumps(value, indent=2) + "\n").encode("utf-8")
            if len(data) > 16 * 1024 * 1024 or hashlib.sha256(data).hexdigest() != expected:
                raise ValueError(f"expanded case digest mismatch: {relative}")
            yield relative, data


def source_files(root: Path):
    catalogs = sorted((root / "benchmarks/cases").glob("*.json"))
    if not catalogs:
        raise ValueError("no benchmark catalogs found")
    retained = sorted(p for p in (root / "benchmarks").rglob("*.json") if p.parent != root / "benchmarks/cases")
    return catalogs, retained


def read_published(path: Path) -> bytes | None:
    # Windows can briefly deny reads while another immutable publisher closes
    # its replacement handle. Retry that condition, but never trust unread bytes.
    for attempt in range(8):
        try:
            return path.read_bytes()
        except FileNotFoundError:
            return None
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(0.002 * (1 << attempt))


def materialize(output: Path, root: Path = ROOT) -> dict:
    catalogs, retained = source_files(root)
    documents = {}
    for path in catalogs:
        for relative, data in catalog_documents(json.loads(path.read_text(encoding="utf-8"))):
            if relative in documents:
                raise ValueError(f"duplicate case: {relative}")
            documents[relative] = data
    for path in retained:
        relative = path.relative_to(root / "benchmarks")
        safe_relative(relative.as_posix())
        if relative in documents:
            raise ValueError(f"retained case duplicates catalog: {relative}")
        documents[relative] = path.read_bytes().replace(b"\r\n", b"\n")
    if len(documents) > 20000 or sum(map(len, documents.values())) > 64 * 1024 * 1024:
        raise ValueError("benchmark corpus size limit")
    output = output.resolve()
    if output.is_relative_to(root.resolve()) and (output == root.resolve() or output.relative_to(root.resolve()).parts[0] not in ("temp", "build", "out")):
        raise ValueError("in-repository case output must use temp/, build/ or out/")
    output.mkdir(parents=True, exist_ok=True)
    parents = {}
    for relative, data in documents.items():
        # Resolve each directory once instead of walking the filesystem for
        # every case. Files may be atomically replaced by concurrent readers.
        if relative.parent not in parents:
            parent = output / relative.parent
            parent.mkdir(parents=True, exist_ok=True)
            parent = parent.resolve()
            if not parent.is_relative_to(output):
                raise ValueError("case destination escapes output")
            parents[relative.parent] = parent
        target = parents[relative.parent] / relative.name
        if target.is_symlink() and not target.resolve().is_relative_to(output):
            raise ValueError("case destination escapes output")
        # Match the repository's native checkout line endings, preserving the
        # historical Windows collector hashes as well as canonical Git hashes.
        native = data.replace(b"\n", b"\r\n") if os.name == "nt" else data
        if read_published(target) == native:
            continue
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(native)
            try:
                os.replace(temporary, target)
            except PermissionError:
                # Concurrent immutable publishers can win while another
                # Windows process holds a reader that forbids replacement.
                # Accept only independently verified identical destination
                # bytes; a different/absent value remains a real failure.
                if read_published(target) != native:
                    raise
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return dict(cases=len(documents), catalogs=len(catalogs), retained_files=len(retained), output=str(output))


@lru_cache(maxsize=1)
def case_root() -> Path:
    catalogs, retained = source_files(ROOT)
    digest = hashlib.sha256()
    for path in catalogs + retained:
        digest.update(path.relative_to(ROOT).as_posix().encode() + b"\0" + path.read_bytes())
    output = ROOT / "temp/benchmark-cases" / digest.hexdigest()[:20]
    materialize(output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "temp/benchmark-cases/current")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = materialize(args.output)
    print(json.dumps(report, indent=2) if args.json else report["output"])


if __name__ == "__main__":
    main()
