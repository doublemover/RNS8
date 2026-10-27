#!/usr/bin/env python3
"""Fetch and verify the full research archive, then restore it outside sources."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from urllib.request import urlopen
import zipfile

from benchmark_cases import ROOT, safe_relative


def restore(archive: Path, output: Path, manifest: dict) -> dict:
    if manifest.get("schema") != "rns8_research_evidence_archive_v1":
        raise ValueError("unsupported research archive manifest")
    if archive.stat().st_size != manifest["bytes"]:
        raise ValueError("archive size mismatch")
    with archive.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != manifest["sha256"]:
            raise ValueError("archive SHA256 mismatch")
    output = output.resolve()
    if output.is_relative_to(ROOT) and (output == ROOT or output.relative_to(ROOT).parts[0] not in ("temp", "build", "out")):
        raise ValueError("in-repository restore must use temp/, build/ or out/")
    with zipfile.ZipFile(archive) as container:
        infos = container.infolist()
        if len(infos) > 10000 or sum(info.file_size for info in infos) > 128 * 1024 * 1024:
            raise ValueError("archive expansion limit")
        names = [info.filename for info in infos]
        if len(names) != len(set(names)) or any(info.is_dir() or ((info.external_attr >> 16) & 0o170000) == 0o120000 for info in infos):
            raise ValueError("archive contains duplicate or non-file entries")
        index = json.loads(container.read(manifest["index_member"]))
        if index.get("schema") != "rns8_research_archive_index_v1" or index.get("source_commit") != manifest["source_commit"]:
            raise ValueError("archive source/index mismatch")
        records = index["files"]
        if len(records) != len(names) - 1 or {r["path"] for r in records} != set(names) - {manifest["index_member"]}:
            raise ValueError("archive index membership mismatch")
        # Validate every member before writing any restored file.
        verified = []
        for record in records:
            relative = safe_relative(record["path"])
            target = output / relative
            if not target.resolve().is_relative_to(output):
                raise ValueError("archive destination escapes output")
            data = container.read(record["path"])
            if len(data) != record["bytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
                raise ValueError("member identity mismatch: " + record["path"])
            verified.append((target, data))
    for target, data in verified:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(data)
            os.replace(temporary, target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return dict(files=len(verified), source_commit=manifest["source_commit"], output=str(output), archive_sha256=manifest["sha256"])


def fetch(manifest: dict, destination: Path):
    expected = manifest["bytes"]
    if type(expected) is not int or not 0 < expected <= 128 * 1024 * 1024 or not manifest["url"].startswith("https://github.com/doublemover/RNS8/releases/download/"):
        raise ValueError("invalid research archive download")
    temporary = None
    try:
        with urlopen(manifest["url"], timeout=30) as response, tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
            temporary = Path(stream.name)
            if not response.geturl().startswith("https://"):
                raise ValueError("archive download left HTTPS")
            received = 0
            digest = hashlib.sha256()
            while chunk := response.read(min(1024 * 1024, expected + 1 - received)):
                received += len(chunk)
                if received > expected:
                    raise ValueError("archive download exceeds pinned size")
                digest.update(chunk)
                stream.write(chunk)
        if received != expected or digest.hexdigest() != manifest["sha256"]:
            raise ValueError("download identity mismatch")
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "metadata/research-evidence-20261026.json")
    parser.add_argument("--archive", type=Path, help="use an existing archive without network access")
    parser.add_argument("--output", type=Path, default=ROOT / "temp/restored-pr15-evidence")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    archive = args.archive
    if archive is None:
        cache = ROOT / "temp/research-evidence-cache"
        cache.mkdir(parents=True, exist_ok=True)
        archive = cache / (manifest["sha256"] + ".zip")
        if not archive.exists():
            fetch(manifest, archive)
    print(json.dumps(restore(archive, args.output, manifest), indent=2))


if __name__ == "__main__":
    main()
