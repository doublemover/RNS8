#!/usr/bin/env python3
"""Explicitly install hash-locked quality tools into a fresh ignored environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import stat
import subprocess
import sys
import tarfile
import time
import urllib.request
import venv
import zipfile

from quality_inventory import ROOT
from quality_policy import load_quality_lock

ARCHIVE_LIMIT = 32 * 1024**2
BINARY_LIMIT = 64 * 1024**2


def archive_executable(path: Path, member: str) -> bytes:
    """Read only a reviewed regular member; never extract archive paths."""
    if path.stat().st_size > ARCHIVE_LIMIT:
        raise ValueError("quality archive exceeds byte budget")
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            if len(archive.infolist()) > 4096:
                raise ValueError("quality archive exceeds entry budget")
            entries = [row for row in archive.infolist() if row.filename == member]
            if len(entries) != 1:
                raise ValueError("native archive member is absent or duplicated")
            entry = entries[0]
            mode = entry.external_attr >> 16
            if entry.is_dir() or stat.S_ISLNK(mode) or entry.file_size > BINARY_LIMIT:
                raise ValueError(
                    "native archive member is not a bounded regular executable"
                )
            return archive.read(entry)
    payload, count, expanded = None, 0, 0
    with tarfile.open(path, "r|gz") as archive:
        for entry in archive:
            count += 1
            expanded += max(0, entry.size) + 512
            if count > 4096 or expanded > 2 * BINARY_LIMIT:
                raise ValueError(
                    "quality archive exceeds expanded byte or entry budget"
                )
            if entry.name != member:
                continue
            if payload is not None or not entry.isfile() or entry.size > BINARY_LIMIT:
                raise ValueError(
                    "native archive member is not one bounded regular executable"
                )
            stream = archive.extractfile(entry)
            if stream is None:
                raise ValueError("native executable cannot be read")
            with stream:
                payload = stream.read(BINARY_LIMIT + 1)
    if payload is None:
        raise ValueError("native archive member is absent")
    return payload


def install(out: Path) -> Path:
    system, machine = platform.system(), platform.machine().lower()
    if system not in {"Windows", "Linux"} or machine not in {"amd64", "x86_64"}:
        raise ValueError(
            "quality artifact lock supports Windows AMD64 and Linux x86_64"
        )
    out = out.resolve()
    if (
        not out.is_relative_to((ROOT / "temp").resolve())
        or out == (ROOT / "temp").resolve()
        or out.exists()
    ):
        raise ValueError(
            "quality environment needs a previously absent child of repository temp/"
        )
    lock = load_quality_lock(ROOT)
    if sys.version_info[:2] != (3, 15):
        raise ValueError("locked quality/type-tool wheels require the selected CPython 3.15 development lane")
    out.parent.mkdir(parents=True, exist_ok=True)
    venv.EnvBuilder(with_pip=True, clear=False).create(out)
    binary_dir = out / ("Scripts" if system == "Windows" else "bin")
    interpreter = binary_dir / ("python.exe" if system == "Windows" else "python")
    subprocess.run(
        [
            str(interpreter),
            "-m",
            "pip",
            "install",
            "--require-hashes",
            "--only-binary=:all:",
            "--no-deps",
            "-r",
            str(ROOT / lock["requirements"]),
            "-r",
            str(ROOT / lock["type_checking"]["requirements"]),
        ],
        cwd=ROOT,
        check=True,
        timeout=600,
    )
    platform_key = "windows-amd64" if system == "Windows" else "linux-x86_64"
    downloads = out / "downloads"
    downloads.mkdir()
    artifacts = []
    for tool in lock["tools"]:
        if tool["distribution"] != "native_archive":
            continue
        artifact = next(
            row for row in tool["artifacts"] if row["platform"] == platform_key
        )
        destination = downloads / artifact["filename"]
        digest = hashlib.sha256()
        size = 0
        deadline = time.monotonic() + 180
        with (
            urllib.request.urlopen(artifact["url"], timeout=60) as response,
            destination.open("xb") as output,
        ):
            if not response.geturl().startswith("https://"):
                raise ValueError("quality artifact redirected outside HTTPS")
            while chunk := response.read(1024**2):
                if time.monotonic() > deadline:
                    raise ValueError("quality archive download time budget exhausted")
                size += len(chunk)
                if size > ARCHIVE_LIMIT:
                    raise ValueError("quality archive exceeds download byte budget")
                digest.update(chunk)
                output.write(chunk)
        if digest.hexdigest() != artifact["sha256"]:
            raise ValueError("quality archive SHA256 mismatch: " + artifact["filename"])
        payload = archive_executable(destination, artifact["member"])
        if not payload or len(payload) > BINARY_LIMIT:
            raise ValueError("quality executable has invalid size")
        binary = binary_dir / (tool["name"] + (".exe" if system == "Windows" else ""))
        with binary.open("xb") as output:
            output.write(payload)
        if os.name != "nt":
            binary.chmod(0o755)
        artifacts.append(
            dict(
                tool=tool["name"],
                artifact=artifact,
                executable=str(binary),
                executable_sha256=hashlib.sha256(payload).hexdigest(),
            )
        )
    (out / "quality-installation.json").write_text(
        json.dumps(
            dict(
                schema="rns8_quality_installation_v1",
                source_lock_sha256=hashlib.sha256(
                    (ROOT / "metadata/quality-tools.json").read_bytes()
                ).hexdigest(),
                artifacts=artifacts,
                qualification="installed_not_validated",
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return binary_dir


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "temp/quality-tools")
    args = parser.parse_args(argv)
    try:
        print(install(args.out))
        return 0
    except (
        OSError,
        ValueError,
        subprocess.SubprocessError,
        tarfile.TarError,
        zipfile.BadZipFile,
    ) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
