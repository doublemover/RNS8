#!/usr/bin/env python3
"""Install the reviewed native CMake prefix without changing system settings."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tarfile
import tempfile
import time
import urllib.parse
import urllib.request
import zipfile

from archive_install import extract_tar, extract_zip
from check_dependencies_lib.version_policy import load_policy

ROOT = Path(__file__).resolve().parents[1]
MAX_DOWNLOAD = 256 * 1024**2
MAX_EXTRACT = 1024**3
MAX_MEMBERS = 50_000
RECEIPT = "rns8-build-tools.json"


def selected_artifact(root: Path = ROOT) -> dict:
    host, machine = platform.system(), platform.machine().lower()
    if host not in {"Windows", "Linux"} or machine not in {"amd64", "x86_64"}:
        raise ValueError("Native CMake lock supports Windows AMD64 and Linux x86_64")
    policy = load_policy(root / "metadata/dependencies.json")
    key = "windows-amd64" if host == "Windows" else "linux-x86_64"
    artifact = next(
        row for row in policy["bootstrap"]["native_artifacts"] if row["platform"] == key
    )
    selected = next(row for row in policy["components"] if row["name"] == "cmake")["selected"]
    if artifact["version"] != selected["version"]:
        raise ValueError("Native CMake archive differs from the selected component version")
    return artifact


def default_root() -> Path:
    if value := os.environ.get("RNS8_BUILD_TOOLS_ROOT"):
        return Path(value).expanduser()
    if platform.system() == "Windows":
        if not (local := os.environ.get("LOCALAPPDATA")):
            raise ValueError("LOCALAPPDATA is absent; set RNS8_BUILD_TOOLS_ROOT")
        return Path(local) / "RNS8/dependencies/build-tools"
    return Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "rns8/build-tools"


def installation(root: Path, artifact: dict) -> Path:
    return root / ("cmake-" + artifact["version"] + "-" + artifact["platform"])


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def prefix_files(prefix: Path) -> dict[str, str]:
    """Receipts cover the complete prefix, including modules and native binaries."""
    result = {}
    expanded = 0
    for path in prefix.rglob("*"):
        if len(result) >= MAX_MEMBERS:
            raise ValueError("Native CMake prefix exceeds member budget")
        relative = path.relative_to(prefix).as_posix()
        if path.is_symlink():
            if not path.resolve().is_relative_to(prefix.resolve()) or not path.exists():
                raise ValueError("Native CMake prefix contains an escaping or dangling symlink")
            result[relative] = "symlink:" + path.readlink().as_posix()
        elif path.is_file():
            expanded += path.stat().st_size
            if expanded > MAX_EXTRACT:
                raise ValueError("Native CMake prefix exceeds byte budget")
            result[relative] = file_hash(path)
        elif path.is_dir():
            result[relative] = "directory"
        else:
            raise ValueError("Native CMake prefix contains a special file")
    return dict(sorted(result.items()))


def binaries(prefix: Path, artifact: dict) -> dict[str, str]:
    suffix = ".exe" if artifact["platform"] == "windows-amd64" else ""
    result = {name: str(prefix / "bin" / (name + suffix)) for name in ("cmake", "ctest", "cpack")}
    if any(not Path(path).is_file() for path in result.values()):
        raise ValueError("Native CMake prefix is missing CMake/CTest/CPack")
    if artifact["platform"] == "linux-x86_64" and any(
        not os.access(path, os.X_OK) for path in result.values()
    ):
        raise ValueError("Native CMake binaries lack executable permissions")
    modules = prefix / "share" / ("cmake-" + ".".join(artifact["version"].split(".")[:2]))
    if not (modules / "Modules/CMake.cmake").is_file():
        raise ValueError("Native CMake prefix is missing its matching module tree")
    return result


def discover(root: Path | None = None, *, artifact: dict | None = None) -> dict[str, str] | None:
    artifact = selected_artifact() if artifact is None else artifact
    destination = installation(default_root() if root is None else root, artifact)
    if not destination.exists():
        return None
    if destination.is_symlink() or not destination.is_dir():
        raise ValueError("Native CMake installation is not an owned directory")
    receipt = json.loads((destination / RECEIPT).read_text(encoding="utf-8"))
    prefix = destination / artifact["prefix"]
    if prefix.is_symlink() or not prefix.is_dir():
        raise ValueError("Native CMake prefix is not an owned directory")
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema") != "rns8_native_build_tools_v1"
        or receipt.get("artifact") != artifact
        or receipt.get("files") != prefix_files(prefix)
    ):
        raise ValueError("Native CMake receipt or prefix integrity mismatch; no overlay attempted")
    return binaries(prefix, artifact)


def download(artifact: dict, destination: Path) -> None:
    digest, size = hashlib.sha256(), 0
    deadline = time.monotonic() + 300
    with (
        urllib.request.urlopen(artifact["url"], timeout=60) as source,
        destination.open("xb") as output,
    ):
        endpoint = urllib.parse.urlsplit(source.geturl())
        if endpoint.scheme != "https" or endpoint.hostname not in {
            "github.com",
            "release-assets.githubusercontent.com",
            "objects.githubusercontent.com",
        }:
            raise ValueError("Native CMake download redirected outside the reviewed release hosts")
        while chunk := source.read(1024**2):
            size += len(chunk)
            if size > MAX_DOWNLOAD or time.monotonic() > deadline:
                raise ValueError("Native CMake download exceeds byte/time budget")
            digest.update(chunk)
            output.write(chunk)
    if not size or digest.hexdigest() != artifact["sha256"]:
        raise ValueError("Native CMake archive SHA256 mismatch")


def install(root: Path | None = None) -> dict[str, str]:
    artifact = selected_artifact()
    root = (default_root() if root is None else root).resolve()
    if existing := discover(root, artifact=artifact):
        return existing
    root.mkdir(parents=True, exist_ok=True)
    destination = installation(root, artifact)
    # Cooperating initializers cannot publish competing prefixes. Leave unrelated
    # or failed installations intact and report them instead of repairing in place.
    lock = root / (destination.name + ".install-lock")
    lock.mkdir()
    try:
        if existing := discover(root, artifact=artifact):
            return existing
        with tempfile.TemporaryDirectory(prefix=".cmake-stage-", dir=root) as scratch:
            staging = Path(scratch)
            archive = staging / artifact["filename"]
            download(artifact, archive)
            payload = staging / "payload"
            extractor = extract_zip if archive.suffix == ".zip" else extract_tar
            extractor(
                archive,
                payload,
                "Windows" if artifact["platform"] == "windows-amd64" else "Linux",
                max_bytes=MAX_EXTRACT,
                max_members=MAX_MEMBERS,
            )
            prefix = payload / artifact["prefix"]
            if list(payload.iterdir()) != [prefix]:
                raise ValueError("Native CMake archive must contain exactly its reviewed prefix")
            binaries(prefix, artifact)
            receipt = dict(
                schema="rns8_native_build_tools_v1",
                artifact=artifact,
                files=prefix_files(prefix),
                qualification="installed_not_project_qualified",
            )
            with (payload / RECEIPT).open("x", encoding="utf-8") as output:
                output.write(json.dumps(receipt, indent=2) + "\n")
            if destination.exists():
                raise ValueError("Native CMake destination appeared during installation")
            payload.rename(destination)
        return binaries(destination / artifact["prefix"], artifact)
    finally:
        lock.rmdir()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="Managed build-tool directory")
    parser.add_argument(
        "--github-path", action="store_true", help="Append the selected bin to GITHUB_PATH"
    )
    args = parser.parse_args(argv)
    try:
        if args.github_path and not os.environ.get("GITHUB_PATH"):
            raise ValueError("--github-path requires the CI GITHUB_PATH file")
        paths = install(args.root)
        if args.github_path:
            with Path(os.environ["GITHUB_PATH"]).open("a", encoding="utf-8") as output:
                output.write(str(Path(paths["cmake"]).parent) + "\n")
        print(json.dumps(paths, indent=2))
        return 0
    except (OSError, ValueError, StopIteration, tarfile.TarError, zipfile.BadZipFile) as error:
        print("RNS8 build tools: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
