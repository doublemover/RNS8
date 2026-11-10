#!/usr/bin/env python3
"""Install the reviewed native CMake prefix without changing system settings."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys
import tarfile
import zipfile

import native_archive
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
    return native_archive.destination(root, "cmake", artifact)


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
    return native_archive.discover(
        installation(default_root() if root is None else root, artifact),
        artifact,
        receipt_name=RECEIPT,
        schema="rns8_native_build_tools_v1",
        validate=binaries,
        max_bytes=MAX_EXTRACT,
        max_members=MAX_MEMBERS,
    )


def download(artifact: dict, destination: Path) -> None:
    native_archive.download(artifact, destination, max_bytes=MAX_DOWNLOAD, seconds=300)


def install(root: Path | None = None) -> dict[str, str]:
    return native_archive.install(
        default_root() if root is None else root,
        selected_artifact(),
        namespace="cmake",
        receipt_name=RECEIPT,
        schema="rns8_native_build_tools_v1",
        validate=binaries,
        downloader=download,
        max_bytes=MAX_EXTRACT,
        max_members=MAX_MEMBERS,
    )


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
