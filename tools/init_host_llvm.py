#!/usr/bin/env python3
"""Plan or explicitly install the reviewed standalone host LLVM archive."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys
import tarfile
import zipfile

from check_dependencies_lib.resolution import binary_architecture
from check_dependencies_lib.version_policy import load_policy
import native_archive

ROOT = Path(__file__).resolve().parents[1]
MAX_DOWNLOAD = 3 * 1024**3
MAX_EXTRACT = 32 * 1024**3
MAX_MEMBERS = 250_000
RECEIPT = "rns8-host-llvm.json"
SCHEMA = "rns8_native_host_llvm_v1"
TOOLS = (
    "clang",
    "clang++",
    "llvm-nm",
    "llvm-ar",
    "llvm-symbolizer",
    "llvm-profdata",
    "llvm-cov",
    "llvm-readobj",
    "llvm-objdump",
)


def selected_artifact(root: Path = ROOT) -> dict:
    host, machine = platform.system(), platform.machine().lower()
    if host not in {"Windows", "Linux"} or machine not in {"amd64", "x86_64"}:
        raise ValueError("Host LLVM archive lock supports native Windows AMD64 and Linux x86_64")
    policy = load_policy(root / "metadata/dependencies.json")
    key = "windows-amd64" if host == "Windows" else "linux-x86_64"
    artifact = next(
        row for row in policy["bootstrap"]["host_llvm_artifacts"] if row["platform"] == key
    )
    if artifact["version"] != policy["candidate_refresh_20261105"]["host_llvm_latest"]["version"]:
        raise ValueError("Host LLVM archive and reviewed compiler candidate differ")
    return artifact


def default_root() -> Path:
    if value := os.environ.get("RNS8_HOST_LLVM_ROOT"):
        return Path(value).expanduser()
    if platform.system() == "Windows":
        if not (local := os.environ.get("LOCALAPPDATA")):
            raise ValueError("LOCALAPPDATA is absent; set RNS8_HOST_LLVM_ROOT")
        return Path(local) / "RNS8/dependencies/host-llvm"
    return Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "rns8/host-llvm"


def binaries(prefix: Path, artifact: dict) -> dict[str, str]:
    windows = artifact["platform"] == "windows-amd64"
    suffix = ".exe" if windows else ""
    names = (*TOOLS, "clang-cl", "lld-link") if windows else (*TOOLS, "ld.lld")
    result = {name: str(prefix / "bin" / (name + suffix)) for name in names}
    for name, raw in result.items():
        path = Path(raw)
        if not path.is_file():
            raise ValueError("Host LLVM prefix is missing native tool: " + name)
        with path.open("rb") as source:
            magic = source.read(4)
        if (
            not (magic.startswith(b"MZ") if windows else magic == b"\x7fELF")
            or binary_architecture(path) != "x86_64"
        ):
            raise ValueError("Host LLVM tool has the wrong OS format/architecture: " + name)
        if not windows and not os.access(path, os.X_OK):
            raise ValueError("Host LLVM native tool lacks executable permissions: " + name)
    resource = prefix / "lib/clang" / artifact["version"].split(".")[0]
    if not (resource / "include/stddef.h").is_file():
        raise ValueError("Host LLVM prefix lacks its matching builtin-header resource layout")
    return dict(
        root=str(prefix), bin=str(prefix / "bin"), expected_resource_layout=str(resource), **result
    )


def discover(root: Path | None = None, *, artifact: dict | None = None) -> dict[str, str] | None:
    artifact = selected_artifact() if artifact is None else artifact
    return native_archive.discover(
        native_archive.destination(default_root() if root is None else root, "llvm", artifact),
        artifact,
        receipt_name=RECEIPT,
        schema=SCHEMA,
        validate=binaries,
        max_bytes=MAX_EXTRACT,
        max_members=MAX_MEMBERS,
    )


def download(artifact: dict, destination: Path) -> None:
    native_archive.download(artifact, destination, max_bytes=MAX_DOWNLOAD, seconds=1800)


def install(root: Path | None = None) -> dict[str, str]:
    return native_archive.install(
        default_root() if root is None else root,
        selected_artifact(),
        namespace="llvm",
        receipt_name=RECEIPT,
        schema=SCHEMA,
        validate=binaries,
        downloader=download,
        max_bytes=MAX_EXTRACT,
        max_members=MAX_MEMBERS,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="Managed host compiler parent directory")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--install",
        action="store_true",
        help="Explicitly download/extract or verify/reuse the reviewed prefix",
    )
    modes.add_argument(
        "--discover",
        action="store_true",
        help="Verify an already installed prefix without downloading",
    )
    args = parser.parse_args(argv)
    try:
        artifact = selected_artifact()
        root = (default_root() if args.root is None else args.root).resolve()
        target = native_archive.destination(root, "llvm", artifact)
        result = dict(
            schema="rns8_host_llvm_bootstrap_plan_v1",
            artifact=artifact,
            installation=str(target),
            prefix=str(target / artifact["prefix"]),
            mode="install" if args.install else "discover" if args.discover else "plan_only",
            max_download_bytes=MAX_DOWNLOAD,
            max_expanded_bytes=MAX_EXTRACT,
            max_members=MAX_MEMBERS,
            limits_are_budgets_not_estimated_disk_requirements=True,
            changes_system_configuration=False,
            selects_hip_compiler=False,
            qualification="not_project_qualified",
        )
        if args.install or args.discover:
            result["paths"] = install(root) if args.install else discover(root, artifact=artifact)
            result["status"] = (
                "installed_integrity_verified" if result["paths"] else "not_installed"
            )
        else:
            result["status"] = "not_executed"
        print(json.dumps(result, indent=2))
        return 0
    except (OSError, ValueError, StopIteration, tarfile.TarError, zipfile.BadZipFile) as error:
        print("RNS8 host LLVM: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
