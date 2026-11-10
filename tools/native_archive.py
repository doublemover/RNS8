"""Shared bounded publication of reviewed native build-tool archive prefixes."""

from __future__ import annotations

from collections.abc import Callable
import hashlib
import json
from pathlib import Path
import re
import tempfile
import time
import urllib.parse
import urllib.request

from archive_install import archive_key, extract_tar, extract_zip


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def prefix_files(prefix: Path, *, max_bytes: int, max_members: int) -> dict[str, str]:
    """Receipts cover the complete prefix, including modules and native binaries."""
    result = {}
    expanded = 0
    for path in prefix.rglob("*"):
        if len(result) >= max_members:
            raise ValueError("Native archive prefix exceeds member budget")
        relative = path.relative_to(prefix).as_posix()
        if path.is_symlink():
            if not path.resolve().is_relative_to(prefix.resolve()) or not path.exists():
                raise ValueError("Native archive prefix contains an escaping or dangling symlink")
            result[relative] = "symlink:" + path.readlink().as_posix()
        elif path.is_file():
            expanded += path.stat().st_size
            if expanded > max_bytes:
                raise ValueError("Native archive prefix exceeds byte budget")
            result[relative] = file_hash(path)
        elif path.is_dir():
            result[relative] = "directory"
        else:
            raise ValueError("Native archive prefix contains a special file")
    return dict(sorted(result.items()))


def download(artifact: dict, destination: Path, *, max_bytes: int, seconds: int) -> None:
    digest, size = hashlib.sha256(), 0
    deadline = time.monotonic() + seconds
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
            raise ValueError(
                "Native archive download redirected outside the reviewed release hosts"
            )
        while chunk := source.read(1024**2):
            size += len(chunk)
            if size > max_bytes or time.monotonic() > deadline:
                raise ValueError("Native archive download exceeds byte/time budget")
            digest.update(chunk)
            output.write(chunk)
    if (
        not size
        or digest.hexdigest() != artifact["sha256"]
        or ("download_bytes" in artifact and size != artifact["download_bytes"])
    ):
        raise ValueError("Native archive SHA256 mismatch")


def destination(root: Path, namespace: str, artifact: dict) -> Path:
    for value in (
        namespace,
        artifact["version"],
        artifact["platform"],
        artifact["prefix"],
        artifact["filename"],
    ):
        if (
            type(value) is not str
            or not re.fullmatch(r"[a-zA-Z0-9+_.-]+", value)
            or value in {".", ".."}
        ):
            raise ValueError("native archive identities must be single safe path components")
    archive_key(artifact["prefix"], "Windows")
    archive_key(artifact["filename"], "Windows")
    return root / (namespace + "-" + artifact["version"] + "-" + artifact["platform"])


def discover(
    target: Path,
    artifact: dict,
    *,
    receipt_name: str,
    schema: str,
    validate: Callable[[Path, dict], dict[str, str]],
    max_bytes: int,
    max_members: int,
) -> dict[str, str] | None:
    if target.is_symlink():
        raise ValueError("native archive installation cannot be a symlink")
    if not target.exists():
        return None
    if not target.is_dir():
        raise ValueError("native archive installation is not an owned directory")
    receipt_path = target / receipt_name
    if receipt_path.is_symlink() or receipt_path.stat().st_size > 128 * 1024**2:
        raise ValueError("native archive receipt exceeds ownership/byte budget")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    prefix = target / artifact["prefix"]
    if prefix.is_symlink() or not prefix.is_dir():
        raise ValueError("native archive prefix is not an owned directory")
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema") != schema
        or receipt.get("artifact") != artifact
        or receipt.get("files")
        != prefix_files(prefix, max_bytes=max_bytes, max_members=max_members)
    ):
        raise ValueError(
            "native archive receipt or prefix integrity mismatch; no overlay attempted"
        )
    return validate(prefix, artifact)


def install(
    root: Path,
    artifact: dict,
    *,
    namespace: str,
    receipt_name: str,
    schema: str,
    validate: Callable[[Path, dict], dict[str, str]],
    downloader: Callable[[dict, Path], None],
    max_bytes: int,
    max_members: int,
) -> dict[str, str]:
    root = root.resolve()
    target = destination(root, namespace, artifact)
    if type(receipt_name) is not str or not re.fullmatch(r"[a-zA-Z0-9_.-]+\.json", receipt_name):
        raise ValueError("native archive receipt needs a single JSON filename")
    options = dict(
        receipt_name=receipt_name,
        schema=schema,
        validate=validate,
        max_bytes=max_bytes,
        max_members=max_members,
    )
    if existing := discover(target, artifact, **options):
        return existing
    root.mkdir(parents=True, exist_ok=True)
    lock = root / (target.name + ".install-lock")
    lock.mkdir()
    try:
        if existing := discover(target, artifact, **options):
            return existing
        with tempfile.TemporaryDirectory(prefix="." + namespace + "-stage-", dir=root) as scratch:
            staging = Path(scratch)
            archive = staging / artifact["filename"]
            downloader(artifact, archive)
            payload = staging / "payload"
            extractor = extract_zip if archive.suffix == ".zip" else extract_tar
            extractor(
                archive,
                payload,
                "Windows" if artifact["platform"] == "windows-amd64" else "Linux",
                max_bytes=max_bytes,
                max_members=max_members,
            )
            prefix = payload / artifact["prefix"]
            if list(payload.iterdir()) != [prefix] or prefix.is_symlink() or not prefix.is_dir():
                raise ValueError("native archive must contain exactly its reviewed prefix")
            validate(prefix, artifact)
            receipt = dict(
                schema=schema,
                artifact=artifact,
                files=prefix_files(prefix, max_bytes=max_bytes, max_members=max_members),
                qualification="installed_not_project_qualified",
            )
            with (payload / receipt_name).open("x", encoding="utf-8") as output:
                output.write(json.dumps(receipt, indent=2) + "\n")
            if target.exists() or target.is_symlink():
                raise ValueError("native archive destination appeared during installation")
            payload.rename(target)
        return validate(target / artifact["prefix"], artifact)
    finally:
        lock.rmdir()
