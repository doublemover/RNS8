"""Bounded dependency archive extraction into private, empty staging directories."""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tarfile
import zipfile


def prepare_destination(destination: Path) -> None:
    if destination.is_symlink():
        raise ValueError("Archive destination cannot be a symlink")
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise ValueError("Archive destination must be empty; overlays are forbidden")


def archive_key(name: str, host: str) -> str:
    parts = PurePosixPath(name).parts
    if not parts or any(p in ("..", "/") or "\\" in p or ":" in p for p in parts):
        raise ValueError("Unsafe dependency archive path: " + name)
    if host == "Windows" and any(
        p.endswith((".", " "))
        or re.fullmatch(r"(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?", p, re.I)
        for p in parts
    ):
        raise ValueError("Unsafe Windows dependency archive path: " + name)
    key = "/".join(parts)
    return key.casefold() if host == "Windows" else key


def extract_tar(
    archive: Path,
    destination: Path,
    host: str,
    *,
    max_bytes: int = 64 * 1024**3,
    max_members: int = 500_000,
) -> None:
    """Bounded tar extraction into a new empty staging directory, never overlays."""
    prepare_destination(destination)
    count = size = 0
    seen: set[str] = set()
    links = []
    with tarfile.open(archive, "r|*") as source:
        for member in source:
            count += 1
            size += max(0, member.size)
            if count > max_members or size > max_bytes:
                raise ValueError("dependency extraction exceeds member/expanded-byte limit")
            key = archive_key(member.name, host)
            if key in seen:
                raise ValueError("Duplicate dependency archive path: " + member.name)
            seen.add(key)
            if member.size < 0 or not (
                member.isfile() or member.isdir() or member.issym() or member.islnk()
            ):
                raise ValueError("Unsupported dependency archive member: " + member.name)
            if (member.issym() or member.islnk()) and (
                PurePosixPath(member.linkname).is_absolute()
                or "\\" in member.linkname
                or ":" in member.linkname
            ):
                raise ValueError("Unsafe dependency archive link: " + member.name)
            # Python's data filter rejects escaping/absolute links and device
            # nodes, and checks the resolved path again at extraction time.
            if member.issym() or member.islnk():
                links.append(member)
            else:
                source.extract(member, destination, filter="data")
    # Resolve links after streaming all payloads. tarfile's fallback for hosts
    # without symlink privilege seeks backwards into the compressed stream;
    # forward references need the payload to exist before link creation too.
    root = destination.resolve()
    depth = 0
    while links:
        depth += 1
        if depth > 64:
            raise ValueError("dependency archive link dependency depth exceeds 64")
        pending = []
        for member in links:
            filtered = tarfile.data_filter(member, destination)
            path = destination / filtered.name
            target = (
                (path.parent if member.issym() else destination) / filtered.linkname
            ).resolve()
            if not target.is_relative_to(root) or not path.resolve().is_relative_to(root):
                raise ValueError("dependency link escapes extraction root")
            if not target.exists():
                pending.append(member)
                continue
            if path.exists() or path.is_symlink():
                raise ValueError("dependency link collides with an extracted path")
            path.parent.mkdir(parents=True, exist_ok=True)
            if member.issym():
                try:
                    path.symlink_to(filtered.linkname, target_is_directory=target.is_dir())
                    continue
                except OSError:
                    if host != "Windows" or not target.is_file():
                        raise
            if not target.is_file():
                raise ValueError("dependency hard links require a regular file")
            try:
                os.link(target, path)
            except OSError:
                size += target.stat().st_size
                if size > max_bytes:
                    raise ValueError("dependency link materialization exceeds expanded-byte limit")
                shutil.copy2(target, path)
        if len(pending) == len(links):
            raise ValueError("dependency archive contains dangling or cyclic links")
        links = pending


def extract_zip(
    archive: Path, destination: Path, host: str, *, max_bytes: int, max_members: int
) -> None:
    """ZIP payloads must be regular files/directories; links are not needed here."""
    prepare_destination(destination)
    seen: set[str] = set()
    expanded = 0
    with zipfile.ZipFile(archive) as source:
        if len(source.infolist()) > max_members:
            raise ValueError("Dependency ZIP exceeds member limit")
        for member in source.infolist():
            key = archive_key(member.filename, host)
            if key in seen:
                raise ValueError("Duplicate dependency ZIP path: " + member.filename)
            seen.add(key)
            mode = member.external_attr >> 16
            kind = stat.S_IFMT(mode)
            if member.flag_bits & 1 or kind not in (0, stat.S_IFREG, stat.S_IFDIR):
                raise ValueError("Unsupported dependency ZIP member")
            expanded += member.file_size
            if member.file_size < 0 or expanded > max_bytes:
                raise ValueError("Dependency ZIP exceeds expanded-byte limit")
            path = destination.joinpath(*PurePosixPath(member.filename).parts)
            if member.is_dir():
                if member.file_size:
                    raise ValueError("Dependency ZIP directory has a payload")
                path.mkdir(parents=True, exist_ok=True)
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            with source.open(member) as payload, path.open("xb") as output:
                copied = 0
                while chunk := payload.read(1024**2):
                    copied += len(chunk)
                    if copied > member.file_size:
                        raise ValueError("Dependency ZIP member exceeds declared size")
                    output.write(chunk)
                if copied != member.file_size:
                    raise ValueError("Truncated dependency ZIP member")
            if host != "Windows":
                path.chmod(0o755 if mode & 0o111 else 0o644)
