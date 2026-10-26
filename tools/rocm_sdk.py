"""Explicit, target-aware ROCm Core tarball provisioning for the init workflow.

No import-time discovery, downloads, system installation or persistent environment
changes. Network artifacts come only from AMD's stable HTTPS repository.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request

from check_dependencies_lib.resolution import Request, resolve

BASE_URL = "https://stable.repo.amd.com/rocm/core/tarball/"
VERSION = "10.1.0"
MAX_DOWNLOAD = 8 * 1024**3
MAX_EXTRACT = 64 * 1024**3
MAX_MEMBERS = 500_000
ROOT = Path(__file__).resolve().parents[1]

# Exact known targets, not a wildcard guess about future GPU architectures.
FAMILIES = {
    **dict.fromkeys(("gfx1010", "gfx1011", "gfx1012"), "gfx101X-dgpu"),
    **dict.fromkeys(("gfx1030", "gfx1031", "gfx1032", "gfx1033", "gfx1034", "gfx1035", "gfx1036"), "gfx103X-all"),
    **dict.fromkeys(("gfx1100", "gfx1101", "gfx1102", "gfx1103"), "gfx110X-all"),
    **dict.fromkeys(("gfx1200", "gfx1201"), "gfx120X-all"),
    **dict.fromkeys(("gfx940", "gfx941", "gfx942"), "gfx94X-dcgpu"),
    "gfx908": "gfx908", "gfx90a": "gfx90a", "gfx950": "gfx950-dcgpu",
    "gfx1150": "gfx1150", "gfx1151": "gfx1151", "gfx1152": "gfx1152", "gfx1153": "gfx1153",
    "gfx1250": "gfx125X-dcgpu", "gfx1251": "gfx125X-dcgpu",
}


def captured(command: list[str]) -> str:
    return subprocess.run(command, capture_output=True, text=True, check=True, timeout=30).stdout


def detect_targets(sdk_root: Path | None = None) -> list[str]:
    """Prefer AMD architecture enumerators; never substitute a default GPU."""
    candidates = []
    if sdk_root:
        candidates.extend(sdk_root / name for name in (
            "lib/llvm/bin/amdgpu-arch.exe", "lib/llvm/bin/amdgpu-arch", "bin/hipInfo.exe", "bin/hipInfo"))
    candidates.extend(Path(found) for name in ("amdgpu-arch", "rocm_agent_enumerator", "hipInfo")
                      if (found := shutil.which(name)))
    for executable in candidates:
        if not executable.is_file():
            continue
        try:
            targets = sorted(set(re.findall(r"\bgfx[0-9a-f]+\b", captured([str(executable)]))) - {"gfx000"})
            if targets:
                return targets
        except (OSError, subprocess.SubprocessError):
            continue
    if platform.system() == "Windows":
        # A driver-only Windows machine has no HIP enumerator. Match the AMD
        # adapter names conservatively; unknown AMD adapters require --sdk-target.
        text = captured(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
                         "Get-CimInstance Win32_VideoController | Select-Object Name,PNPDeviceID | ConvertTo-Json -Compress"])
        adapters = json.loads(text)
        if isinstance(adapters, dict):
            adapters = [adapters]
        targets = []
        for adapter in adapters or []:
            if "VEN_1002" not in str(adapter.get("PNPDeviceID", "")).upper():
                continue
            name = str(adapter.get("Name", ""))
            # Desktop names only: mobile variants can use a different ISA even
            # with the same marketing number (for example 7900M vs 7900 XTX).
            match = re.search(r"\bRX\s+(7900|7800|7700|7600|9070|9060)(?:\s+(?:XTX|XT|GRE))?\s*$", name, re.I)
            if not match:
                raise ValueError(f"Unknown AMD adapter {name!r}; supply exact --sdk-target gfxNNNN")
            targets.append({"7900": "gfx1100", "7800": "gfx1101", "7700": "gfx1101",
                            "7600": "gfx1102", "9070": "gfx1201", "9060": "gfx1200"}[match[1]])
        if targets:
            return sorted(set(targets))
    raise ValueError("No GPU architecture enumerator available; supply exact --sdk-target gfxNNNN")


def plan(targets: list[str], host: str, version: str, prefix: Path) -> list[dict]:
    if host not in ("Windows", "Linux"):
        raise ValueError("SDK tarballs are supported only on Windows and native Linux")
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version):
        raise ValueError("SDK version must be an exact stable MAJOR.MINOR.PATCH")
    if not targets:
        raise ValueError("At least one explicit or detected target is required")
    groups: dict[str, list[str]] = {}
    for target in sorted(set(targets)):
        if target not in FAMILIES:
            raise ValueError(f"No SDK family mapping for {target!r}")
        groups.setdefault(FAMILIES[target], []).append(target)
    lanes = []
    for family, members in sorted(groups.items()):
        filename = f"therock-dist-{host.lower()}-{family}-{version}.tar.gz"
        destination = prefix.resolve() / f"{host.lower()}-{family}-{version}"
        lanes.append(dict(family=family, targets=members, host=host, version=version,
                          filename=filename, url=BASE_URL + filename, destination=str(destination)))
    return lanes


def official_inventory() -> set[str]:
    with urllib.request.urlopen(BASE_URL, timeout=60) as response:
        document = response.read(2 * 1024**2 + 1)
    if len(document) > 2 * 1024**2:
        raise ValueError("AMD artifact index exceeds bounded metadata size")
    match = re.search(rb"const files\s*=\s*(\[.*?\]);", document, re.S)
    if not match:
        raise ValueError("Unrecognized AMD artifact index; refusing to guess archive availability")
    return {item["name"] for item in json.loads(match[1])}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def download(lane: dict, cache: Path, expected: str | None) -> tuple[Path, str]:
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / lane["filename"]
    record = archive.with_suffix(archive.suffix + ".json")
    if archive.exists():
        digest = sha256(archive)
        saved = json.loads(record.read_text()) if record.is_file() else {}
        if digest != (expected or saved.get("sha256")):
            raise ValueError(f"Unverified or corrupted cached archive: {archive}; remove it explicitly to download again")
        return archive, digest
    descriptor, partial_name = tempfile.mkstemp(prefix=lane["filename"] + ".", suffix=".part", dir=cache)
    partial = Path(partial_name)
    try:
        with os.fdopen(descriptor, "wb") as output, urllib.request.urlopen(lane["url"], timeout=120) as response:
            if not response.url.startswith(BASE_URL):
                raise ValueError("SDK download redirected outside AMD's stable tarball repository")
            length = int(response.headers["Content-Length"]) if response.headers.get("Content-Length") else None
            if length is not None and (length < 1 or length > MAX_DOWNLOAD):
                raise ValueError("SDK download exceeds configured byte limit")
            size = 0
            digest = hashlib.sha256()
            for block in iter(lambda: response.read(1024**2), b""):
                size += len(block)
                if size > MAX_DOWNLOAD:
                    raise ValueError("SDK download exceeds configured byte limit")
                output.write(block)
                digest.update(block)
            if not size or (length is not None and size != length):
                raise ValueError("Truncated SDK download")
        observed = digest.hexdigest()
        if expected and observed != expected:
            raise ValueError("SDK archive SHA-256 does not match supplied pin")
        if archive.exists():
            raise ValueError("Another SDK installer published this archive; retry to reuse it")
        partial.rename(archive)
        record.write_text(json.dumps(dict(url=lane["url"], sha256=observed,
                                        integrity="pinned_sha256" if expected else "https_observed_sha256"), indent=2) + "\n")
        return archive, observed
    finally:
        partial.unlink(missing_ok=True)


def extract(archive: Path, destination: Path, host: str) -> None:
    """Bounded tar extraction into a new empty staging directory, never overlays."""
    count = size = 0
    seen: set[str] = set()
    with tarfile.open(archive, "r|gz") as source:
        for member in source:
            count += 1
            size += max(0, member.size)
            if count > MAX_MEMBERS or size > MAX_EXTRACT:
                raise ValueError("SDK extraction exceeds member/expanded-byte limit")
            parts = PurePosixPath(member.name).parts
            if not parts or any(p in ("..", "/") or "\\" in p or ":" in p for p in parts):
                raise ValueError("Unsafe SDK archive path: " + member.name)
            if host == "Windows" and any(p.endswith((".", " ")) or
                    re.fullmatch(r"(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?", p, re.I) for p in parts):
                raise ValueError("Unsafe Windows SDK archive path: " + member.name)
            key = "/".join(parts).casefold() if host == "Windows" else "/".join(parts)
            if key in seen:
                raise ValueError("Duplicate SDK archive path: " + member.name)
            seen.add(key)
            if member.size < 0 or not (member.isfile() or member.isdir() or member.issym() or member.islnk()):
                raise ValueError("Unsupported SDK archive member: " + member.name)
            if (member.issym() or member.islnk()) and (PurePosixPath(member.linkname).is_absolute()
                    or "\\" in member.linkname or ":" in member.linkname):
                raise ValueError("Unsafe SDK archive link: " + member.name)
            # Python's data filter rejects escaping/absolute links and device
            # nodes, and checks the resolved path again at extraction time.
            source.extract(member, destination, filter="data")


def inspect(root: Path, host: str, version: str | None = None) -> dict:
    resolution = resolve(Request(host, overrides={"RNS8_HIP_ROOT": str(root)},
                                 source_root=str(ROOT), require_hip=True, require_vcpkg=False))
    if resolution["errors"]:
        raise ValueError("Invalid SDK extraction: " + "; ".join(map(str, resolution["errors"])))
    if version:
        version_file = root / ".info/version"
        if not version_file.is_file() or version_file.read_text().strip() != version:
            raise ValueError("SDK .info/version does not match requested Core SDK release")
    return resolution


def provision(lane: dict, cache: Path, expected: str | None = None) -> tuple[Path, dict]:
    destination = Path(lane["destination"])
    receipt = destination / "rns8-sdk.json"
    if destination.exists():
        saved = json.loads(receipt.read_text())
        if any(saved.get(key) != lane[key] for key in ("url", "host", "version", "family")):
            raise ValueError("Existing SDK receipt does not match requested release/family")
        if expected and saved.get("archive_sha256") != expected:
            raise ValueError("Existing SDK receipt does not match requested SHA-256")
        root = destination / saved["sdk_relative_root"]
        if not root.resolve().is_relative_to(destination.resolve()):
            raise ValueError("SDK receipt points outside its managed prefix")
        inspect(root, lane["host"], lane["version"])
        return root, saved
    if lane["filename"] not in official_inventory():
        raise ValueError("Requested OS/family/release archive is not listed by AMD: " + lane["filename"])
    archive, digest = download(lane, cache, expected)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # tempfile owns this exact directory; its cleanup cannot target user paths.
    with tempfile.TemporaryDirectory(prefix=".rns8-sdk-", dir=destination.parent) as staging_name:
        staging = Path(staging_name)
        extract(archive, staging, lane["host"])
        if (staging / "rns8-sdk.json").exists() or (staging / "rns8-sdk.json").is_symlink():
            raise ValueError("SDK archive collides with managed receipt path")
        roots = [path.parent for path in staging.glob("**/.info/version")]
        roots = [path.parent for path in roots]
        if len(roots) != 1:
            raise ValueError("SDK archive must contain one .info/version root")
        relative = roots[0].relative_to(staging).as_posix()
        inspect(roots[0], lane["host"], lane["version"])
        saved = {**lane, "sdk_relative_root": relative, "archive_sha256": digest,
                 "integrity": "pinned_sha256" if expected else "https_observed_sha256"}
        (staging / "rns8-sdk.json").write_text(json.dumps(saved, indent=2) + "\n")
        if destination.exists():
            raise ValueError("SDK destination appeared during extraction; refusing to overlay")
        staging.rename(destination)
    return destination / relative, saved


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sdk-install", action="store_true", help="explicitly download and extract AMD stable SDK tarballs")
    parser.add_argument("--sdk-root", type=Path, help="reuse an existing coherent SDK instead of downloading")
    parser.add_argument("--sdk-target", action="append", default=[], help="exact gfx target; repeat for multiple GPU families; otherwise detect devices")
    parser.add_argument("--sdk-version", default=VERSION, help="exact ROCm Core SDK release")
    parser.add_argument("--sdk-prefix", type=Path, default=ROOT / "out/third_party/rocm/sdk")
    parser.add_argument("--sdk-cache", type=Path, default=ROOT / "temp/rocm-sdk-downloads")
    parser.add_argument("--sdk-sha256", action="append", default=[], metavar="FAMILY=HEX", help="independently supplied archive hash pin")
    parser.add_argument("--sdk-build", action="store_true", help="configure/build a separate Release directory per selected family")
    parser.add_argument("--sdk-build-root", type=Path, default=ROOT / "build/rocm-sdk")
    parser.add_argument("--sdk-jobs", type=int, default=4)
    parser.add_argument("--sdk-cmake-arg", action="append", default=[], help="additional CMake option, use --sdk-cmake-arg=-DKEY=VALUE")


def run(args: argparse.Namespace) -> dict:
    host = platform.system()
    if args.sdk_install and args.sdk_root:
        raise ValueError("Choose --sdk-install or --sdk-root")
    if args.sdk_jobs < 1:
        raise ValueError("--sdk-jobs must be positive")
    if any(not value.startswith("-D") or "=" not in value for value in args.sdk_cmake_arg):
        raise ValueError("--sdk-cmake-arg accepts only -DKEY=VALUE options")
    pins = {}
    for value in args.sdk_sha256:
        family, separator, digest = value.partition("=")
        if not separator or family not in set(FAMILIES.values()) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            raise ValueError("--sdk-sha256 requires FAMILY=64_HEX_DIGITS")
        if family in pins:
            raise ValueError("Duplicate SDK hash pin for " + family)
        pins[family] = digest.lower()
    targets = args.sdk_target or ([args.target] if args.target else detect_targets(args.sdk_root))
    lanes = plan(targets, host, args.sdk_version, args.sdk_prefix)
    if set(pins) - {lane["family"] for lane in lanes}:
        raise ValueError("SDK hash pin supplied for an unselected family")
    if args.sdk_root and len(lanes) != 1:
        raise ValueError("An existing --sdk-root must select exactly one GPU family")
    results = []
    for lane in lanes:
        result = dict(lane, dry_run=args.dry_run)
        root = args.sdk_root.resolve() if args.sdk_root else Path(lane["destination"])
        if not args.dry_run:
            if args.sdk_root:
                inspect(root, host, args.sdk_version)
                result["integrity"] = "existing_extraction_archive_origin_unverified"
            else:
                root, result["receipt"] = provision(lane, args.sdk_cache, pins.get(lane["family"]))
            resolution = inspect(root, host, args.sdk_version)
            result["dependency_resolution"] = resolution
        else:
            resolution = None
        bindings = dict(resolution["cmake_bindings"]) if resolution else {"RNS8_HIP_ROOT": str(root)}
        bindings.update(RNS8_ENABLE_HIP="ON", RNS8_AMDGPU_TARGETS=";".join(lane["targets"]))
        bindings.update(RNS8_CK_ROOT=str(root), RNS8_ROCWMMA_ROOT=str(root), RNS8_HIPBLASLT_ROOT=str(root))
        build = args.sdk_build_root.resolve() / f"{host.lower()}-{lane['family']}-{lane['version']}"
        wrapper = [sys.executable, str(ROOT / "tools/windows_dev.py")] if host == "Windows" else []
        # Discovery bindings are appended last so caller options cannot mix SDKs
        # or silently select a different target within this build lane.
        configure = [*wrapper, "cmake", "-S", str(ROOT), "-B", str(build), "-G", "Ninja",
                     "-DCMAKE_BUILD_TYPE=Release", "-DVCPKG_MANIFEST_INSTALL=OFF", *args.sdk_cmake_arg,
                     *(f"-D{key}={value}" for key, value in sorted(bindings.items()))]
        commands = [configure, [*wrapper, "cmake", "--build", str(build), "--parallel", str(args.sdk_jobs)]]
        result.update(sdk_root=str(root), build_dir=str(build), build_requested=args.sdk_build, commands=commands)
        if args.sdk_build and not args.dry_run:
            env = dict(os.environ)
            env.update(resolution["compiler_environment"])
            env["HIP_PLATFORM"] = "amd"
            env["PATH"] = os.pathsep.join((str(root / "bin"), str(Path(bindings["RNS8_HIP_CLANG"]).parent), env.get("PATH", "")))
            for command in commands:
                subprocess.run(command, cwd=ROOT, env=env, check=True, stdout=sys.stderr, stderr=sys.stderr)
        results.append(result)
    return dict(schema="rns8_rocm_sdk_init_v1", dry_run=args.dry_run, lanes=results)
