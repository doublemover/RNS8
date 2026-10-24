"""Shared, write-free root selection for the doctor and pre-project CMake.

Filesystem discovery is not a compile/link or runtime compatibility test. No
compiler, package manager, executable version probe or installer runs here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import platform
import re
import shutil
import sys
from typing import Mapping

SCHEMA = "rns8_dependency_resolution_v1"
NATIVE_PACKAGE_POLICY = "Linux/UNIX builds must use native system packages and native CMake discovery. "
ACTIVE_RESOLUTION = ContextVar("rns8_dependency_resolution", default=None)
ROOT_KEYS = ("RNS8_HIP_ROOT", "RNS8_CK_ROOT", "RNS8_ROCWMMA_ROOT", "RNS8_HIPBLASLT_ROOT", "RNS8_GMP_ROOT", "RNS8_FLINT_ROOT")
HIP_PATH_KEYS = {"RNS8_HIP_HIPCC": "compiler", "RNS8_HIP_INCLUDE_DIR": "include", "RNS8_HIP_LIBRARY": "library",
                 "RNS8_HIP_RUNTIME": "runtime", "RNS8_HIP_DEVICE_LIB_DIR": "device_libraries", "RNS8_HIP_CLANG": "clang"}
OPTIONAL_PATH_KEYS = {"RNS8_CK_INCLUDE_DIR": ("ck", "include"), "RNS8_ROCWMMA_INCLUDE_DIR": ("rocwmma", "include"),
    "RNS8_HIPBLASLT_INCLUDE_DIR": ("hipblaslt", "include"), "RNS8_HIPBLASLT_LIBRARY": ("hipblaslt", "library"),
    "hipblaslt_DIR": ("hipblaslt", "cmake_package"), "RNS8_GMP_INCLUDE_DIR": ("gmp", "include"),
    "RNS8_GMP_LIBRARY": ("gmp", "library"), "RNS8_FLINT_INCLUDE_DIR": ("flint", "include"), "RNS8_FLINT_LIBRARY": ("flint", "library")}
REQUEST_KEYS = (*ROOT_KEYS, *HIP_PATH_KEYS, "CMAKE_TOOLCHAIN_FILE", "VCPKG_ROOT", "VCPKG_TARGET_TRIPLET",
                "CMAKE_PREFIX_PATH", "CMAKE_INCLUDE_PATH", "CMAKE_LIBRARY_PATH", "CMAKE_SYSROOT", "CMAKE_CXX_COMPILER",
                "CMAKE_SYSTEM_NAME", "CMAKE_SYSTEM_PROCESSOR", "RNS8_AMDGPU_TARGETS", "RNS8_ROCM_DEPS_INSTALL_ROOT",
                "RNS8_ROCM_DEPS_BUILD_ROOT", "CMAKE_FIND_ROOT_PATH", "CMAKE_FIND_ROOT_PATH_MODE_PACKAGE", *OPTIONAL_PATH_KEYS)
FEATURE_KEYS = ("RNS8_ENABLE_HIP", "RNS8_ENABLE_CK", "RNS8_ENABLE_ROCWMMA", "RNS8_ENABLE_HIPBLASLT",
                "RNS8_ENABLE_GMP", "RNS8_ENABLE_FLINT", "RNS8_ENABLE_AMDGPU_BUILTINS", "RNS8_PROBE_ACCELERATORS")


def truth(value: object) -> bool:
    return str(value).upper() in {"1", "ON", "TRUE", "YES", "Y"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_paths(value: str, host: str) -> list[str]:
    # CMake cache lists use semicolons on every host. Native environment lists
    # can also use colons; never split a Windows drive letter on Linux/WSL.
    if ";" in value: return [part for part in value.split(";") if part]
    if host != "Windows" and not PureWindowsPath(value).drive:
        return [part for part in value.split(":") if part]
    return [value] if value else []


def forbidden_windows_path(value: str) -> bool:
    text = value.replace("\\", "/").lower()
    return bool(PureWindowsPath(value).drive or re.search(r"(^|/|;)x(?:64|86|86_64)-windows(?:-static)?($|/|;)", text)
                or re.search(r"/mnt/[a-z]/.*vcpkg", text))


@dataclass(frozen=True)
class Request:
    host: str
    overrides: dict[str, str] = field(default_factory=dict)
    features: dict[str, bool] = field(default_factory=dict)
    source_root: str = ""
    require_hip: bool = False
    require_vcpkg: bool = False

    @classmethod
    def from_mapping(cls, value: dict) -> "Request":
        if type(value) is not dict: raise ValueError("discovery request must be an object")
        unknown = set(value) - {"host", "overrides", "features", "source_root", "require_hip", "require_vcpkg"}
        if unknown: raise ValueError("unknown discovery request fields: " + ", ".join(sorted(unknown)))
        overrides = value.get("overrides", {})
        features = value.get("features", {})
        if type(overrides) is not dict or any(key not in REQUEST_KEYS or type(item) is not str for key, item in overrides.items()):
            raise ValueError("discovery overrides must use supported CMake variable names and string values")
        if type(features) is not dict or any(key not in FEATURE_KEYS or type(item) is not bool for key, item in features.items()):
            raise ValueError("discovery feature policy must contain supported boolean flags")
        for name in ("require_hip", "require_vcpkg"):
            if name in value and type(value[name]) is not bool: raise ValueError(name + " must be boolean")
        host = value.get("host", platform.system())
        if host not in {"Windows", "Linux", "Darwin"}: raise ValueError("unsupported discovery host: " + str(host))
        root = value.get("source_root", str(Path(__file__).resolve().parents[2]))
        if type(root) is not str: raise ValueError("source_root must be a path string")
        return cls(host, dict(overrides), dict(features), root, value.get("require_hip", False), value.get("require_vcpkg", False))


@dataclass
class Candidate:
    root: str
    canonical_root: str
    origin: str
    priority: int
    explicit: bool
    paths: dict[str, str] = field(default_factory=dict)
    version: str | None = None
    version_source: str | None = None
    rejected: list[str] = field(default_factory=list)
    aliases: list[dict] = field(default_factory=list)
    file_sha256: dict[str, str] = field(default_factory=dict)
    revision: str | None = None
    binary_architectures: dict[str, str | None] = field(default_factory=dict)


def canonical(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve())


def contains(root: str, path: str) -> bool:
    try: Path(canonical(path)).relative_to(Path(canonical(root))); return True
    except ValueError: return False


def first_file(root: Path, relatives: tuple[str, ...]) -> str | None:
    for relative in relatives:
        path = root / relative
        if path.is_file(): return canonical(path)
    return None


def library_dirs(root: Path, host: str) -> list[Path]:
    result = [root / "lib", root / "lib64", root / "lib" / "hip", root / "bin"]
    if host != "Windows" and (root / "lib").is_dir():
        result += sorted(path for path in (root / "lib").glob("*-linux-gnu") if path.is_dir())
    return result


def pick_library(root: Path, host: str, names: tuple[str, ...]) -> str | None:
    for directory in library_dirs(root, host):
        result = first_file(directory, names)
        if result: return result
    return None


def header_version(header: Path) -> str | None:
    if not header.is_file(): return None
    text = header.read_text(encoding="utf-8", errors="replace")
    parts = [re.search(r"^\s*#\s*define\s+HIP_VERSION_" + part + r"\s+(\d+)\b", text, re.M)
             for part in ("MAJOR", "MINOR", "PATCH")]
    return ".".join(part[1] for part in parts) if all(parts) else None


def git_revision(root: Path) -> str | None:
    directory = root / ".git"
    if directory.is_file():
        text = directory.read_text(encoding="utf-8").strip()
        if not text.startswith("gitdir:"): return None
        directory = (root / text.split(":", 1)[1].strip()).resolve()
    head = directory / "HEAD"
    if not head.is_file(): return None
    value = head.read_text(encoding="utf-8").strip()
    if value.startswith("ref: "):
        ref = value[5:]; loose = directory / ref
        if loose.is_file(): value = loose.read_text(encoding="utf-8").strip()
        elif (directory / "packed-refs").is_file():
            value = next((line.split(" ", 1)[0] for line in (directory / "packed-refs").read_text(encoding="utf-8").splitlines()
                          if line.endswith(" " + ref)), "")
    return value if re.fullmatch(r"[0-9a-f]{40,64}", value) else None


def binary_architecture(path: Path) -> str | None:
    with path.open("rb") as stream:
        head = stream.read(64)
        if head.startswith(b"\x7fELF") and len(head) >= 20:
            endian = "little" if head[5] == 1 else "big"
            return {3: "x86", 62: "x86_64", 183: "aarch64"}.get(int.from_bytes(head[18:20], endian))
        if head.startswith(b"MZ") and len(head) >= 64:
            stream.seek(int.from_bytes(head[60:64], "little")); header = stream.read(6)
            if header[:4] == b"PE\0\0": return {0x14c: "x86", 0x8664: "x86_64", 0xaa64: "aarch64"}.get(int.from_bytes(header[4:6], "little"))
    return None


def inspect_hip(candidate: Candidate, request: Request) -> None:
    root = Path(candidate.canonical_root)
    if not root.is_dir(): candidate.rejected.append("root directory does not exist"); return
    windows = request.host == "Windows"
    compiler = first_file(root, ("bin/hipcc.exe", "bin/hipcc.bat", "bin/hipcc") if windows else ("bin/hipcc", "hip/bin/hipcc"))
    include = root / "include"
    header = include / "hip/hip_runtime_api.h"
    if not header.is_file() and (root / "hip/include/hip/hip_runtime_api.h").is_file():
        include = root / "hip/include"; header = include / "hip/hip_runtime_api.h"
    library = pick_library(root, request.host, ("amdhip64.lib", "libamdhip64.dll.a") if windows else ("libamdhip64.so",))
    if compiler: candidate.paths["compiler"] = compiler
    if header.is_file(): candidate.paths["include"] = canonical(include)
    if library: candidate.paths["library"] = library
    version_header = include / "hip/hip_version.h"
    candidate.version = header_version(version_header)
    if version_header.is_file(): candidate.version_source = canonical(version_header)
    if windows:
        runtimes = sorted(path for path in (root / "bin").glob("amdhip64*.dll") if path.is_file())
        major = candidate.version.split(".")[0] if candidate.version else None
        matched = [path for path in runtimes if path.name.lower() in {"amdhip64.dll", "amdhip64_" + str(major) + ".dll"}]
        if matched: runtimes = matched
        elif major and runtimes and not request.overrides.get("RNS8_HIP_RUNTIME"):
            candidate.rejected.append("runtime DLL major does not match the selected HIP version header")
        if len(runtimes) == 1: candidate.paths["runtime"] = canonical(runtimes[0])
        elif len(runtimes) > 1 and not request.overrides.get("RNS8_HIP_RUNTIME"):
            candidate.rejected.append("multiple HIP runtime DLLs; select RNS8_HIP_RUNTIME explicitly")
    elif library:
        candidate.paths["runtime"] = library
    for relative in ("amdgcn/bitcode", "lib/bitcode", "llvm/amdgcn/bitcode", "lib/llvm/amdgcn/bitcode"):
        directory = root / relative
        if directory.is_dir() and any(directory.glob("*.bc")):
            candidate.paths["device_libraries"] = canonical(directory); break
    if "device_libraries" not in candidate.paths:
        for parent in (root / "lib/clang", root / "llvm/lib/clang", root / "lib/llvm/lib/clang"):
            if not parent.is_dir(): continue
            matches = sorted(path for path in parent.glob("*/lib/amdgcn/bitcode") if path.is_dir() and any(path.glob("*.bc")))
            if len(matches) == 1: candidate.paths["device_libraries"] = canonical(matches[0]); break
            if len(matches) > 1 and not request.overrides.get("RNS8_HIP_DEVICE_LIB_DIR"):
                candidate.rejected.append("multiple compiler device-library directories; select RNS8_HIP_DEVICE_LIB_DIR explicitly")
    clang = first_file(root, ("bin/clang++.exe", "bin/clang.exe", "llvm/bin/clang++.exe") if windows else
                       ("llvm/bin/clang++", "bin/clang++", "bin/amdclang++", "lib/llvm/bin/clang++"))
    if not clang and not windows and (root / "lib").is_dir():
        compilers = sorted(path for path in (root / "lib").glob("llvm-*/bin/clang++") if path.is_file())
        if len(compilers) == 1: clang = canonical(compilers[0])
        elif len(compilers) > 1 and not request.overrides.get("RNS8_HIP_CLANG"):
            candidate.rejected.append("multiple native LLVM compilers; select RNS8_HIP_CLANG explicitly")
    if clang: candidate.paths["clang"] = clang
    for override, component in HIP_PATH_KEYS.items():
        value = request.overrides.get(override)
        if not value: continue
        path = Path(value).expanduser()
        expected_dir = component in {"include", "device_libraries"}
        if not (path.is_dir() if expected_dir else path.is_file()):
            candidate.rejected.append(override + " does not exist or has the wrong file type"); continue
        if not contains(candidate.canonical_root, str(path)):
            candidate.rejected.append(override + " is outside the selected canonical SDK root (mixed SDK evidence)"); continue
        candidate.paths[component] = canonical(path)
    for required in ("compiler", "include", "library", "runtime", "device_libraries", "clang"):
        if required not in candidate.paths: candidate.rejected.append("missing " + required)
    selected_include = candidate.paths.get("include")
    if selected_include and not (Path(selected_include) / "hip/hip_runtime_api.h").is_file():
        candidate.rejected.append("selected include directory lacks hip/hip_runtime_api.h")
    elif selected_include: candidate.paths["header"] = canonical(Path(selected_include) / "hip/hip_runtime_api.h")
    for component, path in candidate.paths.items():
        if not contains(candidate.canonical_root, path): candidate.rejected.append(component + " resolves outside canonical SDK root")
    # Folder labels do not establish a version. A mismatched include override
    # must not retain the original header's version claim.
    if selected_include:
        selected_header = Path(selected_include) / "hip/hip_version.h"
        selected_version = header_version(selected_header)
        if candidate.version and selected_version and candidate.version != selected_version:
            candidate.rejected.append("selected header version differs from SDK header version")
        candidate.version = selected_version
        candidate.version_source = canonical(selected_header) if selected_header.is_file() else None
        if candidate.version_source and not contains(candidate.canonical_root, candidate.version_source):
            candidate.rejected.append("version header resolves outside selected SDK root")
    runtime = candidate.paths.get("runtime", "")
    if windows and candidate.version:
        suffix = re.search(r"amdhip64_(\d+)\.dll$", runtime, re.I)
        if suffix and suffix[1] != candidate.version.split(".")[0]: candidate.rejected.append("runtime/header major versions differ")


def default_hip_roots(host: str, environment: Mapping[str, str]) -> list[Path]:
    if host == "Windows":
        parent = Path(environment.get("ProgramFiles", r"C:\Program Files")) / "AMD/ROCm"
        return sorted(path for path in parent.glob("*") if path.is_dir()) if parent.is_dir() else []
    if host == "Linux":
        roots = [Path("/opt/rocm")]
        roots += sorted(path for path in Path("/opt").glob("rocm-*") if path.is_dir())
        roots += [Path("/usr"), Path("/usr/local")]
        return roots
    return []


def sysroot_prefixes(request: Request) -> list[Path]:
    roots = []
    for key in ("CMAKE_SYSROOT", "CMAKE_FIND_ROOT_PATH"):
        roots += [Path(value) for value in split_paths(request.overrides.get(key, ""), request.host)]
    return list(dict.fromkeys(roots))


def select(name: str, roots: list[tuple[str, str, int, bool]], request: Request, required: bool, inspector) -> dict:
    candidates: list[Candidate] = []
    by_root = {}
    for root, origin, priority, explicit in roots:
        if not root: continue
        resolved = canonical(root)
        if resolved in by_root:
            previous = by_root[resolved]
            previous.aliases.append({"root": root, "origin": origin, "priority": priority})
            if priority < previous.priority:
                previous.origin = origin; previous.priority = priority
            previous.explicit = previous.explicit or explicit
            continue
        item = Candidate(str(root), resolved, origin, priority, explicit)
        if request.host != "Windows" and forbidden_windows_path(str(root)):
            item.rejected.append("Windows path is forbidden for native Linux/WSL discovery")
        elif request.overrides.get("CMAKE_FIND_ROOT_PATH_MODE_PACKAGE", "").upper() == "ONLY" and sysroot_prefixes(request) and not any(
                contains(str(prefix), resolved) for prefix in sysroot_prefixes(request)) and name != "vcpkg":
            item.rejected.append("candidate is outside selected CMAKE_SYSROOT/CMAKE_FIND_ROOT_PATH with package mode ONLY")
        else: inspector(item, request)
        item.revision = git_revision(Path(item.canonical_root))
        for component, path in item.paths.items():
            if component != "triplet" and Path(path).is_file():
                item.file_sha256[component] = digest(Path(path))
                if component in {"library", "runtime", "compiler", "clang"}:
                    item.binary_architectures[component] = binary_architecture(Path(path))
        processor = request.overrides.get("CMAKE_SYSTEM_PROCESSOR", platform.machine()).lower()
        architecture = {"amd64": "x86_64", "x86_64": "x86_64", "aarch64": "aarch64", "arm64": "aarch64", "i686": "x86", "x86": "x86"}.get(processor)
        for component in ("library", "runtime"):
            actual = item.binary_architectures.get(component)
            if architecture and actual and actual != architecture: item.rejected.append(component + " binary architecture differs from selected host/system processor")
        if item.version_source: item.file_sha256["version_header"] = digest(Path(item.version_source))
        by_root[resolved] = item; candidates.append(item)
    candidates.sort(key=lambda item: (item.priority, item.canonical_root))
    explicit_priority = min((item.priority for item in candidates if item.explicit), default=None)
    invalid_explicit = [item for item in candidates if item.explicit and item.priority == explicit_priority and item.rejected]
    usable = [item for item in candidates if not item.rejected]
    best = [item for item in usable if item.priority == usable[0].priority] if usable else []
    errors = []
    if invalid_explicit:
        errors = [name + ": invalid explicit " + item.origin + "=" + item.root + ": " + "; ".join(item.rejected) for item in invalid_explicit]
    elif len(best) > 1:
        if required or any(item.explicit for item in best):
            errors.append(name + ": ambiguous equally eligible installations; select an explicit root: " + ", ".join(item.root for item in best))
    elif not best and required:
        errors.append(name + ": no complete installation discovered; provide an explicit root or install prerequisites separately")
    chosen = best[0] if len(best) == 1 and not errors else None
    return {"status": "invalid" if invalid_explicit else "ambiguous" if len(best) > 1 else "discovered" if chosen else "missing",
            "requested": required, "selected": asdict(chosen) if chosen else None,
            "selection_reason": (chosen.origin + " has the unique best eligible priority" if chosen else "no unique usable selection"),
            "candidates": [asdict(item) for item in candidates], "errors": errors,
            "compatibility": "not_run", "runtime_validation": "not_run"}


def resolve_hip(request: Request, environment: Mapping[str, str]) -> dict:
    roots = []
    if request.overrides.get("RNS8_HIP_ROOT"):
        roots.append((request.overrides["RNS8_HIP_ROOT"], "override:RNS8_HIP_ROOT", 0, True))
    # An explicit component path can identify its SDK in the absence of root.
    if not roots:
        for key, component in HIP_PATH_KEYS.items():
            value = request.overrides.get(key)
            if not value: continue
            path = Path(value).expanduser()
            parent = path if component in {"include", "device_libraries"} else path.parent
            for ancestor in (parent, *list(parent.parents)[:4]):
                if (ancestor / "include/hip/hip_runtime_api.h").is_file():
                    roots.append((str(ancestor), "override:" + key, 0, True)); break
            else: roots.append((str(parent.parent), "override:" + key, 0, True))
    for key in ("RNS8_HIP_ROOT", "HIP_PATH", "ROCM_PATH"):
        if environment.get(key): roots.append((environment[key], "environment:" + key, 5 if key == "RNS8_HIP_ROOT" else 10, key == "RNS8_HIP_ROOT"))
    for prefix in split_paths(request.overrides.get("CMAKE_PREFIX_PATH", ""), request.host): roots.append((prefix, "CMAKE_PREFIX_PATH", 20, False))
    for prefix in split_paths(environment.get("CMAKE_PREFIX_PATH", ""), request.host): roots.append((prefix, "environment:CMAKE_PREFIX_PATH", 20, False))
    for sysroot in sysroot_prefixes(request):
        for suffix in ("", "usr", "usr/local", "opt/rocm"):
            roots.append((str(sysroot / suffix), "selected_sysroot", 15, False))
        if (sysroot / "opt").is_dir():
            roots += [(str(path), "selected_sysroot", 15, False) for path in sorted((sysroot / "opt").glob("rocm-*")) if path.is_dir()]
    on_path = shutil.which("hipcc", path=environment.get("PATH", ""))
    if on_path: roots.append((str(Path(on_path).resolve().parent.parent), "PATH:hipcc", 30, False))
    roots += [(str(root), "bounded_platform_default", 40, False) for root in default_hip_roots(request.host, environment)]
    return select("HIP", roots, request, request.require_hip, inspect_hip)


def inspect_vcpkg(candidate: Candidate, request: Request) -> None:
    root = Path(candidate.canonical_root)
    toolchain = root / "scripts/buildsystems/vcpkg.cmake"
    executable = root / ("vcpkg.exe" if request.host == "Windows" else "vcpkg")
    if not toolchain.is_file(): candidate.rejected.append("missing scripts/buildsystems/vcpkg.cmake")
    else: candidate.paths["toolchain"] = canonical(toolchain)
    if not executable.is_file(): candidate.rejected.append("missing bootstrapped vcpkg executable; bootstrap is a separate explicit action")
    else: candidate.paths["executable"] = canonical(executable)
    triplet = request.overrides.get("VCPKG_TARGET_TRIPLET", "x64-windows")
    if not re.fullmatch(r"(?:x64|x86|arm64)-windows(?:-static(?:-md)?)?", triplet):
        candidate.rejected.append("unsupported Windows triplet: " + triplet)
    else: candidate.paths["triplet"] = triplet
    processor = request.overrides.get("CMAKE_SYSTEM_PROCESSOR", platform.machine()).lower()
    architecture = {"amd64": "x64", "x86_64": "x64", "aarch64": "arm64", "arm64": "arm64", "i686": "x86", "x86": "x86"}.get(processor)
    if architecture and triplet.split("-", 1)[0] != architecture:
        candidate.rejected.append("vcpkg triplet architecture differs from selected host/system processor")
    marker = root / "vcpkg-configuration.json"
    if marker.is_file(): candidate.paths["configuration"] = canonical(marker)


def resolve_vcpkg(request: Request, environment: Mapping[str, str]) -> tuple[dict, dict]:
    toolchain = request.overrides.get("CMAKE_TOOLCHAIN_FILE", "")
    external = {"path": canonical(toolchain), "provider": "external", "status": "discovered"} if toolchain else {}
    is_vcpkg = bool(toolchain and Path(toolchain).name.lower() == "vcpkg.cmake")
    if toolchain and not Path(toolchain).is_file(): external["status"] = "invalid"; external["error"] = "explicit CMAKE_TOOLCHAIN_FILE does not exist: " + toolchain
    if request.host != "Windows":
        errors = []
        if toolchain and (is_vcpkg or "vcpkg" in toolchain.lower()): errors.append(NATIVE_PACKAGE_POLICY + "vcpkg toolchains are unsupported")
        for key in ("CMAKE_PREFIX_PATH", "CMAKE_INCLUDE_PATH", "CMAKE_LIBRARY_PATH", "CMAKE_FIND_ROOT_PATH", "CMAKE_SYSROOT", "VCPKG_TARGET_TRIPLET"):
            if any(forbidden_windows_path(value) for value in split_paths(request.overrides.get(key, ""), request.host)):
                errors.append(NATIVE_PACKAGE_POLICY + key + " contains Windows artifacts forbidden on native Linux/WSL")
        if external.get("error"): errors.append(external["error"])
        return {"status": "unsupported", "requested": False, "selected": None, "candidates": [], "errors": errors,
                "selection_reason": "native Linux/WSL package policy; no Linux-vcpkg lane introduced", "compatibility": "not_run"}, external
    roots = []
    if is_vcpkg: roots.append((str(Path(toolchain).parent.parent.parent), "override:CMAKE_TOOLCHAIN_FILE", 0, True))
    if request.overrides.get("VCPKG_ROOT"): roots.append((request.overrides["VCPKG_ROOT"], "override:VCPKG_ROOT", 0, True))
    if environment.get("VCPKG_ROOT"): roots.append((environment["VCPKG_ROOT"], "environment:VCPKG_ROOT", 5, True))
    command = shutil.which("vcpkg", path=environment.get("PATH", ""))
    if command: roots.append((str(Path(command).resolve().parent), "PATH:vcpkg", 20, False))
    roots += [(str(Path(environment.get("SystemDrive", "C:").rstrip("\\/") + "/") / "vcpkg"), "bounded_platform_default", 30, False)]
    inspect_request = request
    if not request.overrides.get("VCPKG_TARGET_TRIPLET") and environment.get("VCPKG_TARGET_TRIPLET"):
        inspect_request = replace(request, overrides={**request.overrides, "VCPKG_TARGET_TRIPLET": environment["VCPKG_TARGET_TRIPLET"]})
    result = select("vcpkg", roots, inspect_request, request.require_vcpkg and not (toolchain and not is_vcpkg), inspect_vcpkg)
    if external.get("error"): result["errors"].append(external["error"]); result["status"] = "invalid"
    if toolchain and not is_vcpkg:
        result["selection_reason"] += "; explicit external toolchain preserved; discovered vcpkg is not injected"
    elif result["selected"]:
        external = {"path": result["selected"]["paths"]["toolchain"], "provider": "vcpkg", "status": "discovered"}
    return result, external


OPTIONAL = {
    "ck": ("RNS8_CK_ROOT", "RNS8_ENABLE_CK", ("include/ck/ck.hpp",), None),
    "rocwmma": ("RNS8_ROCWMMA_ROOT", "RNS8_ENABLE_ROCWMMA", ("library/include/rocwmma/rocwmma.hpp", "include/rocwmma/rocwmma.hpp"), None),
    "hipblaslt": ("RNS8_HIPBLASLT_ROOT", "RNS8_ENABLE_HIPBLASLT", ("include/hipblaslt/hipblaslt.h", "include/hipblaslt.h"), "hipblaslt"),
    "gmp": ("RNS8_GMP_ROOT", "RNS8_ENABLE_GMP", ("include/gmp.h",), "gmp"),
    "flint": ("RNS8_FLINT_ROOT", "RNS8_ENABLE_FLINT", ("include/flint/flint.h",), "flint"),
}


def optional_components(request: Request, environment: Mapping[str, str], hip: dict, vcpkg: dict) -> dict:
    result = {}
    for name, (key, feature, headers, library_name) in OPTIONAL.items():
        roots = []
        if request.overrides.get(key): roots.append((request.overrides[key], "override:" + key, 0, True))
        if not roots:
            for variable, (dependency, component) in OPTIONAL_PATH_KEYS.items():
                if dependency != name or not request.overrides.get(variable): continue
                value = Path(request.overrides[variable])
                parent = value.parent if component == "library" else value
                for ancestor in (parent, *list(parent.parents)[:4]):
                    if any((ancestor / header).is_file() for header in headers):
                        roots.append((str(ancestor), "override:" + variable, 0, True)); break
                else:
                    inferred = parent.parent.parent if component == "include" and parent.parent.name == "include" else parent.parent
                    roots.append((str(inferred), "override:" + variable, 0, True))
        if environment.get(key): roots.append((environment[key], "environment:" + key, 5, True))
        roots += [(prefix, "CMAKE_PREFIX_PATH", 20, False) for prefix in split_paths(request.overrides.get("CMAKE_PREFIX_PATH", ""), request.host)]
        roots += [(prefix, "environment:CMAKE_PREFIX_PATH", 20, False) for prefix in split_paths(environment.get("CMAKE_PREFIX_PATH", ""), request.host)]
        roots += [(str(root / suffix), "selected_sysroot", 15, False) for root in sysroot_prefixes(request) for suffix in ("", "usr", "usr/local")]
        if hip.get("selected"): roots.append((hip["selected"]["canonical_root"], "selected_HIP_stack", 30, False))
        if vcpkg.get("selected"):
            selected = vcpkg["selected"]
            roots.append((str(Path(selected["canonical_root"]) / "installed" / selected["paths"]["triplet"]), "selected_vcpkg_triplet", 30, False))
            roots.append((str(Path(request.source_root) / "vcpkg_installed" / selected["paths"]["triplet"]), "repository_manifest_triplet", 30, False))
        if name in {"ck", "rocwmma"}:
            source = "composable_kernel" if name == "ck" else "rocWMMA"
            roots.append((str(Path(request.source_root) / "third_party/rocm" / source), "pinned_repository_source", 25, False))
        if request.host != "Windows": roots += [("/usr", "bounded_native_prefix", 50, False), ("/usr/local", "bounded_native_prefix", 50, False)]
        def inspect(candidate, config):
            root = Path(candidate.canonical_root)
            header = first_file(root, headers)
            multiarch_include = None
            if not header and name == "gmp" and config.host != "Windows" and (root / "include").is_dir():
                matches = sorted(path for path in (root / "include").glob("*-linux-gnu/gmp.h") if path.is_file())
                if len(matches) == 1: header = canonical(matches[0]); multiarch_include = matches[0].parent
                elif len(matches) > 1: candidate.rejected.append("multiple GMP multiarch headers; provide an explicit target prefix")
            if header:
                relative = next((value for value in headers if (root / value).is_file()), "include/gmp.h")
                # Strip the include-relative public header path.
                prefix = relative.split("/", 1)[0]
                include = multiarch_include or root / ("library/include" if relative.startswith("library/include/") else prefix)
                candidate.paths["include"] = canonical(include)
                candidate.paths["header"] = header
            if library_name:
                names = (library_name + ".lib", "lib" + library_name + ".lib", "lib" + library_name + ".dll.a") if config.host == "Windows" else (
                    "lib" + library_name + ".so", "lib" + library_name + ".a")
                library = pick_library(root, config.host, names)
                if library: candidate.paths["library"] = library
            for variable, (dependency, component) in OPTIONAL_PATH_KEYS.items():
                value = config.overrides.get(variable)
                if dependency != name or not value: continue
                path = Path(value)
                if not (path.is_file() if component == "library" else path.is_dir()):
                    candidate.rejected.append(variable + " does not exist or has the wrong type"); continue
                if not contains(candidate.canonical_root, str(path)):
                    candidate.rejected.append(variable + " is outside selected root (mixed dependency evidence)"); continue
                candidate.paths[component] = canonical(path)
                if component == "include":
                    header_names = {"ck": ("ck/ck.hpp",), "rocwmma": ("rocwmma/rocwmma.hpp",),
                        "hipblaslt": ("hipblaslt/hipblaslt.h", "hipblaslt.h"), "gmp": ("gmp.h",), "flint": ("flint/flint.h",)}[name]
                    actual_header = first_file(path, header_names)
                    if actual_header: candidate.paths["header"] = actual_header
                    else: candidate.paths.pop("header", None)
            if "header" not in candidate.paths: candidate.rejected.append("required public header not found")
            if library_name and "library" not in candidate.paths: candidate.rejected.append("required link library not found")
            for component, path in candidate.paths.items():
                if not contains(candidate.canonical_root, path): candidate.rejected.append(component + " escapes the selected root")
        result[name] = select(name, roots, request, request.features.get(feature, False), inspect)
        result[name]["feature_enabled"] = request.features.get(feature, False)
        result[name]["scope"] = "candidate paths only; existing optional finders own API/target/link probes; no backend enabled by discovery"
    return result


def resolve(request: Request, environment: Mapping[str, str] | None = None) -> dict:
    env = dict(os.environ if environment is None else environment)
    errors = []
    for root in sysroot_prefixes(request):
        if not root.is_dir(): errors.append("explicit sysroot/find root does not exist: " + str(root))
    if request.host != "Windows":
        for key in ("CMAKE_PREFIX_PATH", "CMAKE_INCLUDE_PATH", "CMAKE_LIBRARY_PATH"):
            if any(forbidden_windows_path(value) for value in split_paths(env.get(key, ""), request.host)):
                errors.append(NATIVE_PACKAGE_POLICY + "environment:" + key + " contains Windows artifacts forbidden on native Linux/WSL")
    hip = resolve_hip(request, env)
    vcpkg, toolchain = resolve_vcpkg(request, env)
    optional = optional_components(request, env, hip, vcpkg)
    for item in (hip, vcpkg, *optional.values()): errors += item["errors"]
    bindings = {}
    selected = hip.get("selected")
    if selected and (request.require_hip or any(request.overrides.get(key) for key in ("RNS8_HIP_ROOT", *HIP_PATH_KEYS))):
        bindings["RNS8_HIP_ROOT"] = selected["canonical_root"]
        for variable, component in HIP_PATH_KEYS.items(): bindings[variable] = selected["paths"][component]
    if toolchain.get("status") == "discovered": bindings["CMAKE_TOOLCHAIN_FILE"] = toolchain["path"]
    if toolchain.get("provider") == "vcpkg" and vcpkg.get("selected"):
        selected_vcpkg = vcpkg["selected"]
        bindings["VCPKG_ROOT"] = selected_vcpkg["canonical_root"]
        bindings["VCPKG_TARGET_TRIPLET"] = selected_vcpkg["paths"]["triplet"]
    for name, component in optional.items():
        selected_optional = component.get("selected")
        has_component_override = any(request.overrides.get(variable) for variable, (dependency, _) in OPTIONAL_PATH_KEYS.items() if dependency == name)
        if selected_optional and (component["feature_enabled"] or request.overrides.get(OPTIONAL[name][0]) or has_component_override):
            key = OPTIONAL[name][0]; bindings[key] = selected_optional["canonical_root"]
            bindings[key.removesuffix("_ROOT") + "_INCLUDE_DIR"] = selected_optional["paths"]["include"]
            if selected_optional["paths"].get("library"):
                bindings[key.removesuffix("_ROOT") + "_LIBRARY"] = selected_optional["paths"]["library"]
            if selected_optional["paths"].get("cmake_package"): bindings["hipblaslt_DIR"] = selected_optional["paths"]["cmake_package"]
    tools = {}
    for name in ("cmake", "ninja", "git", "python", "host_cxx"):
        override = request.overrides.get("CMAKE_CXX_COMPILER") if name == "host_cxx" else None
        if override:
            path = canonical(override) if Path(override).is_file() else shutil.which(override, path=env.get("PATH", ""))
            if not path: errors.append("explicit CMAKE_CXX_COMPILER does not identify an executable: " + override)
        else:
            path = sys.executable if name == "python" else shutil.which("cl" if request.host == "Windows" and name == "host_cxx" else "c++" if name == "host_cxx" else name, path=env.get("PATH", ""))
        tools[name] = {"path": canonical(path) if path else None, "origin": "override:CMAKE_CXX_COMPILER" if override else "running_interpreter" if name == "python" else "PATH",
                       "version": None, "status": "discovered" if path else "missing", "compatibility": "not_run"}
    manifest = Path(request.source_root) / "vcpkg.json"
    report = {"schema": SCHEMA, "request": asdict(request), "status": "invalid" if errors else "resolved", "errors": errors,
              "hip": hip, "vcpkg": vcpkg, "toolchain": toolchain, "optional_components": optional, "tools": tools,
              "cmake_bindings": {key: value.replace("\\", "/") if request.host == "Windows" else value for key, value in bindings.items()}, "feature_policy": request.features,
              "manifest_sha256": digest(manifest) if manifest.is_file() else None,
              "validation_scope": "write-free filesystem discovery only; no compiler/package/runtime probes or installation",
              "remaining_gaps": ["compiler and host/runtime ABI compatibility", "all-target CK/rocWMMA/builtin API probes",
                  "GMP/FLINT version/API/transitive link validation", "dependency lock compatibility and upgrades", "native Linux/WSL hardware qualification"]}
    # Cache materialization changes origins from defaults/environment to cache
    # overrides on the next configure. Bind the effective selection, not that
    # diagnostic history or any generated report/build-output filename.
    def selected_identity(component):
        item = component.get("selected")
        return {key: item[key] for key in ("canonical_root", "paths", "version", "revision", "file_sha256")} if item else {"status": component["status"]}
    identity = {"host": request.host, "hip": selected_identity(hip) if request.require_hip else {"status": "disabled"},
                "toolchain": toolchain, "vcpkg": selected_identity(vcpkg) if toolchain.get("provider") == "vcpkg" else {"status": "unused"},
                "optional": {name: selected_identity(item) if item["feature_enabled"] else {"status": "disabled"} for name, item in optional.items()},
                "tools": {name: {"path": item["path"], "version": item["version"]} for name, item in tools.items()},
                "feature_policy": {key: request.features.get(key, False) for key in FEATURE_KEYS},
                "targets": request.overrides.get("RNS8_AMDGPU_TARGETS", "gfx1100"),
                "manifest_sha256": report["manifest_sha256"]}
    if toolchain.get("path") and Path(toolchain["path"]).is_file(): identity["toolchain_sha256"] = digest(Path(toolchain["path"]))
    report["effective_identity"] = identity
    report["identity_sha256"] = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    report["compiler_environment"] = compiler_environment(report)
    return report


def compiler_environment(report: dict) -> dict[str, str]:
    selected = report["hip"].get("selected")
    if not selected: return {}
    paths = selected["paths"]
    environment = {"HIP_PATH": str(Path(paths["include"]).parent), "ROCM_PATH": selected["canonical_root"],
                   "HIP_CLANG_PATH": str(Path(paths["clang"]).parent), "HIP_DEVICE_LIB_PATH": paths["device_libraries"]}
    # CMake macro wrappers can reparse arguments as source. Native backslashes
    # in paths such as C:\\Program Files become invalid CMake escapes there.
    if report["request"]["host"] == "Windows":
        environment = {key: value.replace("\\", "/") for key, value in environment.items()}
    return environment


def read_cache(path: Path) -> tuple[dict[str, str], dict[str, bool]]:
    overrides = {}; features = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith(("#", "//")) or "=" not in line: continue
        left, value = line.split("=", 1); key = left.split(":", 1)[0]
        if key in REQUEST_KEYS and value and not value.endswith("-NOTFOUND"): overrides[key] = value
        if key in FEATURE_KEYS: features[key] = truth(value)
    return overrides, features


@contextmanager
def selection_context(report: dict):
    token = ACTIVE_RESOLUTION.set(report)
    try: yield
    finally: ACTIVE_RESOLUTION.reset(token)


def selected_command(name: str) -> tuple[bool, str | None]:
    """Return handled/path so a rejected SDK cannot leak back to PATH fallback."""
    report = ACTIVE_RESOLUTION.get()
    if not report:
        if name not in {"hipcc", "hipconfig", "hipInfo", "rocminfo", "vcpkg"}: return False, None
        host = platform.system()
        if name == "vcpkg" and host != "Windows": return False, None
        request = Request(host)
        report = {"request": {"host": host}, "hip": resolve_hip(request, os.environ),
                  "vcpkg": resolve_vcpkg(request, os.environ)[0] if name == "vcpkg" else {}}
    if name in {"hipcc", "hipconfig", "hipInfo", "rocminfo"}:
        selected = report["hip"].get("selected")
        if not selected: return True, None
        if name == "hipcc": return True, selected["paths"]["compiler"]
        root = Path(selected["canonical_root"])
        return True, first_file(root, ("bin/" + name, "bin/" + name + ".exe", "bin/" + name + ".bat"))
    if name == "vcpkg" and report["request"]["host"] == "Windows":
        selected = report["vcpkg"].get("selected")
        return True, selected["paths"]["executable"] if selected else None
    return False, None
