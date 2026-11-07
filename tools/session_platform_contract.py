"""Frozen native Linux session inputs; no installation or GPU execution.

An operator selects the tuple from AMD's versioned support matrix. Matching
files and native host observations establish reproducibility, not upstream
support, loaded firmware, transitive ABI correctness or hardware qualification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import stat
from urllib.parse import urlsplit

from json_evidence import parse_evidence_json

SCHEMA = "rns8_linux_platform_lock_v1"
FIELDS = (
    "os_id",
    "os_version",
    "kernel",
    "machine",
    "glibc",
    "driver_version",
    "driver_srcversion",
)
ROLES = {
    "host_compiler",
    "stdlib",
    "openmp",
    "hip_compiler",
    "hip_runtime",
    "device_library",
    "reference",
    "firmware",
    "package_inventory",
}


def load_lock(path: Path) -> dict:
    with path.open("rb") as handle:
        payload = handle.read(1024 * 1024 + 1)
    if len(payload) > 1024 * 1024:
        raise ValueError("platform lock exceeds 1MiB")
    value = parse_evidence_json(payload.decode("utf-8"))
    if (
        not isinstance(value, dict)
        or set(value)
        != {
            "schema",
            "target",
            "matrix_url",
            "reviewed_date_utc",
            "image_digest",
            "expected",
            "files",
        }
        or value["schema"] != SCHEMA
    ):
        raise ValueError("complete versioned Linux platform lock required")
    if value["target"] != "gfx942":
        raise ValueError("this session lock currently supports gfx942 only")
    url = urlsplit(value["matrix_url"]) if isinstance(value["matrix_url"], str) else None
    if (
        url is None
        or url.scheme != "https"
        or url.netloc != "rocm.docs.amd.com"
        or not re.fullmatch(r"/en/docs-[0-9.]+/compatibility/compatibility-matrix.html", url.path)
        or url.query
        or url.fragment
    ):
        raise ValueError("versioned official AMD compatibility matrix URL required")
    from datetime import date

    try:
        date.fromisoformat(value["reviewed_date_utc"])
    except (TypeError, ValueError) as error:
        raise ValueError("ISO review date required") from error
    image = value["image_digest"]
    if image is not None and (
        not isinstance(image, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", image)
    ):
        raise ValueError("image must be an immutable SHA256 or null for a non-image host")
    expected = value["expected"]
    if not isinstance(expected, dict) or set(expected) != set(FIELDS):
        raise ValueError("complete native platform observation fields required")
    for field, text in expected.items():
        if text is None and field in {"driver_version", "driver_srcversion"}:
            continue
        if (
            not isinstance(text, str)
            or not text
            or len(text) > 512
            or any(ord(c) < 32 for c in text)
        ):
            raise ValueError("invalid expected native field: " + field)
    if not expected["driver_version"] and not expected["driver_srcversion"]:
        raise ValueError("at least one native amdgpu module identity required")
    files = value["files"]
    if not isinstance(files, list) or not 1 <= len(files) <= 256:
        raise ValueError("1..256 locked platform files required")
    paths, roles, total = set(), set(), 0
    for row in files:
        if not isinstance(row, dict) or set(row) != {"role", "path", "bytes", "sha256"}:
            raise ValueError("typed platform file identity required")
        name = row["path"]
        if (
            not isinstance(row["role"], str)
            or row["role"] not in ROLES
            or not isinstance(name, str)
            or not name.startswith("/")
            or "\\" in name
            or len(name) > 4096
            or any(ord(c) < 32 for c in name)
            or any(p in {"", ".", ".."} for p in name.split("/")[1:])
            or name in paths
            or type(row["bytes"]) is not int
            or not 0 < row["bytes"] <= 2 * 1024**3
            or not isinstance(row["sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"])
        ):
            raise ValueError("invalid, repeated or unbounded platform file")
        paths.add(name)
        roles.add(row["role"])
        total += row["bytes"]
    if (
        total > 16 * 1024**3
        or not {"host_compiler", "stdlib", "hip_compiler", "hip_runtime", "device_library"} <= roles
    ):
        raise ValueError("bounded host and HIP compiler/runtime/device-library identities required")
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(payload).hexdigest(), lock=value)


def observe(lock: dict) -> dict:
    if platform.system() != "Linux":
        raise ValueError("native Linux platform observation required")
    release = platform.freedesktop_os_release()

    def driver(name):
        try:
            with (Path("/sys/module/amdgpu") / name).open(encoding="utf-8") as stream:
                return stream.read(513).strip() or None
        except OSError:
            return None

    native = dict(
        os_id=release.get("ID"),
        os_version=release.get("VERSION_ID"),
        kernel=platform.release(),
        machine=platform.machine(),
        glibc=os.confstr("CS_GNU_LIBC_VERSION"),
        driver_version=driver("version"),
        driver_srcversion=driver("srcversion"),
    )
    files = []
    for expected in lock["files"]:
        row = dict(path=expected["path"], role=expected["role"], bytes=None, sha256=None)
        try:
            path = Path(row["path"])
            with os.fdopen(
                os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC), "rb"
            ) as stream:
                before = os.fstat(stream.fileno())

                if not stat.S_ISREG(before.st_mode) or before.st_size != expected["bytes"]:
                    raise ValueError("locked file is not regular or its length changed")
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
                after = os.fstat(stream.fileno())
            if (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            ) != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ) or path.stat() != after:
                raise ValueError("locked file changed during observation")
            row.update(bytes=after.st_size, sha256=digest)
        except (OSError, ValueError) as error:
            row["error"] = str(error)
        files.append(row)
    return dict(native=native, files=files)


def evaluate(reference: dict, observed: dict) -> dict:
    lock = reference["lock"]
    blockers = []
    native, files = observed.get("native"), observed.get("files")
    if not isinstance(native, dict) or set(native) != set(FIELDS):
        blockers.append("complete native platform observations missing")
    else:
        blockers += [
            "native platform mismatch: " + name
            for name in FIELDS
            if lock["expected"][name] is not None and native[name] != lock["expected"][name]
        ]
    if not isinstance(files, list) or len(files) != len(lock["files"]):
        blockers.append("complete locked file observations missing")
    else:
        for expected, actual in zip(lock["files"], files):
            if (
                not isinstance(actual, dict)
                or actual.get("error")
                or any(
                    type(actual.get(key)) is not type(value) or actual[key] != value
                    for key, value in expected.items()
                )
            ):
                blockers.append("platform input changed or unavailable: " + expected["path"])
    return dict(
        schema="rns8_linux_platform_receipt_v1",
        platform_lock_sha256=reference["sha256"],
        observed=observed,
        blockers=blockers,
        reproducible_inputs_match=not blockers,
        image_digest=lock["image_digest"],
        image_identity_scope="operator_declared_not_native_attestation",
        supported_tuple_qualified=False,
        loaded_firmware_qualified=False,
        runtime_transitive_abi_qualified=False,
        promotion_allowed=False,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        reference = load_lock(args.lock)
        observed = observe(reference["lock"])
        if load_lock(args.lock) != reference:
            raise ValueError("platform lock changed during native observation")
        receipt = evaluate(reference, observed)
        receipt["process_id"] = os.getpid()
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(receipt, sort_keys=True))
    return 1 if receipt["blockers"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
