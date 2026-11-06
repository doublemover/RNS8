"""Bounded native ELF build-ID custody; symbols still come from native perf."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import struct

SOURCE = "https://github.com/torvalds/linux/blob/47f05f71ad988c3180bdba807151e6e86ed75aa3/include/uapi/linux/elf.h"
BUILD_ID_SCOPE = "https://sourceware.org/binutils/docs/ld/Options.html#index-_002d_002dbuild_002did"
HEADER = struct.Struct("<16sHHIQQQIHHHHHH")
PROGRAM = struct.Struct("<IIQQQQQQ")
SECTION = struct.Struct("<IIQQQQIIQQ")


def bounded_digest(stream, size):
    stream.seek(0)
    total, value = 0, hashlib.sha256()
    while True:
        chunk = stream.read(min(1024**2, size + 1 - total))
        if not chunk:
            break
        total += len(chunk)
        if total > size:
            raise ModuleChangedError("ELF module grew during bounded inspection")
        value.update(chunk)
    if total != size:
        raise ModuleChangedError("ELF module size changed during inspection")
    return value.hexdigest()


class ModuleChangedError(ValueError):
    pass


class ModuleBudgetError(ValueError):
    pass


def identity(path: Path, *, max_bytes=512 * 1024**2) -> dict:
    path = path.resolve(strict=True)
    if type(max_bytes) is not int or not 64 <= max_bytes <= 2 * 1024**3:
        raise ValueError("bounded ELF module byte budget required")
    if not path.is_file():
        raise ValueError("regular ELF module required")
    with path.open("rb") as stream:
        original = os.fstat(stream.fileno())
        size = original.st_size
        if size < HEADER.size:
            raise ValueError("ELF module header is truncated")
        if size > max_bytes:
            raise ModuleBudgetError("ELF module exceeds byte budget")

        def read(offset, length):
            if offset < 0 or length < 0 or offset > size or length > size - offset:
                raise ValueError("ELF table/note extends outside the actual module")
            stream.seek(offset)
            data = stream.read(length)
            if len(data) != length:
                raise ModuleChangedError("ELF module truncated during inspection")
            return data

        header = HEADER.unpack(read(0, HEADER.size))
        (
            ident,
            kind,
            machine,
            version,
            _,
            phoff,
            shoff,
            _,
            ehsize,
            phsize,
            phcount,
            shsize,
            shcount,
            _,
        ) = header
        if (
            ident[:7] != b"\x7fELF\x02\x01\x01"
            or version != 1
            or ehsize != HEADER.size
            or kind not in (2, 3)
            or machine not in (62, 183)
        ):
            raise ValueError(
                "supported native little-endian x86-64/AArch64 executable ELF required"
            )
        before = bounded_digest(stream, size)
        if HEADER.unpack(read(0, HEADER.size)) != header:
            raise ModuleChangedError("ELF module header changed during inspection")
        first = None
        if shoff:
            if shsize != SECTION.size:
                raise ValueError("unsupported ELF section-header size")
            first = SECTION.unpack(read(shoff, shsize))
            if first[1] != 0:
                raise ValueError("ELF section zero must be null")
            if shcount == 0:
                shcount = first[5]
        elif shcount:
            raise ValueError("ELF section table missing")
        if phcount == 65535:
            if first is None:
                raise ValueError("ELF extended program count lacks section zero")
            phcount = first[7]
        if not 0 <= phcount <= 4096 or not 0 <= shcount <= 65536:
            raise ModuleBudgetError("ELF header-count budget exhausted")
        notes, symbol_tables = set(), []
        if phcount:
            if not phoff or phsize != PROGRAM.size:
                raise ValueError("unsupported ELF program table")
            for row in PROGRAM.iter_unpack(read(phoff, phcount * phsize)):
                if row[0] == 4 and row[5]:
                    notes.add((row[2], row[5]))
        if shcount:
            for row in SECTION.iter_unpack(read(shoff, shcount * shsize)):
                if row[1] == 7 and row[5]:
                    notes.add((row[4], row[5]))
                elif row[1] in (2, 11):
                    if row[4] > size or row[5] > size - row[4]:
                        raise ValueError("ELF symbol-table extent is outside the actual file")
                    symbol_tables.append("static" if row[1] == 2 else "dynamic")
        if sum(length for _, length in notes) > 8 * 1024**2:
            raise ModuleBudgetError("ELF note payload budget exhausted")
        build_ids = set()
        for offset, length in sorted(notes):
            if length > 1024**2:
                raise ModuleBudgetError("ELF note extent exceeds budget")
            data, position = read(offset, length), 0
            while position < len(data):
                remaining = len(data) - position
                if remaining < 12:
                    if any(data[position:]):
                        raise ValueError("truncated ELF note header")
                    break
                names, payload, tag = struct.unpack_from("<III", data, position)
                name_start = position + 12
                payload_start = name_start + ((names + 3) & ~3)
                end = payload_start + ((payload + 3) & ~3)
                if end > len(data):
                    raise ValueError("truncated ELF note name/payload")
                if tag == 3 and data[name_start : name_start + names] == b"GNU\0":
                    if not 1 <= payload <= 64:
                        raise ValueError("GNU build-ID payload outside supported budget")
                    build_ids.add(data[payload_start : payload_start + payload].hex())
                position = end
        if len(build_ids) > 1:
            raise ValueError("ELF module contains contradictory GNU build IDs")
        after = os.fstat(stream.fileno())
        visible = path.stat()
        if (
            bounded_digest(stream, size) != before
            or (original.st_dev, original.st_ino, size)
            != (after.st_dev, after.st_ino, after.st_size)
            or (visible.st_dev, visible.st_ino, visible.st_size)
            != (after.st_dev, after.st_ino, size)
        ):
            raise ModuleChangedError("ELF module identity changed during inspection")
    return dict(
        path=str(path),
        sha256=before,
        bytes=size,
        machine=machine,
        elf_type=kind,
        gnu_build_id=next(iter(build_ids), None),
        symbol_table_classes=sorted(set(symbol_tables)),
        scope="current_ELF_file_and_native_recorded_build_ID_not_transient_loaded_image_history",
        build_id_is_content_checksum=False,
    )


def module_inventory(
    samples: list[dict],
    *,
    machine: int,
    max_modules=128,
    max_module_bytes=512 * 1024**2,
    max_total_bytes=2 * 1024**3,
) -> dict:
    """Bind leaf DSO reports; absent/deleted/vDSO/unwind identities stay partial."""
    import re

    if (
        type(machine) is not int
        or machine not in (62, 183)
        or type(max_modules) is not int
        or not 1 <= max_modules <= 128
        or type(max_total_bytes) is not int
        or not 1 <= max_total_bytes <= 8 * 1024**3
        or type(max_module_bytes) is not int
        or not 64 <= max_module_bytes <= 2 * 1024**3
    ):
        raise ValueError("bounded native ELF module inventory required")
    records, files, errors, total = {}, {}, {}, 0
    for sample in samples:
        name, build_id = sample.get("dso_path"), sample.get("dso_build_id")
        # Native perf emits an empty string when a resolved DSO has no build ID.
        name = None if name == "" else name
        build_id = None if build_id == "" else build_id
        if (name is not None and (type(name) is not str or not name or len(name) > 4096)) or (
            build_id is not None
            and (
                type(build_id) is not str or not re.fullmatch(r"(?:[0-9a-fA-F]{2}){1,64}", build_id)
            )
        ):
            raise ValueError("malformed native DSO path/build ID")
        key = name, build_id.lower() if build_id is not None else None
        if key not in records:
            if len(records) == max_modules:
                raise ValueError("native DSO inventory exceeds module budget")
            row = dict(
                native_path=name,
                native_build_id=key[1],
                samples=0,
                symbolized_samples=0,
                source_location_samples=0,
                state="unresolved",
                reason="missing_native_path_or_build_id",
                file=None,
            )
            records[key] = row
            if name is not None and build_id is not None:
                path = Path(name)
                if not path.is_absolute():
                    row["reason"] = "nonabsolute_or_special_native_module"
                else:
                    try:
                        resolved = path.resolve(strict=True)
                        if resolved not in files:
                            size = resolved.stat().st_size
                            if size > max_module_bytes or size > max_total_bytes - total:
                                raise ModuleBudgetError(
                                    "native DSO aggregate/per-module byte budget exhausted"
                                )
                            # Charge failed-format files too, and decode each path once.
                            total += size
                            try:
                                files[resolved] = identity(resolved, max_bytes=max_module_bytes)
                            except (ModuleChangedError, ModuleBudgetError):
                                raise
                            except ValueError as error:
                                files[resolved] = None
                                errors[resolved] = str(error)
                        if files[resolved] is None:
                            raise ValueError(errors[resolved])
                        row["file"] = files[resolved]
                        if row["file"]["machine"] != machine:
                            row.update(
                                state="mismatch",
                                reason="native_machine_differs_from_current_module",
                            )
                        elif row["file"]["gnu_build_id"] != key[1]:
                            row.update(
                                state="mismatch",
                                reason="native_build_id_differs_from_current_module",
                            )
                        else:
                            row.update(
                                state="matched", reason="native_build_id_matches_current_ELF"
                            )
                    except (ModuleChangedError, ModuleBudgetError):
                        raise
                    except OSError as error:
                        row["reason"] = "current_module_unavailable: " + str(error)
                    except ValueError as error:
                        row["reason"] = "current_module_unsupported_or_malformed: " + str(error)
        row = records[key]
        row["samples"] += 1
        row["symbolized_samples"] += bool(sample.get("symbol"))
        source = sample.get("source")
        row["source_location_samples"] += bool(
            isinstance(source, dict) and source.get("path") and source.get("line", 0) > 0
        )
    matched = sum(row["samples"] for row in records.values() if row["state"] == "matched")
    for row in records.values():
        if row["file"] is not None and Path(row["native_path"]).resolve(strict=True) != Path(
            row["file"]["path"]
        ):
            raise ModuleChangedError("native DSO path changed its resolved target")
    for path, file in files.items():
        if file is not None:
            with path.open("rb") as stream:
                if bounded_digest(stream, file["bytes"]) != file["sha256"]:
                    raise ModuleChangedError("earlier native DSO changed during module inventory")
    return dict(
        schema="rns8_perf_leaf_module_custody_v1",
        modules=list(records.values()),
        selected_samples=len(samples),
        matched_build_id_samples=matched,
        unresolved_or_mismatched_samples=len(samples) - matched,
        module_file_byte_budget_charged=total,
        complete=bool(samples) and matched == len(samples),
        matched_leaf_symbolized_samples=sum(
            row["symbolized_samples"] for row in records.values() if row["state"] == "matched"
        ),
        matched_leaf_source_location_samples=sum(
            row["source_location_samples"] for row in records.values() if row["state"] == "matched"
        ),
        callchain_module_identities="unverified_native_paths_only",
        source_content_binding="native_debug_locations_not_verified_source_file_contents",
        loaded_image_history_verified=False,
        build_id_is_content_checksum=False,
        producer_source=SOURCE,
        build_id_scope_source=BUILD_ID_SCOPE,
    )
