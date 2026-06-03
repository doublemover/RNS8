#!/usr/bin/env python3
"""Install reviewed RNS8 autotune cache entries into a destination cache."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any


REVIEWED_RELEASE_PREFIX = "reviewed_release_"
PUBLIC_ACCELERATOR_AUTOTUNE_BACKENDS = {"hipblaslt", "ck", "wmma"}
PUBLIC_ACCELERATOR_AUTOTUNE_SEMANTICS = {
    "bounded_i64",
    "bounded_u64",
    "exact_wide_signed",
    "exact_wide_unsigned",
    "finite_ring_u8",
    "finite_field_u8",
}
BOUNDED_SEMANTICS = {"bounded_i64", "bounded_u64"}
EXACT_WIDE_SEMANTICS = {"exact_wide_signed", "exact_wide_unsigned"}
FINITE_U8_SEMANTICS = {"finite_ring_u8", "finite_field_u8"}


class AutotuneCacheInstallError(ValueError):
    pass


def default_autotune_cache_path() -> Path:
    override = os.environ.get("RNS8_AUTOTUNE_CACHE_PATH")
    if override:
        return Path(override)
    if os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            return Path(local_app_data) / "rns8-gemm" / "autotune.json"
        user_profile = os.environ.get("USERPROFILE")
        if user_profile:
            return Path(user_profile) / "AppData" / "Local" / "rns8-gemm" / "autotune.json"
    else:
        xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
        if xdg_cache_home:
            return Path(xdg_cache_home) / "rns8-gemm" / "autotune.json"
        home = os.environ.get("HOME")
        if home:
            return Path(home) / ".cache" / "rns8-gemm" / "autotune.json"
    return Path("rns8-gemm") / "autotune.json"


def key_fields(key: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for part in key.split(";"):
        if not part or "=" not in part:
            continue
        name, value = part.split("=", 1)
        if name:
            fields[name] = value
    return fields


def require_string(item: dict[str, Any], name: str) -> str:
    value = item.get(name)
    if not isinstance(value, str) or not value:
        raise AutotuneCacheInstallError(f"{name} must be a nonempty string")
    return value


def require_int(item: dict[str, Any], name: str, *, minimum: int = 0) -> int:
    value = item.get(name)
    if not isinstance(value, int) or value < minimum:
        raise AutotuneCacheInstallError(f"{name} must be an integer >= {minimum}")
    return value


def require_number(item: dict[str, Any], name: str) -> float:
    value = item.get(name)
    if not isinstance(value, (int, float)):
        raise AutotuneCacheInstallError(f"{name} must be numeric")
    return float(value)


def require_key_field(fields: dict[str, str], name: str, expected: str) -> None:
    value = fields.get(name)
    if value != expected:
        raise AutotuneCacheInstallError(f"key_{name}_mismatch")


def require_key_int(fields: dict[str, str], name: str, expected: int) -> None:
    value = fields.get(name)
    if value is None:
        raise AutotuneCacheInstallError(f"missing_key_{name}")
    try:
        parsed = int(value, 10)
    except ValueError as exc:
        raise AutotuneCacheInstallError(f"invalid_key_{name}") from exc
    if parsed != expected:
        raise AutotuneCacheInstallError(f"key_{name}_mismatch")


def optional_key_field(fields: dict[str, str], name: str, expected: str) -> None:
    value = fields.get(name)
    if value is not None and value != expected:
        raise AutotuneCacheInstallError(f"key_{name}_mismatch")


def optional_key_int(fields: dict[str, str], name: str, expected: int) -> None:
    if name not in fields:
        return
    require_key_int(fields, name, expected)


def reviewed_backend_supports_semantic_contract(selected_backend: str, semantic_contract: str) -> bool:
    return (
        selected_backend in PUBLIC_ACCELERATOR_AUTOTUNE_BACKENDS
        and semantic_contract in PUBLIC_ACCELERATOR_AUTOTUNE_SEMANTICS
    )


def reviewed_kernel_supported_for_contract(
    selected_backend: str, semantic_contract: str, selected_kernel: str
) -> bool:
    if selected_backend == "hipblaslt":
        return selected_kernel == "hipblaslt_int8_i32_scratch_reduce_baseline_v1"
    if selected_backend == "ck":
        if semantic_contract in FINITE_U8_SEMANTICS:
            return selected_kernel == "ck_wmma_cshuffle_finite_u8_centered_epilogue_v1"
        if semantic_contract in EXACT_WIDE_SEMANTICS:
            return selected_kernel == "ck_wmma_cshuffle_i8_i32_centered_epilogue_v1"
        if semantic_contract in BOUNDED_SEMANTICS:
            return selected_kernel in {
                "ck_wmma_cshuffle_i8_i32_centered_epilogue_v1",
                "ck_wmma_cshuffle_tiled_i8_i32_centered_epilogue_v1",
            }
    if selected_backend == "wmma":
        if semantic_contract in FINITE_U8_SEMANTICS:
            return selected_kernel == "rocwmma_i8_i32_signed_finite_u8_hot_residue_v1"
        if semantic_contract in EXACT_WIDE_SEMANTICS:
            return selected_kernel == "rocwmma_i8_i32_signed_hot_residue_v1"
        if semantic_contract in BOUNDED_SEMANTICS:
            return selected_kernel in {
                "rocwmma_i8_i32_signed_hot_residue_v1",
                "rocwmma_i8_i32_signed_tiled_hot_residue_v1",
            }
    return False


def reviewed_epilogue_supported_for_contract(
    selected_backend: str, semantic_contract: str, epilogue: str
) -> bool:
    if selected_backend == "hipblaslt":
        if semantic_contract in FINITE_U8_SEMANTICS:
            return epilogue == "separate_i32_scratch_reduce_then_canonical_u8_export"
        if semantic_contract in EXACT_WIDE_SEMANTICS:
            return epilogue == "separate_i32_scratch_reduce_rns_output"
        if semantic_contract in BOUNDED_SEMANTICS:
            return epilogue == "separate_i32_scratch_reduce_then_crt_export"
    if selected_backend == "ck":
        if semantic_contract in FINITE_U8_SEMANTICS:
            return epilogue == "ck_fused_i32_to_centered_residue_then_canonical_u8_export"
        if semantic_contract in EXACT_WIDE_SEMANTICS:
            return epilogue == "ck_fused_i32_to_centered_residue_rns_output"
        if semantic_contract in BOUNDED_SEMANTICS:
            return epilogue == "ck_fused_i32_to_centered_residue_then_crt_export"
    if selected_backend == "wmma":
        if semantic_contract in FINITE_U8_SEMANTICS:
            return epilogue == "rocwmma_fused_i32_to_centered_residue_then_canonical_u8_export"
        if semantic_contract in EXACT_WIDE_SEMANTICS:
            return epilogue == "rocwmma_fused_i32_to_centered_residue_rns_output"
        if semantic_contract in BOUNDED_SEMANTICS:
            return epilogue == "rocwmma_fused_i32_to_centered_residue_then_crt_export"
    return False


def validate_entry(entry: Any, *, source: Path, index: int) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise AutotuneCacheInstallError(f"{source}: entry {index} must be an object")
    try:
        key = require_string(entry, "key")
        selected_backend = require_string(entry, "selected_backend")
        selected_kernel = require_string(entry, "selected_kernel")
        target_id = require_string(entry, "target_id")
        version = require_string(entry, "hip_sdk_or_library_version")
        semantic_contract = require_string(entry, "semantic_contract")
        layout = require_string(entry, "layout")
        require_string(entry, "prefix_schedule_hash")
        epilogue = require_string(entry, "epilogue")
        kernel_family = require_string(entry, "kernel_family")
        validation_status = require_string(entry, "validation_status")
        if entry.get("performance_validated") is not True:
            raise AutotuneCacheInstallError("performance_validated must be true")
        if not validation_status.startswith(REVIEWED_RELEASE_PREFIX):
            raise AutotuneCacheInstallError("validation_status must start with reviewed_release_")
        if require_int(entry, "schema_version", minimum=0) != 1:
            raise AutotuneCacheInstallError("entry schema_version must be 1")
        if not reviewed_backend_supports_semantic_contract(selected_backend, semantic_contract):
            raise AutotuneCacheInstallError("unsupported_autotune_backend_semantic_contract")

        shape = entry.get("shape")
        if not isinstance(shape, dict):
            raise AutotuneCacheInstallError("shape must be an object")
        m = require_int(shape, "m", minimum=1)
        n = require_int(shape, "n", minimum=1)
        k = require_int(shape, "k", minimum=1)
        k_block_size = require_int(entry, "k_block_size", minimum=0)
        tile_m = require_int(entry, "tile_m", minimum=1)
        tile_n = require_int(entry, "tile_n", minimum=1)
        require_int(entry, "workspace_bytes", minimum=0)

        medians = entry.get("measured_medians_us")
        if not isinstance(medians, dict):
            raise AutotuneCacheInstallError("measured_medians_us must be an object")
        for name in ["pack", "rns_gemm", "crt_export", "end_to_end"]:
            require_number(medians, name)

        fields = key_fields(key)
        require_key_field(fields, "backend", selected_backend)
        require_key_field(fields, "semantics", semantic_contract)
        require_key_field(fields, "kernel", selected_kernel)
        require_key_field(fields, "epilogue", epilogue)
        if kernel_family != selected_kernel:
            raise AutotuneCacheInstallError("kernel_family_mismatch")
        require_key_int(fields, "m", m)
        require_key_int(fields, "n", n)
        require_key_int(fields, "k", k)
        require_key_int(fields, "tile_m", tile_m)
        require_key_int(fields, "tile_n", tile_n)
        optional_key_int(fields, "k_block_size", k_block_size)
        optional_key_field(fields, "target", target_id)
        optional_key_field(fields, "target_id", target_id)
        optional_key_field(fields, "version", version)
        optional_key_field(fields, "hip_sdk_or_library_version", version)
        optional_key_field(fields, "layout", layout)

        finite_contract = semantic_contract in {"finite_ring_u8", "finite_field_u8"}
        raw_finite_modulus = entry.get("finite_modulus", 0)
        finite_modulus = 0 if raw_finite_modulus is None else raw_finite_modulus
        if not isinstance(finite_modulus, int) or finite_modulus < 0:
            raise AutotuneCacheInstallError("finite_modulus must be an integer >= 0")
        if finite_contract:
            if finite_modulus == 0:
                raise AutotuneCacheInstallError("missing_entry_finite_modulus")
            require_key_int(fields, "finite_modulus", finite_modulus)
        elif finite_modulus != 0:
            raise AutotuneCacheInstallError("unexpected_entry_finite_modulus")
        elif "finite_modulus" in fields:
            raise AutotuneCacheInstallError("unexpected_key_finite_modulus")
        entry["finite_modulus"] = finite_modulus

        if not reviewed_kernel_supported_for_contract(selected_backend, semantic_contract, selected_kernel):
            raise AutotuneCacheInstallError("unsupported_autotune_kernel_for_contract")
        if not reviewed_epilogue_supported_for_contract(selected_backend, semantic_contract, epilogue):
            raise AutotuneCacheInstallError("unsupported_autotune_epilogue_for_contract")
    except AutotuneCacheInstallError as exc:
        raise AutotuneCacheInstallError(f"{source}: entry {index}: {exc}") from exc
    return entry


def read_cache(path: Path, *, allow_missing: bool = False) -> list[dict[str, Any]]:
    if allow_missing and not path.exists():
        return []
    try:
        root = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        if allow_missing:
            return []
        raise
    if not isinstance(root, dict):
        raise AutotuneCacheInstallError(f"{path}: root must be an object")
    if root.get("schema_version") != 1:
        raise AutotuneCacheInstallError(f"{path}: schema_version must be 1")
    entries = root.get("entries")
    if not isinstance(entries, list):
        raise AutotuneCacheInstallError(f"{path}: entries must be an array")
    return [validate_entry(entry, source=path, index=index) for index, entry in enumerate(entries)]


def write_cache(path: Path, entries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"schema_version": 1, "entries": entries}, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temp_path = Path(handle.name)
        handle.write(payload)
    temp_path.replace(path)


def install_cache(
    sources: list[Path],
    destination: Path,
    *,
    dry_run: bool = False,
    replace_existing: bool = False,
) -> dict[str, Any]:
    if replace_existing:
        existing_entries: list[dict[str, Any]] = []
    else:
        try:
            existing_entries = read_cache(destination, allow_missing=True)
        except AutotuneCacheInstallError as exc:
            raise AutotuneCacheInstallError(
                f"{destination}: existing cache failed reviewed-cache validation; "
                f"rerun with --replace-existing to discard existing entries: {exc}"
            ) from exc
    merged = {entry["key"]: entry for entry in existing_entries}
    initial_keys = set(merged)
    source_entries = 0
    for source in sources:
        for entry in read_cache(source):
            source_entries += 1
            merged[entry["key"]] = entry
    ordered = [merged[key] for key in sorted(merged)]
    added = len(set(merged) - initial_keys)
    replaced = sum(1 for key in initial_keys if merged[key] not in existing_entries)
    if not dry_run:
        write_cache(destination, ordered)
    return {
        "destination": str(destination),
        "dry_run": dry_run,
        "replace_existing": replace_existing,
        "sources": [str(source) for source in sources],
        "source_entries": source_entries,
        "existing_entries": len(existing_entries),
        "installed_entries": len(ordered),
        "added_entries": added,
        "replaced_entries": replaced,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", required=True, help="reviewed cache JSON to install")
    parser.add_argument("--destination", type=Path, help="destination cache path; defaults to RNS8 autotune cache path")
    parser.add_argument("--dry-run", action="store_true", help="validate and report without writing the destination")
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="discard existing destination entries after validating the reviewed sources",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    destination = args.destination or default_autotune_cache_path()
    summary = install_cache(args.source, destination, dry_run=args.dry_run, replace_existing=args.replace_existing)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
