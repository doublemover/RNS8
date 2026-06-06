#!/usr/bin/env python3
"""Install reviewed RNS8 autotune cache entries into a destination cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from check_dependencies_lib.config import LINUX_CDNA_TARGETS
from metadata_registry_constants import BOUNDED_SEMANTICS, EXACT_WIDE_SEMANTICS, FINITE_U8_SEMANTICS


REVIEWED_RELEASE_PREFIX = "reviewed_release_"
PUBLIC_ACCELERATOR_AUTOTUNE_BACKENDS = {"hipblaslt", "ck", "rocwmma"}
NATIVE_VECTOR_AUTOTUNE_BACKEND = "hip-vector-alu-int64"
PUBLIC_ACCELERATOR_AUTOTUNE_SEMANTICS = {
    "bounded_i64",
    "bounded_u64",
    "exact_wide_signed",
    "exact_wide_unsigned",
    "finite_ring_u8",
    "finite_field_u8",
}
CK_FINITE_SPECIALIZED_KERNELS = {
    251: "ck_wmma_cshuffle_finite_u8_mod251_centered_epilogue_v2",
    255: "ck_wmma_cshuffle_finite_u8_mod255_centered_epilogue_v2",
    256: "ck_wmma_cshuffle_finite_u8_mod256_centered_epilogue_v2",
}
ROCWMMA_FINITE_SPECIALIZED_KERNELS = {
    251: "rocwmma_i8_i32_signed_finite_u8_mod251_hot_residue_v2",
    255: "rocwmma_i8_i32_signed_finite_u8_mod255_hot_residue_v2",
    256: "rocwmma_i8_i32_signed_finite_u8_mod256_hot_residue_v2",
}


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


def entry_fingerprint(entry: dict[str, Any] | None) -> str | None:
    if entry is None:
        return None
    payload = json.dumps(entry, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def selector_key_hash(selector_key: str) -> str:
    return hashlib.sha256(selector_key.encode("utf-8")).hexdigest()[:16]


def target_class_for_id(target_id: str | None) -> str:
    target = (target_id or "").lower()
    if target in LINUX_CDNA_TARGETS:
        return "cdna"
    if target.startswith("gfx10") or target.startswith("gfx11") or target.startswith("gfx12"):
        return "rdna"
    return "unknown" if target else "missing"


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
    if selected_backend == NATIVE_VECTOR_AUTOTUNE_BACKEND:
        return semantic_contract in BOUNDED_SEMANTICS
    return (
        selected_backend in PUBLIC_ACCELERATOR_AUTOTUNE_BACKENDS
        and semantic_contract in PUBLIC_ACCELERATOR_AUTOTUNE_SEMANTICS
    )


def expected_vector_alu_kernel(semantic_contract: str, m: int, n: int, k: int) -> str | None:
    gemv_n1 = n == 1 and k >= 4096
    if semantic_contract == "bounded_i64":
        return "hip_vector_alu_i64_gemv_n1_exact_192b_v1" if gemv_n1 else "hip_vector_alu_i64_exact_192b_v1"
    if semantic_contract == "bounded_u64":
        return "hip_vector_alu_u64_gemv_n1_exact_192b_v1" if gemv_n1 else "hip_vector_alu_u64_exact_192b_v1"
    return None


def reviewed_kernel_supported_for_contract(
    selected_backend: str,
    semantic_contract: str,
    selected_kernel: str,
    finite_modulus: int,
    m: int,
    n: int,
    k: int,
) -> bool:
    if selected_backend == NATIVE_VECTOR_AUTOTUNE_BACKEND:
        return selected_kernel == expected_vector_alu_kernel(semantic_contract, m, n, k)
    if selected_backend == "hipblaslt":
        return selected_kernel == "hipblaslt_int8_i32_scratch_reduce_specialized_251_255_256_v2"
    if selected_backend == "ck":
        if semantic_contract in FINITE_U8_SEMANTICS:
            return selected_kernel == CK_FINITE_SPECIALIZED_KERNELS.get(
                finite_modulus, "ck_wmma_cshuffle_finite_u8_centered_epilogue_v1"
            )
        if semantic_contract in EXACT_WIDE_SEMANTICS:
            return selected_kernel == "ck_wmma_cshuffle_i8_i32_mod251_255_256_centered_epilogue_v2"
        if semantic_contract in BOUNDED_SEMANTICS:
            return selected_kernel in {
                "ck_wmma_cshuffle_i8_i32_mod251_255_256_centered_epilogue_v2",
                "ck_wmma_cshuffle_tiled_i8_i32_mod251_255_256_centered_epilogue_v2",
            }
    if selected_backend == "rocwmma":
        if semantic_contract in FINITE_U8_SEMANTICS:
            return selected_kernel == ROCWMMA_FINITE_SPECIALIZED_KERNELS.get(
                finite_modulus, "rocwmma_i8_i32_signed_finite_u8_hot_residue_v1"
            )
        if semantic_contract in EXACT_WIDE_SEMANTICS:
            return selected_kernel == "rocwmma_i8_i32_signed_mod251_255_256_hot_residue_v2"
        if semantic_contract in BOUNDED_SEMANTICS:
            return selected_kernel in {
                "rocwmma_i8_i32_signed_mod251_255_256_hot_residue_v2",
                "rocwmma_i8_i32_signed_tiled_mod251_255_256_hot_residue_v2",
            }
    return False


def reviewed_epilogue_supported_for_contract(
    selected_backend: str, semantic_contract: str, epilogue: str
) -> bool:
    if selected_backend == NATIVE_VECTOR_AUTOTUNE_BACKEND:
        return semantic_contract in BOUNDED_SEMANTICS and epilogue == "direct_int64_export"
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
    if selected_backend == "rocwmma":
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
        require_key_field(fields, "target_id", target_id)
        optional_key_field(fields, "target", target_id)
        optional_key_field(fields, "version", version)
        optional_key_field(fields, "hip_sdk_or_library_version", version)
        optional_key_field(fields, "layout", layout)

        export_variant = entry.get("export_variant", "default")
        reconstruction_variant = entry.get("reconstruction_variant", "default_garner")
        if not isinstance(export_variant, str) or not export_variant:
            raise AutotuneCacheInstallError("export_variant must be a nonempty string")
        if not isinstance(reconstruction_variant, str) or not reconstruction_variant:
            raise AutotuneCacheInstallError("reconstruction_variant must be a nonempty string")
        default_export_contract = export_variant == "default" and reconstruction_variant == "default_garner"
        cache_scope = entry.get("cache_scope", "runtime_exact_autotune")
        if not isinstance(cache_scope, str) or not cache_scope:
            raise AutotuneCacheInstallError("cache_scope must be a nonempty string")
        if default_export_contract:
            if cache_scope != "runtime_exact_autotune":
                raise AutotuneCacheInstallError("default_export_contract_cache_scope_mismatch")
            for stale_field in ("export_variant", "reconstruction_variant", "export_selector_hash"):
                if stale_field in fields:
                    raise AutotuneCacheInstallError(f"unexpected_default_{stale_field}_key")
        else:
            if cache_scope != "selector_review_only_non_default":
                raise AutotuneCacheInstallError("non_default_export_contract_cache_scope_mismatch")
            export_selector_key = entry.get("export_selector_key")
            if not isinstance(export_selector_key, str) or not export_selector_key:
                raise AutotuneCacheInstallError("missing_export_selector_key")
            export_selector_hash = entry.get("export_selector_hash")
            expected_selector_hash = selector_key_hash(export_selector_key)
            if export_selector_hash != expected_selector_hash:
                raise AutotuneCacheInstallError("export_selector_hash_mismatch")
            require_key_field(fields, "export_variant", export_variant)
            require_key_field(fields, "reconstruction_variant", reconstruction_variant)
            require_key_field(fields, "export_selector_hash", expected_selector_hash)

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

        if not reviewed_kernel_supported_for_contract(
            selected_backend, semantic_contract, selected_kernel, finite_modulus, m, n, k
        ):
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


def read_promotion_ledger(path: Path) -> dict[str, dict[str, Any]]:
    try:
        root = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AutotuneCacheInstallError(f"{path}: failed to read promotion ledger: {exc}") from exc
    if not isinstance(root, dict):
        raise AutotuneCacheInstallError(f"{path}: promotion ledger root must be an object")
    entries = root.get("entries")
    if not isinstance(entries, list):
        raise AutotuneCacheInstallError(f"{path}: promotion ledger entries must be an array")
    by_key: dict[str, dict[str, Any]] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise AutotuneCacheInstallError(f"{path}: promotion ledger entry {index} must be an object")
        key = entry.get("autotune_key")
        if not isinstance(key, str) or not key:
            raise AutotuneCacheInstallError(f"{path}: promotion ledger entry {index} missing autotune_key")
        if key in by_key:
            raise AutotuneCacheInstallError(f"{path}: duplicate promotion ledger autotune_key {key}")
        by_key[key] = entry
    return by_key


def validate_promotion_ledger_gate(
    source_entries: list[dict[str, Any]],
    ledger_paths: list[Path],
    *,
    require_variance_gate: bool = False,
    require_target_validation_gate: bool = False,
) -> None:
    ledgers: dict[str, dict[str, Any]] = {}
    for path in ledger_paths:
        for key, entry in read_promotion_ledger(path).items():
            if key in ledgers and ledgers[key] != entry:
                raise AutotuneCacheInstallError(f"{path}: conflicting promotion ledger entry for {key}")
            ledgers[key] = entry
    if require_variance_gate and not ledger_paths:
        raise AutotuneCacheInstallError("--require-variance-gate requires at least one --promotion-ledger")
    for entry in source_entries:
        key = entry["key"]
        ledger_entry = ledgers.get(key)
        if ledger_entry is None:
            raise AutotuneCacheInstallError(f"missing_promotion_ledger_entry:{key}")
        blockers = ledger_entry.get("promotion_blockers")
        if not isinstance(blockers, list):
            raise AutotuneCacheInstallError(f"promotion_ledger_blockers_malformed:{key}")
        if blockers:
            joined = ",".join(str(blocker) for blocker in blockers)
            raise AutotuneCacheInstallError(f"promotion_ledger_blocked:{key}:{joined}")
        if ledger_entry.get("performance_validated") is not True:
            raise AutotuneCacheInstallError(f"promotion_ledger_not_performance_validated:{key}")
        if require_variance_gate:
            if ledger_entry.get("variance_gate_available") is not True:
                raise AutotuneCacheInstallError(f"promotion_ledger_missing_variance_gate:{key}")
            if ledger_entry.get("variance_gate_ready") is not True:
                raise AutotuneCacheInstallError(f"promotion_ledger_variance_gate_not_ready:{key}")
        target_id = str(entry.get("target_id") or "").lower()
        target_gate_required = require_target_validation_gate or target_id in LINUX_CDNA_TARGETS
        if target_gate_required:
            if ledger_entry.get("target_validation_gate_available") is not True:
                raise AutotuneCacheInstallError(f"promotion_ledger_missing_target_validation_gate:{key}")
            if ledger_entry.get("target_validation_gate_ready") is not True:
                raise AutotuneCacheInstallError(f"promotion_ledger_target_validation_gate_not_ready:{key}")
            if ledger_entry.get("target_cache_eligible") is not True:
                raise AutotuneCacheInstallError(f"promotion_ledger_target_cache_not_eligible:{key}")


def write_cache(path: Path, entries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"schema_version": 1, "entries": entries}, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temp_path = Path(handle.name)
        handle.write(payload)
    temp_path.replace(path)


def cache_coverage_summary(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str, str, str], int] = {}
    for entry in entries:
        shape = entry.get("shape") if isinstance(entry.get("shape"), dict) else {}
        if all(isinstance(shape.get(key), int) for key in ("m", "n", "k")):
            max_dim = max(int(shape[key]) for key in ("m", "n", "k"))
            shape_family = "small" if max_dim <= 128 else "medium" if max_dim <= 1024 else "large"
        else:
            shape_family = "unknown"
        target_id = str(entry.get("target_id") or "unknown")
        key = (
            str(entry.get("semantic_contract") or "unknown"),
            shape_family,
            str(entry.get("selected_backend") or "unknown"),
            target_class_for_id(target_id),
            str(entry.get("export_variant") or "default"),
            str(entry.get("reconstruction_variant") or "default_garner"),
        )
        groups[key] = groups.get(key, 0) + 1
    return [
        {
            "semantic_contract": semantic,
            "shape_family": shape_family,
            "backend": backend,
            "target_class": target_class,
            "export_variant": export_variant,
            "reconstruction_variant": reconstruction_variant,
            "entry_count": count,
        }
        for (semantic, shape_family, backend, target_class, export_variant, reconstruction_variant), count in sorted(
            groups.items()
        )
    ]


def install_cache(
    sources: list[Path],
    destination: Path,
    *,
    dry_run: bool = False,
    replace_existing: bool = False,
    promotion_ledgers: list[Path] | None = None,
    require_variance_gate: bool = False,
    require_target_validation_gate: bool = False,
    allow_selector_review_cache: bool = False,
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
    existing_by_key = {entry["key"]: entry for entry in existing_entries}
    merged = dict(existing_by_key)
    initial_keys = set(merged)
    source_entries = 0
    reviewed_source_entries: list[dict[str, Any]] = []
    source_by_key: dict[str, str] = {}
    for source in sources:
        entries = read_cache(source)
        source_entries += len(entries)
        reviewed_source_entries.extend(entries)
        for entry in entries:
            source_by_key[entry["key"]] = str(source)
    if not allow_selector_review_cache:
        for entry in reviewed_source_entries:
            if entry.get("cache_scope", "runtime_exact_autotune") != "runtime_exact_autotune":
                raise AutotuneCacheInstallError(
                    "selector_review_only_cache_entry_not_runtime_installable:" + str(entry.get("key") or "<missing>")
                )
    effective_require_variance_gate = require_variance_gate or bool(promotion_ledgers)
    if not dry_run and not promotion_ledgers:
        raise AutotuneCacheInstallError("promotion_ledger_required_for_cache_install")
    if promotion_ledgers:
        validate_promotion_ledger_gate(
            reviewed_source_entries,
            promotion_ledgers,
            require_variance_gate=effective_require_variance_gate,
            require_target_validation_gate=require_target_validation_gate,
        )
    elif effective_require_variance_gate:
        raise AutotuneCacheInstallError("--require-variance-gate requires at least one --promotion-ledger")
    elif require_target_validation_gate:
        raise AutotuneCacheInstallError("--require-target-validation-gate requires at least one --promotion-ledger")
    replacement_history: list[dict[str, Any]] = []
    for entry in reviewed_source_entries:
        old_entry = merged.get(entry["key"])
        if old_entry != entry:
            replacement_history.append(
                {
                    "action": "replace" if old_entry is not None else "add",
                    "key": entry["key"],
                    "old_entry_fingerprint": entry_fingerprint(old_entry),
                    "new_entry_fingerprint": entry_fingerprint(entry),
                    "selected_backend": entry.get("selected_backend"),
                    "selected_kernel": entry.get("selected_kernel"),
                    "semantic_contract": entry.get("semantic_contract"),
                    "target_id": entry.get("target_id"),
                    "target_class": target_class_for_id(entry.get("target_id")),
                    "export_variant": entry.get("export_variant", "default"),
                    "reconstruction_variant": entry.get("reconstruction_variant", "default_garner"),
                    "export_selector_hash": entry.get("export_selector_hash"),
                    "cache_scope": entry.get("cache_scope", "runtime_exact_autotune"),
                    "evidence_source": source_by_key.get(entry["key"]),
                    "updated_utc": entry.get("updated_utc"),
                    "validation_status": entry.get("validation_status"),
                    "validation_command_family": "reviewed_release_with_promotion_ledger"
                    if promotion_ledgers
                    else "reviewed_release_cache_validation_only",
                }
            )
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
        "promotion_ledgers": [str(path) for path in (promotion_ledgers or [])],
        "require_variance_gate": effective_require_variance_gate,
        "require_target_validation_gate": require_target_validation_gate,
        "allow_selector_review_cache": allow_selector_review_cache,
        "source_entries": source_entries,
        "existing_entries": len(existing_entries),
        "installed_entries": len(ordered),
        "added_entries": added,
        "replaced_entries": replaced,
        "replacement_history": replacement_history,
        "cache_coverage": cache_coverage_summary(ordered),
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
    parser.add_argument(
        "--promotion-ledger",
        type=Path,
        action="append",
        default=[],
        help="promotion_ledger.py output; when supplied every source entry must have an unblocked ledger row",
    )
    parser.add_argument(
        "--require-variance-gate",
        action="store_true",
        help=(
            "require supplied promotion ledger rows to include a ready perf_variance_report gate; "
            "this is now the default whenever --promotion-ledger is supplied"
        ),
    )
    parser.add_argument(
        "--require-target-validation-gate",
        action="store_true",
        help="require supplied promotion ledger rows to include matching target_validation_report cache eligibility",
    )
    parser.add_argument(
        "--allow-selector-review-cache",
        action="store_true",
        help=(
            "allow non-default export/reconstruction selector review entries in the destination cache artifact; "
            "normal runtime cache installs reject these because AUTO cannot route them"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    destination = args.destination or default_autotune_cache_path()
    summary = install_cache(
        args.source,
        destination,
        dry_run=args.dry_run,
        replace_existing=args.replace_existing,
        promotion_ledgers=args.promotion_ledger,
        require_variance_gate=args.require_variance_gate,
        require_target_validation_gate=args.require_target_validation_gate,
        allow_selector_review_cache=args.allow_selector_review_cache,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
