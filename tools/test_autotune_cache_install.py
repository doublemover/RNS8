#!/usr/bin/env python3
"""Self-test reviewed autotune cache installation."""

from __future__ import annotations

import copy
import json
import os
import tempfile
from pathlib import Path

import install_autotune_cache


def entry(key_suffix: str = "", *, finite_modulus: int = 0) -> dict:
    semantics = "finite_ring_u8" if finite_modulus else "bounded_i64"
    selected_kernel = (
        {
            251: "ck_wmma_cshuffle_finite_u8_mod251_centered_epilogue_v2",
            255: "ck_wmma_cshuffle_finite_u8_mod255_centered_epilogue_v2",
            256: "ck_wmma_cshuffle_finite_u8_mod256_centered_epilogue_v2",
        }.get(finite_modulus, "ck_wmma_cshuffle_finite_u8_centered_epilogue_v1")
        if finite_modulus
        else "ck_wmma_cshuffle_i8_i32_mod251_255_256_centered_epilogue_v2"
    )
    epilogue = (
        "ck_fused_i32_to_centered_residue_then_canonical_u8_export"
        if finite_modulus
        else "ck_fused_i32_to_centered_residue_then_crt_export"
    )
    target_id = f"gfx1100{key_suffix}"
    finite = f";finite_modulus={finite_modulus}" if finite_modulus else ""
    key = (
        f"backend=ck;target_id={target_id};version=repo-local release/rocm-rel-7.1;semantics={semantics};"
        f"m=512;n=512;k=512{finite};layout=row_major;k_block_size=512;tile_m=128;tile_n=128;"
        f"kernel={selected_kernel};epilogue={epilogue}"
    )
    return {
        "key": key,
        "selected_backend": "ck",
        "selected_kernel": selected_kernel,
        "target_id": target_id,
        "hip_sdk_or_library_version": "repo-local release/rocm-rel-7.1",
        "semantic_contract": semantics,
        "finite_modulus": finite_modulus,
        "shape": {"m": 512, "n": 512, "k": 512},
        "layout": "row_major",
        "prefix_schedule_hash": "groups=1;adaptive_prefix=0;adaptive_skip=0",
        "k_block_size": 512,
        "tile_m": 128,
        "tile_n": 128,
        "epilogue": epilogue,
        "kernel_family": selected_kernel,
        "workspace_bytes": 4096,
        "measured_medians_us": {"pack": 1.0, "rns_gemm": 2.0, "crt_export": 3.0, "end_to_end": 4.0},
        "performance_validated": True,
        "validation_status": "reviewed_release_same_contract_fastest_windows_gfx1100",
        "schema_version": 1,
        "updated_utc": "2026-06-03T00:00:00Z",
    }


def vector_entry(
    key_suffix: str = "",
    *,
    semantics: str = "bounded_u64",
    m: int = 512,
    n: int = 512,
    k: int = 512,
) -> dict:
    target_id = f"gfx1100{key_suffix}"
    signed = semantics == "bounded_i64"
    gemv_n1 = n == 1 and k >= 4096
    selected_kernel = (
        "hip_vector_alu_i64_gemv_n1_exact_192b_v1"
        if signed and gemv_n1
        else "hip_vector_alu_i64_exact_192b_v1"
        if signed
        else "hip_vector_alu_u64_gemv_n1_exact_192b_v1"
        if gemv_n1
        else "hip_vector_alu_u64_exact_192b_v1"
    )
    epilogue = "direct_int64_export"
    key = (
        f"backend=hip-vector-alu-int64;target_id={target_id};version=repo-local release/rocm-rel-7.1;"
        f"semantics={semantics};m={m};n={n};k={k};layout=row_major;k_block_size={k};tile_m=128;tile_n=128;"
        f"kernel={selected_kernel};epilogue={epilogue}"
    )
    return {
        "key": key,
        "selected_backend": "hip-vector-alu-int64",
        "selected_kernel": selected_kernel,
        "target_id": target_id,
        "hip_sdk_or_library_version": "repo-local release/rocm-rel-7.1",
        "semantic_contract": semantics,
        "finite_modulus": 0,
        "shape": {"m": m, "n": n, "k": k},
        "layout": "row_major",
        "prefix_schedule_hash": "groups=1;adaptive_prefix=0;adaptive_skip=0",
        "k_block_size": k,
        "tile_m": 128,
        "tile_n": 128,
        "epilogue": epilogue,
        "kernel_family": selected_kernel,
        "workspace_bytes": 0,
        "measured_medians_us": {"pack": 1.0, "rns_gemm": 2.0, "crt_export": 3.0, "end_to_end": 4.0},
        "performance_validated": True,
        "validation_status": "reviewed_release_same_contract_fastest_windows_gfx1100",
        "schema_version": 1,
        "updated_utc": "2026-06-03T00:00:00Z",
    }


def write_cache(path: Path, entries: list[dict]) -> None:
    path.write_text(json.dumps({"schema_version": 1, "entries": entries}, indent=2), encoding="utf-8")


def write_ledger(
    path: Path,
    entries: list[dict],
    *,
    blockers: list[str] | None = None,
    variance: bool = False,
    variance_ready: bool = True,
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "policy": "reviewed_release_evidence_required_for_autotune_promotion",
                "entries": [
                    {
                        "autotune_key": item["key"],
                        "performance_validated": True,
                        "promotion_blockers": list(blockers or []),
                        "variance_gate_available": variance,
                        "variance_gate_ready": variance_ready if variance else None,
                    }
                    for item in entries
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


def check_default_path(root: Path) -> None:
    tracked = ["RNS8_AUTOTUNE_CACHE_PATH", "LOCALAPPDATA", "USERPROFILE", "XDG_CACHE_HOME", "HOME"]
    old = {name: os.environ.get(name) for name in tracked}
    try:
        override = root / "override" / "autotune.json"
        os.environ["RNS8_AUTOTUNE_CACHE_PATH"] = str(override)
        assert install_autotune_cache.default_autotune_cache_path() == override

        os.environ.pop("RNS8_AUTOTUNE_CACHE_PATH", None)
        if os.name == "nt":
            local = root / "local-app-data"
            profile = root / "profile"
            os.environ["LOCALAPPDATA"] = str(local)
            os.environ["USERPROFILE"] = str(profile)
            assert install_autotune_cache.default_autotune_cache_path() == local / "rns8-gemm" / "autotune.json"
            os.environ.pop("LOCALAPPDATA", None)
            assert (
                install_autotune_cache.default_autotune_cache_path()
                == profile / "AppData" / "Local" / "rns8-gemm" / "autotune.json"
            )
        else:
            xdg = root / "xdg-cache"
            home = root / "home"
            os.environ["XDG_CACHE_HOME"] = str(xdg)
            os.environ["HOME"] = str(home)
            assert install_autotune_cache.default_autotune_cache_path() == xdg / "rns8-gemm" / "autotune.json"
            os.environ.pop("XDG_CACHE_HOME", None)
            assert install_autotune_cache.default_autotune_cache_path() == home / ".cache" / "rns8-gemm" / "autotune.json"
    finally:
        for name, value in old.items():
            restore_env(name, value)


def main() -> int:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        check_default_path(root)
        source_a = root / "source-a.json"
        source_b = root / "source-b.json"
        destination = root / "autotune.json"
        old = entry("-old")
        replacement = copy.deepcopy(old)
        replacement["measured_medians_us"]["end_to_end"] = 1.5
        finite = entry("-finite", finite_modulus=251)
        vector = vector_entry("-vector")
        vector_gemv = vector_entry("-vector-gemv", semantics="bounded_i64", m=256, n=1, k=4096)
        write_cache(destination, [old])
        write_cache(source_a, [replacement])
        write_cache(source_b, [finite, vector, vector_gemv])

        dry_run = install_autotune_cache.install_cache([source_a, source_b], destination, dry_run=True)
        assert dry_run["dry_run"] is True
        assert dry_run["replace_existing"] is False
        assert dry_run["installed_entries"] == 4
        assert json.loads(destination.read_text(encoding="utf-8"))["entries"][0]["measured_medians_us"][
            "end_to_end"
        ] == 4.0

        summary = install_autotune_cache.install_cache([source_a, source_b], destination)
        assert summary["source_entries"] == 4
        assert summary["existing_entries"] == 1
        assert summary["installed_entries"] == 4
        assert summary["added_entries"] == 3
        assert summary["replaced_entries"] == 1
        installed = json.loads(destination.read_text(encoding="utf-8"))["entries"]
        assert [item["key"] for item in installed] == sorted(item["key"] for item in installed)
        assert any(item["finite_modulus"] == 251 for item in installed)
        assert any(item["selected_backend"] == "hip-vector-alu-int64" for item in installed)
        assert any(item["selected_kernel"] == "hip_vector_alu_i64_gemv_n1_exact_192b_v1" for item in installed)
        assert any(item["measured_medians_us"]["end_to_end"] == 1.5 for item in installed)

        ledger = root / "promotion-ledger.json"
        write_ledger(ledger, [replacement])
        ledger_summary = install_autotune_cache.install_cache(
            [source_a],
            destination,
            dry_run=True,
            promotion_ledgers=[ledger],
        )
        assert ledger_summary["promotion_ledgers"] == [str(ledger)]
        assert ledger_summary["require_variance_gate"] is False

        missing_ledger = root / "missing-promotion-ledger.json"
        write_ledger(missing_ledger, [finite])
        try:
            install_autotune_cache.install_cache([source_a], destination, promotion_ledgers=[missing_ledger])
        except install_autotune_cache.AutotuneCacheInstallError as exc:
            assert "missing_promotion_ledger_entry" in str(exc)
        else:
            raise AssertionError("source cache entry without a ledger row was accepted")

        blocked_ledger = root / "blocked-promotion-ledger.json"
        write_ledger(blocked_ledger, [replacement], blockers=["speedup_inside_variance_margin"])
        try:
            install_autotune_cache.install_cache([source_a], destination, promotion_ledgers=[blocked_ledger])
        except install_autotune_cache.AutotuneCacheInstallError as exc:
            assert "promotion_ledger_blocked" in str(exc)
            assert "speedup_inside_variance_margin" in str(exc)
        else:
            raise AssertionError("blocked promotion ledger row was accepted")

        try:
            install_autotune_cache.install_cache(
                [source_a],
                destination,
                promotion_ledgers=[ledger],
                require_variance_gate=True,
            )
        except install_autotune_cache.AutotuneCacheInstallError as exc:
            assert "promotion_ledger_missing_variance_gate" in str(exc)
        else:
            raise AssertionError("cache install accepted a missing required variance gate")

        variance_ledger = root / "variance-promotion-ledger.json"
        write_ledger(variance_ledger, [replacement], variance=True, variance_ready=True)
        variance_summary = install_autotune_cache.install_cache(
            [source_a],
            destination,
            dry_run=True,
            promotion_ledgers=[variance_ledger],
            require_variance_gate=True,
        )
        assert variance_summary["require_variance_gate"] is True

        bad = copy.deepcopy(finite)
        bad["key"] = bad["key"].replace(";finite_modulus=251", "")
        bad_source = root / "bad.json"
        write_cache(bad_source, [bad])
        try:
            install_autotune_cache.install_cache([bad_source], destination)
        except install_autotune_cache.AutotuneCacheInstallError as exc:
            assert "finite_modulus" in str(exc)
        else:
            raise AssertionError("invalid finite cache entry was accepted")

        stale_finite_kernel = entry("-finite256", finite_modulus=256)
        stale_finite_kernel["selected_kernel"] = "ck_wmma_cshuffle_finite_u8_centered_epilogue_v1"
        stale_finite_kernel["kernel_family"] = "ck_wmma_cshuffle_finite_u8_centered_epilogue_v1"
        stale_finite_kernel["key"] = stale_finite_kernel["key"].replace(
            "kernel=ck_wmma_cshuffle_finite_u8_mod256_centered_epilogue_v2",
            "kernel=ck_wmma_cshuffle_finite_u8_centered_epilogue_v1",
        )
        stale_finite_source = root / "stale-finite-kernel.json"
        write_cache(stale_finite_source, [stale_finite_kernel])
        try:
            install_autotune_cache.install_cache([stale_finite_source], destination)
        except install_autotune_cache.AutotuneCacheInstallError as exc:
            assert "unsupported_autotune_kernel_for_contract" in str(exc)
        else:
            raise AssertionError("stale finite CK kernel cache entry was accepted")

        legacy_target_key = entry()
        legacy_target_key["key"] = legacy_target_key["key"].replace(";target_id=gfx1100;", ";target=gfx1100;")
        legacy_target_source = root / "legacy-target-key.json"
        write_cache(legacy_target_source, [legacy_target_key])
        try:
            install_autotune_cache.install_cache([legacy_target_source], destination)
        except install_autotune_cache.AutotuneCacheInstallError as exc:
            assert "key_target_id_mismatch" in str(exc)
        else:
            raise AssertionError("legacy target-only cache key was accepted")

        wrap64_candidate = entry("-wrap64-candidate")
        wrap64_candidate.update(
            {
                "key": (
                    "backend=rocwmma;target_id=gfx1100;version=repo-local release/rocm-rel-7.1;"
                    "semantics=wrap_u64_mod_2_64;m=64;n=64;k=64;layout=row_major;"
                    "k_block_size=64;tile_m=16;tile_n=16;"
                    "kernel=rocwmma_wrap64_byte_gemm36_candidate_v0;epilogue=low64_wrap_export"
                ),
                "selected_backend": "rocwmma",
                "selected_kernel": "rocwmma_wrap64_byte_gemm36_candidate_v0",
                "semantic_contract": "wrap_u64_mod_2_64",
                "shape": {"m": 64, "n": 64, "k": 64},
                "k_block_size": 64,
                "tile_m": 16,
                "tile_n": 16,
                "epilogue": "low64_wrap_export",
                "kernel_family": "rocwmma_wrap64_byte_gemm36_candidate_v0",
            }
        )
        wrap64_source = root / "wrap64-candidate.json"
        write_cache(wrap64_source, [wrap64_candidate])
        try:
            install_autotune_cache.install_cache([wrap64_source], destination)
        except install_autotune_cache.AutotuneCacheInstallError as exc:
            assert "unsupported_autotune_backend_semantic_contract" in str(exc)
        else:
            raise AssertionError("wrap64 matrix-engine candidate cache entry was accepted")

        direct_baseline = entry("-hip-direct")
        direct_baseline.update(
            {
                "key": (
                    "backend=hip-direct;target_id=gfx1100;version=HIP runtime;semantics=bounded_i64;"
                    "m=512;n=512;k=512;layout=row_major;k_block_size=512;tile_m=128;tile_n=128;"
                    "kernel=direct_hip_tiled_rns_gemm_v1;epilogue=fused_centered_residue_then_crt_export"
                ),
                "selected_backend": "hip-direct",
                "selected_kernel": "direct_hip_tiled_rns_gemm_v1",
                "hip_sdk_or_library_version": "HIP runtime",
                "semantic_contract": "bounded_i64",
                "epilogue": "fused_centered_residue_then_crt_export",
                "kernel_family": "direct_hip_tiled_rns_gemm_v1",
            }
        )
        direct_source = root / "hip-direct-baseline.json"
        write_cache(direct_source, [direct_baseline])
        try:
            install_autotune_cache.install_cache([direct_source], destination)
        except install_autotune_cache.AutotuneCacheInstallError as exc:
            assert "unsupported_autotune_backend_semantic_contract" in str(exc)
        else:
            raise AssertionError("direct-HIP baseline cache entry was accepted")

        vector_exact_wide = vector_entry("-exact-wide")
        vector_exact_wide.update(
            {
                "key": vector_exact_wide["key"]
                .replace("semantics=bounded_u64", "semantics=exact_wide_unsigned")
                .replace("kernel=hip_vector_alu_u64_exact_192b_v1", "kernel=hip_vector_alu_u64_exact_192b_v1"),
                "semantic_contract": "exact_wide_unsigned",
            }
        )
        vector_exact_wide_source = root / "vector-exact-wide.json"
        write_cache(vector_exact_wide_source, [vector_exact_wide])
        try:
            install_autotune_cache.install_cache([vector_exact_wide_source], destination)
        except install_autotune_cache.AutotuneCacheInstallError as exc:
            assert "unsupported_autotune_backend_semantic_contract" in str(exc)
        else:
            raise AssertionError("non-bounded vector cache entry was accepted")

        stale_vector_gemv = vector_entry("-stale-vector-gemv", semantics="bounded_i64", m=256, n=1, k=4096)
        stale_vector_gemv["selected_kernel"] = "hip_vector_alu_i64_exact_192b_v1"
        stale_vector_gemv["kernel_family"] = "hip_vector_alu_i64_exact_192b_v1"
        stale_vector_gemv["key"] = stale_vector_gemv["key"].replace(
            "kernel=hip_vector_alu_i64_gemv_n1_exact_192b_v1",
            "kernel=hip_vector_alu_i64_exact_192b_v1",
        )
        stale_vector_gemv_source = root / "stale-vector-gemv.json"
        write_cache(stale_vector_gemv_source, [stale_vector_gemv])
        try:
            install_autotune_cache.install_cache([stale_vector_gemv_source], destination)
        except install_autotune_cache.AutotuneCacheInstallError as exc:
            assert "unsupported_autotune_kernel_for_contract" in str(exc)
        else:
            raise AssertionError("stale vector GEMV cache kernel was accepted")

        stale_kernel = entry("-stale-kernel")
        stale_kernel.update(
            {
                "key": stale_kernel["key"]
                .replace(
                    "kernel=ck_wmma_cshuffle_i8_i32_mod251_255_256_centered_epilogue_v2",
                    "kernel=rocwmma_wrap64_byte_gemm36_candidate_v0",
                ),
                "selected_kernel": "rocwmma_wrap64_byte_gemm36_candidate_v0",
                "kernel_family": "rocwmma_wrap64_byte_gemm36_candidate_v0",
            }
        )
        stale_kernel_source = root / "stale-kernel.json"
        write_cache(stale_kernel_source, [stale_kernel])
        try:
            install_autotune_cache.install_cache([stale_kernel_source], destination)
        except install_autotune_cache.AutotuneCacheInstallError as exc:
            assert "unsupported_autotune_kernel_for_contract" in str(exc)
        else:
            raise AssertionError("stale public-backend cache kernel was accepted")

        stale_bounded_v1 = entry("-stale-bounded-v1")
        stale_bounded_v1["selected_kernel"] = "ck_wmma_cshuffle_i8_i32_centered_epilogue_v1"
        stale_bounded_v1["kernel_family"] = "ck_wmma_cshuffle_i8_i32_centered_epilogue_v1"
        stale_bounded_v1["key"] = stale_bounded_v1["key"].replace(
            "kernel=ck_wmma_cshuffle_i8_i32_mod251_255_256_centered_epilogue_v2",
            "kernel=ck_wmma_cshuffle_i8_i32_centered_epilogue_v1",
        )
        stale_bounded_v1_source = root / "stale-bounded-v1.json"
        write_cache(stale_bounded_v1_source, [stale_bounded_v1])
        try:
            install_autotune_cache.install_cache([stale_bounded_v1_source], destination)
        except install_autotune_cache.AutotuneCacheInstallError as exc:
            assert "unsupported_autotune_kernel_for_contract" in str(exc)
        else:
            raise AssertionError("stale bounded CK v1 cache kernel was accepted")

        stale_epilogue = entry("-stale-epilogue")
        stale_epilogue.update(
            {
                "key": stale_epilogue["key"].replace(
                    "epilogue=ck_fused_i32_to_centered_residue_then_crt_export",
                    "epilogue=low64_wrap_export",
                ),
                "epilogue": "low64_wrap_export",
            }
        )
        stale_epilogue_source = root / "stale-epilogue.json"
        write_cache(stale_epilogue_source, [stale_epilogue])
        try:
            install_autotune_cache.install_cache([stale_epilogue_source], destination)
        except install_autotune_cache.AutotuneCacheInstallError as exc:
            assert "unsupported_autotune_epilogue_for_contract" in str(exc)
        else:
            raise AssertionError("stale public-backend cache epilogue was accepted")

        stale = root / "stale.json"
        stale_entry = copy.deepcopy(old)
        stale_entry["validation_status"] = "smoke_only"
        write_cache(stale, [stale_entry])
        try:
            install_autotune_cache.install_cache([source_a], stale)
        except install_autotune_cache.AutotuneCacheInstallError as exc:
            assert "--replace-existing" in str(exc)
        else:
            raise AssertionError("stale destination cache was silently preserved")

        replaced = install_autotune_cache.install_cache([source_a], stale, replace_existing=True)
        assert replaced["replace_existing"] is True
        assert replaced["existing_entries"] == 0
        assert replaced["installed_entries"] == 1
        installed_replacement = json.loads(stale.read_text(encoding="utf-8"))["entries"]
        assert installed_replacement[0]["validation_status"].startswith("reviewed_release_")

    print("autotune cache install self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
