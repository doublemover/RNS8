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
    finite = f";finite_modulus={finite_modulus}" if finite_modulus else ""
    key = (
        f"backend=ck;target=gfx1100;version=repo-local release/rocm-rel-7.1;semantics={semantics};"
        f"m=512;n=512;k=512{finite};layout=row_major;k_block_size=512;tile_m=128;tile_n=128;"
        f"kernel=unit_kernel{key_suffix};epilogue=unit_epilogue"
    )
    return {
        "key": key,
        "selected_backend": "ck",
        "selected_kernel": f"unit_kernel{key_suffix}",
        "target_id": "gfx1100",
        "hip_sdk_or_library_version": "repo-local release/rocm-rel-7.1",
        "semantic_contract": semantics,
        "finite_modulus": finite_modulus,
        "shape": {"m": 512, "n": 512, "k": 512},
        "layout": "row_major",
        "prefix_schedule_hash": "groups=1;adaptive_prefix=0;adaptive_skip=0",
        "k_block_size": 512,
        "tile_m": 128,
        "tile_n": 128,
        "epilogue": "unit_epilogue",
        "kernel_family": f"unit_kernel{key_suffix}",
        "workspace_bytes": 4096,
        "measured_medians_us": {"pack": 1.0, "rns_gemm": 2.0, "crt_export": 3.0, "end_to_end": 4.0},
        "performance_validated": True,
        "validation_status": "reviewed_release_same_contract_fastest_windows_gfx1100",
        "schema_version": 1,
        "updated_utc": "2026-06-03T00:00:00Z",
    }


def write_cache(path: Path, entries: list[dict]) -> None:
    path.write_text(json.dumps({"schema_version": 1, "entries": entries}, indent=2), encoding="utf-8")


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
        write_cache(destination, [old])
        write_cache(source_a, [replacement])
        write_cache(source_b, [finite])

        dry_run = install_autotune_cache.install_cache([source_a, source_b], destination, dry_run=True)
        assert dry_run["dry_run"] is True
        assert dry_run["replace_existing"] is False
        assert dry_run["installed_entries"] == 2
        assert json.loads(destination.read_text(encoding="utf-8"))["entries"][0]["measured_medians_us"][
            "end_to_end"
        ] == 4.0

        summary = install_autotune_cache.install_cache([source_a, source_b], destination)
        assert summary["source_entries"] == 2
        assert summary["existing_entries"] == 1
        assert summary["installed_entries"] == 2
        assert summary["added_entries"] == 1
        assert summary["replaced_entries"] == 1
        installed = json.loads(destination.read_text(encoding="utf-8"))["entries"]
        assert [item["key"] for item in installed] == sorted(item["key"] for item in installed)
        assert any(item["finite_modulus"] == 251 for item in installed)
        assert any(item["measured_medians_us"]["end_to_end"] == 1.5 for item in installed)

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
