#!/usr/bin/env python3
"""Self-test release tree checks without mutating the repository."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import check_release_tree as release_tree


MIT_LICENSE = "MIT License\n\nPermission is hereby granted, free of charge, to any person obtaining a copy\n"
MIT_NOTICE = "RNS8\n\nRNS8 is licensed under the MIT License. See LICENSE.\n"


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def write_metadata(root: Path, license_text: str = MIT_LICENSE, notice_text: str = MIT_NOTICE) -> None:
    (root / "LICENSE").write_text(license_text, encoding="utf-8")
    (root / "NOTICE").write_text(notice_text, encoding="utf-8")
    (root / "vcpkg.json").write_text(
        json.dumps(
            {
                "name": "rns8-gemm",
                "version-string": "0.1.0",
                "license": "MIT",
                "dependencies": ["boost-multiprecision", "catch2", "nlohmann-json"],
                "features": {
                    "optional-exact-libs": {
                        "dependencies": ["gmp", "flint"],
                    },
                },
            }
        ),
        encoding="utf-8",
    )


def main() -> int:
    clean_paths = [
        release_tree.normalize_path("src/core/api_context.cpp"),
        release_tree.normalize_path("include/rns8/rns8.h"),
        release_tree.normalize_path("temp/local-smoke.exe"),
        release_tree.normalize_path("build/cpu-debug/CMakeCache.txt"),
    ]
    expect(not release_tree.path_violations(clean_paths), "ignored work areas should not fail path checks")

    bad_paths = [
        release_tree.normalize_path("src/leaked.obj"),
        release_tree.normalize_path("docs/__pycache__/tool.cpython-311.pyc"),
    ]
    violations = release_tree.path_violations(bad_paths)
    expect(len(violations) == 2, "native artifacts and __pycache__ paths should fail")

    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        write_metadata(root)
        expect(not release_tree.metadata_violations(root), "valid MIT metadata should pass")

        write_metadata(root, notice_text="RNS8 is licensed under the " + "Apa" + "che License, Version 2.0.\n")
        expect(release_tree.metadata_violations(root), "stale non-MIT notice should fail")

        write_metadata(root)
        manifest = json.loads((root / "vcpkg.json").read_text(encoding="utf-8"))
        manifest["dependencies"].append("fmt")
        (root / "vcpkg.json").write_text(json.dumps(manifest), encoding="utf-8")
        expect(release_tree.metadata_violations(root), "unused default dependency should fail")

    print("release tree checker self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
