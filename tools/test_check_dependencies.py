#!/usr/bin/env python3
"""Self-test repo-local accelerator dependency reporting."""

from __future__ import annotations

import tempfile
from pathlib import Path

import check_dependencies as deps


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    root = deps.repo_root()

    ck = deps.repo_local_dependency_report("ck")
    expect(ck["relative_path"] == "third_party/rocm/composable_kernel", "CK submodule path changed")
    expect(ck["expected_branch"] == "release/rocm-rel-7.1", "CK branch changed")
    expect(ck["expected_sha"] == deps.EXPECTED_ROCM_SUBMODULES["ck"]["sha"], "CK expected SHA changed")
    if ck["status"] == "present":
        expect(bool(ck["sha_matches"]), "initialized CK submodule is not at the pinned release SHA")
    expect("ck" in deps.ACCELERATOR_PRIMITIVE_PROBE_SOURCES, "CK primitive probe source missing")
    expect(
        "DeviceGemmWmma_CShuffle" in deps.ACCELERATOR_PRIMITIVE_PROBE_SOURCES["ck"],
        "CK primitive probe no longer instantiates the WMMA CShuffle GEMM primitive",
    )

    rocwmma = deps.repo_local_dependency_report("rocwmma")
    expect(rocwmma["relative_path"] == "third_party/rocm/rocWMMA", "rocWMMA submodule path changed")
    expect(rocwmma["expected_branch"] == "release/rocm-rel-7.1", "rocWMMA branch changed")
    expect(
        rocwmma["expected_sha"] == deps.EXPECTED_ROCM_SUBMODULES["rocwmma"]["sha"],
        "rocWMMA expected SHA changed",
    )
    if rocwmma["status"] == "present":
        expect(bool(rocwmma["sha_matches"]), "initialized rocWMMA submodule is not at the pinned release SHA")
    expect("rocwmma" in deps.ACCELERATOR_PRIMITIVE_PROBE_SOURCES, "rocWMMA primitive probe source missing")
    expect(
        "mma_sync" in deps.ACCELERATOR_PRIMITIVE_PROBE_SOURCES["rocwmma"],
        "rocWMMA primitive probe no longer instantiates an int8 mma_sync path",
    )

    roc_header = root / "third_party" / "rocm" / "rocWMMA" / "library" / "include" / "rocwmma" / "rocwmma.hpp"
    expect(
        deps.include_root_for_header(str(roc_header)).replace("\\", "/").endswith("rocWMMA/library/include"),
        "rocWMMA raw-source include root was not derived from library/include",
    )

    with tempfile.TemporaryDirectory() as temp:
        temp_path = Path(temp)
        generated = deps.write_ck_generated_headers(temp_path, "abc123")
        expect((generated / "ck" / "config.h").exists(), "CK generated config.h missing")
        expect((generated / "ck" / "version.h").exists(), "CK generated version.h missing")
        roots = deps.include_roots_for_component(
            "ck",
            {
                "header": str(root / "third_party" / "rocm" / "composable_kernel" / "include" / "ck" / "ck.hpp"),
                "repo_local_dependency": {"actual_sha": "abc123"},
            },
            temp_path,
        )
        expect(roots[0] == str(temp_path / "generated" / "ck" / "include"), "CK generated include root not first")

        probes = deps.accelerator_compile_probes({}, False, temp_path / "probes", None)
        expect(probes["requested"] is False, "not-requested probe report changed")
        expect(probes["items"]["ck"]["status"] == "NOT_REQUESTED", "CK not-requested probe status changed")
        expect(
            probes["items"]["ck"]["primitive_probe_status"] == "NOT_REQUESTED",
            "CK not-requested primitive probe status changed",
        )
        expect(
            probes["items"]["rocwmma"]["primitive_probe_status"] == "NOT_REQUESTED",
            "rocWMMA not-requested primitive probe status changed",
        )
        expect(
            probes["items"]["hipblaslt"]["primitive_probe_status"] == "NOT_APPLICABLE",
            "hipBLASLt primitive probe applicability changed",
        )

    print("dependency checker self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
