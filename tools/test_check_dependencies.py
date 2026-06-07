#!/usr/bin/env python3
"""Self-test repo-local accelerator dependency reporting."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import check_dependencies as deps
import check_dependencies_lib.discovery as discovery_mod


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    root = deps.repo_root()
    expect((root / "CMakePresets.json").exists(), "dependency checker repo_root must resolve to the repository root")
    expect((root / "vcpkg.json").exists(), "dependency checker repo_root must find the repository vcpkg manifest")

    expect(deps.vcpkg_required_for_host("Windows") is True, "Windows vcpkg requirement policy changed")
    expect(deps.vcpkg_required_for_host("Linux") is False, "Linux must not require Windows vcpkg")
    expect(deps.command_required_for_host("vcpkg", "Windows") is True, "vcpkg should be Windows-required")
    expect(deps.command_required_for_host("vcpkg", "Linux") is False, "vcpkg should not be Linux-required")
    expect(
        deps.command_version_ok("cmake", "cmake version 3.22.1")[0] is False,
        "CMake below the project minimum must fail dependency readiness",
    )
    expect(
        deps.command_version_ok("cmake", "cmake version 3.28.0")[0] is True,
        "CMake at the project minimum should satisfy dependency readiness",
    )
    with patch("check_dependencies_lib.system.shutil.which", side_effect=lambda name: "/usr/bin/python3" if name == "python3" else None):
        expect(deps.find_command("python") == "/usr/bin/python3", "Linux python command slot must accept python3")
    linux_commands = deps.command_names_for_host("Linux")
    expect("vcpkg" in linux_commands, "Linux report should still show vcpkg if present")
    expect("hipInfo" not in linux_commands, "Linux report should not require Windows HIP SDK tools")
    expect("rocprofv3" in linux_commands, "Linux profiler readiness command missing")
    expect("rocprofv3-avail" in linux_commands, "Linux rocprof counter-list command missing")
    expect("numactl" in linux_commands and "lstopo" in linux_commands, "Linux topology commands missing")
    expect("all_reduce_perf" in linux_commands, "rccl-tests discovery command missing")
    expect(deps.rccl_discovery_report()["required_for_single_device_smoke"] is False, "RCCL must not block CDNA smoke")
    rocminfo_output = """
Agent 2:
  Name:                    gfx942:sramecc+:xnack-
  Marketing Name:          AMD Instinct MI300X
"""
    rocminfo_details = deps.parse_rocminfo_details(rocminfo_output)
    expect(rocminfo_details["target"] == "gfx942", "rocminfo parser must normalize target feature suffixes")
    expect(rocminfo_details["gcn_arch"] == "gfx942:sramecc+:xnack-", "rocminfo parser must retain full arch detail")
    with (
        patch("check_dependencies_lib.discovery.platform.system", return_value="Linux"),
        patch("check_dependencies_lib.discovery.find_command", return_value="/opt/rocm/bin/rocminfo"),
        patch("check_dependencies_lib.discovery.run", return_value=(0, rocminfo_output)),
    ):
        hip_info = deps.hip_info_report(None)
    expect(hip_info["ok"] is True, "Linux GPU architecture detection must fall back to rocminfo when hipInfo is absent")
    expect(hip_info["target"] == "gfx942", "rocminfo fallback must produce a supported base target id")
    with patch("check_dependencies_lib.discovery.platform.system", return_value="Linux"):
        linux_ck_roots = [str(path).replace("\\", "/") for path in deps.repo_local_accelerator_roots("ck")]
        linux_rocwmma_roots = [str(path).replace("\\", "/") for path in deps.repo_local_accelerator_roots("rocwmma")]
    expect(
        all("windows-gfx1100" not in path for path in linux_ck_roots + linux_rocwmma_roots),
        "Linux dependency discovery must not inspect Windows-gfx1100 generated accelerator roots",
    )
    with tempfile.TemporaryDirectory() as temp:
        temp_root = Path(temp)
        placeholder = temp_root / "third_party" / "rocm" / "composable_kernel"
        placeholder.mkdir(parents=True)
        (temp_root / ".gitmodules").write_text(
            "[submodule \"third_party/rocm/composable_kernel\"]\n"
            "\tpath = third_party/rocm/composable_kernel\n"
            "\turl = https://github.com/ROCm/composable_kernel.git\n"
            "\tbranch = release/rocm-rel-7.1\n",
            encoding="utf-8",
        )
        with patch.object(discovery_mod, "repo_root", return_value=temp_root):
            placeholder_ck = deps.repo_local_dependency_report("ck")
        expect(placeholder_ck["status"] == "missing", "uninitialized CK submodule placeholder must be missing")
        expect(placeholder_ck["actual_sha"] == "", "uninitialized CK submodule must not inherit superproject SHA")

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
