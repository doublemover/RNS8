from __future__ import annotations

import argparse
import glob
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

from msvc_env import command_in_msvc_environment, find_visual_studio_installation

from .config import *

def print_human(report: dict[str, object]) -> None:
    print("RNS8 dependency check")
    print("======================")
    system = report["system"]
    assert isinstance(system, dict)
    print(f"platform: {system['platform']}")
    print(f"python:   {system['python']}")
    print()

    print("Commands")
    commands = report["commands"]
    assert isinstance(commands, dict)
    for name in sorted(commands):
        item = commands[name]
        assert isinstance(item, dict)
        status = "OK" if item["ok"] else "MISSING"
        req = "required" if item["required"] else "optional"
        path = item.get("path", "")
        detail = item.get("detail", "")
        print(f"[{status}] {name} ({req})")
        if path:
            print(f"  path:   {path}")
        if detail:
            print(f"  detail: {detail}")
    print()

    msvc = report["msvc"]
    assert isinstance(msvc, dict)
    print(f"MSVC: {'OK' if msvc['ok'] else 'MISSING'} - {msvc['detail']}")
    print()

    print("Python packages")
    py_packages = report["python_packages"]
    assert isinstance(py_packages, dict)
    for name in PYTHON_PACKAGES:
        version = py_packages[name]
        print(f"[{'OK' if version else 'MISSING'}] {name}: {version or 'not found'}")
    print()

    print("CMake presets")
    cmake_presets = report["cmake_presets"]
    assert isinstance(cmake_presets, dict)
    print(f"[{'OK' if cmake_presets['ok'] else 'MISSING'}] {cmake_presets['path']}")
    print(f"  configure: {', '.join(cmake_presets['configure_presets'])}")
    print(f"  build:     {', '.join(cmake_presets['build_presets'])}")
    print(f"  test:      {', '.join(cmake_presets['test_presets'])}")
    windows = cmake_presets["windows_hip_debug"]
    linux = cmake_presets["linux_rocm_debug"]
    assert isinstance(windows, dict)
    assert isinstance(linux, dict)
    print(f"  windows HIP debug: {'OK' if windows['ok'] else 'MISSING'}")
    print(f"    toolchain: {windows.get('toolchain')}")
    print(f"    HIP root:  {windows.get('hip_root')}")
    print(f"    targets:   {windows.get('amdgpu_targets')}")
    print(f"    triplet:   {windows.get('vcpkg_triplet')}")
    print(f"  linux ROCm debug represented: {'OK' if linux['represented'] else 'MISSING'}")
    print(f"    toolchain: {linux.get('toolchain')}")
    print(f"    HIP root:  {linux.get('hip_root')}")
    print(f"    active targets:   {linux.get('amdgpu_targets')}")
    coverage_targets = linux.get("coverage_targets") or []
    assert isinstance(coverage_targets, list)
    print(f"    coverage targets: {', '.join(coverage_targets) if coverage_targets else 'not represented'}")
    missing_coverage = linux.get("missing_coverage_targets") or []
    assert isinstance(missing_coverage, list)
    print(f"    missing coverage: {', '.join(missing_coverage) if missing_coverage else 'none'}")
    print(f"    triplet:   {linux.get('vcpkg_triplet')}")
    print(f"    detail:    {linux.get('detail')}")
    print()

    print("vcpkg packages")
    manifest = report["vcpkg_manifest"]
    assert isinstance(manifest, dict)
    manifest_packages = manifest["packages"]
    assert isinstance(manifest_packages, list)
    print(f"manifest: {manifest['path']}")
    vcpkg_packages = report["vcpkg_packages"]
    assert isinstance(vcpkg_packages, dict)
    for name in manifest_packages:
        version = vcpkg_packages[name]
        role = vcpkg_package_role(name)
        print(f"[{'OK' if version else 'MISSING'}] {name} ({role}): {version or 'not installed'}")
    print()

    print("Optional CPU reference packages")
    optional_refs = report["optional_cpp_references"]
    assert isinstance(optional_refs, dict)
    for name in OPTIONAL_CPP_PACKAGES:
        item = optional_refs[name]
        assert isinstance(item, dict)
        print(f"[{'OK' if item['ok'] else 'MISSING'}] {name}: {item.get('version') or 'not installed'}")
    print()

    print("RCCL / rccl-tests readiness")
    rccl = report.get("rccl", {})
    assert isinstance(rccl, dict)
    print(f"RCCL: {'OK' if rccl.get('ok') else 'MISSING'} ({rccl.get('readiness_lane')})")
    print(f"  header: {rccl.get('header') or 'not found'}")
    print(f"  library: {rccl.get('library') or 'not found'}")
    print(f"  rccl-tests: {'OK' if rccl.get('rccl_tests_ready') else 'MISSING'}")
    tests = rccl.get("rccl_test_commands") or {}
    assert isinstance(tests, dict)
    for name in sorted(tests):
        print(f"    {name}: {tests[name] or 'not found'}")
    print(f"  detail: {rccl.get('detail')}")
    print()

    print("Accelerator components")
    accelerators = report["accelerator_components"]
    assert isinstance(accelerators, dict)
    for name in sorted(accelerators):
        item = accelerators[name]
        assert isinstance(item, dict)
        status = "OK" if item["ok"] else "MISSING"
        print(f"[{status}] {name} ({item['backend_stage']}, {item['experiment']}, optional)")
        print(f"  probe: {item['probe']}")
        print(f"  enable flag: {item['enable_flag']} ({item['enable_policy']})")
        print(f"  backend enablement: {item['backend_enablement']}")
        print(f"  evidence class: {item['evidence_class']}")
        print(f"  candidate evidence is correctness validation: {item['candidate_evidence_is_correctness_validation']}")
        print(f"  validated correctness backend: {item['validated_correctness_backend']}")
        print(f"  cmake module: {item.get('cmake_module') or 'not found'}")
        if item.get("cmake_config"):
            print(f"  cmake config: {item['cmake_config']}")
        if item.get("cmake_target"):
            print(f"  cmake target: {item['cmake_target']}")
        if item.get("header"):
            print(f"  header: {item['header']}")
        if item.get("library"):
            print(f"  library: {item['library']}")
        if item.get("library_format"):
            print(f"  library format: {item['library_format']}")
        if item.get("msvc_import_library"):
            print(f"  MSVC import library: {item['msvc_import_library']}")
        if item.get("import_archive"):
            print(f"  import archive: {item['import_archive']}")
        if item.get("runtime_library"):
            print(f"  runtime library: {item['runtime_library']}")
        if item.get("tool"):
            print(f"  tool: {item['tool']}")
        if item.get("link_guidance"):
            print(f"  link guidance: {item['link_guidance']}")
        repo_dependency = item.get("repo_local_dependency")
        if isinstance(repo_dependency, dict):
            print(f"  repo-local dependency: {repo_dependency.get('status')}")
            print(f"    path: {repo_dependency.get('path')}")
            print(f"    expected branch: {repo_dependency.get('expected_branch')}")
            print(f"    actual branch: {repo_dependency.get('actual_branch') or 'not initialized'}")
            print(f"    expected sha: {repo_dependency.get('expected_sha')}")
            print(f"    actual sha: {repo_dependency.get('actual_sha') or 'not initialized'}")
            print(f"    readiness: {repo_dependency.get('readiness')}")
        print(f"  readiness: {item['readiness']}")
    print()

    print("Accelerator compile/run probes")
    accelerator_probes = report["accelerator_compile_probes"]
    assert isinstance(accelerator_probes, dict)
    print(f"requested: {accelerator_probes['requested']}")
    print(f"probe root: {accelerator_probes['probe_root']}")
    print(f"policy: {accelerator_probes['policy']}")
    print(f"evidence class: {accelerator_probes['evidence_class']}")
    print(f"candidate evidence is correctness validation: {accelerator_probes['candidate_evidence_is_correctness_validation']}")
    probe_items = accelerator_probes["items"]
    assert isinstance(probe_items, dict)
    for name in sorted(probe_items):
        item = probe_items[name]
        assert isinstance(item, dict)
        print(f"[{item['status']}] {name}")
        print(f"  compiled: {item.get('compiled_probe_ok', False)}")
        print(f"  runtime:  {item.get('runtime_probe_ok', False)}")
        print(f"  primitive: {item.get('primitive_probe_status')} ok={item.get('primitive_probe_ok')}")
        print(f"  backend enablement: {item.get('backend_enablement')}")
        print(f"  enable flag: {item.get('enable_flag')} ({item.get('enable_policy')})")
        print(f"  evidence class: {item.get('evidence_class')}")
        print(f"  candidate evidence is correctness validation: {item.get('candidate_evidence_is_correctness_validation')}")
        print(f"  validated correctness backend: {item.get('validated_correctness_backend')}")
        print(f"  source: {item.get('source')}")
        print(f"  binary: {item.get('binary')}")
        if item.get("primitive_source"):
            print(f"  primitive source: {item.get('primitive_source')}")
        if item.get("primitive_object"):
            print(f"  primitive object: {item.get('primitive_object')}")
        if item.get("output"):
            print(f"  compile output: {item['output']}")
        if item.get("primitive_output"):
            print(f"  primitive output: {item['primitive_output']}")
        if item.get("run_output"):
            print(f"  runtime output: {item['run_output']}")
        print(f"  detail: {item['detail']}")
        print(f"  primitive detail: {item.get('primitive_detail')}")
    print()

    cmake_hip = report["cmake_hip_language"]
    assert isinstance(cmake_hip, dict)
    print(f"CMake HIP language: {'OK' if cmake_hip['ok'] else 'NOT REQUIRED'} - {cmake_hip['detail']}")
    print()

    hip_info = report["hip_info"]
    assert isinstance(hip_info, dict)
    print("HIP device")
    print(f"[{'OK' if hip_info['ok'] else 'MISSING'}] {hip_info['detail']}")
    print(f"  target: {hip_info.get('target') or 'not parsed'}")
    print(f"  spec target: {'OK' if hip_info.get('target_supported_by_spec') else 'MISSING'}")
    if hip_info.get("target_info"):
        target_info = hip_info["target_info"]
        assert isinstance(target_info, dict)
        print(f"  tier: {target_info['tier']} {target_info['family']} - {target_info['role']}")
    print()

    readiness = report["readiness"]
    assert isinstance(readiness, dict)

    print("Accelerator enablement policy")
    accelerator_enablement = readiness["accelerator_enablement"]
    assert isinstance(accelerator_enablement, dict)
    print(f"backend enablement: {accelerator_enablement['backend_enablement']}")
    print(f"correctness backends enabled: {accelerator_enablement['correctness_backends_enabled']}")
    print(f"validated correctness backend count: {accelerator_enablement['validated_correctness_backend_count']}")
    print(f"enable flags fail fast: {accelerator_enablement['enable_flags_fail_fast']}")
    print(f"evidence class: {accelerator_enablement['evidence_class']}")
    print(f"candidate evidence is correctness validation: {accelerator_enablement['candidate_evidence_is_correctness_validation']}")
    print(f"policy: {accelerator_enablement['policy']}")
    accelerator_flags = accelerator_enablement["flags"]
    assert isinstance(accelerator_flags, dict)
    for name in sorted(accelerator_flags):
        item = accelerator_flags[name]
        assert isinstance(item, dict)
        print(
            f"  {name}: {item['enable_flag']} -> {item['backend_enablement']} "
            f"({item['probe_status']}, validated={item['validated_correctness_backend']}, "
            f"primitive={item.get('primitive_probe_status')}, evidence_class={item['evidence_class']})"
        )
    print()

    print("Correctness backend validation boundary")
    correctness_validation = readiness["correctness_backend_validation"]
    assert isinstance(correctness_validation, dict)
    print(f"scope: {correctness_validation['checker_scope']}")
    print(f"validated by this report: {len(correctness_validation['validated_by_this_report'])}")
    print(f"validated correctness backend count: {correctness_validation['validated_correctness_backend_count']}")
    print(
        "candidate accelerator evidence is correctness validation: "
        f"{correctness_validation['candidate_accelerator_evidence_is_correctness_validation']}"
    )
    implemented = correctness_validation["implemented_correctness_backends"]
    assert isinstance(implemented, dict)
    for name in sorted(implemented):
        item = implemented[name]
        assert isinstance(item, dict)
        print(
            f"  {name}: {item['backend_enablement']} "
            f"(validated_by_this_report={item['validated_by_this_report']})"
        )
    print(f"policy: {correctness_validation['policy']}")
    print()

    print("Readiness gates")
    print(f"host readiness: {'OK' if readiness['host_readiness_ok'] else 'NOT READY'}")
    print(f"policy: {readiness['policy']}")
    gates = readiness["gates"]
    assert isinstance(gates, dict)
    for name in sorted(gates):
        gate = gates[name]
        assert isinstance(gate, dict)
        req = "host-required" if gate["required_for_host_readiness"] else "not host-required"
        print(f"[{gate['status']}] {name} ({req})")
        print(f"  detail: {gate['detail']}")
    platform_gates = readiness["platform_gates"]
    assert isinstance(platform_gates, dict)
    for name in sorted(platform_gates):
        gate = platform_gates[name]
        assert isinstance(gate, dict)
        req = "host-required" if gate["required_for_host_readiness"] else "not host-required"
        print(f"[{gate['status']}] {name} ({req})")
        print(f"  detail: {gate['detail']}")
    print()

    print("Exact-wide platform validation")
    exact_wide_platform = readiness["exact_wide_platform_validation"]
    assert isinstance(exact_wide_platform, dict)
    print(f"host: {exact_wide_platform['host']}")
    print(f"current host target: {exact_wide_platform['current_host_target']}")
    print(f"Windows gfx1100 evidence by this host: {exact_wide_platform['windows_gfx1100_evidence_by_this_host']}")
    print(f"Windows gfx1100 scope: {exact_wide_platform['windows_gfx1100_scope']}")
    print(f"Linux ROCm validated by this host: {exact_wide_platform['linux_rocm_validated_by_this_host']}")
    print(f"Instinct validated by this host: {exact_wide_platform['instinct_validated_by_this_host']}")
    print(f"Windows evidence validates Linux ROCm: {exact_wide_platform['windows_evidence_validates_linux_rocm']}")
    print(f"Windows evidence validates Instinct: {exact_wide_platform['windows_evidence_validates_instinct']}")
    print(f"Linux ROCm validation evidence: {exact_wide_platform['linux_rocm_validation_evidence']}")
    print(f"Instinct validation evidence: {exact_wide_platform['instinct_validation_evidence']}")
    print(f"validation boundary: {exact_wide_platform['validation_boundary']}")
    print(f"policy: {exact_wide_platform['policy']}")
    print()

    hard_cut_checks = report["hard_cut_self_checks"]
    assert isinstance(hard_cut_checks, dict)
    print("Hard-cut readiness self-checks")
    print(f"ok: {hard_cut_checks['ok']}")
    print(f"scope: {hard_cut_checks['scope']}")
    print(f"checker validation scope: {hard_cut_checks['checker_validation_scope']}")
    checks = hard_cut_checks["checks"]
    assert isinstance(checks, dict)
    for name in sorted(checks):
        item = checks[name]
        assert isinstance(item, dict)
        print(f"[{'OK' if item['ok'] else 'FAIL'}] {name}: {item['detail']}")
    print()

    print("Project tools")
    tools = report["project_tools"]
    assert isinstance(tools, dict)
    for name in sorted(tools):
        item = tools[name]
        assert isinstance(item, dict)
        print(f"[{'OK' if item['ok'] else 'MISSING'}] {name}: {item['detail']}")


