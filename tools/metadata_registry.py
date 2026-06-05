#!/usr/bin/env python3
"""Validate and generate constants from the RNS8 metadata registry.

The registry files use a YAML-compatible JSON subset so this tool can run with
the Python standard library on clean Windows developer machines.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
METADATA_DIR = REPO_ROOT / "metadata"
PYTHON_CONSTANTS_PATH = REPO_ROOT / "tools" / "metadata_registry_constants.py"
CPP_HEADER_PATH = REPO_ROOT / "include" / "rns8" / "generated" / "metadata_registry.hpp"
REGISTRY_FILES = {
    "benchmark_modes": "benchmark_modes.yaml",
    "output_policies": "output_policies.yaml",
    "event_phases": "event_phases.yaml",
    "epilogues": "epilogues.yaml",
    "workspace_modes": "workspace_modes.yaml",
    "kernels": "kernels.yaml",
    "claim_labels": "claim_labels.yaml",
}


class MetadataRegistryError(RuntimeError):
    pass


@dataclass(frozen=True)
class Registry:
    files: dict[str, dict[str, Any]]

    def benchmark_modes(self) -> dict[str, Any]:
        return self.files["benchmark_modes"]

    def output_policies(self) -> dict[str, Any]:
        return self.files["output_policies"]


def _load_json_yaml(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MetadataRegistryError(f"{path}: metadata files must use the JSON-compatible YAML subset: {exc}") from exc
    if not isinstance(data, dict):
        raise MetadataRegistryError(f"{path}: top-level registry document must be an object")
    return data


def load_registry(metadata_dir: Path = METADATA_DIR) -> Registry:
    files: dict[str, dict[str, Any]] = {}
    for key, filename in REGISTRY_FILES.items():
        path = metadata_dir / filename
        if not path.exists():
            raise MetadataRegistryError(f"missing metadata registry file: {path}")
        files[key] = _load_json_yaml(path)
    return Registry(files)


def _require_list(document: dict[str, Any], key: str, path: str) -> list[Any]:
    value = document.get(key)
    if not isinstance(value, list):
        raise MetadataRegistryError(f"{path}.{key} must be a list")
    return value


def _require_string_list(document: dict[str, Any], key: str, path: str) -> list[str]:
    values = _require_list(document, key, path)
    if not all(isinstance(item, str) and item for item in values):
        raise MetadataRegistryError(f"{path}.{key} must contain only nonempty strings")
    _require_unique(values, f"{path}.{key}")
    return list(values)


def _require_unique(values: list[str], path: str) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    if duplicates:
        raise MetadataRegistryError(f"{path} contains duplicate ids: {sorted(duplicates)}")


def _strategy_ids(registry: Registry) -> list[str]:
    strategies = _require_list(
        registry.benchmark_modes(),
        "grouped_dispatch_execution_strategies",
        "benchmark_modes",
    )
    ids: list[str] = []
    for index, item in enumerate(strategies):
        if not isinstance(item, dict):
            raise MetadataRegistryError(f"benchmark_modes.grouped_dispatch_execution_strategies[{index}] must be an object")
        value = item.get("id")
        if not isinstance(value, str) or not value:
            raise MetadataRegistryError(f"benchmark_modes.grouped_dispatch_execution_strategies[{index}].id must be a string")
        if not isinstance(item.get("batched_exact_wide_export"), bool):
            raise MetadataRegistryError(
                f"benchmark_modes.grouped_dispatch_execution_strategies[{index}].batched_exact_wide_export must be boolean"
            )
        policy = item.get("device_descriptor_policy")
        if not isinstance(policy, str) or not policy:
            raise MetadataRegistryError(
                f"benchmark_modes.grouped_dispatch_execution_strategies[{index}].device_descriptor_policy must be a string"
            )
        ids.append(value)
    _require_unique(ids, "benchmark_modes.grouped_dispatch_execution_strategies")
    return ids


def _strategy_policy_map(registry: Registry) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in registry.benchmark_modes()["grouped_dispatch_execution_strategies"]:
        result[item["id"]] = item["device_descriptor_policy"]
    return result


def _batched_exact_wide_strategies(registry: Registry) -> list[str]:
    return [
        item["id"]
        for item in registry.benchmark_modes()["grouped_dispatch_execution_strategies"]
        if item["batched_exact_wide_export"]
    ]


def validate_registry(registry: Registry) -> None:
    for key, document in registry.files.items():
        version = document.get("schema_version")
        if version != 1:
            raise MetadataRegistryError(f"{key}.schema_version must be 1")

    benchmark = registry.benchmark_modes()
    _require_string_list(benchmark, "benchmark_execution_modes", "benchmark_modes")
    _require_string_list(benchmark, "grouped_dispatch_statuses", "benchmark_modes")
    strategy_ids = _strategy_ids(registry)

    descriptor = benchmark.get("grouped_task_descriptor")
    if not isinstance(descriptor, dict):
        raise MetadataRegistryError("benchmark_modes.grouped_task_descriptor must be an object")
    for key in [
        "descriptor_layouts",
        "bucket_policies",
        "source_version_policies",
        "workspace_policies",
        "checksum_policies",
        "status_policies",
        "device_descriptor_policies",
    ]:
        _require_string_list(descriptor, key, "benchmark_modes.grouped_task_descriptor")

    descriptor_policies = set(descriptor["device_descriptor_policies"])
    for strategy, policy in _strategy_policy_map(registry).items():
        if policy not in descriptor_policies:
            raise MetadataRegistryError(f"grouped strategy {strategy} references unknown descriptor policy {policy}")
    if "not_requested" not in strategy_ids:
        raise MetadataRegistryError("grouped strategies must include not_requested")

    output = registry.output_policies()
    _require_string_list(output, "destination_layouts", "output_policies")
    _require_string_list(output, "status_handling", "output_policies")
    _require_string_list(output, "output_contract_domains", "output_policies")
    _require_string_list(output, "direct_hip_export_staging_policies", "output_policies")

    _require_string_list(registry.files["event_phases"], "timing_phases", "event_phases")
    _require_string_list(registry.files["epilogues"], "epilogue_types", "epilogues")
    _require_string_list(registry.files["epilogues"], "backend_epilogue_modes", "epilogues")
    _require_string_list(registry.files["workspace_modes"], "workspace_modes", "workspace_modes")
    _require_string_list(registry.files["kernels"], "selected_kernels", "kernels")
    _require_string_list(registry.files["claim_labels"], "comparison_baseline_statuses", "claim_labels")
    _require_string_list(registry.files["claim_labels"], "release_gate_review_statuses", "claim_labels")


def _python_repr_set(values: list[str]) -> str:
    inner = ",\n".join(f"    {value!r}" for value in sorted(values))
    return "{\n" + inner + (",\n" if inner else "") + "}"


def _python_repr_dict(values: dict[str, str]) -> str:
    inner = ",\n".join(f"    {key!r}: {value!r}" for key, value in sorted(values.items()))
    return "{\n" + inner + (",\n" if inner else "") + "}"


def _constant_suffix(value: str) -> str:
    chars = [char.upper() if char.isalnum() else "_" for char in value]
    suffix = "".join(chars)
    while "__" in suffix:
        suffix = suffix.replace("__", "_")
    return suffix.strip("_")


def render_python_constants(registry: Registry) -> str:
    benchmark = registry.benchmark_modes()
    descriptor = benchmark["grouped_task_descriptor"]
    output = registry.output_policies()
    strategies = _strategy_ids(registry)
    generated = [
        "# Generated by tools/metadata_registry.py. Do not edit by hand.",
        "from __future__ import annotations",
        "",
        f"BENCHMARK_EXECUTION_MODES = {_python_repr_set(benchmark['benchmark_execution_modes'])}",
        f"GROUPED_DISPATCH_STATUSES = {_python_repr_set(benchmark['grouped_dispatch_statuses'])}",
        f"GROUPED_DISPATCH_EXECUTION_STRATEGIES = {_python_repr_set(strategies)}",
        f"GROUPED_DISPATCH_BATCHED_EXACT_WIDE_EXPORT_STRATEGIES = {_python_repr_set(_batched_exact_wide_strategies(registry))}",
        f"GROUPED_TASK_DESCRIPTOR_LAYOUTS = {_python_repr_set(descriptor['descriptor_layouts'])}",
        f"GROUPED_TASK_BUCKET_POLICIES = {_python_repr_set(descriptor['bucket_policies'])}",
        f"GROUPED_TASK_SOURCE_VERSION_POLICIES = {_python_repr_set(descriptor['source_version_policies'])}",
        f"GROUPED_TASK_WORKSPACE_POLICIES = {_python_repr_set(descriptor['workspace_policies'])}",
        f"GROUPED_TASK_CHECKSUM_POLICIES = {_python_repr_set(descriptor['checksum_policies'])}",
        f"GROUPED_TASK_STATUS_POLICIES = {_python_repr_set(descriptor['status_policies'])}",
        f"GROUPED_TASK_DEVICE_DESCRIPTOR_POLICIES = {_python_repr_set(descriptor['device_descriptor_policies'])}",
        f"GROUPED_STRATEGY_DEVICE_DESCRIPTOR_POLICIES = {_python_repr_dict(_strategy_policy_map(registry))}",
        f"OUTPUT_DESTINATION_LAYOUTS = {_python_repr_set(output['destination_layouts'])}",
        f"STATUS_HANDLING = {_python_repr_set(output['status_handling'])}",
        f"OUTPUT_CONTRACT_DOMAINS = {_python_repr_set(output['output_contract_domains'])}",
        f"DIRECT_HIP_EXPORT_STAGING_POLICIES = {_python_repr_set(output['direct_hip_export_staging_policies'])}",
        "",
    ]
    strategy_constants = [
        f"GROUPED_DISPATCH_STRATEGY_{_constant_suffix(strategy)} = {strategy!r}" for strategy in sorted(strategies)
    ]
    if strategy_constants:
        generated.extend(strategy_constants)
        generated.append("")
    return "\n".join(generated)


def _cpp_array(name: str, values: list[str]) -> str:
    items = ",\n".join(f'    "{value}"' for value in sorted(values))
    return (
        f"inline constexpr std::array<std::string_view, {len(values)}> {name}{{{{\n"
        f"{items}\n"
        "}};\n"
    )


def render_cpp_header(registry: Registry) -> str:
    benchmark = registry.benchmark_modes()
    descriptor = benchmark["grouped_task_descriptor"]
    output = registry.output_policies()
    return "\n".join(
        [
            "// Generated by tools/metadata_registry.py. Do not edit by hand.",
            "#pragma once",
            "",
            "#include <array>",
            "#include <string_view>",
            "",
            "namespace rns8::generated_metadata {",
            "",
            _cpp_array("benchmark_execution_modes", benchmark["benchmark_execution_modes"]),
            _cpp_array("grouped_dispatch_statuses", benchmark["grouped_dispatch_statuses"]),
            _cpp_array("grouped_dispatch_execution_strategies", _strategy_ids(registry)),
            _cpp_array("grouped_task_device_descriptor_policies", descriptor["device_descriptor_policies"]),
            _cpp_array("output_destination_layouts", output["destination_layouts"]),
            _cpp_array("status_handling_modes", output["status_handling"]),
            "",
            "}  // namespace rns8::generated_metadata",
            "",
        ]
    )


def write_if_changed(path: Path, content: str) -> bool:
    old = path.read_text(encoding="utf-8") if path.exists() else None
    if old == content:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")
    return True


def check_generated(path: Path, expected: str) -> list[str]:
    if not path.exists():
        return [f"missing generated file: {path}"]
    actual = path.read_text(encoding="utf-8")
    if actual != expected:
        return [f"stale generated file: {path}; run python tools\\metadata_registry.py --write-generated"]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-generated", action="store_true", help="rewrite generated Python/C++ constants")
    parser.add_argument("--check", action="store_true", help="validate registry and check generated outputs")
    args = parser.parse_args()

    registry = load_registry()
    validate_registry(registry)
    python_constants = render_python_constants(registry)
    cpp_header = render_cpp_header(registry)

    if args.write_generated:
        changed = [
            write_if_changed(PYTHON_CONSTANTS_PATH, python_constants),
            write_if_changed(CPP_HEADER_PATH, cpp_header),
        ]
        print("metadata registry generated files updated" if any(changed) else "metadata registry generated files unchanged")
        return 0

    errors = check_generated(PYTHON_CONSTANTS_PATH, python_constants)
    errors.extend(check_generated(CPP_HEADER_PATH, cpp_header))
    if errors:
        for error in errors:
            print(error)
        return 1
    print("metadata registry: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
