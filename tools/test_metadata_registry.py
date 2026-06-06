#!/usr/bin/env python3
"""Self-test for the checked-in metadata registry."""

from __future__ import annotations

import copy

import metadata_registry


def main() -> int:
    registry = metadata_registry.load_registry()
    metadata_registry.validate_registry(registry)
    errors = metadata_registry.check_generated(
        metadata_registry.PYTHON_CONSTANTS_PATH,
        metadata_registry.render_python_constants(registry),
    )
    errors.extend(
        metadata_registry.check_generated(
            metadata_registry.CPP_HEADER_PATH,
            metadata_registry.render_cpp_header(registry),
        )
    )
    if errors:
        raise SystemExit("\n".join(errors))

    stale_kernel_group = metadata_registry.Registry(copy.deepcopy(registry.files))
    stale_kernel_group.files["kernels"]["selected_kernel_groups"]["vector_alu"].append(
        "hip_vector_alu_unregistered_kernel_v9"
    )
    try:
        metadata_registry.validate_registry(stale_kernel_group)
    except metadata_registry.MetadataRegistryError as exc:
        if "references unknown kernels" not in str(exc):
            raise
    else:
        raise SystemExit("expected stale kernel group metadata to fail validation")

    stale_grouped_strategy = metadata_registry.Registry(copy.deepcopy(registry.files))
    stale_grouped_strategy.files["benchmark_modes"]["grouped_dispatch_execution_strategies"].append(
        {
            "id": "device_grouped_unregistered_descriptor_policy",
            "batched_exact_wide_export": False,
            "batched_bounded_export": False,
            "batched_finite_export": False,
            "device_descriptor_policy": "unregistered_device_policy",
        }
    )
    try:
        metadata_registry.validate_registry(stale_grouped_strategy)
    except metadata_registry.MetadataRegistryError as exc:
        if "references unknown descriptor policy" not in str(exc):
            raise
    else:
        raise SystemExit("expected stale grouped strategy metadata to fail validation")

    print("metadata registry self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
