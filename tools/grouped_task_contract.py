"""Shared grouped task composition and physical descriptor admission."""

from __future__ import annotations
from typing import Any
from metadata_registry_constants import GROUPED_STRATEGY_DEVICE_DESCRIPTOR_POLICIES

SINGLE_GROUPED_DESCRIPTOR_LAYOUT = "same_shape_resident_task_triplets_v1"
BUCKETED_GROUPED_DESCRIPTOR_LAYOUT = "same_contract_bucketed_resident_task_triplets_v1"
SINGLE_GROUPED_BUCKET_POLICY = "single_same_shape_bucket"
BUCKETED_GROUPED_BUCKET_POLICY = "same_contract_shape_buckets"


def grouped_task_descriptor_contract(capture: dict[str, Any] | None) -> dict[str, Any]:
    if capture is None:
        return {}
    grouped = (
        capture.get("grouped_dispatch") if isinstance(capture.get("grouped_dispatch"), dict) else {}
    )
    contract = grouped.get("task_descriptor_contract") if isinstance(grouped, dict) else None
    return contract if isinstance(contract, dict) else {}


def expected_grouped_output_domain(capture: dict[str, Any]) -> str:
    semantics = capture.get("semantics")
    if semantics in {"finite_ring_u8", "finite_field_u8"}:
        return "finite_u8_host"
    if semantics in {"exact_wide_signed", "exact_wide_unsigned"}:
        return "exact_wide_limb_host"
    return "native_i64_u64_host"


def expected_grouped_shape_key(capture: dict[str, Any]) -> str:
    selected_prefix = capture.get("selected_prefix", capture.get("prefix"))
    return (
        f"m={capture.get('m')};n={capture.get('n')};k={capture.get('k')};"
        f"tile_m={capture.get('tile_m')};tile_n={capture.get('tile_n')};"
        f"prefix={selected_prefix}"
    )


def expected_grouped_device_descriptor_policy(capture: dict[str, Any]) -> str | None:
    grouped = (
        capture.get("grouped_dispatch") if isinstance(capture.get("grouped_dispatch"), dict) else {}
    )
    strategy = grouped.get("execution_strategy")
    return GROUPED_STRATEGY_DEVICE_DESCRIPTOR_POLICIES.get(strategy)


def grouped_task_descriptor_valid(capture: dict[str, Any]) -> bool:
    contract = grouped_task_descriptor_contract(capture)
    common_valid = (
        (capture.get("grouped_dispatch") or {}).get("requested") is True
        and contract.get("schema_version") == 1
        and type(contract.get("task_count")) is int
        and contract.get("task_count") == (capture.get("grouped_dispatch") or {}).get("task_count")
        and 2 <= contract["task_count"] <= 65536
        and contract.get("semantics") == capture.get("semantics")
        and contract.get("output_domain") == expected_grouped_output_domain(capture)
        and contract.get("source_version_policy") == "per_task_monotonic_source_version_repack"
        and contract.get("workspace_policy") == "one_workspace_per_task_shared_plan"
        and contract.get("matrix_ownership_policy")
        == "benchmark_owns_all_task_triplets_until_capture_end"
        and contract.get("descriptor_reuse_policy")
        == "reuse_after_shape_workspace_source_validation"
        and contract.get("stride_policy")
        == "matrix_ld_matches_logical_shape_host_output_ld_explicit"
        and contract.get("output_currentness_policy")
        == "device_residue_current_after_grouped_gemm_host_output_after_export"
        and contract.get("lifetime_policy")
        == "task_matrices_and_workspaces_destroyed_after_capture"
        and contract.get("checksum_policy") == "combined_per_task_checksum_u64"
        and contract.get("status_policy") == "fail_fast_per_task_operation_status"
        and expected_grouped_device_descriptor_policy(capture) is not None
        and contract.get("device_descriptor_policy")
        == expected_grouped_device_descriptor_policy(capture)
        and contract.get("promotion_eligible") is False
    )
    if not common_valid:
        return False
    if contract.get("bucket_policy") == SINGLE_GROUPED_BUCKET_POLICY:
        return (
            contract.get("descriptor_layout") == SINGLE_GROUPED_DESCRIPTOR_LAYOUT
            and contract.get("bucket_count") == 1
            and contract.get("same_shape_required") is True
            and contract.get("shape_key") == expected_grouped_shape_key(capture)
        )
    if contract.get("bucket_policy") != BUCKETED_GROUPED_BUCKET_POLICY:
        return False
    if (
        contract.get("descriptor_layout") != BUCKETED_GROUPED_DESCRIPTOR_LAYOUT
        or type(contract.get("bucket_count")) is not int
        or contract.get("bucket_count") < 2
        or contract.get("same_shape_required") is not False
    ):
        return False
    buckets = contract.get("buckets")
    if not isinstance(buckets, list) or len(buckets) != contract.get("bucket_count"):
        return False
    offset = 0
    for index, bucket in enumerate(buckets):
        if not isinstance(bucket, dict):
            return False
        bucket_task_count = bucket.get("task_count")
        if (
            bucket.get("bucket_index") != index
            or bucket.get("task_offset") != offset
            or type(bucket_task_count) is not int
            or bucket_task_count <= 1
            or not isinstance(bucket.get("shape_key"), str)
            or not bucket.get("shape_key")
            or bucket.get("semantics") != capture.get("semantics")
            or bucket.get("output_domain") != expected_grouped_output_domain(capture)
        ):
            return False
        offset += bucket_task_count
    return offset == (capture.get("grouped_dispatch") or {}).get("task_count")


def uniform_grouped_task_count(capture: dict[str, Any]) -> int | None:
    """Only the registered same-shape producer permits uniform task scaling."""
    grouped = capture.get("grouped_dispatch") or {}
    contract = grouped_task_descriptor_contract(capture)
    fields = {
        "schema_version",
        "descriptor_layout",
        "bucket_policy",
        "bucket_count",
        "task_count",
        "same_shape_required",
        "shape_key",
        "semantics",
        "output_domain",
        "source_version_policy",
        "workspace_policy",
        "matrix_ownership_policy",
        "descriptor_reuse_policy",
        "stride_policy",
        "output_currentness_policy",
        "lifetime_policy",
        "checksum_policy",
        "status_policy",
        "device_descriptor_policy",
        "promotion_eligible",
        "buckets",
    }
    if (
        not grouped_task_descriptor_valid(capture)
        or contract.get("bucket_policy") != SINGLE_GROUPED_BUCKET_POLICY
        or set(contract) - fields
        or contract.get("buckets") not in (None, [])
        or grouped.get("source_hash") != str(capture.get("seed"))
        or grouped.get("output_hash") != "final_checksum_u64"
        or grouped.get("descriptor_identity")
        != (
            f"same_shape_m={capture.get('m')};n={capture.get('n')};k={capture.get('k')};"
            f"semantics={capture.get('semantics')}"
        )
    ):
        return None
    return contract["task_count"]
