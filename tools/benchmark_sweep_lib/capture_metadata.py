from __future__ import annotations

import argparse
from typing import Any

from .config import PHASES, PLACEHOLDER_TARGET_IDS, WRAP64_ROCWMMA_CANDIDATE_BACKEND

def capture_contract_key(capture: dict[str, Any]) -> str:
    tile_bounds = capture.get("tile_bounds_u64")
    tile_hash = tile_bounds.get("hash_u64") if isinstance(tile_bounds, dict) else None
    wrap64_contract = capture.get("semantics") == "wrap_u64_mod_2_64"
    tile_m = "wrap64_semantic_contract" if wrap64_contract else capture.get("tile_m")
    tile_n = "wrap64_semantic_contract" if wrap64_contract else capture.get("tile_n")
    bound_source = capture_bound_source(capture)
    timing_metadata = capture.get("timing_metadata")
    output_policy = capture.get("output_policy")
    requested_next_op = capture.get("requested_next_op")
    export_variant = capture_export_variant_name(capture)
    reconstruction_variant = capture_reconstruction_variant_name(capture)
    residue_output_mode = capture.get("residue_output_mode", "host_export")
    if residue_output_mode == "host_export":
        next_op_contract = "host_export"
    else:
        next_op_contract = requested_next_op.get("resolved") if isinstance(requested_next_op, dict) else None
    parts = [
        f"semantics={capture.get('semantics')}",
        f"finite_modulus={capture.get('finite_modulus')}",
        f"bound_kind={capture.get('bound_kind')}",
        f"bound_mode={capture.get('bound_mode')}",
        f"bound={capture.get('bound')}",
        f"bound_source={bound_source}",
        f"m={capture.get('m')}",
        f"n={capture.get('n')}",
        f"k={capture.get('k')}",
        f"output_logical_ld={capture.get('output_logical_ld', capture.get('n'))}",
        f"output_ld_padding={capture.get('output_ld_padding', 0)}",
        f"prefix={capture.get('prefix')}",
        f"layout={capture.get('layout')}",
        f"tile_m={tile_m}",
        f"tile_n={tile_n}",
        f"k_block={capture.get('k_block_size')}",
        f"exact_wide_limb_count={capture.get('exact_wide_limb_count')}",
        f"residue_chain_length={capture.get('residue_chain_length', 1)}",
        f"residue_output_mode={residue_output_mode}",
        f"seed={capture.get('seed')}",
        f"input_distribution={capture.get('input_distribution')}",
        f"reuse_packed_inputs={capture.get('reuse_packed_inputs') is True}",
        f"pack_mode={capture_pack_mode(capture)}",
        f"next_op_contract={next_op_contract}",
        f"output_policy={output_policy.get('destination_layout') if isinstance(output_policy, dict) else None}",
        f"status_handling={output_policy.get('status_handling') if isinstance(output_policy, dict) else None}",
        f"export_variant={export_variant}",
        f"reconstruction_variant={reconstruction_variant}",
        f"fusion_mode={timing_metadata.get('fusion_mode') if isinstance(timing_metadata, dict) else None}",
        f"residue_group_width={timing_metadata.get('residue_group_width') if isinstance(timing_metadata, dict) else None}",
        f"tile_hash={tile_hash}",
    ]
    return ";".join(str(part) for part in parts)


def capture_export_variant_name(capture: dict[str, Any]) -> str:
    variant = capture.get("export_variant")
    if isinstance(variant, dict) and isinstance(variant.get("name"), str) and variant["name"]:
        return variant["name"]
    value = capture.get("export_variant_name")
    return str(value) if isinstance(value, str) and value else "default"


def capture_reconstruction_variant_name(capture: dict[str, Any]) -> str:
    variant = capture.get("reconstruction_variant")
    if isinstance(variant, dict) and isinstance(variant.get("name"), str) and variant["name"]:
        return variant["name"]
    value = capture.get("reconstruction_variant_name")
    return str(value) if isinstance(value, str) and value else "default_garner"


def median_phase(capture: dict[str, Any], phase: str) -> float | None:
    summary = capture.get("timing_summary_us")
    if not isinstance(summary, dict):
        return None
    item = summary.get(phase)
    if not isinstance(item, dict):
        return None
    value = item.get("median")
    return float(value) if isinstance(value, (int, float)) else None


def backend_id(capture: dict[str, Any]) -> str:
    if capture.get("backend_requested") == WRAP64_ROCWMMA_CANDIDATE_BACKEND:
        return WRAP64_ROCWMMA_CANDIDATE_BACKEND
    backend = str(capture.get("backend_selected"))
    execution_mode = capture_execution_mode(capture)
    if execution_mode == "public_oneshot_transient_native_inputs":
        return f"{backend}-oneshot"
    if execution_mode == "benchmark_host_api_batch":
        return f"{backend}-hostbatch"
    if execution_mode == "hip_graph_replay_resident_rns_chain":
        return f"{backend}-hipgraph"
    if execution_mode == "hip_graph_replay_bounded_pack_gemm_export":
        return f"{backend}-hipgraph-pack-export"
    if execution_mode == "hip_graph_replay_finite_u8_pack_gemm_export":
        return f"{backend}-hipgraph-finite-pack-export"
    if execution_mode == "hip_graph_replay_wrap64_pack_gemm_export":
        return f"{backend}-hipgraph-wrap64-pack-export"
    return backend


def selected_kernel(capture: dict[str, Any]) -> str:
    kernel = capture.get("selected_kernel")
    return str(kernel) if kernel is not None else ""


def normalized_target_id(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in PLACEHOLDER_TARGET_IDS:
        return None
    return text


def backend_family_id(backend: str) -> str:
    for suffix in ("-oneshot", "-hostbatch", "-hipgraph", "-hipgraph-pack-export"):
        if backend.endswith(suffix):
            return backend[: -len(suffix)]
    return backend


def backend_requires_gpu_target(backend: str) -> bool:
    return backend_family_id(backend) not in {"cpu-reference", "wrap64-byte-limb"}


def normalized_positive_int(value: Any) -> str | None:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return str(value)
    return None


def normalized_identity_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in PLACEHOLDER_TARGET_IDS:
        return None
    return text


def normalized_compiler_identity(capture: dict[str, Any]) -> str | None:
    compiler = capture_compiler(capture)
    compiler_id = normalized_identity_text(compiler.get("id"))
    compiler_version = normalized_identity_text(compiler.get("version"))
    if compiler_id is None or compiler_version is None:
        return None
    return f"{compiler_id} {compiler_version}"


def capture_backend_metadata(capture: dict[str, Any]) -> dict[str, Any]:
    metadata = capture.get("backend_metadata")
    return metadata if isinstance(metadata, dict) else {}


def capture_timing_metadata(capture: dict[str, Any]) -> dict[str, Any]:
    metadata = capture.get("timing_metadata")
    return metadata if isinstance(metadata, dict) else {}


def capture_execution_mode(capture: dict[str, Any]) -> str:
    mode = capture.get("benchmark_execution_mode")
    if not isinstance(mode, str):
        mode = capture_timing_metadata(capture).get("benchmark_execution_mode")
    if isinstance(mode, str):
        return mode
    if capture.get("backend_requested") == WRAP64_ROCWMMA_CANDIDATE_BACKEND:
        return "internal_wrap64_rocwmma_candidate"
    if capture.get("benchmark") in {"rns8_bounded_gemm_public_oneshot", "rns8_finite_u8_public_oneshot"}:
        return "public_oneshot_transient_native_inputs"
    return "persistent_resident_matrices"


def capture_pack_mode(capture: dict[str, Any]) -> str:
    timing = capture_timing_metadata(capture)
    mode = timing.get("pack_mode")
    if mode is None:
        mode = capture.get("pack_mode")
    if isinstance(mode, str):
        return mode
    return "prepacked_reuse" if capture.get("reuse_packed_inputs") is True else "per_repeat_repack"


def capture_prepack_reuse_strategy(capture: dict[str, Any]) -> str:
    timing = capture_timing_metadata(capture)
    strategy = timing.get("prepack_reuse_strategy")
    if strategy is None:
        strategy = capture.get("prepack_reuse_strategy")
    if isinstance(strategy, str):
        return strategy
    return "persistent_matrix_residency" if capture.get("reuse_packed_inputs") is True else "none"


def capture_bound_source(capture: dict[str, Any]) -> str:
    value = capture.get("bound_source")
    return value if isinstance(value, str) else "static_profile"


def requested_pack_mode(args: argparse.Namespace) -> str:
    if getattr(args, "reuse_packed_inputs", False):
        return "prepacked_reuse"
    reuse_a = bool(getattr(args, "reuse_packed_a", False))
    reuse_b = bool(getattr(args, "reuse_packed_b", False))
    if reuse_a and reuse_b:
        return "prepacked_reuse"
    if reuse_a:
        return "prepacked_reuse_a"
    if reuse_b:
        return "prepacked_reuse_b"
    return "per_repeat_repack"


def capture_prepack_reuse_operands(capture: dict[str, Any]) -> tuple[str, ...]:
    operands = capture.get("prepack_reuse_operands")
    if not isinstance(operands, list):
        timing = capture_timing_metadata(capture)
        operands = timing.get("prepack_reuse_operands")
    if isinstance(operands, list):
        return tuple(str(item) for item in operands)
    mode = capture_pack_mode(capture)
    if mode == "prepacked_reuse":
        return ("A", "B")
    if mode == "prepacked_reuse_a":
        return ("A",)
    if mode == "prepacked_reuse_b":
        return ("B",)
    return ()


def capture_device(capture: dict[str, Any]) -> dict[str, Any]:
    device = capture.get("device")
    return device if isinstance(device, dict) else {}


def capture_hip_toolchain(capture: dict[str, Any]) -> dict[str, Any]:
    toolchain = capture.get("hip_toolchain")
    return toolchain if isinstance(toolchain, dict) else {}


def capture_compiler(capture: dict[str, Any]) -> dict[str, Any]:
    compiler = capture.get("compiler")
    return compiler if isinstance(compiler, dict) else {}


def candidate_source_metadata(capture: dict[str, Any]) -> dict[str, Any]:
    metadata = capture_backend_metadata(capture)
    timing = capture_timing_metadata(capture)
    device = capture_device(capture)
    toolchain = capture_hip_toolchain(capture)
    compiler = capture_compiler(capture)
    return {
        "target_id": normalized_target_id(device.get("gcn_arch")),
        "device_name": device.get("name"),
        "hip_runtime_version": device.get("hip_runtime_version"),
        "hip_driver_version": device.get("hip_driver_version"),
        "hip_sdk_or_rocm_version": toolchain.get("hip_sdk_or_rocm_version"),
        "accelerator_library": metadata.get("accelerator_library"),
        "accelerator_version": metadata.get("accelerator_version"),
        "compiler": {"id": compiler.get("id"), "version": compiler.get("version")},
        "configured_amdgpu_targets": capture.get("configured_amdgpu_targets"),
        "git_commit": capture.get("git_commit"),
        "benchmark": capture.get("benchmark"),
        "benchmark_execution_mode": capture_execution_mode(capture),
        "seed": capture.get("seed"),
        "warmups": capture.get("warmups"),
        "repeats": capture.get("repeats"),
        "layout": capture.get("layout"),
        "output_logical_ld": capture.get("output_logical_ld", capture.get("n")),
        "output_ld_padding": capture.get("output_ld_padding", 0),
        "reuse_packed_inputs": capture.get("reuse_packed_inputs") is True,
        "pack_mode": capture_pack_mode(capture),
        "prepack_reuse_strategy": capture_prepack_reuse_strategy(capture),
        "prepack_setup_us": capture.get("prepack_setup_us"),
        "bound_source": capture_bound_source(capture),
        "bound_discovery": capture.get("bound_discovery"),
        "prefix": capture.get("prefix"),
        "selected_prefix": capture.get("selected_prefix"),
        "requested_max_prefix": capture.get("requested_max_prefix"),
        "contract_prefix_policy": capture.get("contract_prefix_policy"),
        "residue_plane_skip_fraction": capture.get("residue_plane_skip_fraction"),
        "k_block_size": capture.get("k_block_size"),
        "tile_m": capture.get("tile_m"),
        "tile_n": capture.get("tile_n"),
        "epilogue": capture.get("epilogue_type"),
        "workspace_bytes": metadata.get("workspace_required_bytes", 0),
        "autotune_key": metadata.get("autotune_key"),
        "event_source": timing.get("gpu_event_timing_source"),
        "event_source_scope": timing.get("gpu_event_timing_source_scope"),
        "event_status": timing.get("gpu_event_timing_status"),
        "event_phase_order": timing.get("gpu_event_phase_order"),
    }


def group_source_metadata(items: list[dict[str, Any]]) -> dict[str, Any]:
    source_items = [candidate_source_metadata(item) for item in items]
    targets = sorted(
        {
            str(item.get("target_id"))
            for item in source_items
            if item.get("target_id")
        }
    )
    hip_versions = sorted(
        {
            str(item.get("hip_sdk_or_rocm_version"))
            for item in source_items
            if item.get("hip_sdk_or_rocm_version")
        }
    )
    accelerator_versions = sorted(
        {
            str(item.get("accelerator_version"))
            for item in source_items
            if item.get("accelerator_version")
        }
    )
    configured_targets = sorted(
        {
            str(item.get("configured_amdgpu_targets"))
            for item in source_items
            if item.get("configured_amdgpu_targets")
        }
    )
    runtime_versions = sorted(
        {
            str(item.get("hip_runtime_version"))
            for item in source_items
            if item.get("hip_runtime_version")
        }
    )
    driver_versions = sorted(
        {
            str(item.get("hip_driver_version"))
            for item in source_items
            if item.get("hip_driver_version")
        }
    )
    compilers = sorted(
        {
            f"{compiler.get('id')} {compiler.get('version')}"
            for item in items
            if (compiler := capture_compiler(item)).get("id") or compiler.get("version")
        }
    )
    return {
        "target_ids": targets,
        "hip_sdk_or_rocm_versions": hip_versions,
        "accelerator_versions": accelerator_versions,
        "configured_amdgpu_targets": configured_targets,
        "hip_runtime_versions": runtime_versions,
        "hip_driver_versions": driver_versions,
        "compilers": compilers,
        "git_commits": sorted({str(item.get("git_commit")) for item in items if item.get("git_commit")}),
        "seeds": sorted({int(item.get("seed")) for item in items if isinstance(item.get("seed"), int)}),
        "warmups": sorted({int(item.get("warmups")) for item in items if isinstance(item.get("warmups"), int)}),
        "repeats": sorted({int(item.get("repeats")) for item in items if isinstance(item.get("repeats"), int)}),
        "reuse_packed_inputs": sorted({bool(item.get("reuse_packed_inputs") is True) for item in items}),
        "benchmark_execution_modes": sorted({capture_execution_mode(item) for item in items}),
        "pack_modes": sorted({capture_pack_mode(item) for item in items}),
        "prepack_reuse_strategies": sorted({capture_prepack_reuse_strategy(item) for item in items}),
        "prepack_reuse_operands": sorted({"/".join(capture_prepack_reuse_operands(item)) or "none" for item in items}),
        "event_sources": sorted(
            {
                str(capture_timing_metadata(item).get("gpu_event_timing_source") or "unavailable")
                for item in items
            }
        ),
    }


