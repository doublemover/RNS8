from __future__ import annotations

import argparse
from typing import Any

from .capture_metadata import requested_pack_mode
from .config import (
    ADAPTIVE_WORKLOAD_CASES,
    BOUNDED_BACKENDS,
    BOUNDED_SEMANTICS,
    DEFAULT_ADAPTIVE_CASES,
    DEFAULT_EXACT_WIDE_LIMB_COUNT,
    DEFAULT_FINITE_FIELD_MODULI,
    DEFAULT_FINITE_RING_MODULI,
    EXACT_WIDE_BACKENDS,
    EXACT_WIDE_LIMB_VARIANTS,
    EXACT_WIDE_SEMANTICS,
    EXPLORATORY_RELEASE_SHAPES,
    FINITE_BACKENDS,
    HOST_API_BATCH_BACKENDS,
    PROMOTABLE_RELEASE_SHAPES,
    PUBLIC_ONESHOT_BACKENDS,
    RNS_CHAIN_SEMANTICS,
    SCENARIO_DATA_DIR,
    SweepCase,
    SweepCommand,
    WRAP64_BACKENDS,
    WRAP64_ROCWMMA_CANDIDATE_BACKEND,
)
from .execution import cli_backend, normalize_semantics, parse_backend_bench
from .parsing import parse_case
from .scenarios import load_scenario_data_catalog

def default_cases(args: argparse.Namespace) -> list[SweepCase]:
    if args.case:
        return [parse_case(value) for value in args.case]
    if args.shapes:
        return [parse_case(f"{shape},{shape},{shape}") for shape in args.shapes]
    shapes = PROMOTABLE_RELEASE_SHAPES if args.release_matrix else [64, 128]
    if args.include_exploratory_large:
        shapes = [*shapes, *EXPLORATORY_RELEASE_SHAPES]
    return [parse_case(f"{shape},{shape},{shape}", promotable=shape in PROMOTABLE_RELEASE_SHAPES) for shape in shapes]


def adaptive_cases(args: argparse.Namespace) -> list[SweepCase]:
    cases: list[SweepCase] = []
    if args.adaptive_case:
        cases.extend(parse_case(value, adaptive=True) for value in args.adaptive_case)
    if args.include_default_adaptive:
        cases.extend(parse_case(value, adaptive=True) for value in DEFAULT_ADAPTIVE_CASES)
    if args.include_adaptive_workloads:
        cases.extend(parse_case(value, adaptive=True) for value in ADAPTIVE_WORKLOAD_CASES)
    return cases


def wrap64_cases(args: argparse.Namespace) -> list[SweepCase]:
    if args.case:
        return [parse_case(value) for value in args.case]
    shapes = PROMOTABLE_RELEASE_SHAPES if args.release_matrix else [64]
    if args.include_exploratory_large:
        shapes = [*shapes, *EXPLORATORY_RELEASE_SHAPES]
    return [
        parse_case(f"wrap64-{shape}:{shape},{shape},{shape}", promotable=shape in PROMOTABLE_RELEASE_SHAPES)
        for shape in shapes
    ]


def wrap64_backends_for(args: argparse.Namespace) -> list[str]:
    backends = list(WRAP64_BACKENDS)
    if args.include_wrap64_rocwmma_candidate:
        backends.append(WRAP64_ROCWMMA_CANDIDATE_BACKEND)
    return backends


def finite_moduli_for(semantics: str, args: argparse.Namespace) -> list[int | None]:
    if semantics == "finite-u8-ring":
        return args.modulus or DEFAULT_FINITE_RING_MODULI
    if semantics == "finite-u8-field":
        return args.modulus or DEFAULT_FINITE_FIELD_MODULI
    return [None]


def exact_wide_limb_counts_for(semantics: str, args: argparse.Namespace) -> list[int | None]:
    if semantics not in EXACT_WIDE_SEMANTICS:
        return [None]
    if args.exact_wide_limbs:
        counts = args.exact_wide_limbs
    elif args.include_exact_wide_limb_variants:
        counts = EXACT_WIDE_LIMB_VARIANTS
    else:
        counts = [DEFAULT_EXACT_WIDE_LIMB_COUNT]
    invalid = [count for count in counts if count < 1 or count > 32]
    if invalid:
        raise SystemExit(f"--exact-wide-limbs values must be in [1, 32], got {invalid}")
    return list(dict.fromkeys(counts))


def scenario_catalog() -> dict[str, list[ScenarioItem]]:
    return load_scenario_data_catalog(SCENARIO_DATA_DIR)

def scenario_names() -> list[str]:
    return sorted(scenario_catalog())


def selected_scenario_items(args: argparse.Namespace) -> list[ScenarioItem]:
    requested = list(dict.fromkeys(getattr(args, "scenario", []) or []))
    if not requested:
        return []
    catalog = scenario_catalog()
    unknown = [name for name in requested if name != "all" and name not in catalog]
    if unknown:
        raise SystemExit(f"--scenario must be one of {scenario_names() + ['all']}, got {unknown}")
    names = scenario_names() if "all" in requested else requested
    items: list[ScenarioItem] = []
    for name in names:
        items.extend(catalog[name])
    return items


def scenario_args_for_item(args: argparse.Namespace, item: ScenarioItem) -> argparse.Namespace:
    scenario_args = argparse.Namespace(**vars(args))
    scenario_args.reuse_packed_inputs = item.pack_mode == "prepacked_reuse"
    scenario_args.reuse_packed_a = item.pack_mode == "prepacked_reuse_a"
    scenario_args.reuse_packed_b = item.pack_mode == "prepacked_reuse_b"
    scenario_args.residue_chain_length = item.residue_chain_length
    scenario_args.residue_chain_final_export = item.residue_chain_final_export
    scenario_args.residue_chain_independent_final_export = item.residue_chain_independent_final_export
    scenario_args.output_ld_padding = item.output_ld_padding
    scenario_args.host_api_batch_size = item.host_api_batch_size
    scenario_args.native_to_rns_bridge = item.native_to_rns_bridge
    scenario_args.vector_to_rns_chain = item.vector_to_rns_chain
    scenario_args.prefix_policy = item.prefix_policy or getattr(args, "prefix_policy", None)
    scenario_args.max_prefix = item.max_prefix if item.max_prefix is not None else getattr(args, "max_prefix", None)
    scenario_args.bound_source = item.bound_source or getattr(args, "bound_source", None)
    scenario_args.next_op_hint = item.next_op_hint or getattr(args, "next_op_hint", None)
    scenario_args.residue_channel_fusion = item.residue_channel_fusion
    scenario_args.modulus_set = item.modulus_set or getattr(args, "modulus_set", "default")
    scenario_args.tile_shape_variant = item.tile_shape_variant or getattr(args, "tile_shape_variant", "default")
    scenario_args.export_variant = item.export_variant or getattr(args, "export_variant", "default")
    scenario_args.reconstruction_variant = item.reconstruction_variant or getattr(args, "reconstruction_variant", "default_garner")
    scenario_args.grouped_dispatch_tasks = item.grouped_dispatch_tasks
    scenario_args.hip_graph_replay = item.hip_graph_replay
    scenario_args.workload_proxy = item.workload_proxy or getattr(args, "workload_proxy", "none")
    scenario_args.resident_lifetime = item.resident_lifetime or getattr(args, "resident_lifetime", False)
    scenario_args.workspace_arena = item.workspace_arena or getattr(args, "workspace_arena", False)
    scenario_args.adaptive_grouped_scheduler = item.adaptive_grouped_scheduler or getattr(
        args, "adaptive_grouped_scheduler", False
    )
    scenario_args.streaming_overlap = item.streaming_overlap or getattr(args, "streaming_overlap", False)
    scenario_args.release_gate = (
        item.release_gate if item.release_gate and item.release_gate != "none" else getattr(args, "release_gate", "none")
    )
    scenario_args.verification_amortization = item.verification_amortization or getattr(
        args, "verification_amortization", "none"
    )
    return scenario_args


def scenario_backends_for_item(args: argparse.Namespace, item: ScenarioItem) -> list[str]:
    backends = list(item.backends or default_backends_for(item.semantics, item.case))
    if item.host_api_batch_size > 1:
        backends = [backend for backend in backends if backend in HOST_API_BATCH_BACKENDS]
    if item.hip_graph_replay:
        backends = [backend for backend in backends if backend == "hip-direct"]
    if item.semantics == "wrap-u64" and item.include_wrap64_candidate and args.include_wrap64_rocwmma_candidate:
        backends.append(WRAP64_ROCWMMA_CANDIDATE_BACKEND)
    if args.backends:
        requested = set(args.backends)
        backends = [backend for backend in backends if backend in requested]
    return list(dict.fromkeys(backends))


def scenario_metadata(
    item: ScenarioItem,
    backend: str,
    modulus: int | None,
    exact_wide_limb_count: int | None,
    scenario_args: argparse.Namespace,
    *,
    oneshot: bool,
) -> dict[str, Any]:
    metadata = {
        "family": item.family,
        "name": item.name,
        "semantics": item.semantics,
        "backend": backend,
        "modulus": modulus,
        "exact_wide_limb_count": exact_wide_limb_count,
        "shape": {"m": item.case.m, "n": item.case.n, "k": item.case.k},
        "tile": {"m": item.case.tile_m, "n": item.case.tile_n},
        "bound_mode": item.case.bound_mode,
        "input_profile": item.case.input_profile,
        "pack_mode": requested_pack_mode(scenario_args),
        "reuse_packed_inputs": requested_pack_mode(scenario_args) != "per_repeat_repack",
        "residue_chain_length": item.residue_chain_length,
        "residue_chain_final_export": item.residue_chain_final_export,
        "residue_chain_independent_final_export": item.residue_chain_independent_final_export,
        "output_ld_padding": item.output_ld_padding,
        "host_api_batch_size": item.host_api_batch_size,
        "native_to_rns_bridge": item.native_to_rns_bridge,
        "vector_to_rns_chain": item.vector_to_rns_chain,
        "next_op_hint": item.next_op_hint,
        "residue_channel_fusion": item.residue_channel_fusion,
        "modulus_set": item.modulus_set,
        "tile_shape_variant": item.tile_shape_variant,
        "export_variant": item.export_variant,
        "reconstruction_variant": item.reconstruction_variant,
        "grouped_dispatch_tasks": item.grouped_dispatch_tasks,
        "hip_graph_replay": item.hip_graph_replay,
        "workload_proxy": item.workload_proxy,
        "resident_lifetime": item.resident_lifetime,
        "workspace_arena": item.workspace_arena,
        "adaptive_grouped_scheduler": item.adaptive_grouped_scheduler,
        "streaming_overlap": item.streaming_overlap,
        "release_gate": item.release_gate,
        "verification_amortization": item.verification_amortization,
        "oneshot": oneshot,
        "evidence_scope": item.evidence_scope,
        "output_domain": item.output_domain,
        "rationale": item.rationale,
        "review_mode_expectation": item.review_mode_expectation,
        "promotion_eligibility": item.promotion_eligibility,
    }
    if item.metadata:
        metadata["metadata"] = item.metadata
    return metadata


def default_backends_for(semantics: str, case: SweepCase) -> list[str]:
    if semantics in {"bounded-i64", "bounded-u64"}:
        return ["cpu", "hip-direct", "hip-vector-alu-int64", "ck", "rocwmma"] if case.bound_mode == "per-tile" else BOUNDED_BACKENDS
    if semantics in {"finite-u8-ring", "finite-u8-field"}:
        return FINITE_BACKENDS
    if semantics in {"exact-wide-signed", "exact-wide-unsigned"}:
        return EXACT_WIDE_BACKENDS
    if semantics == "wrap-u64":
        return WRAP64_BACKENDS
    return []


def backend_allowed_for(semantics: str, case: SweepCase, backend: str) -> bool:
    if semantics == "wrap-u64":
        return backend in WRAP64_BACKENDS or backend == WRAP64_ROCWMMA_CANDIDATE_BACKEND
    if semantics in {"finite-u8-ring", "finite-u8-field"}:
        return backend in FINITE_BACKENDS
    if semantics in {"exact-wide-signed", "exact-wide-unsigned"}:
        return case.bound_mode == "global" and backend in EXACT_WIDE_BACKENDS
    if semantics in {"bounded-i64", "bounded-u64"}:
        if backend not in BOUNDED_BACKENDS:
            return False
        if case.bound_mode == "per-tile" and backend == "hipblaslt":
            return False
        return True
    return False


def capture_name(
    semantics: str,
    case: SweepCase,
    backend: str,
    modulus: int | None,
    pack_mode: str,
    exact_wide_limb_count: int | None = None,
    residue_chain_length: int = 1,
    residue_chain_final_export: bool = False,
    output_ld_padding: int = 0,
    host_api_batch_size: int = 1,
    hip_graph_replay: bool = False,
    oneshot: bool = False,
    residue_chain_independent_final_export: bool = False,
) -> str:
    parts = [semantics, case.name, f"{case.m}x{case.n}x{case.k}"]
    if modulus is not None:
        parts.append(f"mod{modulus}")
    if semantics in EXACT_WIDE_SEMANTICS and exact_wide_limb_count not in (None, DEFAULT_EXACT_WIDE_LIMB_COUNT):
        parts.append(f"limbs{exact_wide_limb_count}")
    if semantics in RNS_CHAIN_SEMANTICS and residue_chain_length > 1:
        parts.append(f"chain{residue_chain_length}")
        if residue_chain_independent_final_export:
            parts.append("indepfinalexport")
        elif residue_chain_final_export:
            parts.append("finalexport")
    if output_ld_padding > 0:
        parts.append(f"outpad{output_ld_padding}")
    if host_api_batch_size > 1:
        parts.append(f"hostbatch{host_api_batch_size}")
    if hip_graph_replay:
        parts.append("hipgraph")
    if oneshot:
        parts.append("oneshot")
    if pack_mode == "prepacked_reuse":
        parts.append("reuse-packed")
    elif pack_mode == "prepacked_reuse_a":
        parts.append("reuse-packed-a")
    elif pack_mode == "prepacked_reuse_b":
        parts.append("reuse-packed-b")
    parts.append(backend)
    return "-".join(parts).replace("/", "_") + ".json"


def command_for(
    bench: Path,
    backend: str,
    semantics: str,
    case: SweepCase,
    modulus: int | None,
    exact_wide_limb_count: int | None,
    args: argparse.Namespace,
    *,
    oneshot: bool = False,
) -> list[str]:
    tile_m = 16 if semantics == "wrap-u64" and backend == WRAP64_ROCWMMA_CANDIDATE_BACKEND else case.tile_m
    tile_n = 16 if semantics == "wrap-u64" and backend == WRAP64_ROCWMMA_CANDIDATE_BACKEND else case.tile_n
    command = [
        str(bench),
        "--backend",
        cli_backend(backend),
        "--semantics",
        semantics,
        "--m",
        str(case.m),
        "--n",
        str(case.n),
        "--k",
        str(case.k),
        "--tile-m",
        str(tile_m),
        "--tile-n",
        str(tile_n),
        "--bound-mode",
        case.bound_mode,
        "--warmups",
        str(args.warmups),
        "--repeats",
        str(args.repeats),
        "--seed",
        str(args.seed),
    ]
    if case.require_adaptive and getattr(args, "prefix_policy", None) != "fixed-requested":
        command.append("--require-adaptive-execution")
    if case.input_profile != "uniform-small":
        command.extend(["--input-profile", case.input_profile])
    if modulus is not None:
        command.extend(["--modulus", str(modulus)])
    if exact_wide_limb_count is not None:
        command.extend(["--exact-wide-limbs", str(exact_wide_limb_count)])
    bound_source = getattr(args, "bound_source", None)
    if bound_source:
        command.extend(["--bound-source", bound_source])
    next_op_hint = getattr(args, "next_op_hint", None)
    if next_op_hint:
        command.extend(["--next-op-hint", next_op_hint])
    prefix_policy = getattr(args, "prefix_policy", None)
    max_prefix = getattr(args, "max_prefix", None)
    if prefix_policy:
        command.extend(["--prefix-policy", prefix_policy])
    if max_prefix is not None:
        command.extend(["--max-prefix", str(max_prefix)])
    if args.residue_chain_length > 1:
        command.extend(["--residue-chain-length", str(args.residue_chain_length)])
    if getattr(args, "residue_chain_independent_final_export", False):
        command.append("--residue-chain-independent-final-export")
    elif getattr(args, "residue_chain_final_export", False):
        command.append("--residue-chain-final-export")
    host_api_batch_size = int(getattr(args, "host_api_batch_size", 1) or 1)
    if host_api_batch_size > 1:
        command.extend(["--host-api-batch-size", str(host_api_batch_size)])
    output_ld_padding = int(getattr(args, "output_ld_padding", 0) or 0)
    if output_ld_padding > 0:
        command.extend(["--output-ld-padding", str(output_ld_padding)])
    if getattr(args, "residue_channel_fusion", False):
        command.append("--residue-channel-fusion")
    if oneshot:
        command.append("--oneshot")
    pack_mode = requested_pack_mode(args)
    if pack_mode == "prepacked_reuse":
        command.append("--reuse-packed-inputs")
    elif pack_mode == "prepacked_reuse_a":
        command.append("--reuse-packed-a")
    elif pack_mode == "prepacked_reuse_b":
        command.append("--reuse-packed-b")
    if getattr(args, "native_to_rns_bridge", False):
        command.append("--native-to-rns-bridge")
    if getattr(args, "vector_to_rns_chain", False):
        command.append("--vector-to-rns-chain")
    modulus_set = getattr(args, "modulus_set", "default")
    if modulus_set and modulus_set != "default":
        command.extend(["--modulus-set", modulus_set])
    tile_shape_variant = getattr(args, "tile_shape_variant", "default")
    if tile_shape_variant and tile_shape_variant != "default":
        command.extend(["--tile-shape-variant", tile_shape_variant])
    export_variant = getattr(args, "export_variant", "default")
    if export_variant and export_variant != "default":
        command.extend(["--export-variant", export_variant])
    reconstruction_variant = getattr(args, "reconstruction_variant", "default_garner")
    if reconstruction_variant and reconstruction_variant != "default_garner":
        command.extend(["--reconstruction-variant", reconstruction_variant])
    grouped_dispatch_tasks = int(getattr(args, "grouped_dispatch_tasks", 1) or 1)
    if grouped_dispatch_tasks > 1:
        command.extend(["--grouped-dispatch", str(grouped_dispatch_tasks)])
    if getattr(args, "hip_graph_replay", False):
        command.append("--hip-graph-replay")
    workload_proxy = getattr(args, "workload_proxy", "none")
    if workload_proxy and workload_proxy != "none":
        command.extend(["--workload-proxy", workload_proxy])
    if getattr(args, "resident_lifetime", False):
        command.append("--resident-lifetime")
    if getattr(args, "workspace_arena", False):
        command.append("--workspace-arena")
    if getattr(args, "adaptive_grouped_scheduler", False):
        command.append("--adaptive-grouped-scheduler")
    if getattr(args, "streaming_overlap", False):
        command.append("--streaming-overlap")
    release_gate = getattr(args, "release_gate", "none")
    if release_gate and release_gate != "none":
        command.extend(["--release-gate", release_gate])
    verification_amortization = getattr(args, "verification_amortization", "none")
    if verification_amortization and verification_amortization != "none":
        command.extend(["--verification-amortization", verification_amortization])
    return command


def default_sweep_command_entries(args: argparse.Namespace) -> list[SweepCommand]:
    backend_benches = parse_backend_bench(args.bench_for)
    commands: list[SweepCommand] = []
    host_api_batch_size = int(getattr(args, "host_api_batch_size", 1) or 1)
    if host_api_batch_size <= 0:
        raise SystemExit("--host-api-batch-size must be positive")
    if host_api_batch_size > 1:
        if any(
            [
                getattr(args, "include_oneshot", False),
                getattr(args, "oneshot_only", False),
                getattr(args, "reuse_packed_inputs", False),
                getattr(args, "reuse_packed_a", False),
                getattr(args, "reuse_packed_b", False),
                getattr(args, "native_to_rns_bridge", False),
                getattr(args, "vector_to_rns_chain", False),
                getattr(args, "residue_channel_fusion", False),
                getattr(args, "residue_chain_independent_final_export", False),
                int(getattr(args, "residue_chain_length", 1) or 1) != 1,
                getattr(args, "bound_source", None) == "input-scan",
            ]
        ):
            raise SystemExit(
                "--host-api-batch-size > 1 cannot be combined with one-shot, reuse, bridge, chain, "
                "residue-fusion, residue-chain, or input-scan modes"
            )
        if args.backends:
            args.backends = [backend for backend in args.backends if backend in HOST_API_BATCH_BACKENDS]
    semantics_values = [normalize_semantics(item) for item in (args.semantics or ["bounded-i64", "bounded-u64"])]
    cases = [*([] if args.adaptive_only else default_cases(args)), *adaptive_cases(args)]
    if args.adaptive_only and not cases:
        raise SystemExit("--adaptive-only requires --adaptive-case, --include-default-adaptive, or --include-adaptive-workloads")
    if (args.include_wrap64 or args.include_wrap64_rocwmma_candidate) and "wrap-u64" not in semantics_values:
        semantics_values.append("wrap-u64")
    if args.include_exact_wide:
        for exact_semantics in EXACT_WIDE_SEMANTICS:
            if exact_semantics not in semantics_values:
                semantics_values.append(exact_semantics)
    if (getattr(args, "prefix_policy", None) or getattr(args, "max_prefix", None) is not None) and any(
        semantics not in {"bounded-i64", "bounded-u64", "exact-wide-signed", "exact-wide-unsigned"}
        for semantics in semantics_values
    ):
        raise SystemExit("--prefix-policy and --max-prefix are only valid for bounded or exact-wide RNS sweeps")
    if getattr(args, "bound_source", None) == "input-scan" and any(
        semantics not in {"bounded-i64", "bounded-u64"} for semantics in semantics_values
    ):
        raise SystemExit("--bound-source input-scan is only valid for bounded RNS sweeps")
    if getattr(args, "max_prefix", None) is not None and args.max_prefix <= 0:
        raise SystemExit("--max-prefix must be positive")
    if int(getattr(args, "output_ld_padding", 0) or 0) < 0:
        raise SystemExit("--output-ld-padding must be nonnegative")
    if getattr(args, "residue_channel_fusion", False):
        if requested_pack_mode(args) != "per_repeat_repack":
            raise SystemExit("--residue-channel-fusion cannot be combined with packed-input reuse")
        if getattr(args, "prefix_policy", None) != "fixed-requested" or getattr(args, "max_prefix", None) not in {9, None}:
            raise SystemExit("--residue-channel-fusion requires --prefix-policy fixed-requested and --max-prefix 9")
        if any(semantics not in BOUNDED_SEMANTICS for semantics in semantics_values):
            raise SystemExit("--residue-channel-fusion is only valid for bounded semantics")
    if getattr(args, "hip_graph_replay", False):
        if getattr(args, "residue_chain_final_export", False) or getattr(
            args, "residue_chain_independent_final_export", False
        ):
            raise SystemExit("--hip-graph-replay cannot be combined with residue-chain final-export modes")
        if requested_pack_mode(args) != "prepacked_reuse":
            raise SystemExit("--hip-graph-replay requires --reuse-packed-inputs")
        if args.residue_chain_length <= 1:
            raise SystemExit("--hip-graph-replay requires --residue-chain-length > 1")
        if getattr(args, "next_op_hint", None) != "rns-gemm":
            raise SystemExit("--hip-graph-replay requires --next-op-hint rns-gemm")
        if host_api_batch_size > 1 or getattr(args, "include_oneshot", False) or getattr(args, "oneshot_only", False):
            raise SystemExit("--hip-graph-replay cannot be combined with host batching or one-shot sweeps")
        if getattr(args, "native_to_rns_bridge", False) or getattr(args, "vector_to_rns_chain", False):
            raise SystemExit("--hip-graph-replay cannot be combined with bridge or vector-to-RNS chain sweeps")
        if getattr(args, "residue_channel_fusion", False):
            raise SystemExit("--hip-graph-replay cannot be combined with residue-channel fusion")
        if getattr(args, "bound_source", None) == "input-scan":
            raise SystemExit("--hip-graph-replay currently requires static-profile bounds")
        if any(semantics not in RNS_CHAIN_SEMANTICS for semantics in semantics_values):
            raise SystemExit("--hip-graph-replay is only valid for bounded or exact-wide RNS sweeps")
    if args.residue_chain_length < 1:
        raise SystemExit("--residue-chain-length must be positive")
    if getattr(args, "residue_chain_independent_final_export", False) and args.residue_chain_length <= 1:
        raise SystemExit("--residue-chain-independent-final-export requires --residue-chain-length > 1")
    if getattr(args, "residue_chain_final_export", False) and args.residue_chain_length <= 1:
        raise SystemExit("--residue-chain-final-export requires --residue-chain-length > 1")
    if (
        getattr(args, "residue_chain_final_export", False)
        or getattr(args, "residue_chain_independent_final_export", False)
    ) and getattr(args, "next_op_hint", None) == "rns-gemm":
        raise SystemExit("residue-chain final-export modes cannot use --next-op-hint rns-gemm")
    if args.residue_chain_length > 1:
        non_rns_chain = [semantics for semantics in semantics_values if semantics not in RNS_CHAIN_SEMANTICS]
        if non_rns_chain:
            raise SystemExit("--residue-chain-length > 1 currently requires bounded or exact-wide RNS semantics")
    if getattr(args, "residue_chain_independent_final_export", False):
        non_independent_chain = [
            semantics
            for semantics in semantics_values
            if semantics not in BOUNDED_SEMANTICS and semantics not in EXACT_WIDE_SEMANTICS
        ]
        if non_independent_chain:
            raise SystemExit("--residue-chain-independent-final-export supports bounded or exact-wide RNS semantics only")
        if requested_pack_mode(args) != "per_repeat_repack":
            raise SystemExit("--residue-chain-independent-final-export cannot be combined with packed-input reuse")
        if any(
            [
                getattr(args, "include_oneshot", False),
                getattr(args, "oneshot_only", False),
                getattr(args, "native_to_rns_bridge", False),
                getattr(args, "vector_to_rns_chain", False),
                getattr(args, "residue_channel_fusion", False),
                host_api_batch_size > 1,
            ]
        ):
            raise SystemExit(
                "--residue-chain-independent-final-export cannot be combined with one-shot, bridge, "
                "vector-to-RNS chain, residue-fusion, or host batching"
            )
    include_oneshot = bool(getattr(args, "include_oneshot", False))
    oneshot_only = bool(getattr(args, "oneshot_only", False))
    if include_oneshot or oneshot_only:
        if args.adaptive_only:
            raise SystemExit("--include-oneshot cannot be combined with --adaptive-only")
        if requested_pack_mode(args) != "per_repeat_repack":
            raise SystemExit("--include-oneshot cannot be combined with packed-input reuse modes")
        if args.residue_chain_length > 1:
            raise SystemExit("--include-oneshot cannot be combined with --residue-chain-length > 1")
    for semantics in semantics_values:
        if semantics == "wrap-u64":
            if args.adaptive_only:
                continue
            active_cases = wrap64_cases(args)
        else:
            active_cases = cases
        for case in active_cases:
            if args.residue_chain_length > 1 and (case.m != case.n or case.n != case.k):
                raise SystemExit("--residue-chain-length > 1 currently requires square RNS cases")
            if args.residue_chain_length > 1 and semantics in BOUNDED_SEMANTICS and case.bound_mode != "global":
                raise SystemExit("bounded --residue-chain-length > 1 currently requires global bound mode")
            backends = args.backends or (
                wrap64_backends_for(args) if semantics == "wrap-u64" else default_backends_for(semantics, case)
            )
            if host_api_batch_size > 1:
                backends = [backend for backend in backends if backend in HOST_API_BATCH_BACKENDS]
            if getattr(args, "residue_channel_fusion", False):
                backends = [backend for backend in backends if backend == "hip-direct"]
            if getattr(args, "hip_graph_replay", False):
                backends = [backend for backend in backends if backend == "hip-direct"]
            for modulus in finite_moduli_for(semantics, args):
                for exact_wide_limb_count in exact_wide_limb_counts_for(semantics, args):
                    if not oneshot_only:
                        for backend in backends:
                            if not backend_allowed_for(semantics, case, backend):
                                continue
                            if (
                                args.residue_chain_length > 1
                                and semantics in BOUNDED_SEMANTICS
                                and backend in {"auto", "hip-vector-alu-int64"}
                            ):
                                continue
                            bench = backend_benches.get(backend, args.bench)
                            if bench is None:
                                raise SystemExit(f"no benchmark executable configured for backend {backend}")
                            name = capture_name(
                                semantics,
                                case,
                                backend,
                                modulus,
                                requested_pack_mode(args),
                                exact_wide_limb_count,
                                args.residue_chain_length,
                                bool(getattr(args, "residue_chain_final_export", False)),
                                int(getattr(args, "output_ld_padding", 0) or 0),
                                int(getattr(args, "host_api_batch_size", 1) or 1),
                                bool(getattr(args, "hip_graph_replay", False)),
                                residue_chain_independent_final_export=bool(
                                    getattr(args, "residue_chain_independent_final_export", False)
                                ),
                            )
                            command = command_for(bench, backend, semantics, case, modulus, exact_wide_limb_count, args)
                            commands.append(SweepCommand(name, command, args.out_root / name))
                    if (include_oneshot or oneshot_only) and (
                        (semantics in BOUNDED_SEMANTICS and case.bound_mode == "global")
                        or semantics in {"finite-u8-ring", "finite-u8-field"}
                    ):
                        oneshot_backends = [
                            backend
                            for backend in (args.backends or PUBLIC_ONESHOT_BACKENDS)
                            if backend in PUBLIC_ONESHOT_BACKENDS
                        ]
                        for backend in oneshot_backends:
                            bench = backend_benches.get(backend, args.bench)
                            if bench is None:
                                raise SystemExit(f"no benchmark executable configured for backend {backend}")
                            name = capture_name(
                                semantics,
                                case,
                                backend,
                                modulus,
                                requested_pack_mode(args),
                                exact_wide_limb_count,
                                args.residue_chain_length,
                                bool(getattr(args, "residue_chain_final_export", False)),
                                int(getattr(args, "output_ld_padding", 0) or 0),
                                int(getattr(args, "host_api_batch_size", 1) or 1),
                                bool(getattr(args, "hip_graph_replay", False)),
                                oneshot=True,
                            )
                            command = command_for(
                                bench,
                                backend,
                                semantics,
                                case,
                                modulus,
                                exact_wide_limb_count,
                                args,
                                oneshot=True,
                            )
                            commands.append(SweepCommand(name, command, args.out_root / name))
    return commands


def scenario_sweep_command_entries(args: argparse.Namespace) -> list[SweepCommand]:
    if any(
        [
            getattr(args, "case", None),
            getattr(args, "adaptive_case", None),
            getattr(args, "shapes", None),
            getattr(args, "include_default_adaptive", False),
            getattr(args, "include_adaptive_workloads", False),
            getattr(args, "adaptive_only", False),
            getattr(args, "include_wrap64", False),
            getattr(args, "include_exact_wide", False),
            getattr(args, "include_oneshot", False),
            getattr(args, "oneshot_only", False),
            getattr(args, "reuse_packed_inputs", False),
            getattr(args, "reuse_packed_a", False),
            getattr(args, "reuse_packed_b", False),
            getattr(args, "release_matrix", False),
            getattr(args, "include_exploratory_large", False),
            getattr(args, "prefix_policy", None),
            getattr(args, "max_prefix", None) is not None,
            getattr(args, "bound_source", None),
            getattr(args, "next_op_hint", None),
            getattr(args, "residue_channel_fusion", False),
            getattr(args, "modulus_set", "default") != "default",
            getattr(args, "tile_shape_variant", "default") != "default",
            getattr(args, "export_variant", "default") != "default",
            getattr(args, "reconstruction_variant", "default_garner") != "default_garner",
            int(getattr(args, "grouped_dispatch_tasks", 1) or 1) != 1,
            getattr(args, "hip_graph_replay", False),
            getattr(args, "residue_chain_final_export", False),
            getattr(args, "residue_chain_independent_final_export", False),
            getattr(args, "workload_proxy", "none") != "none",
            int(getattr(args, "output_ld_padding", 0) or 0) != 0,
            int(getattr(args, "residue_chain_length", 1) or 1) != 1,
            int(getattr(args, "host_api_batch_size", 1) or 1) != 1,
        ]
    ):
        raise SystemExit("--scenario cannot be combined with manual sweep shape/reuse/include flags")

    backend_benches = parse_backend_bench(args.bench_for)
    semantics_filter = {normalize_semantics(item) for item in (args.semantics or [])}
    commands: list[SweepCommand] = []
    for item in selected_scenario_items(args):
        if semantics_filter and item.semantics not in semantics_filter:
            continue
        scenario_args = scenario_args_for_item(args, item)
        backends = scenario_backends_for_item(args, item)
        if not backends:
            continue
        finite_moduli = finite_moduli_for(item.semantics, args) if args.modulus else list(item.finite_moduli)
        if args.exact_wide_limbs:
            exact_wide_counts = exact_wide_limb_counts_for(item.semantics, args)
        else:
            exact_wide_counts = list(item.exact_wide_limb_counts)
        for modulus in finite_moduli:
            for exact_wide_limb_count in exact_wide_counts:
                for backend in backends:
                    if (
                        not item.native_to_rns_bridge
                        and not item.vector_to_rns_chain
                        and not backend_allowed_for(item.semantics, item.case, backend)
                    ):
                        continue
                    if (
                        item.residue_chain_length > 1
                        and item.semantics in BOUNDED_SEMANTICS
                        and backend in {"auto", "hip-vector-alu-int64"}
                    ):
                        continue
                    bench = backend_benches.get(backend)
                    if bench is None and (item.native_to_rns_bridge or item.vector_to_rns_chain) and backend == "auto":
                        bench = backend_benches.get("hip-direct")
                    if bench is None:
                        bench = args.bench
                    if bench is None:
                        raise SystemExit(f"no benchmark executable configured for backend {backend}")
                    base_name = capture_name(
                        item.semantics,
                        item.case,
                        backend,
                        modulus,
                        requested_pack_mode(scenario_args),
                        exact_wide_limb_count,
                        item.residue_chain_length,
                        item.residue_chain_final_export,
                        item.output_ld_padding,
                        item.host_api_batch_size,
                        item.hip_graph_replay,
                        oneshot=item.oneshot,
                        residue_chain_independent_final_export=item.residue_chain_independent_final_export,
                    )
                    name = f"{item.family}-{item.name}-{base_name}"
                    command = command_for(
                        bench,
                        backend,
                        item.semantics,
                        item.case,
                        modulus,
                        exact_wide_limb_count,
                        scenario_args,
                        oneshot=item.oneshot,
                    )
                    metadata = scenario_metadata(
                        item,
                        backend,
                        modulus,
                        exact_wide_limb_count,
                        scenario_args,
                        oneshot=item.oneshot,
                    )
                    output = args.out_root / "scenarios" / item.family / name
                    commands.append(SweepCommand(name, command, output, metadata))
    return commands


def sweep_command_entries(args: argparse.Namespace) -> list[SweepCommand]:
    return (
        scenario_sweep_command_entries(args)
        if getattr(args, "scenario", None)
        else default_sweep_command_entries(args)
    )


def sweep_commands(args: argparse.Namespace) -> list[tuple[str, list[str], Path]]:
    return [(entry.name, entry.command, entry.output) for entry in sweep_command_entries(args)]


