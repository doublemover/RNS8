from __future__ import annotations

import argparse
import json
from pathlib import Path

from evidence_database_lib.isa import load_isa_index
from evidence_database_lib.io import discover_capture_paths, discover_isa_report_paths

from .commands import scenario_names, sweep_command_entries
from .execution import autotune_cache_path, execute_sweep_entries, validate_paths
from .reports import write_markdown_report, write_scenario_manifest
from .review import attach_cache_write_status, review_captures, write_promoted_cache_entries

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", type=Path, help="path to rns8-bench executable")
    parser.add_argument(
        "--bench-for",
        action="append",
        default=[],
        metavar="BACKEND=PATH",
        help="override rns8-bench executable for one backend, useful for opt-in accelerator build dirs",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("temp") / "benchmark-sweeps" / "windows-gfx1100",
        help="ignored output directory for raw captures and review report",
    )
    parser.add_argument("--capture", type=Path, action="append", default=[], help="existing capture to review")
    parser.add_argument(
        "--capture-root",
        type=Path,
        action="append",
        default=[],
        help="directory containing existing benchmark capture JSON files to review",
    )
    parser.add_argument(
        "--isa-report",
        type=Path,
        action="append",
        default=[],
        help="gpu_isa_report.py JSON summary or directory of *-isa-summary.json reports to attach to review candidates",
    )
    parser.add_argument("--review-only", action="store_true", help="only review --capture files")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="reuse valid existing capture outputs instead of rerunning them",
    )
    parser.add_argument(
        "--max-new-captures",
        type=int,
        help="limit the number of newly executed captures while still reviewing existing/skipped captures",
    )
    parser.add_argument(
        "--capture-timeout-seconds",
        type=float,
        help="optional wall-clock timeout for each benchmark capture command; timed-out captures are recorded as .failed.json",
    )
    parser.add_argument(
        "--hip-visible-devices",
        help="set HIP_VISIBLE_DEVICES for every capture command, e.g. 0, 0,1, 0,1,2,3, or 0,1,2,3,4,5,6,7",
    )
    parser.add_argument(
        "--rocr-visible-devices",
        help="set ROCR_VISIBLE_DEVICES for every capture command using the same comma-separated device-list syntax",
    )
    parser.add_argument(
        "--gpu-device-ordinal",
        help="set GPU_DEVICE_ORDINAL for every capture command using the same comma-separated device-list syntax",
    )
    parser.add_argument(
        "--gpu-shards",
        help="duplicate each capture once per listed GPU with HIP_VISIBLE_DEVICES set to that single GPU and outputs under gpuN/",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        choices=[*scenario_names(), "all", "release-candidates"],
        help="run a named scenario corpus family; repeatable, or use all/release-candidates",
    )
    parser.add_argument("--backend", dest="backends", action="append", help="backend to sweep; repeatable")
    parser.add_argument("--semantics", action="append", help="benchmark semantics to sweep; repeatable")
    parser.add_argument("--case", action="append", help="global case NAME:M,N,K; repeatable")
    parser.add_argument(
        "--adaptive-case",
        action="append",
        help="adaptive case NAME:M,N,K,TILE_M,TILE_N[,INPUT_PROFILE]; repeatable",
    )
    parser.add_argument("--shape", dest="shapes", action="append", help="legacy square shape to sweep; repeatable")
    parser.add_argument("--modulus", type=int, action="append", help="finite-u8 modulus; repeatable")
    parser.add_argument("--exact-wide-limbs", type=int, action="append", help="exact-wide output limb count; repeatable")
    parser.add_argument(
        "--bound-source",
        choices=["static-profile", "input-scan"],
        help="bounded RNS bound source to pass through to rns8-bench",
    )
    parser.add_argument(
        "--next-op-hint",
        choices=["final-export", "rns-gemm", "native-gemm", "native-to-rns", "reuse-b"],
        help="benchmark-only next-operation hint to pass through to rns8-bench",
    )
    parser.add_argument(
        "--prefix-policy",
        choices=["minimum-proven", "fixed-requested"],
        help="bounded/exact-wide RNS prefix policy to pass through to rns8-bench",
    )
    parser.add_argument(
        "--max-prefix",
        type=int,
        help="bounded/exact-wide RNS requested maximum prefix to pass through to rns8-bench",
    )
    parser.add_argument(
        "--residue-chain-length",
        type=int,
        default=1,
        help="bounded/exact-wide RNS GEMM chain length; values above 1 leave output residue-current unless final export is requested",
    )
    parser.add_argument(
        "--residue-chain-final-export",
        action="store_true",
        help="measure one final host export inside each residue-chain repeat instead of leaving output residue-current",
    )
    parser.add_argument(
        "--residue-chain-independent-final-export",
        action="store_true",
        help="measure a bounded residue chain as independent GEMMs with host export and repack between steps",
    )
    parser.add_argument(
        "--host-api-batch-size",
        type=int,
        default=1,
        help="benchmark-owned host API batch size for persistent resident matrix captures",
    )
    parser.add_argument(
        "--output-ld-padding",
        type=int,
        default=0,
        help="add padding columns to the benchmark host output leading dimension",
    )
    parser.add_argument(
        "--residue-channel-fusion",
        action="store_true",
        help="benchmark-only direct-HIP residue-channel fusion experiment passthrough",
    )
    parser.add_argument(
        "--modulus-set",
        default="default",
        help="benchmark-only modulus set metadata, default or experimental:NAME",
    )
    parser.add_argument(
        "--tile-shape-variant",
        default="default",
        help="benchmark-only tile-shape variant label",
    )
    parser.add_argument(
        "--export-variant",
        default="default",
        help="benchmark-only export/reconstruction evidence label",
    )
    parser.add_argument(
        "--reconstruction-variant",
        default="default_garner",
        help="benchmark-only reconstruction variant label",
    )
    parser.add_argument(
        "--grouped-dispatch-tasks",
        type=int,
        default=1,
        help="benchmark-only grouped-dispatch task count metadata",
    )
    parser.add_argument(
        "--hip-graph-replay",
        action="store_true",
        help="benchmark-only Direct-HIP HIP Graph replay experiment passthrough",
    )
    parser.add_argument(
        "--workload-proxy",
        default="none",
        help="benchmark-only workload proxy label",
    )
    parser.add_argument(
        "--resident-lifetime",
        action="store_true",
        help="benchmark-only resident matrix lifetime evidence metadata",
    )
    parser.add_argument(
        "--workspace-arena",
        action="store_true",
        help="benchmark-only workspace arena evidence metadata",
    )
    parser.add_argument(
        "--adaptive-grouped-scheduler",
        action="store_true",
        help="benchmark-only adaptive prefix grouped scheduler evidence metadata",
    )
    parser.add_argument(
        "--streaming-overlap",
        action="store_true",
        help="benchmark-only streaming pack/compute/export overlap evidence metadata",
    )
    parser.add_argument(
        "--k-block-policy",
        default="auto",
        help="benchmark-only K-block/tile-K policy label",
    )
    parser.add_argument(
        "--release-gate",
        default="none",
        help="benchmark-only release gate label such as large-release-validation-4096-budgeted",
    )
    parser.add_argument(
        "--verification-amortization",
        default="none",
        help="benchmark-only verification amortization policy label",
    )
    parser.add_argument(
        "--error-detection-policy",
        default="none",
        help="benchmark-only research error-detection policy label; never changes exact API semantics",
    )
    parser.add_argument(
        "--cpu-small-shape-selector",
        default="none",
        help="benchmark-only CPU small-shape selector evidence label; does not change AUTO routing",
    )
    parser.add_argument(
        "--incremental-result-cache",
        default="none",
        help="benchmark-only incremental result-cache research label; never enables default routing",
    )
    parser.add_argument(
        "--include-exact-wide-limb-variants",
        action="store_true",
        help="include exact-wide output limb counts 1, 2, 3, 4, 8, 16, and 32",
    )
    parser.add_argument("--include-default-adaptive", action="store_true", help="include default adaptive bounded cases")
    parser.add_argument(
        "--include-adaptive-workloads",
        action="store_true",
        help="include profile-driven adaptive bounded workload cases",
    )
    parser.add_argument("--adaptive-only", action="store_true", help="run only adaptive cases, skipping global cases")
    parser.add_argument("--include-wrap64", action="store_true", help="include wrap64 CPU/direct-HIP captures")
    parser.add_argument(
        "--include-rocwmma-wrap64-candidate",
        dest="include_wrap64_rocwmma_candidate",
        action="store_true",
        help="include the internal rocWMMA wrap64 byte-GEMM36 candidate in wrap64 sweeps",
    )
    parser.add_argument(
        "--include-exact-wide",
        action="store_true",
        help="include exact-wide signed and unsigned captures",
    )
    parser.add_argument(
        "--reuse-packed-inputs",
        action="store_true",
        help="pack A/B once before warmups and benchmark repeated GEMM/export against persistent packed inputs",
    )
    parser.add_argument(
        "--reuse-packed-a",
        action="store_true",
        help="pack A once before warmups and benchmark repeats that repack B",
    )
    parser.add_argument(
        "--reuse-packed-b",
        action="store_true",
        help="pack B once before warmups and benchmark repeats that repack A",
    )
    parser.add_argument(
        "--include-oneshot",
        action="store_true",
        help="also capture bounded/finite CPU/direct-HIP public one-shot API runs beside persistent matrix baselines",
    )
    parser.add_argument(
        "--oneshot-only",
        action="store_true",
        help="capture only bounded/finite CPU/direct-HIP public one-shot API runs for the selected global cases",
    )
    parser.add_argument("--release-matrix", action="store_true", help="use promotable release bounded shapes 64..1024")
    parser.add_argument("--include-exploratory-large", action="store_true", help="include 2048/4096/8192 exploratory shapes")
    parser.add_argument(
        "--review-mode",
        choices=["smoke", "release"],
        default="smoke",
        help="release mode is required before captures can produce performance_validated autotune entries",
    )
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=0,
        help="pass through to rns8-bench; 0 leaves OpenMP/env default thread selection in control",
    )
    parser.add_argument(
        "--cpu-parallel-threshold",
        type=int,
        default=1 << 20,
        help="pass through to rns8-bench as the CPU OpenMP work-estimate threshold",
    )
    parser.add_argument(
        "--cpu-reference-mode",
        choices=["timed-baseline", "correctness-anchor"],
        default="timed-baseline",
        help="mark CPU captures as timed baselines or single exact correctness anchors",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="stream per-capture start/end progress and pass --progress through to rns8-bench stderr",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--write-autotune-cache",
        action="store_true",
        help="write performance_validated cache entries only for fastest promotable release-reviewed captures",
    )
    parser.add_argument("--autotune-cache", type=Path, help="override autotune cache output path")
    args = parser.parse_args()
    if args.max_new_captures is not None and args.max_new_captures < 0:
        parser.error("--max-new-captures must be non-negative")
    if args.capture_timeout_seconds is not None and args.capture_timeout_seconds <= 0:
        parser.error("--capture-timeout-seconds must be positive")
    if args.cpu_threads < 0:
        parser.error("--cpu-threads must be non-negative")
    if args.cpu_parallel_threshold < 0:
        parser.error("--cpu-parallel-threshold must be non-negative")
    for option_name in ["hip_visible_devices", "rocr_visible_devices", "gpu_device_ordinal", "gpu_shards"]:
        value = getattr(args, option_name, None)
        if value is None:
            continue
        parts = [part.strip() for part in value.split(",")]
        if not parts or any(not part.isdecimal() for part in parts):
            parser.error(f"--{option_name.replace('_', '-')} must be a comma-separated list of nonnegative device indices")
    if args.gpu_shards and (args.hip_visible_devices or args.rocr_visible_devices):
        parser.error(
            "--gpu-shards sets HIP_VISIBLE_DEVICES and ROCR_VISIBLE_DEVICES per shard and cannot be combined "
            "with --hip-visible-devices or --rocr-visible-devices"
        )
    if args.grouped_dispatch_tasks <= 0:
        parser.error("--grouped-dispatch-tasks must be positive")
    if args.modulus_set != "default" and not args.modulus_set.startswith("experimental:"):
        parser.error("--modulus-set must be default or experimental:NAME")
    if not args.release_gate:
        parser.error("--release-gate must not be empty")
    if not args.verification_amortization:
        parser.error("--verification-amortization must not be empty")
    if not args.error_detection_policy:
        parser.error("--error-detection-policy must not be empty")
    if not args.cpu_small_shape_selector:
        parser.error("--cpu-small-shape-selector must not be empty")
    if not args.incremental_result_cache:
        parser.error("--incremental-result-cache must not be empty")
    return args


def review_capture_paths(args: argparse.Namespace) -> list[Path]:
    paths = list(args.capture)
    if args.capture_root:
        paths.extend(discover_capture_paths(args.capture_root))
    if args.review_only and not paths:
        default_root = args.out_root / "scenarios"
        if default_root.exists():
            paths.extend(discover_capture_paths([default_root]))
    return list(dict.fromkeys(paths))


def load_required_isa_index(paths: list[Path]) -> dict:
    for path in paths:
        if not path.exists():
            raise SystemExit(f"--isa-report path does not exist: {path}")
    discovered = discover_isa_report_paths(paths)
    if not discovered:
        joined = ", ".join(str(path) for path in paths)
        raise SystemExit(f"--isa-report found no *-isa-summary.json reports under: {joined}")
    return load_isa_index(paths)


def main() -> int:
    args = parse_args()
    args.out_root = Path(args.out_root)
    args.out_root.mkdir(parents=True, exist_ok=True)

    capture_paths = review_capture_paths(args)
    entries: list[SweepCommand] = []
    execution_stats: dict[str, int] | None = None
    if not args.review_only:
        if args.bench is None:
            raise SystemExit("--bench is required unless --review-only is used")
        entries = sweep_command_entries(args)
        execution_stats = execute_sweep_entries(entries, args, capture_paths)

    captures = validate_paths(capture_paths)
    isa_index = load_required_isa_index(args.isa_report) if args.isa_report else None
    report = review_captures(captures, review_mode=args.review_mode, isa_index=isa_index)
    promoted = 0
    cache_path = args.autotune_cache or autotune_cache_path()
    if args.write_autotune_cache:
        promoted = write_promoted_cache_entries(report, captures, cache_path)
    attach_cache_write_status(report, args.write_autotune_cache, cache_path, promoted)
    report_path = args.out_root / "review_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path = args.out_root / "review_report.md"
    write_markdown_report(report, markdown_path)
    scenario_paths = write_scenario_manifest(entries, args, args.out_root)
    output = {
        "review_report": str(report_path),
        "markdown_report": str(markdown_path),
        "captures": len(captures),
        "promoted_cache_entries": promoted,
        "autotune_cache": str(cache_path) if args.write_autotune_cache else None,
    }
    if execution_stats is not None:
        output["sweep_execution"] = execution_stats
    if scenario_paths is not None:
        output.update(scenario_paths)
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
