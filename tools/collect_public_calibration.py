"""Counterbalanced independent public API observations; no installation/promotion.

This collector executes only when explicitly invoked. Plan-only writes commands
without starting a benchmark. Every output gets the existing artifact/run-ID
capture envelope; instrumented and production programs remain separate cohorts.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from benchmark_sweep_lib.execution import run_command, validate_paths
from observation_cohorts import aggregate_captures
from benchmark_schema.wide_magnitude import MODULI
from benchmark_references import DEFAULT_INDEX, DEFAULT_CACHE, prepare as prepare_references


def reference_cache_supported(arguments):
    def option(flag, default):
        return arguments[arguments.index(flag) + 1] if flag in arguments else default
    return (option("--semantics", "bounded-i64") in ("bounded-i64", "bounded-u64") and
        all(option(flag, "1") == "1" for flag in ("--residue-chain-length", "--host-api-batch-size", "--grouped-dispatch")) and
        not any(flag in arguments for flag in ("--native-to-rns-bridge", "--vector-to-rns-chain",
            "--force-native-to-rns-bridge", "--vector-native-to-rns-chain", "--vector-to-rns-chain-host-repack-control",
            "--vector-native-to-rns-chain-host-repack-control", "--transient-uniform-small-inputs",
            "--transient-uniform-small-i8-inputs", "--residue-channel-fusion", "--streaming-overlap", "--hip-graph-replay")))


def validate_initial_plan_capacity(arguments, name):
    """Check the benchmark's full-native-type planning envelope before any run.

    The ordinary CLI creates its plan before binding actual input values. A
    small magnitude profile or scalar-first proof does not relax that initial
    type-wide capacity requirement; explicit value-bound owners are separate.
    """
    def option(flag, default=None):
        if flag not in arguments:
            return default
        if arguments.count(flag) != 1:
            raise ValueError(f"{name}: duplicate {flag}")
        index = arguments.index(flag)
        if index + 1 >= len(arguments):
            raise ValueError(f"{name}: missing {flag} value")
        return arguments[index + 1]
    semantics = option("--semantics")
    if semantics not in ("exact-wide-signed", "exact-wide-unsigned"):
        return
    prefix = int(option("--max-prefix", "20"))
    k = int(option("--k", "64"))
    if not 1 <= prefix <= len(MODULI) or k < 1:
        raise ValueError(f"{name}: invalid exact-wide planning dimensions/prefix")
    shift = 127 if semantics == "exact-wide-signed" else 128
    if math.prod(MODULI[:prefix]) <= (k << shift):
        raise ValueError(f"{name}: prefix {prefix} cannot admit the initial full-native-type exact-wide plan at K={k}")


def commands(binary, cases, runs, seed, output, *, first_call=False, reference_directory=None, verify_native_output=False):
    if not 2 <= runs <= 100: raise ValueError("need 2..100 independent processes per case/mode")
    result=[]
    names=set()
    for case in cases:
        name=case["name"]
        if not isinstance(name,str) or not name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in name) or name in names:
            raise ValueError("unique simple case name required")
        names.add(name)
        arguments=case["arguments"]
        if not isinstance(arguments,list) or not all(isinstance(x,str) for x in arguments): raise ValueError("arguments list")
        validate_initial_plan_capacity(arguments, name)
        if any(x.startswith(("--host-timing","--sustained-","--warmups","--repeats","--write-autotune","--reference-cache","--populate-reference","--verify-native-output")) for x in arguments):
            raise ValueError("collector owns timing/sampling controls and never writes a cache")
        first=int(hashlib.sha256(f"{seed}:{name}".encode()).hexdigest(),16)&1
        for run in range(runs):
            modes=("production",) if first_call else (("instrumented","production") if (first+run)%2==0 else ("production","instrumented"))
            for position,mode in enumerate(modes):
                command=[str(binary),*arguments,"--warmups","0" if first_call else "3","--repeats","1" if first_call else "9","--host-timing",mode]
                if reference_directory is not None and reference_cache_supported(arguments):
                    command += ["--reference-cache-dir", str(reference_directory)]
                elif verify_native_output and reference_cache_supported(arguments):
                    command += ["--verify-native-output"]
                if not first_call and mode=="production" and "--streaming-overlap" not in arguments:command += ["--sustained-batches","3","--sustained-min-calls","16","--sustained-max-calls","4096","--sustained-min-ms","250"]
                result.append(dict(case=name,run=run,position=position,mode=mode,command=command,
                                   output=str(output/f"{name}-run{run:03d}-{mode}.json")))
    # Interleave physical variants by independent run; reverse case order every
    # other run. This keeps per-block/per-call controls from occupying disjoint
    # time windows while preserving each case's seeded AB/BA instrumentation pair.
    order={case["name"]:i for i,case in enumerate(cases)}
    result.sort(key=lambda e:(e["run"],order[e["case"]] if e["run"]%2==0 else -order[e["case"]],e["position"]))
    return result


def first_call_setup(capture):
    """Sum only disjoint measured setup and one completed call after context creation."""
    if (type(capture.get('warmups')) is not int or capture['warmups']!=0 or
            type(capture.get('repeats')) is not int or capture['repeats']!=1 or
            (capture.get('timing_metadata') or {}).get('host_instrumentation_state')!='hooks_disabled'):
        raise ValueError('first-call evidence requires explicit hooks-disabled zero-warmup one-call capture')
    values={}
    for phase in ('planning','matrix_alloc','end_to_end'):
        samples=(capture.get('raw_timings_us') or {}).get(phase)
        if (not isinstance(samples,list) or len(samples)!=1 or type(samples[0]) not in (int,float) or
                not math.isfinite(samples[0]) or samples[0]<0 or (phase=='end_to_end' and samples[0]==0)):
            raise ValueError('first-call setup requires one measured planning/allocation/completed-call sample')
        values[phase+'_us']=samples[0]
    return dict(**values,setup_plus_first_completed_call_us=sum(values.values()),
        scope='after_context_creation_planning_includes_workspace_then_matrix_allocation_then_completed_call',
        context_creation_and_input_generation_included=False,independent_oracle_included=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary",type=Path,required=True);parser.add_argument("--cases",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True);parser.add_argument("--runs",type=int,default=2)
    parser.add_argument("--order-seed",type=int,default=1);parser.add_argument("--timeout-seconds",type=float,default=600)
    parser.add_argument("--plan-only",action="store_true")
    parser.add_argument("--reference-cache", action=argparse.BooleanOptionalAction, default=True,
                        help="reuse repository exact fixtures for supported native products; cache misses compute a full reference")
    parser.add_argument("--reference-cache-index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--first-call",action="store_true",help="fresh-process hook-disabled first invocation; zero warmups/one sample, no sustained bursts")
    args=parser.parse_args()
    case_bytes=args.cases.read_bytes()
    collector_identity=dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        case_manifest_sha256=hashlib.sha256(case_bytes).hexdigest())
    reference_directory = DEFAULT_CACHE if args.reference_cache else None
    reference_receipt = None
    if reference_directory is not None and not args.plan_only:
        reference_receipt = prepare_references(args.reference_cache_index, reference_directory)
    plan=commands(args.binary,json.loads(case_bytes)["cases"],args.runs,args.order_seed,args.output,first_call=args.first_call,
                  reference_directory=reference_directory, verify_native_output=True)
    args.output.mkdir(parents=True,exist_ok=True)
    order="reversed_case_order_by_independent_run_production_first_invocation_only" if args.first_call else "seeded_AB_BA_instrumentation_and_reversed_case_order_by_independent_run"
    (args.output/"collection-plan.json").write_text(json.dumps(dict(schema="public_calibration_plan_v1",order_policy=order,sampling="first_call" if args.first_call else "steady",collector_identity=collector_identity,reference_fixtures=reference_receipt,commands=plan,promotion_eligible=False),indent=2)+"\n")
    if args.plan_only:return 0
    captures=[]
    for entry in plan:
        output=Path(entry["output"])
        if output.exists():raise ValueError("refusing to overwrite an independent observation")
        if not run_command(entry["command"],output,args.timeout_seconds):raise RuntimeError(f"capture failed: {output}")
        captures.extend(validate_paths([output]))
    # Preserve all observations and ranges; no fixed-order overhead subtraction.
    views=aggregate_captures(captures)
    report=dict(schema="public_calibration_observations_v1",observation_views=views,collector_identity=collector_identity,
                overhead_inference="fresh_process_first_call_not_steady_state" if args.first_call else "counterbalanced_pair_order_recorded_no_causal_overhead_or_speedup_claim",
                promotion_eligible=False)
    if args.first_call:
        report['sampling']='first_call_after_context_creation_not_steady_state'
        report['first_call_observations']=[dict(case=entry['case'],run=entry['run'],execution_id=capture['execution_id'],
            capture_path=entry['output'],measurements=first_call_setup(capture)) for entry,capture in zip(plan,captures)]
    (args.output/"public-calibration-report.json").write_text(json.dumps(report,indent=2)+"\n")
    return 0

if __name__=="__main__":raise SystemExit(main())
