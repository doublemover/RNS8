"""Counterbalanced independent public API observations; no installation/promotion.

This collector executes only when explicitly invoked. Plan-only writes commands
without starting a benchmark. Every output gets the existing artifact/run-ID
capture envelope; instrumented and production programs remain separate cohorts.
"""
import argparse
import hashlib
import json
from pathlib import Path
from benchmark_sweep_lib.execution import run_command, validate_paths
from observation_cohorts import aggregate_captures


def commands(binary, cases, runs, seed, output):
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
        if any(x.startswith(("--host-timing","--sustained-","--warmups","--repeats","--write-autotune")) for x in arguments):
            raise ValueError("collector owns timing/sampling controls and never writes a cache")
        first=int(hashlib.sha256(f"{seed}:{name}".encode()).hexdigest(),16)&1
        for run in range(runs):
            modes=("instrumented","production") if (first+run)%2==0 else ("production","instrumented")
            for position,mode in enumerate(modes):
                command=[str(binary),*arguments,"--warmups","3","--repeats","9","--host-timing",mode]
                if mode=="production" and "--streaming-overlap" not in arguments:command += ["--sustained-batches","3","--sustained-min-calls","16","--sustained-max-calls","4096","--sustained-min-ms","250"]
                result.append(dict(case=name,run=run,position=position,mode=mode,command=command,
                                   output=str(output/f"{name}-run{run:03d}-{mode}.json")))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary",type=Path,required=True);parser.add_argument("--cases",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True);parser.add_argument("--runs",type=int,default=2)
    parser.add_argument("--order-seed",type=int,default=1);parser.add_argument("--timeout-seconds",type=float,default=600)
    parser.add_argument("--plan-only",action="store_true");args=parser.parse_args()
    plan=commands(args.binary,json.loads(args.cases.read_text())["cases"],args.runs,args.order_seed,args.output)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/"collection-plan.json").write_text(json.dumps(dict(schema="public_calibration_plan_v1",order_policy="seeded_AB_BA_by_independent_process_pair",commands=plan,promotion_eligible=False),indent=2)+"\n")
    if args.plan_only:return 0
    captures=[]
    for entry in plan:
        output=Path(entry["output"])
        if output.exists():raise ValueError("refusing to overwrite an independent observation")
        if not run_command(entry["command"],output,args.timeout_seconds):raise RuntimeError(f"capture failed: {output}")
        captures.extend(validate_paths([output]))
    # Preserve all observations and ranges; no fixed-order overhead subtraction.
    views=aggregate_captures(captures)
    report=dict(schema="public_calibration_observations_v1",observation_views=views,
                overhead_inference="counterbalanced_pair_order_recorded_no_causal_overhead_or_speedup_claim",
                promotion_eligible=False)
    (args.output/"public-calibration-report.json").write_text(json.dumps(report,indent=2)+"\n")
    return 0

if __name__=="__main__":raise SystemExit(main())
