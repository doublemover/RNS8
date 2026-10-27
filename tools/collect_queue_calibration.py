"""Collect matched private queue schedules; never installs or promotes a policy."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import uuid
from benchmark_sweep_lib.execution import run_command
from build_manifest import manifest_provenance_complete
from collect_native_calibration import strict_json

MODES = ("explicit", "task-major", "fair", "ordinary")


def validate_case(case):
    keys = {"schema", "workers", "chunk", "reuse", "repeats", "warmups", "metadata_budget_mib", "encoded_b", "mutate_b", "tasks"}
    if not isinstance(case, dict) or set(case) != keys or case["schema"] != "native_queue_case_v1":
        raise ValueError("exact native queue case ABI required")
    limits = {"workers": (1, 1024), "chunk": (1, 32), "reuse": (1, 256), "repeats": (3, 1000), "warmups": (1, 1000), "metadata_budget_mib": (1, 1024)}
    for key, (low, high) in limits.items():
        if type(case[key]) is not int or not low <= case[key] <= high: raise ValueError(f"invalid {key}")
    if any(type(case[k]) is not bool for k in ("encoded_b", "mutate_b")): raise ValueError("explicit boolean representation/mutation controls required")
    tasks = case["tasks"]
    if not isinstance(tasks, list) or not 1 <= len(tasks) <= 4096: raise ValueError("task count")
    work = 0; payload = 0
    for task in tasks:
        if not isinstance(task, dict) or set(task) != {"m", "n", "k", "prefix", "group", "unsigned"}: raise ValueError("task ABI")
        for key, low, high in (("m",1,1024),("n",1,1024),("k",1,65537),("prefix",1,20)):
            if type(task[key]) is not int or not low <= task[key] <= high: raise ValueError(f"task {key}")
        if type(task["group"]) is not int or task["group"] not in (1,2,4) or type(task["unsigned"]) is not bool: raise ValueError("task arithmetic contract")
        m,n,k,p=(task[x] for x in ("m","n","k","prefix"))
        work += m*n*k
        payload += 8*(m*(k+1)+k*(n+2))+(p+1)*m*(n+3)+int(case["encoded_b"])*p*n*((k+15)//16)*16+4
    if work > 134217728 or payload > 536870912: raise ValueError("bounded reference/payload envelope")
    return case


def commands(binary, cases, output, runs, seed, collection_id):
    if type(runs) is not int or not 2 <= runs <= 100: raise ValueError("2..100 independent processes")
    plan = []
    for path in cases:
        case = validate_case(strict_json(path.read_text()))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        offset = int(hashlib.sha256(f"{seed}:{digest}".encode()).hexdigest(),16)%4
        for run in range(runs):
            # Rotating/reversing order, not a fixed candidate-favored sequence.
            order = list(MODES); shift=(offset+run)%4; order=order[shift:]+order[:shift]
            if run%2: order.reverse()
            for position,mode in enumerate(order):
                run_id=f"{collection_id}-{digest[:12]}-{run}-{mode}"
                plan.append(dict(command=[str(binary),str(path),mode,run_id],case=case,case_sha256=digest,
                                 mode=mode,run=run,position=position,run_id=run_id,
                                 output=str(output/f"{run_id}.json")))
    return plan


def validate_capture(capture, entry):
    if capture.get("schema") != "rns8_native_queue_microcase_v1" or capture.get("promotion_eligible") is not False: raise ValueError("queue capture ABI")
    for key,want in (("case",entry["case"]),("run_id",entry["run_id"]),("schedule",entry["mode"]),
                     ("host_instrumentation_state","hooks_disabled"),("exact_reference_passed",True),("oracle_inside_production_interval",False)):
        if capture.get(key)!=want: raise ValueError(f"queue capture {key} mismatch")
    if not manifest_provenance_complete(capture.get("build_manifest")) or not capture.get("target"): raise ValueError("missing compiled artifact/target provenance")
    artifact=capture.get("artifact_capture",{})
    if artifact.get("unchanged_during_run") is not True or len(artifact.get("executable_sha256",""))!=64 or not capture.get("execution_id"): raise ValueError("artifact/execution receipt")
    samples=capture.get("samples",[])
    if len(samples)!=entry["case"]["repeats"]: raise ValueError("sample count")
    def positive(x): return type(x) in (int,float) and math.isfinite(x) and x>0
    if any(not positive(s.get("batch_ns")) or s.get("exact_reference_passed") is not True for s in samples): raise ValueError("invalid completed batch sample")
    tasks=entry["case"]["tasks"]
    counts=[((t["m"]+15)//16)*((t["n"]+15)//16)*((t["prefix"]+t["group"]-1)//t["group"]) for t in tasks]
    if capture.get("work_assignments_per_call")!=sum(counts): raise ValueError("work accounting")
    diagnostic=capture.get("completion_diagnostic")
    if entry["mode"]=="ordinary":
        if diagnostic is not None: raise ValueError("ordinary grid cannot claim queue telemetry")
    else:
        if not isinstance(diagnostic,dict) or diagnostic.get("completed_work")!=counts: raise ValueError("completion coverage")
        ticks=diagnostic.get("completion_ticks",[]);rate=diagnostic.get("wall_clock_rate_khz")
        if len(ticks)!=len(counts) or not positive(rate) or any(type(t) is not int or t<=0 for t in ticks): raise ValueError("wall clock receipt")
        micros=diagnostic.get("completion_us",[])
        if len(micros)!=len(ticks) or any(not math.isclose(v,t*1000/rate,rel_tol=1e-12) for v,t in zip(micros,ticks)): raise ValueError("wall clock units")
    return capture


def summarize(captures):
    groups={}; seen=set()
    for c in captures:
        if c["execution_id"] in seen: raise ValueError("duplicate execution is not independence")
        seen.add(c["execution_id"])
        # Different binaries/targets/recipes cannot be silently pooled.
        key=json.dumps([c["case"],c["schedule"],c["target"],c["build_manifest"],c["artifact_capture"]["executable_sha256"],
                        c["initial_input_checksum"],c["hip_runtime_version"],c["hip_driver_version"]],sort_keys=True)
        groups.setdefault(key,[]).append(c)
    result=[]
    for key,group in groups.items():
        per_process=[statistics.median(s["batch_ns"] for s in c["samples"]) for c in group]
        result.append(dict(identity=json.loads(key),independent_processes=len(group),process_medians_ns=per_process,
                           median_process_ns=statistics.median(per_process),minimum_process_ns=min(per_process),maximum_process_ns=max(per_process),
                           execution_ids=[c["execution_id"] for c in group],promotion_eligible=False,
                           inference="observations_only_no_speedup_or_production_fairness_claim"))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary",type=Path,required=True);parser.add_argument("--case",type=Path,action="append",required=True)
    parser.add_argument("--output",type=Path,required=True);parser.add_argument("--runs",type=int,default=4)
    parser.add_argument("--seed",type=int,default=38);parser.add_argument("--timeout-seconds",type=float,default=600)
    parser.add_argument("--plan-only",action="store_true");args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if any(args.output.iterdir()): raise ValueError("refusing to overwrite collection directory")
    plan=commands(args.binary.resolve(),[p.resolve() for p in args.case],args.output,args.runs,args.seed,uuid.uuid4().hex)
    (args.output/"collection-plan.json").write_text(json.dumps(dict(schema="native_queue_collection_v1",commands=plan,promotion_eligible=False),indent=2)+"\n")
    if args.plan_only:return 0
    captures=[]
    for entry in plan:
        path=Path(entry["command"][1]);output=Path(entry["output"])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=entry["case_sha256"]: raise ValueError("case file changed after planning")
        if output.exists() or not run_command(entry["command"],output,args.timeout_seconds): raise RuntimeError("capture failed/exists")
        if hashlib.sha256(path.read_bytes()).hexdigest()!=entry["case_sha256"]: raise ValueError("case file changed during execution")
        capture=validate_capture(strict_json(output.read_text()),entry)
        capture["case_artifact_sha256"]=entry["case_sha256"];capture["collection_position"]=entry["position"]
        output.write_text(json.dumps(capture,indent=2)+"\n");captures.append(capture)
    (args.output/"observations.json").write_text(json.dumps(dict(schema="native_queue_observations_v1",cohorts=summarize(captures),promotion_eligible=False),indent=2)+"\n")
    return 0

if __name__=="__main__":raise SystemExit(main())
