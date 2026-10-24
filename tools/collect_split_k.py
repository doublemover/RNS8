"""Collect public split-K ownership cohorts; independent exact replay, no promotion.

One child process is one independent observation. Preparation, first use, unchanged
reuse and repack/recreation are separate boundaries, never extrapolated timings.
"""
from __future__ import annotations
import argparse
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import re
import statistics
import subprocess
import sys
import time
import uuid
from build_manifest import manifest_provenance_complete
from collect_finite_verify import demand,integer,positive,same,strict_json,checksum,validate_provenance

MODES=("unsplit","full","window")
KEYS={"device","mode","m","n","k","domain","fragments","window","reuse"}
MASK=(1<<64)-1
BUDGET=512*1024*1024
XOR=0x0123456789abcdef


def validate_case(case):
    demand(type(case) is dict and set(case)==KEYS,"case requires exactly device/mode/m/n/k/domain/fragments/window/reuse")
    demand(integer(case["device"],0,63) and case["mode"] in MODES and case["domain"] in ("signed","unsigned"),"invalid explicit device/mode/signedness")
    demand(all(integer(case[key],1,512) for key in ("m","n")) and integer(case["k"],1,65537),"invalid shape")
    demand(integer(case["reuse"],1,256) and case["m"]*case["n"]*case["k"]*case["reuse"]<=134217728,"batch work exceeds bounded domain")
    f,w=case["fragments"],case["window"]
    demand(integer(f,1,4096) and integer(w,0,4096),"invalid fragment/window types or bounds")
    demand((f==1 and w==0) if case["mode"]=="unsplit" else
        f<=case["k"] and (case["k"]+f-1)//f<=65536 and (w==f if case["mode"]=="full" else 1<=w<f),"irrelevant or inadmissible fragments/window")
    demand(256*(case["m"]*case["k"]+case["k"]*case["n"])+256*case["m"]*case["n"]+16384+(resources(case) or {}).get("scratch_bytes",0)<=BUDGET,"payload envelope exceeds budget")
    return dict(case)


def load_cases(path):
    document=strict_json(Path(path).read_text(encoding="utf-8-sig"))
    demand(type(document) is dict and set(document)=={"schema","cases"} and document["schema"]=="rns8_split_k_cases_v1","invalid case manifest")
    cases=document["cases"];demand(type(cases) is list and 1<=len(cases)<=2048,"case count must be1..2048")
    cases=[validate_case(case) for case in cases];ids=[json.dumps(case,sort_keys=True) for case in cases]
    demand(len(ids)==len(set(ids)),"duplicate case definitions do not establish independent evidence")
    return cases


def resources(case):
    if case["mode"]=="unsplit":return None
    f,w=case["fragments"],case["window"];cells=case["m"]*case["n"]*20
    count=w;levels=0
    while count>1:count=(count+127)//128;levels+=1
    partial=cells*w;merge=cells*((w+127)//128);running=cells if w<f else 0
    return dict(fragments=f,fragment_window=w,window_count=(f+w-1)//w,partial_bytes=partial,merge_bytes=merge,
        running_bytes=running,scratch_bytes=partial+merge+running,merge_levels=levels)


def contract(case,run_id,repeats,warmups):
    return dict(schema="rns8_split_k_microcase_v1",promotion_eligible=False,run_id=run_id,**case,device_actual=case["device"],
        backend_requested="hip-direct",backend_actual="hip-direct",semantics="exact_wide_"+case["domain"],prefix=20,
        repeats=repeats,warmups=warmups,dataset_recipe="splitmix64_fullwidth_seed5117_xor_A0_generation2_v1",dataset_seed=5117,generation2_a0_xor=XOR,
        content_generations=None,generation_contract="opaque_public_owner_generation_binding_source_version_is_separate",split_resources=resources(case),
        reference_method="boost_int256_full_product_with_cpp_int_row_edge_audits",reference_buffers_released_before_timing=True,
        output_contract="three_little_endian_uint64_limbs_per_cell_twos_complement_for_signed",timer="std_chrono_steady_clock_nanoseconds",
        timed_scope="reuse_public_synchronous_compute_only_no_export_or_oracle",
        setup_scope="generation1_context_plan_workspace_matrices_pack_owner_generation2_sum_pack_and_checked_recreate_excludes_stale_probe",
        kernel_events=None,production_timing_hooks_enabled=False,observed_overlap=None,memory_budget_bytes=BUDGET,
        memory_exclusions="allocator_bookkeeping_spare_capacity_context_runtime_fixed_metadata_not_process_RSS_or_measured_peak",
        allocation_scope="queried_matrix_payload_and_cached_staging_plus_separate_split_scratch_not_peak",
        exact_full_output_and_padding_checked=True,verified_outputs_per_batch=1)


def words(case):
    state=5117;arrays=[]
    for cells in (case["m"]*case["k"],case["k"]*case["n"]):
        values=[]
        for i in range(cells):
            state=(state+0x9e3779b97f4a7c15)&MASK;z=state
            z=((z^(z>>30))*0xbf58476d1ce4e5b9)&MASK;z=((z^(z>>27))*0x94d049bb133111eb)&MASK;z^=z>>31
            if i%257==0:z=(1<<63) if case["domain"]=="signed" else MASK
            elif i%257==1:z=(1<<63)-1 if case["domain"]=="signed" else 0
            values.append(z)
        arrays.append(values)
    return arrays


def word_checksum(values):return checksum(b"".join((value&MASK).to_bytes(8,"little") for value in values))


@lru_cache(maxsize=8)
def reference(m,n,k,domain,generation):
    case=dict(m=m,n=n,k=k,domain=domain);a,b=words(case)
    if generation==2:a[0]^=XOR
    def value(word):return word-(1<<64) if domain=="signed" and word>>63 else word
    aa=[value(word) for word in a];bt=[[value(b[t*n+col]) for t in range(k)] for col in range(n)]
    output=[]
    for row in range(m):
        for col in range(n):
            total=sum(aa[row*k+t]*bt[col][t] for t in range(k))
            demand(abs(total)<(1<<145),"independent exact result exceeds admitted full-expression bound")
            output.extend((total>>(64*limb))&MASK for limb in range(3))
    # Cache hashes only, not potentially large operand/output arrays.
    return word_checksum(a),word_checksum(b),word_checksum(output)


def validate_allocations(values,case,envelope):
    fields={"logical_primary_bytes","host_payload_capacity_bytes","device_payload_capacity_bytes","device_upload_staging_bytes",
        "device_export_staging_bytes","device_status_and_schedule_bytes","device_total_capacity_bytes"}
    demand(type(values) is list and len(values)==3,"missing A/B/C allocation boundaries")
    total=0
    for record,cells in zip(values,(case["m"]*case["k"],case["k"]*case["n"],case["m"]*case["n"])):
        demand(type(record) is dict and set(record)==fields and all(integer(v,0,BUDGET) for v in record.values()),"invalid allocation fields")
        demand(record["logical_primary_bytes"]==cells*20 and record["device_payload_capacity_bytes"]>=cells*20
            and record["device_total_capacity_bytes"]==sum(record[key] for key in ("device_payload_capacity_bytes","device_upload_staging_bytes","device_export_staging_bytes","device_status_and_schedule_bytes")),"incoherent matrix capacities")
        total+=record["host_payload_capacity_bytes"]+record["device_total_capacity_bytes"]
    demand(total<=envelope,"matrix capacity exceeds logical budget")
    return total


def validate_capture(capture,case,run_id,repeats,warmups):
    demand(type(capture) is dict,"capture must be object")
    for key,value in contract(case,run_id,repeats,warmups).items():demand(key in capture and same(capture[key],value),f"metadata mismatch: {key}")
    for key in ("device_name","selected_unsplit_kernel"):demand(type(capture.get(key)) is str and bool(capture[key].strip()),f"missing identity: {key}")
    demand(type(capture.get("target")) is str and re.fullmatch(r"gfx[0-9a-f]+(?::[A-Za-z0-9_+:-]+)?",capture["target"]),"invalid target")
    for key in ("hip_runtime_version","hip_driver_version","clock_period_numerator","clock_period_denominator"):demand(integer(capture.get(key),1,(1<<63)-1),f"invalid runtime/timer: {key}")
    demand(integer(capture.get("owner_release_ns"),0 if case["mode"]=="unsplit" else 1,0 if case["mode"]=="unsplit" else (1<<63)-1),"invalid owner release timing")
    ids=capture.get("source_instance_ids");out=capture.get("output_instance_id")
    demand(type(ids) is list and len(ids)==2 and all(integer(v,1,MASK) for v in ids) and integer(out,1,MASK) and len(set([*ids,out]))==3,"invalid matrix instance identities")
    ab=case["m"]*case["k"]+case["k"]*case["n"];mn=case["m"]*case["n"]
    width=capture.get("reference_fixed_operand_object_bytes");work=capture.get("plan_workspace_bytes")
    demand(integer(width,32,128) and same(capture.get("reference_operand_storage_bytes"),ab*width) and integer(work,0,BUDGET),"invalid reference/workspace memory")
    scratch=(resources(case) or {}).get("scratch_bytes",0)
    expected=max(8*ab+ab*width+24*mn+2048,256*ab+256*mn+scratch+16384+work)
    demand(integer(capture.get("logical_live_payload_envelope_bytes"),1,BUDGET) and capture["logical_live_payload_envelope_bytes"]==expected,"invalid aggregate payload envelope")
    demand(validate_allocations(capture.get("matrix_allocations_at_setup"),case,expected)+work+scratch<=expected,"setup allocations exceed aggregate envelope")
    phases=capture.get("phases");demand(type(phases) is list and len(phases)==2,"two input generations required")
    for generation,phase in enumerate(phases,1):
        fields={"generation","source_versions","binding_action","stale_owner_status","stale_owner_preserved_output","a_checksum","b_checksum","output_checksum",
            "oracle_ns","phase_setup_ns","changed_input_pack_ns","owner_prepare_or_recreate_ns","first_batch_ns","setup_plus_first_batch_ns",
            "first_batch_exact_reference_passed","samples","median_batch_ns","gemm_calls_per_second","matrix_allocations_after_audit","sustained"}
        demand(type(phase) is dict and set(phase)==fields and same(phase.get("generation"),generation) and same(phase.get("source_versions"),[generation,generation]),"phase fields or input versions mismatch")
        split=case["mode"]!="unsplit";changed=generation==2
        action=("repack_destroy_recreate" if split else "repack") if changed else ("initial_create" if split else "initial_pack")
        demand(phase.get("binding_action")==action and same(phase.get("stale_owner_status"),"RNS8_INVALID_ARGUMENT" if split and changed else None)
            and same(phase.get("stale_owner_preserved_output"),True if split and changed else None),"stale binding admission/preservation mismatch")
        demand(tuple(phase.get(key) for key in ("a_checksum","b_checksum","output_checksum"))==reference(*(case[key] for key in ("m","n","k","domain")),generation),"independent complete input/product checksum mismatch")
        for key in ("oracle_ns","phase_setup_ns","first_batch_ns","setup_plus_first_batch_ns"):demand(integer(phase.get(key),1,(1<<63)-1),f"invalid phase timing: {key}")
        pack,prepare=phase.get("changed_input_pack_ns"),phase.get("owner_prepare_or_recreate_ns")
        demand(integer(pack,1 if changed else 0,(1<<63)-1 if changed else 0) and integer(prepare,1 if split else 0,(1<<63)-1 if split else 0),"invalid pack/preparation phase")
        demand((phase["phase_setup_ns"]==pack+prepare if changed else phase["phase_setup_ns"]>=prepare)
            and phase["setup_plus_first_batch_ns"]==phase["phase_setup_ns"]+phase["first_batch_ns"] and phase.get("first_batch_exact_reference_passed") is True,"incoherent setup/first-use boundary")
        samples=phase.get("samples");demand(type(samples) is list and len(samples)==repeats,"sample count mismatch")
        for sample in samples:demand(type(sample) is dict and set(sample)=={"wall_ns","exact_reference_passed"} and integer(sample["wall_ns"],1,(1<<63)-1) and sample["exact_reference_passed"] is True,"invalid wall sample/audit")
        median=statistics.median(sample["wall_ns"] for sample in samples)
        demand(positive(phase.get("median_batch_ns")) and phase["median_batch_ns"]==median and positive(phase.get("gemm_calls_per_second")) and math.isclose(phase["gemm_calls_per_second"],1e9*case["reuse"]/median,rel_tol=1e-12),"inconsistent median/GEMM rate")
        sustained=phase.get("sustained");demand(type(sustained) is dict,"missing sustained interval")
        calls,elapsed=sustained.get("calls"),sustained.get("elapsed_ns")
        demand(integer(calls,16,4096) and integer(elapsed,1,(1<<63)-1) and same(sustained.get("gemm_calls"),calls*case["reuse"])
            and same(sustained.get("minimum_calls"),16) and same(sustained.get("maximum_calls"),4096) and same(sustained.get("target_ns"),250000000)
            and sustained.get("independent_audit_inside_interval") is False and sustained.get("final_exact_reference_passed") is True
            and positive(sustained.get("mean_batch_ns")) and math.isclose(sustained["mean_batch_ns"]*calls,elapsed,rel_tol=1e-12)
            and ((elapsed>=250000000 and sustained.get("stop_reason")=="duration") or (elapsed<250000000 and calls==4096 and sustained.get("stop_reason")=="call_cap")),"invalid sustained timing/work/audit")
        demand(validate_allocations(phase.get("matrix_allocations_after_audit"),case,expected)+work+scratch<=expected,"final capacities exceed envelope")
    validate_provenance(capture.get("build_manifest"))
    return {key:capture[key] for key in ("build_manifest","device_actual","device_name","target","hip_runtime_version","hip_driver_version","reference_fixed_operand_object_bytes")}


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(observations):
    groups={}
    for observation in observations:
        case=observation["case"];key=tuple(case[field] for field in ("device","m","n","k","domain","reuse"))
        groups.setdefault(key,[]).append(observation)
    result=[]
    for key,group in groups.items():
        modes={item["case"]["mode"] for item in group}
        result.append(dict(matched_identity=dict(zip(("device","m","n","k","domain","reuse"),key)),
            complete_three_mode_cohort=set(MODES)<=modes,observed_modes=sorted(modes),
            observations=group,comparison_rule="compare observed phase-matched independent process values only; no reuse-count or break-even extrapolation"))
    return result


def collect(args):
    demand(integer(args.runs,1,100) and integer(args.repeats,1,1000) and integer(args.warmups,1,1000) and integer(args.timeout,1,86400) and type(args.order_seed) is int,"invalid collection controls")
    case_hash=digest(args.cases);cases=load_cases(args.cases);demand(digest(args.cases)==case_hash,"cases changed while loading")
    exe=Path(args.exe).resolve(strict=True);artifact=digest(exe)
    sources=[Path(__file__),Path(__file__).with_name("collect_finite_verify.py"),Path(__file__).with_name("build_manifest.py")]
    tools={path.name:digest(path) for path in sources};python_hash=digest(sys.executable)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True);campaign=str(uuid.uuid4())
    tasks=[(case,repeat) for repeat in range(args.runs) for case in cases];random.Random(args.order_seed).shuffle(tasks)
    cohort=None;observations=[]
    def unchanged():demand(digest(exe)==artifact and digest(args.cases)==case_hash and all(digest(path)==tools[path.name] for path in sources),"artifact/tools/cases changed during collection")
    for ordinal,(case,repeat) in enumerate(tasks):
        unchanged();run_id=str(uuid.uuid4());command=[str(exe),str(case["device"]),case["mode"],*(str(case[key]) for key in ("m","n","k")),case["domain"],*(str(case[key]) for key in ("fragments","window","reuse")),str(args.repeats),str(args.warmups),run_id]
        record=dict(schema="rns8_split_k_collection_v1",campaign_id=campaign,run_id=run_id,ordinal=ordinal,case=case,independent_repeat=repeat,
            order_seed=args.order_seed,command=command,started_unix_ns=time.time_ns(),artifact_sha256=artifact,cases_sha256=case_hash,
            collector_sha256=tools[Path(__file__).name],tool_dependencies_sha256=tools,python_executable=sys.executable,python_version=platform.python_version(),python_sha256=python_hash,
            validated=False,promotion_eligible=False,provenance_complete=False,exit_code=None,stdout="",stderr="")
        path=output/f"{run_id}.json"
        try:
            try:
                process=subprocess.run(command,text=True,encoding="utf-8",errors="strict",capture_output=True,timeout=args.timeout,check=False)
                record.update(exit_code=process.returncode,stdout=process.stdout,stderr=process.stderr)
            except subprocess.TimeoutExpired as error:
                record.update(stdout=(error.stdout.decode("utf-8",errors="replace") if isinstance(error.stdout,bytes) else error.stdout) or "",
                    stderr=(error.stderr.decode("utf-8",errors="replace") if isinstance(error.stderr,bytes) else error.stderr) or "",timeout_seconds=args.timeout)
                raise ValueError("split-K process timed out") from error
            demand(process.returncode==0,"split-K process failed")
            capture=strict_json(process.stdout);identity=validate_capture(capture,case,run_id,args.repeats,args.warmups)
            if cohort is None:cohort=identity
            demand(same(identity,cohort),"compiled build/device/runtime cohort mismatch")
            unchanged();record.update(validated=True,provenance_complete=manifest_provenance_complete(capture["build_manifest"]))
            observations.append(dict(case=case,run_id=run_id,independent_repeat=repeat,
                scratch_bytes=(resources(case) or {}).get("scratch_bytes",0),
                phases=[{key:phase[key] for key in ("generation","binding_action","phase_setup_ns","first_batch_ns","setup_plus_first_batch_ns","median_batch_ns","sustained")} for phase in capture["phases"]]))
        except Exception as error:
            record["failure"]=str(error)
            with path.open("x",encoding="utf-8") as stream:json.dump(record,stream,indent=2,allow_nan=False)
            raise
        with path.open("x",encoding="utf-8") as stream:json.dump(record,stream,indent=2,allow_nan=False)
    summary=dict(schema="rns8_split_k_campaign_v1",campaign_id=campaign,artifact_sha256=artifact,cases_sha256=case_hash,tool_dependencies_sha256=tools,
        cohort=cohort,independent_processes=len(observations),promotion_eligible=False,provenance_complete=manifest_provenance_complete(cohort["build_manifest"]),matched_cohorts=summarize(observations))
    with (output/f"{campaign}.summary.json").open("x",encoding="utf-8") as stream:json.dump(summary,stream,indent=2,allow_nan=False)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("exe","cases","output"):parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--runs",type=int,default=5);parser.add_argument("--repeats",type=int,default=9);parser.add_argument("--warmups",type=int,default=3)
    parser.add_argument("--timeout",type=int,default=1800);parser.add_argument("--order-seed",type=int,default=5117)
    args=parser.parse_args()
    try:summary=collect(args)
    except (ValueError,OSError) as error:parser.error(str(error))
    print(f"Collected {summary['independent_processes']} public split-K processes; qualification remains pending");return 0

if __name__=="__main__":raise SystemExit(main())
