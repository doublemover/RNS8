"""Collect public serial-panel versus in-core native host-to-host controls.

One process is one independent observation. All capacities are queried payloads,
all modeled transfer bytes are absent, and no overlap or qualification is inferred.
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

MODES=("incore","stream")
KEYS={"device","mode","m","n","k","domain","bits","prefix","panel_m","panel_n","panel_k","device_budget_bytes","host_budget_bytes","retained_b_budget_bytes","reuse"}
MATCH=("device","m","n","k","domain","bits","prefix","reuse")
MASK=(1<<64)-1
BUDGET=512*1024*1024
# Same public default ladder, independently enumerated for admission replay.
MODULI=(256,255,253,251,247,239,233,229,227,223,217,211,199,197,193,191,181,179,173,167)


def resources(case):
    if case["mode"]=="incore":return None
    m,n,k=(min(case[axis],case["panel_"+axis]) for axis in ("m","n","k"));p=case["prefix"]
    align=lambda size:(size+255)//256*256
    slots=(m*k*p,k*n*p,m*n*p,m*n*p,8*m*k,8*k*n,8*m*n,4)
    result=dict(device_bytes=sum(map(align,slots)),host_bytes=8*(m*k+k*n+m*n),minimum_device_bytes=2048,minimum_host_bytes=24,
        source_a_span_bytes=8*((case["m"]-1)*(case["k"]+3)+case["k"]),
        source_b_span_bytes=8*((case["k"]-1)*(case["n"]+3)+case["n"]),
        destination_span_bytes=8*((case["m"]-1)*(case["n"]+3)+case["n"]),
        output_tiles=((case["m"]+m-1)//m)*((case["n"]+n-1)//n),k_panels_per_tile=(case["k"]+k-1)//k)
    stride=align(k*n*p);panels=((case["n"]+n-1)//n)*result["k_panels_per_tile"]
    entries=max(0,min(panels,case["retained_b_budget_bytes"]//stride,
        (case["device_budget_bytes"]-result["device_bytes"])//stride,
        (case["host_budget_bytes"]-result["host_bytes"])//32))
    if entries<=panels//2:entries=0
    singleton=max(0,2*entries-panels) if entries else 0
    row_tiles=(case["m"]+m-1)//m
    result.update(retained_b_entries=entries,retained_b_device_bytes=entries*stride,
        cold_b_panel_loads=panels+(row_tiles-1)*(panels-singleton),warm_b_panel_loads=row_tiles*(panels-singleton))
    result["device_bytes"]+=entries*stride;result["host_bytes"]+=entries*32
    return result


def memory(case,workspace=0):
    m,n,k=(case[key] for key in ("m","n","k"));ab=m*k+k*n;mn=m*n
    caller=8*(m*(k+3)+k*(n+3)+m*(n+3));r=resources(case) or {}
    return caller,512*(ab+mn)+caller+8*mn+16384+r.get("device_bytes",0)+r.get("host_bytes",0)+workspace


def validate_case(case):
    demand(type(case) is dict and set(case)==KEYS,"case requires exactly explicit device/mode/shape/domain/bits/prefix/panels/budgets/reuse")
    demand(integer(case["device"],0,63) and case["mode"] in MODES and case["domain"] in ("signed","unsigned"),"invalid explicit device/mode/domain")
    demand(all(integer(case[key],1,512) for key in ("m","n")) and integer(case["k"],1,65537),"invalid shape")
    demand(integer(case["bits"],1,31) and integer(case["prefix"],1,20) and integer(case["reuse"],1,256),"invalid bits/prefix/reuse")
    demand(case["m"]*case["n"]*case["k"]*case["reuse"]<=134217728,"batch reference work exceeds limit")
    maximum=(1<<case["bits"])-1;bound=case["k"]*maximum*maximum
    demand(bound<=(1<<(63 if case["domain"]=="signed" else 64))-1,"full-expression bound exceeds native output")
    product=math.prod(MODULI[:case["prefix"]])
    demand((2*bound if case["domain"]=="signed" else bound)<product,"CRT prefix does not admit bound")
    demand(all(integer(case[key],0,65537) for key in ("panel_m","panel_n","panel_k")) and all(integer(case[key],0,BUDGET) for key in ("device_budget_bytes","host_budget_bytes","retained_b_budget_bytes")),"invalid panel/budget types")
    if case["mode"]=="incore":demand(all(case[key]==0 for key in ("panel_m","panel_n","panel_k","device_budget_bytes","host_budget_bytes","retained_b_budget_bytes")),"irrelevant panel controls for in-core")
    else:
        demand(all(case[key]>0 for key in ("panel_m","panel_n","panel_k","device_budget_bytes","host_budget_bytes")),"stream requires positive panels/budgets")
        r=resources(case);demand(r["device_bytes"]<=case["device_budget_bytes"] and r["host_bytes"]<=case["host_budget_bytes"],"explicit panel budgets too small")
    demand(memory(case)[1]<=BUDGET,"aggregate logical payload exceeds512MiB")
    return dict(case)


def load_cases(path):
    doc=strict_json(Path(path).read_text(encoding="utf-8-sig"))
    demand(type(doc) is dict and set(doc)=={"schema","cases"} and doc["schema"]=="rns8_stream_panel_cases_v2","invalid manifest schema")
    demand(type(doc["cases"]) is list and 1<=len(doc["cases"])<=2048,"invalid manifest case count")
    cases=[validate_case(case) for case in doc["cases"]];ids=[json.dumps(case,sort_keys=True) for case in cases]
    demand(len(ids)==len(set(ids)),"duplicate cases do not establish independent evidence")
    return cases


def contract(case,run_id,repeats,warmups):
    panel=case["mode"]=="stream";m,n,k=(case[key] for key in ("m","n","k"));maximum=(1<<case["bits"])-1
    return dict(schema="rns8_stream_panel_microcase_v2",promotion_eligible=False,run_id=run_id,**case,device_actual=case["device"],
        backend_requested="hip-direct",backend_actual="hip-direct",repeats=repeats,warmups=warmups,semantics="bounded_i64" if case["domain"]=="signed" else "bounded_u64",
        max_abs_a=maximum,max_abs_b=maximum,full_expression_bound=k*maximum*maximum,lda=k+3,ldb=n+3,ldc=n+3,
        dataset_recipe="splitmix64_seed6811_bounded_native_A0_B0_max_then_A0_zero_v1",dataset_seed=6811,source_instance_labels=[101,202],
        source_label_scope="dataset_labels_passed_to_stream_owner_only_not_matrix_instances_or_cache_hits",incore_pack_source_version=None if panel else 0,input_scan="public_owner_full_input_scan_each_call" if panel else "recipe_bound_independently_verified_before_timing",
        reference_method="boost_int256_full_product_with_cpp_int_row_edge_audits",reference_buffers_released_before_timing=True,
        output_contract="native64_word_per_cell_twos_complement_for_signed",exact_full_output_and_padding_checked=True,
        timer="std_chrono_steady_clock_nanoseconds",timed_scope="reuse_native_host_inputs_to_checked_native_host_output_no_independent_oracle",
        setup_scope="query_then_context_and_owner_or_plan_workspace_matrices_no_input_conversion",kernel_events=None,production_timing_hooks_enabled=False,observed_overlap=None,
        panel_resources=resources(case),budget_preflight_receipts=dict(cache_disabled_for_preflight=True,baseline_device_minus_one="RNS8_WORKSPACE_TOO_SMALL",baseline_host_minus_one="RNS8_WORKSPACE_TOO_SMALL",queried_resources_preserved=True) if panel else None,
        caller_native_allocation_bytes=memory(case)[0],reference_output_bytes=8*m*n,memory_budget_bytes=BUDGET,full_incore_residue_payload_bytes=(m*k+k*n+m*n)*case["prefix"],
        capacity_scope="public_query_required_payload_and_staging_not_measured_peak_or_bus_traffic",
        memory_exclusions="allocator_bookkeeping_spare_capacity_context_runtime_fixed_metadata_not_process_RSS_or_measured_peak")


def word_checksum(values):return checksum(b"".join((value&MASK).to_bytes(8,"little") for value in values))


def words(m,n,k,domain,bits):
    state=6811;maximum=(1<<bits)-1;arrays=[]
    for cells in (m*k,k*n):
        values=[]
        for _ in range(cells):
            state=(state+0x9e3779b97f4a7c15)&MASK;z=state;z=((z^(z>>30))*0xbf58476d1ce4e5b9)&MASK;z=((z^(z>>27))*0x94d049bb133111eb)&MASK;z^=z>>31
            values.append(z%(2*maximum+1)-maximum if domain=="signed" else z%(maximum+1))
        values[0]=maximum;arrays.append(values)
    return arrays


@lru_cache(maxsize=8)
def reference(m,n,k,domain,bits,generation):
    a,b=words(m,n,k,domain,bits)
    if generation==2:a[0]=0
    bt=[[b[q*n+col] for q in range(k)] for col in range(n)]
    out=[sum(a[row*k+q]*bt[col][q] for q in range(k)) for row in range(m) for col in range(n)]
    maximum=(1<<bits)-1;demand(all(abs(v)<=k*maximum*maximum for v in out),"exact independent output exceeds bound")
    return word_checksum(a),word_checksum(b),word_checksum(out)


def coverage(record,case,epoch):
    if case["mode"]=="incore":demand(record is None,"in-core cannot claim panel coverage");return
    r=resources(case);expected=dict(epoch=epoch,completed_tiles=r["output_tiles"],completed_panels=r["output_tiles"]*r["k_panels_per_tile"],output_started=True,output_valid=True,quarantined=False)
    loads=r["cold_b_panel_loads"] if epoch==1 else r["warm_b_panel_loads"]
    expected.update(b_panel_loads=loads,b_panel_cache_hits=expected["completed_panels"]-loads)
    demand(type(record) is dict and same(record,expected),"incomplete observed panel coverage/epoch/cache")


def allocations(values,case,envelope,workspace):
    demand(type(values) is list and len(values)==(3 if case["mode"]=="incore" else 0),"invalid matrix allocation boundaries")
    fields={"logical_primary_bytes","host_payload_capacity_bytes","device_payload_capacity_bytes","device_upload_staging_bytes","device_export_staging_bytes","device_status_and_schedule_bytes","device_total_capacity_bytes"}
    r=resources(case) or {};total=memory(case)[0]+8*case["m"]*case["n"]+workspace+r.get("device_bytes",0)+r.get("host_bytes",0)
    for item,cells in zip(values,(case["m"]*case["k"],case["k"]*case["n"],case["m"]*case["n"])):
        demand(type(item) is dict and set(item)==fields and all(integer(v,0,BUDGET) for v in item.values()),"invalid matrix allocation fields")
        demand(item["logical_primary_bytes"]==cells*case["prefix"] and item["device_payload_capacity_bytes"]>=cells*case["prefix"] and item["device_total_capacity_bytes"]==sum(item[key] for key in ("device_payload_capacity_bytes","device_upload_staging_bytes","device_export_staging_bytes","device_status_and_schedule_bytes")),"incoherent matrix capacities")
        total+=item["host_payload_capacity_bytes"]+item["device_total_capacity_bytes"]
    demand(total<=envelope,"retained allocations exceed logical envelope")


def validate_capture(capture,case,run_id,repeats,warmups):
    demand(type(capture) is dict,"capture must be object")
    required=contract(case,run_id,repeats,warmups)
    dynamic={"device_name","target","hip_runtime_version","hip_driver_version","build_manifest","reference_fixed_operand_object_bytes","reference_operand_storage_bytes",
        "clock_period_numerator","clock_period_denominator","selected_control","query_ns","initial_setup_ns","owner_prepare_ns","cold_setup_plus_first_batch_ns",
        "logical_live_payload_envelope_bytes","plan_workspace_bytes","matrix_allocations_after_final_audit","production_native_calls","out_of_bound_probe","recreation","owner_release_ns","phases"}
    demand(set(capture)==set(required)|dynamic,"unexpected or missing capture fields")
    for key,value in required.items():demand(same(capture[key],value),f"metadata mismatch: {key}")
    panel=case["mode"]=="stream";reuse=case["reuse"]
    demand(type(capture.get("device_name")) is str and capture["device_name"].strip() and type(capture.get("target")) is str and re.fullmatch(r"gfx[0-9a-f]+(?::[A-Za-z0-9_+:-]+)?",capture["target"]),"missing target/device identity")
    demand(type(capture.get("selected_control")) is str and bool(capture["selected_control"].strip()) and (not panel or capture["selected_control"]=="serial_panel_public_rns_axpby"),"missing or false control identity")
    for key in ("hip_runtime_version","hip_driver_version","clock_period_numerator","clock_period_denominator","initial_setup_ns"):demand(integer(capture.get(key),1,(1<<63)-1),f"invalid identity/timer: {key}")
    for key in ("query_ns","owner_prepare_ns","owner_release_ns"):demand(integer(capture.get(key),1 if panel else 0,(1<<63)-1 if panel else 0),f"invalid owner lifecycle timing: {key}")
    demand(capture["initial_setup_ns"]>=capture["owner_prepare_ns"],"owner preparation outside setup boundary")
    width=capture.get("reference_fixed_operand_object_bytes");workspace=capture.get("plan_workspace_bytes");ab=case["m"]*case["k"]+case["k"]*case["n"]
    demand(integer(width,32,128) and same(capture.get("reference_operand_storage_bytes"),ab*width) and integer(workspace,0,BUDGET) and (not panel or workspace==0),"invalid reference/workspace admission")
    envelope=memory(case,workspace)[1];demand(integer(capture.get("logical_live_payload_envelope_bytes"),1,BUDGET) and capture["logical_live_payload_envelope_bytes"]==envelope,"invalid aggregate memory model")
    allocations(capture.get("matrix_allocations_after_final_audit"),case,envelope,workspace)
    phases=capture.get("phases");demand(type(phases) is list and len(phases)==2,"two input generations required");epoch=0
    phase_fields={"generation","source_binding","binding_action","a_checksum","b_checksum","output_checksum","oracle_ns","first_batch_ns","first_coverage","first_exact_reference_passed","samples","median_batch_ns","gemm_calls_per_second","matrix_allocations_after_audit","sustained"}
    for gen,phase in enumerate(phases,1):
        demand(type(phase) is dict and set(phase)==phase_fields and same(phase.get("generation"),gen) and same(phase.get("source_binding"),[101,gen,202,1]) and phase.get("binding_action")==("initial_inputs" if gen==1 else "changed_A0_same_plan"),"phase fields or binding mismatch")
        demand(tuple(phase.get(key) for key in ("a_checksum","b_checksum","output_checksum"))==reference(*(case[key] for key in ("m","n","k","domain","bits")),gen),"independent full input/output checksum mismatch")
        for key in ("oracle_ns","first_batch_ns"):demand(integer(phase.get(key),1,(1<<63)-1),f"invalid phase timing: {key}")
        demand(phase.get("first_exact_reference_passed") is True,"first full output not audited");epoch+=reuse;coverage(phase.get("first_coverage"),case,epoch);epoch+=warmups*reuse
        samples=phase.get("samples");demand(type(samples) is list and len(samples)==repeats,"sample count mismatch")
        for sample in samples:
            demand(type(sample) is dict and set(sample)=={"wall_ns","exact_reference_passed","coverage"} and integer(sample["wall_ns"],1,(1<<63)-1) and sample["exact_reference_passed"] is True,"invalid sampled timing/audit")
            epoch+=reuse;coverage(sample["coverage"],case,epoch)
        median=statistics.median(sample["wall_ns"] for sample in samples)
        demand(positive(phase.get("median_batch_ns")) and phase["median_batch_ns"]==median and positive(phase.get("gemm_calls_per_second")) and math.isclose(phase["gemm_calls_per_second"],1e9*reuse/median,rel_tol=1e-12),"incoherent median/rate")
        sustained=phase.get("sustained");demand(type(sustained) is dict and set(sustained)=={"batches","gemm_calls","elapsed_ns","mean_batch_ns","minimum_batches","maximum_batches","target_ns","stop_reason","independent_audit_inside_interval","final_exact_reference_passed","coverage"},"missing sustained fields")
        batches,elapsed=sustained.get("batches"),sustained.get("elapsed_ns")
        demand(integer(batches,16,4096) and integer(elapsed,1,(1<<63)-1) and same(sustained.get("gemm_calls"),batches*reuse) and same(sustained.get("minimum_batches"),16) and same(sustained.get("maximum_batches"),4096) and same(sustained.get("target_ns"),250000000)
            and sustained.get("independent_audit_inside_interval") is False and sustained.get("final_exact_reference_passed") is True and positive(sustained.get("mean_batch_ns")) and math.isclose(sustained["mean_batch_ns"]*batches,elapsed,rel_tol=1e-12)
            and ((elapsed>=250000000 and sustained.get("stop_reason")=="duration") or (elapsed<250000000 and batches==4096 and sustained.get("stop_reason")=="call_cap")),"invalid sustained timing/stop/audit")
        epoch+=batches*reuse;coverage(sustained["coverage"],case,epoch);allocations(phase.get("matrix_allocations_after_audit"),case,envelope,workspace)
    cold=capture.get("cold_setup_plus_first_batch_ns")
    demand(integer(cold,1,(1<<63)-1) and cold==capture["query_ns"]+capture["initial_setup_ns"]+phases[0]["first_batch_ns"],"cold setup/first boundary mismatch")
    if panel:
        demand(same(capture.get("out_of_bound_probe"),dict(status="RNS8_RANGE_ERROR",output_preserved=True,epoch_preserved=True)),"missing preflight preservation evidence")
        recreate=capture.get("recreation");demand(type(recreate) is dict and set(recreate)=={"destroy_create_ns","first_batch_ns","setup_plus_first_batch_ns","coverage","exact_reference_passed"},"missing recreation boundaries")
        demand(all(integer(recreate.get(key),1,(1<<63)-1) for key in ("destroy_create_ns","first_batch_ns","setup_plus_first_batch_ns")) and recreate["setup_plus_first_batch_ns"]==recreate["destroy_create_ns"]+recreate["first_batch_ns"] and recreate["exact_reference_passed"] is True,"invalid recreation timing/audit")
        coverage(recreate["coverage"],case,reuse);epoch+=reuse
    else:demand(capture.get("recreation") is None and capture.get("out_of_bound_probe") is None,"in-core claimed stream-only lifecycle")
    demand(same(capture.get("production_native_calls"),epoch),"executed native call count mismatch")
    validate_provenance(capture.get("build_manifest"))
    return {key:capture[key] for key in ("build_manifest","device_actual","device_name","target","hip_runtime_version","hip_driver_version","reference_fixed_operand_object_bytes")}


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(observations):
    groups={}
    for item in observations:groups.setdefault(tuple(item["case"][key] for key in MATCH),[]).append(item)
    return [dict(matched_identity=dict(zip(MATCH,key)),complete_two_mode_cohort=set(MODES)<={item["case"]["mode"] for item in group},observations=group,
        comparison_rule="observed phase-matched independent processes only; panel sizes/budgets remain separate observations; no peak/overlap/crossover extrapolation") for key,group in groups.items()]


def collect(args):
    demand(integer(args.runs,1,100) and integer(args.repeats,1,1000) and integer(args.warmups,1,1000) and integer(args.timeout,1,86400) and type(args.order_seed) is int,"invalid collection controls")
    case_hash=digest(args.cases);cases=load_cases(args.cases);demand(digest(args.cases)==case_hash,"cases changed while loading")
    exe=Path(args.exe).resolve(strict=True);artifact=digest(exe);sources=[Path(__file__),Path(__file__).with_name("collect_finite_verify.py"),Path(__file__).with_name("build_manifest.py")]
    tools={path.name:digest(path) for path in sources};python_hash=digest(sys.executable);output=Path(args.output);output.mkdir(parents=True,exist_ok=True);campaign=str(uuid.uuid4())
    tasks=[(case,repeat) for repeat in range(args.runs) for case in cases];random.Random(args.order_seed).shuffle(tasks);cohort=None;observations=[]
    def unchanged():demand(digest(exe)==artifact and digest(args.cases)==case_hash and digest(sys.executable)==python_hash and all(digest(path)==tools[path.name] for path in sources),"artifact/tools/interpreter/cases changed during collection")
    for ordinal,(case,repeat) in enumerate(tasks):
        unchanged();run_id=str(uuid.uuid4());command=[str(exe),str(case["device"]),case["mode"],*(str(case[key]) for key in ("m","n","k")),case["domain"],*(str(case[key]) for key in ("bits","prefix","panel_m","panel_n","panel_k","device_budget_bytes","host_budget_bytes","reuse")),str(args.repeats),str(args.warmups),run_id,str(case["retained_b_budget_bytes"])]
        record=dict(schema="rns8_stream_panel_collection_v2",campaign_id=campaign,run_id=run_id,ordinal=ordinal,case=case,independent_repeat=repeat,order_seed=args.order_seed,command=command,started_unix_ns=time.time_ns(),artifact_sha256=artifact,cases_sha256=case_hash,
            collector_sha256=tools[Path(__file__).name],tool_dependencies_sha256=tools,python_executable=sys.executable,python_version=platform.python_version(),python_sha256=python_hash,validated=False,promotion_eligible=False,provenance_complete=False,exit_code=None,stdout="",stderr="")
        path=output/f"{run_id}.json"
        try:
            try:
                process=subprocess.run(command,text=True,encoding="utf-8",errors="strict",capture_output=True,timeout=args.timeout,check=False);record.update(exit_code=process.returncode,stdout=process.stdout,stderr=process.stderr)
            except subprocess.TimeoutExpired as error:
                record.update(stdout=(error.stdout.decode("utf-8",errors="replace") if isinstance(error.stdout,bytes) else error.stdout) or "",stderr=(error.stderr.decode("utf-8",errors="replace") if isinstance(error.stderr,bytes) else error.stderr) or "",timeout_seconds=args.timeout)
                raise ValueError("serial panel process timed out") from error
            demand(process.returncode==0,"serial panel process failed");capture=strict_json(process.stdout);identity=validate_capture(capture,case,run_id,args.repeats,args.warmups)
            if cohort is None:cohort=identity
            demand(same(identity,cohort),"compiled build/device/runtime cohort mismatch");unchanged();record.update(validated=True,provenance_complete=manifest_provenance_complete(capture["build_manifest"]))
            observations.append(dict(case=case,run_id=run_id,independent_repeat=repeat,panel_resources=resources(case),cold_setup_plus_first_batch_ns=capture["cold_setup_plus_first_batch_ns"],recreation=capture["recreation"],
                phases=[{key:phase[key] for key in ("generation","binding_action","first_batch_ns","median_batch_ns","sustained")} for phase in capture["phases"]]))
        except Exception as error:
            record["failure"]=str(error)
            with path.open("x",encoding="utf-8") as stream:json.dump(record,stream,indent=2,allow_nan=False)
            raise
        with path.open("x",encoding="utf-8") as stream:json.dump(record,stream,indent=2,allow_nan=False)
    summary=dict(schema="rns8_stream_panel_campaign_v2",campaign_id=campaign,artifact_sha256=artifact,cases_sha256=case_hash,tool_dependencies_sha256=tools,cohort=cohort,independent_processes=len(observations),promotion_eligible=False,provenance_complete=manifest_provenance_complete(cohort["build_manifest"]),matched_cohorts=summarize(observations))
    with (output/f"{campaign}.summary.json").open("x",encoding="utf-8") as stream:json.dump(summary,stream,indent=2,allow_nan=False)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("exe","cases","output"):parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--runs",type=int,default=5);parser.add_argument("--repeats",type=int,default=9);parser.add_argument("--warmups",type=int,default=3);parser.add_argument("--timeout",type=int,default=1800);parser.add_argument("--order-seed",type=int,default=6811)
    args=parser.parse_args()
    try:summary=collect(args)
    except (ValueError,OSError) as error:parser.error(str(error))
    print(f"Collected {summary['independent_processes']} public serial-panel processes; qualification remains pending");return 0


if __name__=="__main__":raise SystemExit(main())
