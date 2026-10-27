"""Collect exact instruction-chain observations, preserving diagnostic provenance.

Wave jobs and source instruction counts are models. Neither normalized event
time nor an unreviewed ISA-file hash establishes hardware cycle latency.
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

MODES=("raw","reduce251")
CHAINS=(1,2,4,8)
KEYS={"device","mode","chains","iterations","waves"}
LEFT=(-128,127,-128,1,0,-1,63,-64)
RIGHT=(-128,127,127,-128,127,-1,-64,63)
BUDGET=256*1024*1024


def validate_case(case):
    demand(type(case) is dict and set(case)==KEYS,"case requires exactly device/mode/chains/iterations/waves")
    demand(integer(case["device"],0,63) and case["mode"] in MODES,"invalid explicit device/mode")
    demand(type(case["chains"]) is int and case["chains"] in CHAINS,"chains must be1/2/4/8")
    demand(integer(case["iterations"],1,4096) and integer(case["waves"],1,4096),"iterations/wave jobs must be1..4096")
    return dict(case)


def load_cases(path):
    document=strict_json(Path(path).read_text(encoding="utf-8-sig"))
    demand(type(document) is dict and set(document)=={"schema","cases"} and document["schema"]=="rns8_instruction_roof_cases_v1","invalid case manifest")
    cases=document["cases"];demand(type(cases) is list and 1<=len(cases)<=1024,"case count must be1..1024")
    cases=[validate_case(case) for case in cases];keys=[json.dumps(case,sort_keys=True) for case in cases]
    demand(len(keys)==len(set(keys)),"duplicate cases are not independent observations")
    return cases


def contract(case,run_id,repeats,warmups):
    chains,iterations,waves=(case[key] for key in ("chains","iterations","waves"));reduce=case["mode"]=="reduce251"
    instructions=chains*iterations*waves;outputs=waves*chains*256;allocation=outputs*4+64
    return dict(schema="rns8_instruction_roof_microcase_v1",run_id=run_id,**case,
        device_requested=case["device"],device_actual=case["device"],backend_requested="hip-direct",backend_actual="hip-direct",
        seed=0,dataset_recipe="fixed_signed_runtime_chain_extrema_v1",a_values=list(LEFT[:chains]),b_values=list(RIGHT[:chains]),
        repeats=repeats,warmups=warmups,input_contract="one_runtime_signed_byte_per_chain_replicated_into_uniform_16x16_fragments",
        output_contract="wave_chain_row_major_int32_cells_then16_int32_canaries",reference_method="int64_full_16_term_recurrence_every_iteration_all_cells",
        semantic_contract="signed_int8_wmma_then_centered_mod251_each_iteration" if reduce else "exact_signed_int8_wmma_int32_accumulation",
        k_equivalent=iterations*16,maximum_accumulator_magnitude=16*128*128+125 if reduce else iterations*16*128*128,
        modeled_wave_instructions=instructions,modeled_integer_operations=instructions*8192,modeled_output_cells=outputs,
        operation_count_contract="two_operations_per_int8_mac_16cubed_per_wave_instruction_excludes_reduction_probe_and_stores",
        accumulator_bound_scope="before_each_reduction" if reduce else "entire_raw_chain",
        modeled_output_store_bytes=outputs*4,input_bytes_each=chains,output_allocation_bytes=allocation,capability_bytes=4,canary_bytes=64,
        logical_payload_envelope_bytes=allocation*4+6*chains+8+16384,memory_budget_bytes=BUDGET,
        threads_per_block=128,waves_per_block=4,grid_blocks=(waves+3)//4,warp_width=32,
        kernel_event_label="instruction_roof_wmma_reduce251" if reduce else "instruction_roof_wmma_raw",timer="std_chrono_steady_clock_nanoseconds",
        host_timing_scope="resident_checked_call_including_capability_probe_status_transfer_and_required_sync",
        event_timing_scope="chain_kernel_only_including_fragment_setup_loop_reduction_if_requested_and_all_output_stores",
        normalization_contract="kernel_event_ns_divided_by_modeled_wave_instructions_not_hardware_cycles_or_isolated_instruction_latency",
        actual_issued_instructions=None,observed_resident_waves=None,achieved_occupancy=None,hardware_clock_hz=None,hardware_cycle_latency=None,
        compiler_private_scratch_bytes=None,actual_hbm_bytes=None,timing_resolution_ns=None,isa_sha256=None,isa_verified=False,promotion_eligible=False,
        production_timing_hooks_enabled=False)


def dataset(case):
    chains,iterations,waves=(case[key] for key in ("chains","iterations","waves"))
    a=bytes(value&255 for value in LEFT[:chains]);b=bytes(value&255 for value in RIGHT[:chains]);tiles=[]
    for left,right in zip(LEFT[:chains],RIGHT[:chains]):
        value=0
        for _ in range(iterations):
            for _ in range(16):value+=left*right
            if case["mode"]=="reduce251":
                value%=251
                if value>=126:value-=251
        demand(-(1 << 31)<=value<(1 << 31),"independent recurrence exceeds INT32")
        tiles.append(value.to_bytes(4,"little",signed=True)*256)
    gold=b"".join(tiles)*waves+(0x12345678).to_bytes(4,"little")*16
    return a,b,gold


@lru_cache(maxsize=32)
def reference_checksums(chains,iterations,waves,mode):
    # Retain only three short checksums, never full outputs or timing evidence.
    return tuple(checksum(data) for data in dataset(dict(chains=chains,iterations=iterations,waves=waves,mode=mode)))


def validate_capture(capture,case,run_id,repeats,warmups):
    demand(type(capture) is dict,"capture must be object")
    expected=contract(case,run_id,repeats,warmups)
    for key,value in expected.items():demand(key in capture and same(capture[key],value),f"metadata mismatch: {key}")
    demand(type(capture.get("device_name")) is str and bool(capture["device_name"].strip()),"missing device identity")
    target=capture.get("target")
    demand(type(target) is str and re.fullmatch(r"gfx[0-9a-f]+(?::[A-Za-z0-9_+:-]+)?",target)
        and target.split(":")[0] in {"gfx1100","gfx1101","gfx1102"},"target lacks admitted native WMMA route")
    demand(integer(capture.get("runtime_version"),1,(1 << 31)-1),"invalid runtime identity")
    for field in ("oracle_and_dataset_ns","resident_setup_ns","resource_release_ns"):demand(positive(capture.get(field)),f"invalid host phase: {field}")
    demand(tuple(capture.get(key) for key in ("a_checksum","b_checksum","output_checksum"))==reference_checksums(
        *(case[key] for key in ("chains","iterations","waves","mode"))),"full input/recurrence/output checksum mismatch")
    samples=capture.get("samples");demand(type(samples) is list and len(samples)==repeats,"invalid sample count")
    for sample in samples:
        demand(type(sample) is dict and positive(sample.get("host_call_ns")) and positive(sample.get("uninstrumented_host_call_ns"))
            and sample.get("exact_reference_passed") is True and sample.get("uninstrumented_exact_reference_passed") is True,"invalid paired sample")
        capability=sample.get("capability_events")
        demand(type(capability) is list and len(capability)<=1,"at most one capability transfer event required")
        for event in capability:
            demand(type(event) is dict and set(event)=={"label","elapsed_ns"} and event["label"]=="residue_d2h_sync",
                   "capability transfer event label mismatch")
            demand(event["elapsed_ns"] is None or positive(event["elapsed_ns"]),"invalid capability transfer duration")
        events=sample.get("events");demand(type(events) is list and len(events)<=1,"one kernel event or explicit unavailable state required")
        kernel=sample.get("kernel_event_ns","missing");normalized=sample.get("event_ns_per_modeled_wave_instruction","missing")
        if events:
            event=events[0]
            demand(type(event) is dict and set(event)=={"label","elapsed_ns"} and event["label"]==expected["kernel_event_label"],"kernel event label mismatch")
            duration=event["elapsed_ns"];demand(duration is None or positive(duration),"invalid kernel event duration")
            demand(same(kernel,duration) and sample.get("kernel_event_state")==("below_resolution" if duration is None else "measured"),"kernel event state/value mismatch")
        else:demand(kernel is None and sample.get("kernel_event_state")=="unavailable","missing event must remain unavailable")
        if kernel is None:demand(normalized is None,"unmeasured kernel cannot have normalized timing")
        else:demand(positive(normalized) and math.isclose(normalized*expected["modeled_wave_instructions"],kernel,rel_tol=1e-12),"event normalization differs from modeled instructions")
    sustained=capture.get("sustained")
    demand(type(sustained) is dict and integer(sustained.get("calls"),16,4096) and positive(sustained.get("elapsed_ns"))
        and positive(sustained.get("mean_call_ns")) and sustained.get("final_exact_reference_passed") is True,"invalid sustained evidence")
    demand(same(sustained.get("minimum_calls"),16) and same(sustained.get("maximum_calls"),4096) and same(sustained.get("target_ns"),250000000)
        and sustained.get("independent_audit_inside_interval") is False,"invalid sustained controls")
    demand(math.isclose(sustained["mean_call_ns"]*sustained["calls"],sustained["elapsed_ns"],rel_tol=1e-12)
        and (sustained["calls"]==4096 or sustained["elapsed_ns"]>=250000000),"sustained duration mismatch")
    validate_provenance(capture.get("build_manifest"))
    return {key:capture[key] for key in ("build_manifest","device","device_name","target","runtime_version")}


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collect(args):
    demand(integer(args.runs,1,100) and integer(args.repeats,1,10000) and integer(args.warmups,1,1000)
        and integer(args.timeout,1,86400) and type(args.order_seed) is int,"invalid collection bounds")
    case_hash=digest(args.cases);cases=load_cases(args.cases);demand(digest(args.cases)==case_hash,"cases changed while loading")
    exe=Path(args.exe).resolve(strict=True);artifact=digest(exe)
    sources=[Path(__file__),Path(__file__).with_name("collect_finite_verify.py"),Path(__file__).with_name("build_manifest.py")]
    identities={path.name:digest(path) for path in sources};python_hash=digest(sys.executable)
    isa_path=getattr(args,"isa_file",None);isa_path=Path(isa_path).resolve(strict=True) if isa_path else None
    isa=dict(path=str(isa_path),sha256=digest(isa_path),correspondence_verified=False,review_status="unreviewed") if isa_path else None
    campaign=str(uuid.uuid4());output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    tasks=[(case,repeat) for repeat in range(args.runs) for case in cases];random.Random(args.order_seed).shuffle(tasks)
    cohort=None;observations=[]
    def unchanged():
        demand(digest(exe)==artifact and digest(args.cases)==case_hash and all(digest(path)==identities[path.name] for path in sources),"artifact/tools/cases changed during collection")
        if isa_path:demand(digest(isa_path)==isa["sha256"],"unreviewed ISA artifact changed during collection")
    for ordinal,(case,repeat) in enumerate(tasks):
        unchanged();run_id=str(uuid.uuid4())
        command=[str(exe),str(case["device"]),case["mode"],*(str(case[key]) for key in ("chains","iterations","waves")),str(args.repeats),str(args.warmups),run_id]
        record=dict(schema="rns8_instruction_roof_collection_v1",campaign_id=campaign,run_id=run_id,case=case,command=command,ordinal=ordinal,
            independent_repeat=repeat,order_seed=args.order_seed,started_unix_ns=time.time_ns(),artifact_sha256=artifact,
            collector_sha256=identities[Path(__file__).name],tool_dependencies_sha256=identities,cases_sha256=case_hash,
            python_executable=sys.executable,python_sha256=python_hash,python_version=platform.python_version(),unreviewed_isa_artifact=isa,
            promotion_eligible=False,validated=False,provenance_complete=False,exit_code=None,stdout="",stderr="")
        path=output/f"{run_id}.json"
        try:
            try:
                result=subprocess.run(command,text=True,encoding="utf-8",errors="strict",capture_output=True,timeout=args.timeout,check=False)
                record.update(exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr)
            except subprocess.TimeoutExpired as error:
                record.update(stdout=(error.stdout.decode("utf-8",errors="replace") if isinstance(error.stdout,bytes) else error.stdout) or "",
                    stderr=(error.stderr.decode("utf-8",errors="replace") if isinstance(error.stderr,bytes) else error.stderr) or "",timeout_seconds=args.timeout)
                raise ValueError("instruction-roof process timed out") from error
            demand(result.returncode==0,"instruction-roof process failed")
            capture=strict_json(result.stdout);identity=validate_capture(capture,case,run_id,args.repeats,args.warmups)
            if cohort is None:cohort=identity
            demand(same(identity,cohort),"compiled build/device/runtime cohort mismatch")
            unchanged();record.update(validated=True,provenance_complete=manifest_provenance_complete(capture["build_manifest"]))
            measured=[sample["kernel_event_ns"] for sample in capture["samples"] if sample["kernel_event_state"]=="measured"]
            observations.append(dict(case=case,run_id=run_id,independent_repeat=repeat,sustained_mean_host_ns=capture["sustained"]["mean_call_ns"],
                median_production_host_ns=statistics.median(sample["uninstrumented_host_call_ns"] for sample in capture["samples"]),
                median_measured_kernel_event_ns=statistics.median(measured) if measured else None,measured_kernel_events=len(measured),
                censored_kernel_events=sum(sample["kernel_event_state"]=="below_resolution" for sample in capture["samples"]),
                unavailable_kernel_events=sum(sample["kernel_event_state"]=="unavailable" for sample in capture["samples"]),
                event_normalization="per_modeled_wave_instruction_not_hardware_cycles"))
        except Exception as error:
            record["failure"]=str(error)
            with path.open("x",encoding="utf-8") as stream:json.dump(record,stream,indent=2,allow_nan=False)
            raise
        with path.open("x",encoding="utf-8") as stream:json.dump(record,stream,indent=2,allow_nan=False)
    summary=dict(schema="rns8_instruction_roof_campaign_v1",campaign_id=campaign,promotion_eligible=False,artifact_sha256=artifact,
        tool_dependencies_sha256=identities,cases_sha256=case_hash,cohort=cohort,unreviewed_isa_artifact=isa,
        independent_processes=len(observations),provenance_complete=manifest_provenance_complete(cohort["build_manifest"]),observations=observations,
        comparison_rule="same device/build/runtime; raw and reduce251 are distinct programs; launched wave jobs are not observed occupancy; no hardware cycle latency inference")
    with (output/f"{campaign}.summary.json").open("x",encoding="utf-8") as stream:json.dump(summary,stream,indent=2,allow_nan=False)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("exe","cases","output"):parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--isa-file",type=Path,help="Optional unreviewed artifact hash; does not verify correspondence or qualification")
    parser.add_argument("--runs",type=int,default=5);parser.add_argument("--repeats",type=int,default=9);parser.add_argument("--warmups",type=int,default=3)
    parser.add_argument("--timeout",type=int,default=600);parser.add_argument("--order-seed",type=int,default=3553)
    args=parser.parse_args()
    try:summary=collect(args)
    except (ValueError,OSError) as error:parser.error(str(error))
    print(f"Collected {summary['independent_processes']} exact chain processes; ISA and hardware qualification remain pending");return 0

if __name__=="__main__":raise SystemExit(main())
