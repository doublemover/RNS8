"""Independent public ragged lifecycle receipts, with explicit host/resident scopes.

Uniform export and ragged reconstruct+export have matched host results. Resident
reconstruction has no public uniform-copy control. Traffic fields are models.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
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
from modulus_set_search import DEFAULT_MODULI

MODES=("uniform_export","ragged_export","ragged_resident")
KEYS={"backend","device","mode","rows","cols","pattern","domain","tile_rows","tile_cols","reuse"}
MASK=(1<<64)-1
BUDGET=512*1024*1024


def validate_case(case):
    demand(type(case) is dict and set(case)==KEYS,"case requires exact backend/device/mode/shape/recipe/signedness/tile/reuse keys")
    demand(case["backend"] in ("cpu-reference","hip-direct") and integer(case["device"],-1,63)
        and (case["device"]==-1 if case["backend"]=="cpu-reference" else case["device"]>=0),"explicit backend/device mismatch")
    demand(case["mode"] in MODES and case["pattern"] in ("mixed","fullwidth","wide140") and case["domain"] in ("signed","unsigned"),"invalid mode/recipe/signedness")
    demand(all(integer(case[key],1,512) for key in ("rows","cols")) and all(integer(case[key],1,128) for key in ("tile_rows","tile_cols"))
        and integer(case["reuse"],1,256),"invalid shape/tile/reuse")
    cells=case["rows"]*case["cols"]
    demand(cells*case["reuse"]<=16777216 and (case["pattern"]!="wide140" or cells*16384<=134217728),"construction or batch work exceeds domain")
    demand(envelope(case)<=BUDGET,"aggregate payload envelope exceeds budget")
    return dict(case)


def envelope(case):
    cells=case["rows"]*case["cols"];tiles=((case["rows"]+case["tile_rows"]-1)//case["tile_rows"])*((case["cols"]+case["tile_cols"]-1)//case["tile_cols"])
    construction=(case["rows"]+case["cols"])*16384 if case["pattern"]=="wide140" else 0
    upper=tiles*case["tile_rows"]*case["tile_cols"]*20
    return 512*cells+64*construction+(0 if case["mode"]=="uniform_export" else 2*(upper+tiles*256+4)+20*cells)+16384


def load_cases(path):
    document=strict_json(Path(path).read_text(encoding="utf-8-sig"))
    demand(type(document) is dict and set(document)=={"schema","cases"} and document["schema"]=="rns8_ragged_cases_v1","invalid cases schema")
    demand(type(document["cases"]) is list and 1<=len(document["cases"])<=2048,"invalid case count")
    cases=[validate_case(case) for case in document["cases"]];ids=[json.dumps(case,sort_keys=True) for case in cases]
    demand(len(ids)==len(set(ids)),"duplicate cases are not independent evidence")
    return cases


def signed(case,value):return value-(1<<64) if case["domain"]=="signed" and value>>63 else value


def factor(case,left,index):
    band=(index//case["tile_rows" if left else "tile_cols"])%4
    return (0,1,1<<32,1<<63)[band]


def native(case):
    state=5152;values=[]
    for row in range(case["rows"]):
        for col in range(case["cols"]):
            i=row*case["cols"]+col;state=(state+0x9e3779b97f4a7c15)&MASK;word=state
            word=((word^(word>>30))*0xbf58476d1ce4e5b9)&MASK;word=((word^(word>>27))*0x94d049bb133111eb)&MASK;word^=word>>31
            if i%257==0:word=(1<<63) if case["domain"]=="signed" else MASK
            elif i%257==1:word=(1<<63)-1 if case["domain"]=="signed" else 0
            if case["pattern"]=="mixed":
                band=(col//case["tile_cols"])%4
                if band==0:word=0
                elif band<3:
                    word=word%127 if band==1 else (1<<32)+word%65536
                    if case["domain"]=="signed" and (row+col)%2:word=(-word)&MASK
            values.append(word)
    return values


def word_checksum(values):return checksum(b"".join((word&MASK).to_bytes(8,"little") for word in values))


def values(case):
    if case["pattern"]=="wide140":return [16384*signed(case,factor(case,True,row))*signed(case,factor(case,False,col)) for row in range(case["rows"]) for col in range(case["cols"])]
    return [signed(case,word) for word in native(case)]


@lru_cache(maxsize=8)
def reference(rows,cols,pattern,domain,tile_rows,tile_cols):
    case=dict(rows=rows,cols=cols,pattern=pattern,domain=domain,tile_rows=tile_rows,tile_cols=tile_cols)
    data=values(case);output=[]
    for value in data:output.extend((value>>(64*limb))&MASK for limb in range(3))
    prefixes=[];payload=0
    for tr in range((rows+tile_rows-1)//tile_rows):
        for tc in range((cols+tile_cols-1)//tile_cols):
            bound=max(abs(data[row*cols+col]) for row in range(tr*tile_rows,min(rows,(tr+1)*tile_rows)) for col in range(tc*tile_cols,min(cols,(tc+1)*tile_cols)))
            product=1
            for prefix in range(1,21):
                product*=DEFAULT_MODULI[prefix-1]
                if product>(2*bound if domain=="signed" else bound):break
            demand(product>(2*bound if domain=="signed" else bound),"tile cannot fit active ladder")
            prefixes.append(prefix);payload+=prefix*tile_rows*tile_cols
    return dict(source_checksum=word_checksum(output),output_checksum=word_checksum(output),native_input_checksum=None if pattern=="wide140" else word_checksum(native(case)),
        modeled_layout=dict(tile_count=len(prefixes),payload_bytes=payload,tile_prefixes=prefixes,minimum_prefix=min(prefixes),maximum_prefix=max(prefixes)))


def contract(case,run_id,repeats,warmups):
    return dict(schema="rns8_ragged_microcase_v1",promotion_eligible=False,run_id=run_id,**{key:case[key] for key in KEYS if key not in ("backend","device")},
        backend_requested=case["backend"],backend_actual=case["backend"],device_requested=case["device"],device_actual=case["device"],
        repeats=repeats,warmups=warmups,prefix=20,semantics="exact_wide_"+case["domain"],recipe="ragged_tile_bands_seed5152_or_constant_dot16384_v1",dataset_seed=5152,
        reference_method="cpp_int_every_value_constant_dot_closed_form_with_scalar_row_edge_audits_or_native_decode",source_construction_k=16384 if case["pattern"]=="wide140" else 0,
        memory_budget_bytes=BUDGET,allocation_scope="public_matrix_capacities_and_owner_payload_directory_metadata_staging_at_named_boundaries_not_peak",
        memory_exclusions="allocator_bookkeeping_spare_capacity_context_runtime_fixed_metadata_not_RSS",timer="std_chrono_steady_clock_nanoseconds",
        timed_scope="uniform_resident_export_each_reuse" if case["mode"]=="uniform_export" else "ragged_reconstruct_then_export_each_reuse" if case["mode"]=="ragged_export" else "ragged_reconstruct_resident_each_reuse",
        kernel_events=None,observed_hbm_bytes=None,observed_transfer_bytes=None,observed_overlap=None,
        modeled_compact_payload_h2d_bytes=0,modeled_full_uniform_snapshot_d2h_bytes=0,exports_per_batch=0 if case["mode"]=="ragged_resident" else case["reuse"],
        exact_full_value_and_padding_checked=True,first_exact_reference_passed=True,
        **deepcopy(reference(*(case[key] for key in ("rows","cols","pattern","domain","tile_rows","tile_cols")))))


def validate_resource(resource,case,capture,staged):
    fields={"tile_count","payload_bytes","directory_bytes","minimum_prefix","maximum_prefix","device_metadata_bytes","materialization_capacity_bytes","source_instance","source_generation"}
    demand(type(resource) is dict and set(resource)==fields and all(integer(v,0,BUDGET) for key,v in resource.items() if key not in ("source_instance","source_generation")),"invalid resource fields")
    for key in ("tile_count","payload_bytes","minimum_prefix","maximum_prefix"):demand(same(resource[key],capture["modeled_layout"][key]),f"independent ragged resource mismatch: {key}")
    directory=resource["directory_bytes"];tiles=resource["tile_count"]
    demand(directory%tiles==0 and 32<=directory//tiles<=128,"invalid opaque directory entry capacity")
    demand(same(resource["source_instance"],capture["source_instance"]) and integer(resource["source_generation"],1,MASK),"invalid owner source binding")
    hip=case["backend"]=="hip-direct"
    demand(resource["device_metadata_bytes"]==(directory+4 if hip else 0) and resource["materialization_capacity_bytes"]==(20*case["rows"]*case["cols"] if staged and hip else 0),"retained scratch capacity disagrees with stage/backend")
    return resource["payload_bytes"]+directory+resource["device_metadata_bytes"]+resource["materialization_capacity_bytes"]


def validate_allocations(records,case,owner_bytes):
    fields={"logical_primary_bytes","host_payload_capacity_bytes","device_payload_capacity_bytes","device_upload_staging_bytes","device_export_staging_bytes","device_status_and_schedule_bytes","device_total_capacity_bytes"}
    demand(type(records) is list and len(records)==(1 if case["mode"]=="uniform_export" else 2),"matrix allocation boundary mismatch")
    total=owner_bytes;cells=case["rows"]*case["cols"]
    for record in records:
        demand(type(record) is dict and set(record)==fields and all(integer(v,0,BUDGET) for v in record.values()),"invalid matrix allocation fields")
        demand(record["logical_primary_bytes"]==20*cells and record["device_total_capacity_bytes"]==sum(record[key] for key in ("device_payload_capacity_bytes","device_upload_staging_bytes","device_export_staging_bytes","device_status_and_schedule_bytes")),"invalid matrix allocation accounting")
        demand(record["device_payload_capacity_bytes" if case["backend"]=="hip-direct" else "host_payload_capacity_bytes"]>=20*cells,"missing persistent uniform storage")
        total+=record["host_payload_capacity_bytes"]+record["device_total_capacity_bytes"]
    demand(total<=envelope(case),"queried retained payload exceeds conservative live envelope")


def validate_capture(capture,case,run_id,repeats,warmups):
    demand(type(capture) is dict,"capture must be object")
    for key,value in contract(case,run_id,repeats,warmups).items():demand(key in capture and same(capture[key],value),f"metadata mismatch: {key}")
    demand(type(capture.get("device_name")) is str and capture["device_name"].strip(),"missing device name")
    hip=case["backend"]=="hip-direct";uniform=case["mode"]=="uniform_export"
    demand(type(capture.get("target")) is str and (re.fullmatch(r"gfx[0-9a-f]+(?::[A-Za-z0-9_+:-]+)?",capture["target"]) if hip else capture["target"]=="cpu"),"invalid target identity")
    for key in ("hip_runtime_version","hip_driver_version"):demand(integer(capture.get(key),1 if hip else 0,(1<<31)-1 if hip else 0),"runtime identity mismatch")
    for key in ("oracle_ns","setup_ns","first_batch_ns","setup_plus_first_batch_ns","clock_period_numerator","clock_period_denominator"):demand(integer(capture.get(key),1,(1<<63)-1),f"invalid timing: {key}")
    demand(capture["setup_plus_first_batch_ns"]==capture["setup_ns"]+capture["first_batch_ns"],"setup boundary sum mismatch")
    for key in ("first_prepare_ns","owner_destruction_ns"):demand(integer(capture.get(key),0 if uniform else 1,0 if uniform else (1<<63)-1),"irrelevant or invalid preparation/destruction")
    demand(capture["setup_ns"]>=capture["first_prepare_ns"],"preparation exceeds setup")
    wide=case["pattern"]=="wide140"
    demand(integer(capture.get("source_construction_ns"),1 if wide else 0,(1<<63)-1 if wide else 0),"source construction timing mismatch")
    work=capture.get("source_construction_workspace_bytes")
    demand(integer(work,0,BUDGET) and (wide or work==0) and same(capture.get("logical_payload_envelope_bytes"),envelope(case)+work)
        and envelope(case)+work<=BUDGET,"invalid aggregate construction/live memory budget")
    demand(integer(capture.get("source_instance"),1,MASK) and integer(capture.get("source_version"),1,MASK)
        and (wide or capture["source_version"]==1),"source identity/version mismatch")
    demand(integer(capture.get("output_instance"),0 if uniform else 1,0 if uniform else MASK)
        and (uniform or capture["output_instance"]!=capture["source_instance"]),"output identity mismatch")
    initial=capture.get("ragged_resources_at_prepare");first=capture.get("ragged_resources_after_first");final=capture.get("ragged_resources_final")
    if uniform:demand(all(key in capture and capture[key] is None for key in ("ragged_resources_at_prepare","ragged_resources_after_first","ragged_resources_final")),"uniform route cannot claim a ragged owner");owner_bytes=[0,0,0];directory=0
    else:
        owner_bytes=[validate_resource(initial,case,capture,False),validate_resource(first,case,capture,True),validate_resource(final,case,capture,True)]
        demand(same(first,final) and first["source_generation"]==initial["source_generation"] and first["directory_bytes"]==initial["directory_bytes"],"owner binding/capacity changed after first use")
        directory=initial["directory_bytes"]
    for key,expected in (("modeled_prepare_directory_d2h_bytes",directory if hip else 0),("modeled_prepare_status_d2h_bytes",4 if hip and not uniform else 0),
        ("modeled_prepare_directory_h2d_bytes",directory if hip else 0),("modeled_materialization_directory_h2d_bytes_per_call",directory if hip else 0),
        ("modeled_materialization_status_d2h_bytes_per_call",4 if hip and not uniform else 0)):
        demand(same(capture.get(key),expected),f"modeled transfer contract mismatch: {key}")
    for key,owned in zip(("matrix_allocations_at_setup","matrix_allocations_after_first","matrix_allocations_final"),owner_bytes):validate_allocations(capture.get(key),case,owned)
    preparations=capture.get("preparation_samples");demand(type(preparations) is list and len(preparations)==(0 if uniform else repeats),"fresh preparation sample count mismatch")
    for sample in preparations:demand(type(sample) is dict and set(sample)=={"preparation_ns","destruction_ns","resources"} and integer(sample["preparation_ns"],1,(1<<63)-1)
        and integer(sample["destruction_ns"],1,(1<<63)-1) and same(sample["resources"],initial),"invalid fresh preparation/destruction/resource sample")
    samples=capture.get("samples");demand(type(samples) is list and len(samples)==repeats,"sample count mismatch")
    for sample in samples:demand(type(sample) is dict and set(sample)=={"wall_ns","exact_reference_passed"} and integer(sample["wall_ns"],1,(1<<63)-1) and sample["exact_reference_passed"] is True,"invalid production sample/audit")
    median=statistics.median(sample["wall_ns"] for sample in samples)
    demand(positive(capture.get("median_batch_ns")) and capture["median_batch_ns"]==median and positive(capture.get("materializations_per_second"))
        and math.isclose(capture["materializations_per_second"],1e9*case["reuse"]/median,rel_tol=1e-12),"batch median/rate mismatch")
    sustained=capture.get("sustained");demand(type(sustained) is dict,"missing sustained interval")
    calls,elapsed=sustained.get("calls"),sustained.get("elapsed_ns")
    demand(integer(calls,16,4096) and integer(elapsed,1,(1<<63)-1) and same(sustained.get("materializations"),calls*case["reuse"])
        and same(sustained.get("minimum_calls"),16) and same(sustained.get("maximum_calls"),4096) and same(sustained.get("target_ns"),250000000)
        and sustained.get("independent_audit_inside_interval") is False and sustained.get("final_exact_reference_passed") is True and positive(sustained.get("mean_batch_ns"))
        and math.isclose(sustained["mean_batch_ns"]*calls,elapsed,rel_tol=1e-12)
        and ((elapsed>=250000000 and sustained.get("stop_reason")=="duration") or (elapsed<250000000 and calls==4096 and sustained.get("stop_reason")=="call_cap")),"invalid sustained evidence")
    validate_provenance(capture.get("build_manifest"))
    return {key:capture[key] for key in ("build_manifest","backend_actual","device_actual","device_name","target","hip_runtime_version","hip_driver_version")}


def summarize(observations):
    groups={}
    for observation in observations:
        c=observation["case"];key=tuple(c[field] for field in ("backend","device","rows","cols","pattern","domain","tile_rows","tile_cols","reuse"))
        groups.setdefault(key,[]).append(observation)
    return [dict(identity=dict(zip(("backend","device","rows","cols","pattern","domain","tile_rows","tile_cols","reuse"),key)),
        matched_host_boundary_available={"uniform_export","ragged_export"}<={item["case"]["mode"] for item in group},observations=group,
        comparison_rule="only uniform_export and ragged_export share a host result boundary; resident reconstruction has no public copy baseline; modeled bytes are not measured traffic") for key,group in groups.items()]


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collect(args):
    demand(integer(args.runs,1,100) and integer(args.repeats,1,1000) and integer(args.warmups,1,1000) and integer(args.timeout,1,86400) and type(args.order_seed) is int,"invalid collection controls")
    case_hash=digest(args.cases);cases=load_cases(args.cases);demand(digest(args.cases)==case_hash,"cases changed while loading")
    exe=Path(args.exe).resolve(strict=True);artifact=digest(exe)
    sources=[Path(__file__),Path(__file__).with_name("collect_finite_verify.py"),Path(__file__).with_name("build_manifest.py"),Path(__file__).with_name("modulus_set_search.py")]
    tools={path.name:digest(path) for path in sources};python_hash=digest(sys.executable)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True);campaign=str(uuid.uuid4())
    tasks=[(case,repeat) for repeat in range(args.runs) for case in cases];random.Random(args.order_seed).shuffle(tasks)
    cohort=None;observations=[]
    def unchanged():demand(digest(exe)==artifact and digest(args.cases)==case_hash and all(digest(path)==tools[path.name] for path in sources),"artifact/tools/cases changed during collection")
    for ordinal,(case,repeat) in enumerate(tasks):
        unchanged();run_id=str(uuid.uuid4());command=[str(exe),case["backend"],str(case["device"]),case["mode"],*(str(case[key]) for key in ("rows","cols")),case["pattern"],case["domain"],*(str(case[key]) for key in ("tile_rows","tile_cols","reuse")),str(args.repeats),str(args.warmups),run_id]
        record=dict(schema="rns8_ragged_collection_v1",campaign_id=campaign,run_id=run_id,ordinal=ordinal,case=case,independent_repeat=repeat,
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
                raise ValueError("ragged process timed out") from error
            demand(process.returncode==0,"ragged process failed")
            capture=strict_json(process.stdout);identity=validate_capture(capture,case,run_id,args.repeats,args.warmups)
            if cohort is None:cohort=identity
            demand(same(identity,cohort),"compiled build/device/runtime cohort mismatch")
            unchanged();record.update(validated=True,provenance_complete=manifest_provenance_complete(capture["build_manifest"]))
            observations.append(dict(case=case,run_id=run_id,independent_repeat=repeat,setup_ns=capture["setup_ns"],first_prepare_ns=capture["first_prepare_ns"],
                first_batch_ns=capture["first_batch_ns"],median_batch_ns=capture["median_batch_ns"],sustained=capture["sustained"],
                retained_resources=capture["ragged_resources_final"]))
        except Exception as error:
            record["failure"]=str(error)
            with path.open("x",encoding="utf-8") as stream:json.dump(record,stream,indent=2,allow_nan=False)
            raise
        with path.open("x",encoding="utf-8") as stream:json.dump(record,stream,indent=2,allow_nan=False)
    summary=dict(schema="rns8_ragged_campaign_v1",campaign_id=campaign,artifact_sha256=artifact,cases_sha256=case_hash,tool_dependencies_sha256=tools,
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
    print(f"Collected {summary['independent_processes']} public ragged processes; qualification remains pending");return 0

if __name__=="__main__":raise SystemExit(main())
