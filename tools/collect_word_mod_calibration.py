"""Collect private canonical word-modular integer/scalar-binary64 controls; no promotion."""
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
from collect_finite_verify import demand, integer, positive, same, strict_json, checksum, validate_provenance

INTEGER_METHODS=("baseline","packed_dot4","canonical_dot4","byte_wmma")
METHODS=(*INTEGER_METHODS,"asymmetric_binary64")
SCOPES=("resident","cold")
MASK=(1 << 64)-1
BUDGET=512*1024*1024
KEYS={"device","method","scope","m","n","k","domain","modulus"}
FLOAT_KEYS={"left_digits","right_digits","slab_k"}


def prime(n):
    if n<2:return False
    for p in (2,3,5,7,11,13,17,19,23,29,31,37):
        if n%p==0:return n==p
    d=n-1;s=0
    while d%2==0:s+=1;d//=2
    for witness in (2,325,9375,28178,450775,9780504,1795265022):
        a=witness%n
        if not a:continue
        x=pow(a,d,n)
        if x in (1,n-1):continue
        for _ in range(s-1):
            x=x*x%n
            if x==n-1:break
        else:return False
    return True


def validate_case(case):
    demand(type(case) is dict,"case must be object")
    floating=case.get("method")=="asymmetric_binary64"
    expected_keys=KEYS|FLOAT_KEYS if floating else KEYS
    demand(set(case)==expected_keys,"case requires exactly documented controls; floating digits/slab are explicit and method-specific")
    demand(integer(case["device"],0,63) and case["method"] in METHODS and case["scope"] in SCOPES,"invalid device/method/scope")
    demand(all(integer(case[key],1,65537 if key=="k" else 512) for key in ("m","n","k"))
        and case["m"]*case["n"]*case["k"]<=134217728,"shape/reference work exceeds limits")
    demand(case["domain"] in ("ring","prime_field") and integer(case["modulus"],2,MASK),"invalid domain/modulus")
    demand(case["domain"]!="prime_field" or prime(case["modulus"]),"prime field requires a prime modulus")
    demand(contract(case,"validation",1,1)["logical_payload_envelope_bytes"]<=BUDGET,"aggregate logical payload exceeds cap")
    return dict(case)


def ceil_root(value,degree):
    """Independent exact root: no float approximation or imported planner proof."""
    low,high=1,value
    while low<high:
        middle=(low+high)//2
        if middle**degree>=value:high=middle
        else:low=middle+1
    return low


def floating_schedule(case):
    left,right,slab=(case[key] for key in ("left_digits","right_digits","slab_k"))
    demand(integer(left,1,4) and integer(right,1,4) and integer(slab,1,65536),"invalid floating digit/slab request")
    p=case["modulus"];alpha=ceil_root(p,left);beta=ceil_root(p,right)
    lhs=slab*alpha*beta+p-1
    demand(lhs<=1 << 53,"binary64 schedule fails exact lambda*alpha*beta+p-1<=2^53")
    weights=[0]*16
    for i in range(left):
        for j in range(right):weights[i*4+j]=(pow(alpha,i,p)*pow(beta,j,p))%p
    return dict(version=1,arithmetic="scalar_binary64_fma_integer_coefficients",left_digits=left,right_digits=right,slab_k=slab,
        alpha=alpha,beta=beta,mantissa_bits=53,admission_lhs_decimal=str(lhs),admission_limit_decimal=str(1 << 53),
        coefficient_max_decimal=str(slab*(alpha-1)*(beta-1)),weights_stride=4,weights=weights,
        weighting="overflow_safe_integer_modular_recombination",matrix_engine=False)


def load_cases(path):
    document=strict_json(Path(path).read_text(encoding="utf-8-sig"))
    demand(type(document) is dict and set(document)=={"schema","cases"} and document["schema"]=="rns8_word_mod_cases_v1","invalid case manifest")
    cases=document["cases"];demand(type(cases) is list and 1<=len(cases)<=1024,"case count must be1..1024")
    cases=[validate_case(case) for case in cases];identities=[json.dumps(case,sort_keys=True) for case in cases]
    demand(len(identities)==len(set(identities)),"duplicate cases are not independent observations");return cases


def contract(case,run_id,repeats,warmups):
    m,n,k,p=case["m"],case["n"],case["k"],case["modulus"];ac=m*(k+2);bc=k*(n+3);cc=m*(n+3)+8
    canonical=case["method"]=="canonical_dot4";dot=canonical or case["method"]=="packed_dot4";wmma=case["method"]=="byte_wmma";floating=case["method"]=="asymmetric_binary64";cold=case["scope"]=="cold"
    operand_bytes=((p-1).bit_length()+7)//8
    result={"schema":"rns8_word_mod_microcase_v1","run_id":run_id,**case,"lda":k+2,"ldb":n+3,"ldc":n+3,
        "operand_bits":(p-1).bit_length(),"modulus_bits":p.bit_length(),"device_requested":case["device"],"device_actual":case["device"],
        "backend_requested":"hip-direct","backend_actual":"hip-direct","seed":6651,
        "dataset_recipe":"splitmix64_word_mod_seed6651_canonical_extrema_padding_v1","repeats":repeats,"warmups":warmups,
        "input_contract":"canonical_u64_padded_rows","output_contract":"canonical_u64_modulus_with_padding","output_boundary":"host" if cold else "device",
        "reference_method":"boost_int256_all_products_modulus_with_cpp_int_row_edge_audits","a_allocation_bytes":ac*8,"b_allocation_bytes":bc*8,
        "output_allocation_bytes":cc*8,"explicit_scratch_peak_bytes":4,"compiler_private_scratch_bytes":None,"canary_bytes":64,"logical_payload_envelope_bytes":(3*(ac+bc)+4*cc)*8+16384,
        "memory_budget_bytes":BUDGET,"slab_k":case["slab_k"] if floating else 4096 if dot or wmma else None,"digit_diagonals":2*operand_bytes-1 if wmma or canonical else 15 if dot else 0,
        "max_diagonal_accumulator":4096*operand_bytes*65025 if wmma or canonical else 2130739200 if dot else None,
        "arithmetic_kernel":"word_mod_byte_dot4_canonical_width" if canonical else "word_mod_byte_wmma" if wmma else "word_mod_asymmetric_binary64" if floating else "word_mod_byte_dot4" if dot else "word_mod_double_add",
        "timing_scope":"allocate_upload_checked_compute_readback_checked_free" if cold else "resident_checked_compute_including_canonical_scan_and_domain_admission",
        "input_preservation_scope":"one_separate_resident_audit_before_warmups" if cold else "outside_each_verified_call_and_final_burst",
        "timer":"std_chrono_steady_clock_nanoseconds","production_timing_hooks_enabled":False,"public_ownership_measured":False,
        "asymmetric_floating_schedule_implemented":floating,"actual_hbm_bytes":None,"timing_resolution_ns":None,"isa_verified":False,"promotion_eligible":False}
    if floating:result["floating_schedule"]=floating_schedule(case)
    if canonical:
        result["canonical_dot4_schedule"]=dict(version=1,bytes_per_operand=operand_bytes,diagonals=2*operand_bytes-1,
            source_dot4_updates_per_four_k=operand_bytes**2,slab_k=4096,maximum_accumulator=4096*operand_bytes*65025,
            width_proof="canonical_inputs_less_than_modulus",zero_pad_k_tail=True,full_width_control="packed_dot4",
            issued_instruction_count_measured=False)
    if wmma:
        result["operand_bytes"]=operand_bytes
        result["byte_wmma_schedule"]=dict(version=1,arithmetic="unsigned_radix256_wmma_diagonal_integer_recombination",bytes_per_operand=operand_bytes,
            diagonals=2*operand_bytes-1,slab_k=4096,maximum_accumulator=4096*operand_bytes*65025,tile_m=16,tile_n=16,tile_k=16,threads_per_block=32,
            output_tile_count=((m+15)//16)*((n+15)//16),zero_pad_spatial_and_k_tails=True,left_signed=False,right_signed=False,
            saturating_accumulator=False,native_operands_reread_per_diagonal=True,asymmetric_floating_decomposition=False)
    return result


def dataset(case):
    m,n,k,p=(case[key] for key in ("m","n","k","modulus"));state=6651
    def word():
        nonlocal state
        state=(state+0x9e3779b97f4a7c15)&MASK;z=state;z=((z^(z>>30))*0xbf58476d1ce4e5b9)&MASK
        z=((z^(z>>27))*0x94d049bb133111eb)&MASK;return z^(z>>31)
    def fill(rows,cols,ld):
        data=[MASK]*(rows*ld);edges=(0,1,p-1,p//2);index=0
        for row in range(rows):
            for col in range(cols):
                raw=word();data[row*ld+col]=edges[index%257] if index%257<4 else raw%p;index+=1
        return data
    a=fill(m,k,k+2);b=fill(k,n,n+3);gold=[MASK]*(m*(n+3)+8)
    for row in range(m):
        for col in range(n):gold[row*(n+3)+col]=sum(a[row*(k+2)+q]*b[q*(n+3)+col] for q in range(k))%p
    return tuple(b"".join(value.to_bytes(8,"little") for value in values) for values in (a,b,gold))


@lru_cache(maxsize=32)
def reference_checksums(m,n,k,modulus):
    # Cache only three short digests, never payload arrays or sampled evidence.
    return tuple(checksum(data) for data in dataset(dict(m=m,n=n,k=k,modulus=modulus)))


def validate_capture(capture,case,run_id,repeats,warmups):
    demand(type(capture) is dict,"capture must be object")
    for key,value in contract(case,run_id,repeats,warmups).items():
        demand(key in capture and same(capture[key],value),f"metadata mismatch: {key}")
    if case["method"]!="asymmetric_binary64":
        demand(not {"floating_schedule","left_digits","right_digits"}&capture.keys(),"irrelevant floating schedule on integer method")
    if case["method"]!="byte_wmma":demand(not {"operand_bytes","byte_wmma_schedule"}&capture.keys(),"irrelevant WMMA schedule on another method")
    for key in ("device_name","target"):
        demand(type(capture.get(key)) is str and bool(capture[key].strip()),f"missing identity: {key}")
    demand(re.fullmatch(r"gfx[0-9a-f]+(?::[A-Za-z0-9_+:-]+)?",capture["target"]),"invalid target")
    if case["method"] in ("packed_dot4","canonical_dot4","byte_wmma"):demand(capture["target"].split(":")[0] in ("gfx1100","gfx1101","gfx1102"),"byte-instruction target not admitted")
    demand(integer(capture.get("runtime_version"),1,(1 << 31)-1),"invalid runtime identity")
    demand(positive(capture.get("oracle_and_dataset_ns")),"invalid oracle timing")
    demand(positive(capture.get("resident_setup_ns")) if case["scope"]=="resident" else capture.get("resident_setup_ns","absent") is None,"invalid setup scope")
    expected=reference_checksums(*(case[key] for key in ("m","n","k","modulus")))
    demand(tuple(capture.get(key) for key in ("a_checksum","b_checksum","output_checksum"))==expected,"full replay checksum mismatch")
    samples=capture.get("samples");demand(type(samples) is list and len(samples)==repeats,"invalid sample count")
    for sample in samples:
        demand(type(sample) is dict and positive(sample.get("host_call_ns")) and positive(sample.get("uninstrumented_host_call_ns"))
            and sample.get("exact_reference_passed") is True and sample.get("uninstrumented_exact_reference_passed") is True
            and type(sample.get("events")) is list,"invalid sample evidence")
        for event in sample["events"]:
            demand(type(event) is dict and type(event.get("label")) is str and bool(event["label"])
                and "elapsed_ns" in event and (event["elapsed_ns"] is None or positive(event["elapsed_ns"])),"invalid HIP event")
        if case["method"]=="byte_wmma":
            arithmetic=[event for event in sample["events"] if event["label"]=="word_mod_byte_wmma"]
            demand(len(arithmetic)<=1,"duplicate selected arithmetic event")
            demand(not any(event["label"].startswith("word_mod_") and event["label"]!="word_mod_byte_wmma" for event in sample["events"]),"different arithmetic kernel in WMMA observation")
    sustained=capture.get("sustained")
    demand(type(sustained) is dict and integer(sustained.get("calls"),16,4096) and positive(sustained.get("elapsed_ns"))
        and positive(sustained.get("mean_call_ns")) and sustained.get("final_exact_reference_passed") is True,"invalid sustained evidence")
    demand(same(sustained.get("minimum_calls"),16) and same(sustained.get("maximum_calls"),4096)
        and same(sustained.get("target_ns"),250000000) and sustained.get("independent_audit_inside_interval") is False,"invalid sustained controls")
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
    campaign=str(uuid.uuid4());output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    tasks=[(case,repeat) for repeat in range(args.runs) for case in cases];random.Random(args.order_seed).shuffle(tasks)
    cohort=None;observations=[];outputs={}
    def unchanged():demand(digest(exe)==artifact and digest(args.cases)==case_hash
        and all(digest(path)==identities[path.name] for path in sources),"artifact/tools/cases changed during collection")
    for ordinal,(case,repeat) in enumerate(tasks):
        unchanged();run_id=str(uuid.uuid4())
        command=[str(exe),str(case["device"]),case["method"],case["scope"],*(str(case[key]) for key in ("m","n","k")),case["domain"],str(case["modulus"]),str(args.repeats),str(args.warmups)]
        if case["method"]=="asymmetric_binary64":command.extend(str(case[key]) for key in ("left_digits","right_digits","slab_k"))
        command.append(run_id)
        record=dict(schema="rns8_word_mod_collection_v1",campaign_id=campaign,run_id=run_id,case=case,command=command,ordinal=ordinal,
            independent_repeat=repeat,order_seed=args.order_seed,started_unix_ns=time.time_ns(),artifact_sha256=artifact,collector_sha256=identities[Path(__file__).name],
            tool_dependencies_sha256=identities,cases_sha256=case_hash,python_executable=sys.executable,python_sha256=python_hash,python_version=platform.python_version(),
            promotion_eligible=False,validated=False,provenance_complete=False,exit_code=None,stdout="",stderr="")
        path=output/f"{run_id}.json"
        try:
            try:
                result=subprocess.run(command,text=True,encoding="utf-8",errors="strict",capture_output=True,timeout=args.timeout,check=False)
                record.update(exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr)
            except subprocess.TimeoutExpired as error:
                record.update(stdout=(error.stdout.decode("utf-8",errors="replace") if isinstance(error.stdout,bytes) else error.stdout) or "",
                    stderr=(error.stderr.decode("utf-8",errors="replace") if isinstance(error.stderr,bytes) else error.stderr) or "",timeout_seconds=args.timeout)
                raise ValueError("word modular process timed out") from error
            demand(result.returncode==0,"word modular process failed")
            capture=strict_json(result.stdout);identity=validate_capture(capture,case,run_id,args.repeats,args.warmups)
            if cohort is None:cohort=identity
            demand(same(identity,cohort),"compiled build/device/runtime cohort mismatch")
            key=tuple(case[field] for field in ("m","n","k","modulus","domain"))
            demand(capture["output_checksum"]==outputs.setdefault(key,capture["output_checksum"]),"matched output mismatch")
            unchanged();record.update(validated=True,provenance_complete=manifest_provenance_complete(capture["build_manifest"]))
            observations.append(dict(case=case,run_id=run_id,independent_repeat=repeat,sustained_mean_ns=capture["sustained"]["mean_call_ns"],
                median_production_ns=statistics.median(sample["uninstrumented_host_call_ns"] for sample in capture["samples"])))
        except Exception as error:
            record["failure"]=str(error)
            with path.open("x",encoding="utf-8") as stream:json.dump(record,stream,indent=2,allow_nan=False)
            raise
        with path.open("x",encoding="utf-8") as stream:json.dump(record,stream,indent=2,allow_nan=False)
    summary=dict(schema="rns8_word_mod_campaign_v1",campaign_id=campaign,promotion_eligible=False,artifact_sha256=artifact,
        tool_dependencies_sha256=identities,cases_sha256=case_hash,cohort=cohort,independent_processes=len(observations),
        provenance_complete=manifest_provenance_complete(cohort["build_manifest"]),observations=observations,
        comparison_rule="same shape/modulus/domain/scope/source boundary; byte_wmma is integer radix256, floating digit orientation/slab stays explicit; independent processes only; scalar binary64 is not WMMA; no extrapolated speedup")
    with (output/f"{campaign}.summary.json").open("x",encoding="utf-8") as stream:json.dump(summary,stream,indent=2,allow_nan=False)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("exe","cases","output"):parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--runs",type=int,default=5);parser.add_argument("--repeats",type=int,default=9);parser.add_argument("--warmups",type=int,default=3)
    parser.add_argument("--timeout",type=int,default=600);parser.add_argument("--order-seed",type=int,default=6651)
    args=parser.parse_args()
    try:summary=collect(args)
    except (ValueError,OSError) as error:parser.error(str(error))
    print(f"Collected {summary['independent_processes']} exact word-modular processes; qualification remains pending");return 0

if __name__=="__main__":raise SystemExit(main())
