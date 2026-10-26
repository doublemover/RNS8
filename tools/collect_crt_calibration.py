"""Collect exact import/CRT controls in existing conversion executable; no promotion."""
from __future__ import annotations
import argparse
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
from modulus_set_search import DEFAULT_MODULI

BASE_CANDIDATES=("garner", "direct", "balanced", "fixed_dot4", "tile_balanced", "tile_convert_balanced", "tile_convert_garner",
            "import_scalar_signed", "import_scalar_unsigned", "import_dot4_signed", "import_dot4_unsigned", "fixed_wmma", "import_wmma_signed", "import_wmma_unsigned",
            "import_public_signed", "import_fixed_signed", "import_generic_signed",
            "import_public_padded_signed", "import_fixed_padded_signed", "import_generic_padded_signed")
QUARTER_CANDIDATES=tuple(kind+sign for kind in ("quarter_","garner_quarter_","direct_quarter_","balanced_quarter_",
    "quarter_narrow_","garner_narrow_","public_quarter_","public_garner_") for sign in ("signed","unsigned"))
CANDIDATES=BASE_CANDIDATES+QUARTER_CANDIDATES

SCOPES=("resident", "cold")
MASK=(1 << 64)-1
BUDGET=768*1024*1024
EDGES=(0, 1, MASK, 1 << 63, 0x8080808080808080, 0x7f8081ff007fff80)


def validate_case(case):
    demand(type(case) is dict and set(case)=={"candidate", "cells", "prefix", "scope"}, "case requires candidate,cells,prefix,scope")
    demand(case["candidate"] in CANDIDATES and case["scope"] in SCOPES, "invalid candidate/scope")
    demand(integer(case["cells"],1,1048576) and integer(case["prefix"],1,20), "invalid cells/prefix")
    demand(contract(case,"validation",1,1)["logical_payload_envelope_bytes"]<=BUDGET, "payload budget exceeded")
    return dict(case)


def load_cases(path):
    document=strict_json(Path(path).read_text(encoding="utf-8-sig"))
    demand(type(document) is dict and set(document)=={"schema","cases"} and document["schema"]=="rns8_conversion_cases_v1", "invalid case manifest")
    cases=document["cases"]
    demand(type(cases) is list and 1<=len(cases)<=1024, "case count must be1..1024")
    cases=[validate_case(case) for case in cases]
    identities=[json.dumps(case,sort_keys=True) for case in cases]
    demand(len(identities)==len(set(identities)), "duplicate cases are not independent observations")
    return cases


def contract(case,run_id,repeats,warmups):
    candidate,cells,prefix,scope=(case[key] for key in ("candidate","cells","prefix","scope"))
    public=candidate.startswith("public_")
    quarter=candidate in QUARTER_CANDIDATES;narrow="narrow" in candidate or public
    importing=candidate.startswith("import_");tile=candidate.startswith("tile_")
    cols=256 if cells%256==0 else cells;rows=cells//cols
    padded="_padded_" in candidate;ld=cols+(7 if padded else 0);native_bytes=rows*ld*8
    pack_control=candidate.startswith(("import_public_","import_fixed_","import_generic_"))
    pack_kernel=None
    if pack_control:
        pack_kernel=("generic_per_residue" if candidate.startswith("import_generic_") else
            "fixed_prefix_one_load" if candidate.startswith("import_fixed_") else
            "four_wide_per_plane" if ld==cols and 256<=cells<=4096 else
            "eight_wide_per_plane" if ld==cols and cells>4096 else
            "small_per_residue" if cells<=4096 else "fixed_prefix_one_load" if prefix<=9 or prefix in (18,20) else "generic_per_residue")
    tile_bytes=((rows+7)//8)*((cols+15)//16)*128*prefix if tile else 0
    plain=cells*prefix;source=native_bytes if importing else tile_bytes if tile else plain
    output=cells*(prefix if importing else 8 if narrow else 24)+64
    buffers=dict(native=native_bytes if importing else 0,
        plain=plain if not importing and candidate!="tile_balanced" else 0,tile=tile_bytes,
        output=output,status=4 if candidate in ("garner","tile_convert_garner") or candidate.startswith("garner_") else 0,
        probe=4 if "dot4" in candidate or "wmma" in candidate else 0)
    envelope=native_bytes+plain+output*3+tile_bytes+source+sum(buffers.values())+16384
    result={"schema":"rns8_crt_microcase_v4", "run_id":run_id, **case,
        "seed":3903,"dataset_recipe":("splitmix64_quarter_native64_seed3903_v1" if narrow else "splitmix64_quarter_wide_seed3903_v1") if quarter else "splitmix64_conversion_seed3903_extrema_v1", "repeats":repeats,"warmups":warmups,
        "device_requested":0,"device_actual":0,"backend_requested":"hip-direct","backend_actual":"hip-direct",
        "timer":"std_chrono_steady_clock_nanoseconds","production_timing_hooks_enabled":False,
        "public_ownership_measured":False,"complete_output_and_canaries_checked":True,
        "input_preservation_scope":"one_separate_resident_audit_before_warmups" if scope=="cold" else "outside_each_verified_call_and_final_burst",
        "operation":"native_import" if importing else "narrow_crt" if narrow else "canonical_crt",
        "signed_inputs":candidate.endswith("_signed") if importing or quarter else None,
        "input_contract":"native64_bits" if importing else "quarter_range_centered_residue_planes" if quarter else "tile_centered_residue_planes" if tile else "centered_residue_planes",
        "output_contract":"centered_residue_planes" if importing else ("signed_one_limb" if candidate.endswith("_signed") else "unsigned_one_limb") if narrow else "canonical_three_limbs", "output_boundary":"host" if scope=="cold" else "device",
        "reference_method":"boost_cpp_int_all_operand_residues" if importing else "boost_cpp_int_all_native64_values" if narrow else "boost_cpp_int_all_canonical_values",
        "canary_bytes":64,"source_bytes_modeled":source,"source_physical_bytes":source,
        "source_layout":("native64_padded7_v1" if padded else "native64_compact_v1") if importing else "tile_major_8x16_v1" if tile else "modulus_major_v1",
        "logical_rows":rows,"logical_cols":cols,"logical_ld":ld,"pack_kernel":pack_kernel,
        "public_selector_measured":candidate.startswith("import_public_"),"device_buffer_bytes":buffers,
        "conversion_inside_call":tile and candidate!="tile_balanced",
        "conversion_scratch_bytes":plain if tile and candidate!="tile_balanced" else 0,
        "output_bytes_modeled":cells*(prefix if importing else 8 if narrow else 24),"output_allocation_bytes":output,
        "probe_bytes":buffers["probe"],"common_status_capacity_bytes":buffers["status"],
        "logical_payload_envelope_bytes":envelope,"memory_budget_bytes":BUDGET,
        "timing_scope":"allocate_upload_convert_readback_checked_free" if scope=="cold" else "resident_checked_convert_with_sync_no_upload_or_readback",
        "cache_state":"fresh_buffers_each_call" if scope=="cold" else "repeated_same_buffers",
        "actual_hbm_bytes":None,"timing_resolution_ns":None,"isa_verified":False,"promotion_eligible":False}
    if public:
        for key in ("device_buffer_bytes","probe_bytes","common_status_capacity_bytes"):
            result.pop(key)
        result.update(public_ownership_measured=True,operation="public_narrow_crt",output_boundary="host",
            plan_admission_policy="fixed_prefix_current_value_bound_export_only",gemm_measured=False,
            input_contract="engine_current_quarter_bound_residue_matrix",logical_rows=1,logical_cols=cells,logical_ld=cells,
            logical_payload_envelope_bytes=plain*3+output*5+cells*8+16384,
            timing_scope="public_context_plan_pack_prepare_export_checked_free" if scope=="cold" else "public_current_source_checked_export_stage_readback_publish")
    return result


def dataset(case):
    """Independent integer replay of every input/output byte, including canaries."""
    importing=case["candidate"].startswith("import_");signed=case["candidate"].endswith("_signed")
    quarter=case["candidate"] in QUARTER_CANDIDATES;narrow="narrow" in case["candidate"] or case["candidate"].startswith("public_")
    cells,prefix=case["cells"],case["prefix"];state=3903;product=math.prod(DEFAULT_MODULI[:prefix])
    cols=256 if cells%256==0 else cells;rows=cells//cols;ld=cols+(7 if "_padded_" in case["candidate"] else 0)
    native=bytearray([99])*(rows*ld*8);plain=bytearray(cells*prefix)
    width=prefix if importing else 8 if narrow else 24
    expected=bytearray([99])*(cells*width+64)
    bound=min(product//4,(1 << (63 if signed else 64))-1) if narrow else product//4
    def word():
        nonlocal state
        state=(state+0x9e3779b97f4a7c15)&MASK;z=state
        z=((z^(z>>30))*0xbf58476d1ce4e5b9)&MASK;z=((z^(z>>27))*0x94d049bb133111eb)&MASK
        return z^(z>>31)
    for cell in range(cells):
        r0,r1,r2=word(),word(),word();raw=EDGES[cell%257] if cell%257<6 else r0
        offset=((cell//cols)*ld+cell%cols)*8
        native[offset:offset+8]=raw.to_bytes(8,"little")
        value=(raw-(1 << 64) if signed and raw>>63 else raw) if importing else (r0+(r1 << 64)+(r2 << 128))%product
        if quarter:
            value=(r0+(r1 << 64)+(r2 << 128))%(bound+1)
            if cell<3:value=(0,bound,bound-1)[cell]
            if signed and cell%2:value=-value
        elif not importing and cell<3: value=(0,product-1,product//2)[cell]
        for p,modulus in enumerate(DEFAULT_MODULI[:prefix]):
            residue=value%modulus
            plain[p*cells+cell]=(residue-modulus if residue>=(modulus+1)//2 else residue)&255
        if not importing:
            represented=value+((1 << 64) if narrow else product) if value<0 else value
            expected[cell*width:cell*width+width]=represented.to_bytes(width,"little")
    if importing: expected[:len(plain)]=plain;return native,expected
    if not case["candidate"].startswith("tile_"): return plain,expected
    cols=256 if cells%256==0 else cells;rows=cells//cols;gc=(cols+15)//16
    tiled=bytearray([99])*(((rows+7)//8)*gc*prefix*128)
    for p in range(prefix):
        for row in range(rows):
            for col in range(cols):
                tile=(row//8)*gc+col//16
                tiled[(((tile*prefix+p)*8+row%8)*16+col%16)]=plain[(p*rows+row)*cols+col]
    return tiled,expected


def validate_capture(capture,case,run_id,repeats,warmups):
    demand(type(capture) is dict,"capture must be object")
    for key,value in contract(case,run_id,repeats,warmups).items():
        demand(key in capture and same(capture[key],value), f"metadata mismatch: {key}")
    for key in ("device","target"):
        demand(type(capture.get(key)) is str and bool(capture[key].strip()),f"missing identity: {key}")
    demand(re.fullmatch(r"gfx[0-9a-f]+(?::[A-Za-z0-9_+:-]+)?",capture["target"]),"invalid target")
    if "dot4" in case["candidate"] or "wmma" in case["candidate"]:
        demand(capture["target"].split(":")[0] in ("gfx1100","gfx1101","gfx1102"),"fixed conversion target not admitted")
    demand(integer(capture.get("runtime_version"),1,(1 << 31)-1),"invalid runtime identity")
    demand(positive(capture.get("oracle_and_dataset_ns")),"invalid oracle timing")
    demand(positive(capture.get("resident_setup_ns")) if case["scope"]=="resident" else capture.get("resident_setup_ns","absent") is None,"invalid setup scope")
    source,expected=dataset(case)
    demand(capture.get("input_checksum")==checksum(source) and capture.get("output_checksum")==checksum(expected),"recipe/full output checksum mismatch")
    samples=capture.get("samples")
    demand(type(samples) is list and len(samples)==repeats,"invalid samples")
    for sample in samples:
        demand(type(sample) is dict and positive(sample.get("host_call_ns")) and positive(sample.get("uninstrumented_host_call_ns"))
            and sample.get("exact_reference_passed") is True and sample.get("uninstrumented_exact_reference_passed") is True
            and type(sample.get("events")) is list,"invalid sample evidence")
        for event in sample["events"]:
            demand(type(event) is dict and type(event.get("label")) is str and bool(event["label"])
                and "elapsed_ns" in event and (event["elapsed_ns"] is None or positive(event["elapsed_ns"])),"invalid HIP timing")
    sustained=capture.get("sustained")
    demand(type(sustained) is dict and integer(sustained.get("calls"),16,4096) and positive(sustained.get("elapsed_ns"))
        and positive(sustained.get("mean_call_ns")) and sustained.get("final_exact_reference_passed") is True,"invalid sustained evidence")
    demand(same(sustained.get("minimum_calls"),16) and same(sustained.get("maximum_calls"),4096)
        and same(sustained.get("target_ns"),250000000) and sustained.get("independent_audit_inside_interval") is False,"invalid sustained bounds/audit scope")
    demand(math.isclose(sustained["mean_call_ns"]*sustained["calls"],sustained["elapsed_ns"],rel_tol=1e-12)
        and (sustained["calls"]==4096 or sustained["elapsed_ns"]>=250000000),"invalid sustained duration")
    demand(sustained.get("scope")==("fresh_buffers_transfer_inclusive_no_intermediate_reference" if case["scope"]=="cold" else
        "resident_same_buffers_no_intermediate_poison_or_reference"),"sustained scope mismatch")
    if case["candidate"] in QUARTER_CANDIDATES:
        narrow="narrow" in case["candidate"] or case["candidate"].startswith("public_")
        bound=math.prod(DEFAULT_MODULI[:case["prefix"]])//4
        if narrow:bound=min(bound,(1 << (63 if case["candidate"].endswith("_signed") else 64))-1)
        if case["candidate"].startswith("public_"):
            actual=0 if case["cells"]==1 else bound
            demand(capture.get("certified_magnitude_bound")==str(actual),"public current magnitude proof mismatch")
            if case["scope"]=="resident":
                demand(integer(capture.get("retained_public_device_payload_bytes"),case["cells"]*case["prefix"],BUDGET),"public retained bytes missing")
                if case["candidate"].startswith("public_quarter_"):
                    demand(integer(capture.get("quarter_owner_payload_bytes"),case["cells"]*16,BUDGET),"quarter staging payload mismatch")
                    demand(re.fullmatch(r"[0-9a-f]{64}",capture.get("quarter_certificate_sha256","")),"quarter owner digest missing")
                    demand(capture.get("delayed_digits")==max(1,(actual.bit_length()+1+15)//16),"quarter narrow digits mismatch")
        else:
            q=capture.get("quarter_certificate",{})
            expected_mode=(1 if case["candidate"].endswith("_unsigned") else 0) if narrow else 2
            demand(type(q) is dict and q.get("magnitude_bound")==str(bound) and q.get("output_mode")==expected_mode
                and q.get("digits")== (max(1,(bound.bit_length()+1+15)//16) if narrow else 12)
                and q.get("error_numerator")==168940 and q.get("error_denominator")==33554432
                and q.get("target_execution")=="not_run" and q.get("source_scope")=="private_fixture_immutable_for_process"
                and positive(q.get("setup_ns")) and re.fullmatch(r"[0-9a-f]{64}",q.get("sha256","")),"quarter proof contract mismatch")
    validate_provenance(capture.get("build_manifest"))
    return {key:capture[key] for key in ("build_manifest","device","target","runtime_version")}


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collect(args):
    demand(integer(args.runs,1,100) and integer(args.repeats,1,10000) and integer(args.warmups,1,1000)
        and integer(args.timeout,1,86400) and type(args.order_seed) is int,"invalid collection bounds")
    initial_case_hash=digest(args.cases) if args.cases else None
    if args.cases:
        cases=load_cases(args.cases)
        demand(digest(args.cases)==initial_case_hash,"cases changed while loading")
    else:
        cases=[validate_case(dict(candidate=c,cells=n,prefix=p,scope=s)) for c in args.candidates for n in args.cells for p in args.prefixes for s in args.scopes]
        demand(cases and len({json.dumps(c,sort_keys=True) for c in cases})==len(cases),"duplicate/empty cases")
    exe=Path(args.exe).resolve(strict=True);artifact=digest(exe)
    sources=[Path(__file__),Path(__file__).with_name("collect_finite_verify.py"),Path(__file__).with_name("build_manifest.py"),Path(__file__).with_name("modulus_set_search.py")]
    identities={path.name:digest(path) for path in sources};case_hash=initial_case_hash
    python_hash=digest(sys.executable);campaign=str(uuid.uuid4());output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    tasks=[(case,repeat) for repeat in range(args.runs) for case in cases];random.Random(args.order_seed).shuffle(tasks)
    cohort=None;checksums={};observations=[]
    def unchanged():
        demand(digest(exe)==artifact and all(digest(path)==identities[path.name] for path in sources)
            and (not args.cases or digest(args.cases)==case_hash),"executable/tools/cases changed during collection")
    for ordinal,(case,repeat) in enumerate(tasks):
        unchanged();run_id=str(uuid.uuid4());command=[str(exe),case["candidate"],str(case["cells"]),str(case["prefix"]),str(args.repeats),str(args.warmups),run_id,case["scope"]]
        record=dict(schema="rns8_crt_collection_v3",campaign_id=campaign,run_id=run_id,case=case,command=command,ordinal=ordinal,
            independent_repeat=repeat,order_seed=args.order_seed,started_unix_ns=time.time_ns(),artifact_sha256=artifact,
            collector_sha256=identities[Path(__file__).name],tool_dependencies_sha256=identities,cases_sha256=case_hash,
            python_executable=sys.executable,python_sha256=python_hash,python_version=platform.python_version(),promotion_eligible=False,
            validated=False,provenance_complete=False,exit_code=None,stdout="",stderr="")
        path=output/f"{run_id}.json"
        try:
            try:
                result=subprocess.run(command,text=True,encoding="utf-8",errors="strict",capture_output=True,timeout=args.timeout,check=False)
                record.update(exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr)
            except subprocess.TimeoutExpired as error:
                record.update(exit_code=None,stdout=(error.stdout.decode("utf-8",errors="replace") if isinstance(error.stdout,bytes) else error.stdout) or "",
                    stderr=(error.stderr.decode("utf-8",errors="replace") if isinstance(error.stderr,bytes) else error.stderr) or "",timeout_seconds=args.timeout)
                raise ValueError("conversion process timed out") from error
            demand(result.returncode==0,"conversion process failed")
            capture=strict_json(result.stdout);identity=validate_capture(capture,case,run_id,args.repeats,args.warmups)
            if cohort is None: cohort=identity
            demand(same(identity,cohort),"compiled build/device/runtime cohort mismatch")
            # Import signedness is part of the semantics; CRT layout/candidate is
            # a transport difference, so every canonical control must match.
            key=(case["cells"],case["prefix"],capture["operation"],capture["signed_inputs"],capture["dataset_recipe"],capture["output_contract"])
            expected=checksums.setdefault(key,capture["output_checksum"])
            demand(capture["output_checksum"]==expected,"matched output checksum mismatch")
            unchanged();record.update(validated=True,provenance_complete=manifest_provenance_complete(capture["build_manifest"]))
            observations.append(dict(case=case,run_id=run_id,independent_repeat=repeat,sustained_mean_ns=capture["sustained"]["mean_call_ns"],
                median_production_ns=statistics.median(sample["uninstrumented_host_call_ns"] for sample in capture["samples"])))
        except Exception as error:
            record["failure"]=str(error)
            with path.open("x",encoding="utf-8") as stream: json.dump(record,stream,indent=2,allow_nan=False)
            raise
        with path.open("x",encoding="utf-8") as stream: json.dump(record,stream,indent=2,allow_nan=False)
    summary=dict(schema="rns8_conversion_campaign_v1",campaign_id=campaign,promotion_eligible=False,artifact_sha256=artifact,
        tool_dependencies_sha256=identities,cases_sha256=case_hash,cohort=cohort,independent_processes=len(observations),provenance_complete=manifest_provenance_complete(cohort["build_manifest"]),
        observations=observations,comparison_rule="compare same operation/signedness/cells/prefix/scope; repeats are not independent processes; no inferred speedup or break-even")
    with (output/f"{campaign}.summary.json").open("x",encoding="utf-8") as stream: json.dump(summary,stream,indent=2,allow_nan=False)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--cases",type=Path,help="versioned explicit conversion cases; otherwise use cartesian controls below")
    parser.add_argument("--cells",type=int,nargs="+",default=[1,257,4096]);parser.add_argument("--prefixes",type=int,nargs="+",default=list(range(1,21)))
    parser.add_argument("--candidates",nargs="+",choices=CANDIDATES,default=["garner","direct","balanced"])
    parser.add_argument("--scopes",nargs="+",choices=SCOPES,default=["resident"])
    parser.add_argument("--runs",type=int,default=5);parser.add_argument("--repeats",type=int,default=10);parser.add_argument("--warmups",type=int,default=3)
    parser.add_argument("--timeout",type=int,default=300);parser.add_argument("--order-seed",type=int,default=3903)
    args=parser.parse_args()
    try: summary=collect(args)
    except (ValueError,OSError) as error: parser.error(str(error))
    print(f"Collected {summary['independent_processes']} exact conversion processes; qualification remains pending")
    return 0

if __name__=="__main__": raise SystemExit(main())
