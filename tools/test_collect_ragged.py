"""Authored synthetic ragged receipts; no runtime measurements or qualification."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from collect_ragged import (MODES,KEYS,MASK,BUDGET,collect,contract,envelope,factor,load_cases,native,reference,signed,
    validate_capture,validate_case,values,word_checksum)
from test_capture_identity_support import synthetic_identity

BASE=dict(backend="cpu-reference",device=-1,mode="ragged_export",rows=17,cols=19,pattern="mixed",domain="signed",tile_rows=3,tile_cols=5,reuse=2)


def fixture(case=None,run_id="fixture-run",repeats=2,warmups=1):
    case=case or BASE;capture=contract(case,run_id,repeats,warmups);synthetic_identity(capture)
    hip=case["backend"]=="hip-direct";uniform=case["mode"]=="uniform_export";wide=case["pattern"]=="wide140";cells=case["rows"]*case["cols"]
    def allocation():return dict(logical_primary_bytes=20*cells,host_payload_capacity_bytes=0 if hip else 20*cells,
        device_payload_capacity_bytes=20*cells if hip else 0,device_upload_staging_bytes=0,device_export_staging_bytes=0,device_status_and_schedule_bytes=0,device_total_capacity_bytes=20*cells if hip else 0)
    initial=None;first=None
    if not uniform:
        model=capture["modeled_layout"];directory=model["tile_count"]*40
        initial={key:model[key] for key in ("tile_count","payload_bytes","minimum_prefix","maximum_prefix")}
        initial.update(directory_bytes=directory,source_instance=1,source_generation=4,device_metadata_bytes=directory+4 if hip else 0,materialization_capacity_bytes=0)
        first=dict(initial,materialization_capacity_bytes=20*cells if hip else 0)
    directory=0 if uniform else initial["directory_bytes"]
    capture.update(device_name="synthetic CPU/GPU only",target="gfx1100" if hip else "cpu",hip_runtime_version=70100000 if hip else 0,hip_driver_version=70100000 if hip else 0,
        oracle_ns=100,setup_ns=100,first_batch_ns=100,setup_plus_first_batch_ns=200,clock_period_numerator=1,clock_period_denominator=1000000000,
        first_prepare_ns=0 if uniform else 20,owner_destruction_ns=0 if uniform else 20,source_construction_ns=50 if wide else 0,source_construction_workspace_bytes=0,
        logical_payload_envelope_bytes=envelope(case),source_instance=1,source_version=7 if wide else 1,output_instance=0 if uniform else 2,
        ragged_resources_at_prepare=initial,ragged_resources_after_first=first,ragged_resources_final=deepcopy(first),
        modeled_prepare_directory_d2h_bytes=directory if hip else 0,modeled_prepare_status_d2h_bytes=4 if hip and not uniform else 0,
        modeled_prepare_directory_h2d_bytes=directory if hip else 0,modeled_materialization_directory_h2d_bytes_per_call=0,modeled_materialization_precopy_d2d_bytes_per_call=0,modeled_materialization_commit_d2d_bytes_per_call=20*cells if hip and not uniform else 0,
        modeled_materialization_status_d2h_bytes_per_call=4 if hip and not uniform else 0,
        matrix_allocations_at_setup=[allocation() for _ in range(1 if uniform else 2)],matrix_allocations_after_first=[allocation() for _ in range(1 if uniform else 2)],matrix_allocations_final=[allocation() for _ in range(1 if uniform else 2)],
        preparation_samples=[] if uniform else [dict(preparation_ns=20,destruction_ns=10,resources=deepcopy(initial)) for _ in range(repeats)],
        samples=[dict(wall_ns=100+i,exact_reference_passed=True) for i in range(repeats)],median_batch_ns=100+(repeats-1)/2,
        materializations_per_second=1e9*case["reuse"]/(100+(repeats-1)/2),
        sustained=dict(calls=16,materializations=16*case["reuse"],elapsed_ns=320000000,mean_batch_ns=20000000.,minimum_calls=16,maximum_calls=4096,target_ns=250000000,stop_reason="duration",independent_audit_inside_interval=False,final_exact_reference_passed=True))
    return capture


def arguments(root,cases=None):
    root=Path(root);exe=root/"fake.exe";exe.write_bytes(b"synthetic binary")
    manifest=root/"cases.json";manifest.write_text(json.dumps(dict(schema="rns8_ragged_cases_v1",cases=cases or [BASE])))
    return argparse.Namespace(exe=exe,cases=manifest,output=root/"captures",runs=1,repeats=2,warmups=1,timeout=10,order_seed=5152)


def successful(command,**kwargs):
    case=dict(backend=command[1],device=int(command[2]),mode=command[3],rows=int(command[4]),cols=int(command[5]),pattern=command[6],domain=command[7],tile_rows=int(command[8]),tile_cols=int(command[9]),reuse=int(command[10]))
    return subprocess.CompletedProcess(command,0,json.dumps(fixture(case,command[13],int(command[11]),int(command[12]))),"")


class Captures(unittest.TestCase):
    def test_backends_modes_signedness_and_all_recipes(self):
        for backend,device in (("cpu-reference",-1),("hip-direct",0)):
            for mode in MODES:
                for pattern in ("mixed","fullwidth","wide140"):
                    for domain in ("signed","unsigned"):
                        case=validate_case(dict(BASE,backend=backend,device=device,mode=mode,pattern=pattern,domain=domain))
                        capture=fixture(case);validate_capture(capture,case,"fixture-run",2,1)
                        self.assertEqual(capture["source_checksum"],capture["output_checksum"])
                        self.assertIsNone(capture["observed_hbm_bytes"])

    def test_full_reference_and_mixed_prefix_neighbors(self):
        for domain in ("signed","unsigned"):
            for pattern in ("mixed","fullwidth","wide140"):
                case=dict(BASE,domain=domain,pattern=pattern);data=values(case);expected=[]
                for row in range(case["rows"]):
                    for col in range(case["cols"]):
                        if pattern=="wide140":
                            total=0
                            for _ in range(16384):total+=signed(case,factor(case,True,row))*signed(case,factor(case,False,col))
                        else:total=signed(case,native(case)[row*case["cols"]+col])
                        self.assertEqual(data[row*case["cols"]+col],total)
                        expected.extend((total>>(64*limb))&MASK for limb in range(3))
                ref=reference(*(case[key] for key in ("rows","cols","pattern","domain","tile_rows","tile_cols")))
                self.assertEqual(ref["source_checksum"],word_checksum(expected))
                if pattern=="mixed":self.assertEqual(ref["modeled_layout"]["minimum_prefix"],1);self.assertGreater(ref["modeled_layout"]["maximum_prefix"],8)
                if pattern=="wide140":self.assertIn(1<<140,data)

    def test_partial_tiles_padding_in_resource_model(self):
        for rows,cols,tr,tc in ((1,1,128,128),(17,19,3,5),(128,129,16,17)):
            case=dict(BASE,rows=rows,cols=cols,tile_rows=tr,tile_cols=tc)
            capture=fixture(case);validate_capture(capture,case,"fixture-run",2,1);m=capture["modeled_layout"]
            self.assertEqual(m["tile_count"],((rows+tr-1)//tr)*((cols+tc-1)//tc));self.assertEqual(m["payload_bytes"],tr*tc*sum(m["tile_prefixes"]))

    def test_generated_source_versions_are_positive_opaque_values(self):
        case=dict(BASE,pattern="wide140")
        for version in (1,7,12345):
            capture=fixture(case);capture["source_version"]=version
            validate_capture(capture,case,"fixture-run",2,1)
        for version in (0,False,-1,"7"):
            capture=fixture(case);capture["source_version"]=version
            with self.assertRaises(ValueError):validate_capture(capture,case,"fixture-run",2,1)

    def test_invalid_explicit_controls_and_strict_manifests(self):
        for key in KEYS:
            bad=dict(BASE);del bad[key]
            with self.assertRaises(ValueError):validate_case(bad)
        for key,value in (("backend","auto"),("device",0),("device",True),("mode","copy"),("rows",0),("cols",513),("pattern","random"),("domain","i64"),("tile_rows",0),("tile_cols",129),("reuse",False),("reuse",257)):
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):validate_case(dict(BASE,**{key:value}))
        for case in (dict(BASE,backend="hip-direct",device=-1),dict(BASE,pattern="wide140",rows=512,cols=512),dict(BASE,rows=512,cols=512,reuse=256)):
            with self.assertRaises(ValueError):validate_case(case)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"cases.json"
            for text in ('[]','{"schema":"rns8_ragged_cases_v1","cases":[]}',json.dumps(dict(schema="rns8_ragged_cases_v1",cases=[BASE,BASE])),
                '{"schema":"rns8_ragged_cases_v1","schema":"duplicate","cases":[]}','{"schema":NaN}'):
                path.write_text(text)
                with self.assertRaises(ValueError):load_cases(path)

    def test_every_contract_field_required_and_metadata_replayed(self):
        for key in contract(BASE,"fixture-run",2,1):
            capture=fixture();del capture[key]
            with self.subTest(key=key),self.assertRaises(ValueError):validate_capture(capture,BASE,"fixture-run",2,1)
        for key,value in (("mode","uniform_export"),("domain","unsigned"),("output_checksum","f"*16),("native_input_checksum","f"*16),("source_instance",0),
            ("source_version",2),("output_instance",1),("source_construction_k",16384),("logical_payload_envelope_bytes",0),("observed_hbm_bytes",1),
            ("modeled_compact_payload_h2d_bytes",1),("modeled_full_uniform_snapshot_d2h_bytes",20),("kernel_events",[]),("exports_per_batch",0)):
            capture=fixture();capture[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_capture(capture,BASE,"fixture-run",2,1)
        capture=fixture();capture["modeled_layout"]["tile_prefixes"][0]=20
        with self.assertRaises(ValueError):validate_capture(capture,BASE,"fixture-run",2,1)

    def test_retained_resource_and_allocation_mismatch_rejection(self):
        for field,value in (("payload_bytes",1),("directory_bytes",1),("minimum_prefix",0),("maximum_prefix",1),("device_metadata_bytes",4),("materialization_capacity_bytes",1),("source_generation",0),("source_instance",2)):
            capture=fixture();capture["ragged_resources_final"][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):validate_capture(capture,BASE,"fixture-run",2,1)
        for boundary in ("matrix_allocations_at_setup","matrix_allocations_after_first","matrix_allocations_final"):
            capture=fixture();capture[boundary][0]["device_total_capacity_bytes"]=1
            with self.assertRaises(ValueError):validate_capture(capture,BASE,"fixture-run",2,1)
        hip=dict(BASE,backend="hip-direct",device=0);capture=fixture(hip);capture["modeled_prepare_status_d2h_bytes"]=0
        with self.assertRaises(ValueError):validate_capture(capture,hip,"fixture-run",2,1)

    def test_timing_scope_and_full_audits_required(self):
        for key,value in (("first_prepare_ns",0),("first_batch_ns",0),("setup_plus_first_batch_ns",1),("preparation_samples",[]),("samples",[]),("median_batch_ns",101.),("materializations_per_second",1.),("first_exact_reference_passed",False)):
            capture=fixture();capture[key]=value
            with self.assertRaises(ValueError):validate_capture(capture,BASE,"fixture-run",2,1)
        for key,value in (("calls",15),("materializations",1),("elapsed_ns",1),("stop_reason","call_cap"),("independent_audit_inside_interval",True),("final_exact_reference_passed",False)):
            capture=fixture();capture["sustained"][key]=value
            with self.assertRaises(ValueError):validate_capture(capture,BASE,"fixture-run",2,1)

    def test_explicit_cpu_and_hip_case_sets(self):
        root=Path(__file__).resolve().parents[1]/"benchmarks/ragged"
        for backend,device,prefix in (("cpu-reference",-1,"cpu"),("hip-direct",0,"hip")):
            for kind in ("smoke","large"):
                cases=load_cases(root/(prefix+"-"+kind+".json"));self.assertEqual({c["mode"] for c in cases},set(MODES))
                self.assertEqual({c["backend"] for c in cases},{backend});self.assertEqual({c["device"] for c in cases},{device})
                self.assertEqual({c["domain"] for c in cases},{"signed","unsigned"})
            self.assertEqual({c["pattern"] for c in load_cases(root/(prefix+"-smoke.json"))},{"mixed","fullwidth","wide140"})


class Collection(unittest.TestCase):
    def test_matched_host_boundary_and_distinct_resident_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            args=arguments(directory,[dict(BASE,mode=mode) for mode in MODES]);args.runs=2
            with patch("collect_ragged.subprocess.run",side_effect=successful):summary=collect(args)
            self.assertEqual(summary["independent_processes"],6);self.assertFalse(summary["promotion_eligible"])
            self.assertTrue(summary["matched_cohorts"][0]["matched_host_boundary_available"])
            for path in args.output.glob("*.json"):
                if path.name.endswith(".summary.json"):continue
                receipt=json.loads(path.read_text());self.assertTrue(receipt["validated"]);self.assertIn("modulus_set_search.py",receipt["tool_dependencies_sha256"])

    def test_end_to_end_failures_and_artifact_mutations_remain_receipts(self):
        for failure in ("run","output","recipe","resource","scope","build","timeout","exit","exe","cases","json"):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as directory:
                args=arguments(directory)
                def run(command,**kwargs):
                    if failure=="timeout":raise subprocess.TimeoutExpired(command,10,output=b"partial",stderr=b"timeout")
                    result=successful(command);capture=json.loads(result.stdout)
                    if failure=="run":capture["run_id"]="wrong"
                    elif failure=="output":capture["output_checksum"]="0"*16
                    elif failure=="recipe":capture["pattern"]="fullwidth"
                    elif failure=="resource":capture["ragged_resources_final"]["source_generation"]+=1
                    elif failure=="scope":capture["timed_scope"]="copy"
                    elif failure=="build":capture["build_manifest"]["source_digest"]="bad"
                    elif failure=="exit":result.returncode=77
                    elif failure=="exe":args.exe.write_bytes(b"changed")
                    elif failure=="cases":args.cases.write_text("[]")
                    result.stdout='{"schema":1,"schema":2}' if failure=="json" else json.dumps(capture);return result
                with patch("collect_ragged.subprocess.run",side_effect=run),self.assertRaises(ValueError):collect(args)
                receipts=list(args.output.glob("*.json"));self.assertEqual(len(receipts),1);receipt=json.loads(receipts[0].read_text());self.assertFalse(receipt["validated"]);self.assertIn("failure",receipt)

    def test_diagnostic_provenance_stays_ineligible(self):
        for kind in ("dirty","unknown","garner"):
            with tempfile.TemporaryDirectory() as directory:
                args=arguments(directory)
                def run(command,**kwargs):
                    result=successful(command);capture=json.loads(result.stdout)
                    if kind=="dirty":capture["build_manifest"]["source_dirty"]=True
                    elif kind=="unknown":capture["build_manifest"]["dependency_digest"]="unavailable"
                    else:capture["build_manifest"]["configuration"]+="\ngarner_table=ON"
                    result.stdout=json.dumps(capture);return result
                with patch("collect_ragged.subprocess.run",side_effect=run):summary=collect(args)
                self.assertFalse(summary["provenance_complete"]);self.assertFalse(summary["promotion_eligible"])

    def test_cross_process_build_identity_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args=arguments(directory);args.runs=2;count=0
            def run(command,**kwargs):
                nonlocal count
                result=successful(command);count+=1
                if count==2:
                    capture=json.loads(result.stdout);capture["build_manifest"]["source_digest"]="b"*64;result.stdout=json.dumps(capture)
                return result
            with patch("collect_ragged.subprocess.run",side_effect=run),self.assertRaises(ValueError):collect(args)


@unittest.skipUnless(os.environ.get("RNS8_RAGGED_EXE"),"compiled CLI supplied by final build")
class CompiledCli(unittest.TestCase):
    def test_help_and_precontext_rejections(self):
        exe=os.environ["RNS8_RAGGED_EXE"]
        def invoke(args):return subprocess.run([exe,*args],text=True,capture_output=True,timeout=10)
        result=invoke(["--help"]);self.assertEqual(result.returncode,0);self.assertIn("no synthetic copy baseline",result.stdout)
        base=["cpu-reference","-1","ragged_export","17","19","mixed","signed","3","5","2","2","1","fixture"]
        for index,value in ((0,"auto"),(1,"0"),(2,"copy"),(3,"0"),(4,"513"),(5,"random"),(6,"i64"),(7,"0"),(8,"129"),(9,"0"),(10,"0"),(11,"0"),(12,"bad run"),(3,"17.0")):
            args=base.copy();args[index]=value;self.assertEqual(invoke(args).returncode,1)
        self.assertEqual(invoke(base+["extra"]).returncode,1);self.assertEqual(invoke(base[:-1]).returncode,1)

if __name__=="__main__":unittest.main()
