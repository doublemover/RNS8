"""Authored mock receipts and pre-context CLI fixtures; no measured evidence."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from collect_split_k import (MODES,KEYS,MASK,BUDGET,XOR,collect,contract,load_cases,reference,resources,
    validate_case,validate_capture,words,word_checksum)
from test_capture_identity_support import synthetic_identity

BASE=dict(device=0,mode="window",m=2,n=3,k=17,domain="signed",fragments=3,window=2,reuse=2)


def fixture(case=None,run_id="fixture-run",repeats=2,warmups=1):
    case=case or BASE;capture=contract(case,run_id,repeats,warmups);synthetic_identity(capture)
    ab=case["m"]*case["k"]+case["k"]*case["n"];mn=case["m"]*case["n"];scratch=(resources(case) or {}).get("scratch_bytes",0)
    envelope=max(8*ab+ab*48+24*mn+2048,256*ab+256*mn+scratch+16384)
    allocations=[]
    for cells in (case["m"]*case["k"],case["k"]*case["n"],mn):
        allocations.append(dict(logical_primary_bytes=cells*20,host_payload_capacity_bytes=0,device_payload_capacity_bytes=cells*20,
            device_upload_staging_bytes=0,device_export_staging_bytes=0,device_status_and_schedule_bytes=0,device_total_capacity_bytes=cells*20))
    capture.update(device_name="synthetic mocked GPU",target="gfx1100",hip_runtime_version=70100000,hip_driver_version=70100000,
        clock_period_numerator=1,clock_period_denominator=1000000000,owner_release_ns=0 if case["mode"]=="unsplit" else 10,selected_unsplit_kernel="mock dense control",
        source_instance_ids=[1,2],output_instance_id=3,reference_fixed_operand_object_bytes=48,reference_operand_storage_bytes=48*ab,
        plan_workspace_bytes=0,logical_live_payload_envelope_bytes=envelope,matrix_allocations_at_setup=allocations)
    phases=[];split=case["mode"]!="unsplit"
    for gen in (1,2):
        a,b,out=reference(*(case[key] for key in ("m","n","k","domain")),gen)
        prepare=20 if split else 0;pack=10 if gen==2 else 0;setup=pack+prepare if gen==2 else 100
        phases.append(dict(generation=gen,source_versions=[gen,gen],binding_action=("initial_create" if split else "initial_pack") if gen==1 else ("repack_destroy_recreate" if split else "repack"),
            stale_owner_status="RNS8_INVALID_ARGUMENT" if split and gen==2 else None,stale_owner_preserved_output=True if split and gen==2 else None,
            a_checksum=a,b_checksum=b,output_checksum=out,oracle_ns=100,phase_setup_ns=setup,changed_input_pack_ns=pack,owner_prepare_or_recreate_ns=prepare,
            first_batch_ns=100,setup_plus_first_batch_ns=setup+100,first_batch_exact_reference_passed=True,
            samples=[dict(wall_ns=100+index,exact_reference_passed=True) for index in range(repeats)],median_batch_ns=100+(repeats-1)/2,
            gemm_calls_per_second=1e9*case["reuse"]/(100+(repeats-1)/2),matrix_allocations_after_audit=deepcopy(allocations),
            sustained=dict(calls=16,gemm_calls=16*case["reuse"],elapsed_ns=320000000,mean_batch_ns=20000000.,minimum_calls=16,maximum_calls=4096,target_ns=250000000,
                stop_reason="duration",independent_audit_inside_interval=False,final_exact_reference_passed=True)))
    capture["phases"]=phases;return capture


def arguments(root,cases=None):
    root=Path(root);exe=root/"fake.exe";exe.write_bytes(b"mock artifact")
    manifest=root/"cases.json";manifest.write_text(json.dumps(dict(schema="rns8_split_k_cases_v1",cases=cases or [BASE])))
    return argparse.Namespace(exe=exe,cases=manifest,output=root/"captures",runs=1,repeats=2,warmups=1,timeout=10,order_seed=5117)


def successful(command,**kwargs):
    case=dict(device=int(command[1]),mode=command[2],m=int(command[3]),n=int(command[4]),k=int(command[5]),domain=command[6],
        fragments=int(command[7]),window=int(command[8]),reuse=int(command[9]))
    return subprocess.CompletedProcess(command,0,json.dumps(fixture(case,command[12],int(command[10]),int(command[11]))),"")


class Captures(unittest.TestCase):
    def test_all_modes_signedness_tail_and_changed_bindings(self):
        for domain in ("signed","unsigned"):
            for mode in MODES:
                case=dict(BASE,domain=domain,mode=mode,fragments=1 if mode=="unsplit" else 3,window=0 if mode=="unsplit" else 3 if mode=="full" else 2)
                validate_case(case);capture=fixture(case);validate_capture(capture,case,"fixture-run",2,1)
                self.assertNotEqual(capture["phases"][0]["a_checksum"],capture["phases"][1]["a_checksum"])
                self.assertEqual(capture["phases"][0]["b_checksum"],capture["phases"][1]["b_checksum"])
                self.assertNotEqual(capture["phases"][0]["output_checksum"],capture["phases"][1]["output_checksum"])
                self.assertEqual(capture["phases"][0]["median_batch_ns"],100.5)

    def test_independent_reference_native_extrema_and_generation_mutation(self):
        for domain in ("signed","unsigned"):
            for k in (1,17,129):
                case=dict(BASE,m=1,n=1,k=k,domain=domain);a,b=words(case)
                def val(v):return v-(1<<64) if domain=="signed" and v>>63 else v
                for generation in (1,2):
                    aa=a.copy()
                    if generation==2:aa[0]^=XOR
                    total=0
                    for i in range(k):total+=val(aa[i])*val(b[i])
                    expected=[(total>>(64*limb))&MASK for limb in range(3)]
                    self.assertEqual(reference(1,1,k,domain,generation),(word_checksum(aa),word_checksum(b),word_checksum(expected)))
        self.assertEqual(reference(1,1,1,"signed",1)[2],word_checksum([0,1<<62,0]))

    def test_resource_formula_merge_boundaries_full_and_window_tails(self):
        for f in (1,2,127,128,129,4096):
            for w in (1,min(f,128),f):
                c=dict(BASE,k=65537,fragments=f,window=w,mode="full" if w==f else "window")
                r=resources(c);cells=20*c["m"]*c["n"]
                self.assertEqual(r["partial_bytes"],w*cells);self.assertEqual(r["merge_bytes"],((w+127)//128)*cells)
                self.assertEqual(r["running_bytes"],0 if w==f else cells)
                self.assertEqual(r["window_count"],(f+w-1)//w)
                self.assertEqual(r["scratch_bytes"],sum(r[key] for key in ("partial_bytes","merge_bytes","running_bytes")))
        self.assertEqual(resources(dict(BASE,fragments=4096,window=4096,mode="full"))["merge_levels"],2)

    def test_irrelevant_invalid_or_malformed_case_controls(self):
        for key in KEYS:
            bad=dict(BASE);del bad[key]
            with self.assertRaises(ValueError):validate_case(bad)
        for key,value in (("device",True),("device",64),("mode","auto"),("domain","i64"),("m",0),("n",513),("k",65538),
            ("fragments",True),("fragments",0),("fragments",18),("window",0),("window",3),("window",1.0),("reuse",False),("reuse",257)):
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):validate_case(dict(BASE,**{key:value}))
        for c in (dict(BASE,mode="unsplit"),dict(BASE,mode="full"),dict(BASE,k=65537,fragments=1,window=1,mode="full"),dict(BASE,m=512,n=512,k=65537)):
            with self.assertRaises(ValueError):validate_case(c)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"cases.json"
            for text in ('[]','{"schema":"rns8_split_k_cases_v1","cases":[]}',json.dumps(dict(schema="rns8_split_k_cases_v1",cases=[BASE,BASE])),
                '{"schema":"rns8_split_k_cases_v1","schema":"duplicate","cases":[]}','{"schema":NaN}'):
                path.write_text(text)
                with self.assertRaises(ValueError):load_cases(path)

    def test_every_contract_field_required_and_replayed(self):
        for key in contract(BASE,"fixture-run",2,1):
            capture=fixture();del capture[key]
            with self.subTest(key=key),self.assertRaises(ValueError):validate_capture(capture,BASE,"fixture-run",2,1)
        for key,value in (("promotion_eligible",True),("domain","unsigned"),("device_actual",1),("target","cpu"),("hip_runtime_version",0),
            ("source_instance_ids",[1,1]),("output_instance_id",1),("plan_workspace_bytes",True),("logical_live_payload_envelope_bytes",BUDGET+1),
            ("reference_operand_storage_bytes",0),("kernel_events",[]),("observed_overlap",True),("production_timing_hooks_enabled",True)):
            capture=fixture();capture[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_capture(capture,BASE,"fixture-run",2,1)
        capture=fixture();capture["split_resources"]["running_bytes"]+=1
        with self.assertRaises(ValueError):validate_capture(capture,BASE,"fixture-run",2,1)

    def test_phase_checksum_binding_scope_resource_and_audit_corruptions(self):
        for generation in (0,1):
            for key in fixture()["phases"][generation]:
                capture=fixture();del capture["phases"][generation][key]
                with self.subTest(generation=generation,key=key),self.assertRaises(ValueError):validate_capture(capture,BASE,"fixture-run",2,1)
        mutations=[("binding_action","refresh"),("generation",1),("source_versions",[1,2]),("stale_owner_status",None),
            ("stale_owner_preserved_output",False),("a_checksum","0"*16),("b_checksum","f"*16),("output_checksum","a"*16),
            ("oracle_ns",0),("phase_setup_ns",1),("changed_input_pack_ns",0),("owner_prepare_or_recreate_ns",0),
            ("setup_plus_first_batch_ns",1),("first_batch_exact_reference_passed",False),("samples",[]),("median_batch_ns",101.),("gemm_calls_per_second",1.)]
        for key,value in mutations:
            capture=fixture();capture["phases"][1][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_capture(capture,BASE,"fixture-run",2,1)
        for key,value in (("calls",15),("gemm_calls",31),("elapsed_ns",1),("stop_reason","call_cap"),("independent_audit_inside_interval",True),("final_exact_reference_passed",False)):
            capture=fixture();capture["phases"][1]["sustained"][key]=value
            with self.assertRaises(ValueError):validate_capture(capture,BASE,"fixture-run",2,1)
        for boundary in ("matrix_allocations_at_setup","phase"):
            capture=fixture();allocation=capture["matrix_allocations_at_setup"] if boundary!="phase" else capture["phases"][1]["matrix_allocations_after_audit"]
            allocation[0]["device_total_capacity_bytes"]+=1
            with self.assertRaises(ValueError):validate_capture(capture,BASE,"fixture-run",2,1)

    def test_case_sets_are_matched_and_boundaries_explicit(self):
        root=Path(__file__).resolve().parents[1]/"benchmarks/split-k"
        for name in ("smoke","merge-tails","large-reuse"):
            cases=load_cases(root/(name+".json"));self.assertEqual({case["mode"] for case in cases},set(MODES));self.assertEqual({case["domain"] for case in cases},{"signed","unsigned"})
        tails=load_cases(root/"merge-tails.json");self.assertTrue({127,128,129,4096}<={case["fragments"] for case in tails})
        self.assertTrue({65535,65536,65537}<={case["k"] for case in tails})


class Collection(unittest.TestCase):
    def test_independent_receipts_and_observed_matched_summary(self):
        cases=[dict(BASE,mode=mode,fragments=1 if mode=="unsplit" else 3,window=0 if mode=="unsplit" else 3 if mode=="full" else 2) for mode in MODES]
        with tempfile.TemporaryDirectory() as directory:
            args=arguments(directory,cases);args.runs=2
            with patch("collect_split_k.subprocess.run",side_effect=successful):summary=collect(args)
            self.assertEqual(summary["independent_processes"],6);self.assertFalse(summary["promotion_eligible"])
            self.assertTrue(summary["matched_cohorts"][0]["complete_three_mode_cohort"])
            receipts=[json.loads(path.read_text()) for path in args.output.glob("*.json") if not path.name.endswith(".summary.json")]
            self.assertEqual(len({receipt["run_id"] for receipt in receipts}),6)
            for receipt in receipts:self.assertTrue(receipt["validated"]);self.assertIn("collect_finite_verify.py",receipt["tool_dependencies_sha256"])

    def test_end_to_end_metadata_and_full_output_mismatches_remain_failed_receipts(self):
        for failure in ("run","generation","output","source","device","json","timeout","exit","exe","cases"):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as directory:
                args=arguments(directory)
                def run(command,**kwargs):
                    if failure=="timeout":raise subprocess.TimeoutExpired(command,10,output=b"partial",stderr=b"timeout")
                    result=successful(command);capture=json.loads(result.stdout)
                    if failure=="run":capture["run_id"]="wrong"
                    elif failure=="generation":capture["phases"][1]["binding_action"]="refresh"
                    elif failure=="output":capture["phases"][1]["output_checksum"]="0"*16
                    elif failure=="source":capture["build_manifest"]["source_digest"]="invalid"
                    elif failure=="device":capture["device_actual"]=1
                    elif failure=="exit":result.returncode=77
                    elif failure=="exe":args.exe.write_bytes(b"changed")
                    elif failure=="cases":args.cases.write_text("[]")
                    result.stdout='{"schema":1,"schema":2}' if failure=="json" else json.dumps(capture)
                    return result
                with patch("collect_split_k.subprocess.run",side_effect=run),self.assertRaises(ValueError):collect(args)
                receipts=list(args.output.glob("*.json"));self.assertEqual(len(receipts),1)
                receipt=json.loads(receipts[0].read_text());self.assertFalse(receipt["validated"]);self.assertIn("failure",receipt)

    def test_changed_build_between_independent_children_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args=arguments(directory);args.runs=2;count=0
            def run(command,**kwargs):
                nonlocal count
                result=successful(command);count+=1
                if count==2:
                    capture=json.loads(result.stdout);capture["build_manifest"]["source_digest"]="b"*64;result.stdout=json.dumps(capture)
                return result
            with patch("collect_split_k.subprocess.run",side_effect=run),self.assertRaises(ValueError):collect(args)

    def test_diagnostic_builds_stay_ineligible(self):
        for kind in ("dirty","unknown","garner"):
            with tempfile.TemporaryDirectory() as directory:
                args=arguments(directory)
                def run(command,**kwargs):
                    result=successful(command);capture=json.loads(result.stdout)
                    if kind=="dirty":capture["build_manifest"]["source_dirty"]=True
                    elif kind=="unknown":capture["build_manifest"]["dependency_digest"]="unavailable"
                    else:capture["build_manifest"]["configuration"]+="\ngarner_table=ON"
                    result.stdout=json.dumps(capture);return result
                with patch("collect_split_k.subprocess.run",side_effect=run):summary=collect(args)
                self.assertFalse(summary["provenance_complete"]);self.assertFalse(summary["promotion_eligible"])


@unittest.skipUnless(os.environ.get("RNS8_SPLIT_K_EXE"),"compiled CLI supplied by final build")
class CompiledCli(unittest.TestCase):
    def test_help_and_precontext_rejections(self):
        exe=os.environ["RNS8_SPLIT_K_EXE"]
        def invoke(args):return subprocess.run([exe,*args],text=True,capture_output=True,timeout=10)
        result=invoke(["--help"]);self.assertEqual(result.returncode,0);self.assertIn("recreation",result.stdout)
        base=["0","window","2","3","17","signed","3","2","2","2","1","fixture"]
        for index,value in ((0,"64"),(1,"auto"),(2,"0"),(4,"65538"),(5,"i64"),(6,"0"),(6,"18"),(7,"0"),(7,"3"),(8,"0"),(9,"0"),(10,"0"),(11,"bad run"),(4,"17.0")):
            args=base.copy();args[index]=value;self.assertEqual(invoke(args).returncode,1)
        self.assertEqual(invoke(base+["extra"]).returncode,1);self.assertEqual(invoke(base[:-1]).returncode,1)

if __name__=="__main__":unittest.main()
