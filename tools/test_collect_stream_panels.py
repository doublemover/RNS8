"""Authored panel receipts/CLI fixtures. Mock timings are not measurements."""
from benchmark_cases import case_root
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from collect_stream_panels import (dma_receipt,MODES,KEYS,BUDGET,MASK,MODULI,collect,contract,load_cases,memory,
    reference,resources,validate_case,validate_capture,word_checksum,words)
from test_capture_identity_support import synthetic_identity

BASE=dict(device=0,mode="stream",m=2,n=3,k=17,domain="signed",bits=3,prefix=9,
    panel_m=2,panel_n=2,panel_k=5,device_budget_bytes=8192,host_budget_bytes=8192,retained_b_budget_bytes=0,reuse=2)


def incore(case):return dict(case,mode="incore",panel_m=0,panel_n=0,panel_k=0,device_budget_bytes=0,host_budget_bytes=0,retained_b_budget_bytes=0)


def observed(case,epoch):
    if case["mode"]=="incore":return None
    r=resources(case)
    loads=r["cold_b_panel_loads"] if epoch==1 else r["warm_b_panel_loads"]
    return dict(**dma_receipt(case,epoch),b_panel_loads=loads,b_panel_cache_hits=r["output_tiles"]*r["k_panels_per_tile"]-loads,epoch=epoch,completed_tiles=r["output_tiles"],completed_panels=r["output_tiles"]*r["k_panels_per_tile"],output_started=True,output_valid=True,quarantined=False)


def fixture(case=None,run_id="fixture-run",repeats=2,warmups=1):
    case=case or BASE;capture=contract(case,run_id,repeats,warmups)
    identity={};synthetic_identity(identity);capture["build_manifest"]=identity["build_manifest"]
    panel=case["mode"]!="incore";reuse=case["reuse"]
    allocations=[]
    if not panel:
        for cells in (case["m"]*case["k"],case["k"]*case["n"],case["m"]*case["n"]):
            allocations.append(dict(logical_primary_bytes=cells*case["prefix"],host_payload_capacity_bytes=0,device_payload_capacity_bytes=cells*case["prefix"],device_upload_staging_bytes=0,device_export_staging_bytes=0,device_status_and_schedule_bytes=0,device_total_capacity_bytes=cells*case["prefix"]))
    ab=case["m"]*case["k"]+case["k"]*case["n"]
    capture.update(device_name="synthetic mocked GPU",target="gfx1100",hip_runtime_version=70100000,hip_driver_version=70100000,
        selected_control=("overlapped_panel_public_rns_axpby" if case["mode"]=="stream-overlap" else "serial_panel_public_rns_axpby") if panel else "mock dense control",clock_period_numerator=1,clock_period_denominator=1000000000,
        query_ns=10 if panel else 0,initial_setup_ns=100,owner_prepare_ns=20 if panel else 0,owner_release_ns=10 if panel else 0,
        reference_fixed_operand_object_bytes=48,reference_operand_storage_bytes=48*ab,plan_workspace_bytes=0,logical_live_payload_envelope_bytes=memory(case)[1],matrix_allocations_after_final_audit=deepcopy(allocations))
    phases=[];epoch=0
    for gen in (1,2):
        a,b,out=reference(*(case[key] for key in ("m","n","k","domain","bits")),gen);epoch+=reuse;first=observed(case,epoch);epoch+=warmups*reuse;samples=[]
        for index in range(repeats):
            epoch+=reuse;samples.append(dict(wall_ns=100+index,exact_reference_passed=True,coverage=observed(case,epoch)))
        epoch+=16*reuse
        phases.append(dict(generation=gen,source_binding=[101,gen,202,1],binding_action="initial_inputs" if gen==1 else "changed_A0_same_plan",
            a_checksum=a,b_checksum=b,output_checksum=out,oracle_ns=100,first_batch_ns=100,first_coverage=first,first_exact_reference_passed=True,
            samples=samples,median_batch_ns=100+(repeats-1)/2,gemm_calls_per_second=1e9*reuse/(100+(repeats-1)/2),matrix_allocations_after_audit=deepcopy(allocations),
            sustained=dict(batches=16,gemm_calls=16*reuse,elapsed_ns=320000000,mean_batch_ns=20000000.,minimum_batches=16,maximum_batches=4096,target_ns=250000000,stop_reason="duration",independent_audit_inside_interval=False,final_exact_reference_passed=True,coverage=observed(case,epoch))))
    capture.update(phases=phases,cold_setup_plus_first_batch_ns=capture["query_ns"]+200,
        out_of_bound_probe=dict(status="RNS8_RANGE_ERROR",output_preserved=True,epoch_preserved=True) if panel else None,
        recreation=dict(destroy_create_ns=50,first_batch_ns=100,setup_plus_first_batch_ns=150,coverage=observed(case,reuse),exact_reference_passed=True) if panel else None,
        production_native_calls=epoch+(reuse if panel else 0))
    return capture


def arguments(root,cases=None):
    root=Path(root);exe=root/"fake.exe";exe.write_bytes(b"mock artifact");manifest=root/"cases.json"
    manifest.write_text(json.dumps(dict(schema="rns8_stream_panel_cases_v3",cases=cases or [BASE])))
    return argparse.Namespace(exe=exe,cases=manifest,output=root/"captures",runs=1,repeats=2,warmups=1,timeout=10,order_seed=6811)


def successful(command,**kwargs):
    case=dict(device=int(command[1]),mode=command[2],m=int(command[3]),n=int(command[4]),k=int(command[5]),domain=command[6],
        **{key:int(command[index]) for index,key in enumerate(("bits","prefix","panel_m","panel_n","panel_k","device_budget_bytes","host_budget_bytes","reuse"),7)})
    case["retained_b_budget_bytes"]=int(command[18])
    return subprocess.CompletedProcess(command,0,json.dumps(fixture(case,command[17],int(command[15]),int(command[16]))),"")


class Captures(unittest.TestCase):
    def test_dma_replay_matches_independent_tail_and_cache_walk(self):
        for mode in ('stream','stream-overlap'):
            for retained in (0,4*256,6*256):
                case=dict(BASE,mode=mode,m=7,n=5,k=9,panel_m=3,panel_n=3,panel_k=4,
                    device_budget_bytes=65536,host_budget_bytes=65536,retained_b_budget_bytes=retained)
                r=resources(case);tags=[None]*r['retained_b_entries']
                for epoch in (1,2):
                    uploaded=transfers=downloaded=downloads=0
                    for row in range(0,case['m'],3):
                        for col in range(0,case['n'],3):
                            rows=min(3,case['m']-row);cols=min(3,case['n']-col)
                            for q in range(0,case['k'],4):
                                depth=min(4,case['k']-q);key=(col//3)*3+q//4
                                uploaded+=8*rows*depth;transfers+=1
                                hit=bool(tags) and tags[key%len(tags)]==key
                                if not hit:
                                    uploaded+=8*depth*cols;transfers+=1
                                    if tags:tags[key%len(tags)]=key
                            downloaded+=8*rows*cols+4;downloads+=2
                    self.assertEqual(dma_receipt(case,epoch),dict(h2d_bytes=uploaded,d2h_bytes=downloaded,
                        h2d_transfers=transfers,d2h_transfers=downloads))

    def test_overlap_receipts_do_not_infer_observed_overlap(self):
        case=dict(BASE,mode='stream-overlap');capture=fixture(case)
        validate_capture(capture,case,'fixture-run',2,1)
        self.assertIsNone(capture['observed_overlap'])
        self.assertEqual(capture['panel_resources']['event_count'],8)
        for field in ('h2d_bytes','d2h_bytes','h2d_transfers','d2h_transfers'):
            bad=deepcopy(capture);bad['phases'][0]['samples'][0]['coverage'][field]+=1
            with self.assertRaises(ValueError):validate_capture(bad,case,'fixture-run',2,1)

    def test_retained_b_capacity_conflicts_counters_and_budget_fallback(self):
        base=dict(BASE,m=17,n=19,k=33,panel_m=3,panel_n=4,panel_k=5)
        baseline=resources(base)
        for slots,expected_entries,cold,warm in ((0,0,210,210),(1,0,210,210),(18,18,205,204),(35,35,35,0)):
            case=dict(base,device_budget_bytes=11008,host_budget_bytes=1496,retained_b_budget_bytes=slots*256)
            r=resources(case)
            self.assertEqual((r["retained_b_entries"],r["cold_b_panel_loads"],r["warm_b_panel_loads"]),(expected_entries,cold,warm))
            validate_case(case);capture=fixture(case);validate_capture(capture,case,"fixture-run",2,1)
            for key in ("b_panel_loads","b_panel_cache_hits"):
                corrupted=deepcopy(capture);corrupted["phases"][0]["first_coverage"][key]+=1
                with self.assertRaises(ValueError):validate_capture(corrupted,case,"fixture-run",2,1)
        limited=dict(base,device_budget_bytes=baseline["device_bytes"],host_budget_bytes=baseline["host_bytes"],retained_b_budget_bytes=35*256)
        validate_case(limited);self.assertEqual(resources(limited)["retained_b_entries"],0)
        for value in (-1,True,BUDGET+1):
            with self.assertRaises(ValueError):validate_case(dict(base,retained_b_budget_bytes=value))

    def test_all_modes_domains_full_output_changed_inputs_even_median(self):
        for domain in ("signed","unsigned"):
            for case in (dict(BASE,domain=domain),incore(dict(BASE,domain=domain))):
                validate_case(case);capture=fixture(case);validate_capture(capture,case,"fixture-run",2,1)
                self.assertNotEqual(capture["phases"][0]["a_checksum"],capture["phases"][1]["a_checksum"])
                self.assertEqual(capture["phases"][0]["b_checksum"],capture["phases"][1]["b_checksum"])
                self.assertNotEqual(capture["phases"][0]["output_checksum"],capture["phases"][1]["output_checksum"])
                self.assertEqual(capture["phases"][0]["median_batch_ns"],100.5)

    def test_panel_alignment_clipping_tail_spans_and_hard_budgets(self):
        r=resources(BASE);self.assertEqual(r["device_bytes"],2048);self.assertEqual(r["host_bytes"],192)
        self.assertEqual(r["output_tiles"],2);self.assertEqual(r["k_panels_per_tile"],4)
        self.assertEqual(r["source_a_span_bytes"],8*(20+17));self.assertEqual(r["destination_span_bytes"],8*(6+3))
        for shape in ((1,1,1),(17,19,33),(2,3,65537)):
            c=dict(BASE,m=shape[0],n=shape[1],k=shape[2],panel_m=65537,panel_n=65537,panel_k=65537,device_budget_bytes=BUDGET,host_budget_bytes=BUDGET)
            full=resources(c);self.assertEqual(full["output_tiles"],1);self.assertEqual(full["k_panels_per_tile"],1)
            exact=dict(c,device_budget_bytes=full["device_bytes"],host_budget_bytes=full["host_bytes"]);validate_case(exact)
            for key in ("device_budget_bytes","host_budget_bytes"):
                with self.assertRaises(ValueError):validate_case(dict(exact,**{key:exact[key]-1}))
        self.assertEqual(resources(dict(BASE,m=1,n=1,k=1))["minimum_device_bytes"],2048)

    def test_reference_replay_signed_unsigned_bits_and_generation_change(self):
        for domain in ("signed","unsigned"):
            for bits in (1,3,16,31):
                a,b=words(1,1,1,domain,bits);maximum=(1<<bits)-1
                self.assertEqual(a,[maximum]);self.assertEqual(b,[maximum])
                self.assertEqual(reference(1,1,1,domain,bits,1),(word_checksum(a),word_checksum(b),word_checksum([maximum*maximum])))
                self.assertEqual(reference(1,1,1,domain,bits,2),(word_checksum([0]),word_checksum(b),word_checksum([0])))
        # The collector's independent ladder is the public default basis, including composite channels.
        header=(Path(__file__).resolve().parents[1]/"src/core/internal.hpp").read_text(encoding="utf-8")
        ladder=header.split("kDefaultModuli[RNS8_DEFAULT_MODULUS_COUNT] = {",1)[1].split("}",1)[0]
        self.assertEqual(tuple(int(value.strip()) for value in ladder.split(",")[:20]),MODULI)

    def test_irrelevant_missing_malformed_and_overflow_case_controls(self):
        for key in KEYS:
            bad=dict(BASE);del bad[key]
            with self.assertRaises(ValueError):validate_case(bad)
        for key,value in (("device",True),("device",64),("mode","auto"),("domain","i64"),("m",0),("n",513),("k",65538),("bits",True),("bits",0),("bits",32),("prefix",0),("prefix",21),("panel_m",0),("panel_n",65538),("panel_k",1.0),("device_budget_bytes",True),("host_budget_bytes",0),("reuse",False),("reuse",257)):
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):validate_case(dict(BASE,**{key:value}))
        for case in (dict(BASE,mode="incore"),dict(BASE,bits=31,k=65537),dict(BASE,bits=16,prefix=1),dict(BASE,m=512,n=512,k=65537),dict(BASE,m=512,n=1,k=65537),dict(incore(BASE),panel_m=1)):
            with self.assertRaises(ValueError):validate_case(case)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"cases.json"
            for text in ('[]','{"schema":"rns8_stream_panel_cases_v3","cases":[]}',json.dumps(dict(schema="rns8_stream_panel_cases_v3",cases=[BASE,BASE])),
                '{"schema":1,"schema":2,"cases":[]}','{"schema":NaN}'):
                path.write_text(text)
                with self.assertRaises(ValueError):load_cases(path)

    def test_every_required_contract_field_and_identity_fail_closed(self):
        for key in fixture():
            c=fixture();del c[key]
            with self.subTest(key=key),self.assertRaises(ValueError):validate_capture(c,BASE,"fixture-run",2,1)
        for key,value in (("promotion_eligible",True),("device_actual",1),("target","gfx"),("hip_runtime_version",0),("selected_control","ordinary"),("plan_workspace_bytes",1),
            ("reference_operand_storage_bytes",0),("logical_live_payload_envelope_bytes",BUDGET+1),("kernel_events",[]),("observed_overlap",True),("query_ns",0),("owner_prepare_ns",101),("owner_release_ns",0),("production_native_calls",1),("cold_setup_plus_first_batch_ns",1)):
            c=fixture();c[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_capture(c,BASE,"fixture-run",2,1)
        c=fixture();c["panel_resources"]["device_bytes"]+=256
        with self.assertRaises(ValueError):validate_capture(c,BASE,"fixture-run",2,1)

    def test_phase_boundary_checksum_epoch_and_sustained_corruptions(self):
        for generation in (0,1):
            for key in fixture()["phases"][generation]:
                c=fixture();del c["phases"][generation][key]
                with self.subTest(generation=generation,key=key),self.assertRaises(ValueError):validate_capture(c,BASE,"fixture-run",2,1)
        for key,value in (("generation",1),("source_binding",[101,2,202,2]),("binding_action","repack"),("a_checksum","0"*16),("b_checksum","a"*16),("output_checksum","f"*16),("oracle_ns",0),("first_batch_ns",0),("first_exact_reference_passed",False),("samples",[]),("median_batch_ns",101),("gemm_calls_per_second",1)):
            c=fixture();c["phases"][1][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_capture(c,BASE,"fixture-run",2,1)
        for where in ("first","sample","sustained","recreation"):
            for key,value in (("epoch",1),("completed_tiles",1),("completed_panels",1),("output_started",False),("output_valid",False),("quarantined",True)):
                c=fixture();target=c["recreation"]["coverage"] if where=="recreation" else c["phases"][1]["first_coverage"] if where=="first" else c["phases"][1]["samples"][0]["coverage"] if where=="sample" else c["phases"][1]["sustained"]["coverage"]
                target[key]=value
                with self.subTest(where=where,key=key),self.assertRaises(ValueError):validate_capture(c,BASE,"fixture-run",2,1)
        for key,value in (("batches",15),("gemm_calls",31),("elapsed_ns",1),("stop_reason","call_cap"),("independent_audit_inside_interval",True),("final_exact_reference_passed",False),("mean_batch_ns",1)):
            c=fixture();c["phases"][0]["sustained"][key]=value
            with self.assertRaises(ValueError):validate_capture(c,BASE,"fixture-run",2,1)
        for field in ("output_preserved","epoch_preserved"):
            c=fixture();c["out_of_bound_probe"][field]=False
            with self.assertRaises(ValueError):validate_capture(c,BASE,"fixture-run",2,1)
        c=fixture();c["recreation"]["setup_plus_first_batch_ns"]+=1
        with self.assertRaises(ValueError):validate_capture(c,BASE,"fixture-run",2,1)
        c=fixture(incore(BASE));c["matrix_allocations_after_final_audit"][0]["device_total_capacity_bytes"]+=1
        with self.assertRaises(ValueError):validate_capture(c,incore(BASE),"fixture-run",2,1)

    def test_manifests_cover_bounded_large_shapes_clipped_and_k_tails(self):
        root=case_root() / 'stream-panels'
        for name in ("smoke","tails","large-bounded","retained-b"):
            cases=load_cases(root/(name+".json"));self.assertEqual({c["mode"] for c in cases},set(MODES));self.assertEqual({c["domain"] for c in cases},{"signed","unsigned"})
        cases=load_cases(root/"tails.json");self.assertTrue({65535,65536,65537}<={c["k"] for c in cases})
        self.assertTrue(any(c["panel_m"]>c["m"] for c in cases));self.assertTrue(any(c["n"]%max(c["panel_n"],1) for c in cases if c["mode"]=="stream"))
        self.assertTrue(any(c["m"]==512 for c in load_cases(root/"large-bounded.json")))


class Collection(unittest.TestCase):
    def test_independent_process_order_matched_modes_and_tool_receipts(self):
        with tempfile.TemporaryDirectory() as directory:
            args=arguments(directory,[BASE,incore(BASE)]);args.runs=2
            with patch("collect_stream_panels.subprocess.run",side_effect=successful):summary=collect(args)
            self.assertEqual(summary["independent_processes"],4);self.assertTrue(summary["matched_cohorts"][0]["complete_two_mode_cohort"]);self.assertFalse(summary["promotion_eligible"])
            receipts=[json.loads(p.read_text()) for p in args.output.glob("*.json") if not p.name.endswith(".summary.json")]
            self.assertEqual(len({r["run_id"] for r in receipts}),4)
            for r in receipts:self.assertTrue(r["validated"]);self.assertIn("collect_finite_verify.py",r["tool_dependencies_sha256"])

    def test_end_to_end_mismatches_mutation_timeout_and_raw_failure_retention(self):
        for failure in ("run","bound","coverage","generation","output","source","device","json","timeout","exit","exe","cases"):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as directory:
                args=arguments(directory)
                def run(command,**kwargs):
                    if failure=="timeout":raise subprocess.TimeoutExpired(command,10,output=b"partial",stderr=b"timeout")
                    result=successful(command);c=json.loads(result.stdout)
                    if failure=="run":c["run_id"]="wrong"
                    elif failure=="bound":c["full_expression_bound"]+=1
                    elif failure=="coverage":c["phases"][1]["sustained"]["coverage"]["completed_panels"]-=1
                    elif failure=="generation":c["phases"][1]["source_binding"][1]=1
                    elif failure=="output":c["phases"][1]["output_checksum"]="0"*16
                    elif failure=="source":c["build_manifest"]["source_digest"]="invalid"
                    elif failure=="device":c["device_actual"]=1
                    elif failure=="exit":result.returncode=77
                    elif failure=="exe":args.exe.write_bytes(b"changed")
                    elif failure=="cases":args.cases.write_text("[]")
                    result.stdout='{"schema":1,"schema":2}' if failure=="json" else json.dumps(c);return result
                with patch("collect_stream_panels.subprocess.run",side_effect=run),self.assertRaises(ValueError):collect(args)
                receipts=list(args.output.glob("*.json"));self.assertEqual(len(receipts),1);receipt=json.loads(receipts[0].read_text());self.assertFalse(receipt["validated"]);self.assertIn("failure",receipt)

    def test_build_mismatch_and_diagnostic_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            args=arguments(directory);args.runs=2;count=0
            def run(command,**kwargs):
                nonlocal count
                result=successful(command);count+=1
                if count==2:
                    c=json.loads(result.stdout);c["build_manifest"]["source_digest"]="b"*64;result.stdout=json.dumps(c)
                return result
            with patch("collect_stream_panels.subprocess.run",side_effect=run),self.assertRaises(ValueError):collect(args)
        for kind in ("dirty","unknown","garner"):
            with tempfile.TemporaryDirectory() as directory:
                args=arguments(directory)
                def run(command,**kwargs):
                    result=successful(command);c=json.loads(result.stdout)
                    if kind=="dirty":c["build_manifest"]["source_dirty"]=True
                    elif kind=="unknown":c["build_manifest"]["dependency_digest"]="unavailable"
                    else:c["build_manifest"]["configuration"]+="\ngarner_table=ON"
                    result.stdout=json.dumps(c);return result
                with patch("collect_stream_panels.subprocess.run",side_effect=run):summary=collect(args)
                self.assertFalse(summary["provenance_complete"]);self.assertFalse(summary["promotion_eligible"])


@unittest.skipUnless(os.environ.get("RNS8_STREAM_PANEL_EXE"),"compiled CLI supplied by final build")
class CompiledCli(unittest.TestCase):
    def test_help_and_pre_context_admission(self):
        exe=os.environ["RNS8_STREAM_PANEL_EXE"]
        def invoke(args):return subprocess.run([exe,*args],text=True,capture_output=True,timeout=10)
        result=invoke(["--help"]);self.assertEqual(result.returncode,0);self.assertIn("coverage",result.stdout)
        base=["0","stream","2","3","17","signed","3","9","2","2","5","8192","8192","2","2","1","fixture","0"]
        for index,value in ((0,"64"),(1,"auto"),(1,"incore"),(2,"0"),(3,"513"),(4,"65538"),(5,"i64"),(6,"32"),(6,"31"),(7,"0"),(8,"0"),(9,"65538"),(10,"-1"),(11,"2047"),(12,"191"),(13,"0"),(14,"0"),(15,"0"),(16,"bad run"),(17,"-1"),(4,"17.0")):
            bad=base.copy();bad[index]=value
            with self.subTest(index=index,value=value):self.assertEqual(invoke(bad).returncode,1)
        self.assertEqual(invoke(base+["extra"]).returncode,1);self.assertEqual(invoke(base[:-1]).returncode,1)


if __name__=="__main__":unittest.main()
