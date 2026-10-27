"""Authored instruction-chain receipts and no-GPU CLI fixtures; no measurements."""
from benchmark_cases import case_root
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from collect_instruction_roof import (MODES,CHAINS,LEFT,RIGHT,collect,contract,dataset,load_cases,validate_case,validate_capture,checksum)
from test_capture_identity_support import synthetic_identity

BASE=dict(device=0,mode="raw",chains=2,iterations=17,waves=5)


def fixture(case=None,run_id="fixture-run",repeats=2,warmups=1):
    case=case or BASE;capture=contract(case,run_id,repeats,warmups);synthetic_identity(capture)
    a,b,gold=dataset(case);instructions=capture["modeled_wave_instructions"]
    capture.update(device_name="synthetic mocked GPU only",target="gfx1100",runtime_version=70100000,
        a_checksum=checksum(a),b_checksum=checksum(b),output_checksum=checksum(gold),oracle_and_dataset_ns=1000,
        resident_setup_ns=1000,resource_release_ns=1000,
        samples=[dict(host_call_ns=150+index,uninstrumented_host_call_ns=100+index,
            capability_events=[dict(label="residue_d2h_sync",elapsed_ns=10.)],
            events=[dict(label=capture["kernel_event_label"],elapsed_ns=50.)],kernel_event_ns=50.,kernel_event_state="measured",
            event_ns_per_modeled_wave_instruction=50./instructions,exact_reference_passed=True,uninstrumented_exact_reference_passed=True) for index in range(repeats)],
        sustained=dict(calls=16,minimum_calls=16,maximum_calls=4096,target_ns=250000000,independent_audit_inside_interval=False,
            elapsed_ns=320000000,mean_call_ns=20000000.,final_exact_reference_passed=True))
    return capture


def arguments(directory,cases=None):
    root=Path(directory);exe=root/"fake.exe";exe.write_bytes(b"synthetic binary")
    if cases is None:
        cases=root/"cases.json";cases.write_text(json.dumps(dict(schema="rns8_instruction_roof_cases_v1",cases=[BASE,dict(BASE,mode="reduce251")])))
    return argparse.Namespace(exe=exe,cases=cases,output=root/"captures",runs=1,repeats=2,warmups=1,timeout=10,order_seed=3553,isa_file=None)


def successful(command,**kwargs):
    case=dict(device=int(command[1]),mode=command[2],chains=int(command[3]),iterations=int(command[4]),waves=int(command[5]))
    return subprocess.CompletedProcess(command,0,json.dumps(fixture(case,command[8],int(command[6]),int(command[7]))),"")


class Captures(unittest.TestCase):
    def test_modes_chains_iteration_bounds_and_partial_wave_blocks(self):
        for mode in MODES:
            for chains in CHAINS:
                for iterations in (1,17,4095,4096):
                    for waves in (1,5):
                        case=validate_case(dict(BASE,mode=mode,chains=chains,iterations=iterations,waves=waves))
                        capture=fixture(case);validate_capture(capture,case,"fixture-run",2,1)
                        a,b,gold=dataset(case)
                        self.assertEqual(a,bytes(value&255 for value in LEFT[:chains]));self.assertEqual(b,bytes(value&255 for value in RIGHT[:chains]))
                        self.assertEqual(len(gold),waves*chains*256*4+64)
                        for wave in range(waves):
                            for chain in range(chains):
                                expected=16*LEFT[chain]*RIGHT[chain]*iterations
                                if mode=="reduce251":expected=(expected+125)%251-125
                                tile=gold[(wave*chains+chain)*1024:(wave*chains+chain+1)*1024]
                                self.assertEqual(tile,expected.to_bytes(4,"little",signed=True)*256)
                        self.assertEqual(gold[-64:],(0x12345678).to_bytes(4,"little")*16)
        support=contract(dict(BASE,chains=8,iterations=4096,waves=4096),"x",1,1)
        self.assertEqual(support["maximum_accumulator_magnitude"],1 << 30)
        self.assertEqual(support["modeled_output_store_bytes"],33554432)
        self.assertLess(support["logical_payload_envelope_bytes"],support["memory_budget_bytes"])

    def test_manifests_separate_dependency_wave_sweep_and_tail_controls(self):
        root=case_root() / 'instruction-roof'
        for name,count in (("smoke",8),("one-wave",48),("wave-chain-sweep",120),("wave-tails",160)):
            cases=load_cases(root/(name+".json"));self.assertEqual(len(cases),count)
            self.assertEqual({case["mode"] for case in cases},set(MODES));self.assertEqual({case["chains"] for case in cases},set(CHAINS))
            if name=="one-wave":self.assertEqual({case["waves"] for case in cases},{1})
            if name=="wave-chain-sweep":self.assertEqual({case["waves"] for case in cases},{32,128,512,2048,4096})

    def test_invalid_and_duplicate_cases(self):
        for key,values in {"device":(-1,64,True,0.),"mode":("auto","raw_reduce",None,True),"chains":(0,3,16,True,2.),
            "iterations":(0,4097,True,17.),"waves":(0,4097,True,5.)}.items():
            for value in values:
                with self.subTest(key=key,value=value),self.assertRaises(ValueError):validate_case({**BASE,key:value})
        with self.assertRaises(ValueError):validate_case(dict(BASE,scope="cold"))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"cases.json"
            for cases in ([],[BASE,BASE],[dict(BASE,iterations=True)]):
                path.write_text(json.dumps(dict(schema="rns8_instruction_roof_cases_v1",cases=cases)))
                with self.assertRaises(ValueError):load_cases(path)
            path.write_text('{"schema":"rns8_instruction_roof_cases_v1","cases":[],"cases":[]}')
            with self.assertRaises(ValueError):load_cases(path)

    def test_modeled_counts_provenance_and_no_occupancy_or_cycle_claims(self):
        for key in contract(BASE,"fixture-run",2,1):
            capture=fixture();capture.pop(key)
            with self.subTest(missing=key),self.assertRaises(ValueError):validate_capture(capture,BASE,"fixture-run",2,1)
        changes=[lambda c:c.update(waves=True),lambda c:c.update(backend_actual="cpu-reference"),lambda c:c.update(modeled_integer_operations=1),
            lambda c:c.update(modeled_wave_instructions=1),lambda c:c.update(modeled_output_cells=1),lambda c:c.update(maximum_accumulator_magnitude=1),
            lambda c:c.update(achieved_occupancy=1),lambda c:c.update(hardware_cycle_latency=1),lambda c:c.update(actual_issued_instructions=1),
            lambda c:c.update(isa_sha256="b"*64),lambda c:c.update(isa_verified=True),lambda c:c.update(promotion_eligible=True),
            lambda c:c.update(target="gfx942"),lambda c:c.update(runtime_version=True),lambda c:c["a_values"].__setitem__(0,128),
            lambda c:c.update(output_checksum="0"*16),lambda c:c.update(resource_release_ns=0),
            lambda c:c["build_manifest"].update(source_digest="bad"),lambda c:c["sustained"].update(calls=True),
            lambda c:c["sustained"].update(mean_call_ns=1),lambda c:c["sustained"].update(independent_audit_inside_interval=True)]
        for mutation in changes:
            capture=fixture();mutation(capture)
            with self.assertRaises(ValueError):validate_capture(capture,BASE,"fixture-run",2,1)
        capture=fixture();capture["build_manifest"]["dependency_digest"]="unavailable";validate_capture(capture,BASE,"fixture-run",2,1)

    def test_measured_censored_and_unavailable_kernel_events(self):
        for state in ("below_resolution","unavailable"):
            capture=fixture()
            for sample in capture["samples"]:
                sample.update(kernel_event_ns=None,kernel_event_state=state,event_ns_per_modeled_wave_instruction=None)
                sample["events"]=[] if state=="unavailable" else [dict(label=capture["kernel_event_label"],elapsed_ns=None)]
            validate_capture(capture,BASE,"fixture-run",2,1)
        for transfer in ([], [dict(label="residue_d2h_sync",elapsed_ns=None)]):
            capture=fixture();capture["samples"][0]["capability_events"]=transfer
            validate_capture(capture,BASE,"fixture-run",2,1)
        changes=[lambda s:s["capability_events"][0].update(label="instruction_roof_wmma_raw"),
            lambda s:s["capability_events"][0].update(elapsed_ns=-1),
            lambda s:s["capability_events"].append(dict(s["capability_events"][0])),
            lambda s:s.update(kernel_event_ns=True),lambda s:s.update(kernel_event_state="unavailable"),
            lambda s:s.update(event_ns_per_modeled_wave_instruction=1),lambda s:s["events"][0].update(label="capability_probe"),
            lambda s:s["events"][0].update(elapsed_ns=0),lambda s:s["events"].append(dict(s["events"][0])),
            lambda s:s.update(uninstrumented_exact_reference_passed=False),lambda s:s.update(host_call_ns=float("nan"))]
        for mutation in changes:
            capture=fixture();mutation(capture["samples"][0])
            with self.assertRaises(ValueError):validate_capture(capture,BASE,"fixture-run",2,1)


class Collection(unittest.TestCase):
    def test_independent_immutable_collection_and_unreviewed_isa_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            args=arguments(directory);isa=Path(directory)/"unreviewed.isa";isa.write_text("synthetic fixture only");args.isa_file=isa
            with patch("collect_instruction_roof.subprocess.run",side_effect=successful):summary=collect(args)
            self.assertEqual(summary["independent_processes"],2);self.assertFalse(summary["promotion_eligible"])
            self.assertFalse(summary["unreviewed_isa_artifact"]["correspondence_verified"])
            for path in args.output.glob("*.json"):
                if path.name.endswith(".summary.json"):continue
                record=json.loads(path.read_text());self.assertTrue(record["validated"])
                for key in ("collector_sha256","cases_sha256","python_sha256","artifact_sha256"):self.assertEqual(len(record[key]),64)
                self.assertIn("build_manifest.py",record["tool_dependencies_sha256"])

    def test_mixed_cohorts_schedule_events_outputs_and_failure_retention(self):
        mutations=[lambda c:c.update(mode="wrong"),lambda c:c.update(chains=4),lambda c:c.update(device_name="different GPU"),
            lambda c:c.update(target="gfx1101"),lambda c:c.update(output_checksum="0"*16),lambda c:c["build_manifest"].update(source_dirty=True),
            lambda c:c["samples"][0].update(event_ns_per_modeled_wave_instruction=1)]
        for mutation in mutations:
            with tempfile.TemporaryDirectory() as directory:
                args=arguments(directory);count=0
                def run(command,**kwargs):
                    nonlocal count
                    result=successful(command);count+=1
                    if count==2:
                        capture=json.loads(result.stdout);mutation(capture);result.stdout=json.dumps(capture)
                    return result
                with patch("collect_instruction_roof.subprocess.run",side_effect=run),self.assertRaises(ValueError):collect(args)
                failed=[json.loads(p.read_text()) for p in args.output.glob("*.json") if "failure" in json.loads(p.read_text())]
                self.assertEqual(len(failed),1);self.assertFalse(failed[0]["validated"])
        for failure in ("timeout","exit","artifact","cases","isa","json"):
            with tempfile.TemporaryDirectory() as directory:
                args=arguments(directory);args.isa_file=Path(directory)/"fixture.isa";args.isa_file.write_text("synthetic")
                def run(command,**kwargs):
                    if failure=="timeout":raise subprocess.TimeoutExpired(command,10,output=b"partial",stderr=b"error")
                    if failure=="exit":return subprocess.CompletedProcess(command,77,"","unsupported")
                    result=successful(command)
                    if failure=="artifact":args.exe.write_bytes(b"changed")
                    elif failure=="cases":args.cases.write_text("[]")
                    elif failure=="isa":args.isa_file.write_text("changed")
                    else:result.stdout='{"schema":1,"schema":2}'
                    return result
                with patch("collect_instruction_roof.subprocess.run",side_effect=run),self.assertRaises(ValueError):collect(args)
                records=[json.loads(p.read_text()) for p in args.output.glob("*.json")]
                self.assertEqual(len(records),1);self.assertFalse(records[0]["validated"]);self.assertIn("failure",records[0])

    def test_diagnostic_provenance_never_promotes(self):
        for state in ("dirty","unknown","garner"):
            with tempfile.TemporaryDirectory() as directory:
                args=arguments(directory)
                def run(command,**kwargs):
                    result=successful(command);capture=json.loads(result.stdout)
                    if state=="dirty":capture["build_manifest"]["source_dirty"]=True
                    elif state=="unknown":capture["build_manifest"]["dependency_digest"]="unavailable"
                    else:capture["build_manifest"]["configuration"]+="\ngarner_table=ON"
                    result.stdout=json.dumps(capture);return result
                with patch("collect_instruction_roof.subprocess.run",side_effect=run):summary=collect(args)
                self.assertFalse(summary["provenance_complete"]);self.assertFalse(summary["promotion_eligible"])


@unittest.skipUnless(os.environ.get("RNS8_INSTRUCTION_ROOF_EXE"),"compiled CLI supplied by final build")
class CompiledCli(unittest.TestCase):
    def invoke(self,args):return subprocess.run([os.environ["RNS8_INSTRUCTION_ROOF_EXE"],*args],text=True,capture_output=True,timeout=10)
    def test_help_and_precontext_rejections(self):
        result=self.invoke(["--help"]);self.assertEqual(result.returncode,0);self.assertIn("reduce251",result.stdout);self.assertIn("NOT hardware cycles",result.stdout)
        base=["0","raw","2","17","5","2","1","negative"]
        for index,value in ((0,"-1"),(0,"64"),(1,"auto"),(2,"0"),(2,"3"),(2,"16"),(3,"0"),(3,"4097"),(4,"0"),(4,"4097"),
            (5,"0"),(6,"0"),(7,"bad run"),(3,"17.0"),(3,"17junk")):
            args=base.copy();args[index]=value;result=self.invoke(args);self.assertEqual(result.returncode,1);self.assertTrue(result.stderr)
        self.assertEqual(self.invoke(base+["extra"]).returncode,1)
        self.assertEqual(self.invoke(base[:-1]).returncode,1)

if __name__=="__main__":unittest.main()
