"""Authored exact conversion receipts and malformed collector/CLI fixtures."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from collect_crt_calibration import (CANDIDATES,SCOPES,collect,contract,dataset,load_cases,validate_capture,validate_case,checksum)
from test_capture_identity_support import synthetic_identity

BASE=dict(candidate="fixed_dot4",cells=17,prefix=3,scope="resident")

def fixture(case=None,run_id="fixture-run",repeats=2,warmups=1):
    case=case or BASE;capture=contract(case,run_id,repeats,warmups);synthetic_identity(capture)
    source,expected=dataset(case)
    capture.update(device="synthetic mocked GPU only",target="gfx1100",runtime_version=70100000,
        input_checksum=checksum(source),output_checksum=checksum(expected),oracle_and_dataset_ns=1000,
        resident_setup_ns=1000 if case["scope"]=="resident" else None,
        samples=[dict(host_call_ns=150+index,uninstrumented_host_call_ns=100+index,events=[dict(label="mock",elapsed_ns=1.5)],
            exact_reference_passed=True,uninstrumented_exact_reference_passed=True) for index in range(repeats)],
        sustained=dict(calls=16,minimum_calls=16,maximum_calls=4096,target_ns=250000000,independent_audit_inside_interval=False,elapsed_ns=320000000,mean_call_ns=20000000.,final_exact_reference_passed=True,
            scope="fresh_buffers_transfer_inclusive_no_intermediate_reference" if case["scope"]=="cold" else "resident_same_buffers_no_intermediate_poison_or_reference"))
    return capture


def arguments(directory,cases=None):
    root=Path(directory);exe=root/"fake.exe";exe.write_bytes(b"synthetic executable")
    return argparse.Namespace(exe=exe,output=root/"captures",cases=cases,candidates=["direct","fixed_dot4"],cells=[17],prefixes=[3],scopes=["resident"],
        runs=1,repeats=2,warmups=1,timeout=10,order_seed=3903)


def successful(command,**kwargs):
    case=dict(candidate=command[1],cells=int(command[2]),prefix=int(command[3]),scope=command[7])
    return subprocess.CompletedProcess(command,0,json.dumps(fixture(case,command[6],int(command[4]),int(command[5]))),"")


class Captures(unittest.TestCase):
    def test_tracked_manifests_cover_all_controls_and_prefixes(self):
        root=Path(__file__).resolve().parents[1]/"benchmarks"/"conversion-calibration"
        for name,count in (("smoke",22),("every-prefix",440),("large",88)):
            cases=load_cases(root/(name+".json"))
            self.assertEqual(len(cases),count)
            self.assertEqual({case["candidate"] for case in cases},set(CANDIDATES))
            self.assertEqual({case["scope"] for case in cases},set(SCOPES))
        self.assertEqual({case["prefix"] for case in load_cases(root/"every-prefix.json")},set(range(1,21)))

    def test_every_candidate_scope_prefix_and_layout_tail(self):
        for candidate in CANDIDATES:
            for scope in SCOPES:
                for prefix in range(1,21):
                    for cells in (1,17,256,257):
                        case=validate_case(dict(candidate=candidate,scope=scope,cells=cells,prefix=prefix))
                        validate_capture(fixture(case),case,"fixture-run",2,1)
        validate_case(dict(BASE,cells=1048576,prefix=20))

    def test_dataset_semantics_and_matched_controls(self):
        for prefix in range(1,21):
            direct=dataset(dict(BASE,candidate="direct",prefix=prefix))
            for candidate in ("garner","balanced","fixed_dot4","tile_balanced","tile_convert_balanced","tile_convert_garner"):
                source,expected=dataset(dict(BASE,candidate=candidate,prefix=prefix))
                self.assertEqual(expected,direct[1])
            for signed in ("signed","unsigned"):
                scalar=dataset(dict(BASE,candidate="import_scalar_"+signed,prefix=prefix))
                self.assertEqual(scalar,dataset(dict(BASE,candidate="import_dot4_"+signed,prefix=prefix)))
            if prefix>1:
                self.assertNotEqual(dataset(dict(BASE,candidate="import_scalar_signed",prefix=prefix))[1],
                                    dataset(dict(BASE,candidate="import_scalar_unsigned",prefix=prefix))[1])
        self.assertNotEqual(dataset(dict(BASE,candidate="direct"))[0],dataset(dict(BASE,candidate="tile_balanced"))[0])
        source,expected=dataset(dict(BASE,candidate="import_scalar_unsigned",cells=6))
        self.assertEqual(int.from_bytes(source[16:24],"little"),(1 << 64)-1)
        self.assertEqual(expected[-64:],bytes([99])*64)

    def test_malformed_and_irrelevant_controls(self):
        for key,values in {"candidate":("auto",None,True),"scope":("warm",None,True),"cells":(0,1048577,True,1.,"1"),"prefix":(0,21,True,1.)}.items():
            for value in values:
                with self.subTest(key=key,value=value),self.assertRaises(ValueError): validate_case({**BASE,key:value})
        for case in ({**BASE,"workers":1},{key:value for key,value in BASE.items() if key!="scope"}):
            with self.assertRaises(ValueError): validate_case(case)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"cases.json"
            for cases in ([],[BASE,BASE],[dict(BASE,prefix=True)]):
                path.write_text(json.dumps(dict(schema="rns8_conversion_cases_v1",cases=cases)))
                with self.assertRaises(ValueError):load_cases(path)
            path.write_text('{"schema":"rns8_conversion_cases_v1","cases":[],"cases":[]}')
            with self.assertRaises(ValueError):load_cases(path)

    def test_all_contract_fields_and_integer_types_bound(self):
        for key in contract(BASE,"fixture-run",2,1):
            capture=fixture();capture.pop(key)
            with self.subTest(key=key),self.assertRaises(ValueError):validate_capture(capture,BASE,"fixture-run",2,1)
        for changes in (dict(cells=True),dict(prefix=3.),dict(promotion_eligible=True),dict(probe_bytes=0),dict(output_boundary="host"),dict(memory_budget_bytes=1)):
            capture=fixture();capture.update(changes)
            with self.assertRaises(ValueError):validate_capture(capture,BASE,"fixture-run",2,1)

    def test_malformed_evidence_and_diagnostic_provenance(self):
        mutations=[lambda c:c.update(input_checksum="0"*16),lambda c:c.update(output_checksum="0"*16),lambda c:c.update(runtime_version=True),
            lambda c:c.update(target="gfx942"),lambda c:c.update(resident_setup_ns=None),lambda c:c.update(oracle_and_dataset_ns=0),
            lambda c:c.update(samples=[]),lambda c:c["samples"][0].update(uninstrumented_exact_reference_passed=False),
            lambda c:c["samples"][0].update(host_call_ns=float("nan")),lambda c:c["samples"][0].update(uninstrumented_host_call_ns=0),
            lambda c:c["samples"][0]["events"][0].update(elapsed_ns=0),lambda c:c["samples"][0]["events"][0].update(label=""),
            lambda c:c["sustained"].update(calls=15),lambda c:c["sustained"].update(calls=True),lambda c:c["sustained"].update(mean_call_ns=1),
            lambda c:c["sustained"].update(elapsed_ns=160,mean_call_ns=10),lambda c:c["sustained"].update(final_exact_reference_passed=False),
            lambda c:c["sustained"].update(scope="made_up"),lambda c:c["sustained"].update(independent_audit_inside_interval=True),
            lambda c:c["sustained"].update(maximum_calls=True),lambda c:c["build_manifest"].update(source_digest="bad")]
        for mutation in mutations:
            capture=fixture();mutation(capture)
            with self.assertRaises(ValueError):validate_capture(capture,BASE,"fixture-run",2,1)
        capture=fixture();capture["samples"][0]["events"][0]["elapsed_ns"]=None
        validate_capture(capture,BASE,"fixture-run",2,1)
        for key,value in (("source_dirty",True),("dependency_digest","unavailable")):
            capture=fixture();capture["build_manifest"][key]=value
            validate_capture(capture,BASE,"fixture-run",2,1)
        capture=fixture();capture["build_manifest"]["configuration"]=capture["build_manifest"]["configuration"]+"\ngarner_table=ON"
        validate_capture(capture,BASE,"fixture-run",2,1)
        self.assertFalse(capture["promotion_eligible"])


class Collection(unittest.TestCase):
    def test_matched_records_and_tools_are_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            args=arguments(directory)
            with patch("collect_crt_calibration.subprocess.run",side_effect=successful): summary=collect(args)
            self.assertEqual(summary["independent_processes"],2);self.assertFalse(summary["promotion_eligible"])
            records=[json.loads(path.read_text()) for path in args.output.glob("*.json") if not path.name.endswith(".summary.json")]
            for record in records:
                self.assertTrue(record["validated"]);self.assertEqual(len(record["collector_sha256"]),64)
                self.assertIn("build_manifest.py",record["tool_dependencies_sha256"]);self.assertEqual(len(record["python_sha256"]),64)
                self.assertFalse(record["promotion_eligible"])

    def test_metadata_build_device_output_and_json_mismatch_retained(self):
        for mutation in (lambda c:c.update(candidate="balanced"),lambda c:c.update(scope="cold"),lambda c:c.update(device="another GPU"),
            lambda c:c.update(output_checksum="0"*16),lambda c:c["build_manifest"].update(source_dirty=True)):
            with tempfile.TemporaryDirectory() as directory:
                args=arguments(directory);count=0
                def run(command,**kwargs):
                    nonlocal count
                    result=successful(command);count+=1
                    if count==2:
                        capture=json.loads(result.stdout);mutation(capture);result.stdout=json.dumps(capture)
                    return result
                with patch("collect_crt_calibration.subprocess.run",side_effect=run),self.assertRaises(ValueError):collect(args)
                failed=[json.loads(p.read_text()) for p in args.output.glob("*.json") if "failure" in json.loads(p.read_text())]
                self.assertEqual(len(failed),1);self.assertFalse(failed[0]["validated"]);self.assertIn("stdout",failed[0])
        with tempfile.TemporaryDirectory() as directory:
            args=arguments(directory)
            with patch("collect_crt_calibration.subprocess.run",return_value=subprocess.CompletedProcess([],0,'{"schema":1,"schema":2}',"")),self.assertRaises(ValueError):collect(args)

    def test_diagnostic_provenance_stays_ineligible(self):
        for diagnostic in ("dirty","missing_dependency","experimental_garner"):
            with tempfile.TemporaryDirectory() as directory:
                args=arguments(directory)
                def run(command,**kwargs):
                    result=successful(command);capture=json.loads(result.stdout)
                    if diagnostic=="dirty": capture["build_manifest"]["source_dirty"]=True
                    elif diagnostic=="missing_dependency": capture["build_manifest"]["dependency_digest"]="unavailable"
                    else: capture["build_manifest"]["configuration"]+="\ngarner_table=ON"
                    result.stdout=json.dumps(capture);return result
                with patch("collect_crt_calibration.subprocess.run",side_effect=run):summary=collect(args)
                self.assertFalse(summary["provenance_complete"]);self.assertFalse(summary["promotion_eligible"])

    def test_timeout_nonzero_and_artifact_or_case_changes(self):
        for failure in ("timeout","exit","artifact","cases"):
            with tempfile.TemporaryDirectory() as directory:
                args=arguments(directory)
                if failure=="cases":
                    args.cases=Path(directory)/"cases.json";args.cases.write_text(json.dumps(dict(schema="rns8_conversion_cases_v1",cases=[BASE])))
                def run(command,**kwargs):
                    if failure=="timeout": raise subprocess.TimeoutExpired(command,10,output=b"partial",stderr=b"error")
                    if failure=="exit":return subprocess.CompletedProcess(command,1,"partial","error")
                    result=successful(command)
                    if failure=="artifact":args.exe.write_bytes(b"changed")
                    else:args.cases.write_text("[]")
                    return result
                with patch("collect_crt_calibration.subprocess.run",side_effect=run),self.assertRaises(ValueError):collect(args)
                records=[json.loads(p.read_text()) for p in args.output.glob("*.json")]
                self.assertEqual(len(records),1);self.assertFalse(records[0]["validated"]);self.assertIn("failure",records[0])


@unittest.skipUnless(os.environ.get("RNS8_CRT_CALIBRATION_EXE"),"compiled CLI supplied by final build")
class CompiledCli(unittest.TestCase):
    def invoke(self,args):
        return subprocess.run([os.environ["RNS8_CRT_CALIBRATION_EXE"],*args],text=True,capture_output=True,timeout=10)
    def test_help(self):
        result=self.invoke(["--help"]);self.assertEqual(result.returncode,0)
        for candidate in CANDIDATES:self.assertIn(candidate,result.stdout)
        self.assertIn("resident|cold",result.stdout)
    def test_precontext_rejection(self):
        base=["fixed_dot4","17","3","2","1","negative","resident"]
        for index,value in ((0,"auto"),(1,"0"),(1,"1048577"),(1,"17x"),(2,"0"),(2,"21"),(3,"0"),(4,"0"),(5,"bad run"),(6,"warm")):
            args=base.copy();args[index]=value
            result=self.invoke(args);self.assertEqual(result.returncode,1);self.assertTrue(result.stderr)
        self.assertEqual(self.invoke(base+["extra"]).returncode,1)

if __name__=="__main__":unittest.main()
