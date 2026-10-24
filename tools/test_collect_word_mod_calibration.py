"""Authored exact word-modular receipt/collection/CLI fixtures; no measurements."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from collect_word_mod_calibration import (METHODS,SCOPES,collect,contract,dataset,load_cases,prime,validate_capture,validate_case,checksum)
from test_capture_identity_support import synthetic_identity

BASE=dict(device=0,method="packed_dot4",scope="resident",m=2,n=3,k=17,domain="prime_field",modulus=251)


def fixture(case=None,run_id="fixture-run",repeats=2,warmups=1):
    case=case or BASE;capture=contract(case,run_id,repeats,warmups);synthetic_identity(capture)
    a,b,gold=dataset(case)
    capture.update(device_name="synthetic mocked GPU only",target="gfx1100",runtime_version=70100000,
        a_checksum=checksum(a),b_checksum=checksum(b),output_checksum=checksum(gold),oracle_and_dataset_ns=1000,
        resident_setup_ns=1000 if case["scope"]=="resident" else None,
        samples=[dict(host_call_ns=150+index,uninstrumented_host_call_ns=100+index,events=[dict(label="mock",elapsed_ns=1.5)],
            exact_reference_passed=True,uninstrumented_exact_reference_passed=True) for index in range(repeats)],
        sustained=dict(calls=16,minimum_calls=16,maximum_calls=4096,target_ns=250000000,independent_audit_inside_interval=False,
            elapsed_ns=320000000,mean_call_ns=20000000.,final_exact_reference_passed=True))
    return capture


def arguments(directory,cases=None):
    root=Path(directory);exe=root/"fake.exe";exe.write_bytes(b"synthetic binary")
    if cases is None:
        cases=root/"cases.json";cases.write_text(json.dumps(dict(schema="rns8_word_mod_cases_v1",cases=[BASE,dict(BASE,method="baseline")])))
    return argparse.Namespace(exe=exe,cases=cases,output=root/"captures",runs=1,repeats=2,warmups=1,timeout=10,order_seed=6651)


def successful(command,**kwargs):
    case=dict(device=int(command[1]),method=command[2],scope=command[3],m=int(command[4]),n=int(command[5]),k=int(command[6]),domain=command[7],modulus=int(command[8]))
    return subprocess.CompletedProcess(command,0,json.dumps(fixture(case,command[11],int(command[9]),int(command[10]))),"")


class Captures(unittest.TestCase):
    def test_all_methods_scopes_domains_widths_and_slab_tails(self):
        for bits in (8,16,23,26,27,31,35,39,42,48,50,52,64):
            top=(1 << bits) if bits<64 else (1 << 64)-1
            p=top-1
            while not prime(p):p-=1
            for modulus,domain in ((top,"ring"),(p,"prime_field")):
                for method in METHODS:
                    for scope in SCOPES:
                        case=validate_case(dict(BASE,method=method,scope=scope,domain=domain,modulus=modulus))
                        validate_capture(fixture(case),case,"fixture-run",2,1)
                        self.assertEqual(contract(case,"x",1,1)["operand_bits"],bits)
        for k in (4095,4096,4097,8193,65537):
            case=validate_case(dict(BASE,m=1,n=1,k=k,modulus=(1 << 64)-1,domain="ring"))
            validate_capture(fixture(case),case,"fixture-run",2,1)
        self.assertTrue(prime(18446744073709551557))
        for p in (0,1,4,341,561,3215031751,(1 << 64)-1):self.assertFalse(prime(p))

    def test_tracked_manifests_and_full_width_boundary_words(self):
        root=Path(__file__).resolve().parents[1]/"benchmarks"/"word-mod-calibration"
        for name,count in (("smoke",8),("widths-tail",104),("slab-boundaries",32),("large",96)):
            cases=load_cases(root/(name+".json"));self.assertEqual(len(cases),count)
            self.assertEqual({case["method"] for case in cases},set(METHODS));self.assertEqual({case["scope"] for case in cases},set(SCOPES))
        case=dict(BASE,m=1,n=1,k=4,domain="ring",modulus=(1 << 64)-1)
        a,b,gold=dataset(case);words=[int.from_bytes(a[i:i+8],"little") for i in range(0,len(a),8)]
        self.assertEqual(words,[0,1,(1 << 64)-2,((1 << 64)-1)//2,(1 << 64)-1,(1 << 64)-1])
        self.assertEqual(gold[-64:],b"\xff"*64)
        self.assertEqual(dataset(case),dataset(dict(case,method="baseline",scope="cold")))

    def test_malformed_cases_and_irrelevant_options(self):
        for key,values in {"device":(-1,64,True,0.),"method":("auto",None,True),"scope":("warm",None),"domain":("field",None),
            "m":(0,513,True,1.),"n":(0,513),"k":(0,65538,True),"modulus":(1,1 << 64,True,251.)}.items():
            for value in values:
                with self.subTest(key=key,value=value),self.assertRaises(ValueError):validate_case({**BASE,key:value})
        for case in (dict(BASE,modulus=256),dict(BASE,m=512,n=512,k=513),dict(BASE,m=512,n=1,k=65537),dict(BASE,workers=32)):
            with self.assertRaises(ValueError):validate_case(case)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"cases.json"
            for cases in ([],[BASE,BASE],[dict(BASE,modulus=True)]):
                path.write_text(json.dumps(dict(schema="rns8_word_mod_cases_v1",cases=cases)))
                with self.assertRaises(ValueError):load_cases(path)
            path.write_text('{"schema":"rns8_word_mod_cases_v1","cases":[],"cases":[]}')
            with self.assertRaises(ValueError):load_cases(path)

    def test_contract_fields_and_malformed_samples_provenance(self):
        for key in contract(BASE,"fixture-run",2,1):
            capture=fixture();capture.pop(key)
            with self.subTest(key=key),self.assertRaises(ValueError):validate_capture(capture,BASE,"fixture-run",2,1)
        changes=[lambda c:c.update(device=True),lambda c:c.update(modulus=251.),lambda c:c.update(slab_k=65536),lambda c:c.update(max_diagonal_accumulator=0),
            lambda c:c.update(explicit_scratch_peak_bytes=8),lambda c:c.update(asymmetric_floating_schedule_implemented=True),lambda c:c.update(promotion_eligible=True),
            lambda c:c.update(target="gfx942"),lambda c:c.update(runtime_version=True),lambda c:c.update(a_checksum="0"*16),lambda c:c.update(output_checksum="0"*16),
            lambda c:c.update(resident_setup_ns=None),lambda c:c.update(oracle_and_dataset_ns=0),lambda c:c.update(samples=[]),
            lambda c:c["samples"][0].update(host_call_ns=float("inf")),lambda c:c["samples"][0].update(uninstrumented_exact_reference_passed=False),
            lambda c:c["samples"][0]["events"][0].update(elapsed_ns=0),lambda c:c["sustained"].update(calls=True),lambda c:c["sustained"].update(mean_call_ns=1),
            lambda c:c["sustained"].update(elapsed_ns=160,mean_call_ns=10),lambda c:c["sustained"].update(independent_audit_inside_interval=True),
            lambda c:c["build_manifest"].update(source_digest="bad")]
        for mutation in changes:
            capture=fixture();mutation(capture)
            with self.assertRaises(ValueError):validate_capture(capture,BASE,"fixture-run",2,1)
        capture=fixture();capture["samples"][0]["events"][0]["elapsed_ns"]=None;validate_capture(capture,BASE,"fixture-run",2,1)
        for field,value in (("source_dirty",True),("dependency_digest","unavailable")):
            capture=fixture();capture["build_manifest"][field]=value;validate_capture(capture,BASE,"fixture-run",2,1)


class Collection(unittest.TestCase):
    def test_successful_immutable_independent_collection(self):
        with tempfile.TemporaryDirectory() as directory:
            args=arguments(directory)
            with patch("collect_word_mod_calibration.subprocess.run",side_effect=successful):summary=collect(args)
            self.assertEqual(summary["independent_processes"],2);self.assertFalse(summary["promotion_eligible"])
            for path in args.output.glob("*.json"):
                if path.name.endswith(".summary.json"):continue
                record=json.loads(path.read_text());self.assertTrue(record["validated"])
                for key in ("collector_sha256","cases_sha256","python_sha256","artifact_sha256"):self.assertEqual(len(record[key]),64)
                self.assertIn("build_manifest.py",record["tool_dependencies_sha256"])

    def test_case_build_target_output_mismatch_and_failure_receipts(self):
        mutations=[lambda c:c.update(method="wrong"),lambda c:c.update(scope="cold"),lambda c:c.update(domain="ring"),
            lambda c:c.update(device_name="different GPU"),lambda c:c.update(output_checksum="0"*16),lambda c:c["build_manifest"].update(source_dirty=True)]
        for mutation in mutations:
            with tempfile.TemporaryDirectory() as directory:
                args=arguments(directory);count=0
                def run(command,**kwargs):
                    nonlocal count
                    result=successful(command);count+=1
                    if count==2:
                        capture=json.loads(result.stdout);mutation(capture);result.stdout=json.dumps(capture)
                    return result
                with patch("collect_word_mod_calibration.subprocess.run",side_effect=run),self.assertRaises(ValueError):collect(args)
                failed=[json.loads(p.read_text()) for p in args.output.glob("*.json") if "failure" in json.loads(p.read_text())]
                self.assertEqual(len(failed),1);self.assertFalse(failed[0]["validated"])
        for failure in ("timeout","exit","artifact","cases","json"):
            with tempfile.TemporaryDirectory() as directory:
                args=arguments(directory)
                def run(command,**kwargs):
                    if failure=="timeout":raise subprocess.TimeoutExpired(command,10,output=b"partial",stderr=b"error")
                    if failure=="exit":return subprocess.CompletedProcess(command,77,"","unsupported")
                    result=successful(command)
                    if failure=="artifact":args.exe.write_bytes(b"changed")
                    elif failure=="cases":args.cases.write_text("[]")
                    else:result.stdout='{"schema":1,"schema":2}'
                    return result
                with patch("collect_word_mod_calibration.subprocess.run",side_effect=run),self.assertRaises(ValueError):collect(args)
                records=[json.loads(p.read_text()) for p in args.output.glob("*.json")]
                self.assertEqual(len(records),1);self.assertFalse(records[0]["validated"]);self.assertIn("failure",records[0])

    def test_diagnostic_receipts_never_promote(self):
        for state in ("dirty","unknown","garner"):
            with tempfile.TemporaryDirectory() as directory:
                args=arguments(directory)
                def run(command,**kwargs):
                    result=successful(command);capture=json.loads(result.stdout)
                    if state=="dirty":capture["build_manifest"]["source_dirty"]=True
                    elif state=="unknown":capture["build_manifest"]["dependency_digest"]="unavailable"
                    else:capture["build_manifest"]["configuration"]+="\ngarner_table=ON"
                    result.stdout=json.dumps(capture);return result
                with patch("collect_word_mod_calibration.subprocess.run",side_effect=run):summary=collect(args)
                self.assertFalse(summary["provenance_complete"]);self.assertFalse(summary["promotion_eligible"])


@unittest.skipUnless(os.environ.get("RNS8_WORD_MOD_CALIBRATION_EXE"),"compiled CLI supplied by final build")
class CompiledCli(unittest.TestCase):
    def invoke(self,args):return subprocess.run([os.environ["RNS8_WORD_MOD_CALIBRATION_EXE"],*args],text=True,capture_output=True,timeout=10)
    def test_help_and_precontext_rejections(self):
        result=self.invoke(["--help"]);self.assertEqual(result.returncode,0);self.assertIn("packed_dot4",result.stdout)
        base=["0","baseline","resident","2","3","17","prime_field","251","2","1","negative"]
        for index,value in ((0,"-1"),(0,"64"),(1,"auto"),(2,"warm"),(3,"0"),(5,"65538"),(6,"field"),(7,"256"),(7,"18446744073709551616"),(8,"0"),(9,"0"),(10,"bad run")):
            args=base.copy();args[index]=value;result=self.invoke(args);self.assertEqual(result.returncode,1);self.assertTrue(result.stderr)
        self.assertEqual(self.invoke(base+["extra"]).returncode,1)

if __name__=="__main__":unittest.main()
