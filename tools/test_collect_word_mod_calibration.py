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
from collect_word_mod_calibration import (INTEGER_METHODS,METHODS,SCOPES,collect,contract,dataset,load_cases,prime,validate_capture,validate_case,checksum,floating_schedule,ceil_root)
from test_capture_identity_support import synthetic_identity

BASE=dict(device=0,method="packed_dot4",scope="resident",m=2,n=3,k=17,domain="prime_field",modulus=251)
FLOAT=dict(BASE,method="asymmetric_binary64",left_digits=1,right_digits=2,slab_k=17)


def fixture(case=None,run_id="fixture-run",repeats=2,warmups=1):
    case=case or BASE;capture=contract(case,run_id,repeats,warmups);synthetic_identity(capture)
    a,b,gold=dataset(case)
    capture.update(device_name="synthetic mocked GPU only",target="gfx1100",runtime_version=70100000,
        a_checksum=checksum(a),b_checksum=checksum(b),output_checksum=checksum(gold),oracle_and_dataset_ns=1000,
        resident_setup_ns=1000 if case["scope"]=="resident" else None,
        samples=[dict(host_call_ns=150+index,uninstrumented_host_call_ns=100+index,events=[dict(label="word_mod_byte_wmma" if case["method"]=="byte_wmma" else "mock",elapsed_ns=1.5)],
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
    if case["method"]=="asymmetric_binary64":case.update(left_digits=int(command[11]),right_digits=int(command[12]),slab_k=int(command[13]))
    return subprocess.CompletedProcess(command,0,json.dumps(fixture(case,command[-1],int(command[9]),int(command[10]))),"")


class Captures(unittest.TestCase):
    def test_all_methods_scopes_domains_widths_and_slab_tails(self):
        for bits in (8,16,23,26,27,31,35,39,42,48,50,52,64):
            top=(1 << bits) if bits<64 else (1 << 64)-1
            p=top-1
            while not prime(p):p-=1
            for modulus,domain in ((top,"ring"),(p,"prime_field")):
                for method in INTEGER_METHODS:
                    for scope in SCOPES:
                        case=validate_case(dict(BASE,method=method,scope=scope,domain=domain,modulus=modulus))
                        validate_capture(fixture(case),case,"fixture-run",2,1)
                        self.assertEqual(contract(case,"x",1,1)["operand_bits"],bits)
        for k in (4095,4096,4097,8193,65537):
            for method in INTEGER_METHODS:
                case=validate_case(dict(BASE,method=method,m=1,n=1,k=k,modulus=(1 << 64)-1,domain="ring"))
                validate_capture(fixture(case),case,"fixture-run",2,1)
        self.assertTrue(prime(18446744073709551557))
        for p in (0,1,4,341,561,3215031751,(1 << 64)-1):self.assertFalse(prime(p))

    def test_tracked_manifests_and_full_width_boundary_words(self):
        root=Path(__file__).resolve().parents[1]/"benchmarks"/"word-mod-calibration"
        for name,count in (("smoke",12),("widths-tail",156),("slab-boundaries",48),("large",144),("tile-tails",84)):
            cases=load_cases(root/(name+".json"));self.assertEqual(len(cases),count)
            self.assertEqual({case["method"] for case in cases},set(INTEGER_METHODS));self.assertEqual({case["scope"] for case in cases},set(SCOPES))
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


class ByteWmmaCaptures(unittest.TestCase):
    def test_byte_widths_and_tile_boundaries_replay_independently(self):
        for modulus in (2,255,256,257,65535,65536,65537,(1<<24),(1<<24)+1,1<<32,1<<40,1<<48,1<<52,(1<<64)-1):
            for m,n in ((1,1),(15,17),(16,16),(17,19),(31,33)):
                case=validate_case(dict(BASE,method="byte_wmma",domain="ring",modulus=modulus,m=m,n=n,k=17))
                capture=fixture(case);validate_capture(capture,case,"fixture-run",2,1)
                width=1
                while (1<<(8*width))<modulus:width+=1
                self.assertEqual(capture["operand_bytes"],width)
                self.assertEqual(capture["digit_diagonals"],2*width-1)
                self.assertEqual(capture["max_diagonal_accumulator"],4096*width*255*255)
                self.assertLessEqual(capture["max_diagonal_accumulator"],(1<<31)-1)
                self.assertEqual(capture["byte_wmma_schedule"]["output_tile_count"],((m+15)//16)*((n+15)//16))
                self.assertEqual(dataset(case),dataset(dict(case,method="baseline")))

    def test_missing_tampered_or_irrelevant_schedule_metadata(self):
        case=dict(BASE,method="byte_wmma")
        for key in contract(case,"fixture-run",2,1):
            capture=fixture(case);del capture[key]
            with self.subTest(key=key),self.assertRaises(ValueError):validate_capture(capture,case,"fixture-run",2,1)
        schedule=fixture(case)["byte_wmma_schedule"]
        for key in schedule:
            capture=fixture(case);del capture["byte_wmma_schedule"][key]
            with self.assertRaises(ValueError):validate_capture(capture,case,"fixture-run",2,1)
        for key,value in (("bytes_per_operand",8),("diagonals",15),("maximum_accumulator",2130739200),("slab_k",65536),
            ("tile_m",8),("tile_n",8),("tile_k",8),("threads_per_block",64),("output_tile_count",0),("left_signed",True),("right_signed",True),
            ("saturating_accumulator",True),("zero_pad_spatial_and_k_tails",False),("asymmetric_floating_decomposition",True),
            ("native_operands_reread_per_diagonal",False)):
            capture=fixture(case);capture["byte_wmma_schedule"][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_capture(capture,case,"fixture-run",2,1)
        for key,value in (("operand_bytes",True),("operand_bytes",8),("digit_diagonals",15),("max_diagonal_accumulator",2130739200),
            ("arithmetic_kernel","word_mod_byte_dot4"),("target","gfx942"),("left_digits",1),("floating_schedule",{})):
            capture=fixture(case);capture[key]=value
            with self.assertRaises(ValueError):validate_capture(capture,case,"fixture-run",2,1)
        for other in (BASE,dict(BASE,method="baseline"),FLOAT):
            for key,value in (("operand_bytes",1),("byte_wmma_schedule",schedule)):
                capture=fixture(other);capture[key]=value
                with self.assertRaises(ValueError):validate_capture(capture,other,"fixture-run",2,1)
        for irrelevant in (dict(case,left_digits=1,right_digits=2,slab_k=17),dict(case,operand_bytes=1)):
            with self.assertRaises(ValueError):validate_case(irrelevant)

    def test_selected_arithmetic_event_and_censored_or_unavailable_timing(self):
        case=dict(BASE,method="byte_wmma")
        for events in ([],[dict(label="word_mod_byte_wmma",elapsed_ns=None)]):
            capture=fixture(case);capture["samples"][0]["events"]=events
            validate_capture(capture,case,"fixture-run",2,1)
        for events in ([dict(label="word_mod_byte_wmma",elapsed_ns=1.)]*2,
            [dict(label="word_mod_byte_dot4",elapsed_ns=1.)],[dict(label="word_mod_double_add",elapsed_ns=None)],
            [dict(label="word_mod_asymmetric_binary64",elapsed_ns=1.)],[dict(label="word_mod_typo",elapsed_ns=1.)]):
            capture=fixture(case);capture["samples"][0]["events"]=events
            with self.assertRaises(ValueError):validate_capture(capture,case,"fixture-run",2,1)


class FloatingCaptures(unittest.TestCase):
    def test_exact_roots_swapped_pairs_and_schedule_weights(self):
        for modulus in (2,251,256,65521,65536,2147483647,1 << 42,1 << 52):
            for degree in range(1,5):
                root=ceil_root(modulus,degree)
                self.assertLess((root-1)**degree,modulus);self.assertGreaterEqual(root**degree,modulus)
            for left,right in ((1,2),(2,1),(1,3),(3,1),(2,3),(3,2),(4,4)):
                case=dict(FLOAT,domain="ring",modulus=modulus,left_digits=left,right_digits=right)
                alpha=ceil_root(modulus,left);beta=ceil_root(modulus,right);lhs=17*alpha*beta+modulus-1
                if lhs>1 << 53:
                    with self.assertRaises(ValueError):validate_case(case)
                    continue
                for scope in SCOPES:
                    candidate=validate_case(dict(case,scope=scope));capture=fixture(candidate)
                    validate_capture(capture,candidate,"fixture-run",2,1)
                    schedule=capture["floating_schedule"]
                    self.assertEqual(schedule["admission_lhs_decimal"],str(lhs))
                    self.assertEqual(schedule["coefficient_max_decimal"],str(17*(alpha-1)*(beta-1)))
                    for i in range(4):
                        for j in range(4):self.assertEqual(schedule["weights"][i*4+j],(alpha**i*beta**j)%modulus if i<left and j<right else 0)
                    self.assertEqual(dataset(candidate),dataset(dict(candidate,method="baseline")))

    def test_ineligible_invalid_and_irrelevant_requests(self):
        # Exact admission boundary: alpha=2^26, beta=2^13, lhs adds p-1.
        edge=dict(FLOAT,domain="ring",modulus=1 << 26,slab_k=16383)
        self.assertEqual(floating_schedule(edge)["admission_lhs_decimal"],str(16383*(1 << 39)+(1 << 26)-1))
        validate_case(edge)
        rejected=[dict(edge,slab_k=16384),dict(FLOAT,modulus=(1 << 64)-1,domain="ring",left_digits=2,right_digits=3),
            dict(BASE,left_digits=1,right_digits=2,slab_k=17)]
        rejected.extend({key:value for key,value in FLOAT.items() if key!=missing} for missing in ("left_digits","right_digits","slab_k"))
        for key,values in {"left_digits":(0,5,True,1.,"1"),"right_digits":(0,5,False,2.),"slab_k":(0,65537,True,17.)}.items():
            rejected.extend(dict(FLOAT,**{key:value}) for value in values)
        for case in rejected:
            with self.subTest(case=case),self.assertRaises(ValueError):validate_case(case)

    def test_schedule_receipt_tampering_and_types(self):
        for key in contract(FLOAT,"fixture-run",2,1):
            capture=fixture(FLOAT);capture.pop(key)
            with self.subTest(missing=key),self.assertRaises(ValueError):validate_capture(capture,FLOAT,"fixture-run",2,1)
        for field,value in (("alpha",252),("beta",15),("slab_k",18),("left_digits",2),("right_digits",1),
            ("mantissa_bits",52),("admission_lhs_decimal","1"),("admission_lhs_decimal",str(1 << 53)),
            ("admission_limit_decimal",1 << 53),("coefficient_max_decimal","0"),("weights_stride",3),("matrix_engine",True)):
            capture=fixture(FLOAT);capture["floating_schedule"][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):validate_capture(capture,FLOAT,"fixture-run",2,1)
        for mutate in (lambda c:c.update(slab_k=18),lambda c:c.update(left_digits=True),lambda c:c.update(asymmetric_floating_schedule_implemented=False),
            lambda c:c.update(arithmetic_kernel="word_mod_byte_dot4"),lambda c:c["floating_schedule"]["weights"].__setitem__(15,1),
            lambda c:c["floating_schedule"]["weights"].__setitem__(0,True)):
            capture=fixture(FLOAT);mutate(capture)
            with self.assertRaises(ValueError):validate_capture(capture,FLOAT,"fixture-run",2,1)
        capture=fixture();capture["floating_schedule"]=floating_schedule(FLOAT)
        with self.assertRaises(ValueError):validate_capture(capture,BASE,"fixture-run",2,1)

    def test_matched_floating_manifests(self):
        root=Path(__file__).resolve().parents[1]/"benchmarks"/"word-mod-calibration"
        for name,count in (("floating-smoke",16),("floating-pairs",392),("floating-slabs",144),("floating-large",192)):
            cases=load_cases(root/(name+".json"));self.assertEqual(len(cases),count)
            self.assertEqual({case["method"] for case in cases},{"baseline","packed_dot4","asymmetric_binary64"})
            identities={(case["m"],case["n"],case["k"],case["modulus"],case["domain"],case["scope"],case["method"]) for case in cases}
            for case in cases:
                self.assertIn((*tuple(case[key] for key in ("m","n","k","modulus","domain","scope")),"baseline"),identities)
                if case["method"]=="asymmetric_binary64":
                    swapped=dict(case,left_digits=case["right_digits"],right_digits=case["left_digits"])
                    self.assertIn(swapped,cases)


class Collection(unittest.TestCase):
    def test_byte_wmma_commands_and_metadata_mismatch_receipts(self):
        for corrupt in (False,True):
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory);path=root/"wmma.json"
                path.write_text(json.dumps(dict(schema="rns8_word_mod_cases_v1",cases=[dict(BASE,method=method) for method in INTEGER_METHODS])))
                args=arguments(directory,path)
                def run(command,**kwargs):
                    result=successful(command)
                    if command[2]=="byte_wmma":
                        self.assertEqual(len(command),12)
                        if corrupt:
                            capture=json.loads(result.stdout);capture["byte_wmma_schedule"]["bytes_per_operand"]=8;result.stdout=json.dumps(capture)
                    return result
                with patch("collect_word_mod_calibration.subprocess.run",side_effect=run):
                    if corrupt:
                        with self.assertRaises(ValueError):collect(args)
                        failures=[json.loads(p.read_text()) for p in args.output.glob("*.json") if "failure" in json.loads(p.read_text())]
                        self.assertEqual(len(failures),1);self.assertFalse(failures[0]["validated"])
                    else:self.assertEqual(collect(args)["independent_processes"],3)

    def test_floating_commands_schedule_mismatch_and_retained_failure(self):
        for corrupt in (False,True):
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory);path=root/"floating.json"
                path.write_text(json.dumps(dict(schema="rns8_word_mod_cases_v1",cases=[dict(BASE,method="baseline"),FLOAT,dict(FLOAT,left_digits=2,right_digits=1)])))
                args=arguments(directory,path)
                def run(command,**kwargs):
                    result=successful(command)
                    if command[2]=="asymmetric_binary64":
                        self.assertEqual(len(command),15)
                        if corrupt:
                            capture=json.loads(result.stdout);capture["floating_schedule"]["alpha"]+=1;result.stdout=json.dumps(capture)
                    return result
                with patch("collect_word_mod_calibration.subprocess.run",side_effect=run):
                    if corrupt:
                        with self.assertRaises(ValueError):collect(args)
                        failed=[json.loads(p.read_text()) for p in args.output.glob("*.json") if "failure" in json.loads(p.read_text())]
                        self.assertEqual(len(failed),1);self.assertFalse(failed[0]["validated"])
                    else:self.assertEqual(collect(args)["independent_processes"],3)

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
        result=self.invoke(["--help"]);self.assertEqual(result.returncode,0);self.assertIn("packed_dot4",result.stdout);self.assertIn("byte_wmma",result.stdout);self.assertIn("asymmetric_binary64",result.stdout)
        base=["0","baseline","resident","2","3","17","prime_field","251","2","1","negative"]
        for index,value in ((0,"-1"),(0,"64"),(1,"auto"),(2,"warm"),(3,"0"),(5,"65538"),(6,"field"),(7,"256"),(7,"18446744073709551616"),(8,"0"),(9,"0"),(10,"bad run")):
            args=base.copy();args[index]=value;result=self.invoke(args);self.assertEqual(result.returncode,1);self.assertTrue(result.stderr)
        self.assertEqual(self.invoke(base+["extra"]).returncode,1)
        floating=base[:10]+["1","2","17",base[-1]];floating[1]="asymmetric_binary64"
        for index,value in ((10,"0"),(10,"5"),(11,"0"),(11,"5"),(12,"0"),(12,"65537"),(12,"17.0"),(13,"bad run")):
            args=floating.copy();args[index]=value;result=self.invoke(args);self.assertEqual(result.returncode,1);self.assertTrue(result.stderr)
        self.assertEqual(self.invoke(floating[:-1]).returncode,1)
        self.assertEqual(self.invoke(base[:10]+["1","2","17","irrelevant"]).returncode,1)
        wmma=base.copy();wmma[1]="byte_wmma"
        self.assertEqual(self.invoke(wmma[:10]+["1","2","17","irrelevant"]).returncode,1)
        rejected=floating.copy();rejected[6]="ring";rejected[7]="18446744073709551615"
        result=self.invoke(rejected);self.assertEqual(result.returncode,1);self.assertIn("binary64 schedule rejected",result.stderr)

if __name__=="__main__":unittest.main()
