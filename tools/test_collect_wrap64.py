"""Authored synthetic wrap capture/collector fixtures; no measured evidence."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from collect_wrap64 import (CANDIDATES, RDNA_CANDIDATES, CDNA_CANDIDATES, SCOPES, EDGES, MASK, checksum, collect, contract,
                            dataset, load_cases, summarize, validate_capture, validate_case)
from test_capture_identity_support import synthetic_identity
BASE = dict(device=0, candidate="signed_radix", scope="resident", m=2, n=3, k=17, pattern="carry", reuse_count=1)


def case_for(candidate="signed_radix", scope="resident", **changes):
    return validate_case({**BASE, "candidate": candidate, "scope": scope, **changes})


def fixture(case=None, run_id="fixture-run", repeats=2, warmups=1):
    case = case or case_for(); capture = contract(case, run_id, repeats, warmups,"gfx942" if case["candidate"]=="signed_radix_mfma" else "gfx1100"); synthetic_identity(capture)
    a, b = dataset(case); av = [int.from_bytes(a[i:i+8], "little") for i in range(0, len(a), 8)]; bv = [int.from_bytes(b[i:i+8], "little") for i in range(0, len(b), 8)]
    expected = b"".join((sum(av[r*case["k"]+t]*bv[t*case["n"]+col] for t in range(case["k"])) & MASK).to_bytes(8, "little") for r in range(case["m"]) for col in range(case["n"]))
    ab, mn = len(av)+len(bv), case["m"]*case["n"]
    oracle = 8*ab+8*mn+48*ab+2048
    execution = 24*ab+64*mn+24*case["m"]+4096
    phases = dict(input_buffer_allocation=10 if case["scope"] == "cold" else 0,
                  input_pack_and_upload=20 if case["scope"] != "resident" else 0, gemm_calls=30,
                  output_export=20 if case["scope"] != "resident" else 0,
                  input_and_transport_release=10 if case["scope"] == "cold" else 0)
    capture.update(device_name="synthetic mocked GPU only", target="gfx942" if case["candidate"]=="signed_radix_mfma" else "gfx1100", hip_runtime_version=70100000, hip_driver_version=70100000,
        setup_ns=1000, reference_and_dataset_ns=2000, resident_input_prepare_ns=100 if case["scope"] in ("resident","repeated_b") else None,
        a_checksum=checksum(a), b_checksum=checksum(b), output_checksum=checksum(expected),
        reference_operand_storage_bytes=48*ab, reference_fixed_operand_object_bytes=48,
        logical_live_payload_envelope_bytes=max(oracle, execution), clock_period_numerator=1, clock_period_denominator=1000000000,
        samples=[dict(host_call_ns=500*(index+1), uninstrumented_host_call_ns=100*(index+1),
                      instrumented_phases_ns=deepcopy(phases), events=[dict(label="synthetic event", elapsed_ns=10.5)],
                      exact_reference_passed=True, uninstrumented_exact_reference_passed=True) for index in range(repeats)],
        median_production_batch_ns=(repeats+1)*50., gemm_calls_per_second=1e9*case["reuse_count"]/((repeats+1)*50.),
        sustained=dict(calls=16, gemm_calls=16*case["reuse_count"], elapsed_ns=320000000, mean_batch_ns=20000000., mean_gemm_ns=20000000./case["reuse_count"],
                       minimum_calls=16, maximum_calls=4096, target_ns=250000000, stop_reason="duration", independent_audit_inside_interval=False, final_exact_reference_passed=True))
    return capture


class Captures(unittest.TestCase):
    def test_certificate_file_hashes_match_compiled_identity(self):
        from generate_wrap64_certificate_identity import rendered
        root=Path(__file__).resolve().parents[1]
        self.assertEqual((root/'src/backend_common/wrap64_certificate_identity.hpp').read_text(),rendered(root))

    def test_wrap_capture_isa_adapter_preserves_target_symbol_and_certificate(self):
        from gpu_isa_report import capture_metadata
        with tempfile.TemporaryDirectory() as directory:
            for candidate in CANDIDATES:
                capture=fixture(case_for(candidate));path=Path(directory)/'synthetic.json'
                path.write_text(json.dumps(capture))
                metadata=capture_metadata(path,'synthetic only')
                self.assertEqual(metadata['target_id'],capture['target'])
                self.assertEqual(metadata['selected_kernel'],capture['kernel_selection'])
                self.assertEqual(metadata['generated_reducer_identity'],capture['arithmetic_certificate_sha256'])
                self.assertEqual(metadata['artifact_capture'],capture['artifact_capture'])

    def test_all_candidates_scopes_patterns_reuse_and_boundaries(self):
        for candidate in CANDIDATES:
            for scope in SCOPES:
                for pattern in ("seeded", "carry", "zero"):
                    case = case_for(candidate, scope, pattern=pattern, reuse_count=8)
                    validate_capture(fixture(case), case, "fixture-run", 2, 1)
        for k in (4096, 4097, 32768, 33025, 65535, 65536, 65537, 131071):
            case = case_for("direct_v4", m=1, n=1, k=k)
            validate_case(case)
            self.assertIn("_u32acc_" if k <= 4096 else "_u64acc_", contract(case, "fixture-run", 2, 1)["kernel_selection"])
        for candidate in (*RDNA_CANDIDATES[:-1], "signed_radix_mfma"):
            validate_case(case_for(candidate, k=131071, m=1, n=1))

    def test_fixed_recipe_words_and_ring_output(self):
        a, b = dataset(BASE)
        self.assertEqual([int.from_bytes(a[i:i+8], "little") for i in range(0, 64, 8)], list(EDGES))
        self.assertEqual(dataset(BASE), dataset(BASE))
        self.assertNotEqual(dataset(BASE), dataset({**BASE, "pattern": "seeded"}))
        for operand in dataset({**BASE, "pattern": "zero"}):
            self.assertTrue(all(byte == 0 for byte in operand))
        self.assertEqual(fixture(case_for(pattern="zero"))["output_checksum"], checksum(bytes(BASE["m"]*BASE["n"]*8)))

    def test_invalid_controls_and_irrelevant_fields(self):
        for key, values in {"device": (-1, 64, True, 0.), "candidate": (None, "auto", "rocwmma"), "scope": (None, "warm", "kernel"),
            "m": (0, 513, True, 1.), "n": (0, 513), "k": (0, 131072), "pattern": (None, "random"), "reuse_count": (0, 257, True, 1.)}.items():
            for value in values:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    validate_case({**BASE, key: value})
        for case in ({**BASE, "m": 512, "n": 512, "k": 513}, {**BASE, "candidate": "rocwmma_corrected", "k": 32769}, {**BASE, "prefix": 20}):
            with self.assertRaises(ValueError):
                validate_case(case)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"cases.json"
            for cases in ([], [BASE, BASE], [dict(BASE, surprise=1)]):
                path.write_text(json.dumps(dict(schema="rns8_wrap64_cases_v1", cases=cases)))
                with self.assertRaises(ValueError):
                    load_cases(path)

    def test_all_contract_fields_bound_with_explicit_no_promotion(self):
        for key in contract(BASE, "fixture-run", 2, 1):
            capture = fixture(); capture.pop(key)
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_capture(capture, BASE, "fixture-run", 2, 1)
        for changes in (dict(m=True), dict(production_timing_hooks_enabled=True), dict(mma_terms_per_fold=131071),
                        dict(mathematical_pair_i32_cap=16), dict(static_lds_bytes=0), dict(public_ownership_measured=True)):
            capture = fixture(); capture.update(changes)
            with self.assertRaises(ValueError):
                validate_capture(capture, BASE, "fixture-run", 2, 1)

    def test_malformed_samples_phases_provenance_and_metadata(self):
        mutations = [lambda c: c.update(target="gfx942"), lambda c: c.update(a_checksum="0"*16), lambda c: c.update(hip_runtime_version=0),
            lambda c: c.update(output_checksum="short"), lambda c: c.update(resident_input_prepare_ns=None),
            lambda c: c.update(reference_operand_storage_bytes=0), lambda c: c.update(logical_live_payload_envelope_bytes=1),
            lambda c: c["samples"][0].update(uninstrumented_host_call_ns=0),
            lambda c: c["samples"][0].update(uninstrumented_exact_reference_passed=False),
            lambda c: c["samples"][0]["instrumented_phases_ns"].update(input_pack_and_upload=1),
            lambda c: c["samples"][0]["instrumented_phases_ns"].update(gemm_calls=0),
            lambda c: c["samples"][0]["events"][0].update(elapsed_ns=0),
            lambda c: c["samples"][0]["events"][0].update(elapsed_ns=float("inf")),
            lambda c: c.update(median_production_batch_ns=750), lambda c: c.update(gemm_calls_per_second=1),
            lambda c: c["sustained"].update(gemm_calls=17), lambda c: c["sustained"].update(mean_batch_ns=1),
            lambda c: c["sustained"].update(independent_audit_inside_interval=True),
            lambda c: c["build_manifest"].update(source_digest="invalid")]
        for mutation in mutations:
            capture = fixture(); mutation(capture)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_capture(capture, BASE, "fixture-run", 2, 1)
        censored = fixture(); censored["samples"][0]["host_call_ns"] = None; censored["samples"][0]["events"][0]["elapsed_ns"] = None
        validate_capture(censored, BASE, "fixture-run", 2, 1)
        roc = case_for("rocwmma_corrected"); capture = fixture(roc)
        del capture["build_manifest"]["external_dependencies"]["ROCWMMA"]
        capture["build_manifest"]["configuration"] = "\n".join(line for line in capture["build_manifest"]["configuration"].splitlines() if "ROCWMMA" not in line and not line.startswith("rocwmma="))
        with self.assertRaises(ValueError):
            validate_capture(capture, roc, "fixture-run", 2, 1)

    def test_tracked_case_sets_and_explicit_rocwmma_gap(self):
        root = Path(__file__).resolve().parents[1]/"benchmarks"/"wrap64"
        for name, count in (("smoke.json", 12), ("tail-reuse.json", 36), ("512-reuse.json", 48), ("long-k.json", 45), ("matched-boundaries.json", 12)):
            cases = load_cases(root/name); self.assertEqual(len(cases), count)
            for k in {case["k"] for case in cases}:
                expected = set(RDNA_CANDIDATES) if k <= 32768 else set(RDNA_CANDIDATES[:-1])
                self.assertEqual({case["candidate"] for case in cases if case["k"] == k}, expected)

    def test_cdna_generation_maps_and_repeated_b_scope(self):
        case=case_for("signed_radix_mfma", "repeated_b")
        for target,width in (("gfx90a",16),("gfx942",32),("gfx950",64)):
            capture=fixture(case);capture.update(contract(case,"fixture-run",2,1,target),target=target)
            validate_capture(capture,case,"fixture-run",2,1)
            for key,value in (("mma_terms_per_fold",128),("static_lds_bytes",4096),("target","gfx1100"),
                              ("resident_input_prepare_ns",None)):
                broken=deepcopy(capture);broken[key]=value
                with self.assertRaises(ValueError):validate_capture(broken,case,"fixture-run",2,1)
        for name,count in (("cdna-small.json",27),("cdna-carry-slabs.json",33),("cdna-boundaries.json",24),("cdna-sustained.json",24)):
            cases=load_cases(Path(__file__).resolve().parents[1]/"benchmarks"/"wrap64"/name)
            self.assertEqual(len(cases),count)
            self.assertEqual({x['candidate'] for x in cases},set(CDNA_CANDIDATES))


class Comparison(unittest.TestCase):
    def test_only_complete_independent_cohorts_and_no_roc_extrapolation(self):
        for k in (17, 65537):
            observations = [dict(case=case_for(candidate, k=k), independent_run=index, mean_batch_ns=100+10*CANDIDATES.index(candidate)+index)
                            for candidate in (RDNA_CANDIDATES if k <= 32768 else RDNA_CANDIDATES[:-1]) for index in range(5)]
            record = summarize(observations)[0]
            self.assertTrue(record["complete_matched_cohort"]); self.assertFalse(record["promotion_eligible"])
            self.assertEqual(record["comparison_scope"], "four_way" if k <= 32768 else "rocwmma_unavailable_k_cap_three_way")
            if k > 32768:
                self.assertNotIn("rocwmma_corrected", record["observed_process_statistics"])
            else:
                self.assertIn("rocwmma_corrected", record["observed_process_statistics"])
            self.assertFalse(summarize(observations[:-1])[0]["complete_matched_cohort"])
            self.assertFalse(summarize([*observations, observations[0]])[0]["complete_matched_cohort"])

    def test_cdna_requires_independent_complete_non_rdna_cohort(self):
        observations=[dict(case=case_for(candidate),independent_run=run,mean_batch_ns=100+run)
                      for candidate in CDNA_CANDIDATES for run in range(2)]
        result=summarize(observations)[0]
        self.assertTrue(result['complete_matched_cohort'])
        self.assertEqual(result['comparison_scope'],'cdna_signed_mfma_direct_v4_native_u64')
        self.assertFalse(summarize([x for x in observations if x['independent_run']==0])[0]['complete_matched_cohort'])
        self.assertFalse(summarize(observations+[dict(case=case_for('native_u8'),independent_run=0,mean_batch_ns=1)])[0]['complete_matched_cohort'])


class Collection(unittest.TestCase):
    def setUp(self):
        # Host metadata may itself invoke subprocess.run (uname -p on Linux).
        host = patch("collect_wrap64.platform.platform", return_value="synthetic-platform")
        host.start()
        self.addCleanup(host.stop)

    def scenario(self, mutation=None, status=0):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); exe = root/"fixture"; exe.write_bytes(b"fixture only")
            cases = root/"cases.json"; cases.write_text(json.dumps(dict(schema="rns8_wrap64_cases_v1", cases=[BASE])))
            args = argparse.Namespace(exe=str(exe), cases=str(cases), output=str(root/"output"), runs=2, repeats=2, warmups=1, timeout=10)
            count = 0
            def child(command, **kwargs):
                nonlocal count
                count += 1; capture = fixture(BASE, command[-1])
                if mutation:
                    mutation(capture, count, exe)
                return subprocess.CompletedProcess(command, status, json.dumps(capture), "synthetic stderr")
            with patch.dict(os.environ, {"RNS8_WRAP64_HIP_COLPAIR_EXPERIMENT": "0"}), patch("collect_wrap64.subprocess.run", side_effect=child):
                if mutation or status:
                    with self.assertRaises(ValueError):
                        collect(args)
                else:
                    collect(args)
            records = [json.loads(path.read_text()) for path in sorted((root/"output").glob("*.record.json"))]
            self.assertEqual(len(records), count); self.assertEqual(len({record["run_id"] for record in records}), count)
            self.assertEqual(records[-1]["validated"], not bool(mutation or status))
            for record in records:
                self.assertFalse(record["promotion_eligible"])
                self.assertEqual(set(record["tool_sha256"]), {"collect_wrap64.py", "collect_finite_verify.py", "build_manifest.py"})
            self.assertEqual(len(list((root/"output").glob("comparison-*.json"))), 0 if mutation or status else 1)

    def test_collector_process_and_failure_receipts(self):
        self.scenario()
        for mutation in (lambda c, i, exe: c.update(run_id="wrong"), lambda c, i, exe: c.update(candidate="native_u8"),
                         lambda c, i, exe: c.update(output_checksum="0"*16) if i == 2 else None,
                         lambda c, i, exe: c.update(device_name="changed") if i == 2 else None,
                         lambda c, i, exe: c["build_manifest"].update(source_commit="e"*40) if i == 2 else None,
                         lambda c, i, exe: (exe.parent/"cases.json").write_text("{}"),
                         lambda c, i, exe: exe.write_bytes(b"changed artifact")):
            with self.subTest(mutation=mutation):
                self.scenario(mutation)
        self.scenario(status=77)

    def test_colpair_environment_is_rejected_before_child(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); cases = root/"cases.json"; cases.write_text(json.dumps(dict(schema="rns8_wrap64_cases_v1", cases=[BASE])))
            args = argparse.Namespace(exe="not launched", cases=str(cases), output=str(root/"output"), runs=1, repeats=2, warmups=1, timeout=1)
            with patch.dict(os.environ, {"RNS8_WRAP64_HIP_COLPAIR_EXPERIMENT": "1"}), patch("collect_wrap64.subprocess.run") as child:
                with self.assertRaises(ValueError):
                    collect(args)
                child.assert_not_called()

    def test_timeout_retains_partial_raw_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); exe = root/"fixture"; exe.write_bytes(b"fixture only")
            cases = root/"cases.json"; cases.write_text(json.dumps(dict(schema="rns8_wrap64_cases_v1", cases=[BASE])))
            args = argparse.Namespace(exe=str(exe), cases=str(cases), output=str(root/"output"), runs=1, repeats=2, warmups=1, timeout=1)
            with patch.dict(os.environ, {"RNS8_WRAP64_HIP_COLPAIR_EXPERIMENT": "0"}), patch("collect_wrap64.subprocess.run", side_effect=subprocess.TimeoutExpired("fixture", 1, output=b"partial stdout", stderr=b"partial stderr")):
                with self.assertRaises(subprocess.TimeoutExpired):
                    collect(args)
            self.assertEqual(next((root/"output").glob("*.stdout.json")).read_bytes(), b"partial stdout")
            self.assertEqual(next((root/"output").glob("*.stderr.txt")).read_bytes(), b"partial stderr")
            self.assertFalse(json.loads(next((root/"output").glob("*.record.json")).read_text())["validated"])
            self.assertFalse(list((root/"output").glob("comparison-*.json")))


@unittest.skipUnless(os.environ.get("RNS8_WRAP64_CALIBRATION_EXE"), "set RNS8_WRAP64_CALIBRATION_EXE for pre-context CLI fixtures")
class CompiledCLI(unittest.TestCase):
    def test_help_and_malformed_cases_never_run_gpu(self):
        exe = os.environ["RNS8_WRAP64_CALIBRATION_EXE"]
        self.assertEqual(subprocess.run([exe, "--help"], capture_output=True).returncode, 0)
        valid = ["63", "signed_radix", "resident", "2", "3", "17", "carry", "1", "2", "1", "fixture"]
        invalid = [valid[:-1], valid+["extra"]]
        for index, value in ((0, "-1"), (0, "64"), (1, "auto"), (2, "unknown"), (3, "0"), (4, "513"), (5, "131072"),
                             (6, "unknown"), (7, "0"), (7, "257"), (8, "0"), (9, "0"), (10, "bad/id")):
            args = list(valid); args[index] = value; invalid.append(args)
        invalid.append(["63", "rocwmma_corrected", "resident", "2", "3", "32769", "carry", "1", "2", "1", "fixture"])
        for args in invalid:
            with self.subTest(args=args):
                self.assertEqual(subprocess.run([exe, *args], capture_output=True, timeout=10).returncode, 1)


if __name__ == "__main__":
    unittest.main()
