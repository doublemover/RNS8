"""Authored collector fixtures mock all children; optional CLI checks never run GPU work."""
import copy
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from collect_native_calibration import (
    ADAPTIVE, CANDIDATES, CONVERSION, ENCODED, QUEUED, REUSE, TILED, adaptive_contract, capture_contract, epilogue_contract, main, narrow_contract, tiled_contract,
    reuse_contract, summarize_encoded_reuse, conversion_contract, summarize_conversion,
    scheduling_contract, strict_json, valid_sustained, validate_capture, validate_case,
)
from native_locality_contract import private_contract as locality_contract
from test_capture_identity_support import synthetic_identity


BASE = dict(candidate="wmma", m=17, n=19, k=257, prefix=3, group=2, unsigned_inputs=False)


def case_for(candidate, **changes):
    case = {**BASE, "candidate": candidate}
    if candidate not in (*REUSE, *CONVERSION, "grid_prepared", *QUEUED, *ADAPTIVE, *TILED):
        case["group"] = 1
    if candidate.startswith("narrow_"):
        case["prefix"] = 1
    if candidate.startswith("complex_"):
        case["prefix"] = 20
    if candidate in CONVERSION:
        case.update(k_depth=16,lds_buffers=0,waves=4,conversion="replicated",reuse_count=1,input_change="stable")
    if candidate in TILED:
        case.update(k_depth=16, lds_buffers=0 if candidate == "tiled_register" else 1)
    if candidate in ENCODED:
        case["reuse_count"] = 1
    if candidate in ADAPTIVE:
        case.update(prefix=20, bound_method="axis_l1")
        if candidate == "adaptive_compact": case["workers"] = 32
    return validate_case({**case, **changes})


def capture_for(case, run_id="fixture-run", repeats=3, warmups=1):
    capture = capture_contract(case, run_id, repeats, warmups)
    if case["candidate"] in ("queue_prepared", "grid_prepared"):
        capture.update(queue_prepare_ns=2000.25, queue_descriptor_storage_bytes=1024)
    if case["candidate"].startswith("complex_"):
        capture.update(input_prepare_ns=3000.5)
    synthetic_identity(capture)
    capture.update(device="synthetic mocked GPU", target="gfx1100", runtime_version=70100000,
                   timing_scope="synthetic scoped timing", cache_state="synthetic repeated inputs",
                   sustained=dict(calls=16, elapsed_ns=320000000, mean_call_ns=20000000,
                                  final_exact_reference_passed=True, scope="synthetic final readback"),
                   samples=[dict(host_call_ns=1000.25, uninstrumented_host_call_ns=900.5,
                                 exact_reference_passed=True, uninstrumented_exact_reference_passed=True,
                                 events=[dict(label="synthetic event", elapsed_ns=500.25)]) for _ in range(repeats)])
    if case["candidate"] in REUSE:
        capture["encoded_b_prepare_ns"] = 4000.25 if case["candidate"] == "encoded_b_warm" else None
        capture["sustained"].update(gemm_calls=16*case["reuse_count"],
            mean_gemm_ns=20000000/case["reuse_count"],verified_outputs_per_batch=1)
    if case["candidate"] in CONVERSION:
        capture["conversion_prepare_ns"]=4000.25 if case["candidate"]=="conversion_prepared" else None
        capture["sustained"].update(gemm_calls=16*case["reuse_count"],mean_gemm_ns=20000000/case["reuse_count"],verified_outputs_per_batch=1)
    return capture


class CaseTest(unittest.TestCase):
    def test_native_locality_and_real_displacement_receipts(self):
        for candidate in ('queue_prepared','split_k'):
            for order in ('ordinary','output-neighbor','modulus','same-output'):
                for mib in (0,256):
                    case=case_for(candidate,locality_order=order,cache_displace_mib=mib,cache_model_mib=64 if mib else 0)
                    capture=capture_for(case);capture['locality']=locality_contract(case,'gfx1100')
                    calls=23 if mib else 0;witness=(calls*0x9e3779b9)&0xffffffff
                    capture['cache_displacement']=dict(version=1,allocation_bytes=mib*1024*1024,modeled_cache_bytes=(64 if mib else 0)*1024*1024,
                        line_stride_bytes=64,passes_per_call=1 if mib else 0,submissions=calls,host_ns=100 if mib else 0,
                        first_word=witness,last_word=witness,witness_verified=True,inside_timed_call=True,before_every_sustained_call=True,cold_cache_verified=False)
                    validate_capture(capture,case,'fixture-run',3,1)
                    for field in ('active_entries','unique_source_bytes','permutation_fingerprint','partial_output_bytes'):
                        bad=copy.deepcopy(capture);bad['locality'][field]+=1
                        with self.assertRaises(ValueError):validate_capture(bad,case,'fixture-run',3,1)
                    for field,value in (('cold_cache_verified',True),('before_every_sustained_call',False),('submissions',calls+1)):
                        bad=copy.deepcopy(capture);bad['cache_displacement'][field]=value
                        with self.assertRaises(ValueError):validate_capture(bad,case,'fixture-run',3,1)
        with self.assertRaises(ValueError):case_for('wmma',locality_order='same-output',cache_displace_mib=256,cache_model_mib=64)
        with self.assertRaises(ValueError):case_for('queue_prepared',locality_order='same-output',cache_displace_mib=32,cache_model_mib=64)

    def test_encoded_reuse_controls_storage_and_irrelevant_arguments(self):
        self.assertEqual(validate_case(BASE)["reuse_count"],1)
        self.assertNotIn("reuse_count",BASE)
        for candidate in REUSE:
            for reuse in (1,2,8,32,256):
                for group in (1,2,4):
                    for unsigned in (False,True):
                        case=case_for(candidate,reuse_count=reuse,group=group,unsigned_inputs=unsigned)
                        contract=reuse_contract(case)
                        self.assertEqual(contract["batch_gemm_calls"],reuse)
                        encoded=candidate in ENCODED
                        expected=(3*19*272+4) if encoded else 0
                        self.assertEqual(contract["encoded_b_cache_logical_bytes"],expected)
                        self.assertEqual(contract["encoded_b_k_words"],68 if encoded else 0)
                        self.assertEqual(contract["encoded_b_preparation_in_batch"],candidate=="encoded_b_cold")
                        self.assertFalse(contract["encoded_b_public_ownership_measured"])
                        validate_capture(capture_for(case),case,"fixture-run",3,1)
            for bad in (None,True,False,0,-1,257,1.0,"1",[],{}):
                with self.subTest(candidate=candidate,bad=bad),self.assertRaises(ValueError):
                    case_for(candidate,reuse_count=bad)
        for candidate in ENCODED:
            missing=case_for(candidate);del missing["reuse_count"]
            with self.assertRaises(ValueError):validate_case(missing)
        for candidate in set(CANDIDATES)-set(REUSE)-set(CONVERSION):
            with self.subTest(candidate=candidate),self.assertRaisesRegex(ValueError,"reuse_count applies only"):
                case_for(candidate,reuse_count=1)

    def test_prepared_queue_requires_setup_cost_and_storage(self):
        for candidate in ("queue_prepared", "grid_prepared"):
            case = case_for(candidate)
            for key in ("queue_prepare_ns", "queue_descriptor_storage_bytes"):
                for value in (None, False, 0, -1, "100", float("nan"), float("inf")):
                    with self.subTest(key=key, value=value):
                        capture = capture_for(case)
                        capture[key] = value
                        with self.assertRaises(ValueError):
                            validate_capture(capture, case, "fixture-run", 3, 1)
                capture = capture_for(case)
                del capture[key]
                with self.assertRaises(ValueError):
                    validate_capture(capture, case, "fixture-run", 3, 1)

    def test_all_modes_groups_and_signedness(self):
        for candidate in CANDIDATES:
            for unsigned in (False, True):
                if (candidate == "narrow_scalar" or candidate.startswith("complex_")) and unsigned:
                    continue
                for group in ((1, 2, 4) if candidate in (*REUSE, *CONVERSION, "grid_prepared", *QUEUED, *ADAPTIVE, *TILED) else (1,)):
                    with self.subTest(candidate=candidate, unsigned=unsigned, group=group):
                        case = case_for(candidate, group=group, unsigned_inputs=unsigned)
                        self.assertEqual(validate_case(case), case)
                        validate_capture(capture_for(case), case, "fixture-run", 3, 1)

    def test_worker_defaults_are_normalized_without_mutation(self):
        for candidate in QUEUED:
            original = {**BASE, "candidate": candidate}
            self.assertEqual(validate_case(original)["workers"], 32)
            self.assertNotIn("workers", original)
            for workers in (1, 2, 32, 1024):
                self.assertEqual(case_for(candidate, workers=workers)["workers"], workers)
        for candidate in set(CANDIDATES) - set(QUEUED) - {"adaptive_compact"}:
            with self.subTest(candidate=candidate), self.assertRaisesRegex(ValueError, "only to queue/queue_prepared/split_k"):
                case_for(candidate, workers=32)

    def test_schedule_count_clamping_and_split_boundaries(self):
        for candidate in QUEUED:
            for group in (1, 2, 4):
                for unsigned in (False, True):
                    for k in (1, 2, 3, 65535, 65536, 65537):
                        for workers in (1, 32, 1024):
                            case = case_for(candidate, m=1, n=17, k=k, prefix=5,
                                            group=group, unsigned_inputs=unsigned, workers=workers)
                            contract = scheduling_contract(case)
                            splits = min(k, 3) if candidate == "split_k" else 1
                            count = 2 * ((5 + group - 1) // group) * splits
                            self.assertEqual(contract["queue_descriptors"], count)
                            self.assertEqual(contract["splits"], splits)
                            self.assertEqual(contract["requested_workers"], workers)
                            self.assertEqual(contract["effective_workers"], min(count, workers))
                            self.assertEqual(contract["queue_kernel_selection"],
                                             f"homogeneous_group_{group}_" + ("unsigned" if unsigned else "signed"))
        for candidate in set(CANDIDATES) - set(QUEUED) - {"adaptive_compact"}:
            contract = scheduling_contract(case_for(candidate))
            self.assertEqual(contract, dict(splits=1, requested_workers=0, effective_workers=0,
                                           queue_descriptors=0, queue_kernel_selection="not_applicable"))

    def test_adaptive_workers_are_exact_and_counts_remain_device_only(self):
        for group in (1, 2, 4):
            for unsigned in (False, True):
                for workers in (1, 32, 512, 65535):
                    case = case_for("adaptive_compact", m=1, n=1, k=1, group=group,
                                    unsigned_inputs=unsigned, workers=workers)
                    schedule = scheduling_contract(case)
                    self.assertEqual(schedule["requested_workers"], workers)
                    self.assertEqual(schedule["effective_workers"], workers)
                    self.assertEqual(schedule["queue_descriptors"], 0)
                    contract = adaptive_contract(case)
                    self.assertEqual(contract["adaptive_dense_group_upper_bound"], (20 + group - 1) // group)
                    self.assertIsNone(contract["adaptive_actual_compact_groups"])
                    self.assertEqual(contract["bound_method"], "axis_l1")
                    self.assertIn("no_count_clamp", contract["adaptive_worker_policy"])
        missing = case_for("adaptive_compact"); del missing["workers"]
        with self.assertRaises(ValueError): validate_case(missing)

    def test_tiled_storage_controls_and_exact_static_resources(self):
        for group in (1, 2, 4):
            for unsigned in (False, True):
                for candidate, depth, buffers, lds in (
                        ("tiled_register", 16, 0, 0),
                        ("tiled_lds", 16, 1, 8576), ("tiled_lds", 16, 2, 17152),
                        ("tiled_lds", 32, 1, 16896), ("tiled_lds", 32, 2, 33792)):
                    case = case_for(candidate, group=group, unsigned_inputs=unsigned,
                                    k_depth=depth, lds_buffers=buffers)
                    with self.subTest(case=case):
                        contract = tiled_contract(case)
                        self.assertEqual(contract["tiled_raw_lds_bytes"], lds)
                        self.assertEqual(contract["tiled_waves"], 4)
                        self.assertEqual(contract["tiled_lds_budget_bytes"], 65536)
                        self.assertFalse(contract["tiled_resource_query_is_device_qualification"])
                        self.assertTrue(contract["tiled_kernel_selection"].endswith(f"_d{depth}_b{buffers}"))
                        validate_capture(capture_for(case), case, "fixture-run", 3, 1)
        for candidate in TILED:
            for key in ("k_depth", "lds_buffers"):
                missing = case_for(candidate); del missing[key]
                with self.assertRaises(ValueError): validate_case(missing)
        for candidate in TILED:
            for depth in (None, True, False, 0, 15, 17, 31, 33, 64, "16", 16.0, [], {}):
                with self.assertRaises(ValueError): case_for(candidate, k_depth=depth)
            for buffers in (None, True, False, -1, 3, "1", 1.0, [], {}):
                with self.assertRaises(ValueError): case_for(candidate, lds_buffers=buffers)
        for depth, buffers in ((32, 0), (16, 1), (16, 2)):
            with self.assertRaises(ValueError): case_for("tiled_register", k_depth=depth, lds_buffers=buffers)
        with self.assertRaises(ValueError): case_for("tiled_lds", lds_buffers=0)
        for candidate in set(CANDIDATES) - set(TILED) - set(CONVERSION):
            for changes in (dict(k_depth=16), dict(lds_buffers=0)):
                with self.assertRaises(ValueError): case_for(candidate, **changes)

    def test_tiled_resources_and_storage_identity_cannot_be_rounded_or_invented(self):
        case = case_for("tiled_lds", k_depth=32, lds_buffers=2)
        for key, values in (
                ("k_depth", (16, True, 32.0)), ("lds_buffers", (1, True, 2.0)),
                ("tiled_raw_lds_bytes", (0, 32768, True, 33792.0)),
                ("tiled_resource_query_is_device_qualification", (True, 0)),
                ("tiled_prefetch", ("asynchronous_dma_overlap", "direct_native_register_loads")),
                ("tiled_waves", (1, 2, 4.0)), ("tiled_output_rows", (16, True, 32.0))):
            for value in values:
                capture = capture_for(case); capture[key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    validate_capture(capture, case, "fixture-run", 3, 1)

    def test_shape_boundaries(self):
        for key, high in (("m", 1024), ("n", 1024), ("k", 65537), ("prefix", 20)):
            for value in (1, high):
                validate_case({**BASE, "m": 1, "n": 1, key: value})
        validate_case({**BASE, "m": 1024, "n": 1024, "k": 1024})
        with self.assertRaises(ValueError):
            validate_case({**BASE, "m": 1024, "n": 1024, "k": 1025})

    def test_tracked_case_sets_cover_every_mode_and_adaptive_controls(self):
        root = Path(__file__).resolve().parents[1] / "benchmarks/native-calibration"
        smoke = [validate_case(row) for row in strict_json((root / "all-modes-smoke.json").read_text(encoding="utf-8"))]
        self.assertEqual({row["candidate"] for row in smoke}, set(CANDIDATES))
        adaptive = [validate_case(row) for row in strict_json((root / "adaptive-native.json").read_text(encoding="utf-8"))]
        self.assertLessEqual(len(adaptive), 64)
        self.assertEqual(len({json.dumps(row, sort_keys=True) for row in adaptive}), len(adaptive))
        for candidate in ADAPTIVE:
            for group in (1, 2, 4):
                for unsigned in (False, True):
                    self.assertTrue(any(row["candidate"] == candidate and row["group"] == group
                                        and row["unsigned_inputs"] == unsigned for row in adaptive))
        self.assertEqual({row["workers"] for row in adaptive if row["candidate"] == "adaptive_compact"}, {32, 512})
        for filename in ("tiled-native-tail.json", "tiled-native-512.json"):
            rows = [validate_case(row) for row in strict_json((root / filename).read_text(encoding="utf-8"))]
            self.assertLessEqual(len(rows), 64)
            self.assertEqual(len({json.dumps(row, sort_keys=True) for row in rows}), len(rows))
            for group in (1, 2, 4):
                for unsigned in (False, True):
                    modes = {(row["candidate"], row.get("k_depth"), row.get("lds_buffers"))
                             for row in rows if row["group"] == group and row["unsigned_inputs"] == unsigned}
                    self.assertTrue({("wmma", None, None), ("tiled_register", 16, 0),
                                     ("tiled_lds", 16, 1), ("tiled_lds", 16, 2),
                                     ("tiled_lds", 32, 1), ("tiled_lds", 32, 2)} <= modes)
        for filename in ("encoded-b-tail.json","encoded-b-512.json"):
            rows=[validate_case(row) for row in strict_json((root/filename).read_text(encoding="utf-8"))]
            self.assertEqual(len(rows),54)
            self.assertEqual(len({json.dumps(row,sort_keys=True) for row in rows}),54)
            for unsigned in (False,True):
                for group in (1,2,4):
                    for reuse in (1,8,32):
                        self.assertEqual({row["candidate"] for row in rows if row["unsigned_inputs"]==unsigned
                            and row["group"]==group and row["reuse_count"]==reuse},set(REUSE))

    def test_malformed_cases(self):
        malformed = [None, [], "case", {}, {**BASE, "extra": 1}, {k: v for k, v in BASE.items() if k != "k"}]
        for key in ("m", "n", "k", "prefix", "group"):
            for value in (None, True, False, 0, -1, 1.0, "1", [], {}, 10**100):
                malformed.append({**BASE, key: value})
        for value in (None, "unknown", [], {}, 1, False):
            malformed.append({**BASE, "candidate": value})
        for value in (None, 0, 1, "false", [], {}):
            malformed.append({**BASE, "unsigned_inputs": value})
        malformed += [{**BASE, "m": 1025}, {**BASE, "n": 1025}, {**BASE, "k": 65538},
                      {**BASE, "prefix": 21}, {**BASE, "group": 3}]
        for candidate in QUEUED:
            for value in (None, True, False, 0, -1, 1025, 1.0, "32", [], {}):
                malformed.append({**BASE, "candidate": candidate, "workers": value})
        for candidate in (c for c in CANDIDATES if c not in (*REUSE, *CONVERSION, "grid_prepared", *QUEUED, *ADAPTIVE, *TILED)):
            malformed.append({**case_for(candidate), "group": 2})
        for candidate in (c for c in CANDIDATES if c.startswith("narrow_")):
            malformed += [{**case_for(candidate), "prefix": 2}, {**case_for(candidate), "k": 32769}]
        malformed.append({**case_for("narrow_scalar"), "unsigned_inputs": True})
        for candidate in ("complex_3m", "complex_4m"):
            malformed += [{**case_for(candidate), "prefix": 19}, {**case_for(candidate), "unsigned_inputs": True}]
        for candidate in ADAPTIVE:
            malformed.append({**case_for(candidate), "prefix": 19})
            for method in (None, True, 0, "tile_max", "panel_l1", "axis_l1 ", [], {}):
                malformed.append({**case_for(candidate), "bound_method": method})
        for workers in (None, True, False, 0, -1, 65536, 1.0, "32", [], {}):
            malformed.append({**case_for("adaptive_compact"), "workers": workers})
        malformed.append({**case_for("adaptive_grid"), "workers": 1})
        for candidate in set(CANDIDATES) - set(ADAPTIVE):
            malformed.append({**case_for(candidate), "bound_method": "axis_l1"})
        for case in malformed:
            with self.subTest(case=case), self.assertRaises(ValueError):
                validate_case(case)

    def test_narrow_and_epilogue_contracts(self):
        for unsigned in (False, True):
            for candidate in (c for c in CANDIDATES if c.startswith("narrow_")):
                if candidate == "narrow_scalar" and unsigned:
                    continue
                case = case_for(candidate, unsigned_inputs=unsigned, k=32768)
                contract = narrow_contract(case)
                self.assertEqual(contract["abs_partial_sum_bound"], 32768 * (255 if unsigned else 128)**2)
                self.assertLessEqual(contract["abs_partial_sum_bound"], (1 << 31) - 1)
                self.assertEqual(contract["selected_rns_prefix"] > 0, candidate == "narrow_rns")
            for candidate in (c for c in CANDIDATES if c.startswith("epilogue_")):
                contract = epilogue_contract(case_for(candidate, unsigned_inputs=unsigned))
                self.assertEqual(contract["expression"]["alpha"], (1 << 63) - 1 if unsigned else -(1 << 63))
                self.assertEqual(contract["expression"]["beta"], (1 << 63) - 1)
                self.assertEqual(bool(contract["materialized_intermediate_bytes"]), candidate == "epilogue_unfused")


class ConversionContracts(unittest.TestCase):
    def test_every_typed_recipe_repeats_and_preparation_boundary(self):
        recipes=((16,0,1,"replicated"),(16,0,1,"cooperative"),(16,0,1,"paired_columns"),(16,0,4,"cooperative"),
                 (16,1,4,"replicated"),(32,2,4,"replicated"),(16,1,4,"encoded_lds"),(32,1,4,"encoded_lds"))
        for depth,buffers,waves,mode in recipes:
            for candidate in ("conversion_cold","conversion_prepared"):
                for group in (1,2,4):
                    if mode=="paired_columns" and group==4:continue
                    case=case_for(candidate,k_depth=depth,lds_buffers=buffers,waves=waves,conversion=mode,group=group,reuse_count=8,input_change="reverse_k")
                    capture=capture_for(case);validate_capture(capture,case,"fixture-run",3,1)
                    contract=conversion_contract(case)
                    self.assertFalse(contract["conversion_source_counts_measured"])
                    if mode=="encoded_lds":
                        self.assertEqual(contract["conversion_raw_lds_bytes"],0)
                        self.assertEqual(contract["conversion_scalars_per_interior_panel"],2*32*depth*group)
                    if mode=="paired_columns":
                        self.assertEqual(contract['conversion_output_tile_cols'],32)
                        self.assertEqual(contract['conversion_column_map']['columns'],[2,4,8,16,1])
                        self.assertEqual(contract['conversion_accumulator_i32_per_lane'],16*group)
                        for field,value in [('columns',[1,2,4,8,16]),('wave_size',64),('fingerprint',0)]:
                            wrong=copy.deepcopy(capture);wrong['conversion_column_map'][field]=value
                            with self.assertRaises(ValueError):validate_capture(wrong,case,"fixture-run",3,1)
                    for field in ("conversion_identity","conversion_preparation_in_batch","conversion_total_lds_bytes"):
                        wrong=copy.deepcopy(capture);wrong[field]=None
                        with self.assertRaises(ValueError):validate_capture(wrong,case,"fixture-run",3,1)
        for changes in (dict(waves=2),dict(conversion="encoded_lds",lds_buffers=2),dict(conversion="cooperative",lds_buffers=1),dict(reuse_count=True),dict(input_change="unrecorded")):
            with self.assertRaises(ValueError):case_for("conversion_prepared",**changes)

    def test_setup_is_added_within_each_process_and_permutations_preserve_result(self):
        observations=[]
        for run,batch,setup in (("one",100,50),("two",120,10),("three",140,0)):
            observations.append(dict(case=case_for("conversion_prepared"),run_id=run,independent_repeat=len(observations),
                                     sampled_batch_ns=batch,prepare_ns=setup,sustained_batch_ns=batch,provenance_complete=False))
        forward=summarize_conversion(observations)
        self.assertEqual(forward,summarize_conversion(list(reversed(observations))))
        row=forward[0]["recipes"][0]
        self.assertEqual(row["sampled_batch_ns_median"],120)
        self.assertEqual(row["setup_plus_sampled_batch_ns_median"],140)
        self.assertEqual(row["independent_processes"],3)
        self.assertFalse(forward[0]["promotion_eligible"])


class CaptureTest(unittest.TestCase):
    def test_encoded_preparation_and_gemm_count_scope_are_strict(self):
        for candidate in REUSE:
            case=case_for(candidate,reuse_count=8)
            for key in ("encoded_b_prepare_ns",):
                missing=capture_for(case);del missing[key]
                with self.assertRaises(ValueError):validate_capture(missing,case,"fixture-run",3,1)
                bad=(None,False,0,-1,"1",float("nan"),float("inf")) if candidate=="encoded_b_warm" else (0,False,1,"not_applicable")
                for value in bad:
                    capture=capture_for(case);capture[key]=value
                    with self.subTest(candidate=candidate,key=key,value=value),self.assertRaises(ValueError):
                        validate_capture(capture,case,"fixture-run",3,1)
            for key in ("gemm_calls","mean_gemm_ns","verified_outputs_per_batch"):
                for value in (None,True,0,-1,"1",float("nan"),float("inf")):
                    capture=capture_for(case);capture["sustained"][key]=value
                    with self.subTest(candidate=candidate,key=key,value=value),self.assertRaises(ValueError):
                        validate_capture(capture,case,"fixture-run",3,1)
                capture=capture_for(case);del capture["sustained"][key]
                with self.assertRaises(ValueError):validate_capture(capture,case,"fixture-run",3,1)
            capture=capture_for(case);capture["sustained"]["gemm_calls"]=16
            with self.assertRaises(ValueError):validate_capture(capture,case,"fixture-run",3,1)

    def test_adaptive_never_substitutes_dense_count_for_device_compact_count(self):
        for candidate in ADAPTIVE:
            case = case_for(candidate)
            for key, values in (("adaptive_actual_compact_groups", (0, 1, 40, False, "unknown")),
                                ("adaptive_dense_group_upper_bound", (True, 40.0, None, -1)),
                                ("effective_workers", (True, 32.0, None)),
                                ("bound_method", ("tile_max", None)),
                                ("adaptive_worker_policy", ("clamped_to_compact_count", None))):
                for value in values:
                    with self.subTest(candidate=candidate, key=key, value=value):
                        capture = capture_for(case); capture[key] = value
                        with self.assertRaises(ValueError): validate_capture(capture, case, "fixture-run", 3, 1)

    def test_complex_costs_and_component_contract(self):
        for candidate, products, scratch in (("complex_3m", 3, (17*257+257*19+3*17*19)*20),
                                              ("complex_4m", 4, 4*17*19*20)):
            case = case_for(candidate)
            record = capture_for(case)
            self.assertEqual(record["real_products"], products)
            self.assertEqual(record["scratch_peak_bytes"], scratch)
            self.assertEqual(record["residue_output_bytes"], 2*17*19*20)
            self.assertEqual(record["component_bound"], str(257*(1 << 127)))
            for invalid in (None, False, 0, -1, "100", float("nan")):
                with self.subTest(candidate=candidate, invalid=invalid):
                    record["input_prepare_ns"] = invalid
                    with self.assertRaises(ValueError): validate_capture(record, case, "fixture-run", 3, 1)

    def test_every_required_metadata_field_is_checked_for_all_modes(self):
        for candidate in CANDIDATES:
            case = case_for(candidate)
            capture = capture_for(case)
            for key in capture_contract(case, "fixture-run", 3, 1):
                with self.subTest(candidate=candidate, key=key):
                    missing = copy.deepcopy(capture)
                    del missing[key]
                    with self.assertRaises(ValueError):
                        validate_capture(missing, case, "fixture-run", 3, 1)
                    wrong = copy.deepcopy(capture)
                    wrong[key] = "wrong" if type(wrong[key]) is not str else ""
                    with self.assertRaises(ValueError):
                        validate_capture(wrong, case, "fixture-run", 3, 1)

    def test_integer_controls_cannot_be_bools_or_floats(self):
        case = case_for("queue", m=1, n=1, prefix=1, group=1, workers=1)
        for key in ("m", "n", "group", "prefix", "requested_workers", "effective_workers", "queue_descriptors", "splits"):
            for value in (True, 1.0):
                capture = capture_for(case)
                capture[key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    validate_capture(capture, case, "fixture-run", 3, 1)

    def test_malformed_capture_and_measurements(self):
        case = case_for("wmma")
        for bad in (None, [], True, "json", 1):
            with self.assertRaises(ValueError):
                validate_capture(bad, case, "fixture-run", 3, 1)
        changes = [dict(samples=None), dict(samples={}), dict(samples=[None]*3), dict(samples=[]),
                   dict(sustained=None), dict(runtime_version=True), dict(runtime_version=0),
                   dict(device=""), dict(target=None), dict(timing_scope=""), dict(cache_state=None)]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_capture({**capture_for(case), **change}, case, "fixture-run", 3, 1)
        for key in ("host_call_ns", "uninstrumented_host_call_ns"):
            for bad in (None, 0, -1, True, "1", float("nan"), float("inf"), 10**1000):
                capture = capture_for(case)
                capture["samples"][0][key] = bad
                with self.subTest(key=key, bad=str(bad)[:20]), self.assertRaises(ValueError):
                    validate_capture(capture, case, "fixture-run", 3, 1)
        for bad in (None, {}, [None], [dict(label="", elapsed_ns=1)], [dict(label="x", elapsed_ns=-1)],
                    [dict(label="x", elapsed_ns=True)], [dict(label="x")]):
            capture = capture_for(case)
            capture["samples"][0]["events"] = bad
            with self.assertRaises(ValueError):
                validate_capture(capture, case, "fixture-run", 3, 1)
        capture = capture_for(case)
        capture["samples"][0]["events"][0]["elapsed_ns"] = None
        validate_capture(capture, case, "fixture-run", 3, 1)

    def test_sustained_boundaries(self):
        value = capture_for(case_for("wmma"))["sustained"]
        self.assertTrue(valid_sustained(value))
        self.assertTrue(valid_sustained({**value, "calls": 4096, "elapsed_ns": 4096, "mean_call_ns": 1}))
        for change in ({"calls": True}, {"calls": 15}, {"calls": 4097}, {"calls": 16.0},
                       {"elapsed_ns": float("nan")}, {"elapsed_ns": 10**1000}, {"mean_call_ns": 0},
                       {"mean_call_ns": True}, {"final_exact_reference_passed": False}, {"scope": ""},
                       {"elapsed_ns": 1600, "mean_call_ns": 100}):
            self.assertFalse(valid_sustained({**value, **change}))

    def test_provenance_is_structured_and_dirty_diagnostics_are_allowed(self):
        case = case_for("queue")
        capture = capture_for(case)
        for change in ({"version": 1}, {"source_dirty": 1}, {"source_commit": "unknown"},
                       {"source_digest": "x"}, {"external_dependencies": []},
                       {"dependency_digest": "bad"}, {"configuration": ""}):
            changed = copy.deepcopy(capture)
            changed["build_manifest"].update(change)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "provenance"):
                validate_capture(changed, case, "fixture-run", 3, 1)
        capture["build_manifest"].update(source_dirty=True, dependency_digest="unavailable")
        validate_capture(capture, case, "fixture-run", 3, 1)

    def test_garner_on_and_off_are_valid_diagnostics(self):
        case = case_for("queue")
        for enabled in ("ON", "OFF", "TRUE", "1"):
            capture = capture_for(case)
            capture["build_manifest"]["configuration"] += "\ngarner_table=" + enabled
            validate_capture(capture, case, "fixture-run", 3, 1)

    def test_strict_json_rejects_duplicates_and_nonfinite_numbers(self):
        for text in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":-Infinity}'):
            with self.assertRaises(ValueError):
                strict_json(text)


class EncodedComparisonTest(unittest.TestCase):
    def observation(self,candidate,reuse,independent,value,**changes):
        return {"case":case_for(candidate,reuse_count=reuse),"independent_repeat":independent,
                "run_id":f"{candidate}-{reuse}-{independent}","sustained_batch_ns":value,
                "prepare_ns":1000 if candidate=="encoded_b_warm" else None,**changes}

    def test_even_median_no_winner_and_missing_or_unmatched_controls(self):
        values=[self.observation(candidate,8,i,100+i*20) for candidate in REUSE for i in (0,1)]
        row,=summarize_encoded_reuse(values)
        point,=row["points"]
        self.assertEqual(point["candidates"]["wmma"]["sustained_batch_ns_median"],110)
        self.assertEqual(point["candidates"]["wmma"]["per_gemm_ns_from_batch_median"],110/8)
        self.assertFalse(point["cold_faster_than_direct_median"])
        self.assertIsNone(row["first_observed_cold_winning_reuse_count"])
        missing,=summarize_encoded_reuse([value for value in values if value["case"]["candidate"]!="wmma"])
        self.assertEqual(missing["points"][0]["status"],"incomplete_cohort")
        self.assertEqual(missing["points"][0]["missing_candidates"],["wmma"])
        unmatched,=summarize_encoded_reuse(values[:-1])
        self.assertEqual(unmatched["points"][0]["status"],"unmatched_independent_repeats")
        with self.assertRaisesRegex(ValueError,"duplicate independent"):
            summarize_encoded_reuse(values+[values[0]])
        for bad in (None,False,0,-1,float("nan"),float("inf")):
            malformed=copy.deepcopy(values);malformed[0]["sustained_batch_ns"]=bad
            with self.assertRaises(ValueError):summarize_encoded_reuse(malformed)

    def test_shapes_groups_signs_and_reuse_do_not_silently_merge(self):
        values=[]
        for m,group,unsigned in ((1,1,False),(2,1,False),(1,2,False),(1,1,True)):
            for candidate in REUSE:
                value=self.observation(candidate,32,0,100)
                value["case"]=case_for(candidate,m=m,group=group,unsigned_inputs=unsigned,reuse_count=32)
                values.append(value)
        self.assertEqual(len(summarize_encoded_reuse(values)),4)


class CollectorTest(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parents[1] / "temp"
        scratch.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.exe = self.root / "mock.exe"
        self.exe.write_bytes(b"synthetic mocked child; never executed")
        self.cases = self.root / "cases.json"
        self.output = self.root / "records"

    def collect(self, cases, mutate=None, runs=1, child_action=None):
        self.cases.write_text(json.dumps(cases), encoding="utf-8")
        argv = ["collector", "--exe", str(self.exe), "--cases", str(self.cases), "--output", str(self.output),
                "--runs", str(runs), "--repeats", "3", "--warmups", "1"]
        calls = []
        def child(command, **kwargs):
            calls.append(command)
            case = validate_case(dict(candidate=command[1], m=int(command[2]), n=int(command[3]),
                                      k=int(command[4]), prefix=int(command[5]), group=int(command[6]),
                                      unsigned_inputs=bool(int(command[7])),
                                      **({"workers": int(command[11])} if command[1] in (*QUEUED, "adaptive_compact") else {}),
                                      **({"fragments":int(command[12]),"merge_policy":command[13]} if command[1]=="split_k" else {}),
                                      **({"bound_method": command[-1]} if command[1] in ADAPTIVE else {}),
                                      **({"reuse_count": int(command[11])} if command[1] in REUSE else {}),
                                      **({"coefficient_profile": command[11]} if command[1].startswith("epilogue_") else {}),
                                      **({"k_depth":int(command[11]),"lds_buffers":int(command[12]),"waves":int(command[13]),
                                          "conversion":command[14],"reuse_count":int(command[15]),"input_change":command[16]} if command[1] in CONVERSION else {}),
                                      **({"k_depth": int(command[11]), "lds_buffers": int(command[12])} if command[1] in TILED else {})))
            capture = capture_for(case, command[10])
            if mutate:
                mutate(capture, len(calls))
            if child_action:
                return child_action(command, capture)
            return subprocess.CompletedProcess(command, 0, json.dumps(capture), "mocked stderr")
        with patch.object(sys, "argv", argv), patch("collect_native_calibration.subprocess.run", side_effect=child), contextlib.redirect_stdout(io.StringIO()):
            result = main()
        return result, calls

    def records(self):
        return [json.loads(p.read_text(encoding="utf-8")) for p in self.output.glob("*.json")
                if not p.name.startswith(("encoded-b-reuse-comparison-", "conversion-comparison-"))]

    def test_all_modes_end_to_end_and_distinct_worker_identity(self):
        cases = [case_for(name) for name in CANDIDATES]
        cases += [case_for("queue", workers=1), case_for("queue", workers=1024),
                  case_for("adaptive_compact", workers=65535),
                  case_for("split_k", m=1, n=1, k=2, group=4, prefix=1, workers=1024)]
        result, calls = self.collect(cases, runs=2)
        self.assertEqual(result, 0)
        records = self.records()
        self.assertEqual(len(records), 2 * len(cases))
        self.assertEqual(len({r["run_id"] for r in records}), len(records))
        self.assertEqual(len({r["campaign_id"] for r in records}), 1)
        self.assertEqual({r["independent_repeat"] for r in records}, {0, 1})
        for record in records:
            self.assertEqual(record["schema"], "rns8_native_collection_v2")
            self.assertTrue(record["validated"])
            self.assertTrue(record["provenance_complete"])
            self.assertFalse(record["promotion_eligible"])
            import hashlib
            for name, digest in record["collector_sources_sha256"].items():
                self.assertEqual(digest, hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest())
            self.assertEqual(record["collector_sha256"], record["collector_sources_sha256"]["collect_native_calibration.py"])
            self.assertEqual(record["command"][10], record["run_id"])
            self.assertIn(record["command"], calls)
            self.assertEqual(record["comparison_identity"]["case"], record["case"])
            self.assertEqual(record["comparison_identity"]["artifact_sha256"], record["artifact_sha256"])
        queue_identities = [r["comparison_identity"] for r in records if r["case"]["candidate"] == "queue"]
        self.assertEqual({r["requested_workers"] for r in queue_identities}, {1, 32, 1024})
        split = next(r for r in records if r["case"].get("workers") == 1024 and r["case"]["candidate"] == "split_k")
        self.assertEqual(split["comparison_identity"]["effective_workers"], 2)
        adaptive = [r for r in records if r["case"]["candidate"] in ADAPTIVE]
        for record in adaptive:
            self.assertEqual(record["command"][-1], "axis_l1")
            self.assertIsNone(record["comparison_identity"]["adaptive_actual_compact_groups"])
        compact = next(r for r in adaptive if r["case"].get("workers") == 65535)
        self.assertEqual(compact["comparison_identity"]["effective_workers"], 65535)
        for record in (r for r in records if r["case"]["candidate"] in TILED):
            self.assertEqual(record["command"][-2:], [str(record["case"][key]) for key in ("k_depth", "lds_buffers")])
            for key, value in tiled_contract(record["case"]).items():
                self.assertEqual(record["comparison_identity"][key], value)

    def test_all_tiled_storage_variants_have_distinct_comparison_identity(self):
        cases = [case_for("tiled_register"), *(case_for("tiled_lds", k_depth=depth, lds_buffers=buffers)
                  for depth in (16, 32) for buffers in (1, 2))]
        self.collect(cases)
        selections = {row["comparison_identity"]["tiled_kernel_selection"] for row in self.records()}
        self.assertEqual(len(selections), 5)

    def test_encoded_batches_have_distinct_identity_and_complete_observed_comparison(self):
        cases=[case_for(candidate,reuse_count=reuse) for candidate in REUSE for reuse in (1,8)]
        def mutate(capture,count):
            reuse=capture["reuse_count"]
            mean={"wmma":20000000,"encoded_b_cold":30000000 if reuse==1 else 19000000,
                  "encoded_b_warm":18000000}[capture["candidate"]]
            capture["sustained"].update(elapsed_ns=16*mean,mean_call_ns=mean,mean_gemm_ns=mean/reuse)
        self.collect(cases,mutate=mutate,runs=2)
        records=self.records()
        self.assertEqual(len(records),12)
        for record in records:
            self.assertEqual(record["command"][-1],str(record["case"]["reuse_count"]))
            for key,value in reuse_contract(record["case"]).items():
                self.assertEqual(record["comparison_identity"][key],value)
        summary,=self.output.glob("encoded-b-reuse-comparison-*.json")
        result=json.loads(summary.read_text(encoding="utf-8"))
        self.assertFalse(result["promotion_eligible"])
        self.assertTrue(result["provenance_complete"])
        row,=result["groups"]
        self.assertEqual(row["first_observed_cold_winning_reuse_count"],8)
        self.assertEqual([point["status"] for point in row["points"]],["matched_independent_cohort"]*2)
        self.assertEqual([point["cold_faster_than_direct_median"] for point in row["points"]],[False,True])
        self.assertEqual(result["comparison_identity"]["artifact_sha256"],records[0]["artifact_sha256"])

    def test_encoded_setup_and_batch_mismatches_retain_failed_captures(self):
        for candidate,key,value in (("encoded_b_warm","encoded_b_prepare_ns",None),
                                    ("encoded_b_cold","encoded_b_prepare_ns",100),
                                    ("encoded_b_warm","encoded_b_cache_logical_bytes",1)):
            self.output=self.root/(candidate+key)
            def mutate(capture,count):capture[key]=value
            with self.assertRaisesRegex(RuntimeError,"contract/reference mismatch"):
                self.collect([case_for(candidate)],mutate=mutate)
            record,=self.records()
            self.assertFalse(record["validated"])
            self.assertFalse(list(self.output.glob("encoded-b-reuse-comparison-*.json")))

    def test_omitted_workers_get_explicit_command_and_identity(self):
        result, calls = self.collect([{**BASE, "candidate": "queue"}])
        self.assertEqual(result, 0)
        self.assertEqual(calls[0][11], "32")
        self.assertEqual(self.records()[0]["case"]["workers"], 32)

    def test_worker_and_all_contract_mismatches_retain_raw_capture(self):
        for candidate in CANDIDATES:
            case = case_for(candidate)
            for key in capture_contract(case, "fixture-run", 3, 1):
                with self.subTest(candidate=candidate, key=key):
                    self.output = self.root / f"mismatch-{candidate}-{key}"
                    def mutate(capture, count):
                        capture[key] = None if capture[key] is not None else "wrong"
                    with self.assertRaisesRegex(RuntimeError, "contract/reference mismatch"):
                        self.collect([case], mutate=mutate)
                    record, = self.records()
                    self.assertFalse(record["validated"])
                    self.assertIn(key, record["validation_error"])
                    self.assertIn("stdout", record)
                    self.assertEqual(record["stderr"], "mocked stderr")

    def test_changed_build_device_or_runtime_is_retained_and_rejected(self):
        for key in ("device", "target", "runtime_version", "build_manifest"):
            self.output = self.root / ("changed-" + key)
            def mutate(capture, count):
                if count == 2:
                    if key == "build_manifest":
                        capture[key]["source_commit"] = "d" * 40
                    elif key == "runtime_version":
                        capture[key] += 1
                    else:
                        capture[key] += " changed"
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, "identity changed"):
                self.collect([BASE], mutate=mutate, runs=2)
            self.assertEqual(len(self.records()), 2)
            self.assertEqual(sum(r["validated"] for r in self.records()), 1)

    def test_dirty_or_incomplete_provenance_stays_diagnostic(self):
        def mutate(capture, count):
            capture["build_manifest"].update(source_dirty=True, dependency_digest="unavailable")
        self.collect([BASE], mutate=mutate)
        record, = self.records()
        self.assertTrue(record["validated"])
        self.assertFalse(record["provenance_complete"])
        self.assertFalse(record["promotion_eligible"])

    def test_garner_diagnostics_preserve_release_ineligibility(self):
        for enabled, complete in (("ON", False), ("OFF", True)):
            self.output = self.root / ("garner-" + enabled)
            def mutate(capture, count):
                capture["build_manifest"]["configuration"] += "\ngarner_table=" + enabled
            self.collect([BASE], mutate=mutate)
            record, = self.records()
            self.assertTrue(record["validated"])
            self.assertEqual(record["provenance_complete"], complete)
            self.assertFalse(record["promotion_eligible"])
            self.assertIn("garner_table=" + enabled, record["comparison_identity"]["build_manifest"]["configuration"])

    def test_timeout_partial_output_failure_and_invalid_json(self):
        def timeout(command, capture):
            raise subprocess.TimeoutExpired(command, 1, output=b"partial stdout", stderr=b"partial stderr")
        actions = [timeout, lambda c, j: subprocess.CompletedProcess(c, 3, "failure output", "failure stderr"),
                   lambda c, j: subprocess.CompletedProcess(c, 0, "not json", ""),
                   lambda c, j: subprocess.CompletedProcess(c, 0, "[]", ""),
                   lambda c, j: subprocess.CompletedProcess(c, 0, '{"x":1,"x":2}', ""),
                   lambda c, j: subprocess.CompletedProcess(c, 0, '{"x":NaN}', "")]
        for index, action in enumerate(actions):
            self.output = self.root / f"failure-{index}"
            with self.subTest(index=index), self.assertRaises(RuntimeError):
                self.collect([BASE], child_action=action)
            record, = self.records()
            self.assertFalse(record["validated"])
            self.assertNotIn("comparison_identity", record)
            if index == 0:
                self.assertEqual(record["reason"], "timeout")
                self.assertEqual(record["stdout"], "partial stdout")
                self.assertEqual(record["stderr"], "partial stderr")

    def test_executable_changes_after_child_are_rejected(self):
        def action(command, capture):
            self.exe.write_bytes(b"changed binary")
            return subprocess.CompletedProcess(command, 0, json.dumps(capture), "")
        with self.assertRaisesRegex(RuntimeError, "executable changed"):
            self.collect([BASE], child_action=action)
        self.assertFalse(self.records()[0]["validated"])

    def test_invalid_cases_fail_before_launch_or_output_creation(self):
        for index, cases in enumerate((None, {}, [], [BASE] * 65, [BASE, BASE],
                                              [{**BASE, "candidate": "queue"}, {**BASE, "candidate": "queue", "workers": 32}],
                                              [{**BASE, "workers": 32}], [{**BASE, "m": True}])):
            self.output = self.root / f"bad-input-{index}"
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.collect(cases)
            self.assertFalse(self.output.exists())


@unittest.skipUnless(os.environ.get("RNS8_NATIVE_CALIBRATION_EXE"), "set RNS8_NATIVE_CALIBRATION_EXE for compiled CLI rejection/help checks")
class CompiledCliTest(unittest.TestCase):
    def invoke(self, arguments):
        return subprocess.run([os.environ["RNS8_NATIVE_CALIBRATION_EXE"], *arguments],
                              capture_output=True, text=True, check=False, timeout=30)

    def test_help_lists_all_modes_and_worker_contract(self):
        result = self.invoke(["--help"])
        self.assertEqual(result.returncode, 0, result.stderr)
        for name in CANDIDATES:
            self.assertIn(name, result.stdout)
        self.assertIn("1..1024", result.stdout)
        self.assertIn("default 32", result.stdout)

    def test_malformed_and_irrelevant_workers_fail_before_device_work(self):
        for candidate in (name for name in CANDIDATES if name not in (*ADAPTIVE, *TILED, *REUSE, *CONVERSION)):
            case = case_for(candidate)
            base = [candidate, *(str(case[k]) for k in ("m", "n", "k", "prefix", "group")), "0", "1", "1", "cli-reject"]
            workers = ("0", "1025", "-1", "1.0", "+1", "", "4294967296", "32x") if candidate in QUEUED else ("32",)
            for value in workers:
                with self.subTest(candidate=candidate, workers=value):
                    result = self.invoke([*base, value])
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("numeric argument" if candidate in QUEUED else "workers applies only", result.stderr)
        result = self.invoke([])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("usage:", result.stderr)

    def test_encoded_invalid_reuse_never_runs_device_work(self):
        for candidate in REUSE:
            case=case_for(candidate)
            base=[candidate,*(str(case[k]) for k in ("m","n","k","prefix","group")),
                  "0","1","1","cli-reject"]
            invalid=[[value] for value in ("0","257","-1","1.0","+1","", "4294967296","32x")]
            invalid += [["1","1"]]
            if candidate in ENCODED:invalid.append([])
            for suffix in invalid:
                with self.subTest(candidate=candidate,suffix=suffix):
                    result=self.invoke([*base,*suffix])
                    self.assertNotEqual(result.returncode,0)

    def test_adaptive_invalid_controls_fail_before_device_work(self):
        for candidate in ADAPTIVE:
            case = case_for(candidate)
            base = [candidate, *(str(case[k]) for k in ("m", "n", "k", "prefix", "group")), "0", "1", "1", "cli-reject"]
            invalid = [[], ["tile_max"], ["32"]] if candidate == "adaptive_grid" else [[], ["32"], ["axis_l1"],
                ["0", "axis_l1"], ["65536", "axis_l1"], ["1.0", "axis_l1"], ["32", "panel_l1"]]
            for suffix in invalid:
                with self.subTest(candidate=candidate, suffix=suffix):
                    result = self.invoke([*base, *suffix])
                    self.assertNotEqual(result.returncode, 0)

    def test_tiled_invalid_controls_fail_before_device_work(self):
        for candidate in TILED:
            case = case_for(candidate)
            base = [candidate, *(str(case[key]) for key in ("m", "n", "k", "prefix", "group")),
                    "0", "1", "1", "cli-reject"]
            invalid = [[], ["16"], ["17", "1"], ["64", "2"], ["16", "3"], ["16", "-1"],
                       ["16.0", "1"], ["16", "true"], ["16", "+1"]]
            invalid += [["32", "0"], ["16", "1"]] if candidate == "tiled_register" else [["16", "0"]]
            for suffix in invalid:
                with self.subTest(candidate=candidate, suffix=suffix):
                    result = self.invoke([*base, *suffix])
                    self.assertNotEqual(result.returncode, 0)



class NativeSplitMergeControls(unittest.TestCase):
    def test_fragment_boundaries_and_capture_accounting(self):
        for fragments in (1,2,3,127,128,129,257,4096):
            for policy in ('fused-first','separate-control'):
                case=case_for('split_k',m=2,n=3,k=4096,fragments=fragments,merge_policy=policy)
                capture=capture_for(case);validate_capture(capture,case,'fixture-run',3,1)
                self.assertEqual(capture['splits'],fragments)
                self.assertEqual(capture['standalone_canonicalization_logical_bytes'],
                                 2*fragments*case['prefix']*6 if policy=='separate-control' else 0)
                for key in ('native_partial_format','merge_output_format','merge_maximum_integer_sum','standalone_canonicalization_logical_bytes'):
                    bad=copy.deepcopy(capture);bad[key]=None
                    with self.assertRaises(ValueError):validate_capture(bad,case,'fixture-run',3,1)
        for value in (0,258,True,3.0):
            with self.assertRaises(ValueError):case_for('split_k',fragments=value)
        with self.assertRaises(ValueError):case_for('wmma',fragments=3)
        with self.assertRaises(ValueError):case_for('split_k',merge_policy='reinterpret-unsigned')

if __name__ == "__main__":
    unittest.main()
