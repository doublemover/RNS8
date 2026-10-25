"""Collect independent native conversion/WMMA pipeline processes, without promotion."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import time
import uuid
from build_manifest import manifest_provenance_complete
from modulus_set_search import DEFAULT_MODULI
from native_dataset import PROFILES as DATASET_PROFILES, identity as dataset_identity, model as dataset_model


CONVERSION = ("conversion_cold", "conversion_prepared", "conversion_pack_gemm")
CANDIDATES = (*CONVERSION, "pack_pair", "packed_gemm", "pack_gemm", "wmma", "grid_prepared", "queue", "queue_prepared", "split_k",
              "adaptive_grid", "adaptive_compact",
              "tiled_register", "tiled_lds",
              "encoded_b_cold", "encoded_b_warm",
              "complex_3m", "complex_4m",
              "epilogue_direct", "epilogue_rocwmma", "epilogue_unfused",
              "narrow_wmma", "narrow_scalar", "narrow_vector", "narrow_rns")
QUEUED = ("queue", "queue_prepared", "split_k")
ADAPTIVE = ("adaptive_grid", "adaptive_compact")
TILED = ("tiled_register", "tiled_lds")
ENCODED = ("encoded_b_cold", "encoded_b_warm")
REUSE = ("wmma", *ENCODED)


def strict_json(text):
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError(f"non-finite JSON number: {value}")
    return json.loads(text, object_pairs_hook=object_pairs, parse_constant=invalid_constant)


def validate_case(case):
    """Canonical controls: queues default32 workers; ordinary WMMA defaults1 reuse."""
    keys = {"candidate", "m", "n", "k", "prefix", "group", "unsigned_inputs"}
    if not isinstance(case, dict) or not keys <= set(case) or set(case) - keys - {"workers", "bound_method", "k_depth", "lds_buffers", "reuse_count", "waves", "conversion", "input_change", "coefficient_profile", "dataset_profile", "dataset_seed", "fragments", "merge_policy"}:
        raise ValueError("native case needs candidate,m,n,k,prefix,group,unsigned_inputs; workers for queues/compact; adaptive requires bound_method; tiled requires k_depth/lds_buffers; encoded B requires reuse_count")
    limits = {"m": (1, 1024), "n": (1, 1024), "k": (1, 65537), "prefix": (1, 20)}
    if any(type(case[key]) is not int or not low <= case[key] <= high for key, (low, high) in limits.items()):
        raise ValueError("invalid native calibration shape/prefix")
    native = case["candidate"] in (*REUSE, *CONVERSION, "grid_prepared", *QUEUED, *ADAPTIVE, *TILED)
    narrow = case["candidate"] in ("narrow_wmma", "narrow_scalar", "narrow_vector", "narrow_rns")
    if (case["candidate"] not in CANDIDATES
            or type(case["group"]) is not int or case["group"] not in (1, 2, 4)
            or (not native and case["group"] != 1) or type(case["unsigned_inputs"]) is not bool
            or case["m"] * case["n"] * case["k"] > 1073741824):
        raise ValueError("invalid native calibration candidate/group/reference budget")
    if narrow and (case["prefix"] != 1 or case["k"] > 32768
                   or (case["candidate"] == "narrow_scalar" and case["unsigned_inputs"])):
        raise ValueError("narrow cases require prefix=1, K<=32768; scalar is signed only")
    if case["candidate"].startswith("complex_") and (case["prefix"] != 20 or case["unsigned_inputs"]):
        raise ValueError("complex cases require signed inputs, prefix20 and group1")
    if case["candidate"] in ADAPTIVE:
        if case["prefix"] != 20 or case.get("bound_method") != "axis_l1":
            raise ValueError("adaptive cases require prefix20 and explicit bound_method=axis_l1")
    elif "bound_method" in case:
        raise ValueError("bound_method applies only to adaptive cases")
    if case["candidate"] in CONVERSION:
        depth,buffers,waves,mode=(case.get(key) for key in ("k_depth","lds_buffers","waves","conversion"))
        if (type(depth) is not int or depth not in (16,32) or type(buffers) is not int or buffers not in (0,1,2)
                or type(waves) is not int or waves not in (1,4) or mode not in ("replicated","cooperative","encoded_lds")
                or (not buffers and depth!=16) or (waves==1 and (buffers or depth!=16))
                or (mode=="cooperative" and buffers) or (mode=="encoded_lds" and (buffers!=1 or waves!=4))
                or case.get("input_change") not in ("stable","reverse_k")
                or type(case.get("reuse_count")) is not int or not 1<=case["reuse_count"]<=256
                or (case["candidate"]=="conversion_pack_gemm" and (depth,buffers,waves,mode)!=(16,0,4,"replicated"))):
            raise ValueError("invalid explicit conversion ownership/storage/reuse/input control")
    elif any(key in case for key in ("waves","conversion","input_change")):
        raise ValueError("conversion ownership controls apply only to conversion candidates")
    if case["candidate"] in TILED:
        depth, buffers = case.get("k_depth"), case.get("lds_buffers")
        if (type(depth) is not int or type(buffers) is not int
                or (case["candidate"] == "tiled_register" and (depth != 16 or buffers != 0))
                or (case["candidate"] == "tiled_lds" and (depth not in (16, 32) or buffers not in (1, 2)))):
            raise ValueError("tiled_register requires integer k_depth16/lds_buffers0; tiled_lds depth16|32/buffers1|2")
    elif case["candidate"] not in CONVERSION and ("k_depth" in case or "lds_buffers" in case):
        raise ValueError("k_depth/lds_buffers apply only to tiled cases")
    normalized = dict(case)
    if 'dataset_profile' in case or 'dataset_seed' in case:
        if (case['candidate'] not in (*ADAPTIVE,'wmma','pack_gemm') or case['prefix']!=20
                or case.get('dataset_profile') not in DATASET_PROFILES
                or type(case.get('dataset_seed')) is not int or not 0<=case['dataset_seed']<=0xffffffff):
            raise ValueError('dataset controls require paired adaptive/wmma/pack_gemm capacity20, versioned profile and uint32 seed')
        # Reject tiny shapes that cannot contain the advertised spatial bands.
        if case['dataset_profile'] in ('mixed-tiles-v2','zero-tiles-v2','localized-outlier-v2') and min(case['m'],case['n'])<=16:
            raise ValueError('heterogeneous datasets require at least two row and column tiles')
    if case["candidate"].startswith("epilogue_"):
        profile=case.get("coefficient_profile","extreme")
        if profile not in ("extreme","zero","copy","unit","general","channel-zero"):raise ValueError("unknown coefficient profile")
        normalized["coefficient_profile"]=profile
    elif "coefficient_profile" in case:raise ValueError("coefficient profiles apply only to epilogues")
    if case["candidate"] in REUSE:
        reuse = case.get("reuse_count", 1 if case["candidate"] == "wmma" else None)
        if type(reuse) is not int or not 1 <= reuse <= 256:
            raise ValueError("encoded B requires explicit integer reuse_count1..256; wmma defaults1")
        normalized["reuse_count"] = reuse
    elif "reuse_count" in case and case["candidate"] not in CONVERSION:
        raise ValueError("reuse_count applies only to wmma/encoded_b_cold/encoded_b_warm")
    if case["candidate"] == "adaptive_compact":
        workers = case.get("workers")
        if type(workers) is not int or not 1 <= workers <= 65535:
            raise ValueError("adaptive_compact requires explicit integer workers1..65535")
    elif case["candidate"] in QUEUED:
        workers = case.get("workers", 32)
        if type(workers) is not int or not 1 <= workers <= 1024:
            raise ValueError("queue/queue_prepared/split_k workers must be an integer in 1..1024")
        normalized["workers"] = workers
    elif "workers" in case:
        raise ValueError("workers applies only to queue/queue_prepared/split_k/adaptive_compact candidates")
    if case['candidate']=='split_k':
        fragments=case.get('fragments',min(case['k'],3));policy=case.get('merge_policy','fused-first')
        if type(fragments) is not int or not 1<=fragments<=min(case['k'],4096) or policy not in ('fused-first','separate-control'):
            raise ValueError('split_k requires fragments1..min(K,4096) and fused-first|separate-control')
        normalized.update(fragments=fragments,merge_policy=policy)
    elif 'fragments' in case or 'merge_policy' in case:raise ValueError('fragment controls apply only to native split_k')
    if 'dataset_profile' in normalized and normalized.get('reuse_count',1)!=1:
        raise ValueError('dataset comparisons require one complete product per call')
    return normalized


def scheduling_contract(case):
    queued = case["candidate"] in QUEUED
    splits = case.get("fragments",min(case["k"],3)) if case["candidate"] == "split_k" else 1
    descriptors = (((case["m"] + 15) // 16) * ((case["n"] + 15) // 16)
                   * ((case["prefix"] + case["group"] - 1) // case["group"]) * splits) if queued else 0
    compact = case["candidate"] == "adaptive_compact"
    requested = case["workers"] if compact else case.get("workers", 32) if queued else 0
    return {"splits": splits, "requested_workers": requested,
            "effective_workers": requested if compact else min(requested, descriptors), "queue_descriptors": descriptors,
            "queue_kernel_selection": (f"homogeneous_group_{case['group']}_"
                                       + ("unsigned" if case["unsigned_inputs"] else "signed"))
                                      if queued else "not_applicable"}


def adaptive_contract(case):
    if case["candidate"] not in ADAPTIVE:
        return {}
    compact = case["candidate"] == "adaptive_compact"
    return {"bound_method": "axis_l1", "adaptive_launch": "compact_workers" if compact else "dense_grid",
            "adaptive_dataset": dataset_identity(case.get("dataset_profile","full-width-v1"),case.get("dataset_seed",3135)),
            "reference_method": "boost_int256_full_product_with_cpp_int_row_edge_audits_and_epilogue",
            "adaptive_dense_group_upper_bound": ((case["m"] + 15) // 16) * ((case["n"] + 15) // 16)
                * ((case["prefix"] + case["group"] - 1) // case["group"]),
            "adaptive_actual_compact_groups": None, "adaptive_compact_count_source": "device_offsets_not_read_back",
            "adaptive_worker_policy": "exact_requested_worker_launch_no_count_clamp" if compact else "dense_grid_control",
            "adaptive_kernel_selection": ("compact_group_" if compact else "grid_group_") + str(case["group"])
                + ("_unsigned_axis_l1" if case["unsigned_inputs"] else "_signed_axis_l1"),
            "adaptive_output_scope": "fresh bounds and per-tile prefix; exact suffix completes every capacity20 output plane"}


def tiled_contract(case):
    if case["candidate"] not in TILED:
        return {}
    depth, buffers = case["k_depth"], case["lds_buffers"]
    return {"tiled_output_rows": 32, "tiled_output_cols": 32, "tiled_waves": 4,
            "tiled_raw_lds_bytes": buffers * (32 * (depth + 1) + depth * 33) * 8,
            "tiled_lds_budget_bytes": 65536, "tiled_resource_query_is_device_qualification": False,
            "tiled_dataset": "full_width_native64_mt19937_64_seed3135_forced_extrema_v1",
            "reference_method": "boost_int256_full_product_with_cpp_int_row_edge_audits_and_epilogue",
            "tiled_storage": "raw_native_padded_lds_encoded_fragments_in_registers" if buffers else "direct_native_register_loads",
            "tiled_prefetch": ("next_disjoint_buffer_loaded_before_current_arithmetic" if buffers == 2 else
                               "single_buffer_load_barrier_reuse" if buffers else "direct_native_register_loads"),
            "tiled_kernel_selection": (f"native32x32_w4_g{case['group']}_"
                                       + ("unsigned" if case["unsigned_inputs"] else "signed")
                                       + f"_d{depth}_b{buffers}"),
            "tiled_control_scope": "four_wave_32x32_same_output_residency_reference_and_slab_cap"}


def conversion_contract(case):
    if case["candidate"] not in CONVERSION: return {}
    depth,buffers,waves,mode=(case[key] for key in ("k_depth","lds_buffers","waves","conversion"))
    packed=case["candidate"]=="conversion_pack_gemm"
    raw=0 if mode=="encoded_lds" or packed else buffers*(32*(depth+1)+depth*33)*8
    encoded=case["group"]*2*32*(depth//4+1)*4 if mode=="encoded_lds" and not packed else 0
    distinct=2*(16 if waves==1 else 32)*depth
    conversions=distinct*case["group"] if mode=="encoded_lds" else waves*2*16*depth*case["group"]*(1 if mode=="cooperative" else 2)
    identity={"replicated":"rdna3_replicated_native_conversion_v1",
              "cooperative":"rdna3_wave32_own_ab_half_xor16_packed_v1",
              "encoded_lds":"rdna3_cta32x32_encode_once_ab_axis_kword_pad1_v1"}[mode]
    return {"conversion_identity":identity,"batch_gemm_calls":case["reuse_count"],
        "conversion_preparation_in_batch":case["candidate"]=="conversion_cold",
        "conversion_prepared_capability_bytes":0 if packed else 4,"conversion_public_ownership_measured":False,
        "conversion_raw_lds_bytes":raw,"conversion_encoded_lds_bytes":encoded,"conversion_total_lds_bytes":raw+encoded,
        "conversion_output_tile_rows":0 if packed else (16 if waves==1 else 32),
        "conversion_output_tile_cols":0 if packed else (16 if waves==1 else 32),
        "conversion_grid_workgroups":0 if packed else ((case["m"]+(16 if waves==1 else 32)-1)//(16 if waves==1 else 32))*
            ((case["n"]+(16 if waves==1 else 32)-1)//(16 if waves==1 else 32))*((case["prefix"]+case["group"]-1)//case["group"]),
        "conversion_source_counts_measured":False,"conversion_scalars_per_interior_panel":0 if packed else conversions,
        "conversion_raw_scalars_per_interior_panel":0 if packed else distinct,
        "conversion_accumulator_i32_per_lane":0 if packed else 8*case["group"],
        "conversion_fragment_words_per_lane":0 if packed else 8*case["group"],
        "conversion_lds_word_layout":"A_row_Kword_B_column_Kword_odd_padded_stride" if mode=="encoded_lds" else "not_applicable",
        "conversion_dataset":"full_width_native64_mt19937_64_seed3135_forced_extrema_v1",
        "conversion_input_upload_scope":"alternate_common_K_reversal_before_each_sample_and_once_before_sustained_burst_outside_interval"
            if case["input_change"]=="reverse_k" else "initial_upload_only"}


def epilogue_contract(case):
    if not case["candidate"].startswith("epilogue_"):
        return {}
    size = case["m"] * case["n"] * case["prefix"]
    profile=case.get("coefficient_profile","extreme")
    alpha,beta={"extreme":((1<<63)-1 if case["unsigned_inputs"] else -(1<<63),(1<<63)-1),
        "zero":(0,0),"copy":(0,1),"unit":(1,1),"general":(3 if case["unsigned_inputs"] else -3,5),
        "channel-zero":(255,256)}[profile]
    return {"coefficient_profile":profile,"input_domain": "resident_centered_residues",
            "output_contract": "integer_alpha_ab_plus_beta_d_residues_no_integer_export",
            "expression": {"alpha": alpha,
                           "beta": beta, "coefficients": "original_integers",
                           "d_dataset": "seed_3135_after_a_b"},
            "reserved_intermediate_bytes": size,
            "materialized_intermediate_bytes": size if case["candidate"] == "epilogue_unfused" else 0}


def reuse_contract(case):
    if case["candidate"] not in REUSE:
        return {}
    encoded, warm = case["candidate"] in ENCODED, case["candidate"] == "encoded_b_warm"
    words = ((case["k"]+15)//16)*4 if encoded else 0
    size = case["prefix"]*case["n"]*words*4+4 if encoded else 0
    return {"reuse_count":case["reuse_count"], "batch_gemm_calls":case["reuse_count"],
            "reuse_batch_boundary":"R_synchronous_calls_same_native_A_B_one_final_verified_output",
            "reuse_comparison_boundary":"private_raw_residue_no_public_matrix_proof_or_generation",
            "reuse_dataset":dataset_identity(case.get("dataset_profile","full-width-v1"),case.get("dataset_seed",3135)),
            "reference_method":"boost_int256_full_product_with_cpp_int_row_edge_audits_and_epilogue",
            "encoded_b_layout":"modulus_n_k_packed_word32_v1" if encoded else "none",
            "encoded_b_k_words":words,"encoded_b_cache_logical_bytes":size,
            "encoded_b_preparation_in_batch":encoded and not warm,
            "encoded_b_preparation_scope": "one_preparation_outside_batches_cached_consume" if warm else
                "per_batch_allocate_probe_encode_consume_release" if encoded else "none_direct_native_grouped_each_call",
            "encoded_b_resident_cache_bytes":size if warm else 0,
            "encoded_b_transient_cache_bytes":size if encoded and not warm else 0,
            "encoded_b_same_B_immutable":True,"encoded_b_public_ownership_measured":False,
            "encoded_b_kernel_selection":("encoded_b_grouped_g" if encoded else "native_grouped_g")
                +str(case["group"])+("_unsigned" if case["unsigned_inputs"] else "_signed")}


def positive_number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value) and value > 0
    except OverflowError:
        return False


def valid_sustained(value):
    if not isinstance(value, dict) or value.get("final_exact_reference_passed") is not True:
        return False
    calls, elapsed, mean = (value.get(k) for k in ("calls", "elapsed_ns", "mean_call_ns"))
    return (type(calls) is int and 16 <= calls <= 4096
            and all(positive_number(x) for x in (elapsed, mean))
            and (elapsed >= 250000000 or calls == 4096)
            and math.isclose(mean * calls, elapsed, rel_tol=1e-12)
            and type(value.get("scope")) is str and bool(value["scope"]))


def narrow_contract(case):
    if not case["candidate"].startswith("narrow_"):
        return {}
    bound = case["k"] * (255 ** 2 if case["unsigned_inputs"] else 128 ** 2)
    prefix, product = 0, 1
    if case["candidate"] == "narrow_rns":
        while product <= bound * (1 if case["unsigned_inputs"] else 2):
            product *= DEFAULT_MODULI[prefix]
            prefix += 1
    return {"dataset": "byte_domain_mt19937_64_seed3135_forced_maxima_v1",
            "input_domain": "resident_native_64_byte_values",
            "output_contract": "bounded_exact_native_64",
            "abs_partial_sum_bound": bound,
            "selected_rns_prefix": prefix,
            "residue_storage_bytes": (case["m"] * case["k"] + case["k"] * case["n"] + case["m"] * case["n"]) * prefix,
            "native_input_storage_bytes": (case["m"] * case["k"] + case["k"] * case["n"]) * 8,
            "native_output_bytes": case["m"] * case["n"] * 8,
            "reserved_status_bytes": 4,
            "reference": "exact_int64_all_outputs_with_boost_boundary_columns"}


def capture_contract(case, run_id, repeats, warmups):
    expected = {key: value for key, value in case.items() if key != "workers"}
    expected.update(schema="rns8_native_microcase_v2", run_id=run_id, repeats=repeats,
                    warmups=warmups, seed=case.get("dataset_seed",3135), promotion_eligible=False, isa_verified=False,
                    actual_hbm_bytes=None, timing_resolution_ns=None)
    if 'dataset_profile' in case:
        expected['dataset_identity']=dataset_identity(case['dataset_profile'],case['dataset_seed'])
        expected['dataset_model']=dataset_model(case['m'],case['n'],case['k'],case['unsigned_inputs'],case['group'],
            case['dataset_profile'],case['dataset_seed'],case['candidate'] in ADAPTIVE)
    if case['candidate']=='split_k':
        fragments=case['fragments'];cells=case['m']*case['n']*case['prefix'];separate=case['merge_policy']=='separate-control'
        expected.update(native_partial_format='centered_int8',merge_output_format='canonical_uint8',
            native_split_scratch_bytes=cells*(fragments+(fragments+127)//128),
            standalone_canonicalization_logical_bytes=2*cells*fragments if separate else 0,
            standalone_canonicalization_launches=1 if separate else 0,merge_maximum_integer_sum=128*255,
            source_counts_not_measured_HBM=True)
    expected.update(scheduling_contract(case))
    expected.update(adaptive_contract(case))
    expected.update(tiled_contract(case))
    expected.update(narrow_contract(case))
    if case["candidate"].startswith("complex_"):
        m, n, k, prefix = (case[key] for key in ("m", "n", "k", "prefix"))
        three = case["candidate"] == "complex_3m"
        expected.update(dataset="split_complex_full_signed64_mt19937_3135_extrema_v1",
                        input_domain="resident_split_complex_centered_residues",
                        output_contract="paired_exact_signed_residues_no_integer_export",
                        real_products=3 if three else 4,
                        scratch_peak_bytes=((m*k+k*n)+3*m*n)*prefix if three else 4*m*n*prefix,
                        scratch_accounting="explicit_queried_capacity_not_measured_peak_or_HBM",
                        scratch_layout_version=2,reused_internal_stage_bytes=2*m*n*prefix,
                        stage_ownership="p0_p1_after_cell_local_product_loads_public_paired_commit_unchanged",
                        native_input_storage_bytes=2*(m*k+k*n)*8,
                        residue_input_storage_bytes=2*(m*k+k*n)*prefix,
                        residue_output_bytes=2*m*n*prefix, component_bound=str(k*(1 << 127)),
                        reference_method="boost_int256_four_products_all_outputs_with_cpp_int_boundary_columns",
                        input_preparation_scope="native upload, four input packs and common allocations; outside resident repeated calls")
        return expected
    if case["candidate"] in ("queue_prepared", "grid_prepared"):
        expected["queue_preparation_scope"] = "one host preparation including allocation, capability check and descriptor upload; excluded from steady calls"
    if not case["candidate"].startswith("narrow_"):
        epilogue = case["candidate"].startswith("epilogue_")
        expected.update(
            input_domain="resident_centered_residues" if epilogue or case["candidate"] == "packed_gemm" else "resident_native_64",
            output_contract="operand_residue_planes" if case["candidate"] == "pack_pair" else "product_residues_default_basis_no_integer_export",
            expression=None, reserved_intermediate_bytes=0, materialized_intermediate_bytes=0,
            native_input_storage_bytes=(case["m"] * case["k"] + case["k"] * case["n"]) * 8,
            residue_input_storage_bytes=(case["m"] * case["k"] + case["k"] * case["n"]
                                         + (case["m"] * case["n"] if epilogue else 0)) * case["prefix"],
            residue_output_bytes=case["m"] * case["n"] * case["prefix"])
        expected.update(epilogue_contract(case))
    expected.update(reuse_contract(case))
    expected.update(conversion_contract(case))
    if case["candidate"] in CONVERSION and case["candidate"]!="conversion_pack_gemm":
        expected["residue_input_storage_bytes"]=0
    return expected


def same_value(actual, expected):
    """JSON bools and floats cannot impersonate integer controls or bounds."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(same_value(actual[k], v) for k, v in expected.items())
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(same_value(a, b) for a, b in zip(actual, expected))
    return actual == expected


def validate_capture(capture, case, run_id, repeats, warmups):
    if not isinstance(capture, dict):
        raise ValueError("capture must be a JSON object")
    for key, value in capture_contract(case, run_id, repeats, warmups).items():
        if key not in capture or not same_value(capture[key], value):
            raise ValueError(f"metadata mismatch: {key}")
    if not valid_sustained(capture.get("sustained")):
        raise ValueError("invalid sustained verification/timing")
    if case["candidate"] in CONVERSION:
        setup=capture.get("conversion_prepare_ns")
        if "conversion_prepare_ns" not in capture or (not positive_number(setup) if case["candidate"]=="conversion_prepared" else setup is not None):
            raise ValueError("invalid conversion preparation measurement")
        sustained=capture["sustained"];calls=sustained["calls"]*case["reuse_count"]
        if (type(sustained.get("gemm_calls")) is not int or sustained["gemm_calls"]!=calls
                or type(sustained.get("verified_outputs_per_batch")) is not int or sustained["verified_outputs_per_batch"]!=1
                or not positive_number(sustained.get("mean_gemm_ns"))
                or not math.isclose(sustained["mean_gemm_ns"]*calls,sustained["elapsed_ns"],rel_tol=1e-12)):
            raise ValueError("invalid conversion batch/GEMM accounting")
    if case["candidate"] in REUSE:
        warm = case["candidate"] == "encoded_b_warm"
        setup = capture.get("encoded_b_prepare_ns")
        if "encoded_b_prepare_ns" not in capture or (not positive_number(setup) if warm else setup is not None):
            raise ValueError("invalid encoded B preparation timing/scope")
        sustained = capture["sustained"]
        calls = sustained["calls"]*case["reuse_count"]
        if (type(sustained.get("gemm_calls")) is not int or sustained["gemm_calls"] != calls
                or type(sustained.get("verified_outputs_per_batch")) is not int
                or sustained["verified_outputs_per_batch"] != 1
                or not positive_number(sustained.get("mean_gemm_ns"))
                or not math.isclose(sustained["mean_gemm_ns"]*calls,sustained["elapsed_ns"],rel_tol=1e-12)):
            raise ValueError("invalid encoded B batch/GEMM counts or timing")
    if case["candidate"].startswith("complex_") and not positive_number(capture.get("input_prepare_ns")):
        raise ValueError("invalid complex input preparation timing")
    if case["candidate"] in ("queue_prepared", "grid_prepared"):
        if not positive_number(capture.get("queue_prepare_ns")):
            raise ValueError("invalid prepared queue setup timing")
        size = capture.get("queue_descriptor_storage_bytes")
        if type(size) is not int or size <= 0:
            raise ValueError("invalid prepared queue descriptor storage")
    samples = capture.get("samples")
    if not isinstance(samples, list) or len(samples) != repeats:
        raise ValueError("invalid sample count")
    for sample in samples:
        if (not isinstance(sample, dict) or sample.get("exact_reference_passed") is not True
                or sample.get("uninstrumented_exact_reference_passed") is not True
                or not positive_number(sample.get("host_call_ns"))
                or not positive_number(sample.get("uninstrumented_host_call_ns"))
                or not isinstance(sample.get("events"), list)):
            raise ValueError("invalid sample verification/timing")
        for event in sample["events"]:
            if (not isinstance(event, dict) or type(event.get("label")) is not str or not event["label"]
                    or "elapsed_ns" not in event
                    or (event["elapsed_ns"] is not None and not positive_number(event["elapsed_ns"]))):
                raise ValueError("invalid HIP event timing")
    for key in ("device", "target", "timing_scope", "cache_state"):
        if type(capture.get(key)) is not str or not capture[key]:
            raise ValueError(f"missing identity/scope: {key}")
    if type(capture.get("runtime_version")) is not int or capture["runtime_version"] <= 0:
        raise ValueError("invalid runtime identity")
    manifest = capture.get("build_manifest")
    # Diagnostic runs may use dirty sources or unavailable dependency identity.
    # Validate their shape without misrepresenting them as release-qualified.
    if not isinstance(manifest, dict) or type(manifest.get("source_dirty")) is not bool:
        raise ValueError("invalid build provenance")
    qualified_shape = dict(manifest, source_dirty=False)
    if manifest.get("dependency_digest") == "unavailable":
        qualified_shape["dependency_digest"] = "0" * 64
    # Experimental Garner is valid diagnostic provenance, but the original
    # manifest must still fail the separate release-provenance check below.
    configuration = manifest.get("configuration")
    if isinstance(configuration, str):
        lines = configuration.splitlines()
        qualified_shape["configuration"] = "\n".join(
            "garner_table=OFF" if line.partition("=")[0] == "garner_table" else line
            for line in lines)
    backend = "rocwmma" if case["candidate"] == "epilogue_rocwmma" else None
    if not manifest_provenance_complete(qualified_shape, backend):
        raise ValueError("invalid build provenance")
    return {key: capture[key] for key in ("build_manifest", "device", "target", "runtime_version")}


def summarize_encoded_reuse(observations):
    """One sustained batch observation per independent process, no extrapolation.

    Caller supplies only contract-validated records from one immutable artifact
    and build/device/runtime cohort. Missing controls stay explicitly incomplete.
    """
    groups = {}
    dimensions = ("m","n","k","prefix","group","unsigned_inputs")
    for observation in observations:
        case = observation["case"]
        if case["candidate"] not in REUSE:
            continue
        key = tuple(case[field] for field in dimensions)
        groups.setdefault(key,{}).setdefault(case["reuse_count"],{}).setdefault(case["candidate"],[]).append(observation)
    result = []
    for key, points in sorted(groups.items()):
        row = {"case":dict(zip(dimensions,key)),"points":[],
               "break_even_definition":"first_observed_cold_win_among_tested_reuse_counts_no_interpolation",
               "first_observed_cold_winning_reuse_count":None}
        for reuse_count, candidates in sorted(points.items()):
            point = {"reuse_count":reuse_count,"candidates":{},"missing_candidates":sorted(set(REUSE)-set(candidates)),
                     "status":"incomplete_cohort", "cold_faster_than_direct_median":None,
                     "warm_faster_than_direct_median":None}
            repeats = []
            for candidate, records in sorted(candidates.items()):
                ids = [record["independent_repeat"] for record in records]
                if len(ids) != len(set(ids)):
                    raise ValueError("duplicate independent encoded-B observation")
                repeats.append(set(ids))
                values = [record["sustained_batch_ns"] for record in records]
                if not all(positive_number(value) for value in values):
                    raise ValueError("invalid encoded-B comparison observation")
                if candidate == "encoded_b_warm" and not all(positive_number(record["prepare_ns"]) for record in records):
                    raise ValueError("invalid encoded-B comparison preparation observation")
                median = statistics.median(values)
                point["candidates"][candidate] = {
                    "independent_processes":len(records),"run_ids":[record["run_id"] for record in records],
                    "sustained_batch_ns_median":median,"sustained_batch_ns_min":min(values),
                    "sustained_batch_ns_max":max(values),"per_gemm_ns_from_batch_median":median/reuse_count,
                    "warm_prepare_ns_median":statistics.median(record["prepare_ns"] for record in records)
                        if candidate == "encoded_b_warm" else None}
            if not point["missing_candidates"] and all(ids == repeats[0] for ids in repeats):
                point["status"]="matched_independent_cohort"
                timing={name:entry["sustained_batch_ns_median"] for name,entry in point["candidates"].items()}
                point["cold_faster_than_direct_median"]=timing["encoded_b_cold"] < timing["wmma"]
                point["warm_faster_than_direct_median"]=timing["encoded_b_warm"] < timing["wmma"]
                if point["cold_faster_than_direct_median"] and row["first_observed_cold_winning_reuse_count"] is None:
                    row["first_observed_cold_winning_reuse_count"]=reuse_count
            elif not point["missing_candidates"]:
                point["status"]="unmatched_independent_repeats"
            row["points"].append(point)
        result.append(row)
    return result


def summarize_conversion(observations):
    """Source/preparation ablations; no promotion or unmeasured extrapolation."""
    dimensions=("m","n","k","prefix","group","unsigned_inputs","reuse_count","input_change")
    groups={}
    for observation in observations:
        case=observation["case"];key=tuple(case[field] for field in dimensions)
        recipe=tuple(case[field] for field in ("candidate","conversion","waves","k_depth","lds_buffers"))
        groups.setdefault(key,{}).setdefault(recipe,[]).append(observation)
    output=[]
    for key,recipes in sorted(groups.items()):
        rows=[]
        for recipe,items in sorted(recipes.items()):
            items=sorted(items,key=lambda item:item["run_id"])
            if len({item["run_id"] for item in items})!=len(items):raise ValueError("duplicate conversion process")
            samples=[item["sampled_batch_ns"] for item in items]
            complete=[item["sampled_batch_ns"]+(item["prepare_ns"] or 0) for item in items]
            rows.append({"recipe":dict(zip(("candidate","conversion","waves","k_depth","lds_buffers"),recipe)),
                "run_ids":[item["run_id"] for item in items],"independent_processes":len(items),
                "sampled_batch_ns_median":statistics.median(samples),"sampled_batch_ns_min":min(samples),"sampled_batch_ns_max":max(samples),
                "setup_plus_sampled_batch_ns_median":statistics.median(complete),
                "sustained_batch_mean_ns_process_median":statistics.median(item["sustained_batch_ns"] for item in items),
                "preparation_ns_process_median":statistics.median(item["prepare_ns"] or 0 for item in items),
                "observations":items})
        output.append({"work_contract":dict(zip(dimensions,key)),"recipes":rows,
            "comparison_scope":"matched_exact_residue_work_separate_cold_prepared_and_pack_boundaries",
            "promotion_eligible":False,"hardware_qualification":"not_established_by_this_report"})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True,
                        help="JSON cases; queue workers1..1024 default32; adaptive prefix20/bound_method=axis_l1; compact requires workers1..65535; tiled requires k_depth/lds_buffers; encoded B requires reuse_count1..256, wmma defaults1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--order-seed", type=int, default=3135)
    parser.add_argument("--plan-only",action="store_true",help="emit exact dataset work preflight without opening or launching the executable")
    args = parser.parse_args()
    try:
        cases = strict_json(args.cases.read_text(encoding="utf-8"))
        if (not isinstance(cases, list) or not 1 <= len(cases) <= 64 or not 1 <= args.runs <= 100
                or not 1 <= args.repeats <= 10000 or not 1 <= args.warmups <= 1000 or args.timeout <= 0):
            raise ValueError("invalid case/run/repeat/timeout domain")
        cases = [validate_case(c) for c in cases]
        if len({json.dumps(case, sort_keys=True) for case in cases}) != len(cases):
            raise ValueError("duplicate normalized cases; use --runs for independent processes")
    except (ValueError, OSError) as error:
        parser.error(str(error))
    preflight=[]
    for case in cases:
        if 'dataset_profile' in case:
            preflight.append({'case':case,'dataset_identity':dataset_identity(case['dataset_profile'],case['dataset_seed']),
                'model':dataset_model(case['m'],case['n'],case['k'],case['unsigned_inputs'],case['group'],
                    case['dataset_profile'],case['dataset_seed'],case['candidate'] in ADAPTIVE)})
    if args.plan_only:
        print(json.dumps({'schema':'rns8_native_dataset_preflight_v2','cases':preflight,'gpu_work_executed':False},indent=2));return 0
    exe = args.exe.resolve(strict=True)
    artifact = hashlib.sha256(exe.read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    collector_sources = [Path(__file__).resolve(), Path(__file__).with_name("build_manifest.py").resolve(),
                         Path(__file__).with_name("modulus_set_search.py").resolve(), Path(__file__).with_name("native_dataset.py").resolve()]
    collector_identity = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in collector_sources}
    distribution_observations=[]
    campaign = str(uuid.uuid4())
    jobs = [(c, independent) for independent in range(args.runs) for c in cases]
    random.Random(args.order_seed).shuffle(jobs)
    campaign_identity = None
    reuse_observations = []
    conversion_observations = []
    for ordinal, (case, independent) in enumerate(jobs):
        if hashlib.sha256(exe.read_bytes()).hexdigest() != artifact:
            raise RuntimeError("native executable changed during collection")
        run_id = str(uuid.uuid4())
        command = [str(exe), case["candidate"], *(str(case[k]) for k in ("m", "n", "k", "prefix", "group")),
                   str(int(case["unsigned_inputs"])), str(args.repeats), str(args.warmups), run_id]
        if case["candidate"] in (*QUEUED, "adaptive_compact"):
            command.append(str(case["workers"]))
        if case["candidate"]=='split_k':command.extend([str(case['fragments']),case['merge_policy']])
        if case["candidate"] in ADAPTIVE:
            command.append(case["bound_method"])
        if case["candidate"] in TILED:
            command.extend(str(case[key]) for key in ("k_depth", "lds_buffers"))
        if case["candidate"] in CONVERSION:
            command.extend(str(case[key]) for key in ("k_depth","lds_buffers","waves","conversion","reuse_count","input_change"))
        if case["candidate"] in REUSE:
            command.append(str(case["reuse_count"]))
        if case["candidate"].startswith("epilogue_"):command.append(case["coefficient_profile"])
        if 'dataset_profile' in case:command.extend(['--dataset',case['dataset_profile'],'--seed',str(case['dataset_seed'])])
        record = {"schema": "rns8_native_collection_v2", "campaign_id": campaign, "run_id": run_id,
                  "artifact_sha256": artifact, "command": command, "case": case,
                  "collector_sha256": collector_identity["collect_native_calibration.py"],
                  "collector_sources_sha256": collector_identity,
                  "started_unix_ns": time.time_ns(), "ordinal": ordinal, "independent_repeat": independent,
                  "order_seed": args.order_seed, "promotion_eligible": False}
        failure = None
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout, check=False)
            record.update(exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr)
            if hashlib.sha256(exe.read_bytes()).hexdigest() != artifact:
                raise ValueError("native executable changed while executing; capture is ineligible")
            if result.returncode != 0:
                raise ValueError("native microcase failed")
            capture = strict_json(result.stdout)
            identity = validate_capture(capture, case, run_id, args.repeats, args.warmups)
            if campaign_identity is not None and not same_value(identity, campaign_identity):
                raise ValueError("build/device/runtime identity changed during collection")
            campaign_identity = identity
            record["comparison_identity"] = {"artifact_sha256": artifact, **identity,
                                             "case": case, **scheduling_contract(case),
                                             **adaptive_contract(case),
                                             **tiled_contract(case),
                                             **conversion_contract(case),
                                             **reuse_contract(case),
                                             "input_domain": capture["input_domain"],
                                             "output_contract": capture["output_contract"],
                                             "dataset_identity":capture.get("dataset_identity"), "dataset_model":capture.get("dataset_model"),
                                             "seed": capture["seed"], "repeats": args.repeats,
                                             "warmups": args.warmups, "timing_scope": capture["timing_scope"],
                                             "cache_state": capture["cache_state"]}
            record["provenance_complete"] = manifest_provenance_complete(
                capture["build_manifest"], "rocwmma" if case["candidate"] == "epilogue_rocwmma" else None)
            record["validated"] = True
            if 'dataset_profile' in case:
                distribution_observations.append({'case':case,'run_id':run_id,'sustained_mean_call_ns':capture['sustained']['mean_call_ns'],
                    'model':capture['dataset_model'],'samples':capture['samples'],
                    'scratch_bytes':capture.get('adaptive_peak_scratch_bytes'),
                    'comparison_identity':record['comparison_identity']})
            if case["candidate"] in CONVERSION:
                conversion_observations.append({"case":case,"run_id":run_id,"independent_repeat":independent,
                    "sampled_batch_ns":statistics.median(sample["uninstrumented_host_call_ns"] for sample in capture["samples"]),
                    "sustained_batch_ns":capture["sustained"]["mean_call_ns"],"prepare_ns":capture["conversion_prepare_ns"],
                    "provenance_complete":record["provenance_complete"]})
            if case["candidate"] in REUSE:
                reuse_observations.append({"case":case,"independent_repeat":independent,"run_id":run_id,
                    "sustained_batch_ns":capture["sustained"]["mean_call_ns"],
                    "prepare_ns":capture["encoded_b_prepare_ns"],"provenance_complete":record["provenance_complete"]})
        except subprocess.TimeoutExpired as error:
            def timeout_text(value):
                return value.decode(errors="replace") if isinstance(value, bytes) else value or ""
            record.update(exit_code=None, reason="timeout", timeout_seconds=args.timeout,
                          stdout=timeout_text(error.stdout), stderr=timeout_text(error.stderr))
            failure = "native microcase timeout"
        except (ValueError, OSError) as error:
            record.setdefault("exit_code", None)
            record.update(reason="contract/reference mismatch", validation_error=str(error))
            failure = f"native microcase contract/reference mismatch: {error}"
        record.setdefault("validated", False)
        with (args.output / (run_id + ".json")).open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2)
        if failure:
            raise RuntimeError(f"{failure}; retained {run_id}")
    if any(case["candidate"] in ENCODED for case in cases):
        comparison = {"schema":"rns8_encoded_b_reuse_comparison_v1", "campaign_id":campaign,
            "comparison_identity":{"artifact_sha256":artifact,**campaign_identity},
            "collector_sources_sha256":collector_identity,
            "observation":"one sustained batch mean per independent process; median across processes",
            "boundary":"private raw residue calls; cold includes preparation/allocation/release; warm setup reported separately",
            "promotion_eligible":False,"provenance_complete":all(row["provenance_complete"] for row in reuse_observations),
            "groups":summarize_encoded_reuse(reuse_observations)}
        with (args.output/("encoded-b-reuse-comparison-"+campaign+".json")).open("x",encoding="utf-8") as stream:
            json.dump(comparison,stream,indent=2)
    if conversion_observations:
        comparison={"schema":"rns8_native_conversion_comparison_v1","campaign_id":campaign,
            "comparison_identity":{"artifact_sha256":artifact,**campaign_identity},
            "collector_sources_sha256":collector_identity,"promotion_eligible":False,
            "estimator":"per_process_sample_median_then_process_median_setup_added_within_process",
            "groups":summarize_conversion(conversion_observations)}
        with (args.output/("conversion-comparison-"+campaign+".json")).open("x",encoding="utf-8") as stream:
            json.dump(comparison,stream,indent=2)
    if distribution_observations:
        groups={}
        for row in distribution_observations:
            case=row['case'];key=tuple(case[k] for k in ('m','n','k','prefix','group','unsigned_inputs','dataset_profile','dataset_seed'))
            groups.setdefault(key,{}).setdefault(case['candidate'],[]).append(row)
        summary=[]
        for key,candidates in groups.items():
            item={'work_contract':dict(zip(('m','n','k','prefix','group','unsigned_inputs','dataset_profile','dataset_seed'),key)),
                  'candidates':{},'missing_controls':sorted({'wmma','pack_gemm','adaptive_grid','adaptive_compact'}-set(candidates))}
            for candidate,observations in candidates.items():
                times=[row['sustained_mean_call_ns'] for row in observations]
                item['candidates'][candidate]={'independent_processes':len(times),'complete_call_ns_process_median':statistics.median(times),
                    'complete_call_ns_process_min':min(times),'complete_call_ns_process_max':max(times),
                    'source_work_model':observations[0]['model'],'observations':observations}
            summary.append(item)
        with (args.output/('adaptive-distribution-comparison-'+campaign+'.json')).open('x',encoding='utf-8') as stream:
            json.dump({'schema':'rns8_adaptive_distribution_comparison_v2','groups':summary,'promotion_eligible':False,
                'global_minimal_completed_output_control_implemented':False,
                'estimator':'one complete sustained mean per independent process; no pooling different input distributions',
                'warning':'modeled channel/group savings are not measured speedups; all complete discovery/compaction/suffix/commit costs remain charged'},stream,indent=2)
    print(f"Collected {len(jobs)} independent native microcase records; qualification remains pending")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
