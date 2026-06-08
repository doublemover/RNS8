def as_reused_pack_capture(capture: dict) -> dict:
    reused = copy.deepcopy(capture)
    repeats = reused["repeats"]
    reused["reuse_packed_inputs"] = True
    reused["pack_mode"] = "prepacked_reuse"
    reused["prepack_reuse_operands"] = ["A", "B"]
    reused["prepack_reuse_strategy"] = "persistent_matrix_residency"
    reused["prepack_setup_us"] = 123
    reused["avg_prepack_setup_us"] = 123.0
    reused["avg_pack_us"] = 0.0
    reused["raw_timings_us"]["pack"] = [0] * repeats
    reused["timing_summary_us"]["pack"] = zero_summary()
    reused["timing_metadata"]["pack_mode"] = "prepacked_reuse"
    reused["timing_metadata"]["prepack_reuse_operands"] = ["A", "B"]
    reused["timing_metadata"]["prepack_reuse_strategy"] = "persistent_matrix_residency"
    reused["timing_metadata"]["phase_notes"]["pack"] = (
        "zero-valued per-repeat phase; A and B were packed once into persistent matrices before warmups"
    )
    reused["timing_metadata"]["phase_notes"]["end_to_end"] = (
        "per-repeat rns_gemm plus crt_export host timing; excludes one-time prepack_setup_us"
    )
    reused["timing_metadata"]["phase_availability"]["prepack_setup"] = {
        "timed": True,
        "timing_key": "prepack_setup_us",
        "scope": "one_time_before_warmups",
        "reason": "A and B were packed once before warmups and reused for every measured repeat",
    }
    for phase in ["pack_h2d", "pack_kernel", "finite_pack_h2d", "finite_pack_kernel", "pack"]:
        timings = reused.get("gpu_event_timings_us")
        summaries = reused.get("gpu_event_timing_summary_us")
        if isinstance(timings, dict) and phase in timings:
            timings[phase] = [0.0] * repeats
        if isinstance(summaries, dict) and phase in summaries:
            summaries[phase] = zero_summary()
    return reused


def as_hipblaslt_reused_ab_capture(capture: dict) -> dict:
    reused = as_reused_pack_capture(capture)
    phase = "hipblaslt_pack_transpose_centered"
    phase_order = reused["timing_metadata"].get("gpu_event_phase_order")
    if isinstance(phase_order, list) and phase in phase_order:
        phase_order.remove(phase)
    for field in ["gpu_event_timings_us", "gpu_event_timing_summary_us"]:
        values = reused.get(field)
        if isinstance(values, dict):
            values.pop(phase, None)
    return reused


def as_reused_a_capture(capture: dict) -> dict:
    reused = copy.deepcopy(capture)
    reused["reuse_packed_inputs"] = True
    reused["pack_mode"] = "prepacked_reuse_a"
    reused["prepack_reuse_operands"] = ["A"]
    reused["prepack_reuse_strategy"] = "persistent_matrix_residency"
    reused["prepack_setup_us"] = 77
    reused["avg_prepack_setup_us"] = 77.0
    reused["timing_metadata"]["pack_mode"] = "prepacked_reuse_a"
    reused["timing_metadata"]["prepack_reuse_operands"] = ["A"]
    reused["timing_metadata"]["prepack_reuse_strategy"] = "persistent_matrix_residency"
    reused["timing_metadata"]["phase_notes"]["pack"] = (
        "per-repeat host timing for packing B; A was packed once into a persistent matrix before warmups"
    )
    reused["timing_metadata"]["phase_notes"]["end_to_end"] = (
        "per-repeat pack of non-reused input plus rns_gemm plus crt_export host timing; excludes one-time "
        "prepack_setup_us"
    )
    reused["timing_metadata"]["phase_availability"]["prepack_setup"] = {
        "timed": True,
        "timing_key": "prepack_setup_us",
        "scope": "one_time_before_warmups",
        "reason": "prepacked A once before warmups and reused for every measured repeat",
    }
    return reused


def as_exact_wide_capture(capture: dict) -> dict:
    exact = copy.deepcopy(capture)
    exact["benchmark"] = "rns8_exact_wide_persistent_rns"
    exact["semantics"] = "exact_wide_signed"
    exact["bound_kind"] = "none"
    exact["bound_mode"] = "global"
    exact["bound"] = 0
    exact["prefix"] = 20
    exact["finite_modulus"] = None
    exact["tile_bounds_u64"] = None
    exact["epilogue_type"] = "exact_wide_signed_limb_export"
    exact["exact_wide_limb_count"] = 4
    exact["residue_chain_length"] = 1
    exact["residue_output_mode"] = "host_export"
    exact["input_distribution"] = "signed_uniform_-16_16"
    exact["comparison_baseline"]["required_before_speedup_claim"] = [
        "same_contract_cpu_reference",
        "same_contract_direct_hip_correctness",
    ]
    exact["backend_metadata"]["epilogue_mode"] = "ck_fused_i32_to_centered_residue_rns_output"
    exact["backend_metadata"]["autotune_key"] = with_accumulator_key_fields(
        (
        "backend=ck;semantics=exact_wide_signed;m=64;n=128;k=64;prefix=20;tile_m=128;tile_n=128;"
        "groups=1;adaptive_prefix=0;adaptive_skip=0;kernel=ck_wmma_cshuffle_i8_i32_mod251_255_256_centered_epilogue_v2;"
        "epilogue=ck_fused_i32_to_centered_residue_rns_output"
        ),
        exact,
    )
    exact["schedule_metadata"]["min_selected_prefix"] = 20
    exact["schedule_metadata"]["max_selected_prefix"] = 20
    exact["schedule_metadata"]["prefix_group_count"] = 1
    exact["schedule_metadata"]["adaptive_execution_applied"] = False
    exact["avg_per_modulus_gemm_estimate_us"] = float(exact["avg_rns_gemm_us"]) / 20.0
    exact["timing_note"] = (
        "host wall-clock timings for persistent exact-wide RNS packing, RNS GEMM, and fixed-width little-endian "
        "limb export; GPU event timing names exact-wide export operation groups when backend hooks are available"
    )
    exact["timing_metadata"]["phase_notes"]["crt_export"] = (
        "per-repeat host timing for fixed-width exact-wide limb export"
    )

    renamed = {
        "crt_export_status_memset": "exact_wide_export_status_memset",
        "crt_export_kernel": "exact_wide_export_kernel",
        "crt_export_status_d2h": "exact_wide_export_status_d2h",
        "crt_export_d2h": "exact_wide_export_d2h",
    }
    phase_order = exact["timing_metadata"].get("gpu_event_phase_order")
    if isinstance(phase_order, list):
        exact["timing_metadata"]["gpu_event_phase_order"] = [renamed.get(item, item) for item in phase_order]
    for field in ["gpu_event_timings_us", "gpu_event_timing_summary_us"]:
        values = exact.get(field)
        if isinstance(values, dict):
            for old, new in renamed.items():
                if old in values:
                    values[new] = values.pop(old)
    return exact

def as_residue_current_chain_capture(capture: dict) -> dict:
    chain = as_exact_wide_capture(capture)
    repeats = chain["repeats"]
    chain["m"] = 64
    chain["n"] = 64
    chain["k"] = 64
    chain["epilogue_type"] = "residue_current_rns_output"
    chain["residue_chain_length"] = 3
    chain["residue_chain_final_export"] = False
    chain["residue_output_mode"] = "residue_current_rns"
    chain["benchmark_execution_mode"] = "residue_current_rns_chain"
    chain["timing_metadata"]["benchmark_execution_mode"] = "residue_current_rns_chain"
    chain["timing_note"] = (
        "host wall-clock timings for an exact-wide residue-current RNS GEMM chain; each measured repeat runs "
        "3 resident RNS GEMM calls before host export, raw_timings_us.crt_export is intentionally zero, and one "
        "final fixed-width limb export runs after measured repeats only to produce checksum_u64"
    )
    chain["timing_metadata"]["phase_notes"]["rns_gemm"] = (
        "per-repeat host timing for 3 chained rns8_gemm_rns calls that keep the intermediate output resident "
        "in RNS form"
    )
    chain["timing_metadata"]["phase_notes"]["crt_export"] = (
        "zero-valued per-repeat phase; residue-current chain mode defers host limb export until one final checksum "
        "export after measured repeats"
    )
    chain["timing_metadata"]["phase_notes"]["end_to_end"] = (
        "per-repeat pack plus chained rns_gemm host timing; excludes the final checksum-only limb export"
    )
    chain["raw_timings_us"]["crt_export"] = [0 for _ in range(repeats)]
    chain["timing_summary_us"]["crt_export"] = {"avg": 0.0, "median": 0.0, "p95": 0.0}
    chain["avg_crt_export_us"] = 0.0
    add_target_variant_fields(chain)
    add_requested_next_op_fields(chain, resolved="rns-gemm")
    add_output_policy_fields(
        chain,
        status_handling="structurally_elided",
        per_repeat_export=False,
        final_checksum_export=True,
    )
    add_ck_chain_gpu_events(chain, 20)
    return chain


def as_residue_chain_final_export_capture(capture: dict) -> dict:
    chain = as_exact_wide_capture(capture)
    repeats = chain["repeats"]
    chain["benchmark"] = "rns8_residue_chain_final_host_export"
    chain["benchmark_execution_mode"] = "residue_chain_final_host_export"
    chain["command_line"] = (
        "rns8-bench --backend ck --semantics exact-wide-signed --m 64 --n 64 --k 64 "
        "--residue-chain-length 3 --residue-chain-final-export --warmups 1 --repeats 2 --seed 7"
    )
    chain["m"] = 64
    chain["n"] = 64
    chain["k"] = 64
    chain["epilogue_type"] = "exact_wide_signed_limb_export"
    chain["residue_chain_length"] = 3
    chain["residue_chain_final_export"] = True
    chain["residue_output_mode"] = "host_export"
    chain["timing_note"] = (
        "host wall-clock timings for a final-output RNS GEMM chain; each measured repeat runs 3 resident "
        "RNS GEMM calls with intermediate outputs kept in RNS form, then exports the final logical output "
        "inside the measured repeat"
    )
    chain["timing_metadata"]["benchmark_execution_mode"] = "residue_chain_final_host_export"
    chain["timing_metadata"]["residue_chain_final_export"] = True
    chain["timing_metadata"]["phase_notes"]["rns_gemm"] = (
        "per-repeat host timing for 3 chained rns8_gemm_rns calls that keep intermediate outputs resident "
        "in RNS form"
    )
    chain["timing_metadata"]["phase_notes"]["crt_export"] = (
        "per-repeat host timing for exporting the final chained exact-wide limb output inside the measured repeat"
    )
    chain["timing_metadata"]["phase_notes"]["end_to_end"] = (
        "per-repeat pack plus 3 chained rns_gemm calls plus final logical export host timing"
    )
    chain["raw_timings_us"]["crt_export"] = [90, 100][:repeats]
    chain["raw_timings_us"]["end_to_end"] = [
        p + g + e
        for p, g, e in zip(
            chain["raw_timings_us"]["pack"],
            chain["raw_timings_us"]["rns_gemm"],
            chain["raw_timings_us"]["crt_export"],
        )
    ]
    chain["timing_summary_us"]["crt_export"] = summary(chain["raw_timings_us"]["crt_export"])
    chain["timing_summary_us"]["end_to_end"] = summary(chain["raw_timings_us"]["end_to_end"])
    chain["avg_crt_export_us"] = chain["timing_summary_us"]["crt_export"]["avg"]
    chain["avg_end_to_end_us"] = chain["timing_summary_us"]["end_to_end"]["avg"]
    for phase in ["exact_wide_export_status_memset", "exact_wide_export_status_d2h"]:
        if phase in chain.get("gpu_event_timings_us", {}):
            chain["gpu_event_timings_us"][phase] = [0.0 for _ in range(repeats)]
        if phase in chain.get("gpu_event_timing_summary_us", {}):
            chain["gpu_event_timing_summary_us"][phase] = zero_summary()
    add_target_variant_fields(chain)
    add_requested_next_op_fields(chain, resolved="final-export")
    add_output_policy_fields(
        chain,
        status_handling=chain["output_policy"]["status_handling"] if "output_policy" in chain else "structurally_elided",
        per_repeat_export=True,
        final_checksum_export=False,
    )
    return chain


def as_bounded_residue_chain_independent_final_export_capture(capture: dict) -> dict:
    chain = as_bounded_residue_current_chain_capture(capture)
    repeats = chain["repeats"]
    chain["benchmark"] = "rns8_residue_chain_independent_final_host_export"
    chain["benchmark_execution_mode"] = "residue_chain_independent_final_host_export"
    chain["command_line"] = (
        "rns8-bench --backend ck --semantics bounded-i64 --m 64 --n 64 --k 64 "
        "--residue-chain-length 3 --residue-chain-independent-final-export --warmups 1 --repeats 2 --seed 7"
    )
    chain["epilogue_type"] = "crt_export"
    chain["residue_chain_final_export"] = True
    chain["residue_chain_independent_final_export"] = True
    chain["residue_output_mode"] = "host_export"
    chain["pack_mode"] = "per_repeat_repack"
    chain["reuse_packed_inputs"] = False
    chain["prepack_reuse_operands"] = []
    chain["prepack_reuse_strategy"] = "none"
    chain["prepack_setup_us"] = None
    chain["avg_prepack_setup_us"] = None
    chain["timing_note"] = (
        "host wall-clock timings for an independent final-output RNS GEMM chain; each measured repeat performs "
        "3 GEMM calls, exports every intermediate host output, repacks non-final intermediates as the next native "
        "input, and includes all materialization work"
    )
    chain["timing_metadata"]["benchmark_execution_mode"] = "residue_chain_independent_final_host_export"
    chain["timing_metadata"]["residue_chain_final_export"] = True
    chain["timing_metadata"]["residue_chain_independent_final_export"] = True
    chain["timing_metadata"]["pack_mode"] = "per_repeat_repack"
    chain["timing_metadata"]["prepack_reuse_operands"] = []
    chain["timing_metadata"]["prepack_reuse_strategy"] = "none"
    chain["timing_metadata"]["phase_notes"]["pack"] = (
        "per-repeat host timing for packing original A/B plus repacking each non-final host intermediate output"
    )
    chain["timing_metadata"]["phase_notes"]["rns_gemm"] = (
        "per-repeat host timing for 3 independent rns8_gemm_rns calls separated by host export and repack"
    )
    chain["timing_metadata"]["phase_notes"]["crt_export"] = (
        "per-repeat host timing for exporting every intermediate and final chained logical output"
    )
    chain["timing_metadata"]["phase_notes"]["end_to_end"] = (
        "per-repeat original input pack plus intermediate repacks, 3 rns_gemm calls, and every logical export"
    )
    chain["raw_timings_us"]["pack"] = [40, 45][:repeats]
    chain["raw_timings_us"]["rns_gemm"] = [70, 75][:repeats]
    chain["raw_timings_us"]["per_modulus_gemm_estimate"] = [
        value / 9.0 for value in chain["raw_timings_us"]["rns_gemm"]
    ]
    chain["raw_timings_us"]["crt_export"] = [55, 60][:repeats]
    chain["raw_timings_us"]["end_to_end"] = [
        p + g + e
        for p, g, e in zip(
            chain["raw_timings_us"]["pack"],
            chain["raw_timings_us"]["rns_gemm"],
            chain["raw_timings_us"]["crt_export"],
        )
    ]
    for phase in ["pack", "rns_gemm", "crt_export", "end_to_end"]:
        chain["timing_summary_us"][phase] = summary(chain["raw_timings_us"][phase])
    chain["timing_summary_us"]["per_modulus_gemm_estimate"] = summary(
        chain["raw_timings_us"]["per_modulus_gemm_estimate"]
    )
    chain["avg_pack_us"] = chain["timing_summary_us"]["pack"]["avg"]
    chain["avg_rns_gemm_us"] = chain["timing_summary_us"]["rns_gemm"]["avg"]
    chain["avg_per_modulus_gemm_estimate_us"] = chain["timing_summary_us"]["per_modulus_gemm_estimate"]["avg"]
    chain["avg_crt_export_us"] = chain["timing_summary_us"]["crt_export"]["avg"]
    chain["avg_end_to_end_us"] = chain["timing_summary_us"]["end_to_end"]["avg"]
    export_phases = [
        "crt_export_status_memset",
        "crt_export_kernel",
        "crt_export_status_d2h",
        "crt_export_d2h",
        "crt_export",
    ]
    for phase in export_phases:
        if phase not in chain["timing_metadata"]["gpu_event_phase_order"]:
            chain["timing_metadata"]["gpu_event_phase_order"].append(phase)
        chain["gpu_event_timings_us"][phase] = [1.0 for _ in range(repeats)]
        chain["gpu_event_timing_summary_us"][phase] = summary(chain["gpu_event_timings_us"][phase])
    add_requested_next_op_fields(chain, resolved="final-export")
    add_output_policy_fields(
        chain,
        status_handling="required",
        per_repeat_export=True,
        final_checksum_export=False,
    )
    return chain


def as_exact_wide_residue_chain_independent_final_export_capture(capture: dict) -> dict:
    chain = as_residue_chain_final_export_capture(capture)
    repeats = chain["repeats"]
    chain["benchmark"] = "rns8_residue_chain_independent_final_host_export"
    chain["benchmark_execution_mode"] = "residue_chain_independent_final_host_export"
    chain["command_line"] = (
        "rns8-bench --backend ck --semantics exact-wide-signed --m 64 --n 64 --k 64 "
        "--residue-chain-length 3 --residue-chain-independent-final-export --warmups 1 --repeats 2 --seed 7"
    )
    chain["residue_chain_final_export"] = True
    chain["residue_chain_independent_final_export"] = True
    chain["residue_output_mode"] = "host_export"
    chain["pack_mode"] = "per_repeat_repack"
    chain["reuse_packed_inputs"] = False
    chain["prepack_reuse_operands"] = []
    chain["prepack_reuse_strategy"] = "none"
    chain["prepack_setup_us"] = None
    chain["avg_prepack_setup_us"] = None
    chain["timing_note"] = (
        "host wall-clock timings for an independent exact-wide final-output RNS GEMM chain; each measured repeat "
        "performs 3 GEMM calls, exports every intermediate fixed-limb host output, repacks non-final "
        "intermediates as the next RNS input, and includes all materialization work"
    )
    chain["timing_metadata"]["benchmark_execution_mode"] = "residue_chain_independent_final_host_export"
    chain["timing_metadata"]["residue_chain_final_export"] = True
    chain["timing_metadata"]["residue_chain_independent_final_export"] = True
    chain["timing_metadata"]["pack_mode"] = "per_repeat_repack"
    chain["timing_metadata"]["prepack_reuse_operands"] = []
    chain["timing_metadata"]["prepack_reuse_strategy"] = "none"
    chain["timing_metadata"]["phase_notes"]["pack"] = (
        "per-repeat host timing for packing original A/B plus repacking each non-final exact-wide limb output"
    )
    chain["timing_metadata"]["phase_notes"]["rns_gemm"] = (
        "per-repeat host timing for 3 independent rns8_gemm_rns calls separated by fixed-limb export and RNS repack"
    )
    chain["timing_metadata"]["phase_notes"]["crt_export"] = (
        "per-repeat host timing for exporting every intermediate and final chained exact-wide limb output"
    )
    chain["timing_metadata"]["phase_notes"]["end_to_end"] = (
        "per-repeat original input pack plus intermediate exact-wide repacks, 3 rns_gemm calls, and every logical export"
    )
    chain["raw_timings_us"]["pack"] = [80, 85][:repeats]
    chain["raw_timings_us"]["rns_gemm"] = [140, 150][:repeats]
    chain["raw_timings_us"]["per_modulus_gemm_estimate"] = [
        value / 20.0 for value in chain["raw_timings_us"]["rns_gemm"]
    ]
    chain["raw_timings_us"]["crt_export"] = [110, 120][:repeats]
    chain["raw_timings_us"]["end_to_end"] = [
        p + g + e
        for p, g, e in zip(
            chain["raw_timings_us"]["pack"],
            chain["raw_timings_us"]["rns_gemm"],
            chain["raw_timings_us"]["crt_export"],
        )
    ]
    for phase in ["pack", "rns_gemm", "crt_export", "end_to_end"]:
        chain["timing_summary_us"][phase] = summary(chain["raw_timings_us"][phase])
    chain["timing_summary_us"]["per_modulus_gemm_estimate"] = summary(
        chain["raw_timings_us"]["per_modulus_gemm_estimate"]
    )
    chain["avg_pack_us"] = chain["timing_summary_us"]["pack"]["avg"]
    chain["avg_rns_gemm_us"] = chain["timing_summary_us"]["rns_gemm"]["avg"]
    chain["avg_per_modulus_gemm_estimate_us"] = chain["timing_summary_us"]["per_modulus_gemm_estimate"]["avg"]
    chain["avg_crt_export_us"] = chain["timing_summary_us"]["crt_export"]["avg"]
    chain["avg_end_to_end_us"] = chain["timing_summary_us"]["end_to_end"]["avg"]
    return chain


def as_bounded_residue_current_chain_capture(capture: dict) -> dict:
    chain = copy.deepcopy(capture)
    repeats = chain["repeats"]
    chain["m"] = 64
    chain["n"] = 64
    chain["k"] = 64
    chain["bound_mode"] = "global"
    chain["bound_kind"] = "global_max_abs"
    chain["bound"] = 1099511627776
    chain["tile_bounds_u64"] = None
    chain["schedule_metadata"]["tile_rows"] = 1
    chain["schedule_metadata"]["tile_cols"] = 1
    chain["schedule_metadata"]["tile_count"] = 1
    chain["schedule_metadata"]["min_required_prefix"] = 9
    chain["schedule_metadata"]["max_required_prefix"] = 9
    chain["schedule_metadata"]["min_selected_prefix"] = 9
    chain["schedule_metadata"]["max_selected_prefix"] = 9
    chain["schedule_metadata"]["prefix_group_count"] = 1
    chain["schedule_metadata"]["adaptive_prefix_active"] = False
    chain["schedule_metadata"]["adaptive_skip_active"] = False
    chain["schedule_metadata"]["adaptive_execution_applied"] = False
    chain["schedule_metadata"]["range_bit_length"] = 41
    chain["epilogue_type"] = "residue_current_rns_output"
    chain["residue_chain_length"] = 3
    chain["residue_chain_final_export"] = False
    chain["residue_output_mode"] = "residue_current_rns"
    chain["benchmark_execution_mode"] = "residue_current_rns_chain"
    chain["timing_metadata"]["benchmark_execution_mode"] = "residue_current_rns_chain"
    chain["backend_metadata"]["autotune_key"] = with_accumulator_key_fields(
        (
        "backend=ck;semantics=bounded_i64;m=64;n=64;k=64;prefix=9;tile_m=128;tile_n=128;"
        "groups=1;adaptive_prefix=0;adaptive_skip=0;kernel=ck_wmma_cshuffle_i8_i32_mod251_255_256_centered_epilogue_v2;"
        "epilogue=ck_fused_i32_to_centered_residue_then_crt_export"
        ),
        chain,
    )
    chain["timing_note"] = (
        "host wall-clock timings for a residue-current RNS GEMM chain; each measured repeat runs 3 resident "
        "RNS GEMM calls before host export, raw_timings_us.crt_export is intentionally zero, and one final "
        "logical export runs after measured repeats only to produce checksum_u64"
    )
    chain["timing_metadata"]["phase_notes"]["rns_gemm"] = (
        "per-repeat host timing for 3 chained rns8_gemm_rns calls that keep the intermediate output resident "
        "in RNS form"
    )
    chain["timing_metadata"]["phase_notes"]["crt_export"] = (
        "zero-valued per-repeat phase; residue-current chain mode defers host logical export until one final "
        "checksum export after measured repeats"
    )
    chain["timing_metadata"]["phase_notes"]["end_to_end"] = (
        "per-repeat pack plus chained rns_gemm host timing; excludes the final checksum-only logical export"
    )
    chain["raw_timings_us"]["crt_export"] = [0 for _ in range(repeats)]
    chain["timing_summary_us"]["crt_export"] = {"avg": 0.0, "median": 0.0, "p95": 0.0}
    chain["avg_crt_export_us"] = 0.0
    add_target_variant_fields(chain)
    add_requested_next_op_fields(chain, resolved="rns-gemm")
    add_output_policy_fields(
        chain,
        status_handling="structurally_elided",
        per_repeat_export=False,
        final_checksum_export=True,
    )
    add_ck_chain_gpu_events(chain, 9)
    return chain


def as_hip_graph_replay_capture(capture: dict) -> dict:
    graph = as_reused_pack_capture(copy.deepcopy(capture))
    repeats = graph["repeats"]
    graph["benchmark"] = "rns8_hip_graph_replay_resident_rns_chain"
    graph["benchmark_execution_mode"] = "hip_graph_replay_resident_rns_chain"
    graph["command_line"] = (
        "rns8-bench --backend hip-direct --semantics bounded-i64 --m 64 --n 64 --k 64 "
        "--reuse-packed-inputs --residue-chain-length 3 --next-op-hint rns-gemm --hip-graph-replay "
        "--warmups 1 --repeats 3 --seed 7"
    )
    graph["m"] = 64
    graph["n"] = 64
    graph["k"] = 64
    graph["bound_kind"] = "global_max_abs"
    graph["bound_mode"] = "global"
    graph["bound_source"] = "static_profile"
    graph["bound"] = 1099511627776
    graph["tile_bounds_u64"] = None
    graph["schedule_metadata"]["tile_rows"] = 1
    graph["schedule_metadata"]["tile_cols"] = 1
    graph["schedule_metadata"]["tile_count"] = 1
    graph["schedule_metadata"]["min_required_prefix"] = 9
    graph["schedule_metadata"]["max_required_prefix"] = 9
    graph["schedule_metadata"]["min_selected_prefix"] = 9
    graph["schedule_metadata"]["max_selected_prefix"] = 9
    graph["schedule_metadata"]["prefix_group_count"] = 1
    graph["schedule_metadata"]["adaptive_prefix_active"] = False
    graph["schedule_metadata"]["adaptive_skip_active"] = False
    graph["schedule_metadata"]["adaptive_execution_applied"] = False
    graph["schedule_metadata"]["range_bit_length"] = 41
    graph["epilogue_type"] = "residue_current_rns_output"
    graph["residue_chain_length"] = 3
    graph["residue_output_mode"] = "residue_current_rns"
    graph["per_modulus_gemm_estimate_applicable"] = False
    graph["raw_timings_us"].pop("tile_bound_scan", None)
    graph["timing_summary_us"].pop("tile_bound_scan", None)
    graph["raw_timings_us"]["rns_gemm"] = [70, 71, 72][:repeats]
    graph["raw_timings_us"]["crt_export"] = [0 for _ in range(repeats)]
    graph["raw_timings_us"]["end_to_end"] = [70, 71, 72][:repeats]
    graph["timing_summary_us"]["rns_gemm"] = summary(graph["raw_timings_us"]["rns_gemm"])
    graph["timing_summary_us"]["crt_export"] = zero_summary()
    graph["timing_summary_us"]["end_to_end"] = summary(graph["raw_timings_us"]["end_to_end"])
    graph["avg_rns_gemm_us"] = graph["timing_summary_us"]["rns_gemm"]["avg"]
    graph["avg_crt_export_us"] = 0.0
    graph["avg_end_to_end_us"] = graph["timing_summary_us"]["end_to_end"]["avg"]
    graph["avg_per_modulus_gemm_estimate_us"] = graph["avg_rns_gemm_us"]
    graph["backend_metadata"]["autotune_key"] = with_accumulator_key_fields(
        (
        "backend=hip-direct;semantics=bounded_i64;m=64;n=64;k=64;prefix=9;tile_m=64;tile_n=64;"
        "groups=1;adaptive_prefix=0;adaptive_skip=0;kernel=direct_hip_tiled_active_prefix_rns_gemm_v2;"
        "epilogue=fused_centered_residue_then_crt_export"
        ),
        graph,
    )
    graph["timing_note"] = (
        "host wall-clock timings for benchmark-only HIP Graph replay of a Direct-HIP residue-current RNS GEMM chain"
    )
    metadata = graph["timing_metadata"]
    metadata["benchmark_execution_mode"] = "hip_graph_replay_resident_rns_chain"
    metadata["gpu_event_timing"] = False
    metadata["gpu_event_timing_reason"] = "hip_graph_replay_wall_clock_only"
    metadata["gpu_event_timing_status"] = "not_requested_graph_replay"
    metadata["gpu_event_timing_source"] = None
    metadata["gpu_event_timing_source_scope"] = None
    metadata["gpu_event_timing_caveat"] = None
    metadata["gpu_event_phase_order"] = None
    metadata["phase_order"] = ["planning", "scheduling", "matrix_alloc", "pack", "rns_gemm", "crt_export", "end_to_end"]
    metadata["phase_notes"]["pack"] = (
        "zero-valued per-repeat phase; A and B were packed once into persistent RNS matrices before HIP Graph capture"
    )
    metadata["phase_notes"]["rns_gemm"] = (
        "per-repeat host timing for one hipGraphLaunch plus stream synchronization containing 3 captured Direct-HIP "
        "resident RNS GEMM launches"
    )
    metadata["phase_notes"]["crt_export"] = (
        "zero-valued per-repeat phase; residue-current graph replay defers host logical export until one final "
        "checksum export after measured repeats"
    )
    metadata["phase_notes"]["end_to_end"] = (
        "same measured duration as rns_gemm for one hipGraphLaunch plus stream synchronization"
    )
    metadata["phase_availability"].pop("tile_bound_scan", None)
    metadata["hip_graph_replay_enabled"] = True
    metadata["hip_graph_replay_status"] = "available"
    metadata["hip_graph_replay_scope"] = "direct_hip_reused_inputs_residue_current_rns_chain"
    metadata["hip_graph_capture_us"] = 11
    metadata["hip_graph_instantiate_us"] = 17
    metadata["hip_graph_total_launches"] = graph["warmups"] + graph["repeats"]
    graph["gpu_event_timings_us"] = None
    graph["gpu_event_timing_summary_us"] = None
    graph["hip_graph_replay"] = {
        "requested": True,
        "available": True,
        "used": True,
        "status": "available",
        "scope": "direct_hip_reused_inputs_residue_current_rns_chain",
        "descriptor_identity": "fixed_plan_workspace_descriptor:m=64;n=64;k=64",
        "plan_identity": graph["backend_metadata"]["autotune_key"],
        "setup_scope": "benchmark_hip_graph_replay_resident_rns_chain",
        "capture_status": "replayed",
        "unsupported_reason": None,
        "promotion_eligible": False,
        "capture_us": 11,
        "instantiate_us": 17,
        "graph_launches_per_measured_repeat": 1,
        "total_graph_launches": graph["warmups"] + graph["repeats"],
        "captured_chain_length": 3,
        "timing_policy": "raw_timings_us.rns_gemm_and_end_to_end_measure_one_hipGraphLaunch_plus_stream_sync",
        "setup_policy": "A_B_prepack_before_capture_capture_and_instantiate_before_warmups",
        "final_export_policy": "one_final_logical_export_after_measured_repeats_for_checksum_only",
        "caveat": "captures resident Direct-HIP RNS GEMM launches only; A/B prepack setup and final checksum export are outside the graph",
    }
    add_target_variant_fields(graph)
    add_requested_next_op_fields(graph, resolved="rns-gemm", requested="rns-gemm", source="cli")
    add_output_policy_fields(
        graph,
        status_handling="structurally_elided",
        per_repeat_export=False,
        final_checksum_export=True,
    )
    return graph


def as_hip_graph_full_bounded_capture(capture: dict) -> dict:
    graph = copy.deepcopy(capture)
    repeats = graph["repeats"]
    graph["benchmark"] = "rns8_hip_graph_replay_bounded_pack_gemm_export"
    graph["benchmark_execution_mode"] = "hip_graph_replay_bounded_pack_gemm_export"
    graph["command_line"] = (
        "rns8-bench --backend hip-direct --semantics bounded-i64 --m 64 --n 64 --k 64 "
        "--hip-graph-replay --warmups 1 --repeats 3 --seed 7"
    )
    graph["backend_requested"] = "hip-direct"
    graph["backend_selected"] = "hip-direct"
    graph["semantics"] = "bounded_i64"
    graph["m"] = 64
    graph["n"] = 64
    graph["k"] = 64
    graph["bound_kind"] = "global_max_abs"
    graph["bound_mode"] = "global"
    graph["bound_source"] = "static_profile"
    graph["bound"] = 1099511627776
    graph["tile_bounds_u64"] = None
    graph["prefix"] = 9
    graph["selected_prefix"] = 9
    graph["requested_max_prefix"] = 9
    graph["contract_prefix_policy"] = "minimum_proven"
    graph["schedule_metadata"]["tile_rows"] = 1
    graph["schedule_metadata"]["tile_cols"] = 1
    graph["schedule_metadata"]["tile_count"] = 1
    graph["schedule_metadata"]["min_required_prefix"] = 9
    graph["schedule_metadata"]["max_required_prefix"] = 9
    graph["schedule_metadata"]["min_selected_prefix"] = 9
    graph["schedule_metadata"]["max_selected_prefix"] = 9
    graph["schedule_metadata"]["prefix_group_count"] = 1
    graph["schedule_metadata"]["adaptive_prefix_active"] = False
    graph["schedule_metadata"]["adaptive_skip_active"] = False
    graph["schedule_metadata"]["adaptive_execution_applied"] = False
    graph["schedule_metadata"]["range_bit_length"] = 41
    graph["reuse_packed_inputs"] = False
    graph["pack_mode"] = "per_repeat_repack"
    graph["prepack_reuse_operands"] = []
    graph["prepack_reuse_strategy"] = "none"
    graph["prepack_setup_us"] = None
    graph.pop("avg_prepack_setup_us", None)
    graph["epilogue_type"] = "crt_export"
    graph["residue_chain_length"] = 1
    graph["residue_output_mode"] = "host_export"
    graph["per_modulus_gemm_estimate_applicable"] = False
    graph["raw_timings_us"].pop("tile_bound_scan", None)
    graph["timing_summary_us"].pop("tile_bound_scan", None)
    graph["raw_timings_us"]["pack"] = [0 for _ in range(repeats)]
    graph["raw_timings_us"]["rns_gemm"] = [90, 91, 92][:repeats]
    graph["raw_timings_us"]["crt_export"] = [0 for _ in range(repeats)]
    graph["raw_timings_us"]["end_to_end"] = [90, 91, 92][:repeats]
    graph["timing_summary_us"]["pack"] = zero_summary()
    graph["timing_summary_us"]["rns_gemm"] = summary(graph["raw_timings_us"]["rns_gemm"])
    graph["timing_summary_us"]["crt_export"] = zero_summary()
    graph["timing_summary_us"]["end_to_end"] = summary(graph["raw_timings_us"]["end_to_end"])
    graph["avg_pack_us"] = 0.0
    graph["avg_rns_gemm_us"] = graph["timing_summary_us"]["rns_gemm"]["avg"]
    graph["avg_crt_export_us"] = 0.0
    graph["avg_end_to_end_us"] = graph["timing_summary_us"]["end_to_end"]["avg"]
    graph["avg_per_modulus_gemm_estimate_us"] = 0.0
    graph["backend_metadata"]["autotune_key"] = with_accumulator_key_fields(
        (
            "backend=hip-direct;semantics=bounded_i64;m=64;n=64;k=64;prefix=9;tile_m=64;tile_n=64;"
            "groups=1;adaptive_prefix=0;adaptive_skip=0;kernel=direct_hip_tiled_active_prefix_rns_gemm_v2;"
            "epilogue=fused_centered_residue_then_crt_export"
        ),
        graph,
    )
    add_prefix_policy_fields(graph, "minimum_proven")
    graph["timing_note"] = (
        "host wall-clock timings for benchmark-only HIP Graph replay of a Direct-HIP bounded full pack/GEMM/export path"
    )
    metadata = graph["timing_metadata"]
    metadata["benchmark_execution_mode"] = "hip_graph_replay_bounded_pack_gemm_export"
    metadata["pack_mode"] = "per_repeat_repack"
    metadata["prepack_reuse_operands"] = []
    metadata["prepack_reuse_strategy"] = "none"
    metadata["gpu_event_timing"] = False
    metadata["gpu_event_timing_reason"] = "hip_graph_replay_wall_clock_only"
    metadata["gpu_event_timing_status"] = "not_requested_graph_replay"
    metadata["gpu_event_timing_source"] = None
    metadata["gpu_event_timing_source_scope"] = None
    metadata["gpu_event_timing_caveat"] = None
    metadata["gpu_event_phase_order"] = None
    metadata["phase_order"] = ["planning", "scheduling", "matrix_alloc", "pack", "rns_gemm", "crt_export", "end_to_end"]
    metadata["phase_notes"]["pack"] = (
        "captured inside one HIP Graph launch; raw per-phase GPU events are unavailable for graph replay"
    )
    metadata["phase_notes"]["rns_gemm"] = (
        "captured inside one HIP Graph launch with the Direct-HIP bounded pack and export work"
    )
    metadata["phase_notes"]["crt_export"] = (
        "captured inside one HIP Graph launch including status memset, export kernel, status D2H, and output D2H"
    )
    metadata["phase_notes"]["end_to_end"] = (
        "measured duration for one hipGraphLaunch plus stream synchronization containing the complete bounded pack, "
        "RNS GEMM, export, status, and D2H path"
    )
    metadata["phase_availability"].pop("tile_bound_scan", None)
    metadata["hip_graph_replay_enabled"] = True
    metadata["hip_graph_replay_status"] = "available"
    metadata["hip_graph_replay_scope"] = "direct_hip_bounded_pack_gemm_export"
    metadata["hip_graph_capture_us"] = 13
    metadata["hip_graph_instantiate_us"] = 19
    metadata["hip_graph_total_launches"] = graph["warmups"] + graph["repeats"]
    graph["gpu_event_timings_us"] = None
    graph["gpu_event_timing_summary_us"] = None
    graph["hip_graph_replay"] = {
        "requested": True,
        "available": True,
        "used": True,
        "status": "available",
        "scope": "direct_hip_bounded_pack_gemm_export",
        "descriptor_identity": "fixed_plan_workspace_descriptor:m=64;n=64;k=64",
        "plan_identity": graph["backend_metadata"]["autotune_key"],
        "setup_scope": "benchmark_hip_graph_replay_bounded_pack_gemm_export",
        "capture_status": "replayed",
        "unsupported_reason": None,
        "promotion_eligible": False,
        "capture_us": 13,
        "instantiate_us": 19,
        "graph_launches_per_measured_repeat": 1,
        "total_graph_launches": graph["warmups"] + graph["repeats"],
        "captured_chain_length": 1,
        "timing_policy": "raw_timings_us.end_to_end_measure_one_full_pack_gemm_export_hipGraphLaunch_plus_stream_sync",
        "setup_policy": "capture_and_instantiate_before_warmups_no_prepack_reuse",
        "final_export_policy": "logical_export_and_output_d2h_captured_inside_graph_each_repeat",
        "caveat": (
            "captures Direct-HIP bounded pack, one RNS GEMM, export status, export kernel, and output D2H inside "
            "the graph"
        ),
    }
    add_target_variant_fields(graph)
    add_requested_next_op_fields(graph)
    add_output_policy_fields(graph, status_handling="required", per_repeat_export=True, final_checksum_export=False)
    return graph


def as_hip_graph_full_finite_capture(capture: dict) -> dict:
    graph = as_hip_graph_full_bounded_capture(capture)
    repeats = graph["repeats"]
    modulus = 251
    kernel = "direct_hip_tiled_finite_u8_gemm_mod251_v1"
    autotune_key_base = (
        "backend=hip-direct;semantics=finite_ring_u8;m=64;n=64;k=64;bound_kind=none;"
        "finite_modulus=251;prefix=0;tile_m=64;tile_n=64;groups=0;adaptive_prefix=0;"
        "adaptive_skip=0;kernel=direct_hip_tiled_finite_u8_gemm_mod251_v1;"
        "epilogue=fused_centered_residue_then_canonical_u8_export"
    )
    graph["benchmark"] = "rns8_hip_graph_replay_finite_u8_pack_gemm_export"
    graph["benchmark_execution_mode"] = "hip_graph_replay_finite_u8_pack_gemm_export"
    graph["command_line"] = (
        "rns8-bench --backend hip-direct --semantics finite-u8-ring --modulus 251 "
        "--m 64 --n 64 --k 64 --hip-graph-replay --warmups 1 --repeats 3 --seed 7"
    )
    graph["semantics"] = "finite_ring_u8"
    graph["bound_kind"] = "none"
    graph["bound"] = 0
    graph["input_distribution"] = "u8_uniform_0_modulus_minus_1"
    graph["finite_modulus"] = modulus
    graph["prefix"] = 0
    graph["selected_prefix"] = 0
    graph["requested_max_prefix"] = 0
    graph["contract_prefix_policy"] = "semantic_specific_no_rns_prefix"
    graph["residue_planes_requested"] = 0
    graph["residue_planes_selected"] = 0
    graph["residue_planes_skipped"] = 0
    graph["residue_plane_skip_fraction"] = 0.0
    graph["packed_layout_version"] = None
    graph["epilogue_type"] = "canonical_u8_export"
    graph["selected_kernel"] = kernel
    graph["backend_metadata"]["selected_kernel"] = kernel
    graph["backend_metadata"]["epilogue_mode"] = "fused_centered_residue_then_canonical_u8_export"
    graph["backend_metadata"]["workspace_mode"] = "resident_device_buffers"
    graph["backend_metadata"]["isa_evidence"] = "rns8_hip_direct_finite_specialized_reducer_isa_gate_no_divide"
    graph["backend_metadata"]["accumulator_safety"].update(
        {
            "input_domain": "centered_i8_finite_u8_residues",
            "modulus_policy": "finite_u8_modulus",
            "modulus": modulus,
            "k_block_size": 64,
            "k_block_cap": 65536,
            "safe_for_k_block": True,
            "status": "safe_int32_k_block_split",
        }
    )
    autotune_key = with_accumulator_key_fields(autotune_key_base, graph)
    graph["backend_metadata"]["autotune_key"] = autotune_key
    graph["schedule_metadata"].update(
        {
            "bound_kind": "none",
            "effective_bound": 0,
            "lhs_bound": 0,
            "rhs_bound": 0,
            "bound_contract": "finite_modulus",
            "min_required_prefix": 0,
            "max_required_prefix": 0,
            "min_selected_prefix": 0,
            "max_selected_prefix": 0,
            "prefix_group_count": 0,
            "range_bit_length": 0,
        }
    )
    graph["backend_metadata"]["autotune_key"] = autotune_key
    graph["timing_note"] = (
        "host wall-clock timings for benchmark-only HIP Graph replay of a Direct-HIP finite-u8 full pack/GEMM/export path"
    )
    graph["timing_metadata"]["benchmark_execution_mode"] = "hip_graph_replay_finite_u8_pack_gemm_export"
    graph["timing_metadata"]["hip_graph_replay_scope"] = "direct_hip_finite_u8_pack_gemm_export"
    graph["timing_metadata"]["phase_notes"]["rns_gemm"] = (
        "captured inside one HIP Graph launch with the Direct-HIP finite-u8 pack and canonical export work"
    )
    graph["timing_metadata"]["phase_notes"]["crt_export"] = (
        "captured inside one HIP Graph launch including finite export kernel and output D2H"
    )
    graph["timing_metadata"]["phase_notes"]["end_to_end"] = (
        "measured duration for one hipGraphLaunch plus stream synchronization containing the complete finite-u8 "
        "pack, modular GEMM, canonical export, and D2H path"
    )
    graph["hip_graph_replay"].update(
        {
            "scope": "direct_hip_finite_u8_pack_gemm_export",
            "plan_identity": autotune_key,
            "setup_scope": "benchmark_hip_graph_replay_finite_u8_pack_gemm_export",
            "timing_policy": (
                "raw_timings_us.end_to_end_measure_one_full_finite_pack_gemm_export_hipGraphLaunch_plus_stream_sync"
            ),
            "final_export_policy": "finite_canonical_export_and_output_d2h_captured_inside_graph_each_repeat",
            "caveat": (
                "captures Direct-HIP finite-u8 pack, one modular GEMM, canonical finite export, and output D2H "
                "inside the graph"
            ),
        }
    )
    graph["raw_timings_us"]["rns_gemm"] = [70, 71, 72][:repeats]
    graph["raw_timings_us"]["end_to_end"] = [70, 71, 72][:repeats]
    graph["timing_summary_us"]["rns_gemm"] = summary(graph["raw_timings_us"]["rns_gemm"])
    graph["timing_summary_us"]["end_to_end"] = summary(graph["raw_timings_us"]["end_to_end"])
    graph["avg_rns_gemm_us"] = graph["timing_summary_us"]["rns_gemm"]["avg"]
    graph["avg_end_to_end_us"] = graph["timing_summary_us"]["end_to_end"]["avg"]
    add_prefix_policy_fields(graph, "semantic_specific_no_rns_prefix")
    add_target_variant_fields(graph)
    add_requested_next_op_fields(graph)
    add_output_policy_fields(graph, status_handling="structurally_elided", per_repeat_export=True, final_checksum_export=False)
    return graph


def as_hip_graph_full_wrap64_capture(capture: dict) -> dict:
    graph = copy.deepcopy(capture)
    repeats = graph["repeats"]
    graph["benchmark"] = "rns8_hip_graph_replay_wrap64_pack_gemm_export"
    graph["benchmark_execution_mode"] = "hip_graph_replay_wrap64_pack_gemm_export"
    graph["command_line"] = (
        "rns8-bench --backend hip-direct --semantics wrap-u64 "
        "--m 4 --n 4 --k 8 --hip-graph-replay --warmups 1 --repeats 3 --seed 7"
    )
    graph["reuse_packed_inputs"] = False
    graph["pack_mode"] = "per_repeat_repack"
    graph["prepack_reuse_operands"] = []
    graph["prepack_reuse_strategy"] = "none"
    graph["prepack_setup_us"] = None
    graph.pop("avg_prepack_setup_us", None)
    graph["residue_chain_length"] = 1
    graph["residue_output_mode"] = "host_export"
    graph["output_logical_ld"] = graph["n"]
    graph["output_ld_padding"] = 0
    graph["raw_timings_us"]["pack"] = [0 for _ in range(repeats)]
    graph["raw_timings_us"]["rns_gemm"] = [80, 81, 82][:repeats]
    graph["raw_timings_us"]["crt_export"] = [0 for _ in range(repeats)]
    graph["raw_timings_us"]["end_to_end"] = [80, 81, 82][:repeats]
    graph["timing_summary_us"]["pack"] = zero_summary()
    graph["timing_summary_us"]["rns_gemm"] = summary(graph["raw_timings_us"]["rns_gemm"])
    graph["timing_summary_us"]["crt_export"] = zero_summary()
    graph["timing_summary_us"]["end_to_end"] = summary(graph["raw_timings_us"]["end_to_end"])
    graph["avg_pack_us"] = 0.0
    graph["avg_rns_gemm_us"] = graph["timing_summary_us"]["rns_gemm"]["avg"]
    graph["avg_crt_export_us"] = 0.0
    graph["avg_end_to_end_us"] = graph["timing_summary_us"]["end_to_end"]["avg"]
    graph["timing_note"] = (
        "host wall-clock timings for benchmark-only HIP Graph replay of a Direct-HIP wrap64 full pack/GEMM/export path"
    )
    metadata = graph["timing_metadata"]
    metadata["benchmark_execution_mode"] = "hip_graph_replay_wrap64_pack_gemm_export"
    metadata["pack_mode"] = "per_repeat_repack"
    metadata["prepack_reuse_operands"] = []
    metadata["prepack_reuse_strategy"] = "none"
    metadata["gpu_event_timing"] = False
    metadata["gpu_event_timing_reason"] = "hip_graph_replay_wall_clock_only"
    metadata["gpu_event_timing_status"] = "not_requested_graph_replay"
    metadata["gpu_event_timing_source"] = None
    metadata["gpu_event_timing_source_scope"] = None
    metadata["gpu_event_timing_caveat"] = None
    metadata["gpu_event_phase_order"] = None
    metadata["phase_order"] = ["planning", "scheduling", "matrix_alloc", "pack", "rns_gemm", "crt_export", "end_to_end"]
    metadata["phase_notes"]["pack"] = (
        "captured inside one HIP Graph launch; raw per-phase GPU events are unavailable for graph replay"
    )
    metadata["phase_notes"]["rns_gemm"] = (
        "captured inside one HIP Graph launch with Direct-HIP wrap64 byte-limb pack and low64 export work"
    )
    metadata["phase_notes"]["crt_export"] = (
        "captured inside one HIP Graph launch including wrap64 low64 export kernel and output D2H"
    )
    metadata["phase_notes"]["end_to_end"] = (
        "measured duration for one hipGraphLaunch plus stream synchronization containing the complete wrap64 "
        "byte-limb pack, byte-GEMM36, low64 export, and D2H path"
    )
    metadata["hip_graph_replay_enabled"] = True
    metadata["hip_graph_replay_status"] = "available"
    metadata["hip_graph_replay_scope"] = "direct_hip_wrap64_pack_gemm_export"
    metadata["hip_graph_capture_us"] = 17
    metadata["hip_graph_instantiate_us"] = 23
    metadata["hip_graph_total_launches"] = graph["warmups"] + graph["repeats"]
    graph["gpu_event_timings_us"] = None
    graph["gpu_event_timing_summary_us"] = None
    graph["hip_graph_replay"] = {
        "requested": True,
        "available": True,
        "used": True,
        "status": "available",
        "scope": "direct_hip_wrap64_pack_gemm_export",
        "descriptor_identity": "fixed_wrap64_byte_limb_descriptor:m=4;n=4;k=8",
        "plan_identity": graph["backend_metadata"]["autotune_key"],
        "setup_scope": "benchmark_hip_graph_replay_wrap64_pack_gemm_export",
        "capture_status": "replayed",
        "unsupported_reason": None,
        "promotion_eligible": False,
        "capture_us": 17,
        "instantiate_us": 23,
        "graph_launches_per_measured_repeat": 1,
        "total_graph_launches": graph["warmups"] + graph["repeats"],
        "captured_chain_length": 1,
        "timing_policy": "raw_timings_us.end_to_end_measure_one_full_wrap64_pack_gemm_export_hipGraphLaunch_plus_stream_sync",
        "setup_policy": "capture_and_instantiate_before_warmups_no_prepack_reuse",
        "final_export_policy": "wrap64_low64_export_and_output_d2h_captured_inside_graph_each_repeat",
        "caveat": (
            "captures Direct-HIP wrap64 byte-limb pack, byte-GEMM36, low64 export, and output D2H inside the graph"
        ),
    }
    add_target_variant_fields(graph)
    add_requested_next_op_fields(graph)
    add_output_policy_fields(graph, status_handling="not_applicable", per_repeat_export=True, final_checksum_export=False)
    return graph


def as_wrap64_rocwmma_candidate_capture(capture: dict) -> dict:
    candidate = copy.deepcopy(capture)
    repeats = candidate["repeats"]
    candidate["backend_requested"] = "rocwmma-wrap64-candidate"
    candidate["backend_selected"] = "rocwmma"
    candidate["selected_kernel"] = "rocwmma_wrap64_byte_gemm36_candidate_v0"
    candidate["tile_m"] = 16
    candidate["tile_n"] = 16
    candidate["command_line"] = (
        "rns8-bench --backend rocwmma-wrap64-candidate --semantics wrap-u64 "
        "--m 4 --n 4 --k 8 --tile-m 16 --tile-n 16"
    )
    candidate["backend_metadata"].update(
        {
            "source": "rns8_bench_wrap64_rocwmma_candidate",
            "selected_kernel": "rocwmma_wrap64_byte_gemm36_candidate_v0",
            "accelerator_backend": True,
            "correctness_backend": False,
            "matrix_engine_backend": True,
            "compiled_kernel_available": True,
            "exact_differential_validated": True,
            "performance_validated": False,
            "accelerator_library": "rocWMMA",
            "accelerator_version": "repo-local release/rocm-rel-7.1",
            "capability_status": "internal_wrap64_matrix_engine_candidate",
            "epilogue_mode": "low64_wrap_export",
            "workspace_mode": "benchmark_owned_compact_byte_limb_device_buffers",
            "workspace_required_bytes": 640,
            "isa_evidence": "rocwmma_wrap64_byte_gemm36_matrix_isa_gate_no_divide",
            "autotune_key": (
                "backend=rocwmma-wrap64-candidate;target_id=gfx1100;semantics=wrap_u64_mod_2_64;m=4;n=4;k=8;"
                "prefix=0;tile_m=16;tile_n=16;groups=0;adaptive_prefix=0;adaptive_skip=0;"
                "accumulator_type=int32_then_int64_diagonal;accumulator_signedness=unsigned_u8x_unsigned_u8;"
                "accumulator_modulus_policy=mod_2_64_wraparound_byte_limb;k_block_size=8;k_block_cap=32768;"
                "kernel=rocwmma_wrap64_byte_gemm36_candidate_v0;epilogue=low64_wrap_export"
            ),
            "accumulator_safety": {
                "input_domain": "compact_u8_byte_limb_pairs",
                "signedness": "unsigned_u8x_unsigned_u8",
                "accumulator_type": "int32_then_int64_diagonal",
                "modulus_policy": "mod_2_64_wraparound_byte_limb",
                "modulus": 0,
                "uses_int32_inner_product": True,
                "k_block_size": 8,
                "k_block_cap": 32768,
                "max_lhs_abs": 255,
                "max_rhs_abs": 255,
                "max_product": 255 * 255,
                "safe_for_k_block": True,
                "status": "safe_int32_byte_limb_gemm36_k_block",
            },
        }
    )
    candidate["k_block_size"] = 8
    candidate["schedule_metadata"].update(
        {
            "source": "rns8_bench_wrap64_rocwmma_candidate_static_schedule",
            "tile_m": 16,
            "tile_n": 16,
            "tile_rows": 1,
            "tile_cols": 1,
            "tile_count": 1,
        }
    )
    candidate["timing_note"] = (
        "host wall-clock timings for the internal rocWMMA wrap64 byte-GEMM36 candidate; GPU event timing uses "
        "direct-HIP byte-limb pack/export labels plus one candidate operation-group label and this path is not "
        "public or AUTO-selected"
    )
    candidate["timing_metadata"]["gpu_event_timing_reason"] = (
        "captured_by_internal_rocwmma_wrap64_candidate_hooks"
    )
    candidate["timing_metadata"]["gpu_event_timing_source_scope"] = (
        "rocwmma_wrap64_byte_gemm36_candidate_default_stream_operation_groups"
    )
    candidate["timing_metadata"]["gpu_event_timing_caveat"] = (
        "HIP event timings record direct-HIP byte-limb pack/export operation groups plus one internal rocWMMA "
        "wrap64 byte-GEMM36 candidate operation group; host wall-clock timings remain required for CPU scheduling "
        "overhead, API dispatch, allocations, and synchronous host-side overhead not represented on the HIP stream"
    )
    candidate["timing_metadata"]["phase_notes"].update(
        {
            "planning": "one-time benchmark-owned metadata initialization for the internal rocWMMA wrap64 candidate",
            "scheduling": "one-time fixed 16x16 WMMA candidate schedule derivation from the matrix shape",
            "matrix_alloc": "one-time benchmark-owned compact byte-limb HIP device buffer allocation host timing",
            "pack": "per-repeat host timing for direct-HIP packing of A and B into compact byte-limb device buffers",
            "rns_gemm": "per-repeat host timing for the internal rocWMMA wrap64 byte-GEMM36 candidate",
            "crt_export": "per-repeat host timing for direct-HIP low-64-bit byte-limb export",
        }
    )
    candidate["timing_metadata"]["phase_availability"]["scheduling"] = {
        "timed": True,
        "timing_key": "scheduling",
        "scope": "benchmark_static_wrap64_rocwmma_candidate_schedule",
        "reason": "measured with host steady_clock around fixed 16x16 candidate schedule metadata initialization",
    }
    renamed = {
        "wrap64_byte_gemm36_tiled_2d_kernel": "wrap64_rocwmma_candidate_gemm36_kernel_group",
    }
    phase_order = candidate["timing_metadata"].get("gpu_event_phase_order")
    if isinstance(phase_order, list):
        candidate["timing_metadata"]["gpu_event_phase_order"] = [renamed.get(item, item) for item in phase_order]
    for field in ["gpu_event_timings_us", "gpu_event_timing_summary_us"]:
        values = candidate.get(field)
        if isinstance(values, dict):
            for old, new in renamed.items():
                if old in values:
                    values[new] = values.pop(old)
    assert len(candidate["gpu_event_timings_us"]["wrap64_rocwmma_candidate_gemm36_kernel_group"]) == repeats
    return candidate


def as_large_wrap64_colpair_capture(capture: dict) -> dict:
    colpair = copy.deepcopy(capture)
    old_kernel = "direct_hip_wrap64_byte_gemm36_u32acc_tiled_2d_v4"
    new_kernel = "direct_hip_wrap64_byte_gemm36_u32acc_colpair_2d_v5"
    old_event = "wrap64_byte_gemm36_tiled_2d_kernel"
    new_event = "wrap64_byte_gemm36_colpair_2d_kernel"
    colpair["m"] = 256
    colpair["n"] = 256
    colpair["k"] = 256
    colpair["k_block_size"] = 256
    colpair["selected_kernel"] = new_kernel
    colpair["command_line"] = (
        "rns8-bench --backend hip-direct --semantics wrap-u64 --m 256 --n 256 --k 256 "
        "--warmups 1 --repeats 2 --seed 11"
    )
    colpair["backend_metadata"]["selected_kernel"] = new_kernel
    colpair["backend_metadata"]["accumulator_safety"]["k_block_size"] = 256
    colpair["backend_metadata"]["autotune_key"] = (
        colpair["backend_metadata"]["autotune_key"]
        .replace(";m=4;", ";m=256;")
        .replace(";n=4;", ";n=256;")
        .replace(";k=8;", ";k=256;")
        .replace(";k_block_size=8;", ";k_block_size=256;")
        .replace(f"kernel={old_kernel}", f"kernel={new_kernel}")
    )
    phase_order = colpair["timing_metadata"].get("gpu_event_phase_order")
    if isinstance(phase_order, list):
        colpair["timing_metadata"]["gpu_event_phase_order"] = [
            new_event if item == old_event else item for item in phase_order
        ]
    for field in ["gpu_event_timings_us", "gpu_event_timing_summary_us"]:
        values = colpair.get(field)
        if isinstance(values, dict) and old_event in values:
            values[new_event] = values.pop(old_event)
    return colpair


