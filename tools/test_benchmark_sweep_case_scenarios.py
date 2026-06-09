parsed = benchmark_sweep.parse_case("rect:64,128,256")
assert parsed.name == "rect"
assert (parsed.m, parsed.n, parsed.k) == (64, 128, 256)
adaptive = benchmark_sweep.parse_case("adaptive:65,65,64,64,64", adaptive=True)
assert adaptive.bound_mode == "per-tile"
assert adaptive.input_profile == "uniform-small"
assert adaptive.require_adaptive is True
adaptive_profile = benchmark_sweep.parse_case("adaptive-bands:256,256,512,64,64,adaptive-bands", adaptive=True)
assert adaptive_profile.input_profile == "adaptive-bands"
assert benchmark_sweep.backend_allowed_for("wrap-u64", parsed, "wrap64-byte-limb") is True
assert benchmark_sweep.backend_allowed_for("bounded-i64", parsed, "wrap64-byte-limb") is False
assert benchmark_sweep.backend_allowed_for("finite-u8-ring", parsed, "hip-vector-alu-int64") is False
assert benchmark_sweep.backend_allowed_for("exact-wide-signed", parsed, "ck") is True
assert benchmark_sweep.backend_allowed_for("exact-wide-unsigned", parsed, "hip-vector-alu-int64") is False
assert benchmark_sweep.backend_allowed_for("exact-wide-signed", adaptive, "ck") is False
assert benchmark_sweep.backend_allowed_for("bounded-u64", adaptive, "hipblaslt") is False
assert benchmark_sweep.backend_allowed_for("bounded-i64", adaptive, "amdgpu-builtins") is False
assert benchmark_sweep.backend_allowed_for("bounded-u64", adaptive, "amdgpu-builtins") is False
assert benchmark_sweep.backend_allowed_for("bounded-u64", parsed, "ck") is False
assert "ck" not in benchmark_sweep.default_backends_for("bounded-u64", parsed)
assert benchmark_sweep.cli_backend("rocwmma") == "rocwmma"
assert benchmark_sweep.cli_backend("hip-vector-alu-int64") == "hip-vector-alu-int64-runtime"
assert benchmark_sweep.cli_backend("hip-direct") == "hip-direct"

catalog = benchmark_sweep.scenario_catalog()
scenario_files = sorted(benchmark_sweep.SCENARIO_DATA_DIR.glob("*.json"))
assert len(scenario_files) == len(catalog)
loaded_families = set(catalog)
for path in scenario_files:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["family"] in loaded_families
    assert payload["items"]
    for item in payload["items"]:
        assert isinstance(item.get("case"), dict)
        assert isinstance(item.get("backends"), list) and item["backends"]
        assert isinstance(item.get("review_mode_expectation"), str) and item["review_mode_expectation"]
        assert item["review_mode_expectation"] in {"smoke", "release"}
        assert isinstance(item.get("promotion_eligibility"), str) and item["promotion_eligibility"]
        for key in ["name", "m", "n", "k", "tile_m", "tile_n", "bound_mode", "input_profile"]:
            assert key in item["case"]
release_candidate_items = benchmark_sweep.selected_scenario_items(
    argparse.Namespace(scenario=["release-candidates"])
)
assert release_candidate_items
assert {
    item.promotion_eligibility for item in release_candidate_items
} == {"release_review_candidate"}
all_scenario_items = benchmark_sweep.selected_scenario_items(argparse.Namespace(scenario=["all"]))
assert len(all_scenario_items) > len(release_candidate_items)
assert any(item.promotion_eligibility != "release_review_candidate" for item in all_scenario_items)
try:
    benchmark_sweep.selected_scenario_items(argparse.Namespace(scenario=["release-candidates", "repeated-b"]))
except SystemExit as exc:
    assert "cannot be combined" in str(exc)
else:
    raise AssertionError("expected release-candidates selector combination to fail validation")
repeated_b_items = catalog["repeated-b"]
assert repeated_b_items
assert {item.promotion_eligibility for item in repeated_b_items} == {
    "release_review_candidate",
    "reuse_contract_evidence_only",
}
assert {item.pack_mode for item in repeated_b_items} == {"per_repeat_repack", "prepacked_reuse_b"}
production_reuse_items = [
    item
    for item in repeated_b_items
    if item.promotion_eligibility == "release_review_candidate" and item.pack_mode == "prepacked_reuse_b"
]
assert production_reuse_items
assert {item.semantics for item in production_reuse_items} == {"bounded-i64", "bounded-u64"}
assert {item.name for item in production_reuse_items} == {
    "bounded-i64-512-production-reuse-b",
    "bounded-i64-1024-production-reuse-b",
    "bounded-u64-512-production-reuse-b",
    "bounded-u64-1024-production-reuse-b",
}
for item in production_reuse_items:
    assert item.backends == ("rocwmma",)
    assert item.metadata and item.metadata["reuse_contract_role"] == "stable_b_production_candidate"
baseline_items = [
    item
    for item in repeated_b_items
    if item.promotion_eligibility == "release_review_candidate" and item.pack_mode == "per_repeat_repack"
]
assert baseline_items
assert {item.semantics for item in baseline_items} == {"bounded-i64", "bounded-u64"}
assert {item.name for item in baseline_items} == {
    "bounded-i64-512-production-baselines",
    "bounded-i64-1024-production-baselines",
    "bounded-u64-512-production-baselines",
    "bounded-u64-1024-production-baselines",
}
for item in baseline_items:
    assert {"cpu", "hip-direct", "hip-vector-alu-int64", "rocwmma"}.issubset(set(item.backends))
for item in repeated_b_items:
    if item.promotion_eligibility == "reuse_contract_evidence_only":
        assert item.metadata and item.metadata["promotion_scope"] == "reuse_contract_evidence_only"

skinny_items = catalog["skinny-gemv"]
assert skinny_items
assert {item.case.n for item in skinny_items} == {1, 4, 8}
skinny_release_items = [item for item in skinny_items if item.promotion_eligibility == "release_review_candidate"]
skinny_control_items = [item for item in skinny_items if item.promotion_eligibility == "tile_shape_evidence_only"]
assert {item.review_mode_expectation for item in skinny_release_items} == {"release"}
assert {item.review_mode_expectation for item in skinny_control_items} == {"smoke"}
small_n_items = [item for item in skinny_items if item.case.n in {4, 8}]
small_n_release_items = [item for item in small_n_items if item.promotion_eligibility == "release_review_candidate"]
small_n_control_items = [item for item in small_n_items if item.promotion_eligibility == "tile_shape_evidence_only"]
assert {item.name for item in small_n_release_items} == {
    "bounded-i64-n4-512",
    "bounded-u64-n4-1024",
    "bounded-i64-n8-512",
    "bounded-u64-n8-1024",
}
assert {item.name for item in small_n_control_items} == {
    "bounded-i64-n4-512-tiled-control",
    "bounded-u64-n4-1024-tiled-control",
    "bounded-i64-n8-512-tiled-control",
    "bounded-u64-n8-1024-tiled-control",
}
for item in small_n_release_items:
    assert item.metadata and item.metadata["workflow_name"] == "gemv_small_n"
    assert item.metadata["optimization_status"] == "needs_measured_small_n_kernel_decision"
    assert "hip-vector-alu-int64" not in item.backends
for item in small_n_control_items:
    assert item.metadata and item.metadata["workflow_name"] == "skinny_tiled_control"
    assert item.backends == ("hip-direct",)
    assert item.tile_shape_variant == "direct-hip-skinny-tiled-control-128x128"

multi_modulus_items = catalog["multi-modulus-pack"]
assert multi_modulus_items
assert {item.promotion_eligibility for item in multi_modulus_items} == {"execution_path_evidence"}
assert {item.review_mode_expectation for item in multi_modulus_items} == {"release"}
assert {item.max_prefix for item in multi_modulus_items} == {3, 5, 9, 20}
for item in multi_modulus_items:
    assert item.metadata and item.metadata["promotion_scope"] == "execution_path_evidence"
    assert (
        item.metadata["cache_promotion_blocker"]
        == "pack_prefix_sweep_requires_same_contract_release_baselines"
    )

native_bridge_items = catalog["native-to-rns-bridge"]
assert native_bridge_items
assert {item.review_mode_expectation for item in native_bridge_items} == {"release"}
assert {item.promotion_eligibility for item in native_bridge_items} == {"execution_path_evidence"}
assert {item.backends for item in native_bridge_items} == {("auto",)}
assert {item.semantics for item in native_bridge_items} == {"bounded-i64", "bounded-u64"}
assert {item.case.m for item in native_bridge_items} == {64, 128}
for item in native_bridge_items:
    assert item.native_to_rns_bridge is True
    assert item.metadata and item.metadata["selected_backend_requirement"] == "hip-direct"

exact_export_items = catalog["exact-wide-export"]
assert {
    item.name for item in exact_export_items if item.review_mode_expectation == "release"
} == {
    "signed-limbs4-512",
    "unsigned-limbs4-512",
    "signed-limbs4-1024",
    "unsigned-limbs4-1024",
}
assert {
    item.case.m for item in exact_export_items if item.review_mode_expectation == "release"
} == {512, 1024}
assert {
    item.case.m for item in exact_export_items if item.review_mode_expectation == "smoke"
} == {64, 128, 2048}
assert {item.review_mode_expectation for item in catalog["export-bound-limb-variants"]} == {"release"}
assert {item.review_mode_expectation for item in catalog["reconstruction-zoo"]} == {"release"}
assert {item.review_mode_expectation for item in catalog["fused-pack-gemm-small"]} == {"release"}

bounded_rns_chain_release_items = [
    item
    for item in catalog["rns-chain"]
    if item.semantics in {"bounded-i64", "bounded-u64"}
    and item.promotion_eligibility == "release_review_candidate"
]
assert bounded_rns_chain_release_items
for item in bounded_rns_chain_release_items:
    assert item.residue_chain_length > 1
    assert item.pack_mode == "per_repeat_repack"
    assert "cpu" in item.backends, f"{item.name} must include a CPU correctness anchor"
    assert "hip-direct" in item.backends
    assert "hip-vector-alu-int64" not in item.backends

bounded_rns_chain_reuse_items = [
    item
    for item in catalog["rns-chain"]
    if item.semantics in {"bounded-i64", "bounded-u64"} and item.pack_mode != "per_repeat_repack"
]
assert bounded_rns_chain_reuse_items
for item in bounded_rns_chain_reuse_items:
    assert item.promotion_eligibility == "reuse_contract_evidence_only"
    assert item.metadata and item.metadata["promotion_scope"] == "reuse_contract_evidence_only"
assert {item.review_mode_expectation for item in catalog["rns-chain"]} == {"release"}
assert {item.review_mode_expectation for item in catalog["rns-chain-final-output"]} == {"release"}
for scenario_name in [
    "bound-discovery",
    "generated-prefix-reducers",
    "large-release-validation",
    "layout-search",
    "multi-modulus-pack",
    "residue-channel-fusion",
    "fused-pack-gemm-small",
    "large-release-validation-4096-budgeted",
    "hipblaslt-bounded-i64-1024-ab",
    "finite-modulus-map",
    "modulus-set-autotune",
    "tile-shape-sweeps",
    "exact-wide-output-chain",
    "exact-wide-output-chain-broader",
    "export-bound-limb-variants",
    "reconstruction-zoo",
    "hip-graph-replay",
    "finite-distributions",
    "rns-chain-final-output",
    "rns-chain-final-output-broader",
    "grouped-dispatch",
    "resident-lifetime-arena",
    "adaptive-grouped-scheduler",
    "streaming-overlap",
    "release-gate-closeout",
    "fhe-lattice-proxy-starfoundry",
    "cpu-small-shape-selector",
    "incremental-result-cache",
    "error-detecting-fast-path",
    "sparse-a-4-to-2",
]:
    assert scenario_name in catalog
    assert catalog[scenario_name]

cpu_selector_policies_with_cpu = {
    item.cpu_small_shape_selector
    for items in catalog.values()
    for item in items
    if item.cpu_small_shape_selector != "none" and "cpu" in item.backends
}
result_cache_groups_with_cpu = {
    (
        item.metadata.get("result_cache_contract_group", item.name)
        if isinstance(item.metadata, dict)
        else item.name
    )
    for items in catalog.values()
    for item in items
    if item.incremental_result_cache != "none" and item.backends and "cpu" in item.backends
}

for family, items in catalog.items():
    for item in items:
        if item.verification_amortization != "none":
            assert "cpu" in item.backends, f"{family}/{item.name} verification amortization requires CPU baseline"
        if item.error_detection_policy != "none":
            assert "cpu" in item.backends, f"{family}/{item.name} error detection policy requires CPU baseline"
        if item.cpu_small_shape_selector != "none":
            assert (
                item.cpu_small_shape_selector in cpu_selector_policies_with_cpu
            ), f"{family}/{item.name} CPU selector review requires paired CPU baseline"
            assert item.promotion_eligibility == "cpu_selector_threshold_evidence_only"
        if item.incremental_result_cache != "none":
            result_cache_group = (
                item.metadata.get("result_cache_contract_group", item.name)
                if isinstance(item.metadata, dict)
                else item.name
            )
            assert (
                result_cache_group in result_cache_groups_with_cpu
            ), f"{family}/{item.name} result-cache review group requires CPU baseline"
            assert item.promotion_eligibility in {
                "result_cache_research_only",
                "result_cache_contract_candidate",
            }

with tempfile.TemporaryDirectory() as temp_dir:
    scenario_path = Path(temp_dir) / "bad_scenario.json"
    payload = json.loads((benchmark_sweep.SCENARIO_DATA_DIR / "grouped_dispatch.json").read_text(encoding="utf-8"))
    payload["items"][0]["review_mode_expectation"] = "unregistered_review_mode"
    scenario_path.write_text(json.dumps(payload), encoding="utf-8")
    try:
        benchmark_sweep.load_scenario_data_family(scenario_path)
    except SystemExit as exc:
        assert "registered scenario review mode" in str(exc)
    else:
        raise AssertionError("expected stale scenario review mode to fail validation")

    payload = json.loads((benchmark_sweep.SCENARIO_DATA_DIR / "grouped_dispatch.json").read_text(encoding="utf-8"))
    payload["items"][0]["promotion_eligibility"] = "unregistered_promotion_scope"
    scenario_path.write_text(json.dumps(payload), encoding="utf-8")
    try:
        benchmark_sweep.load_scenario_data_family(scenario_path)
    except SystemExit as exc:
        assert "registered promotion scope" in str(exc)
    else:
        raise AssertionError("expected stale scenario promotion scope to fail validation")

    payload = json.loads((benchmark_sweep.SCENARIO_DATA_DIR / "generated_prefix_reducers.json").read_text(encoding="utf-8"))
    bad_item = copy.deepcopy(payload["items"][0])
    bad_item["name"] = "bounded-i64-invalid-prefix1"
    bad_item["max_prefix"] = 1
    bad_item["prefix_policy"] = "fixed-requested"
    bad_item["output_domain"] = "host_export"
    bad_item["next_op_hint"] = "final-export"
    payload["items"] = [bad_item]
    scenario_path.write_text(json.dumps(payload), encoding="utf-8")
    try:
        benchmark_sweep.load_scenario_data_family(scenario_path)
    except SystemExit as exc:
        assert "cannot host-export bounded-i64 uniform-small" in str(exc)
    else:
        raise AssertionError("expected invalid fixed-prefix host-export scenario to fail validation")

    payload = json.loads((benchmark_sweep.SCENARIO_DATA_DIR / "sparse_a_4_to_2.json").read_text(encoding="utf-8"))
    bad_item = copy.deepcopy(payload["items"][0])
    bad_item["case"]["k"] = 130
    payload["items"] = [bad_item]
    scenario_path.write_text(json.dumps(payload), encoding="utf-8")
    try:
        benchmark_sweep.load_scenario_data_family(scenario_path)
    except SystemExit as exc:
        assert "sparse_a_4_to_2 requires K divisible by 4" in str(exc)
    else:
        raise AssertionError("expected invalid sparse-A K shape to fail validation")

    bounded_item = copy.deepcopy(payload["items"][0])
    bounded_item["case"]["k"] = 128
    bounded_item["semantics"] = "bounded-i64"
    bounded_item["name"] = "bounded-i64-sparse-a-valid"
    bounded_item["finite_moduli"] = [None]
    bounded_item["backends"] = ["cpu", "amdgpu-builtins"]
    payload["items"] = [bounded_item]
    scenario_path.write_text(json.dumps(payload), encoding="utf-8")
    loaded_bounded_sparse = benchmark_sweep.load_scenario_data_family(scenario_path)
    assert loaded_bounded_sparse[0].semantics == "bounded-i64"
    assert loaded_bounded_sparse[0].sparse_a_4_to_2 is True

    exact_wide_item = copy.deepcopy(bounded_item)
    exact_wide_item["semantics"] = "exact-wide-signed"
    exact_wide_item["name"] = "exact-wide-sparse-a-valid"
    exact_wide_item["exact_wide_limb_counts"] = [4]
    payload["items"] = [exact_wide_item]
    scenario_path.write_text(json.dumps(payload), encoding="utf-8")
    loaded_exact_wide_sparse = benchmark_sweep.load_scenario_data_family(scenario_path)
    assert loaded_exact_wide_sparse[0].semantics == "exact-wide-signed"
    assert loaded_exact_wide_sparse[0].sparse_a_4_to_2 is True

    wrap_item = copy.deepcopy(bounded_item)
    wrap_item["semantics"] = "wrap-u64"
    wrap_item["name"] = "wrap64-sparse-a-invalid"
    payload["items"] = [wrap_item]
    scenario_path.write_text(json.dumps(payload), encoding="utf-8")
    try:
        benchmark_sweep.load_scenario_data_family(scenario_path)
    except SystemExit as exc:
        assert "sparse_a_4_to_2 requires finite-u8, bounded RNS, or exact-wide RNS semantics" in str(exc)
    else:
        raise AssertionError("expected wrap64 sparse-A scenario to fail validation")

    bad_item = copy.deepcopy(payload["items"][0])
    bad_item["sparse_a_4_to_2"] = False
    bad_item["sparse_a_4_to_2_dense_baseline"] = True
    payload["items"] = [bad_item]
    scenario_path.write_text(json.dumps(payload), encoding="utf-8")
    try:
        benchmark_sweep.load_scenario_data_family(scenario_path)
    except SystemExit as exc:
        assert "sparse_a_4_to_2_dense_baseline requires sparse_a_4_to_2=true" in str(exc)
    else:
        raise AssertionError("expected sparse-A dense baseline without sparse input to fail validation")

fusion_item = catalog["residue-channel-fusion"][0]
assert fusion_item.residue_channel_fusion is True
assert fusion_item.next_op_hint == "final-export"
fusion_items = catalog["residue-channel-fusion"]
assert {item.review_mode_expectation for item in fusion_items} == {"release"}
assert {item.promotion_eligibility for item in fusion_items} == {"release_review_candidate"}
assert {item.max_prefix for item in fusion_items} == {9}
prefix_reducer_items = catalog["generated-prefix-reducers"]
assert {item.review_mode_expectation for item in prefix_reducer_items} == {"release"}
assert {item.promotion_eligibility for item in prefix_reducer_items} == {"release_review_candidate"}
assert {item.max_prefix for item in prefix_reducer_items} == {3, 5, 9, 20}
assert all(item.max_prefix and item.max_prefix > 1 for item in prefix_reducer_items)
assert all(item.prefix_policy == "fixed-requested" for item in prefix_reducer_items)
scenario_base_args = argparse.Namespace(
    warmups=1,
    repeats=2,
    seed=11,
    reuse_packed_inputs=False,
    reuse_packed_a=False,
    reuse_packed_b=False,
    residue_chain_length=1,
    residue_chain_final_export=False,
    output_ld_padding=0,
    prefix_policy=None,
    max_prefix=None,
    bound_source=None,
    next_op_hint=None,
    residue_channel_fusion=False,
)
backend_filter_args = copy.copy(scenario_base_args)
backend_filter_args.backends = ["cpu", "hip-direct", "amdgpu-builtins"]
backend_filter_args.include_wrap64_rocwmma_candidate = False
ordinary_requested_item = next(
    item
    for item in catalog["adaptive-bands"]
    if item.name == "bounded-i64-256"
)
adaptive_band_items = catalog["adaptive-bands"]
assert {item.review_mode_expectation for item in adaptive_band_items} == {"release"}
assert {item.promotion_eligibility for item in adaptive_band_items} == {"release_review_candidate"}
assert {item.bound_source for item in adaptive_band_items} == {"input-scan"}
assert all(item.case.require_adaptive for item in adaptive_band_items)
computational_algebra_items = catalog["computational-algebra-proxies"]
assert {item.review_mode_expectation for item in computational_algebra_items} == {"release"}
assert {item.promotion_eligibility for item in computational_algebra_items} == {"release_review_candidate"}
assert {item.metadata["source_role"] for item in computational_algebra_items} == {"computational_algebra_proxy"}
assert {item.semantics for item in computational_algebra_items} == {"finite-u8-field", "exact-wide-signed"}
ordinary_backends = benchmark_sweep.scenario_backends_for_item(backend_filter_args, ordinary_requested_item)
assert ordinary_backends == ["cpu", "hip-direct"]
locked_requested_item = next(item for item in catalog["hip-graph-replay"] if item.hip_graph_replay)
locked_backends = benchmark_sweep.scenario_backends_for_item(backend_filter_args, locked_requested_item)
assert locked_backends == ["hip-direct"]
fusion_args = benchmark_sweep.scenario_args_for_item(scenario_base_args, fusion_item)
fusion_command = benchmark_sweep.command_for(
    Path("rns8-bench"),
    "hip-direct",
    fusion_item.semantics,
    fusion_item.case,
    None,
    None,
    fusion_args,
)
assert "--residue-channel-fusion" in fusion_command
assert "--next-op-hint" in fusion_command and "final-export" in fusion_command
assert "--prefix-policy" in fusion_command and "fixed-requested" in fusion_command
assert "--max-prefix" in fusion_command and "9" in fusion_command
modulus_item = catalog["modulus-set-autotune"][0]
modulus_args = benchmark_sweep.scenario_args_for_item(scenario_base_args, modulus_item)
modulus_command = benchmark_sweep.command_for(
    Path("rns8-bench"),
    "hip-direct",
    modulus_item.semantics,
    modulus_item.case,
    None,
    None,
    modulus_args,
)
assert "--modulus-set" in modulus_command and "experimental:prefix5-byte-ladder-search" in modulus_command
tile_item = next(item for item in catalog["tile-shape-sweeps"] if item.tile_shape_variant == "direct-hip-bounded-512-64x64")
tile_args = benchmark_sweep.scenario_args_for_item(scenario_base_args, tile_item)
tile_command = benchmark_sweep.command_for(
    Path("rns8-bench"),
    "hip-direct",
    tile_item.semantics,
    tile_item.case,
    None,
    None,
    tile_args,
)
assert "--tile-shape-variant" in tile_command and "direct-hip-bounded-512-64x64" in tile_command
amdgpu_mfma_tile_items = [
    item
    for item in catalog["tile-shape-sweeps"]
    if item.tile_shape_variant.startswith("amdgpu-cdna3-mfma-")
]
assert {item.tile_shape_variant for item in amdgpu_mfma_tile_items} == {
    "amdgpu-cdna3-mfma-16x16x32",
    "amdgpu-cdna3-mfma-32x32x16",
}
assert {
    (item.tile_shape_variant, item.semantics)
    for item in amdgpu_mfma_tile_items
} == {
    ("amdgpu-cdna3-mfma-16x16x32", "bounded-i64"),
    ("amdgpu-cdna3-mfma-32x32x16", "bounded-i64"),
    ("amdgpu-cdna3-mfma-16x16x32", "bounded-u64"),
    ("amdgpu-cdna3-mfma-32x32x16", "bounded-u64"),
    ("amdgpu-cdna3-mfma-16x16x32", "exact-wide-signed"),
    ("amdgpu-cdna3-mfma-32x32x16", "exact-wide-signed"),
    ("amdgpu-cdna3-mfma-16x16x32", "exact-wide-unsigned"),
    ("amdgpu-cdna3-mfma-32x32x16", "exact-wide-unsigned"),
    ("amdgpu-cdna3-mfma-16x16x32", "finite-u8-ring"),
    ("amdgpu-cdna3-mfma-32x32x16", "finite-u8-ring"),
}
for item in amdgpu_mfma_tile_items:
    assert item.backends == ("amdgpu-builtins",)
    assert item.promotion_eligibility == "tile_shape_evidence_only"
    assert item.metadata and item.metadata["resource_report_required"] == "mfma_isa_histogram_and_phase_timings"
graph_item = next(item for item in catalog["hip-graph-replay"] if item.hip_graph_replay)
assert len(catalog["hip-graph-replay"]) == 40
assert {item.review_mode_expectation for item in catalog["hip-graph-replay"]} == {"release"}
assert {item.promotion_eligibility for item in catalog["hip-graph-replay"]} == {"hip_graph_replay_evidence_only"}
assert {item.case.m for item in catalog["hip-graph-replay"]} == {512, 1024}
exact_graph_items = [
    item
    for item in catalog["hip-graph-replay"]
    if item.hip_graph_replay
    and item.semantics in {"exact-wide-signed", "exact-wide-unsigned"}
    and "full-pack-export" in item.name
]
assert {item.semantics for item in exact_graph_items} == {"exact-wide-signed", "exact-wide-unsigned"}
for item in exact_graph_items:
    assert item.output_domain == "host_export"
    assert item.exact_wide_limb_counts == (4,)
    assert item.case.m == 512 and item.case.n == 512 and item.case.k == 512
graph_args = benchmark_sweep.scenario_args_for_item(scenario_base_args, graph_item)
graph_command = benchmark_sweep.command_for(
    Path("rns8-bench"),
    "hip-direct",
    graph_item.semantics,
    graph_item.case,
    None,
    None,
    graph_args,
)
assert "--hip-graph-replay" in graph_command
grouped_items = catalog["grouped-dispatch"]
assert {item.review_mode_expectation for item in grouped_items} == {"release"}
assert {item.promotion_eligibility for item in grouped_items} == {"grouped_dispatch_evidence_only"}
assert any(item.semantics == "finite-u8-ring" for item in grouped_items)
assert any(item.semantics == "finite-u8-field" for item in grouped_items)
many_small_items = catalog["many-small"]
assert many_small_items
assert {item.review_mode_expectation for item in many_small_items} == {"release"}
assert {item.promotion_eligibility for item in many_small_items} == {"release_review_candidate"}
assert any(item.host_api_batch_size > 1 for item in many_small_items)
assert any(item.oneshot for item in many_small_items)
small_oneshot_items = catalog["small-oneshot"]
assert small_oneshot_items
assert {item.review_mode_expectation for item in small_oneshot_items} == {"release"}
assert {item.promotion_eligibility for item in small_oneshot_items} == {"release_review_candidate"}
assert any(item.oneshot for item in small_oneshot_items)
assert any(not item.oneshot for item in small_oneshot_items)
grouped_item = grouped_items[0]
grouped_args = benchmark_sweep.scenario_args_for_item(scenario_base_args, grouped_item)
grouped_command = benchmark_sweep.command_for(
    Path("rns8-bench"),
    "hip-direct",
    grouped_item.semantics,
    grouped_item.case,
    None,
    None,
    grouped_args,
)
assert "--grouped-dispatch" in grouped_command and "32" in grouped_command
resident_items = catalog["resident-lifetime-arena"]
assert {item.promotion_eligibility for item in resident_items} == {"resident_lifetime_arena_evidence_only"}
assert {item.semantics for item in resident_items} == {
    "bounded-i64",
    "bounded-u64",
    "exact-wide-signed",
    "exact-wide-unsigned",
}
assert {
    item.name for item in resident_items if item.semantics == "bounded-u64"
} == {
    "bounded-u64-512-reuse-b-arena",
    "bounded-u64-1024-reuse-b-arena",
    "bounded-u64-2048-arena",
}
assert {
    item.name for item in resident_items if item.semantics == "exact-wide-unsigned"
} == {
    "exact-wide-unsigned-chain3-arena",
    "exact-wide-unsigned-chain3-1024-arena",
}
for item in resident_items:
    assert item.resident_lifetime is True
    assert item.workspace_arena is True
resident_item = resident_items[0]
resident_args = benchmark_sweep.scenario_args_for_item(scenario_base_args, resident_item)
resident_command = benchmark_sweep.command_for(
    Path("rns8-bench"),
    "hip-direct",
    resident_item.semantics,
    resident_item.case,
    None,
    None,
    resident_args,
)
assert "--resident-lifetime" in resident_command
assert "--workspace-arena" in resident_command
resident_redesign_item = benchmark_sweep.ScenarioItem(
    "direct-hip-resident-redesign",
    "grouped-active-schedule",
    "bounded-i64",
    benchmark_sweep.SweepCase(
        "adaptive-redesign",
        512,
        512,
        512,
        tile_m=64,
        tile_n=64,
        bound_mode="per-tile",
        input_profile="adaptive-bands",
        require_adaptive=True,
    ),
    "resident-redesign-candidate",
    "rns_residue_current",
    "rank-51 grouped active-schedule candidate",
    "release",
    "benchmark_evidence_only",
    backends=("hip-direct",),
    resident_redesign_candidate="grouped_active_schedule_v3",
    resident_redesign_dimensions=(
        "data_layout",
        "tile_shape",
        "export_interaction",
        "schedule_upload",
        "workspace_reuse",
    ),
)
resident_redesign_args = benchmark_sweep.scenario_args_for_item(scenario_base_args, resident_redesign_item)
resident_redesign_command = benchmark_sweep.command_for(
    Path("rns8-bench"),
    "hip-direct",
    resident_redesign_item.semantics,
    resident_redesign_item.case,
    None,
    None,
    resident_redesign_args,
)
assert "--resident-redesign-candidate" in resident_redesign_command
assert "grouped_active_schedule_v3" in resident_redesign_command
assert "--resident-redesign-dimensions" in resident_redesign_command
assert "data_layout,tile_shape,export_interaction,schedule_upload,workspace_reuse" in resident_redesign_command
adaptive_group_item = catalog["adaptive-grouped-scheduler"][0]
adaptive_group_args = benchmark_sweep.scenario_args_for_item(scenario_base_args, adaptive_group_item)
adaptive_group_command = benchmark_sweep.command_for(
    Path("rns8-bench"),
    "hip-direct",
    adaptive_group_item.semantics,
    adaptive_group_item.case,
    None,
    None,
    adaptive_group_args,
)
assert "--adaptive-grouped-scheduler" in adaptive_group_command
overlap_items = catalog["streaming-overlap"]
assert {item.review_mode_expectation for item in overlap_items} == {"release"}
assert {item.promotion_eligibility for item in overlap_items} == {"streaming_overlap_evidence_only"}
assert {item.semantics for item in overlap_items} == {"bounded-i64", "bounded-u64"}
assert {item.case.m for item in overlap_items} == {512, 1024}
assert sum(1 for item in overlap_items if item.streaming_overlap) == 4
assert sum(1 for item in overlap_items if item.backends == ("cpu",)) == 4
assert sum(1 for item in overlap_items if item.pack_mode == "prepacked_reuse_b") == 8
overlap_item = next(item for item in overlap_items if item.streaming_overlap)
overlap_args = benchmark_sweep.scenario_args_for_item(scenario_base_args, overlap_item)
overlap_command = benchmark_sweep.command_for(
    Path("rns8-bench"),
    "hip-direct",
    overlap_item.semantics,
    overlap_item.case,
    None,
    None,
    overlap_args,
)
assert "--streaming-overlap" in overlap_command
release_item = catalog["release-gate-closeout"][0]
release_args = benchmark_sweep.scenario_args_for_item(scenario_base_args, release_item)
release_command = benchmark_sweep.command_for(
    Path("rns8-bench"),
    "hip-direct",
    release_item.semantics,
    release_item.case,
    None,
    None,
    release_args,
)
assert "--release-gate" in release_command and "large-release-validation-4096-budgeted" in release_command
error_detection_item = catalog["error-detecting-fast-path"][0]
error_detection_args = benchmark_sweep.scenario_args_for_item(scenario_base_args, error_detection_item)
error_detection_command = benchmark_sweep.command_for(
    Path("rns8-bench"),
    "hip-direct",
    error_detection_item.semantics,
    error_detection_item.case,
    None,
    None,
    error_detection_args,
)
assert "--error-detection-policy" in error_detection_command
assert "freivalds_two_round_product_check_research" in error_detection_command
cpu_selector_item = catalog["cpu-small-shape-selector"][0]
cpu_selector_args = benchmark_sweep.scenario_args_for_item(scenario_base_args, cpu_selector_item)
cpu_selector_command = benchmark_sweep.command_for(
    Path("rns8-bench"),
    "hip-direct",
    cpu_selector_item.semantics,
    cpu_selector_item.case,
    None,
    None,
    cpu_selector_args,
)
assert "--cpu-small-shape-selector" in cpu_selector_command
assert "bounded_i64_32_cpu_cutoff_review" in cpu_selector_command
incremental_item = catalog["incremental-result-cache"][0]
incremental_args = benchmark_sweep.scenario_args_for_item(scenario_base_args, incremental_item)
incremental_command = benchmark_sweep.command_for(
    Path("rns8-bench"),
    "hip-direct",
    incremental_item.semantics,
    incremental_item.case,
    None,
    None,
    incremental_args,
)
assert "--incremental-result-cache" in incremental_command
assert "bounded_i64_dirty_tile_partial_recompute_research" in incremental_command
sparse_item = catalog["sparse-a-4-to-2"][0]
assert sparse_item.sparse_a_4_to_2 is True
assert sparse_item.backends == ("cpu", "hip-direct", "amdgpu-builtins")
sparse_args = benchmark_sweep.scenario_args_for_item(scenario_base_args, sparse_item)
sparse_command = benchmark_sweep.command_for(
    Path("rns8-bench"),
    "amdgpu-builtins",
    sparse_item.semantics,
    sparse_item.case,
    251,
    None,
    sparse_args,
)
assert "--sparse-a-4-to-2" in sparse_command
assert "--modulus" in sparse_command and "251" in sparse_command
assert sparse_item.metadata["sparse_contract"] == "a_4_to_2_structured_k_v1"
dense_sparse_item = next(item for item in catalog["sparse-a-4-to-2"] if item.sparse_a_4_to_2_dense_baseline)
dense_sparse_args = benchmark_sweep.scenario_args_for_item(scenario_base_args, dense_sparse_item)
dense_sparse_command = benchmark_sweep.command_for(
    Path("rns8-bench"),
    "amdgpu-builtins",
    dense_sparse_item.semantics,
    dense_sparse_item.case,
    251,
    None,
    dense_sparse_args,
)
assert "--sparse-a-4-to-2" in dense_sparse_command
assert "--sparse-a-4-to-2-dense-baseline" in dense_sparse_command
assert dense_sparse_item.metadata["dense_baseline_role"] == "same_expanded_sparse_input_amdgpu_builtin_dense"
assert (
    benchmark_sweep.backend_id(
        {
            "backend_selected": "amdgpu-builtins",
            "selected_kernel": "amdgpu_builtin_cdna3_smfmac_i32_16x16x64_i8_sparse_a_v1",
            "scenario_metadata": {"sparse_a_4_to_2": True},
        }
    )
    == "amdgpu-builtins-sparse-a-runtime"
)
sparse_runtime_backend = benchmark_sweep.backend_id(
    {
        "backend_selected": "amdgpu-builtins",
        "selected_kernel": "amdgpu_builtin_cdna3_smfmac_i32_16x16x64_i8_sparse_a_v1",
        "scenario_metadata": {"sparse_a_4_to_2": True},
    }
)
assert backend_family_id(sparse_runtime_backend) == "amdgpu-builtins"
assert (
    benchmark_sweep.backend_id(
        {
            "backend_selected": "amdgpu-builtins",
            "selected_kernel": "amdgpu_builtin_cdna3_mfma_i32_16x16x32_i8_finite_u8_epilogue_v1",
            "scenario_metadata": {"sparse_a_4_to_2": True, "sparse_a_4_to_2_dense_baseline": True},
        }
    )
    == "amdgpu-builtins-dense-sparse-a-input"
)
dense_sparse_backend = benchmark_sweep.backend_id(
    {
        "backend_selected": "amdgpu-builtins",
        "selected_kernel": "amdgpu_builtin_cdna3_mfma_i32_16x16x32_i8_finite_u8_epilogue_v1",
        "scenario_metadata": {"sparse_a_4_to_2": True, "sparse_a_4_to_2_dense_baseline": True},
    }
)
assert backend_family_id(dense_sparse_backend) == "amdgpu-builtins"

with tempfile.TemporaryDirectory() as temp_dir:
    capture_path = Path(temp_dir) / "capture.json"
    capture_path.write_text(json.dumps(finite_capture("ck", 190)), encoding="utf-8")
    metadata = {
        "family": "finite-modulus-map",
        "name": "finite-ring-map-1024",
        "promotion_eligibility": "non_promoting_modulus_map",
        "metadata": {"promotion_scope": "non_promoting_modulus_map"},
    }
    benchmark_sweep.annotate_scenario_metadata(capture_path, metadata)
    annotated = load_capture(capture_path)
    validate_capture(annotated, capture_path)
    assert annotated["scenario_metadata"] == metadata

broader_chain_items = catalog["rns-chain-final-output-broader"]
assert len(broader_chain_items) == 16
assert {
    (item.semantics, item.case.m, item.residue_chain_independent_final_export)
    for item in broader_chain_items
} == {
    (semantics, shape, independent)
    for semantics in ["bounded-i64", "bounded-u64", "exact-wide-signed", "exact-wide-unsigned"]
    for shape in [512, 1024]
    for independent in [False, True]
}
broader_chain_args = benchmark_sweep.scenario_args_for_item(scenario_base_args, broader_chain_items[0])
broader_chain_command = benchmark_sweep.command_for(
    Path("rns8-bench"),
    "hip-direct",
    broader_chain_items[0].semantics,
    broader_chain_items[0].case,
    None,
    None,
    broader_chain_args,
)
assert "--residue-chain-length" in broader_chain_command and "3" in broader_chain_command
assert "--residue-chain-final-export" in broader_chain_command

compact_output_chain_items = catalog["exact-wide-output-chain"]
assert len(compact_output_chain_items) == 2
assert {item.review_mode_expectation for item in compact_output_chain_items} == {"release"}
assert {
    item.promotion_eligibility for item in compact_output_chain_items
} == {"lazy_export_chain_evidence_only", "lazy_export_chain_reuse_evidence_only"}
assert {item.residue_chain_length for item in compact_output_chain_items} == {3}
assert all(item.next_op_hint == "rns-gemm" for item in compact_output_chain_items)
exact_wide_output_chain_items = catalog["exact-wide-output-chain-broader"]
assert len(exact_wide_output_chain_items) == 12
assert {
    (item.semantics, item.case.m, item.residue_chain_length, item.residue_chain_final_export)
    for item in exact_wide_output_chain_items
} == {
    (semantics, shape, chain, final_export)
    for semantics in ["exact-wide-signed", "exact-wide-unsigned"]
    for shape in [512, 1024]
    for chain, final_export in [(3, False), (4, False), (4, True)]
}
residue_item = next(item for item in exact_wide_output_chain_items if not item.residue_chain_final_export)
residue_args = benchmark_sweep.scenario_args_for_item(scenario_base_args, residue_item)
residue_command = benchmark_sweep.command_for(
    Path("rns8-bench"),
    "hip-direct",
    residue_item.semantics,
    residue_item.case,
    None,
    4,
    residue_args,
)
assert "--residue-chain-length" in residue_command and "3" in residue_command
assert "--residue-chain-final-export" not in residue_command
assert "--next-op-hint" in residue_command and "rns-gemm" in residue_command

