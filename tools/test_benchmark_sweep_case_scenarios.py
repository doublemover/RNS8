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
    "rns-chain-final-output",
    "rns-chain-final-output-broader",
    "grouped-dispatch",
    "resident-lifetime-arena",
    "adaptive-grouped-scheduler",
    "streaming-overlap",
    "release-gate-closeout",
    "fhe-lattice-proxy-starfoundry",
    "error-detecting-fast-path",
]:
    assert scenario_name in catalog
    assert catalog[scenario_name]

for family, items in catalog.items():
    for item in items:
        if item.verification_amortization != "none":
            assert "cpu" in item.backends, f"{family}/{item.name} verification amortization requires CPU baseline"
        if item.error_detection_policy != "none":
            assert "cpu" in item.backends, f"{family}/{item.name} error detection policy requires CPU baseline"

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

fusion_item = catalog["residue-channel-fusion"][0]
assert fusion_item.residue_channel_fusion is True
assert fusion_item.next_op_hint == "final-export"
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
tile_item = catalog["tile-shape-sweeps"][0]
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
graph_item = next(item for item in catalog["hip-graph-replay"] if item.hip_graph_replay)
assert len(catalog["hip-graph-replay"]) == 16
assert {item.review_mode_expectation for item in catalog["hip-graph-replay"]} == {"release"}
assert {item.promotion_eligibility for item in catalog["hip-graph-replay"]} == {"hip_graph_replay_evidence_only"}
assert {item.case.m for item in catalog["hip-graph-replay"]} == {512, 1024}
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
grouped_item = catalog["grouped-dispatch"][0]
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
resident_item = catalog["resident-lifetime-arena"][0]
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
overlap_item = catalog["streaming-overlap"][0]
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

