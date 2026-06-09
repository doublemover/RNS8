large_bounded_args = copy.copy(scenario_args)
large_bounded_args.backends = ["hip-direct"]
large_bounded_args.scenario = ["large-exploratory"]
large_bounded_args.semantics = ["bounded-i64", "bounded-u64"]
large_bounded_entries = benchmark_sweep.sweep_command_entries(large_bounded_args)
assert [entry.scenario["name"] for entry in large_bounded_entries] == [
    "bounded-i64-2048",
    "bounded-i64-2048-reuse-b",
    "bounded-u64-2048",
    "bounded-u64-2048-reuse-b",
    "bounded-i64-4096",
    "bounded-i64-4096-reuse-b",
    "bounded-u64-4096",
    "bounded-u64-4096-reuse-b",
]
assert {entry.scenario["semantics"] for entry in large_bounded_entries} == {"bounded-i64", "bounded-u64"}
assert {entry.scenario["pack_mode"] for entry in large_bounded_entries} == {
    "per_repeat_repack",
    "prepacked_reuse_b",
}
assert any("--reuse-packed-b" in entry.command for entry in large_bounded_entries)

large_validation_args = copy.copy(scenario_args)
large_validation_args.backends = None
large_validation_args.scenario = ["large-release-validation"]
large_validation_entries = benchmark_sweep.sweep_command_entries(large_validation_args)
assert len(large_validation_entries) == 54
assert {entry.scenario["family"] for entry in large_validation_entries} == {"large-release-validation"}
assert {entry.scenario["name"] for entry in large_validation_entries} == {
    "bounded-i64-2048-required-baselines",
    "bounded-u64-2048-required-baselines",
    "bounded-i64-2048-reuse-b-required-baselines",
    "bounded-u64-2048-reuse-b-required-baselines",
    "exact-wide-signed-2048-required-baselines",
    "exact-wide-unsigned-2048-required-baselines",
    "finite-ring-2048-hot-required-baselines",
    "finite-field-2048-hot-required-baselines",
    "wrap64-2048-required-baselines",
}
assert {
    entry.scenario["backend"]
    for entry in large_validation_entries
    if entry.scenario["name"] == "bounded-i64-2048-required-baselines"
} == {"cpu", "hip-direct", "hip-vector-alu-int64", "hipblaslt", "ck", "rocwmma"}
assert {
    entry.scenario["backend"]
    for entry in large_validation_entries
    if entry.scenario["name"] == "bounded-u64-2048-required-baselines"
} == {"cpu", "hip-direct", "hip-vector-alu-int64", "hipblaslt", "rocwmma"}
assert {
    entry.scenario["backend"]
    for entry in large_validation_entries
    if entry.scenario["name"] == "wrap64-2048-required-baselines"
} == {"wrap64-byte-limb", "hip-direct"}
assert sorted(
    entry.scenario["modulus"]
    for entry in large_validation_entries
    if entry.scenario["name"] == "finite-ring-2048-hot-required-baselines"
    and entry.scenario["backend"] == "cpu"
) == [251, 255, 256]
assert all(
    entry.scenario.get("metadata", {}).get("workflow_name") == "large_shape_release_validation"
    for entry in large_validation_entries
)
assert any("--reuse-packed-b" in entry.command for entry in large_validation_entries)
assert any(
    entry.scenario.get("metadata", {}).get("validation_contract")
    == "same_contract_cpu_direct_vector_accelerator_release_review"
    for entry in large_validation_entries
)

large_4096_budgeted_args = copy.copy(scenario_args)
large_4096_budgeted_args.backends = None
large_4096_budgeted_args.scenario = ["large-release-validation-4096-budgeted"]
large_4096_budgeted_entries = benchmark_sweep.sweep_command_entries(large_4096_budgeted_args)
assert len(large_4096_budgeted_entries) == 43
assert {
    entry.scenario["backend"]
    for entry in large_4096_budgeted_entries
    if entry.scenario["name"] == "bounded-i64-4096-budgeted-baselines"
} == {"cpu", "hip-direct", "hip-vector-alu-int64", "hipblaslt", "ck", "rocwmma"}
assert {
    entry.scenario["backend"]
    for entry in large_4096_budgeted_entries
    if entry.scenario["name"] == "bounded-u64-4096-budgeted-baselines"
} == {"cpu", "hip-direct", "hip-vector-alu-int64", "hipblaslt", "rocwmma"}
assert {
    entry.scenario["backend"]
    for entry in large_4096_budgeted_entries
    if entry.scenario["name"] == "exact-wide-signed-4096-budgeted-export"
} == {"cpu", "hip-direct", "hipblaslt", "ck", "rocwmma"}
assert {
    entry.scenario["backend"]
    for entry in large_4096_budgeted_entries
    if entry.scenario["name"] == "exact-wide-unsigned-4096-budgeted-export"
} == {"cpu", "hip-direct", "hipblaslt", "ck", "rocwmma"}
assert sorted(
    (entry.scenario["modulus"], entry.scenario["backend"])
    for entry in large_4096_budgeted_entries
    if entry.scenario["name"] == "finite-ring-4096-hot-budgeted-baselines"
) == [
    (251, "ck"),
    (251, "cpu"),
    (251, "hip-direct"),
    (251, "hipblaslt"),
    (251, "rocwmma"),
    (255, "ck"),
    (255, "cpu"),
    (255, "hip-direct"),
    (255, "hipblaslt"),
    (255, "rocwmma"),
    (256, "ck"),
    (256, "cpu"),
    (256, "hip-direct"),
    (256, "hipblaslt"),
    (256, "rocwmma"),
]
assert {
    entry.scenario["backend"]
    for entry in large_4096_budgeted_entries
    if entry.scenario["name"] == "finite-field-4096-hot-budgeted-baselines"
} == {"cpu", "hip-direct", "hipblaslt", "ck", "rocwmma"}
assert {
    entry.scenario["backend"]
    for entry in large_4096_budgeted_entries
    if entry.scenario["name"] == "wrap64-4096-budgeted-baselines"
} == {"wrap64-byte-limb", "hip-direct"}
assert all(
    entry.scenario.get("metadata", {}).get("promotion_scope") == "non_promoting_budgeted_dry_run"
    for entry in large_4096_budgeted_entries
)
assert all(
    "--release-gate" in entry.command and "large-release-validation-4096-budgeted" in entry.command
    for entry in large_4096_budgeted_entries
)

finite_generic_args = copy.copy(scenario_args)
finite_generic_args.backends = ["hip-direct", "ck", "rocwmma"]
finite_generic_args.scenario = ["finite-generic-moduli"]
finite_generic_entries = benchmark_sweep.sweep_command_entries(finite_generic_args)
assert len(finite_generic_entries) == 5
assert {entry.scenario["backend"] for entry in finite_generic_entries} == {"hip-direct"}
assert {entry.scenario["modulus"] for entry in finite_generic_entries} == {127, 253}
assert {entry.scenario["name"] for entry in finite_generic_entries} == {
    "ring-prime-127-512",
    "field-prime-127-512",
    "ring-composite-253-512",
    "ring-prime-127-2048",
    "ring-composite-253-2048",
}
assert all(
    entry.scenario.get("metadata", {}).get("workflow_name") == "finite_u8_generic_modulus"
    for entry in finite_generic_entries
)
assert any(
    entry.scenario.get("metadata", {}).get("prime_or_composite") == "composite"
    for entry in finite_generic_entries
)
assert any(
    entry.scenario["shape"]["m"] == 2048
    and entry.scenario.get("metadata", {}).get("large_shape_role") == "finite_generic_modulus_probe"
    for entry in finite_generic_entries
)

finite_modulus_map_args = copy.copy(scenario_args)
finite_modulus_map_args.backends = None
finite_modulus_map_args.scenario = ["finite-modulus-map"]
finite_modulus_map_entries = benchmark_sweep.sweep_command_entries(finite_modulus_map_args)
assert len(finite_modulus_map_entries) == 200
assert {entry.scenario["shape"]["m"] for entry in finite_modulus_map_entries} == {128, 512, 1024, 2048}
assert {
    entry.scenario["modulus"]
    for entry in finite_modulus_map_entries
    if entry.scenario["semantics"] == "finite-u8-ring"
} == {127, 241, 243, 251, 253, 255, 256}
assert {
    entry.scenario["modulus"]
    for entry in finite_modulus_map_entries
    if entry.scenario["semantics"] == "finite-u8-field"
} == {127, 241, 251}
assert {entry.scenario["backend"] for entry in finite_modulus_map_entries} == {
    "cpu",
    "hip-direct",
    "hipblaslt",
    "ck",
    "rocwmma",
}
assert all(
    entry.scenario.get("metadata", {}).get("promotion_scope") == "non_promoting_modulus_map"
    for entry in finite_modulus_map_entries
)

bridge_args = copy.copy(scenario_args)
bridge_args.backends = None
bridge_args.bench_for = ["hip-direct=hip-direct-release-bench"]
bridge_args.scenario = ["native-to-rns-bridge"]
bridge_entries = benchmark_sweep.sweep_command_entries(bridge_args)
assert len(bridge_entries) == 4
assert {entry.scenario["name"] for entry in bridge_entries} == {
    "bounded-i64-64",
    "bounded-u64-64",
    "bounded-i64-128",
    "bounded-u64-128",
}
assert {entry.scenario["backend"] for entry in bridge_entries} == {"auto"}
assert all(entry.command[0] == "hip-direct-release-bench" for entry in bridge_entries)
assert all(entry.command[entry.command.index("--backend") + 1] == "auto" for entry in bridge_entries)
assert all("--native-to-rns-bridge" in entry.command for entry in bridge_entries)
assert all("--reuse-packed-inputs" not in entry.command for entry in bridge_entries)
assert all(entry.scenario["native_to_rns_bridge"] is True for entry in bridge_entries)
assert all(
    entry.scenario.get("metadata", {}).get("workflow_name") == "native_to_rns_bridge"
    for entry in bridge_entries
)
assert any(
    entry.scenario.get("metadata", {}).get("conversion_event_required") == "native_u64_to_rns_kernel"
    for entry in bridge_entries
)

vector_chain_args = copy.copy(scenario_args)
vector_chain_args.backends = None
vector_chain_args.bench_for = ["hip-direct=hip-direct-release-bench"]
vector_chain_args.scenario = ["vector-to-rns-chain"]
vector_chain_entries = benchmark_sweep.sweep_command_entries(vector_chain_args)
assert len(vector_chain_entries) == 32
assert {entry.scenario["semantics"] for entry in vector_chain_entries} == {"bounded-i64", "bounded-u64"}
assert {entry.scenario["shape"]["m"] for entry in vector_chain_entries} == {64, 128, 512, 1024}
assert {
    entry.scenario.get("metadata", {}).get("chain_control_mode")
    for entry in vector_chain_entries
} == {"fused_device_native_to_rns", "host_export_repack_control"}
assert {entry.scenario["backend"] for entry in vector_chain_entries} == {"auto"}
assert all(entry.command[0] == "hip-direct-release-bench" for entry in vector_chain_entries)
assert all(entry.command[entry.command.index("--backend") + 1] == "auto" for entry in vector_chain_entries)
assert any("--vector-to-rns-chain" in entry.command for entry in vector_chain_entries)
assert any("--vector-to-rns-chain-host-repack-control" in entry.command for entry in vector_chain_entries)
assert all("--native-to-rns-bridge" not in entry.command for entry in vector_chain_entries)
assert all(entry.scenario["native_to_rns_bridge"] is False for entry in vector_chain_entries)
assert all(entry.scenario["vector_to_rns_chain"] is True for entry in vector_chain_entries)
assert {
    entry.scenario["vector_to_rns_chain_host_repack_control"]
    for entry in vector_chain_entries
} == {False, True}
assert {
    entry.scenario["pack_mode"]
    for entry in vector_chain_entries
} == {"per_repeat_repack", "prepacked_reuse_b"}
assert any("--reuse-packed-b" in entry.command for entry in vector_chain_entries)
assert any(
    entry.scenario.get("metadata", {}).get("reuse_contract") == "consumer_b_prepacked_before_warmups"
    for entry in vector_chain_entries
)
assert all(
    entry.scenario.get("metadata", {}).get("workflow_name") == "vector_to_rns_chain"
    for entry in vector_chain_entries
)
assert any(
    entry.scenario.get("metadata", {}).get("conversion_event_required") == "native_u64_to_rns_kernel"
    for entry in vector_chain_entries
)
assert any(
    entry.scenario.get("metadata", {}).get("host_repack_event_required") == "vector_to_rns_host_repack_a"
    for entry in vector_chain_entries
)

algebra_args = copy.copy(scenario_args)
algebra_args.backends = ["ck"]
algebra_args.scenario = ["computational-algebra-proxies"]
algebra_entries = benchmark_sweep.sweep_command_entries(algebra_args)
assert len(algebra_entries) == 5
assert all(entry.scenario["family"] == "computational-algebra-proxies" for entry in algebra_entries)
assert {entry.scenario.get("metadata", {}).get("source_role") for entry in algebra_entries} == {
    "computational_algebra_proxy"
}
assert any(entry.scenario.get("metadata", {}).get("workflow_name") == "F4" for entry in algebra_entries)
assert any(entry.scenario["semantics"] == "exact-wide-signed" for entry in algebra_entries)

fhe_args = copy.copy(scenario_args)
fhe_args.backends = ["hip-direct"]
fhe_args.scenario = ["fhe-lattice-proxies"]
fhe_entries = benchmark_sweep.sweep_command_entries(fhe_args)
assert len(fhe_entries) == 4
assert all(entry.scenario["family"] == "fhe-lattice-proxies" for entry in fhe_entries)
assert {entry.scenario.get("metadata", {}).get("source_role") for entry in fhe_entries} == {
    "fhe_lattice_proxy"
}
assert any(entry.scenario.get("metadata", {}).get("workflow_name") == "ntt_intt_pressure" for entry in fhe_entries)
assert any("--reuse-packed-b" in entry.command for entry in fhe_entries)
assert any("--residue-chain-length" in entry.command and "4" in entry.command for entry in fhe_entries)

with tempfile.TemporaryDirectory() as tmp:
    manifest_paths = benchmark_sweep.write_scenario_manifest(scenario_entries, scenario_args, Path(tmp))
    assert manifest_paths is not None
    manifest = json.loads(Path(manifest_paths["scenario_manifest"]).read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["scenario_families"] == ["repeated-b"]
    assert manifest["capture_count"] == 6
    assert manifest["entries"][0]["output_domain"] == "host_export"
    assert Path(manifest_paths["scenario_markdown"]).exists()

with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)
    existing = tmp_path / "existing.json"
    existing.write_text(
        (FIXTURE_DIR / "v4_finite_ring_u8_ck.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    missing = tmp_path / "missing.json"
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")
    assert benchmark_sweep.existing_capture_valid(existing) is True
    assert benchmark_sweep.existing_capture_valid(missing) is False
    assert benchmark_sweep.existing_capture_valid(invalid) is False
    resume_args = argparse.Namespace(skip_existing=True, max_new_captures=0)
    capture_paths = []
    stats = benchmark_sweep.execute_sweep_entries(
        [
            benchmark_sweep.SweepCommand("existing", ["not-run"], existing),
            benchmark_sweep.SweepCommand("missing", ["not-run"], missing),
        ],
        resume_args,
        capture_paths,
    )
    assert capture_paths == [existing]
    assert stats == {
        "planned_captures": 2,
        "skipped_existing_captures": 1,
        "new_captures_attempted": 0,
        "new_captures_completed": 0,
        "deduped_cpu_captures": 0,
        "deferred_captures": 1,
    }

