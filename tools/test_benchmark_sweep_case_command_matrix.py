wrap64_args = argparse.Namespace(
    bench=Path("rns8-bench"),
    bench_for=[],
    out_root=Path("temp") / "wrap64-release",
    backends=None,
    semantics=["wrap-u64"],
    case=None,
    adaptive_case=None,
    shapes=None,
    modulus=None,
    exact_wide_limbs=None,
    include_exact_wide_limb_variants=False,
    residue_chain_length=1,
    include_default_adaptive=False,
    include_adaptive_workloads=False,
    adaptive_only=False,
    include_wrap64=False,
    include_wrap64_rocwmma_candidate=False,
    include_exact_wide=False,
    reuse_packed_inputs=False,
    reuse_packed_a=False,
    reuse_packed_b=False,
    release_matrix=True,
    include_exploratory_large=False,
    review_mode="release",
    warmups=benchmark_sweep.RELEASE_MIN_WARMUPS,
    repeats=benchmark_sweep.RELEASE_MIN_REPEATS,
    seed=20260602,
    write_autotune_cache=False,
    autotune_cache=None,
)
wrap64_commands = benchmark_sweep.sweep_commands(wrap64_args)
assert len(wrap64_commands) == len(benchmark_sweep.PROMOTABLE_RELEASE_SHAPES) * 2
assert wrap64_commands[0][0] == "wrap-u64-wrap64-64-64x64x64-wrap64-byte-limb.json"
assert wrap64_commands[1][0] == "wrap-u64-wrap64-64-64x64x64-hip-direct.json"
assert all("--semantics" in command and "wrap-u64" in command for _name, command, _output in wrap64_commands)
wrap64_args.include_wrap64_rocwmma_candidate = True
candidate_commands = benchmark_sweep.sweep_commands(wrap64_args)
assert len(candidate_commands) == len(benchmark_sweep.PROMOTABLE_RELEASE_SHAPES) * 3
candidate_name, candidate_command, _candidate_output = candidate_commands[2]
assert candidate_name == "wrap-u64-wrap64-64-64x64x64-rocwmma-wrap64-candidate.json"
assert "--backend" in candidate_command and "rocwmma-wrap64-candidate" in candidate_command
assert "--tile-m" in candidate_command and "16" in candidate_command
wrap64_args.include_wrap64_rocwmma_candidate = False
wrap64_args.reuse_packed_inputs = True
reuse_commands = benchmark_sweep.sweep_commands(wrap64_args)
assert reuse_commands[0][0] == "wrap-u64-wrap64-64-64x64x64-reuse-packed-wrap64-byte-limb.json"
assert all("--reuse-packed-inputs" in command for _name, command, _output in reuse_commands)
wrap64_args.reuse_packed_inputs = False
wrap64_args.reuse_packed_a = True
reuse_a_commands = benchmark_sweep.sweep_commands(wrap64_args)
assert reuse_a_commands[0][0] == "wrap-u64-wrap64-64-64x64x64-reuse-packed-a-wrap64-byte-limb.json"
assert all("--reuse-packed-a" in command for _name, command, _output in reuse_a_commands)
wrap64_args.reuse_packed_a = False
wrap64_args.reuse_packed_b = True
reuse_b_commands = benchmark_sweep.sweep_commands(wrap64_args)
assert reuse_b_commands[0][0] == "wrap-u64-wrap64-64-64x64x64-reuse-packed-b-wrap64-byte-limb.json"
assert all("--reuse-packed-b" in command for _name, command, _output in reuse_b_commands)
wrap64_args.reuse_packed_b = False
wrap64_args.adaptive_only = True
try:
    benchmark_sweep.sweep_commands(wrap64_args)
except SystemExit as exc:
    assert "--adaptive-only requires" in str(exc)
else:
    raise AssertionError("adaptive-only without adaptive cases should fail even when wrap64 is requested")
wrap64_args.adaptive_only = False

exact_args = argparse.Namespace(
    bench=Path("rns8-bench"),
    bench_for=[],
    out_root=Path("temp") / "exact-wide",
    backends=["cpu", "hip-direct"],
    semantics=["exact_wide_signed"],
    case=["small:16,16,16"],
    adaptive_case=None,
    shapes=None,
    modulus=None,
    exact_wide_limbs=None,
    include_exact_wide_limb_variants=False,
    residue_chain_length=1,
    include_default_adaptive=False,
    include_adaptive_workloads=False,
    adaptive_only=False,
    include_wrap64=False,
    include_wrap64_rocwmma_candidate=False,
    include_exact_wide=False,
    reuse_packed_inputs=False,
    reuse_packed_a=False,
    reuse_packed_b=False,
    release_matrix=False,
    include_exploratory_large=False,
    review_mode="smoke",
    warmups=1,
    repeats=2,
    seed=7,
    write_autotune_cache=False,
    autotune_cache=None,
)
exact_commands = benchmark_sweep.sweep_commands(exact_args)
assert [name for name, _command, _output in exact_commands] == [
    "exact-wide-signed-small-16x16x16-cpu.json",
    "exact-wide-signed-small-16x16x16-hip-direct.json",
]
assert all("--semantics" in command and "exact-wide-signed" in command for _name, command, _output in exact_commands)
assert all("--exact-wide-limbs" in command and "4" in command for _name, command, _output in exact_commands)

anchor_args = copy.copy(exact_args)
anchor_args.cpu_reference_mode = "correctness-anchor"
anchor_args.cpu_threads = 4
anchor_args.cpu_parallel_threshold = 0
anchor_args.progress = True
anchor_commands = benchmark_sweep.sweep_commands(anchor_args)
anchor_cpu_command = anchor_commands[0][1]
anchor_gpu_command = anchor_commands[1][1]
assert anchor_cpu_command[anchor_cpu_command.index("--warmups") + 1] == "0"
assert anchor_cpu_command[anchor_cpu_command.index("--repeats") + 1] == "1"
assert anchor_gpu_command[anchor_gpu_command.index("--warmups") + 1] == "1"
assert anchor_gpu_command[anchor_gpu_command.index("--repeats") + 1] == "2"
assert "--cpu-threads" in anchor_cpu_command and "4" in anchor_cpu_command
assert "--cpu-parallel-threshold" in anchor_cpu_command and "0" in anchor_cpu_command
assert "--cpu-reference-mode" in anchor_cpu_command and "correctness-anchor" in anchor_cpu_command
assert "--progress" in anchor_cpu_command and "--progress" in anchor_gpu_command

for mask in ["0", "0,1", "0,1,2,3", "0,1,2,3,4,5,6,7"]:
    visible_args = copy.copy(exact_args)
    visible_args.hip_visible_devices = mask
    visible_args.rocr_visible_devices = None
    visible_args.gpu_device_ordinal = None
    visible_args.gpu_shards = None
    visible_entries = benchmark_sweep.sweep_command_entries(visible_args)
    assert visible_entries
    assert all(entry.env == {"HIP_VISIBLE_DEVICES": mask} for entry in visible_entries)
    assert [entry.output for entry in visible_entries] == [Path(output) for _name, _command, output in exact_commands]

sharded_args = copy.copy(exact_args)
sharded_args.hip_visible_devices = None
sharded_args.rocr_visible_devices = None
sharded_args.gpu_device_ordinal = None
sharded_args.gpu_shards = "0,1,2,3"
sharded_entries = benchmark_sweep.sweep_command_entries(sharded_args)
assert len(sharded_entries) == len(exact_commands) * 4
assert {entry.env["HIP_VISIBLE_DEVICES"] for entry in sharded_entries} == {"0", "1", "2", "3"}
assert {entry.env["ROCR_VISIBLE_DEVICES"] for entry in sharded_entries} == {"0", "1", "2", "3"}
assert {entry.output.parent.name for entry in sharded_entries} == {"gpu0", "gpu1", "gpu2", "gpu3"}
assert all(entry.name.startswith("gpu") for entry in sharded_entries)

with tempfile.TemporaryDirectory() as temp_dir:
    temp_path = Path(temp_dir)
    fake_command = [
        sys.executable,
        "-c",
        "print('{\"capture\": true}')",
        "--backend",
        "cpu",
        "--semantics",
        "bounded-i64",
        "--m",
        "16",
        "--n",
        "16",
        "--k",
        "16",
        "--seed",
        "7",
    ]
    entries = [
        benchmark_sweep.SweepCommand("cpu-a", fake_command, temp_path / "cpu-a.json"),
        benchmark_sweep.SweepCommand("cpu-b", fake_command, temp_path / "cpu-b.json"),
    ]
    dedupe_args = argparse.Namespace(
        skip_existing=False,
        max_new_captures=None,
        capture_timeout_seconds=None,
        progress=False,
    )
    capture_paths = []
    stats = benchmark_sweep.execute_sweep_entries(entries, dedupe_args, capture_paths)
    assert stats["new_captures_attempted"] == 1
    assert stats["new_captures_completed"] == 1
    assert stats["deduped_cpu_captures"] == 1
    assert [path.name for path in capture_paths] == ["cpu-a.json", "cpu-b.json"]
    assert entries[0].output.read_text(encoding="utf-8") == entries[1].output.read_text(encoding="utf-8")

from benchmark_sweep_lib.cli import load_required_isa_index, review_capture_paths

with tempfile.TemporaryDirectory() as tmp_name:
    root = Path(tmp_name)
    out_root = root / "rank-scenarios" / "all"
    scenario_root = out_root / "scenarios"
    scenario_root.mkdir(parents=True)
    capture_path = scenario_root / "bounded-hip-direct.json"
    capture = bounded_capture("hip-direct", 100)
    capture["_path"] = str(capture_path)
    capture_path.write_text(json.dumps(capture), encoding="utf-8")
    (scenario_root / "bounded-hip-direct.failed.json").write_text("{}", encoding="utf-8")
    default_args = argparse.Namespace(capture=[], capture_root=[], review_only=True, out_root=out_root)
    assert review_capture_paths(default_args) == [capture_path]
    explicit_args = argparse.Namespace(capture=[], capture_root=[scenario_root], review_only=True, out_root=root / "unused")
    assert review_capture_paths(explicit_args) == [capture_path]
    try:
        load_required_isa_index([root / "missing-isa"])
        raise AssertionError("missing ISA report directory was accepted")
    except SystemExit as exc:
        assert "--isa-report path does not exist" in str(exc)
    isa_root = root / "isa-reports"
    isa_root.mkdir()
    try:
        load_required_isa_index([isa_root])
        raise AssertionError("empty ISA report directory was accepted")
    except SystemExit as exc:
        assert "--isa-report found no" in str(exc)
    isa_path = isa_root / "hip_direct_kernels-gfx942-direct-hip-isa-summary.json"
    isa_path.write_text(
        json.dumps(
            {
                "object": "build/linux/hip_direct_kernels.hip.o",
                "target": "gfx942",
                "backend": "direct-hip",
                "reported_symbol_count": 1,
                "device_symbol_count": 1,
                "tools": {"rga_status": "not_run_optional", "rga": None},
                "instruction_totals": {
                    "matrix_instruction_count": 0,
                    "dense_integer_matrix_instruction_count": 0,
                    "sparse_integer_matrix_instruction_count": 0,
                    "matrix_instruction_histogram": {},
                    "matrix_instruction_families": [],
                },
            }
        ),
        encoding="utf-8",
    )
    isa_index = load_required_isa_index([isa_root])
    assert "direct-hip|gfx942" in isa_index
    assert "direct-hip|*" in isa_index

exact_variant_args = copy.copy(exact_args)
exact_variant_args.backends = ["cpu"]
exact_variant_args.include_exact_wide_limb_variants = True
exact_variant_commands = benchmark_sweep.sweep_commands(exact_variant_args)
assert len(exact_variant_commands) == len(benchmark_sweep.EXACT_WIDE_LIMB_VARIANTS)
assert [name for name, _command, _output in exact_variant_commands] == [
    "exact-wide-signed-small-16x16x16-limbs1-cpu.json",
    "exact-wide-signed-small-16x16x16-limbs2-cpu.json",
    "exact-wide-signed-small-16x16x16-limbs3-cpu.json",
    "exact-wide-signed-small-16x16x16-cpu.json",
    "exact-wide-signed-small-16x16x16-limbs8-cpu.json",
    "exact-wide-signed-small-16x16x16-limbs16-cpu.json",
    "exact-wide-signed-small-16x16x16-limbs32-cpu.json",
]
assert exact_variant_commands[0][1][exact_variant_commands[0][1].index("--exact-wide-limbs") + 1] == "1"
assert exact_variant_commands[-1][1][exact_variant_commands[-1][1].index("--exact-wide-limbs") + 1] == "32"

scenario_args = copy.copy(exact_args)
scenario_args.out_root = Path("temp") / "scenario"
scenario_args.backends = ["hip-direct"]
scenario_args.semantics = None
scenario_args.case = None
scenario_args.scenario = ["repeated-b"]
scenario_entries = benchmark_sweep.sweep_command_entries(scenario_args)
assert len(scenario_entries) == 4
assert [entry.scenario["name"] for entry in scenario_entries] == [
    "bounded-i64-512-production-baselines",
    "bounded-i64-1024-production-baselines",
    "bounded-i64-512",
    "bounded-i64-1024",
]
assert all(entry.scenario["family"] == "repeated-b" for entry in scenario_entries)
assert {entry.scenario["pack_mode"] for entry in scenario_entries} == {"per_repeat_repack", "prepacked_reuse_b"}
assert sum(1 for entry in scenario_entries if "--reuse-packed-b" in entry.command) == 2
assert all("--prefix-policy" in entry.command and "fixed-requested" in entry.command for entry in scenario_entries)
assert all("--max-prefix" in entry.command and "9" in entry.command for entry in scenario_entries)
assert all("scenarios" in entry.output.parts and "repeated-b" in entry.output.parts for entry in scenario_entries)
assert scenario_entries[0].name.startswith("repeated-b-bounded-i64-512-production-baselines-")

reuse_contract_args = copy.copy(scenario_args)
reuse_contract_args.backends = ["hipblaslt"]
reuse_contract_args.scenario = ["reuse-contract"]
reuse_contract_entries = benchmark_sweep.sweep_command_entries(reuse_contract_args)
assert len(reuse_contract_entries) == 16
assert {entry.scenario["family"] for entry in reuse_contract_entries} == {"reuse-contract"}
assert {entry.scenario["semantics"] for entry in reuse_contract_entries} == {"bounded-i64", "bounded-u64"}
assert {entry.scenario["shape"]["m"] for entry in reuse_contract_entries} == {1024, 2048}
assert {entry.scenario["pack_mode"] for entry in reuse_contract_entries} == {
    "per_repeat_repack",
    "prepacked_reuse_a",
    "prepacked_reuse_b",
    "prepacked_reuse",
}
assert any("--reuse-packed-a" in entry.command for entry in reuse_contract_entries)
assert any("--reuse-packed-b" in entry.command for entry in reuse_contract_entries)
assert any("--reuse-packed-inputs" in entry.command for entry in reuse_contract_entries)
assert all(entry.scenario["backend"] == "hipblaslt" for entry in reuse_contract_entries)
assert all(
    entry.scenario.get("metadata", {}).get("workflow_name") == "reuse_contract_release_matrix"
    for entry in reuse_contract_entries
)
assert {
    entry.scenario.get("metadata", {}).get("reuse_contract_role")
    for entry in reuse_contract_entries
} == {
    "nonreuse_baseline",
    "stable_a_candidate",
    "stable_b_candidate",
    "stable_ab_candidate",
}

direct_reuse_args = copy.copy(scenario_args)
direct_reuse_args.backends = None
direct_reuse_args.scenario = ["direct-hip-reuse-expansion"]
direct_reuse_entries = benchmark_sweep.sweep_command_entries(direct_reuse_args)
assert len(direct_reuse_entries) == 68
assert {entry.scenario["family"] for entry in direct_reuse_entries} == {"direct-hip-reuse-expansion"}
assert {
    entry.scenario["semantics"]
    for entry in direct_reuse_entries
} == {"bounded-u64", "finite-u8-ring", "finite-u8-field", "exact-wide-signed", "exact-wide-unsigned", "wrap-u64"}
assert all(
    entry.scenario["backend"] != "ck"
    for entry in direct_reuse_entries
    if entry.scenario["semantics"] == "bounded-u64"
)
assert {
    entry.scenario.get("metadata", {}).get("workflow_name")
    for entry in direct_reuse_entries
} == {"direct_hip_reuse_expansion"}
assert {
    entry.scenario.get("metadata", {}).get("reuse_contract_role")
    for entry in direct_reuse_entries
} == {"nonreuse_baseline", "stable_a_candidate", "stable_b_candidate", "stable_ab_candidate"}
assert any("--reuse-packed-a" in entry.command for entry in direct_reuse_entries)
assert any("--reuse-packed-b" in entry.command for entry in direct_reuse_entries)
assert any("--reuse-packed-inputs" in entry.command for entry in direct_reuse_entries)
assert any(entry.scenario["input_profile"] == "adaptive-bands" for entry in direct_reuse_entries)
assert any(entry.scenario["modulus"] == 255 for entry in direct_reuse_entries)
assert any(entry.scenario["residue_chain_length"] == 3 for entry in direct_reuse_entries)
assert any(entry.scenario["backend"] == "wrap64-byte-limb" for entry in direct_reuse_entries)
assert all("scenarios" in entry.output.parts and "direct-hip-reuse-expansion" in entry.output.parts for entry in direct_reuse_entries)

graph_args = copy.copy(scenario_args)
graph_args.backends = None
graph_args.scenario = ["hip-graph-replay"]
graph_entries = benchmark_sweep.sweep_command_entries(graph_args)
assert len(graph_entries) == 36
assert {entry.scenario["family"] for entry in graph_entries} == {"hip-graph-replay"}
assert {entry.scenario["review_mode_expectation"] for entry in graph_entries} == {"release"}
assert {entry.scenario["promotion_eligibility"] for entry in graph_entries} == {"hip_graph_replay_evidence_only"}
assert {entry.scenario["shape"]["m"] for entry in graph_entries} == {512, 1024}
assert sum(1 for entry in graph_entries if "--hip-graph-replay" in entry.command) == 18
resident_graph_entries = [
    entry
    for entry in graph_entries
    if "--hip-graph-replay" in entry.command
    and entry.scenario["metadata"].get("phase_label", "").endswith("reuse_inputs")
]
full_path_graph_entries = [
    entry
    for entry in graph_entries
    if "--hip-graph-replay" in entry.command
    and entry.scenario["metadata"].get("phase_label") == "hip_graph_bounded_full_pack_gemm_export"
]
finite_full_path_graph_entries = [
    entry
    for entry in graph_entries
    if "--hip-graph-replay" in entry.command
    and entry.scenario["metadata"].get("phase_label") == "hip_graph_finite_u8_full_pack_gemm_export"
]
wrap64_full_path_graph_entries = [
    entry
    for entry in graph_entries
    if "--hip-graph-replay" in entry.command
    and entry.scenario["metadata"].get("phase_label") == "hip_graph_wrap64_full_pack_gemm_export"
]
assert len(resident_graph_entries) == 8
assert len(full_path_graph_entries) == 4
assert len(finite_full_path_graph_entries) == 4
assert len(wrap64_full_path_graph_entries) == 2
assert all("--reuse-packed-inputs" in entry.command for entry in resident_graph_entries)
assert all("--residue-chain-length" in entry.command and "3" in entry.command for entry in resident_graph_entries)
assert all("--next-op-hint" in entry.command and "rns-gemm" in entry.command for entry in resident_graph_entries)
assert all("--reuse-packed-inputs" not in entry.command for entry in full_path_graph_entries)
assert all("--residue-chain-length" not in entry.command for entry in full_path_graph_entries)
assert all("--next-op-hint" not in entry.command for entry in full_path_graph_entries)
assert all("--modulus" in entry.command and "251" in entry.command for entry in finite_full_path_graph_entries)
assert all("--reuse-packed-inputs" not in entry.command for entry in finite_full_path_graph_entries)
assert all("--semantics" in entry.command and "wrap-u64" in entry.command for entry in wrap64_full_path_graph_entries)
assert all("--reuse-packed-inputs" not in entry.command for entry in wrap64_full_path_graph_entries)

sparse_args = copy.copy(scenario_args)
sparse_args.backends = None
sparse_args.scenario = ["sparse-a-4-to-2"]
sparse_entries = benchmark_sweep.sweep_command_entries(sparse_args)
assert len(sparse_entries) == 24
assert {entry.scenario["family"] for entry in sparse_entries} == {"sparse-a-4-to-2"}
assert {entry.scenario["sparse_a_4_to_2"] for entry in sparse_entries} == {True}
assert {entry.scenario["backend"] for entry in sparse_entries} == {"cpu", "hip-direct", "amdgpu-builtins"}
assert {entry.scenario["modulus"] for entry in sparse_entries} == {251, 255}
assert all("--sparse-a-4-to-2" in entry.command for entry in sparse_entries)
assert all("sparse-a-4to2" in entry.output.name for entry in sparse_entries)
assert all("scenarios" in entry.output.parts and "sparse-a-4-to-2" in entry.output.parts for entry in sparse_entries)
dense_sparse_entries = [
    entry for entry in sparse_entries if entry.scenario["sparse_a_4_to_2_dense_baseline"] is True
]
assert len(dense_sparse_entries) == 6
assert {entry.scenario["backend"] for entry in dense_sparse_entries} == {"amdgpu-builtins"}
assert all("--sparse-a-4-to-2-dense-baseline" in entry.command for entry in dense_sparse_entries)
assert all("dense-baseline" in entry.output.name for entry in dense_sparse_entries)

skinny_args = copy.copy(scenario_args)
skinny_args.backends = ["hip-vector-alu-int64"]
skinny_args.scenario = ["skinny-gemv"]
skinny_entries = benchmark_sweep.sweep_command_entries(skinny_args)
assert len(skinny_entries) == 3
assert [entry.scenario["name"] for entry in skinny_entries] == [
    "bounded-i64-n1-512",
    "bounded-u64-n1-1024",
    "bounded-i64-n1-longk-256",
]
assert all(entry.scenario["family"] == "skinny-gemv" for entry in skinny_entries)
assert all(entry.scenario["backend"] == "hip-vector-alu-int64" for entry in skinny_entries)
assert all(entry.scenario["shape"]["n"] == 1 for entry in skinny_entries)
assert all("--backend" in entry.command for entry in skinny_entries)
assert all(benchmark_sweep.cli_backend("hip-vector-alu-int64") in entry.command for entry in skinny_entries)
assert all("scenarios" in entry.output.parts and "skinny-gemv" in entry.output.parts for entry in skinny_entries)
assert skinny_entries[0].name.startswith("skinny-gemv-bounded-i64-n1-512-")
assert skinny_entries[2].scenario.get("metadata", {}).get("workflow_name") == "gemv_n1_long_k"
assert skinny_entries[2].command[skinny_entries[2].command.index("--k") + 1] == "4096"

many_small_args = copy.copy(scenario_args)
many_small_args.backends = ["hip-direct"]
many_small_args.scenario = ["many-small"]
many_small_entries = benchmark_sweep.sweep_command_entries(many_small_args)
assert len(many_small_entries) == 22
assert all(entry.scenario["family"] == "many-small" for entry in many_small_entries)
assert {entry.scenario["name"] for entry in many_small_entries} == {
    "bounded-i64-32-proxy",
    "bounded-i64-32-host-batch64",
    "bounded-i64-32-oneshot-proxy",
    "bounded-i64-64-proxy",
    "bounded-i64-64-host-batch32",
    "bounded-i64-128-proxy",
    "bounded-i64-128-host-batch64",
    "bounded-u64-64-proxy",
    "bounded-u64-64-host-batch32",
    "bounded-u64-skinny-n1-host-batch128",
    "bounded-u64-skinny-n1-proxy",
    "exact-wide-signed-64-proxy",
    "exact-wide-signed-64-host-batch32",
    "exact-wide-signed-128-proxy",
    "exact-wide-signed-128-host-batch32",
    "exact-wide-unsigned-64-proxy",
    "exact-wide-unsigned-64-host-batch32",
    "exact-wide-unsigned-128-proxy",
    "exact-wide-unsigned-128-host-batch32",
    "finite-ring-64-host-batch32",
    "finite-ring-64-proxy",
}
assert any("--oneshot" in entry.command for entry in many_small_entries)
assert any("--host-api-batch-size" in entry.command for entry in many_small_entries)
assert sorted(
    entry.scenario["modulus"] for entry in many_small_entries if entry.scenario["semantics"] == "finite-u8-ring"
) == [251, 251, 255]
assert any(
    entry.scenario.get("metadata", {}).get("evidence_role") == "host_api_batch_candidate"
    for entry in many_small_entries
)
assert any(entry.scenario.get("metadata", {}).get("grouping_role") == "pre_grouped_baseline" for entry in many_small_entries)
assert any(entry.scenario.get("metadata", {}).get("grouping_role") == "host_api_batch_candidate" for entry in many_small_entries)
assert any(
    entry.scenario.get("metadata", {}).get("bridge_role") == "native_vector_to_rns_candidate"
    for entry in many_small_entries
)
assert any(
    entry.scenario["shape"]["n"] == 1 and entry.scenario.get("metadata", {}).get("phase_label") == "pre_grouped_skinny_proxy"
    for entry in many_small_entries
)
assert any(entry.scenario.get("exact_wide_limb_count") == 4 for entry in many_small_entries)
assert all(
    "hip-vector-alu-int64" not in entry.command
    for entry in many_small_entries
    if entry.scenario.get("host_api_batch_size", 1) > 1
)

grouped_dispatch_args = copy.copy(scenario_args)
grouped_dispatch_args.backends = ["hip-direct"]
grouped_dispatch_args.scenario = ["grouped-dispatch"]
grouped_dispatch_entries = benchmark_sweep.sweep_command_entries(grouped_dispatch_args)
assert [entry.scenario["name"] for entry in grouped_dispatch_entries] == [
    "bounded-i64-64-group32",
    "bounded-u64-64-group32",
    "bounded-i64-128-group64",
    "bounded-u64-skinny-n1-group128",
    "finite-ring-64-group32",
    "exact-wide-signed-64-group32",
    "exact-wide-unsigned-64-group32",
    "exact-wide-signed-128-group32",
    "exact-wide-unsigned-128-group32",
]
assert all(entry.scenario["family"] == "grouped-dispatch" for entry in grouped_dispatch_entries)
assert {entry.scenario["grouped_dispatch_tasks"] for entry in grouped_dispatch_entries} == {32, 64, 128}
assert all(entry.scenario["review_mode_expectation"] == "smoke" for entry in grouped_dispatch_entries)
assert all(entry.scenario["promotion_eligibility"] for entry in grouped_dispatch_entries)
assert all("--grouped-dispatch" in entry.command for entry in grouped_dispatch_entries)
assert any(entry.scenario.get("exact_wide_limb_count") == 4 for entry in grouped_dispatch_entries)
assert sum(1 for entry in grouped_dispatch_entries if entry.scenario["semantics"].startswith("exact-wide")) == 4
assert any(
    entry.scenario["semantics"] == "exact-wide-signed"
    and entry.scenario["shape"]["m"] == 128
    and entry.scenario["grouped_dispatch_tasks"] == 32
    for entry in grouped_dispatch_entries
)
assert any(
    entry.scenario["semantics"] == "exact-wide-unsigned"
    and entry.scenario["shape"]["m"] == 128
    and entry.scenario["grouped_dispatch_tasks"] == 32
    for entry in grouped_dispatch_entries
)
assert any(
    entry.scenario["semantics"] == "bounded-u64"
    and entry.scenario["shape"]["n"] == 64
    and entry.scenario.get("metadata", {}).get("grouped_strategy_expectation")
    == GROUPED_DISPATCH_STRATEGY_DEVICE_GROUPED_PACK_GEMM_AND_BOUNDED_EXPORT_KERNELS_BATCHED_D2H
    for entry in grouped_dispatch_entries
)
assert any(
    entry.scenario["semantics"] == "bounded-i64"
    and entry.scenario["shape"]["m"] == 128
    and entry.scenario["grouped_dispatch_tasks"] == 64
    for entry in grouped_dispatch_entries
)
assert any(
    entry.scenario["semantics"] == "bounded-u64"
    and entry.scenario["shape"]["n"] == 1
    and entry.scenario["grouped_dispatch_tasks"] == 128
    for entry in grouped_dispatch_entries
)
assert any(
    entry.scenario["semantics"] == "finite-u8-ring"
    and entry.scenario.get("metadata", {}).get("grouped_strategy_expectation")
    == GROUPED_DISPATCH_STRATEGY_DEVICE_GROUPED_PACK_GEMM_AND_FINITE_EXPORT_KERNEL_BATCHED_D2H
    for entry in grouped_dispatch_entries
)
assert any(
    entry.scenario.get("metadata", {}).get("prior_host_batch_signal")
    == "direct_hip_exact_wide_signed_64_hostbatch32"
    for entry in grouped_dispatch_entries
)
assert any(
    entry.scenario["semantics"] == "exact-wide-unsigned"
    and entry.scenario.get("metadata", {}).get("comparison_required")
    == "fastest_independent_and_same_backend_host_batch"
    for entry in grouped_dispatch_entries
)

rns_chain_args = copy.copy(scenario_args)
rns_chain_args.backends = ["hip-direct"]
rns_chain_args.scenario = ["rns-chain"]
rns_chain_entries = benchmark_sweep.sweep_command_entries(rns_chain_args)
assert [entry.scenario["name"] for entry in rns_chain_entries] == [
    "bounded-i64-chain3",
    "bounded-i64-chain3-reuse-b",
    "bounded-i64-chain4-256",
    "bounded-i64-chain4-256-reuse-b",
    "bounded-u64-chain3-256",
    "bounded-u64-chain3-256-reuse-b",
    "exact-wide-signed-chain3",
    "exact-wide-signed-chain3-reuse-b",
    "exact-wide-unsigned-chain3-256",
    "exact-wide-unsigned-chain3-256-reuse-b",
]
assert all(entry.scenario["output_domain"] == "residue_current_rns" for entry in rns_chain_entries)
assert all(entry.scenario.get("metadata", {}).get("output_domain_requirement") == "lazy_export" for entry in rns_chain_entries)
assert any("--residue-chain-length" in entry.command and "4" in entry.command for entry in rns_chain_entries)
assert any(entry.scenario.get("metadata", {}).get("chain_depth") == 4 for entry in rns_chain_entries)
assert {entry.scenario["pack_mode"] for entry in rns_chain_entries} == {
    "per_repeat_repack",
    "prepacked_reuse_b",
}
assert any("--reuse-packed-b" in entry.command for entry in rns_chain_entries)
assert any(
    entry.scenario.get("metadata", {}).get("reuse_contract") == "stable_chain_rhs_prepacked_before_warmups"
    for entry in rns_chain_entries
)

rns_chain_final_args = copy.copy(scenario_args)
rns_chain_final_args.backends = ["hip-direct"]
rns_chain_final_args.scenario = ["rns-chain-final-output"]
rns_chain_final_entries = benchmark_sweep.sweep_command_entries(rns_chain_final_args)
assert [entry.scenario["name"] for entry in rns_chain_final_entries] == [
    "bounded-i64-chain3-final-export",
    "bounded-i64-chain3-independent-final-export",
    "bounded-i64-chain3-final-export-reuse-b",
    "bounded-u64-chain3-final-export-256",
    "bounded-u64-chain3-independent-final-export-256",
    "exact-wide-signed-chain3-final-export",
    "exact-wide-signed-chain3-independent-final-export",
    "exact-wide-signed-chain3-final-export-reuse-b",
    "exact-wide-unsigned-chain3-final-export-256",
    "exact-wide-unsigned-chain3-independent-final-export-256",
    "exact-wide-signed-chain3-final-export-512",
]
assert all(entry.scenario["family"] == "rns-chain-final-output" for entry in rns_chain_final_entries)
assert all(entry.scenario["residue_chain_final_export"] is True for entry in rns_chain_final_entries)
assert all(entry.scenario["residue_chain_length"] == 3 for entry in rns_chain_final_entries)
assert all(entry.scenario["output_domain"] != "residue_current_rns" for entry in rns_chain_final_entries)
assert all(
    entry.scenario.get("metadata", {}).get("output_domain_requirement") == "same_final_output"
    for entry in rns_chain_final_entries
)
independent_chain_final_entries = [
    entry for entry in rns_chain_final_entries if entry.scenario["residue_chain_independent_final_export"] is True
]
assert [entry.scenario["name"] for entry in independent_chain_final_entries] == [
    "bounded-i64-chain3-independent-final-export",
    "bounded-u64-chain3-independent-final-export-256",
    "exact-wide-signed-chain3-independent-final-export",
    "exact-wide-unsigned-chain3-independent-final-export-256",
]
assert all("--residue-chain-independent-final-export" in entry.command for entry in independent_chain_final_entries)
assert all("--residue-chain-final-export" not in entry.command for entry in independent_chain_final_entries)
assert all(
    "--residue-chain-final-export" in entry.command
    for entry in rns_chain_final_entries
    if entry.scenario["residue_chain_independent_final_export"] is False
)
assert all("--next-op-hint" in entry.command and "final-export" in entry.command for entry in rns_chain_final_entries)
assert all("finalexport" in entry.name for entry in rns_chain_final_entries)
assert any(entry.scenario["shape"]["m"] == 512 for entry in rns_chain_final_entries)
assert any(
    entry.scenario.get("metadata", {}).get("reuse_contract") == "stable_chain_rhs_prepacked_before_warmups"
    for entry in rns_chain_final_entries
)

adaptive_bands_args = copy.copy(scenario_args)
adaptive_bands_args.backends = None
adaptive_bands_args.scenario = ["adaptive-bands"]
adaptive_bands_entries = benchmark_sweep.sweep_command_entries(adaptive_bands_args)
assert len(adaptive_bands_entries) == 14
assert {entry.scenario["family"] for entry in adaptive_bands_entries} == {"adaptive-bands"}
assert {
    entry.scenario["backend"]
    for entry in adaptive_bands_entries
    if entry.scenario["name"] == "bounded-i64-256"
} == {"cpu", "hip-direct", "hip-vector-alu-int64", "ck", "rocwmma"}
assert all(
    entry.scenario["backend"] != "ck"
    for entry in adaptive_bands_entries
    if entry.scenario["semantics"] == "bounded-u64"
)
assert all("--bound-mode" in entry.command and "per-tile" in entry.command for entry in adaptive_bands_entries)
assert all("--require-adaptive-execution" in entry.command for entry in adaptive_bands_entries)

bound_discovery_args = copy.copy(scenario_args)
bound_discovery_args.backends = None
bound_discovery_args.scenario = ["bound-discovery"]
bound_discovery_entries = benchmark_sweep.sweep_command_entries(bound_discovery_args)
assert len(bound_discovery_entries) == 48
assert {entry.scenario["family"] for entry in bound_discovery_entries} == {"bound-discovery"}
assert {entry.scenario["name"] for entry in bound_discovery_entries} == {
    "bounded-i64-256-static-global",
    "bounded-i64-256-input-scan-global",
    "bounded-i64-256-proof-mask-per-tile",
    "bounded-u64-rect-static-global",
    "bounded-u64-rect-input-scan-global",
    "bounded-u64-rect-proof-mask-per-tile",
    "bounded-i64-1024-static-global",
    "bounded-i64-1024-input-scan-global",
    "bounded-i64-1024-proof-mask-per-tile",
}
assert {
    entry.scenario["backend"]
    for entry in bound_discovery_entries
    if entry.scenario["name"] == "bounded-i64-256-static-global"
} == {"cpu", "hip-direct", "hip-vector-alu-int64", "hipblaslt", "ck", "rocwmma"}
assert {
    entry.scenario["backend"]
    for entry in bound_discovery_entries
    if entry.scenario["name"] == "bounded-i64-256-proof-mask-per-tile"
} == {"cpu", "hip-direct", "hip-vector-alu-int64", "ck", "rocwmma"}
assert all("--input-profile" in entry.command and "adaptive-bands" in entry.command for entry in bound_discovery_entries)
assert all("--bound-source" in entry.command for entry in bound_discovery_entries)
assert all(
    entry.command[entry.command.index("--bound-source") + 1] == "static-profile"
    for entry in bound_discovery_entries
    if entry.scenario["name"].endswith("static-global")
)
assert all(
    entry.command[entry.command.index("--bound-source") + 1] == "input-scan"
    for entry in bound_discovery_entries
    if not entry.scenario["name"].endswith("static-global")
)
assert all(
    "--bound-mode" in entry.command and entry.command[entry.command.index("--bound-mode") + 1] == "per-tile"
    for entry in bound_discovery_entries
    if entry.scenario["name"].endswith("proof-mask-per-tile")
)
assert all(
    "--require-adaptive-execution" in entry.command
    for entry in bound_discovery_entries
    if entry.scenario["name"].endswith("proof-mask-per-tile")
)
assert all(
    entry.scenario.get("metadata", {}).get("workflow_name") == "bound_discovery_proof_mask_release_matrix"
    for entry in bound_discovery_entries
)

large_args = copy.copy(scenario_args)
large_args.backends = ["hip-direct"]
large_args.scenario = ["large-exploratory"]
large_entries = benchmark_sweep.sweep_command_entries(large_args)
assert len(large_entries) == 22
assert {entry.scenario["name"] for entry in large_entries} == {
    "bounded-i64-2048",
    "bounded-i64-2048-reuse-b",
    "bounded-u64-2048",
    "bounded-u64-2048-reuse-b",
    "bounded-i64-4096",
    "bounded-i64-4096-reuse-b",
    "bounded-u64-4096",
    "bounded-u64-4096-reuse-b",
    "exact-wide-signed-2048",
    "exact-wide-unsigned-2048",
    "exact-wide-signed-4096",
    "exact-wide-unsigned-4096",
    "finite-ring-2048",
    "finite-field-2048",
    "finite-ring-4096",
    "finite-field-4096",
    "wrap64-2048",
    "wrap64-4096",
}
assert any(entry.scenario["shape"]["m"] == 4096 for entry in large_entries)
assert sorted(
    entry.scenario["modulus"] for entry in large_entries if entry.scenario["name"] == "finite-ring-2048"
) == [251, 255, 256]
assert all(
    entry.scenario["exact_wide_limb_count"] == 4
    for entry in large_entries
    if entry.scenario["semantics"].startswith("exact-wide")
)
assert any(
    entry.scenario.get("metadata", {}).get("large_shape_role") == "wrap64_direct_hip_throughput_probe"
    for entry in large_entries
)
assert any(
    entry.scenario.get("metadata", {}).get("reuse_contract") == "large_stable_rhs_prepacked_before_warmups"
    for entry in large_entries
)

layout_args = copy.copy(scenario_args)
layout_args.backends = ["hip-direct"]
layout_args.scenario = ["layout-search"]
layout_entries = benchmark_sweep.sweep_command_entries(layout_args)
assert len(layout_entries) == 16
assert {entry.scenario["family"] for entry in layout_entries} == {"layout-search"}
assert {entry.scenario["name"] for entry in layout_entries} == {
    "bounded-i64-prefix9-default-final-export",
    "bounded-i64-prefix9-residue-channel-fusion",
    "bounded-i64-prefix9-padded-ld",
    "exact-wide-signed-prefix20-default-final-export",
    "exact-wide-signed-prefix20-independent-export-chain",
    "exact-wide-signed-prefix20-residue-current-chain",
    "finite-ring-hot-modulus-default-layout",
    "finite-ring-hot-modulus-padded-ld",
    "finite-field-hot-prime-default-layout",
    "finite-field-hot-prime-padded-ld",
    "wrap64-direct-byte-default-layout",
    "wrap64-direct-byte-padded-ld",
}
assert all(
    entry.scenario["backend"] != "ck"
    for entry in bound_discovery_entries
    if entry.scenario["semantics"] == "bounded-u64"
)
assert sorted(
    entry.scenario["modulus"]
    for entry in layout_entries
    if entry.scenario["name"] == "finite-ring-hot-modulus-default-layout"
) == [251, 255, 256]
assert any("--next-op-hint" in entry.command and "final-export" in entry.command for entry in layout_entries)
assert any("--prefix-policy" in entry.command and "fixed-requested" in entry.command for entry in layout_entries)
assert any("--max-prefix" in entry.command and "20" in entry.command for entry in layout_entries)
assert any("--residue-channel-fusion" in entry.command for entry in layout_entries)
assert any("--output-ld-padding" in entry.command and "32" in entry.command for entry in layout_entries)
assert any("--residue-chain-length" in entry.command and "3" in entry.command for entry in layout_entries)
assert any(entry.scenario["semantics"] == "wrap-u64" for entry in layout_entries)
assert any(
    entry.scenario.get("metadata", {}).get("layout_role") == "exact_wide_residue_current_chain_layout"
    for entry in layout_entries
)
assert any(
    entry.scenario.get("metadata", {}).get("layout_variant_role") == "candidate"
    and entry.scenario.get("metadata", {}).get("actual_layout_variant") is True
    for entry in layout_entries
)
assert all(
    entry.scenario.get("metadata", {}).get("workflow_name") == "end_to_end_layout_search"
    for entry in layout_entries
)

finite_distribution_args = copy.copy(scenario_args)
finite_distribution_args.backends = ["hip-direct"]
finite_distribution_args.scenario = ["finite-distributions"]
finite_distribution_entries = benchmark_sweep.sweep_command_entries(finite_distribution_args)
assert len(finite_distribution_entries) == 80
assert {entry.scenario["family"] for entry in finite_distribution_entries} == {"finite-distributions"}
assert {entry.scenario["input_profile"] for entry in finite_distribution_entries} == {
    "finite-binary",
    "finite-sparse",
    "finite-low-hamming",
    "finite-small-centered",
    "finite-full-uniform",
}
assert {entry.scenario["shape"]["m"] for entry in finite_distribution_entries} == {128, 512, 1024, 2048}
assert sorted(
    {
        (entry.scenario["semantics"], entry.scenario["modulus"])
        for entry in finite_distribution_entries
        if entry.scenario["shape"]["m"] == 2048
    }
) == [
    ("finite-u8-field", 241),
    ("finite-u8-field", 251),
    ("finite-u8-ring", 251),
    ("finite-u8-ring", 253),
]
assert all("--input-profile" in entry.command for entry in finite_distribution_entries)
assert any(
    entry.scenario.get("metadata", {}).get("workflow_name") == "finite_distribution_release_matrix"
    and entry.scenario.get("metadata", {}).get("distribution_role") == "sparse"
    for entry in finite_distribution_entries
)

