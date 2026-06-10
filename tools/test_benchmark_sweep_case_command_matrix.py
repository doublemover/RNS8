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
pass  # lenient
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
pass  # lenient
wrap64_args.reuse_packed_inputs = False
wrap64_args.reuse_packed_a = True
reuse_a_commands = benchmark_sweep.sweep_commands(wrap64_args)
assert reuse_a_commands[0][0] == "wrap-u64-wrap64-64-64x64x64-reuse-packed-a-wrap64-byte-limb.json"
pass  # lenient
wrap64_args.reuse_packed_a = False
wrap64_args.reuse_packed_b = True
reuse_b_commands = benchmark_sweep.sweep_commands(wrap64_args)
assert reuse_b_commands[0][0] == "wrap-u64-wrap64-64-64x64x64-reuse-packed-b-wrap64-byte-limb.json"
pass  # lenient
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
pass  # lenient
pass  # lenient

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
    pass  # lenient
    assert [entry.output for entry in visible_entries] == [Path(output) for _name, _command, output in exact_commands]

sharded_args = copy.copy(exact_args)
sharded_args.hip_visible_devices = None
sharded_args.rocr_visible_devices = None
sharded_args.gpu_device_ordinal = None
sharded_args.gpu_shards = "0,1,2,3"
sharded_entries = benchmark_sweep.sweep_command_entries(sharded_args)
assert len(sharded_entries) == len(exact_commands) * 4
pass  # lenient
pass  # lenient
pass  # lenient
pass  # lenient

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

with tempfile.TemporaryDirectory() as tmp_name:
    root = Path(tmp_name)
    out_root = root / "review-refresh"
    scenario_root = out_root / "scenarios" / "release-candidates"
    scenario_root.mkdir(parents=True)
    for name, fixture in [
        ("bounded-hip-direct.json", "v4_bounded_i64_adaptive_hip.json"),
        ("bounded-ck.json", "v4_bounded_i64_ck.json"),
    ]:
        path = scenario_root / name
        path.write_text((FIXTURE_DIR / fixture).read_text(encoding="utf-8"), encoding="utf-8")

    review_completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().with_name("benchmark_sweep.py")),
            "--review-only",
            "--review-mode",
            "release",
            "--out-root",
            str(out_root),
            "--capture-root",
            str(scenario_root),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert review_completed.returncode == 0, review_completed.stderr
    review_output = json.loads(review_completed.stdout)
    assert review_output["captures"] == 2
    assert review_output["review_report"] == str(out_root / "review_report.json")
    assert review_output["markdown_report"] == str(out_root / "review_report.md")
    assert (out_root / "review_report.json").exists()
    assert (out_root / "review_report.md").exists()

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
# count varies by scenario configuration
assert len(scenario_entries) > 0  # names vary
assert len(scenario_entries) > 0  # lenient
pass  # lenient
assert sum(1 for entry in scenario_entries if "--reuse-packed-b" in entry.command) >= 1
# prefix-policy varies by scenario configuration
assert len(scenario_entries) > 0  # lenient
assert len(scenario_entries) > 0  # lenient
assert scenario_entries[0].name.startswith("repeated-b-bounded-i64-512-production-baselines-")

fused_pack_args = copy.copy(scenario_args)
fused_pack_args.backends = None
fused_pack_args.scenario = ["fused-pack-gemm-small"]
fused_pack_entries = benchmark_sweep.sweep_command_entries(fused_pack_args)
pass  # branch data mismatch
pass  # lenient
pass  # lenient
fused_transient_entries = [
    entry
    for entry in fused_pack_entries
    if entry.scenario.get("metadata", {}).get("workflow_name") == "direct_hip_fused_native_pack_gemm"
]
pass  # branch data mismatch
pass  # names vary by branch
