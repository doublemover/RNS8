with tempfile.TemporaryDirectory() as tmp:
    timeout_output = Path(tmp) / "timeout.json"
    ok = benchmark_sweep.run_command(
        [sys.executable, "-c", "import time; time.sleep(5)"],
        timeout_output,
        timeout_seconds=0.01,
    )
    failure = json.loads(timeout_output.with_suffix(".failed.json").read_text(encoding="utf-8"))
    assert ok is False
    assert timeout_output.exists() is False
    assert failure["timed_out"] is True
    assert failure["timeout_seconds"] == 0.01
    assert failure["returncode"] is None

bad_scenario_args = copy.copy(scenario_args)
bad_scenario_args.include_oneshot = True
try:
    benchmark_sweep.sweep_commands(bad_scenario_args)
except SystemExit as exc:
    assert "--scenario cannot be combined" in str(exc)
else:
    raise AssertionError("scenario mode should reject manual include flags")

for attr, value in [
    ("workspace_arena", True),
    ("resident_lifetime", True),
    ("adaptive_grouped_scheduler", True),
    ("streaming_overlap", True),
    ("k_block_policy", "fixed-safe-kblock-cdna-candidate"),
    ("release_gate", "golden_regression_required"),
    ("verification_amortization", "repeat_window"),
    ("error_detection_policy", "freivalds_two_round_product_check_research"),
]:
    bad_scenario_args = copy.copy(scenario_args)
    setattr(bad_scenario_args, attr, value)
    try:
        benchmark_sweep.sweep_commands(bad_scenario_args)
    except SystemExit as exc:
        assert "--scenario cannot be combined" in str(exc)
    else:
        raise AssertionError(f"scenario mode should reject manual {attr}")

exact_chain_args = copy.copy(exact_args)
exact_chain_args.backends = ["cpu"]
exact_chain_args.residue_chain_length = 3
exact_chain_commands = benchmark_sweep.sweep_commands(exact_chain_args)
assert [name for name, _command, _output in exact_chain_commands] == [
    "exact-wide-signed-small-16x16x16-chain3-cpu.json",
]
exact_chain_command = exact_chain_commands[0][1]
assert "--residue-chain-length" in exact_chain_command
assert exact_chain_command[exact_chain_command.index("--residue-chain-length") + 1] == "3"
assert "--residue-chain-final-export" not in exact_chain_command

exact_chain_final_args = copy.copy(exact_chain_args)
exact_chain_final_args.residue_chain_final_export = True
exact_chain_final_args.next_op_hint = "final-export"
exact_chain_final_commands = benchmark_sweep.sweep_commands(exact_chain_final_args)
assert [name for name, _command, _output in exact_chain_final_commands] == [
    "exact-wide-signed-small-16x16x16-chain3-finalexport-cpu.json",
]
exact_chain_final_command = exact_chain_final_commands[0][1]
assert "--residue-chain-final-export" in exact_chain_final_command
assert "--next-op-hint" in exact_chain_final_command
assert exact_chain_final_command[exact_chain_final_command.index("--next-op-hint") + 1] == "final-export"

bounded_independent_chain_args = copy.copy(exact_args)
bounded_independent_chain_args.backends = ["cpu"]
bounded_independent_chain_args.semantics = ["bounded-i64"]
bounded_independent_chain_args.residue_chain_length = 3
bounded_independent_chain_args.residue_chain_independent_final_export = True
bounded_independent_chain_args.next_op_hint = "final-export"
bounded_independent_chain_commands = benchmark_sweep.sweep_commands(bounded_independent_chain_args)
assert [name for name, _command, _output in bounded_independent_chain_commands] == [
    "bounded-i64-small-16x16x16-chain3-indepfinalexport-cpu.json",
]
bounded_independent_chain_command = bounded_independent_chain_commands[0][1]
assert "--residue-chain-independent-final-export" in bounded_independent_chain_command
assert "--residue-chain-final-export" not in bounded_independent_chain_command

exact_independent_chain_args = copy.copy(exact_chain_args)
exact_independent_chain_args.residue_chain_independent_final_export = True
exact_independent_chain_args.next_op_hint = "final-export"
exact_independent_chain_commands = benchmark_sweep.sweep_commands(exact_independent_chain_args)
assert [name for name, _command, _output in exact_independent_chain_commands] == [
    "exact-wide-signed-small-16x16x16-chain3-indepfinalexport-cpu.json",
]
exact_independent_chain_command = exact_independent_chain_commands[0][1]
assert "--residue-chain-independent-final-export" in exact_independent_chain_command
assert "--residue-chain-final-export" not in exact_independent_chain_command

vector_args = copy.copy(exact_args)
vector_args.out_root = Path("temp") / "vector-runtime"
vector_args.backends = ["hip-vector-alu-int64"]
vector_args.semantics = ["bounded-i64"]
vector_args.case = ["small:16,16,16"]
vector_commands = benchmark_sweep.sweep_commands(vector_args)
vector_name, vector_command, _vector_output = vector_commands[0]
assert vector_name == "bounded-i64-small-16x16x16-hip-vector-alu-int64.json"
assert vector_command[vector_command.index("--backend") + 1] == "hip-vector-alu-int64-runtime"

padded_output_args = copy.copy(vector_args)
padded_output_args.backends = ["hip-direct"]
padded_output_args.output_ld_padding = 7
padded_output_commands = benchmark_sweep.sweep_commands(padded_output_args)
padded_name, padded_command, _padded_output = padded_output_commands[0]
assert padded_name == "bounded-i64-small-16x16x16-outpad7-hip-direct.json"
assert "--output-ld-padding" in padded_command
assert padded_command[padded_command.index("--output-ld-padding") + 1] == "7"

input_scan_args = copy.copy(vector_args)
input_scan_args.bound_source = "input-scan"
input_scan_commands = benchmark_sweep.sweep_commands(input_scan_args)
input_scan_command = input_scan_commands[0][1]
assert "--bound-source" in input_scan_command
assert input_scan_command[input_scan_command.index("--bound-source") + 1] == "input-scan"

exact_input_scan_args = copy.copy(exact_args)
exact_input_scan_args.bound_source = "input-scan"
try:
    benchmark_sweep.sweep_commands(exact_input_scan_args)
except SystemExit as exc:
    assert "--bound-source input-scan is only valid for bounded RNS sweeps" in str(exc)
else:
    raise AssertionError("input-scan bound discovery should reject non-bounded sweeps")

oneshot_args = copy.copy(exact_args)
oneshot_args.out_root = Path("temp") / "oneshot"
oneshot_args.backends = ["hip-direct"]
oneshot_args.semantics = ["bounded-i64"]
oneshot_args.case = ["small:16,16,16"]
oneshot_args.include_oneshot = True
oneshot_args.oneshot_only = False
oneshot_commands = benchmark_sweep.sweep_commands(oneshot_args)
assert [name for name, _command, _output in oneshot_commands] == [
    "bounded-i64-small-16x16x16-hip-direct.json",
    "bounded-i64-small-16x16x16-oneshot-hip-direct.json",
]
assert "--oneshot" not in oneshot_commands[0][1]
assert "--oneshot" in oneshot_commands[1][1]

oneshot_only_args = copy.copy(oneshot_args)
oneshot_only_args.oneshot_only = True
oneshot_only_commands = benchmark_sweep.sweep_commands(oneshot_only_args)
assert [name for name, _command, _output in oneshot_only_commands] == [
    "bounded-i64-small-16x16x16-oneshot-hip-direct.json",
]
assert "--oneshot" in oneshot_only_commands[0][1]

finite_oneshot_args = copy.copy(exact_args)
finite_oneshot_args.out_root = Path("temp") / "finite-oneshot"
finite_oneshot_args.backends = ["hip-direct"]
finite_oneshot_args.semantics = ["finite-u8-ring"]
finite_oneshot_args.case = ["small:16,16,16"]
finite_oneshot_args.modulus = [255]
finite_oneshot_args.include_oneshot = True
finite_oneshot_args.oneshot_only = False
finite_oneshot_commands = benchmark_sweep.sweep_commands(finite_oneshot_args)
assert [name for name, _command, _output in finite_oneshot_commands] == [
    "finite-u8-ring-small-16x16x16-mod255-hip-direct.json",
    "finite-u8-ring-small-16x16x16-mod255-oneshot-hip-direct.json",
]
assert "--oneshot" not in finite_oneshot_commands[0][1]
assert "--oneshot" in finite_oneshot_commands[1][1]

finite_oneshot_only_args = copy.copy(finite_oneshot_args)
finite_oneshot_only_args.oneshot_only = True
finite_oneshot_only_commands = benchmark_sweep.sweep_commands(finite_oneshot_only_args)
assert [name for name, _command, _output in finite_oneshot_only_commands] == [
    "finite-u8-ring-small-16x16x16-mod255-oneshot-hip-direct.json",
]
assert "--oneshot" in finite_oneshot_only_commands[0][1]

exact_include_args = argparse.Namespace(
    bench=Path("rns8-bench"),
    bench_for=[],
    out_root=Path("temp") / "exact-wide-release",
    backends=None,
    semantics=None,
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
    include_exact_wide=True,
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
exact_include_commands = benchmark_sweep.sweep_commands(exact_include_args)
exact_include_names = [name for name, _command, _output in exact_include_commands]
assert "exact-wide-signed-small-16x16x16-cpu.json" in exact_include_names
assert "exact-wide-unsigned-small-16x16x16-rocwmma.json" in exact_include_names
assert "exact-wide-unsigned-small-16x16x16-amdgpu-builtins.json" in exact_include_names
assert len(exact_include_commands) == 25
assert not any(
    "--semantics" in command and "bounded-u64" in command and "--backend" in command and "ck" in command
    for _name, command, _output in exact_include_commands
)

adaptive_only_args = argparse.Namespace(
    bench=Path("rns8-bench"),
    bench_for=[],
    out_root=Path("temp") / "adaptive-only",
    backends=["cpu"],
    semantics=["bounded-i64"],
    case=None,
    adaptive_case=None,
    shapes=None,
    modulus=None,
    exact_wide_limbs=None,
    include_exact_wide_limb_variants=False,
    residue_chain_length=1,
    include_default_adaptive=True,
    include_adaptive_workloads=False,
    adaptive_only=True,
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
commands = benchmark_sweep.sweep_commands(adaptive_only_args)
assert len(commands) == 2
assert all("--require-adaptive-execution" in command for _name, command, _output in commands)
assert all("--bound-mode" in command and "per-tile" in command for _name, command, _output in commands)
assert [name for name, _command, _output in commands] == [
    "bounded-i64-tiny-adaptive-65x65x64-cpu.json",
    "bounded-i64-medium-adaptive-1024x1024x1024-cpu.json",
]
adaptive_input_scan_args = copy.copy(adaptive_only_args)
adaptive_input_scan_args.bound_source = "input-scan"
adaptive_input_scan_commands = benchmark_sweep.sweep_commands(adaptive_input_scan_args)
assert len(adaptive_input_scan_commands) == 2
assert all("--bound-source" in command for _name, command, _output in adaptive_input_scan_commands)
assert all(
    command[command.index("--bound-source") + 1] == "input-scan"
    for _name, command, _output in adaptive_input_scan_commands
)
assert all("--bound-mode" in command and "per-tile" in command for _name, command, _output in adaptive_input_scan_commands)

adaptive_only_args.include_default_adaptive = False
try:
    benchmark_sweep.sweep_commands(adaptive_only_args)
except SystemExit as exc:
    assert "--adaptive-only requires" in str(exc)
else:
    raise AssertionError("adaptive-only without adaptive cases should fail")
adaptive_only_args.include_adaptive_workloads = True
workload_commands = benchmark_sweep.sweep_commands(adaptive_only_args)
assert len(workload_commands) == len(benchmark_sweep.ADAPTIVE_WORKLOAD_CASES)
assert all("--input-profile" in command and "adaptive-bands" in command for _name, command, _output in workload_commands)
assert workload_commands[0][0] == "bounded-i64-banded-adaptive-256-256x256x512-cpu.json"
adaptive_only_args.include_adaptive_workloads = False

