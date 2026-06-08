from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from metadata_registry_constants import PLACEHOLDER_GPU_TARGET_IDS

BOUNDED_BACKENDS = ["cpu", "hip-direct", "hip-vector-alu-int64", "hipblaslt", "ck", "rocwmma", "amdgpu-builtins"]
HOST_API_BATCH_BACKENDS = ["hip-direct"]
PUBLIC_ONESHOT_BACKENDS = ["cpu", "hip-direct"]
EXACT_WIDE_BACKENDS = ["cpu", "hip-direct", "hipblaslt", "ck", "rocwmma", "amdgpu-builtins"]
FINITE_BACKENDS = ["cpu", "hip-direct", "hipblaslt", "ck", "rocwmma", "amdgpu-builtins"]
WRAP64_BACKENDS = ["wrap64-byte-limb", "hip-direct"]
WRAP64_ROCWMMA_CANDIDATE_BACKEND = "rocwmma-wrap64-candidate"
BOUNDED_SEMANTICS = ["bounded-i64", "bounded-u64"]
EXACT_WIDE_SEMANTICS = ["exact-wide-signed", "exact-wide-unsigned"]
RNS_CHAIN_SEMANTICS = BOUNDED_SEMANTICS + EXACT_WIDE_SEMANTICS
PHASES = ["pack", "rns_gemm", "crt_export", "end_to_end"]
REVIEW_SCHEMA_VERSION = 3
PLACEHOLDER_TARGET_IDS = PLACEHOLDER_GPU_TARGET_IDS
RELEASE_MIN_WARMUPS = 3
RELEASE_MIN_REPEATS = 9
PROMOTABLE_RELEASE_SHAPES = [64, 128, 512, 1024]
EXPLORATORY_RELEASE_SHAPES = [2048, 4096, 8192]
DEFAULT_ADAPTIVE_CASES = [
    "tiny-adaptive:65,65,64,64,64",
    "medium-adaptive:1024,1024,1024,128,128",
]
ADAPTIVE_WORKLOAD_CASES = [
    "banded-adaptive-256:256,256,512,64,64,adaptive-bands",
    "banded-adaptive-1024:1024,1024,1024,128,128,adaptive-bands",
    "banded-rect-adaptive:512,1024,512,128,128,adaptive-bands",
]
DEFAULT_FINITE_RING_MODULI = [251, 255]
DEFAULT_FINITE_FIELD_MODULI = [251]
SCENARIO_DATA_DIR = Path(__file__).resolve().parents[2] / "benchmarks" / "scenarios"
INPUT_PROFILES = {"uniform-small", "adaptive-bands"}
DEFAULT_EXACT_WIDE_LIMB_COUNT = 4
EXACT_WIDE_LIMB_VARIANTS = [1, 2, 3, 4, 8, 16, 32]


@dataclass(frozen=True)
class SweepCase:
    name: str
    m: int
    n: int
    k: int
    tile_m: int = 128
    tile_n: int = 128
    bound_mode: str = "global"
    input_profile: str = "uniform-small"
    require_adaptive: bool = False
    promotable: bool = True


@dataclass(frozen=True)
class ScenarioItem:
    family: str
    name: str
    semantics: str
    case: SweepCase
    evidence_scope: str
    output_domain: str
    rationale: str
    review_mode_expectation: str
    promotion_eligibility: str
    backends: tuple[str, ...] | None = None
    pack_mode: str = "per_repeat_repack"
    finite_moduli: tuple[int | None, ...] = (None,)
    exact_wide_limb_counts: tuple[int | None, ...] = (None,)
    residue_chain_length: int = 1
    residue_chain_final_export: bool = False
    residue_chain_independent_final_export: bool = False
    output_ld_padding: int = 0
    host_api_batch_size: int = 1
    oneshot: bool = False
    native_to_rns_bridge: bool = False
    vector_to_rns_chain: bool = False
    vector_to_rns_chain_host_repack_control: bool = False
    sparse_a_4_to_2: bool = False
    prefix_policy: str | None = None
    max_prefix: int | None = None
    bound_source: str | None = None
    next_op_hint: str | None = None
    residue_channel_fusion: bool = False
    modulus_set: str = "default"
    tile_shape_variant: str = "default"
    export_variant: str = "default"
    reconstruction_variant: str = "default_garner"
    grouped_dispatch_tasks: int = 1
    hip_graph_replay: bool = False
    workload_proxy: str = "none"
    resident_lifetime: bool = False
    workspace_arena: bool = False
    adaptive_grouped_scheduler: bool = False
    streaming_overlap: bool = False
    k_block_policy: str = "auto"
    resident_redesign_candidate: str = ""
    resident_redesign_dimensions: tuple[str, ...] = ()
    release_gate: str = "none"
    verification_amortization: str = "none"
    error_detection_policy: str = "none"
    cpu_small_shape_selector: str = "none"
    incremental_result_cache: str = "none"
    include_wrap64_candidate: bool = False
    metadata: dict[str, Any] | None = None



@dataclass(frozen=True)
class SweepCommand:
    name: str
    command: list[str]
    output: Path
    scenario: dict[str, Any] | None = None
    env: dict[str, str] | None = None


