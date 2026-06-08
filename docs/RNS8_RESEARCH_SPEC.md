# RNS8 Technical Specification

Exact integer matrix multiplication on AMD GPU matrix engines with HIP, RNS,
CRT, and Ozaki-style decomposition.

Date: 2026-06-01

Project name: `RNS8`

Repository name: `rns8-gemm`

## 1. Executive Decision Register

This is a greenfield specification. It is not bound to the current checkout
shape, installed toolchain, or a single AMD GPU generation.

| ID | Decision |
|---|---|
| D1 | Build a new RNS-first library named `RNS8`, packaged as `rns8-gemm`. |
| D2 | Windows and Linux are first-class targets. Windows is the local bring-up path; Linux ROCm remains the full production and cluster path. |
| D3 | The implementation must not require hipBLASLt, CK, or rocWMMA to prove correctness. A portable CPU backend and a direct HIP backend are required first. |
| D4 | The first production semantic target is bounded exact signed and unsigned 64-bit GEMM. |
| D5 | The core representation is persistent residue number system storage, not temporary conversion around a BLAS call. |
| D6 | The core compute primitive is INT8 x INT8 -> INT32 matrix GEMM per modulus. |
| D7 | The default modulus ladder is pairwise-coprime, composite-inclusive, and ordered by descending range contribution. |
| D8 | Full signed or unsigned bounded 64-bit output uses the first 9 moduli by default. |
| D9 | Adaptive per-tile modulus counts are enabled for performance, but correctness never depends on probabilistic early termination. |
| D10 | Strict `mod 2^64` wraparound is implemented by a byte-limb backend, not by odd-modulus CRT alone. |
| D11 | INT4 is not a v1 production backend. It is retained only as a measured research path with a retirement rule. |
| D12 | Target hardware covers AMD Radeon RDNA2/RDNA3/RDNA4 and AMD Instinct CDNA2/CDNA3/CDNA4 where officially supported by the active ROCm or HIP SDK release. |
| D13 | The optimization ladder is portable HIP, hipBLASLt when available, CK grouped/fused kernels when available, then rocWMMA or AMDGPU builtin kernels for hot paths. |
| D14 | Fused INT32-to-residue modulo reduction is required for production performance. Separate INT32 store plus reduction is only a baseline. |
| D15 | GPU CRT reconstruction is required for production bounded `int64` and `uint64` export on supported GPU targets. CPU CRT is the reference and debug path. |
| D16 | Public APIs are explicit about semantics and never infer signed, unsigned, bounded, wide, or wraparound behavior from a C++ type alone. |
| D17 | The build system must use portable CMake for host code and explicit HIP compiler integration that works on Windows, where CMake HIP language support is not available. |
| D18 | First-party code and documentation use the MIT License, with third-party notices for AMD, academic, and CPU reference dependencies. |
| D19 | RNS8 development uses hard cutovers only. Retired APIs, schema fields, backend paths, kernels, tests, docs, and tools are deleted or rewritten in-place instead of preserved as legacy, compatibility, fallback, or shim layers. |

### 1.1 Hard-Cutover Discipline

RNS8 is greenfield. Implementation milestones must land as direct hard
cutovers, not compatibility transitions. When a contract changes, the old
contract is removed from production code in the same slice that installs the
new contract.

Forbidden implementation patterns:

- legacy aliases, compatibility wrappers, adapter classes, or forwarding
  shims for retired APIs,
- fallback runtime paths that silently preserve old semantics after a new
  backend, kernel, schema, or storage layout has replaced them,
- duplicate old/new tests that assert retired behavior continues to work,
- benchmark-schema compatibility paths that validate obsolete current-version
  captures,
- docs that present old behavior as an accepted option after a hard cutover.

Allowed historical artifacts:

- old benchmark captures, logs, and notes may remain under ignored `temp/` or
  in clearly labeled historical evidence sections,
- source comments may briefly name a retired behavior only to explain why it was
  deleted,
- ABI size/version fields may reject incompatible callers or enable future
  fields, but they must not keep retired semantics alive.

Unsupported backends, platforms, or semantic contracts fail explicitly with
their status code. They do not fall through to an older backend, narrower
semantic interpretation, or hidden compatibility mode.

## 2. Product Scope

### 2.1 In Scope

- Exact dense matrix multiplication over integer domains.
- Windows HIP SDK implementation for local Radeon development.
- Linux ROCm implementation for Radeon and Instinct production systems.
- CPU reference backend for correctness and CI.
- Direct HIP GPU backend that can run without hipBLASLt or CK.
- Finite-ring `mod m` GEMM primitive for `2 <= m <= 256`.
- Finite-field `mod p` GEMM primitive for prime `p <= 251`.
- Bounded exact `int64_t` and `uint64_t` GEMM.
- Exact-wide integer GEMM with RNS output.
- Strict wraparound `mod 2^64` GEMM through a separate byte-limb backend.
- Persistent RNS matrix storage and reuse.
- Explicit A-side 4:2 structured sparse input/storage contract for future
  SMFMAC/SWMMAC acceleration. This is a caller-supplied sparse-A contract, not
  automatic pruning or general sparse matrix multiplication.
- Per-tile adaptive modulus counts using deterministic bounds.
- Grouped and persistent scheduling across `(modulus, tile)` work.
- Fused modulo reduction in CK, rocWMMA, AMDGPU builtin, or direct HIP kernels.
- Benchmarks against normal vector-ALU `int64` GPU kernels and CPU exact
  linear-algebra baselines.

### 2.2 Out Of Scope For v1

- Drop-in BLAS interception.
- Approximate integer output.
- General sparse matrix multiplication as a primary product.
- Automatic dense-to-sparse pruning, B-side sparsity, unstructured sparsity,
  and sampled sparse correctness.
- Compiler integration.
- Automatic proof of user-provided numerical bounds.
- Default probabilistic correctness.
- INT4 production kernels.
- Required multi-GPU production support.
- Assuming Windows has the same ROCm component surface as Linux.

## 3. Target Hardware

RNS8 targets hardware by capability tier, not by a single launch GPU.

### 3.1 Bring-Up And Optimization Targets

| Tier | GPU family | Example GPUs | LLVM targets | Role |
|---|---|---|---|---|
| W0 | Radeon RDNA3 | RX 7900 XTX, RX 7900 XT | `gfx1100` | Local Windows bring-up and RDNA3 optimization. |
| W1 | Radeon RDNA4 | RX 9070 XT, RX 9070, RX 9060 XT | `gfx1201`, `gfx1200` | Current consumer matrix-core target. |
| W2 | Radeon RDNA2 | RX 6950 XT, RX 6900 XT, RX 6800 XT | `gfx1030` | Functional HIP regression target where supported; matrix-core acceleration is not assumed. |
| I0 | Instinct CDNA4 | MI355X, MI350X | `gfx950` | Current Instinct production target. |
| I1 | Instinct CDNA3 | MI325X, MI300X, MI300A | `gfx942` | Previous-generation Instinct production target. |
| I2 | Instinct CDNA2 | MI250X, MI250, MI210 | `gfx90a` | Supported cluster target where the active ROCm release supports it. |

Hardware support is accepted only when the active ROCm or HIP SDK release
officially supports that GPU and the needed libraries or compiler intrinsics.
Unsupported GPUs may be used for experiments, but they cannot define release
requirements.

### 3.2 Architecture Policy

- RDNA targets use wave32-aware packing and matrix-core paths where available.
- CDNA targets use wave64-aware packing and MFMA/XDLOPS-oriented paths.
- Backend kernels must be selected by runtime feature detection and a persisted
  autotune key, not by hard-coded GPU names.
- Correctness must pass on CPU before a GPU backend can be accepted.
- Production performance gates are per architecture family. Failing one family
  does not invalidate correctness or releases on another family.

## 4. Required Software Stack

### 4.1 Cross-Platform Core

Required everywhere:

- C++17 compiler.
- CMake 3.22 or newer.
- Ninja or another explicitly supported generator.
- Python 3.11 or newer for benchmarks, plotting, and result analysis.
- Git.
- Boost.Multiprecision headers.
- A test framework selected during scaffold, default `Catch2`.

Optional comparison dependencies:

- GMP or MPIR.
- FLINT.
- NTL.
- FFLAS-FFPACK.
- LinBox.

The optional comparison libraries are not required for core build success.
When present, `rns8-verify` uses them for differential testing.

### 4.2 Windows HIP SDK Stack

Required for Windows GPU execution:

- Windows 11 x86-64 on a HIP SDK-supported GPU.
- AMD HIP SDK, latest production release available for Windows.
- HIP SDK Core, HIP Libraries development files, HIP Runtime Compiler
  development files, and AMD GPU driver.
- Visual Studio 2022 Build Tools with MSVC C++ and Windows SDK.
- CMake and Ninja.
- Python packages: `numpy`, `pandas`, `matplotlib`, `pytest`, and `scipy`.

Windows build rules:

- Do not require CMake `enable_language(HIP)`.
- Compile HIP translation units through `hipcc` or the HIP SDK `clang++`
  frontend selected by a CMake toolchain file.
- Use `hipInfo` and `hipconfig` for capability inspection.
- Treat hipBLASLt, CK, rocWMMA, and debugger/profiler components as
  feature-detected optional accelerators unless AMD documents support for the
  active Windows GPU target.

### 4.3 Linux ROCm Stack

Required for Linux direct-HIP GPU execution:

- ROCm production release matching the supported OS and GPU matrix.
- HIP compiler and runtime.
- ROCm LLVM.
- `rocminfo`, `rocm-smi` or `amd-smi`, and ROCProfiler tooling.
- CMake, Ninja, Python, and CPU reference dependencies.

Feature-detected production/performance accelerators, enabled only after
capability probes and exact differentials pass on the target GPU:

- hipBLASLt for baseline INT8 GEMM when supported on the target GPU.
- Composable Kernel or CK Tile for grouped GEMM and custom epilogues when
  supported on the target GPU.
- rocWMMA or AMDGPU builtins for custom hot kernels when supported.

Linux is the required platform for full Instinct validation, multi-GPU
experiments, production profiling, and cluster reproducibility.

### 4.4 Dependency Detection Contract

The repository must provide:

- `cmake/presets/windows-hip-sdk.json`
- `cmake/presets/linux-rocm.json`
- `tools/rns8_inspect`
- `tools/check_dependencies.py`

The dependency checker reports:

- OS and version.
- GPU name, architecture, and LLVM target.
- HIP/ROCm/HIP SDK version.
- Compiler paths and versions.
- Whether CMake HIP language is available.
- hipBLASLt shallow discovery, optional compile/run probe evidence, and backend
  enablement status.
- CK shallow discovery, optional compile/run probe evidence, and backend
  enablement status.
- rocWMMA shallow discovery, optional compile/run probe evidence, and backend
  enablement status.
- AMDGPU builtin readiness status. This has no shallow discovery-only
  correctness pass: the public backend identity is opt-in and runtime contexts
  require `RNS8_ENABLE_AMDGPU_BUILTINS`, compiled target-specific kernels, exact
  CPU differentials, and ISA evidence.
- Accelerator enablement policy as a first-class readiness object. hipBLASLt
  must report as an explicit opt-in baseline backend only after the dedicated
  build/test preset validates exact CPU/direct-HIP differentials; dependency
  discovery alone remains evidence-only. CK and rocWMMA may report explicit
  opt-in correctness backends only after real compiled kernels, semantic
  coverage, exact CPU/direct-HIP differentials, schema fixtures, and ISA
  evidence exist for the target. AMDGPU builtin enablement must continue to
  report disabled runtime capability with
  `correctness_backend=not_implemented` until real target-specific exact
  correctness kernels exist.
- Correctness-backend validation as a separate readiness object. Candidate
  accelerator evidence must report
  `candidate_evidence_is_correctness_validation=false`, and discovery or tiny
  compile/run probes must not be promoted to enabled correctness backends.
- Exact-wide platform validation scope as a first-class readiness object.
  Windows `gfx1100` exact-wide evidence must not be reported as Linux ROCm,
  Radeon Linux, or Instinct CDNA validation; those entries stay unvalidated
  until a real supported Linux ROCm host runs exact CPU differentials.
- Hard-cut self-check metadata that reports the dependency checker's own scope:
  dependency/readiness reporting only, with no build, test, smoke, schema,
  benchmark, or correctness validation implied by running the checker.
- Boost, GMP/MPIR, FLINT, NTL, FFLAS-FFPACK, and LinBox discovery.
- Python package versions.

## 5. Build Outputs

The repository produces:

- `rns8.dll` and import library on Windows.
- `librns8.so` on Linux.
- `librns8_static.a` or platform equivalent where supported.
- `rns8-bench`: benchmark runner.
- `rns8-verify`: correctness and differential-test runner.
- `rns8-inspect`: device, backend, and autotune cache inspector.
- `tools/install_autotune_cache.py`: validates and merges reviewed release
  autotune cache entries into an explicit or default cache path, with an
  explicit replacement mode for discarding stale or non-reviewed destination
  entries. Cache replacement reviews can require a promotion-ledger artifact
  with optional variance-gate enforcement before any source entry is merged.
- Python package `rns8bench` for benchmark sweeps only.

## 6. Integer Semantics

RNS8 exposes integer semantics explicitly.

```cpp
enum rns8_semantics {
  RNS8_BOUNDED_I64,
  RNS8_BOUNDED_U64,
  RNS8_EXACT_WIDE_SIGNED,
  RNS8_EXACT_WIDE_UNSIGNED,
  RNS8_WRAP_U64_MOD_2_64,
  RNS8_FINITE_RING_U8,
  RNS8_FINITE_FIELD_U8
};
```

### 6.1 Bounded Exact Signed 64-bit

Contract:

```text
C = A * B
C_ij is in [-B_ij, B_ij]
B_ij <= 2^63 - 1
```

Correctness condition for a tile `T`:

```text
M_T = product(selected moduli for T)
M_T > 2 * max_abs_output(T)
```

Full signed 64-bit range uses 9 moduli.

### 6.2 Bounded Exact Unsigned 64-bit

Contract:

```text
C = A * B
0 <= C_ij <= B_ij
B_ij <= 2^64 - 1
```

Correctness condition for a tile `T`:

```text
M_T > max_output(T)
```

Full unsigned 64-bit range uses 9 moduli.

### 6.3 Exact-Wide Signed And Unsigned

For arbitrary signed 64-bit inputs:

```text
|sum_k A_ik * B_kj| <= K * 2^126
M > K * 2^127
required_bits = 127 + ceil(log2(K)) + margin
```

For arbitrary unsigned 64-bit inputs:

```text
sum_k A_ik * B_kj < K * 2^128
M > K * 2^128
required_bits = 128 + ceil(log2(K)) + margin
```

The v1 exact-wide GPU compute path stores RNS output. Reconstruction to
multi-limb integers is supported through explicit little-endian limb export.
Signed exact-wide export interprets the CRT result as a centered exact integer
and uses fixed-width two's-complement limbs. Unsigned exact-wide export
interprets the canonical nonnegative CRT result and uses fixed-width magnitude
limbs. Both export exactly the caller-requested limb width, interpret `ld` as
an element stride, require `limb_count` in `[1, 32]`, and return
`RNS8_RANGE_ERROR` without modifying the destination if that width cannot
represent the reconstructed value. Exact-wide descriptors require
`RNS8_BOUND_NONE`, `bound = 0`, and no tile-bound metadata; they are not bounded
i64/u64 exports and are not strict wrap64 exports. CPU Boost.Multiprecision
reconstruction remains the reference path. Direct HIP reconstructs fixed-width
limbs on device for the supported prefix range and copies only the requested
host limb layout.

### 6.4 Strict Wraparound `mod 2^64`

Contract:

```text
C = A * B mod 2^64
```

This is not implemented with the odd-modulus CRT path unless the caller also
supplies a range bound that makes the exact result recoverable. The production
wraparound backend is:

```text
base-256 limbs
36 low-64-relevant byte-product lanes across the low eight Comba diagonals
Comba diagonal accumulation
delayed carry propagation
low 64-bit export
```

The current public implementation exposes a CPU byte-limb reference backend
through `RNS8_BACKEND_WRAP64_BYTE_LIMB` and a direct-HIP correctness path
through `RNS8_BACKEND_HIP_DIRECT`. Both require `RNS8_WRAP_U64_MOD_2_64`,
`RNS8_BOUND_NONE`, and no CRT prefix. They support
`rns8_gemm_wrap_u64_oneshot` and persistent byte-limb matrices with
`rns8_pack_u64`, `rns8_gemm_wrap_u64`, and `rns8_export_wrap_u64`. The direct
HIP path owns device byte-limb matrix storage and uses an inspectable tiled
byte-limb correctness kernel for comparison against the CPU reference. Each
output sums the low eight Comba product diagonals with device-side signed-INT8
correction algebra for the 36 byte-product pairs that can affect the low 64
bits, then performs carry propagation into the low 64 bits.
Persistent direct-HIP wrap64 GEMM and export require device-current byte limbs:
`rns8_pack_u64` is the host-to-device ingress for inputs, GEMM is the
device-current producer for outputs, and GEMM/export must not upload stale
host-current wrap matrices as a hidden route. RNS residue matrices, bounded CRT
metadata, and odd-modulus export paths remain invalid for strict wrap
descriptors.
Optimized matrix-engine byte-GEMM kernels remain later production milestones.

### 6.5 Finite Ring And Finite Field `uint8_t`

Finite-ring and finite-field GEMM are explicit-modulus contracts:

```text
C = A * B mod q
```

`RNS8_FINITE_RING_U8` accepts an explicit modulus `q` in `[2, 256]`.
`RNS8_FINITE_FIELD_U8` accepts an explicit prime modulus `q <= 251`. Inputs are
canonical `uint8_t` values reduced modulo `q`, and outputs are canonical
`uint8_t` residues. For `q <= 255`, every output is in `[0, q - 1]`; for
`q = 256`, every byte value is canonical.

The finite-field contract is a prime-field `GF(q)` contract only. RNS8 does not
currently implement extension fields such as `GF(2^e)`: `RNS8_FINITE_RING_U8`
with `q = 2^e` is arithmetic in the ring `Z/(2^e)Z`, not arithmetic in a binary
extension field with an irreducible polynomial. Word-size prime fields above
`251` likewise require a future finite backend or explicit multimodular
lowering; they are not implied by the byte-sized finite-u8 API.

Finite APIs do not use the CRT prefix ladder. Descriptors must use
`RNS8_BOUND_NONE`, `bound = 0`, `max_prefix = 0`, no tile-bound metadata, and the
matching finite semantic. CPU and direct HIP implementations pack canonical
bytes to centered residues for the explicit modulus, run the K-split
INT8xINT8->INT32 ring GEMM with fused centered reduction, and export canonical
bytes.

Persistent finite matrices use one resident centered-residue plane with the
modulus supplied explicitly to pack, GEMM, and export. Successful pack stamps
the matrix with that modulus; resident GEMM and export reject cross-modulus
inputs or stale outputs. Finite resident storage is prefix-zero storage and is
not an odd-modulus CRT route, exact-wide export route, or strict `mod 2^64`
byte-limb route.

## 7. Modulus Ladder

### 7.1 Default Ordered Set

The default CRT ladder is:

```text
256, 255, 253, 251, 247, 239, 233, 229,
227, 223, 217, 211, 199, 197, 193, 191,
181, 179, 173, 167, 163, 157, 151, 149,
139, 137, 131, 127
```

All values are `<= 256` and pairwise coprime in this order. Composite values
are intentional because CRT requires pairwise coprime rings, not fields.
Prefix selection uses the strict condition `product(moduli[0:s]) > range`.
Therefore prefix 8 is not sufficient for full signed or unsigned 64-bit bounded
output, while prefix 9 is the first default ladder prefix that satisfies both
`2 * 2^63` for signed magnitude bounds and `UINT64_MAX` for unsigned bounds.

### 7.2 Prefix Range Table

| Prefix | Last modulus | Range bits | Use |
|---:|---:|---:|---|
| 4 | 251 | 31.949 | small exact outputs |
| 5 | 247 | 39.897 | 40-bit outputs |
| 6 | 239 | 47.798 | 48-bit outputs |
| 7 | 233 | 55.662 | 56-bit outputs |
| 8 | 229 | 63.502 | signed 63-bit nonnegative or narrower signed bounded outputs |
| 9 | 227 | 71.328 | full signed and unsigned 64-bit bounded outputs |
| 10 | 223 | 79.129 | extra guard range |
| 12 | 211 | 94.612 | medium exact-wide outputs |
| 16 | 191 | 125.040 | fixed 125-bit bounded outputs |
| 18 | 179 | 140.024 | arbitrary signed 64-bit inputs up to K=8192 |
| 19 | 173 | 147.458 | arbitrary signed 64-bit inputs up to K=1048576 |
| 20 | 167 | 154.842 | arbitrary signed 64-bit inputs up to K=134217728 |

### 7.3 Centered Residues

Each residue is stored as signed `int8_t` in centered form:

```text
[-floor(m / 2), ceil(m / 2) - 1]
```

Examples:

- `m = 256`: `[-128, 127]`
- `m = 255`: `[-127, 127]`
- `m = 251`: `[-125, 125]`

Centered residues reduce product magnitude and increase safe INT32
accumulation length.

## 8. Core Algorithm

For each selected modulus `m_i`:

```text
A_i = center(A mod m_i)
B_i = center(B mod m_i)
R_i = A_i * B_i mod m_i
```

Each raw GEMM uses INT8 operands and INT32 accumulation:

```text
int8 x int8 -> int32
```

### 8.1 INT32 Accumulation Limit

The production K-block limit is:

```text
K_block <= 65536
```

This bound is valid for the default centered modulus ladder because:

```text
65536 * 128 * 128 = 1073741824 < 2^31 - 1
```

Large-K GEMM is split into K-blocks. After each block, the partial INT32 sum is
reduced modulo `m_i` and accumulated into the residue accumulator.

### 8.2 Modular Reduction

Production epilogue:

```text
int32 partial
+ previous int8 residue accumulator
-> reduce modulo constant m
-> recenter to int8
-> store int8
```

Reduction implementation order:

1. Special-case `m = 256`.
2. Special-case named finite moduli when the algebra is simple and exact.
   Direct HIP currently names `m = 251`, using byte folding from
   `256 == 5 (mod 251)`, `m = 255`, using byte-sum folding from
   `256 == 1 (mod 255)`, and `m = 256`, using the low byte directly. These
   report `direct_hip_tiled_finite_u8_gemm_mod251_v1`,
   `direct_hip_tiled_finite_u8_gemm_mod255_v1`, and
   `direct_hip_tiled_finite_u8_gemm_mod256_v1` with
   `rns8_hip_direct_finite_specialized_reducer_isa_gate_no_divide` plan
   evidence.
3. Constant reciprocal multiply-high reduction for all other fixed moduli.
4. Branchless correction into canonical centered range.
5. Barrett reduction only where reciprocal reduction fails benchmark gates.

No production kernel writes full INT32 output matrices to global memory except
for baseline backends.

### 8.3 CRT Reconstruction

Bounded 64-bit export uses fixed-limb mixed-radix Garner reconstruction.

Implementation details:

- Prefixes up to 16 moduli use two 64-bit limbs for intermediate values.
- Prefixes 17 through 20 use three 64-bit limbs.
- CPU reference uses Boost.Multiprecision.
- GPU export for bounded `int64_t` and `uint64_t` is required on supported GPU
  targets before a target is considered production-ready.
- GPU export for exact-wide multi-limb output uses the same fixed-limb Garner
  strategy after bounded export passes all correctness gates.

Signed reconstruction:

```text
if x > M / 2:
  output = x - M
else:
  output = x
```

Unsigned reconstruction:

```text
output = x
```

The API returns `RNS8_RANGE_ERROR` when the supplied bound does not satisfy the
selected modulus product.

Computational-algebra rational reconstruction is a separate optional export
surface if added in the future. It must be requested through an explicit
semantic or API, with its own denominator, failure, and verification metadata.
It must not reinterpret bounded `int64_t`/`uint64_t` export, exact-wide limb
export, finite-u8 output, or strict wrap64 output.

## 9. Bounds And Adaptive Moduli

The caller must provide one of:

```cpp
enum rns8_bound_kind {
  RNS8_BOUND_GLOBAL_MAX_ABS,
  RNS8_BOUND_GLOBAL_MAX_UNSIGNED,
  RNS8_BOUND_PER_TILE_MAX_ABS,
  RNS8_BOUND_PER_TILE_MAX_UNSIGNED,
  RNS8_BOUND_INPUT_RANGE_AND_K,
  RNS8_BOUND_NONE
};
```

Rules:

- `RNS8_BOUNDED_I64` rejects `RNS8_BOUND_NONE`.
- `RNS8_BOUNDED_U64` rejects `RNS8_BOUND_NONE`.
- `RNS8_BOUND_NONE` selects exact-wide or wraparound semantics only.
- `RNS8_BOUND_INPUT_RANGE_AND_K` is a bounded i64/u64 plan descriptor
  contract, not a persistent matrix storage kind. `rns8_gemm_desc.bound` must
  be zero at the API boundary, `tile_bounds` must be null, and
  `lhs_bound`/`rhs_bound` provide trusted per-operand input magnitude limits.
  Plan creation derives the effective output bound as
  `k * lhs_bound * rhs_bound`, rejects products outside the requested bounded
  output semantic, stores that derived value in the plan, and schedules the
  same minimum proven prefix as an equivalent global output-bound contract.
  Persistent A/B/C matrices for such plans use ordinary
  `RNS8_BOUND_GLOBAL_MAX_ABS` or `RNS8_BOUND_GLOBAL_MAX_UNSIGNED` storage.
- User-provided bounds are trusted contract inputs and checked by debug
  verification runs when enabled.

The default adaptive bound tile is `128 x 128` output elements. The tile size
is configurable in powers of two from 64 to 512.

For `RNS8_BOUND_PER_TILE_MAX_ABS` and `RNS8_BOUND_PER_TILE_MAX_UNSIGNED`,
`rns8_gemm_desc.bound` must be zero and `rns8_gemm_desc.tile_bounds` points to
row-major output-tile bounds with
`ceil(m / tile_m) * ceil(n / tile_n)` entries. `rns8_create_plan` copies this
array, so the caller only needs to keep it alive through plan creation.
Current CPU reference and direct HIP plans execute and export with these
per-tile selected prefixes. Direct HIP support is a correctness path with
grouped direct tile launches and tile-local device CRT export; direct HIP
tiled wrappers must reject non-covering tile grids, duplicate tile coordinates,
stale selected-prefix group metadata, and invalid prefix metadata before launch
or export-buffer growth. Zero-output tiles may have zero range bits and remain
valid when their required and selected prefixes are otherwise well formed.
Direct HIP all-zero scheduled exports zero-fill the compact native export
buffer directly and do not upload tile schedule/bounds metadata for the export
because no tile-local CRT work can run on that contract.
Per-tile bounded descriptors may also opt into
`RNS8_PLAN_ALLOW_PROVEN_ZERO_ROW_COL_SKIPS` with `zero_a_rows` and
`zero_b_cols` proof masks whose lengths are exactly `m` and `n`. A set A-row
mask byte proves that every output cell in that row is zero for the specific
input pair; a set B-column mask byte proves the same for that output column.
These masks are trusted caller or benchmark proof metadata, not inferred or
verified by RNS8 during plan creation. Plan creation copies the masks, reports
aggregate proof counts through `rns8_plan_schedule_info`, marks intersecting
tiles with `RNS8_TILE_SCHEDULE_ZERO_ROW_COL_PRODUCT`, and includes the copied
masks in workspace fingerprints. Direct HIP scheduled GEMM and scheduled
bounded export can use those uploaded masks to write proven-zero row/column
products without doing the corresponding dot product or CRT reconstruction.
Optimized matrix engine grouped kernels remain a separate validation target.

For each tile:

```text
required_prefix(T) = smallest s such that product(moduli[0:s]) > range(T)
```

where signed bounded range uses `2 * max_abs_output(T)` and unsigned bounded
range uses `max_output(T)`.

Default exact APIs do not use early termination. Research early termination
requires explicit probabilistic mode, verification primes, RNG seed recording,
and false-acceptance metadata.

## 10. Storage Layout

### 10.1 Persistent Matrix Type

```cpp
struct rns8_matrix_desc {
  int64_t rows;
  int64_t cols;
  int64_t logical_ld;
  rns8_semantics semantics;
  rns8_layout logical_layout;
  rns8_bound_kind bound_kind;
  uint32_t tile_m;
  uint32_t tile_n;
  uint32_t max_prefix;
};
```

`Rns8Matrix` owns:

- canonical residue storage,
- backend-packed storage,
- per-tile required prefix,
- per-tile bounds,
- source version token,
- architecture-specific packing metadata,
- autotune result key.

Canonical residue storage is modulus-major:

```text
residue[modulus][tile_m][tile_n][element]
```

Backend-packed storage is also modulus-major:

```text
packed[modulus][macro_tile][fragment]
```

Device allocations are 256-byte aligned. Packed panels are padded to backend
tile multiples. Workspace is caller-owned through `rns8_workspace`. Temporary
allocations inside hot calls are forbidden after plan creation.

Workspaces are bound to the plan contract that created them. Backend, shape,
prefix, semantics, bound kind, bound value, input-range bounds, tile geometry,
selected-prefix schedule metadata, copied per-tile schedule identity, and
copied zero row/column proof-mask identity must match before a workspace can be
used for GEMM. Same-shape workspaces from bounded, exact-wide, input-range
bounded, per-tile bounded, wrap64, or different per-tile schedule/proof-mask
contracts are rejected instead of being reused across semantic boundaries.
Per-tile bounded matrices must also carry the plan's tile geometry before
GEMM/export dispatch.

## 11. Public API Specification

The ABI is C. Public structs include `struct_size` and `abi_version` fields for
validation and future extension. These fields are not compatibility shims:
callers using retired layouts or retired semantics are rejected rather than
translated.

```c
typedef struct rns8_context rns8_context;
typedef struct rns8_plan rns8_plan;
typedef struct rns8_matrix rns8_matrix;
typedef struct rns8_workspace rns8_workspace;

rns8_status rns8_create_context(
    int device_id,
    const rns8_context_options* options,
    rns8_context** out);

rns8_status rns8_destroy_context(rns8_context* ctx);

rns8_status rns8_get_device_info(
    rns8_context* ctx,
    rns8_device_info* out);

rns8_status rns8_create_plan(
    rns8_context* ctx,
    const rns8_gemm_desc* desc,
    rns8_plan** out);

rns8_status rns8_destroy_plan(rns8_plan* plan);

rns8_status rns8_get_plan_schedule_info(
    const rns8_plan* plan,
    rns8_plan_schedule_info* out);

rns8_status rns8_get_plan_tile_schedule(
    const rns8_plan* plan,
    rns8_plan_tile_schedule_entry* entries,
    uint64_t capacity,
    uint64_t* written);

rns8_status rns8_create_workspace(
    rns8_context* ctx,
    const rns8_plan* plan,
    rns8_workspace** out);

rns8_status rns8_destroy_workspace(rns8_workspace* workspace);

rns8_status rns8_create_matrix(
    rns8_context* ctx,
    const rns8_matrix_desc* desc,
    rns8_matrix** out);

rns8_status rns8_destroy_matrix(rns8_matrix* matrix);

rns8_status rns8_pack_i64(
    rns8_context* ctx,
    rns8_matrix* matrix,
    const int64_t* src,
    int64_t ld,
    uint64_t source_version);

rns8_status rns8_pack_u64(
    rns8_context* ctx,
    rns8_matrix* matrix,
    const uint64_t* src,
    int64_t ld,
    uint64_t source_version);

rns8_status rns8_gemm_rns(
    rns8_context* ctx,
    const rns8_plan* plan,
    const rns8_matrix* A,
    const rns8_matrix* B,
    rns8_matrix* C,
    rns8_workspace* workspace);

rns8_status rns8_export_i64(
    rns8_context* ctx,
    const rns8_plan* plan,
    const rns8_matrix* C,
    int64_t* dst,
    int64_t ld);

rns8_status rns8_export_u64(
    rns8_context* ctx,
    const rns8_plan* plan,
    const rns8_matrix* C,
    uint64_t* dst,
    int64_t ld);

rns8_status rns8_gemm_wrap_u64(
    rns8_context* ctx,
    const rns8_plan* plan,
    const rns8_matrix* A,
    const rns8_matrix* B,
    rns8_matrix* C,
    rns8_workspace* workspace);

rns8_status rns8_export_wrap_u64(
    rns8_context* ctx,
    const rns8_plan* plan,
    const rns8_matrix* C,
    uint64_t* dst,
    int64_t ld);

rns8_status rns8_export_exact_wide_signed_limbs(
    rns8_context* ctx,
    const rns8_plan* plan,
    const rns8_matrix* C,
    uint64_t* dst,
    int64_t ld,
    uint32_t limb_count);

rns8_status rns8_export_exact_wide_unsigned_limbs(
    rns8_context* ctx,
    const rns8_plan* plan,
    const rns8_matrix* C,
    uint64_t* dst,
    int64_t ld,
    uint32_t limb_count);

rns8_status rns8_gemm_i64_oneshot(
    rns8_context* ctx,
    const rns8_gemm_desc* desc,
    const int64_t* A,
    int64_t lda,
    const int64_t* B,
    int64_t ldb,
    int64_t* C,
    int64_t ldc);

rns8_status rns8_gemm_u64_oneshot(
    rns8_context* ctx,
    const rns8_gemm_desc* desc,
    const uint64_t* A,
    int64_t lda,
    const uint64_t* B,
    int64_t ldb,
    uint64_t* C,
    int64_t ldc);

rns8_status rns8_gemm_wrap_u64_oneshot(
    rns8_context* ctx,
    const rns8_gemm_desc* desc,
    const uint64_t* A,
    int64_t lda,
    const uint64_t* B,
    int64_t ldb,
    uint64_t* C,
    int64_t ldc);

rns8_status rns8_pack_finite_u8(
    rns8_context* ctx,
    rns8_matrix* matrix,
    uint16_t modulus,
    const uint8_t* src,
    int64_t ld,
    uint64_t source_version);

rns8_status rns8_gemm_finite_u8(
    rns8_context* ctx,
    const rns8_plan* plan,
    uint16_t modulus,
    const rns8_matrix* A,
    const rns8_matrix* B,
    rns8_matrix* C,
    rns8_workspace* workspace);

rns8_status rns8_export_finite_u8(
    rns8_context* ctx,
    const rns8_plan* plan,
    uint16_t modulus,
    const rns8_matrix* C,
    uint8_t* dst,
    int64_t ld);

rns8_status rns8_gemm_finite_ring_u8_oneshot(
    rns8_context* ctx,
    const rns8_gemm_desc* desc,
    uint16_t modulus,
    const uint8_t* A,
    int64_t lda,
    const uint8_t* B,
    int64_t ldb,
    uint8_t* C,
    int64_t ldc);

rns8_status rns8_gemm_finite_field_u8_oneshot(
    rns8_context* ctx,
    const rns8_gemm_desc* desc,
    uint16_t modulus,
    const uint8_t* A,
    int64_t lda,
    const uint8_t* B,
    int64_t ldb,
    uint8_t* C,
    int64_t ldc);

const char* rns8_status_string(rns8_status status);
```

The original plan-only pack sketch was replaced during the Phase 0 scaffold:
packing needs an explicit matrix descriptor because A, B, and C have different
dimensions. Hidden pack-role inference is not allowed in the ABI.
For bounded RNS matrices, `source_version` is caller-supplied pack metadata.
For HIP-resident direct-input storage, repeating a successful pack with the
same nonzero `source_version`, matching semantic storage, and already-current
device data is an idempotent no-op; a changed source must use a changed source
version so the backend performs a new H2D pack. `rns8_get_plan_packing_info`
reports this capability through source-versioned input and same-version pack
elision flags.
Successful bounded persistent GEMM writes an internal deterministic output
version to C from the packed A/B source versions, and rejected GEMM dispatch
must not mutate C's existing version.
Native-to-RNS device handoff requires a nonzero native producer source version
and copies that version onto the materialized Direct-HIP RNS input; zero-version
native outputs are treated as stale/unidentified producers and rejected.

Exact-wide limb export layout is row-major by element. `ld` is a leading
dimension in matrix elements, not in limbs. For element `(row, col)`, limb
`limb` is stored at:

```text
dst[((row * ld) + col) * limb_count + limb]
```

Limb `0` is the least significant 64 bits. Signed export is two's-complement in
exactly `limb_count` limbs, where `limb_count` is in `[1, 32]`, and returns
`RNS8_RANGE_ERROR` unless the centered integer fits:

```text
-2^(64 * limb_count - 1) <= value <= 2^(64 * limb_count - 1) - 1
```

The signed centered CRT representative uses the same threshold convention as
per-modulus centered residues: a canonical reconstruction `x` maps to
`x - P` when `x >= ceil(P / 2)`, where `P` is the selected modulus product.
For even products, the exact `P / 2` residue class is therefore represented as
`-P / 2`, not as a positive value.

Unsigned export is magnitude in exactly `limb_count` limbs, where `limb_count`
is in `[1, 32]`, and returns `RNS8_RANGE_ERROR` unless the canonical integer
fits:

```text
0 <= value <= 2^(64 * limb_count) - 1
```

These APIs do not truncate, wrap, or infer bounded 64-bit behavior from the
destination type. They are separate from bounded i64/u64 export and strict
wraparound semantics.
Null `ctx`, `plan`, matrix, or destination pointers, `limb_count` outside
`[1, 32]`, and output leading dimensions smaller than the matrix width are
malformed ABI calls and return `RNS8_INVALID_ARGUMENT`.
`RNS8_RANGE_ERROR` is all-or-nothing for the caller's destination: CPU export
stages reconstructed cells before writing the padded host layout, and direct HIP
checks device status before copying compact output back to host. Direct HIP
exact-wide export requires device-current resident RNS output and rejects
host-current stale device residues instead of using export as a hidden upload
route.

Exact-wide descriptors use `RNS8_BOUND_NONE`, `bound = 0`, and no tile-bound
storage. Global bounded descriptors also carry no tile-bound pointer/count.
Stale bound metadata is rejected as `RNS8_INVALID_ARGUMENT` rather than ignored
or reported as an unsupported backend.
The production target for `RNS8_BACKEND_AUTO` is reviewed-evidence selection,
not semantic coercion. A valid AUTO plan may select a concrete backend only when
the reviewed autotune key exactly matches the explicit semantic contract, shape,
layout, target id, HIP or accelerator library version, prefix schedule, K-block,
tile size, epilogue, and kernel family. It must reject unvalidated, stale,
debug-only, wrong-target, wrong-version, wrong-shape, or wrong-semantic cache
entries with inspectable rationale. hipBLASLt, CK, and rocWMMA reviewed entries
are residue-current accelerator selections; `hip-vector-alu-int64` reviewed
entries are accepted only for bounded i64/u64 final/native-output plans. Without
a validated exact cache hit or reviewed fastest accelerator entry, AUTO uses the
direct-HIP GPU correctness path for GPU-supported RNS and wrap64 semantics, and
uses CPU only when GPU support is unavailable. It does not translate valid
semantic descriptors across backend families: CPU fallback does not reinterpret
wrap64 as bounded CRT, and wrap64 storage does not route bounded or exact-wide
descriptors to an unrelated RNS path. finite-u8 descriptors carry the explicit
finite modulus in the plan descriptor and autotune key, so reviewed-cache AUTO
selection is shape-and-modulus scoped and cannot alias different rings or
fields.

Strict wrap output is row-major `uint64_t` with caller-supplied leading
dimension in both the one-shot API and `rns8_export_wrap_u64`. It is a
finite-ring low-64-bit result and does not report CRT range errors. Descriptors
carrying bounds or CRT prefixes are rejected instead of being interpreted as
odd-modulus CRT metadata. Valid descriptors on a backend that does not implement
the requested semantic return `RNS8_UNSUPPORTED_BACKEND`; malformed descriptors
return `RNS8_INVALID_ARGUMENT`.

Finite ring/field output is row-major `uint8_t` with caller-supplied leading
dimensions. The public finite APIs require an explicit modulus argument;
`max_prefix` must be zero because the CRT ladder is not a finite-modulus
selector. Persistent finite pack stamps matrices with the explicit modulus, and
persistent finite GEMM/export reject mismatched or stale matrix modulus state.
Ring moduli outside `[2, 256]`, non-prime field moduli, field moduli above 251,
stale bounds, tile-bound metadata, wrong finite semantics, and attempts to use
finite handles through bounded RNS/CRT or wrap64 APIs return
`RNS8_INVALID_ARGUMENT`. Valid finite descriptors requesting unsupported
backends return `RNS8_UNSUPPORTED_BACKEND`.

Required status codes:

- `RNS8_SUCCESS`
- `RNS8_INVALID_ARGUMENT`
- `RNS8_UNSUPPORTED_OS`
- `RNS8_UNSUPPORTED_ARCH`
- `RNS8_UNSUPPORTED_BACKEND`
- `RNS8_RANGE_ERROR`
- `RNS8_ACCUMULATION_OVERFLOW_RISK`
- `RNS8_WORKSPACE_TOO_SMALL`
- `RNS8_BACKEND_FAILURE`
- `RNS8_VERIFICATION_FAILED`
- `RNS8_INTERNAL_ERROR`

`rns8_status_string` is the stable user-visible text surface for these status
codes. It returns lowercase diagnostic strings for every public status and
`unknown status` for out-of-range status values. CLI tools may add context such
as the requested backend name, but they must not reinterpret
`RNS8_UNSUPPORTED_BACKEND` as successful cross-routing or as evidence that
an accelerator correctness backend exists.

Backend capability and plan backend metadata are public ABI surfaces. Backend
capability queries report whether a backend is a correctness backend, an
accelerator candidate, compiled, exact-differential validated, performance
validated, feature-detected, fail-fast, or evidence-only. Plan backend metadata
reports the selected kernel, accelerator library/version, capability status,
epilogue mode, workspace mode, workspace byte requirement, ISA evidence, and
autotune key. Workspaces must preserve this metadata from the plan, and runtime
validation must reject mismatched metadata instead of silently routing through a
different backend path.
Plan packing metadata also reports the selected input/output domain,
host/device output currentness expectation, and next-operation flags. These
flags distinguish final export, residue-current RNS GEMM continuation,
native-current bounded GEMM continuation, native-to-RNS conversion eligibility,
and reusable B prepack availability without inferring semantics from a C++ type
or backend name alone.
Internal plan-lowering diagnostics derive an operation-level description from
the same backend, packing, and schedule metadata. The first description is
`MatMul`-focused and records semantic contract, backend family, desired output,
schedule strategy, packing strategy, reuse strategy, conversion strategy, and
lowering path for inspect/debug tooling. This is planning vocabulary, not a new
public algebra API.

Thread-safety rules:

- Contexts are not internally synchronized.
- One context is used by one host thread at a time.
- Multiple contexts on separate streams are supported.
- Plan and matrix descriptors are immutable after creation.

## 12. Backend Architecture

### 12.1 Backend Ladder

| Stage | Backend | Windows | Linux | Production role |
|---|---|---|---|---|
| B0 | CPU scalar and multiprecision | required | required | correctness reference |
| B1 | Direct HIP vector/matrix baseline | required | required | portable GPU correctness baseline |
| B2 | Direct HIP fused modulo kernels | required | required | portable fused path |
| B3 | hipBLASLt INT8 per modulus | optional by feature detection | optional by feature detection | vendor baseline |
| B4 | hipBLASLt grouped/batched | optional by feature detection | optional by feature detection | launch amortization |
| B5 | CK grouped GEMM | optional by feature detection | optional by feature detection | adaptive scheduling |
| B6 | CK custom epilogue | optional by feature detection | optional by feature detection | fused reduction |
| B7 | rocWMMA or AMDGPU builtins | optional by feature detection | optional for hot production targets | architecture hot paths |
| B8 | Native vector-ALU bounded i64/u64 | HIP feature detection | bounded i64/u64 only | no-RNS same-contract GPU baseline |

The direct HIP backend exists to prevent the project from being blocked by
library availability differences between Windows and Linux.

Windows `gfx1100` performance validation also includes
`hip-vector-alu-int64` for bounded i64/u64. The public runtime backend is
bounded-only and owns compact native HIP device storage rather than persistent
RNS residue planes. It provides same-contract GPU vector-ALU evidence, with
exact 192-bit-limb accumulation and direct logical output export, before a
matrix-engine backend can claim speedup. It must not be used for exact-wide,
finite, or wrap64 semantics; strict `mod 2^64` wraparound remains a byte-limb
backend responsibility.

Feature detection does not enable a backend. hipBLASLt is the baseline
correctness exception only under the explicit `RNS8_ENABLE_HIPBLASLT=ON`
build/test preset; discovery still remains candidate evidence. CK and rocWMMA
become opt-in correctness backends only after compiled kernels, explicit
semantic coverage, exact CPU differentials, direct-HIP differentials,
benchmark-schema coverage, and target ISA evidence exist. Production promotion
still requires measured performance evidence. AMDGPU builtin paths remain
accelerator candidates until the same correctness and evidence gates exist.
CK integration patches to pinned repo-local CK headers must be deterministic,
exact-match guarded at configure time, emitted as build-tree include overlays,
and tracked as source dependencies for the compiled HIP object they affect. If
the expected upstream or patched header block is not present, the CK enable
flag must fail fast instead of compiling an untested CK variant.
Accelerator runtime enablement must continue to fail explicitly while only
evidence probes exist. The test suite should include coverage so discovery
probes cannot become placeholder correctness backends.

### 12.2 Backend Selection Policy

At plan creation, RNS8 selects the first backend that satisfies:

1. OS support,
2. architecture support,
3. compiler support,
4. data layout support,
5. semantic support,
6. workspace limit,
7. autotune result.

The default sequence is:

```text
CPU reference for verification
direct HIP correctness backend
direct HIP fused backend
hipBLASLt baseline after explicit exact CPU/direct-HIP differentials
CK only after a real exact correctness backend exists and is faster
rocWMMA/builtins only after target-specific exact hot kernels exist
```

Production AUTO selection is stricter than this ladder. It is:

```text
exact validated autotune cache hit for the full plan key
reviewed fastest accelerator entry for the same full plan key
direct HIP correctness fallback on available GPU
CPU reference only when GPU support is unavailable
```

Dependency discovery, raw benchmark captures, and unreviewed cache entries are
never backend enablement signals. An accelerator dependency can be present,
compile-probed, and reported by `rns8-inspect` while still being ineligible for
AUTO selection. The implementation status of this policy is tracked in
`docs/roadmap-status.md`; the selector may consume reviewed release cache
entries for HIP-resident accelerator candidates, including finite-u8 entries
whose full plan keys include the explicit modulus, after those entries have
been installed by the reviewed-cache installer. `rns8-bench --backend auto` is
an allowed validation surface for reviewed-cache runtime selection: captures
must report `backend_requested=auto`, the concrete `backend_selected`, and
reviewed-release comparison metadata when the selected plan is
`performance_validated=true`. Hermetic AUTO validation must be able to redirect
the default cache root and prove selection without relying on the developer
machine's real cache state.

### 12.3 Signedness And INT8 Contract

Centered residues use signed `int8_t`. Backends that expose only signed INT8
GEMM can consume centered residues directly.

The wraparound byte-limb backend uses unsigned byte semantics. If a backend
only supports signed INT8 matrix multiply, byte limbs are transformed through a
documented biasing correction or routed to a backend with native unsigned/mixed
signed support. This signedness path is part of correctness testing and cannot
be hidden inside backend-specific assumptions.

### 12.4 Accelerator Control Plane

Accelerator plan metadata is backend-owned and immutable after plan creation.
Every concrete accelerator plan records selected backend, selected kernel,
target id, accelerator library version, workspace bytes, epilogue mode, ISA
evidence state, validation status, and the full autotune key. Workspaces are
created for a specific plan and must reject mismatched backend metadata instead
of silently sharing scratch buffers across semantic contracts or backend
families.

Disabled accelerator descriptors must describe the real state of the backend:
missing dependency, present but unprobed, compile-probed dependency evidence,
compiled opt-in correctness backend, ISA validated, release-reviewed, or
AUTO-promoted. Phrases such as "pending" or "evidence-only" are acceptable only
when they describe the precise unavailable state; they must not appear as
selected kernels for implemented opt-in correctness backends.

Malformed descriptors always win status precedence. Invalid ABI sizes, unknown
semantics, stale bound metadata, unsupported finite modulus arguments, invalid
tile-bound storage, and wrong layout values return `RNS8_INVALID_ARGUMENT`
before accelerator support checks or fail-fast paths run.

### 12.5 Packed Low-Bit Matrix-Engine Pipeline

The packed low-bit pipeline is the long-term path toward near-ideal matrix
engine utilization while preserving exact arithmetic semantics. It is a
separate roadmap track, not a license to approximate the public exact APIs.

Persistent layout versions:

- `rns_i8_modulus_major_v2`: centered `int8_t` RNS planes stored by modulus,
  with backend-aligned leading dimensions and explicit source-version stamps.
- `rns_i8_tile_swizzled_b_v1`: pre-transposed or tile-swizzled B panels for
  repeated-B workloads and rocWMMA/CK hot kernels.
- `finite_u8_centered_plane_v2`: canonical `uint8_t` finite inputs packed into
  centered signed residue planes for matrix-engine consumption, with explicit
  modulus metadata.
- `wrap64_byte_limb_gemm36_v2`: unsigned byte-limb panels containing only the
  36 low-64-relevant byte-product diagonals, with signed-INT8 correction
  metadata when native unsigned byte matrix multiply is unavailable.
- `rns_i4_packed_v0`: research-only INT4/IU4 packing for narrow residue subsets
  or finite fields; never production until exactness and performance gates are
  met.

Prepack caches are keyed by matrix source version, backend, target id, layout
version, tile shape, K-block, modulus, prefix schedule hash, operand role,
transpose state, and epilogue family. Cache hits are valid only when the source
matrix version is unchanged and the exact backend layout contract matches the
selected plan. The benchmark model must separate one-shot packing cost from
repeated-A, repeated-B, and repeated-A/B amortized workloads.

Production dataflow target:

```text
host or resident source matrix
explicit semantic pack or prepack
matrix-engine GEMM over selected moduli or byte pairs
fused residue, finite-u8, exact-wide, or low64 epilogue
checked export or persistent RNS output stamp
```

INT8 is the default production low-bit lane for bounded RNS, exact-wide RNS
output, and finite-u8. CK owns grouped/fused production scheduling, rocWMMA owns
RNS8 hot kernels for `gfx1100`, hipBLASLt remains a baseline or shape-specific
winner, and AMDGPU builtins are admitted only when a measured bottleneck needs
instruction-level control. Global INT32 scratch is a baseline/debug artifact
and cannot be the production path for fused CK/rocWMMA/builtin kernels unless a
reviewed target proves it faster than direct fused reduction.

Strict wrap64 remains separate from odd-modulus CRT. A matrix-engine wrap64
candidate must compute the 36 byte-product pairs that can affect the low 64
bits, apply unsigned-byte or signed-correction algebra explicitly, fuse
diagonal accumulation and carry propagation, and write low-64 output without
materializing full per-cell INT32 scratch. It ships only after exact CPU
byte-limb and direct-HIP wrap64 differentials, ISA matrix-instruction evidence,
and release timings beat `direct_hip_wrap64_byte_gemm36_u32acc_tiled_2d_v4`
for local `K <= 4096` shapes or the corresponding direct-HIP v4
u64-accumulator fallback for larger K shapes.

Low-bit research retirement rules are mandatory:

- retire INT4/IU4 for a semantic/target if it fails to beat tuned INT8 after
  layout, epilogue, and ISA-confirmed matrix-instruction tuning;
- retire a packed layout version when pack amortization is negative for
  one-shot and repeated-A/B workloads after tuning;
- retire FP8/Ozaki from exact integer production routing unless it is isolated
  as a research mode with explicit verification metadata;
- retire a builtin kernel when it does not beat CK/rocWMMA for the same
  semantic contract, shape, target, and release review conditions.

### 12.6 Target-Specific Notes

- `gfx1030` RDNA2: functional HIP/vector target; matrix-core acceleration is
  not assumed.
- `gfx1100`, `gfx1101`, `gfx1102` RDNA3: wave32 packing and WMMA paths.
- `gfx1200`, `gfx1201` RDNA4: wave32 packing and matrix-core hot kernels.
- `gfx90a` CDNA2: wave64 and MFMA/XDLOPS paths where current ROCm supports it.
- `gfx942` CDNA3: primary previous-generation Instinct target.
- `gfx950` CDNA4: primary current Instinct target.

## 13. Performance Model

Ideal upper bound:

```text
effective_int64_tops = dense_int8_matrix_tops / selected_modulus_count
```

This ceiling excludes packing, modular reduction, reconstruction, launch
overhead, and memory traffic.

The gates below are research targets and acceptance thresholds, not current
performance claims. They require reviewed raw captures, matching semantic
contracts, and target-family baselines before they can be used to claim a
speedup or production performance level.

Performance gates are architecture-family-specific:

| Target family | Persistent RNS output minimum | Reconstructed `int64_t` minimum |
|---|---:|---:|
| RDNA2 functional | correctness only | correctness only |
| RDNA3 Radeon | 7.5 TOPS-eq | 5.0 TOPS-eq |
| RDNA4 Radeon | 25.0 TOPS-eq | 15.0 TOPS-eq |
| CDNA2 Instinct | measured gate after baseline | measured gate after baseline |
| CDNA3 Instinct | 35.0 TOPS-eq | 24.0 TOPS-eq |
| CDNA4 Instinct | 50.0 TOPS-eq | 35.0 TOPS-eq |

Instinct gates are provisional until raw INT8 matrix-engine baselines are
measured on the actual deployment hardware.

Minimum relative speedup gates for 8192 square bounded signed 64-bit GEMM:

| Target family | Persistent RNS output | Reconstructed `int64_t` output |
|---|---:|---:|
| RDNA3 Radeon | 4x over optimized vector-ALU baseline | 2.5x over optimized vector-ALU baseline |
| RDNA4 Radeon | 8x over optimized vector-ALU baseline | 4x over optimized vector-ALU baseline |
| CDNA3/CDNA4 Instinct | 8x over optimized vector-ALU baseline | 4x over optimized vector-ALU baseline |

Benchmarks must report speedups separately for each semantic contract.
Bounded exact RNS output is never compared against wraparound vector output as
if they were equivalent.

## 14. Correctness Specification

Required invariants:

- Moduli in a CRT set are pairwise coprime.
- Field algorithms use prime moduli only.
- INT32 accumulators never overflow.
- Centered residue conversion is deterministic.
- Backend signedness transformations are deterministic and tested.
- Signed and unsigned output semantics are explicit.
- Bounds metadata satisfies the selected modulus product.
- Exact APIs never return probabilistic results.
- K-block reduction is performed before the safe accumulation bound is crossed.
- CRT export detects and reports insufficient range.
- CPU, Windows HIP, and Linux ROCm backends agree for identical seeds and
  semantic contracts.

Required test classes:

1. Exhaustive tiny GEMM for dimensions 1 through 8.
2. Random bounded signed 64-bit outputs.
3. Random bounded unsigned 64-bit outputs.
4. Full 64-bit boundary values.
5. Alternating-sign cancellation.
6. Worst-case positive accumulation.
7. Worst-case negative accumulation.
8. K at 65536 and K above 65536.
9. Composite moduli.
10. Prime moduli.
11. Exact-wide signed output.
12. Exact-wide unsigned output.
13. Strict `mod 2^64` wraparound.
14. Per-tile adaptive prefixes.
15. Corrupt-residue detection in debug verification.
16. GPU reconstruction versus CPU multiprecision reconstruction.
17. Windows HIP versus CPU reference.
18. Linux ROCm versus CPU reference.
19. Backend signedness edge cases.

## 15. Repository Shape

```text
rns8-gemm/
  CMakeLists.txt
  CMakePresets.json
  LICENSE
  NOTICE
  README.md
  cmake/
    toolchains/
      windows-hip-sdk.cmake
      linux-rocm.cmake
    modules/
      FindRNS8HIP.cmake
      FindRNS8CK.cmake
      FindRNS8ThirdParty.cmake
  docs/
    design.md
    platform-windows.md
    platform-linux.md
    platform-readiness.md
    performance-model.md
    correctness.md
    backend-notes.md
  include/rns8/
    rns8.h
    rns8.hpp
    status.h
    semantics.h
    bounds.h
    moduli.h
  src/
    core/
    cpu/
    backend_hip_direct/
    backend_hipblaslt/
    backend_ck/
    backend_rocwmma/
    backend_wrap64/
    reconstruct/
    pack/
  benchmarks/
    rns8_bench.cpp
    sweeps/
  tests/
    unit/
    property/
    differential/
  tools/
    rns8_inspect.cpp
    check_dependencies.py
    result_compare.py
  third_party/
    README.md
```

## 16. Autotuning And Reproducibility

Autotune keys include:

- OS and version,
- GPU architecture,
- ROCm or HIP SDK version,
- backend,
- compiler path and version,
- matrix shape,
- layout,
- K-block size,
- prefix count,
- adaptive tile size,
- epilogue type,
- packed layout version.

Autotune results are stored in:

```text
$XDG_CACHE_HOME/rns8-gemm/autotune.json
%LOCALAPPDATA%\rns8-gemm\autotune.json
```

The local implementation also accepts `RNS8_AUTOTUNE_CACHE_PATH` for isolated
tests. Cache entries are keyed by the plan autotune key and record backend,
target, HIP SDK or accelerator library version, semantic contract, shape,
layout, prefix schedule hash, K-block, tile size, epilogue, selected kernel,
workspace bytes, measured medians, validation status, and cache schema version.
Unreviewed benchmark-emitted entries must not be treated as performance
validation or as permission to promote an accelerator backend.
Raw benchmark captures are not allowed to write production autotune entries
directly; reviewed same-contract reports are the promotion boundary.
The reviewed-cache installer must reject stale or non-reviewed destination
entries during normal merges and require an explicit replacement operation
before discarding them.

Benchmark outputs include:

- command line,
- git commit,
- compiler version,
- ROCm or HIP SDK version,
- GPU name and target id,
- clock/power settings when available,
- selected backend,
- selected kernel,
- backend metadata from the public plan backend metadata API,
- matrix shape,
- data distribution,
- semantic contract,
- bound mode,
- per-tile bound source/order/min/max/hash when adaptive bounded captures use
  per-tile bounds,
- adaptive execution applied flag,
- modulus prefix,
- selected-prefix schedule metadata,
- correctness seed,
- warmup count,
- repeat count,
- timing source,
- raw timings,
- median and p95 timings,
- derived TOPS-equivalent,
- comparison baseline.

Exact-wide benchmark captures use explicit `exact_wide_signed` or
`exact_wide_unsigned` semantic contracts, `RNS8_BOUND_NONE`, a nonzero RNS
prefix, fixed-width little-endian limb export epilogues, and same-contract
CPU/direct-HIP baseline requirements. They are not bounded i64/u64, finite-u8,
or strict wrap64 captures.

Exact deterministic benchmarks use fixed seeds. Early-termination research
benchmarks record RNG seeds and false-acceptance bounds.

## 17. Technique Decisions And Retirement Rules

### 17.1 INT4

Decision: not production v1.

Ship rule: INT4 ships only for a named semantic subset if it beats the INT8
path by at least 1.25x at equal correctness on supported target families.

### 17.2 Early Termination

Decision: opt-in research mode only.

Ship rule: never used by default exact APIs. It ships only as
`RNS8_PROBABILISTIC_VERIFIED` when two Freivalds checks are used and the result
metadata records the probability bound.

Computational-algebra CRA early termination, Freivalds product verification,
and redundant/check-residue experiments belong under the same opt-in research
policy. They can reduce verification or reconstruction cost only when their
probability, modulus set, random seed, repetition count, and failure semantics
are recorded; they do not replace deterministic exact default APIs.

### 17.3 Strassen And Winograd

Decision: not v1 production.

Ship rule: ship one-level Strassen per modulus only if it improves square GEMM
time by at least 1.15x at N >= 16384 with memory overhead <= 2.2x.

### 17.4 Structured Sparsity

Decision: support only when the input workload already satisfies the hardware
structured sparsity pattern.

RNS8 sparse v1 is an explicit A-side 4:2 K-structured byte contract for future
SMFMAC/SWMMAC experiments. Dense GEMM calls never route to sparse matrix-core
instructions implicitly. Callers must provide or derive canonical A compression
metadata, with two nonzero values per group of four K entries, ascending 2-bit
indices, dense B, and explicit signedness. RNS8 validates and can round-trip
this packed form before any accelerator kernel may claim sparse evidence.

Ship rule: sparse path ships only if it improves end-to-end exact GEMM by at
least 1.5x after packing overhead.

### 17.5 Multi-GPU

Decision: split by modulus, not by K.

Ship rule: multi-GPU ships only on Linux ROCm after the modulus split reaches
at least 1.55x speedup for 8192 square bounded GEMM.

### 17.6 AMDGPU Builtins

Decision: not production until CK or rocWMMA diagnostics identify a concrete
shape bottleneck that requires builtin-level instruction control.

Ship rule: a builtin path ships only for a named semantic, shape class, and
target id after exact CPU/direct-HIP differentials, ISA evidence for the
expected integer matrix instructions, no scalar divide/remainder/reciprocal
fallback, no unintended global INT32 stores in fused kernels, and reviewed
release timings that beat CK or rocWMMA.

### 17.7 Strict Wrap64 Matrix Engine

Decision: direct-HIP byte-limb GEMM remains production until displaced.

Ship rule: a matrix-engine wrap64 path ships only after it proves unsigned-byte
or signed-corrected byte-GEMM36 algebra against the CPU byte-limb oracle,
matches direct-HIP wrap64 output across carry-heavy and padded/tail cases,
emits real matrix instructions, fuses carry propagation into low-64 export, and
beats `direct_hip_wrap64_byte_gemm36_u32acc_tiled_2d_v4` in reviewed release
captures for local `K <= 4096` shapes or the corresponding direct-HIP v4
u64-accumulator fallback for larger K shapes.

### 17.8 Packed Layout And Prepack Cache

Decision: persistent packed layouts are versioned ABI-adjacent internal
contracts.

Ship rule: a packed layout or prepack cache ships only when correctness tests
cover source-version invalidation, operand role, layout mismatch rejection,
tile tails, K-block splits, and pack amortization across one-shot, repeated-A,
repeated-B, and repeated-A/B workloads.
Benchmark captures for repeated-A, repeated-B, and repeated-A/B workloads use
`--reuse-packed-a`, `--reuse-packed-b`, and `--reuse-packed-inputs`, and must
keep one-time `prepack_setup_us`, reused operand metadata, and
`prepack_reuse_strategy` separate from repeated workload timings. Eligible
rocWMMA repeated-B captures must exercise the real reusable B prepack cache and
report that strategy explicitly; these measurement modes are not themselves a
production prepack cache. Created plans must expose their current packing
contract through `rns8_get_plan_packing_info`: persistent layout versions,
transient pack workspace bytes, operand layout names, selected input/output
domains, next-operation flags, and explicit cache availability flags. Matrix
handles must expose the matching resident storage
contract through `rns8_get_matrix_storage_info`: source version, finite modulus,
host/device currentness, byte counts, and persistent layout version. Plan plus
operand key material must be validated through `rns8_get_prepack_cache_key_info`
before reuse; it must reject backend, semantic, layout, shape, finite-modulus,
operand-role, device-id, currentness, and source-version mismatches using these
public contracts. Created cache handles must expose matching key/hash material,
device id, and allocation byte contract through `rns8_get_prepack_cache_info`
instead of remaining opaque after creation. The first validated reusable cache
slice is intentionally narrow:
rocWMMA may cache non-tiled RNS B operands for `K <= 65536`, then run GEMM with
only A repacked per dispatch. It does not satisfy the broader production cache
ship rule for tiled schedules, finite/wrap64 semantics, A caches, CK/hipBLASLt,
or cross-platform production policy. Until that broader source-versioned cache
policy exists and is validated, every production plan must report no production
prepack cache.

### 17.9 FP8, Ozaki, And Exact-Arithmetic Research

Decision: FP8/Ozaki paths are research modes only.

Ship rule: no default exact integer API may route through FP8/Ozaki. A research
mode may be accepted only with explicit verification metadata, exact reference
differentials for the advertised contract, split-matrix and K-block accounting,
and measured verification overhead.

### 17.10 Finite-U8 Specialized Reducers

Decision: finite-u8 reducers are production candidates for explicit moduli.

Ship rule: a specialized reducer ships only for named moduli or modulus
families such as `251`, `255`, powers/composites in `[2, 256]`, or
prime-field-only cases after CPU/direct-HIP differentials and reviewed release
captures beat the generic finite-u8 path for the same modulus and shape.

Current implementation: direct HIP ships narrow modulus-251, modulus-255, and
modulus-256 reducers with CPU/direct-HIP differential coverage and schema-valid
raw benchmark metadata for the named reducer paths. CK and rocWMMA also expose
explicit common-modulus 251/255/256 selected-kernel identities that route their
finite-u8 epilogues through the shared fixed-modulus reducer helpers. Schema
and cache tooling reject stale generic accelerator identities for those named
moduli. This is not a broad finite-field optimization claim, and the CK/rocWMMA
v2 identities are not promoted performance entries until reviewed release
captures prove same-contract wins over the appropriate generic, direct-HIP, and
historical accelerator baselines.

## 18. Experiment Matrix

### 18.1 Environment And Capability

| ID | Experiment | Decision |
|---|---|---|
| E001 | CPU compiler and Boost reference | Phase 0 gate |
| E002 | Windows HIP SDK detection | Windows GPU path enabled after pass |
| E003 | Linux ROCm detection | Linux GPU path enabled after pass |
| E004 | GPU architecture detection | target-specific backend set selected |
| E005 | hipBLASLt INT8 capability | B3/B4 enabled only after pass |
| E006 | CK capability | B5/B6 enabled only after pass |
| E007 | rocWMMA capability | B7 enabled only after pass |
| E008 | AMDGPU builtin capability | B7 enabled only after pass |
| E009 | signed and unsigned INT8 behavior | backend accepted after exact match |

E005 through E008 require more than file discovery. Shallow headers, libraries,
tools, opt-in tiny compile/run probes, and builtin availability notes are
evidence only until a backend also has target-supported capability checks and
exact CPU differential validation. E008 has no discovery-only readiness path:
AMDGPU builtin kernels are enabled only by the explicit builtin build/test path
with target-specific exact kernels and ISA evidence.

### 18.2 Core Correctness

| ID | Experiment | Decision |
|---|---|---|
| E020 | CPU int64 to centered residues | CPU pack kept as reference |
| E021 | GPU int64 to centered residues | GPU pack required for production |
| E022 | all default ladder moduli | modulus table accepted after pass |
| E023 | direct HIP one-modulus ring GEMM | portable GPU backend accepted |
| E024 | direct HIP fused reduction | portable fused backend accepted |
| E025 | K > 65536 split reduction | large-K correctness accepted |

### 18.3 Bounded 64-bit GEMM

| ID | Experiment | Decision |
|---|---|---|
| E050 | fixed 9-modulus signed GEMM | first correctness milestone |
| E051 | fixed 9-modulus unsigned GEMM | unsigned milestone |
| E052 | persistent RNS input/output | production center |
| E053 | one-shot input/output | same-contract convenience API |
| E054 | CPU Garner reconstruction prefix 4..20 | CPU reference accepted |
| E055 | GPU bounded reconstruction | target production export gate |

### 18.4 Platform Matrix

| ID | Experiment | Decision |
|---|---|---|
| E070 | Windows RDNA3 direct HIP | local bring-up gate |
| E071 | Windows RDNA4 direct HIP | current Radeon gate |
| E072 | Linux RDNA3/RDNA4 ROCm | Radeon Linux gate |
| E073 | Linux CDNA3 Instinct | previous Instinct gate |
| E074 | Linux CDNA4 Instinct | current Instinct gate |
| E075 | Linux CDNA2 Instinct | supported CDNA2 cluster gate |

### 18.5 Advanced Paths

| ID | Experiment | Decision |
|---|---|---|
| E090 | CK grouped adaptive scheduling | production if correct and faster |
| E091 | fused CK epilogue | production if correct and faster |
| E092 | custom WMMA/builtin hot kernel | production if correct and faster |
| E093 | strict wraparound byte-limb backend | wraparound semantics accepted |
| E094 | exact-wide RNS output | exact-wide milestone |
| E095 | multi-GPU modulus split | Linux-only production if >1.55x |
| E096 | reviewed AUTO cache selection | production after exact cache hit and fallback tests |
| E097 | benchmark review V3 phase telemetry | promotion gate for speedup claims |
| E098 | CK tile/scheduler/pipeline sweep | tune or retire per shape |
| E099 | rocWMMA layout/prepack sweep | tune or retire per shape |
| E100 | AMDGPU builtin IU8/IU4 hot kernel | ship only if faster than CK/rocWMMA |
| E101 | wrap64 matrix-engine byte-GEMM36 | ship only if faster than direct-HIP v4 across the reviewed wrap64 release matrix |
| E102 | packed layout prepack cache | production after amortization proof |
| E103 | INT4/IU4 matrix-engine research | retire unless faster than tuned INT8 |
| E104 | FP8/Ozaki exact-arithmetic research | research-only with verification metadata |
| E105 | finite-u8 specialized reducers | ship per modulus after reviewed win |
| E106 | pack amortization repeated A/B | required for persistent packed layouts |
| E107 | durable production autotune promotion | docs updated only after reviewed release cache |

## 19. Development Roadmap

### Phase 0: Portable Foundation

Deliverables:

- repository scaffold,
- C ABI headers,
- CMake host build,
- dependency checker,
- CPU references,
- benchmark result schema,
- device inspector.

Exit gate:

- E001 passes on Windows and Linux host compilers.

### Phase 1: Windows Direct HIP Bring-Up

Deliverables:

- Windows HIP SDK toolchain file,
- HIP device inspection,
- residue conversion,
- direct HIP one-modulus GEMM,
- separate reduction baseline.

Exit gate:

- E002, E004, E020 through E024, and E070 pass on local Radeon hardware.

### Phase 2: Fixed 9-Modulus Bounded GEMM

Deliverables:

- prefix-9 scheduler,
- CPU CRT reconstruction,
- one-shot bounded signed and unsigned GEMM,
- vector-ALU baseline.

Exit gate:

- E050 through E055 pass.

### Phase 3: Persistent RNS Matrices

Deliverables:

- `rns8_matrix`,
- residue cache,
- packed layout cache,
- RNS output mode.

Exit gate:

- persistent A/B reuse speedup and exact correctness pass.

### Phase 4: Linux ROCm Port

Deliverables:

- Linux ROCm CMake preset,
- ROCm package detection,
- Linux direct HIP parity,
- Linux hipBLASLt baseline recorded when available.

Exit gate:

- E003 and E072 pass on a real Linux ROCm host; E005 is recorded and enables
  B3/B4 only when hipBLASLt is available and validated.

### Phase 5: Grouped And Adaptive Scheduling

Deliverables:

- per-tile bounds,
- prefix selection,
- grouped scheduler,
- read-only grouped descriptor/lifetime contract metadata plus narrow resident
  grouped GEMM before broader public grouped pack/export dispatch,
- adaptive skip,
- CK path where supported.

Exit gate:

- E006, E090 pass where CK is supported.

### Phase 6: Fused Reduction And Reconstruction

Deliverables:

- direct HIP fused reduction,
- CK custom epilogue where supported,
- branchless centered correction,
- GPU bounded CRT export.

Exit gate:

- E024, E055, E091 pass.

### Phase 7: Architecture Hot Kernels

Deliverables:

- Windows `gfx1100` CK and rocWMMA optimized hot paths,
- CK Tile or WMMA/CShuffle scheduler and epilogue sweeps,
- rocWMMA layout and prepack variants,
- AMDGPU builtin experiments only for measured CK/rocWMMA bottlenecks,
- reviewed-cache AUTO selection and rejection/fallback tests.

Exit gate:

- E096 through E100 pass for Windows `gfx1100`, and no accelerator is selected
  by AUTO from discovery or raw benchmark evidence.

### Phase 8: Instinct Production

Deliverables:

- `gfx942` validation,
- `gfx950` validation,
- Linux profiling and power runs,
- cluster reproducibility notes.

Exit gate:

- E073 and E074 pass on actual Instinct systems.

### Phase 9: Secondary Semantics

Deliverables:

- exact-wide RNS output and CPU/direct-HIP limb export,
- strict `mod 2^64` CPU and direct-HIP byte-limb correctness backend,
- optimized strict `mod 2^64` byte-GEMM backend decision,
- packed low-bit layout and prepack cache decisions,
- INT4/IU4, FP8/Ozaki, Strassen, sparsity, and multi-GPU decisions,
- finite-u8 specialized reducer decisions.

Exit gate:

- E093 through E107 produce ship or retire outcomes for the supported target
  scope.

## 20. Completeness Audit

| Gap | Closed decision |
|---|---|
| Windows support | first-class HIP SDK target with direct HIP backend |
| Linux support | full ROCm production target |
| CMake HIP mismatch | no dependency on CMake HIP language for Windows |
| Library availability differences | hipBLASLt, CK, and rocWMMA are feature-detected accelerators |
| RDNA-only scope | expanded to RDNA2/RDNA3/RDNA4 and CDNA2/CDNA3/CDNA4 |
| Instinct current and previous gen | CDNA4 `gfx950` and CDNA3 `gfx942` are production targets |
| Local hardware bring-up | RX 7900 XTX / `gfx1100` is the Windows bring-up target |
| Dependency ambiguity | platform-specific dependency stack and checker are required |
| Optional CPU comparisons | GMP/MPIR, FLINT, NTL, FFLAS-FFPACK, and LinBox are optional |
| Backend signedness | signed and unsigned INT8 behavior is explicit and tested |
| Repository scaffold | platform files, dependency checker, and direct HIP backend added |

## 21. Reference Index

### AMD And ROCm

- ROCm GPU hardware specifications:
  https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html
- ROCm Linux system requirements:
  https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html
- HIP SDK for Windows:
  https://rocm.docs.amd.com/projects/install-on-windows/en/latest/index.html
- HIP SDK Windows system requirements:
  https://rocm.docs.amd.com/projects/install-on-windows/en/latest/reference/system-requirements.html
- HIP SDK Windows component support:
  https://rocm.docs.amd.com/projects/install-on-windows/en/develop/conceptual/component-support.html
- HIP installation:
  https://rocm.docs.amd.com/projects/HIP/en/latest/install/install.html
- hipBLASLt data type support:
  https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/reference/data-type-support.html
- Composable Kernel:
  https://rocm.docs.amd.com/projects/composable_kernel/
- rocWMMA:
  https://rocm.docs.amd.com/projects/rocWMMA/en/latest/
- Clang AMDGPU builtins:
  https://clang.llvm.org/docs/AMDGPUBuiltinReference.html

### Ozaki, RNS, And Exact Matrix Multiplication

- Daichi Mukunoki, "DGEMM without FP64 Arithmetic - Using FP64 Emulation and
  FP8 Tensor Cores with Ozaki Scheme", arXiv:2508.00441.
- Ozaki Scheme II: https://arxiv.org/abs/2504.08009
- DGEMM on Integer Matrix Multiplication Unit: https://arxiv.org/abs/2306.11975
- Performance enhancement of Ozaki Scheme on IMMU:
  https://arxiv.org/abs/2409.13313
- Chinese remainder theorem overview:
  https://mathworld.wolfram.com/ChineseRemainderTheorem.html
- Exact linear algebra with early termination:
  https://arxiv.org/abs/cs/0501074
- Freivalds matrix product verification:
  https://dblp.org/rec/conf/mfcs/Freivalds79
- Dense linear algebra over finite fields:
  https://arxiv.org/abs/cs/0601133

### Existing Codebases And Libraries

- GEMMul8: https://github.com/RIKEN-RCCS/GEMMul8
- ozIMMU: https://github.com/enp1s0/ozIMMU
- OzBLAS: https://github.com/RIKEN-RCCS/ozblas
- GRNS: https://github.com/kisupov/grns
- MPRES-BLAS: https://github.com/kisupov/mpres-blas
- FFLAS-FFPACK: https://linbox-team.github.io/fflas-ffpack/
- LinBox: https://linalg.org/linbox/linbox/
- FLINT `fmpz_mat`: https://flintlib.org/doc/fmpz_mat.html
- NTL: https://libntl.org/doc/tour-changes.html

## 22. Final Architecture

Production bounded 64-bit path:

```text
int64 or uint64 source
  -> validated bounds metadata
  -> centered INT8 RNS packing using default modulus ladder
  -> persistent modulus-major residue matrices
  -> validated direct HIP or future validated accelerator INT8 GEMM backend
  -> fused INT32-to-residue reduction where supported
  -> RNS output
  -> bounded CRT export to int64 or uint64
```

Production exact-wide path:

```text
int64 or uint64 source
  -> explicit exact-wide signed or unsigned descriptor with RNS8_BOUND_NONE
  -> centered INT8 RNS packing using the selected prefix
  -> persistent modulus-major residue matrices
  -> validated direct HIP or future validated accelerator INT8 GEMM backend
  -> RNS output
  -> fixed-width little-endian limb export
     signed: centered two's-complement limbs
     unsigned: canonical magnitude limbs
```

Production strict wraparound path:

```text
uint64 source
  -> base-256 byte limbs
  -> 36 low-64-relevant byte-product pairs across the low eight Comba diagonals
  -> Comba diagonal accumulation with unsigned-byte signed-INT8 correction
  -> delayed carry propagation
  -> low 64-bit output
```

If a future accelerator exposes only signed INT8 products, unsigned byte
products must be reconstructed with the tested signed-INT8 correction algebra
before Comba carry propagation. The CPU reference has a 36-byte-pair
decomposition oracle for this production path, and the direct-HIP correctness
kernel consumes the same correction algebra at device source level. This still
does not enable a signed-INT8 matrix-engine backend by itself.

The central performance thesis is:

```text
For bounded exact 64-bit output, RNS8 must prove correctness through the
portable CPU and direct HIP paths first, then earn production status per target
family by amortizing packing, fusing reduction, and selecting the fastest
available AMD matrix-engine backend on that OS and GPU.
```
