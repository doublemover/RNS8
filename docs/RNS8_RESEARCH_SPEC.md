# RNS8 Technical Specification

Exact integer matrix multiplication on INT8 matrix engines with ROCm, RNS,
CRT, and Ozaki-style decomposition.

Date: 2026-06-01

Project name: `RNS8`

Repository name: `rns8-gemm`

## 1. Executive Decision Register

This document is the complete project specification. It contains no dependency
on any current checkout layout. It is written for a new repository.

All ambiguous architecture choices are closed here.

| ID | Decision |
|---|---|
| D1 | Build a new RNS-first library named `RNS8`, packaged as `rns8-gemm`. |
| D2 | The first production target is bounded exact signed and unsigned 64-bit GEMM. |
| D3 | The core representation is persistent residue number system (RNS) storage, not temporary conversion around a BLAS call. |
| D4 | The core compute primitive is INT8 x INT8 -> INT32 matrix GEMM per modulus. |
| D5 | The default modulus ladder is pairwise-coprime, composite-inclusive, and ordered by descending range contribution. |
| D6 | Full signed or unsigned 64-bit bounded output uses the first 9 moduli by default. |
| D7 | Adaptive per-tile modulus counts are enabled for performance, but correctness never depends on early termination. |
| D8 | Probabilistic early termination is not part of the default exact API. It is an opt-in research mode with explicit verification metadata. |
| D9 | Strict `mod 2^64` wraparound is implemented by a byte-limb backend, not by odd-modulus CRT alone. |
| D10 | INT4 is not a v1 production backend. It is retained only as a measured research path with a defined retirement rule. |
| D11 | RX 9070 XT / RDNA4 / `gfx1201` is the first optimization target. RX 7900 XTX / RDNA3 / `gfx1100` is the second target. |
| D12 | The optimized backend sequence is hipBLASLt baseline, CK grouped/fused kernels, then direct rocWMMA or AMDGPU builtin kernels for hot paths. |
| D13 | Fused INT32-to-residue modulo reduction is required for the production performance path. Separate INT32 store plus reduction is only a baseline. |
| D14 | GPU CRT reconstruction is required for production bounded `int64` export. CPU CRT is the reference and debug path. |
| D15 | The public API is explicit about semantics. It never infers signed, unsigned, bounded, wide, or wraparound behavior from the C++ type alone. |
| D16 | v1 supports Linux ROCm builds. Windows is not a GPU execution target for v1. |
| D17 | License is Apache-2.0, with third-party notices for AMD, NVIDIA, academic, and CPU reference dependencies. |

## 2. Product Scope

### 2.1 In Scope

- Exact dense matrix multiplication over integer domains.
- ROCm/HIP implementation for AMD RDNA4 and RDNA3 consumer GPUs.
- A CPU reference backend for correctness and CI.
- A finite-field or finite-ring `mod m` GEMM primitive for `m <= 256`.
- Bounded exact `int64_t` and `uint64_t` GEMM.
- Exact-wide integer GEMM for outputs wider than 64 bits.
- Strict wraparound `mod 2^64` GEMM through a separate byte-limb backend.
- Persistent RNS matrix storage and reuse.
- Per-tile adaptive modulus counts using deterministic bounds.
- Grouped and persistent scheduling across `(modulus, tile)` work.
- Fused modulo reduction in CK or custom WMMA kernels.
- Benchmarks against normal vector-ALU `int64` GPU kernels and CPU exact
  linear-algebra baselines.

### 2.2 Out Of Scope For v1

- Drop-in BLAS interception.
- Approximate integer output.
- General sparse matrix multiplication as a primary product.
- Compiler integration.
- Automatic proof of user-provided numerical bounds.
- Default probabilistic correctness.
- INT4 production kernels.
- Windows GPU execution.

The out-of-scope items remain benchmarked or planned where this document
defines a concrete experiment and retirement rule.

## 3. Target Hardware And Software

### 3.1 Primary GPUs

| GPU | Architecture | Target | Dense INT8 peak | Dense INT4 peak | Priority |
|---|---|---|---:|---:|---:|
| Radeon RX 9070 XT | RDNA4 | `gfx1201` | 389 TOPS | 779 TOPS | 1 |
| Radeon RX 7900 XTX | RDNA3 | `gfx1100` | 123 TOPS | 246 TOPS | 2 |

The RX 9070 XT is the main optimization target because the dense INT8 matrix
ceiling is high enough that packing, reduction, reconstruction, and launch
overhead become the decisive engineering problems.

The RX 7900 XTX remains a required target because it provides the RDNA3 path
and exposes fragment-layout and packing issues that the library must handle.

### 3.2 Required Software Stack

- Linux with ROCm 7.2 or newer.
- HIP C++ compiler with `gfx1201` and `gfx1100` code generation.
- hipBLASLt for baseline INT8 GEMM.
- Composable Kernel (CK) for grouped GEMM and custom epilogues.
- rocWMMA or AMDGPU builtins for custom hot kernels.
- CMake and Ninja.
- Python only for benchmark orchestration and result analysis.
- CPU reference dependencies: C++17 and Boost.Multiprecision.
- Comparison-only CPU libraries: GMP, FLINT, NTL, and FFLAS-FFPACK.

### 3.3 Build Outputs

The repository produces:

- `librns8.so`: shared C ABI library.
- `librns8_static.a`: static library.
- `rns8-bench`: benchmark runner.
- `rns8-verify`: correctness and differential-test runner.
- `rns8-inspect`: device, backend, and autotune cache inspector.
- Python package `rns8bench` for benchmark sweeps only.

## 4. Integer Semantics

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

### 4.1 Bounded Exact Signed 64-bit

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

### 4.2 Bounded Exact Unsigned 64-bit

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

### 4.3 Exact-Wide Signed And Unsigned

Exact-wide output is selected for results wider than 64 bits. For arbitrary
signed 64-bit inputs:

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

The v1 exact-wide GPU compute path stores RNS output. CPU reconstruction to
multi-limb integers is supported. GPU reconstruction for exact-wide output is
a phase-2 feature with fixed two-limb and three-limb implementations.

### 4.4 Strict Wraparound `mod 2^64`

Contract:

```text
C = A * B mod 2^64
```

This is not implemented with the odd-modulus CRT path unless the caller also
supplies a range bound that makes the exact result recoverable. The production
wraparound backend is:

```text
base-256 limbs
36 low-product byte GEMMs
Comba diagonal accumulation
delayed carry propagation
low 64-bit export
```

### 4.5 Finite Ring And Finite Field

Finite-ring primitive:

```text
C = A * B mod m, 2 <= m <= 256
```

Finite-field primitive:

```text
C = A * B mod p, p prime, p <= 251
```

Composite moduli are valid for CRT and ring GEMM. Prime moduli are mandatory
for algorithms that require a field, including field-only Strassen variants
and Freivalds checks.

## 5. Modulus Ladder

### 5.1 Default Ordered Set

The default CRT ladder is:

```text
256, 255, 253, 251, 247, 239, 233, 229,
227, 223, 217, 211, 199, 197, 193, 191,
181, 179, 173, 167, 163, 157, 151, 149,
139, 137, 131, 127
```

All values are `<= 256` and pairwise coprime in this order. Composite values
are intentional because CRT requires pairwise coprime rings, not fields.

### 5.2 Prefix Range Table

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

The default bounded-64 path uses prefix 9. The exact-wide path selects the
smallest prefix that satisfies the formula in Section 4.3.

### 5.3 Centered Residues

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

## 6. Core Algorithm

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

The residues `R_i` are either retained as RNS output or reconstructed into an
integer output type.

### 6.1 INT32 Accumulation Limit

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

### 6.2 Modular Reduction

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
2. Constant reciprocal multiply-high reduction for all fixed moduli.
3. Branchless correction into canonical centered range.
4. Barrett reduction only where reciprocal reduction fails benchmark gates.

No production kernel writes full INT32 output matrices to global memory except
for the baseline backend.

### 6.3 CRT Reconstruction

Bounded 64-bit export uses fixed-limb mixed-radix Garner reconstruction on GPU.

Implementation details:

- Prefixes up to 16 moduli use two 64-bit limbs for intermediate values.
- Prefixes 17 through 20 use three 64-bit limbs.
- CPU reference uses Boost.Multiprecision.
- GPU export for bounded `int64_t` and `uint64_t` is required in v1.
- GPU export for exact-wide multi-limb output is implemented after bounded
  export passes all correctness gates.

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

## 7. Bounds And Adaptive Moduli

### 7.1 Bounds Metadata

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
- User-provided bounds are trusted as contract inputs and checked by debug
  verification runs when the caller enables debug verification.

### 7.2 Tile Size

The default adaptive bound tile is:

```text
128 x 128 output elements
```

The tile size is configurable in powers of two from 64 to 512. Autotuning may
select a larger tile for very large square GEMM, but the default benchmark and
correctness suite uses 128.

### 7.3 Prefix Selection

For each tile:

```text
required_prefix(T) = smallest s such that product(moduli[0:s]) > range(T)
```

where:

- signed bounded range uses `2 * max_abs_output(T)`,
- unsigned bounded range uses `max_output(T)`,
- exact-wide uses the formulas in Section 4.3.

The scheduler never enqueues `(modulus_id, tile)` when:

```text
modulus_id >= required_prefix(tile)
```

### 7.4 Early Termination Policy

Default exact APIs do not use early termination.

Research early termination mode is available only when the caller sets:

```text
allow_probabilistic_result = true
verification_prime_count >= 2
```

It performs stabilization across additional moduli and then two Freivalds
checks over fresh prime moduli. The result object records:

- number of stabilization moduli,
- verification primes,
- RNG seed,
- false-acceptance bound.

Production exact benchmarks report early termination separately and never mix
it with deterministic exact throughput numbers.

## 8. Storage Layout

### 8.1 Persistent Matrix Type

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

### 8.2 Physical Layout

Canonical residue storage is modulus-major:

```text
residue[modulus][tile_m][tile_n][element]
```

Backend-packed storage is also modulus-major:

```text
packed[modulus][macro_tile][wmma_fragment]
```

Modulus-major layout is mandatory because it:

- feeds one residue GEMM with contiguous operands,
- enables grouped GEMM over moduli,
- enables skipping completed adaptive tiles,
- avoids element-major gather overhead,
- keeps A and B reusable across repeated GEMM.

### 8.3 Alignment And Allocation

- Device allocations are 256-byte aligned.
- Packed panels are padded to backend tile multiples.
- Canonical dimensions retain exact logical shape.
- Workspace is caller-owned through `rns8_workspace`.
- Temporary allocations inside hot calls are forbidden after plan creation.

### 8.4 Cache Validity

Packed residues are valid only while all cache key fields match:

- source allocation id,
- source version token,
- dimensions and strides,
- transpose flags,
- modulus ladder id,
- selected max prefix,
- target architecture,
- backend id,
- packing version.

The library cannot detect arbitrary raw pointer mutation. Mutation safety is
solved by caller-managed version tokens.

## 9. Public API Specification

### 9.1 C ABI

The stable ABI is C.

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

rns8_status rns8_create_plan(
    rns8_context* ctx,
    const rns8_gemm_desc* desc,
    rns8_plan** out);

rns8_status rns8_destroy_plan(rns8_plan* plan);

rns8_status rns8_pack_i64(
    rns8_context* ctx,
    const rns8_plan* plan,
    const int64_t* src,
    int64_t ld,
    uint64_t source_version,
    rns8_matrix** out);

rns8_status rns8_pack_u64(
    rns8_context* ctx,
    const rns8_plan* plan,
    const uint64_t* src,
    int64_t ld,
    uint64_t source_version,
    rns8_matrix** out);

rns8_status rns8_gemm_rns(
    rns8_context* ctx,
    const rns8_plan* plan,
    const rns8_matrix* A,
    const rns8_matrix* B,
    rns8_matrix* C);

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

rns8_status rns8_gemm_i64_oneshot(
    rns8_context* ctx,
    const rns8_gemm_desc* desc,
    const int64_t* A,
    int64_t lda,
    const int64_t* B,
    int64_t ldb,
    int64_t* C,
    int64_t ldc);
```

### 9.2 C++ Wrapper

The C++ API is header-only RAII over the C ABI. It does not define separate
semantics.

### 9.3 Error Codes

Required status codes:

- `RNS8_SUCCESS`
- `RNS8_INVALID_ARGUMENT`
- `RNS8_UNSUPPORTED_ARCH`
- `RNS8_UNSUPPORTED_BACKEND`
- `RNS8_RANGE_ERROR`
- `RNS8_ACCUMULATION_OVERFLOW_RISK`
- `RNS8_WORKSPACE_TOO_SMALL`
- `RNS8_BACKEND_FAILURE`
- `RNS8_VERIFICATION_FAILED`
- `RNS8_INTERNAL_ERROR`

### 9.4 Thread Safety

- Contexts are not internally synchronized.
- One context is used by one host thread at a time.
- Multiple contexts on separate streams are supported.
- Plan and matrix descriptors are immutable after creation.

## 10. Backend Architecture

### 10.1 Backend Ladder

| Stage | Backend | Purpose | Production role |
|---|---|---|---|
| B0 | CPU scalar/multiprecision | correctness reference | required |
| B1 | hipBLASLt INT8 per modulus | baseline GPU path | required |
| B2 | hipBLASLt batched/grouped | launch amortization | required |
| B3 | CK grouped GEMM | adaptive tile scheduling | required |
| B4 | CK custom epilogue | fused modulo reduction | required |
| B5 | rocWMMA or AMDGPU builtins | hot kernels | required for RX 9070 XT performance target |

### 10.2 Backend Selection Policy

At plan creation, RNS8 selects the first backend that satisfies:

1. architecture support,
2. data layout support,
3. semantics support,
4. workspace limit,
5. autotune result.

Default backend by phase:

- correctness tests: B0 and B1,
- first benchmark: B1,
- adaptive path: B3,
- production bounded path: B4 or B5.

### 10.3 RDNA3 Path

Target: `gfx1100`

Rules:

- Use RDNA3 WMMA instructions through rocWMMA or builtins.
- Account for A/B fragment replication requirements in packing.
- Favor larger grouped work batches to offset launch and packing overhead.
- Production gate is lower than RDNA4 because raw INT8 peak is lower.

### 10.4 RDNA4 Path

Target: `gfx1201`

Rules:

- Use RDNA4 matrix core intrinsics or rocWMMA support for `gfx12`.
- Implement direct hot kernels earlier than RDNA3.
- Optimize fused epilogue first because raw INT8 throughput makes global INT32
  stores unacceptable.

### 10.5 Scheduler

Work item:

```text
(modulus_id, output_tile_id, k_block_id)
```

Scheduler rules:

- Skip tiles whose `required_prefix` is already satisfied.
- Group by backend-compatible shape.
- Use grouped GEMM for heterogeneous tile/modulus work.
- Use strided-batched GEMM for homogeneous full-prefix work.
- Use persistent kernels for many small or irregular tiles.
- Split K only at the fixed safe K-block boundary.
- Do not split across GPUs by K in the multi-GPU path.

### 10.6 Multi-GPU

Multi-GPU support splits by modulus:

```text
GPU 0: prefix moduli 0..a
GPU 1: prefix moduli a+1..b
```

Rules:

- A and B packed residues are duplicated per GPU.
- Residue outputs are gathered for CRT reconstruction.
- Load balancing uses per-modulus tile counts after adaptive pruning.
- Multi-GPU ships only if two GPUs deliver at least 1.55x speedup over the
  same single-GPU backend for 8192 square bounded GEMM.

## 11. Performance Model

### 11.1 Ideal Dense INT8 Ceiling

Ideal upper bound:

```text
effective_int64_tops = dense_int8_matrix_tops / selected_modulus_count
```

This ceiling excludes packing, modular reduction, reconstruction, launch
overhead, and memory traffic.

| Moduli | Range bits | RX 7900 XTX ideal | RX 9070 XT ideal |
|---:|---:|---:|---:|
| 4 | 31.949 | 30.8 TOPS-eq | 97.2 TOPS-eq |
| 5 | 39.897 | 24.6 TOPS-eq | 77.8 TOPS-eq |
| 6 | 47.798 | 20.5 TOPS-eq | 64.8 TOPS-eq |
| 7 | 55.662 | 17.6 TOPS-eq | 55.6 TOPS-eq |
| 8 | 63.502 | 15.4 TOPS-eq | 48.6 TOPS-eq |
| 9 | 71.328 | 13.7 TOPS-eq | 43.2 TOPS-eq |
| 10 | 79.129 | 12.3 TOPS-eq | 38.9 TOPS-eq |
| 12 | 94.612 | 10.3 TOPS-eq | 32.4 TOPS-eq |
| 16 | 125.040 | 7.7 TOPS-eq | 24.3 TOPS-eq |
| 18 | 140.024 | 6.8 TOPS-eq | 21.6 TOPS-eq |
| 20 | 154.842 | 6.2 TOPS-eq | 19.5 TOPS-eq |

### 11.2 Required Performance Gates

For square 8192 x 8192 x 8192 bounded signed 64-bit GEMM with persistent RNS
inputs and RNS output:

| GPU | Minimum ship gate | Stretch gate |
|---|---:|---:|
| RX 7900 XTX | 7.5 TOPS-eq | 10.0 TOPS-eq |
| RX 9070 XT | 25.0 TOPS-eq | 32.0 TOPS-eq |

For the same workload with `int64_t` reconstruction included:

| GPU | Minimum ship gate | Stretch gate |
|---|---:|---:|
| RX 7900 XTX | 5.0 TOPS-eq | 7.5 TOPS-eq |
| RX 9070 XT | 15.0 TOPS-eq | 24.0 TOPS-eq |

These gates define success for the project. They are not theoretical peak
claims.

### 11.3 Normal `int64` Comparison

The baseline comparison suite includes:

1. vector-ALU low-half `uint64_t mod 2^64` GEMM,
2. vector-ALU signed bounded `int64_t` GEMM with overflow checks disabled only
   when the same range contract is supplied,
3. vector-ALU exact-wide limb GEMM,
4. CPU `__int128` bounded reference,
5. CPU multiprecision reference,
6. AVX512 VNNI byte-limb GEMM where the CPU supports it,
7. AVX512 IFMA52 modular reconstruction where the CPU supports it.

RNS8 must report speedups separately for each semantic contract. The project
does not compare bounded exact RNS output against a wraparound vector kernel
as if they were equivalent.

Minimum relative speedup gates for 8192 square bounded signed 64-bit GEMM:

| GPU | Persistent RNS output | Reconstructed `int64_t` output |
|---|---:|---:|
| RX 7900 XTX | 4x over optimized vector-ALU baseline | 2.5x over optimized vector-ALU baseline |
| RX 9070 XT | 8x over optimized vector-ALU baseline | 4x over optimized vector-ALU baseline |

## 12. Correctness Specification

### 12.1 Invariants

- Moduli in a CRT set are pairwise coprime.
- Field algorithms use prime moduli only.
- INT32 accumulators never overflow.
- Centered residue conversion is deterministic.
- Signed and unsigned output semantics are explicit.
- Bounds metadata satisfies the selected modulus product.
- Exact APIs never return probabilistic results.
- K-block reduction is performed before the safe accumulation bound is crossed.
- CRT export detects and reports insufficient range.

### 12.2 Required Test Classes

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

### 12.3 Reference Implementations

Required references:

- C++ `__int128` bounded signed and unsigned reference.
- Boost.Multiprecision exact-wide reference.
- Per-modulus scalar ring GEMM reference.
- Byte-limb CPU reference for `mod 2^64`.

Optional references:

- GMP or MPIR.
- FLINT `fmpz_mat`.
- NTL.
- FFLAS-FFPACK.
- LinBox.

## 13. Repository Shape

```text
rns8-gemm/
  CMakeLists.txt
  LICENSE
  NOTICE
  README.md
  docs/
    design.md
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
    backend_hipblaslt/
    backend_ck/
    backend_wmma/
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
    result_compare.py
  third_party/
    README.md
```

## 14. Autotuning And Reproducibility

Autotune keys include:

- GPU architecture,
- ROCm version,
- backend,
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
```

Benchmark outputs include:

- command line,
- git commit,
- compiler version,
- ROCm version,
- GPU name and target id,
- clock/power settings,
- selected backend,
- selected kernel,
- matrix shape,
- data distribution,
- semantic contract,
- modulus prefix,
- correctness seed,
- raw timings,
- derived TOPS-equivalent,
- comparison baseline.

Exact deterministic benchmarks use fixed seeds. Early-termination research
benchmarks record RNG seeds and false-acceptance bounds.

## 15. Technique Decisions And Retirement Rules

### 15.1 INT4

Decision: not production v1.

Reason: exact 64-bit output requires enough residue range, and 4-bit residues
reduce range per GEMM too much. The extra operations erase most advertised
INT4 peak advantage for dense exact 64-bit output.

Ship rule: INT4 ships only for a named semantic subset if it beats the INT8
path by at least 1.25x at equal correctness on both target GPUs.

Retirement rule: retire INT4 from the roadmap if it fails the 1.25x gate for:

- bounded 32-bit outputs,
- bounded 48-bit outputs,
- structured sparse 64-bit outputs,
- binary or nibble side-check kernels.

### 15.2 Early Termination

Decision: opt-in research mode only.

Ship rule: never used by default exact APIs. It ships only as
`RNS8_PROBABILISTIC_VERIFIED` when two Freivalds checks are used and the result
metadata records the probability bound.

Retirement rule: remove from performance-critical plans if its verification
overhead exceeds the cost of one additional modulus for 8192 square GEMM.

### 15.3 Strassen And Winograd

Decision: not v1 production.

Ship rule: ship one-level Strassen per modulus only if it improves square
GEMM time by at least 1.15x at N >= 16384 with memory overhead <= 2.2x.

Retirement rule: retire for v1 if the crossover is above N=32768 or memory
overhead exceeds 2.2x.

### 15.4 Structured Sparsity

Decision: support only when the input workload already satisfies the hardware
structured sparsity pattern.

Ship rule: sparse path ships only if it improves end-to-end exact GEMM by at
least 1.5x after packing overhead.

Retirement rule: do not transform dense arbitrary residues to force 2:4
sparsity.

### 15.5 Multi-GPU

Decision: split by modulus, not by K.

Ship rule: two-GPU modulus split must reach at least 1.55x speedup for 8192
square bounded GEMM before it is treated as production.

Retirement rule: if PCIe residue gathering keeps speedup below 1.25x for all
bounded 64-bit shapes up to 16384, keep multi-GPU as exact-wide only.

### 15.6 Byte-Limb Wraparound

Decision: required for strict `mod 2^64`.

Ship rule: the 36-GEMM Comba path ships when it is correct for all wraparound
tests and beats the vector-ALU wraparound baseline by at least 1.5x on RX
9070 XT for N >= 4096.

Retirement rule: Karatsuba and Toom variants retire if they fail to beat Comba
by at least 1.15x after packing overhead.

## 16. Exhaustive Experiment Matrix

Every experiment records correctness, timings, bandwidth, achieved INT8 peak,
effective TOPS-equivalent, memory footprint, and comparison baseline.

### 16.1 Environment And Capability

| ID | Experiment | Decision |
|---|---|---|
| E001 | Detect GPU, target id, ROCm version, matrix instruction support | supported targets are `gfx1201` and `gfx1100` |
| E002 | Compile hipBLASLt INT8 I32 GEMM for each target | B1 backend enabled only after pass |
| E003 | Compile CK grouped GEMM for each target | B3 backend enabled only after pass |
| E004 | Compile CK custom epilogue prototype | B4 backend enabled only after pass |
| E005 | Compile rocWMMA or builtin WMMA minimal kernel | B5 backend enabled only after pass |
| E006 | Verify K-block accumulation safety at K=65536 | K-block constant accepted only after pass |
| E007 | Inspect generated ISA for matrix instructions | kernel rejected if scalarized |

### 16.2 Raw GEMM Baselines

| ID | Experiment | Decision |
|---|---|---|
| E010 | hipBLASLt INT8 GEMM square sweep | establishes raw peak denominator |
| E011 | hipBLASLt INT8 rectangular sweep | establishes non-square backend limits |
| E012 | CK INT8 GEMM square sweep | selected over hipBLASLt if faster |
| E013 | rocWMMA raw microkernel sweep | selected for B5 if faster after epilogue |
| E014 | strided-batched modulus GEMM | selected when homogeneous prefixes dominate |
| E015 | grouped GEMM over modulus list | selected when launch overhead is material |
| E016 | many-small GEMM sweep | persistent scheduler required if utilization is low |

### 16.3 Residue Conversion

| ID | Experiment | Decision |
|---|---|---|
| E020 | CPU int64 to centered residues | CPU pack kept as reference |
| E021 | GPU int64 to centered residues | GPU pack required for production |
| E022 | signed conversion boundary values | conversion accepted after exact match |
| E023 | unsigned conversion boundary values | conversion accepted after exact match |
| E024 | composite modulus conversion | default ladder accepted after exact match |
| E025 | packing cache reuse A only | cache policy retained if speedup > 1.1x |
| E026 | packing cache reuse B only | cache policy retained if speedup > 1.1x |
| E027 | packing cache reuse A and B | persistent mode gate measured here |

### 16.4 Single-Modulus Ring GEMM

| ID | Experiment | Decision |
|---|---|---|
| E030 | `m=256` ring GEMM | bitmask path enabled after pass |
| E031 | `m=255` ring GEMM | composite path enabled after pass |
| E032 | `m=251` field GEMM | prime path enabled after pass |
| E033 | all default ladder moduli | modulus table accepted after pass |
| E034 | separate reduction kernel | retained as baseline only |
| E035 | fused reduction epilogue | required for production path |
| E036 | K > 65536 split reduction | large-K correctness accepted after pass |

### 16.5 CRT Reconstruction

| ID | Experiment | Decision |
|---|---|---|
| E040 | CPU Garner reconstruction prefix 4..20 | CPU reference accepted after pass |
| E041 | GPU two-limb reconstruction prefix 9 | bounded 64 export enabled after pass |
| E042 | GPU two-limb reconstruction prefix 16 | guard-range export enabled after pass |
| E043 | GPU three-limb reconstruction prefix 20 | exact-wide GPU export enabled after pass |
| E044 | signed centered boundary | signed export accepted after pass |
| E045 | unsigned full 64-bit boundary | unsigned export accepted after pass |
| E046 | insufficient range detection | range errors accepted after pass |
| E047 | CPU vs GPU reconstruction throughput | GPU required if output export dominates |

### 16.6 Bounded 64-bit GEMM

| ID | Experiment | Decision |
|---|---|---|
| E050 | fixed 9-modulus signed GEMM | first correctness milestone |
| E051 | fixed 9-modulus unsigned GEMM | unsigned milestone |
| E052 | persistent RNS input/output | production center |
| E053 | one-shot input/output | adoption wrapper |
| E054 | reconstruct every call | export overhead quantified |
| E055 | repeated same A | cache policy selected |
| E056 | repeated same B | cache policy selected |
| E057 | repeated same A and B | persistent performance gate |
| E058 | compare vector-ALU int64 baseline | speedup gate evaluated |

### 16.7 Adaptive Moduli

| ID | Experiment | Decision |
|---|---|---|
| E060 | global bound only | fallback adaptive metadata path |
| E061 | per-tile 128x128 bounds | default adaptive path |
| E062 | tile sizes 64,128,256,512 | autotune tile-size selection |
| E063 | mixed range 6-to-9 prefixes | scheduler skip required |
| E064 | all tiles require 9 prefixes | adaptive overhead must stay below 5% |
| E065 | bound computation on GPU | selected if faster than CPU for dynamic bounds |
| E066 | bound metadata reuse | retained if speedup > 1.1x |

### 16.8 Fused Epilogue

| ID | Experiment | Decision |
|---|---|---|
| E070 | store INT32 then reduce | baseline only |
| E071 | CK custom epilogue reduction | production if faster and correct |
| E072 | custom WMMA epilogue reduction | production for hot shapes if faster |
| E073 | reciprocal reduction | default if correct and fastest |
| E074 | Barrett reduction | selected only when faster |
| E075 | branchless centered correction | required |
| E076 | INT8 residue output bandwidth | fused path must reduce global bytes |

### 16.9 Custom WMMA Kernels

| ID | Experiment | Decision |
|---|---|---|
| E080 | minimal `gfx1100` kernel | RDNA3 B5 enabled after pass |
| E081 | minimal `gfx1201` kernel | RDNA4 B5 enabled after pass |
| E082 | shared-memory staging | selected if faster |
| E083 | direct global panels | selected if faster |
| E084 | double-buffered K loop | required if speedup > 1.05x |
| E085 | multiple moduli per launch | selected if launch overhead dominates |
| E086 | one modulus per launch | retained for simplicity if equal speed |

### 16.10 Exact-Wide Integer

| ID | Experiment | Decision |
|---|---|---|
| E090 | signed arbitrary inputs K<=8192 prefix 18 | exact-wide milestone |
| E091 | signed arbitrary inputs K<=1048576 prefix 19 | extended milestone |
| E092 | signed arbitrary inputs K<=134217728 prefix 20 | maximum default ladder milestone |
| E093 | unsigned arbitrary inputs prefix selection | unsigned wide accepted after pass |
| E094 | RNS output only | v1 exact-wide production mode |
| E095 | CPU multiprecision export | v1 exact-wide export mode |
| E096 | GPU multi-limb export | phase-2 export mode |

### 16.11 Wraparound `mod 2^64`

| ID | Experiment | Decision |
|---|---|---|
| E100 | naive 36 byte-GEMM low product | baseline wrap backend |
| E101 | Comba diagonal scheduling | production if faster |
| E102 | delayed carry propagation | production if faster |
| E103 | carry-save partial storage | production if faster |
| E104 | grouped 36-GEMM schedule | production if launch overhead is material |
| E105 | 2-way Karatsuba | ship only if >1.15x over Comba |
| E106 | Toom-3 short product | ship only if >1.15x over Comba |
| E107 | compare against bounded RNS when bound exists | selects semantic-specific default |

### 16.12 INT4

| ID | Experiment | Decision |
|---|---|---|
| E110 | bounded 32-bit outputs | INT4 retained only if >1.25x |
| E111 | bounded 48-bit outputs | INT4 retained only if >1.25x |
| E112 | structured sparse 64-bit outputs | INT4 retained only if >1.25x |
| E113 | binary side checks | INT4 retained only if >1.25x |
| E114 | dense 64-bit bounded outputs | expected retirement check |

### 16.13 Strassen, Sparsity, And Structure

| ID | Experiment | Decision |
|---|---|---|
| E120 | one-level Strassen N>=16384 | ship only if >1.15x and memory <=2.2x |
| E121 | two-level Strassen | ship only if faster than one-level |
| E122 | Winograd variant | ship only if faster than Strassen |
| E123 | 2:4 structured sparse exact input | ship only if >1.5x end-to-end |
| E124 | block-sparse exact input | ship as separate sparse module only |
| E125 | low-rank exact factors | ship as separate structured module only |

### 16.14 Multi-GPU

| ID | Experiment | Decision |
|---|---|---|
| E130 | split 9 moduli across 2 GPUs | ship if >1.55x |
| E131 | split 18 moduli across 2 GPUs | exact-wide multi-GPU gate |
| E132 | adaptive load balancing | required if adaptive path ships multi-GPU |
| E133 | gather to CPU for CRT | selected for wide export if faster |
| E134 | gather to GPU for CRT | selected for bounded export if faster |

### 16.15 CPU Baselines

| ID | Experiment | Decision |
|---|---|---|
| E140 | scalar C++ reference | required |
| E141 | `__int128` reference | required |
| E142 | Boost.Multiprecision reference | required |
| E143 | GMP or MPIR reference | comparison-only reference |
| E144 | FLINT/NTL/FFLAS-FFPACK | comparison-only exact-linear-algebra reference |
| E145 | AVX512 VNNI byte GEMM | run on supported CPU |
| E146 | AVX512 IFMA52 CRT | run on supported CPU |
| E147 | CPU/GPU overlap reconstruction | ship if >1.1x end-to-end |

### 16.16 End-To-End Workloads

| ID | Experiment | Decision |
|---|---|---|
| E150 | repeated same-B GEMM | demonstrates persistent B reuse |
| E151 | repeated same-A GEMM | demonstrates persistent A reuse |
| E152 | matrix chain in RNS form | demonstrates delayed reconstruction |
| E153 | Krylov-style repeated multiply | demonstrates RNS workflow value |
| E154 | finite-field workload | validates field primitive |
| E155 | graph adjacency multiply | validates sparse/structured follow-up |
| E156 | many-small batched GEMM | validates persistent scheduler |

### 16.17 Power And Efficiency

| ID | Experiment | Decision |
|---|---|---|
| E160 | raw INT8 power | peak efficiency baseline |
| E161 | persistent RNS power | production efficiency number |
| E162 | reconstructed output power | export efficiency number |
| E163 | thermal steady-state 30 minute run | published benchmark requirement |
| E164 | TOPS/W comparison between target GPUs | hardware recommendation |

## 17. Development Roadmap

### Phase 0: Foundation

Deliverables:

- repository scaffold,
- Apache-2.0 license,
- C ABI headers,
- CPU references,
- benchmark result schema,
- device inspector.

Exit gate:

- E001, E140, E141, E142 pass.

### Phase 1: Single-Modulus GPU GEMM

Deliverables:

- residue conversion,
- one-modulus hipBLASLt GEMM,
- separate reduction baseline,
- ring and field correctness tests.

Exit gate:

- E010, E020 through E024, E030 through E034 pass.

### Phase 2: Fixed 9-Modulus Bounded GEMM

Deliverables:

- prefix-9 scheduler,
- CPU CRT reconstruction,
- one-shot bounded signed and unsigned GEMM,
- vector-ALU baseline.

Exit gate:

- E040, E050, E051, E053, E058 pass.

### Phase 3: Persistent RNS Matrices

Deliverables:

- `rns8_matrix`,
- residue cache,
- packed layout cache,
- RNS output mode.

Exit gate:

- E025 through E027, E052, E055 through E057 pass.

### Phase 4: GPU Reconstruction

Deliverables:

- two-limb GPU Garner reconstruction,
- signed and unsigned export,
- range-error detection.

Exit gate:

- E041, E044, E045, E046, E047 pass.

### Phase 5: Grouped And Adaptive Scheduling

Deliverables:

- per-tile bounds,
- prefix selection,
- grouped scheduler,
- adaptive skip.

Exit gate:

- E014, E015, E060 through E066 pass.

### Phase 6: Fused Reduction

Deliverables:

- CK custom epilogue,
- reciprocal reduction,
- branchless centered correction,
- no-INT32-global-write production path.

Exit gate:

- E070 through E076 pass.

### Phase 7: Custom RDNA Kernels

Deliverables:

- `gfx1201` hot kernels,
- `gfx1100` hot kernels,
- backend autotune selection.

Exit gate:

- E080 through E086 pass and performance gates in Section 11.2 pass.

### Phase 8: Secondary Semantics

Deliverables:

- exact-wide RNS output,
- exact-wide CPU export,
- strict `mod 2^64` byte-limb backend.

Exit gate:

- E090 through E096 and E100 through E107 pass.

### Phase 9: Advanced Paths

Deliverables:

- INT4 decision,
- Strassen decision,
- sparsity decision,
- multi-GPU decision,
- CPU/GPU overlap decision.

Exit gate:

- E110 through E147 produce ship or retire outcomes according to Section 15.

## 18. Completeness Audit

The original research plan left several items open. This specification closes
them as follows.

| Gap | Closed decision |
|---|---|
| Repo-specific context | removed; this is a new-repository spec |
| Project name | `RNS8`, repo `rns8-gemm` |
| Primary semantic target | bounded exact signed/unsigned 64-bit GEMM |
| Default modulus count | 9 for full bounded 64-bit output |
| Modulus set | fixed ordered ladder in Section 5 |
| K-block safety | fixed at 65536 elements |
| Bound metadata | required for bounded APIs |
| Adaptive tile size | default 128 x 128 |
| CRT implementation | GPU fixed-limb Garner for bounded export, CPU multiprecision reference |
| Early termination | disabled by default, opt-in probabilistic mode only |
| Wraparound semantics | byte-limb backend, not odd-modulus CRT |
| INT4 | research only, explicit 1.25x ship gate |
| Fused epilogue | required production path |
| Backend order | hipBLASLt baseline, CK grouped/fused, custom WMMA hot kernels |
| Target OS | Linux ROCm v1 |
| License | Apache-2.0 |
| Performance goals | fixed ship and stretch gates |
| Normal `int64` comparison | separate semantic baselines and speedup gates |
| Multi-GPU | modulus split with 1.55x ship gate |
| Experiment plan | concrete IDs and decision rules |

## 19. Reference Index

### Ozaki, RNS, And Exact Matrix Multiplication

- Daichi Mukunoki, "DGEMM without FP64 Arithmetic - Using FP64 Emulation and
  FP8 Tensor Cores with Ozaki Scheme", arXiv:2508.00441.
- Ozaki Scheme II: https://arxiv.org/abs/2504.08009
- DGEMM on Integer Matrix Multiplication Unit: https://arxiv.org/abs/2306.11975
- Performance enhancement of Ozaki Scheme on IMMU: https://arxiv.org/abs/2409.13313
- DGEMM using Tensor Cores, accurate and reproducible versions:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC7295351/
- Recovering single precision accuracy from Tensor Cores:
  https://arxiv.org/abs/2203.03341
- Cascading GEMM: https://arxiv.org/abs/2303.04353
- Matrix multiplication in multiword arithmetic:
  https://eprints.maths.manchester.ac.uk/2846/
- Chinese remainder theorem overview:
  https://mathworld.wolfram.com/ChineseRemainderTheorem.html
- Dumas, Gautier, and Roch, Chinese remaindering:
  https://www-verimag.imag.fr/~rochj/perso_html/papers/2010-pasco-chinese-remainder.pdf
- LinBox ChineseRemainderSequential:
  https://linalg.org/linbox-html/struct_lin_box_1_1_chinese_remainder_sequential.html
- Exact linear algebra with early termination:
  https://arxiv.org/abs/cs/0501074
- Freivalds matrix product verification:
  https://dblp.org/rec/conf/mfcs/Freivalds79
- Dense linear algebra over finite fields:
  https://arxiv.org/abs/cs/0601133
- M4RI dense GF(2):
  https://arxiv.org/abs/0811.1714
- M4RIE:
  https://arxiv.org/abs/1111.6900
- Strassen original reference:
  https://eudml.org/doc/131927
- Van der Hoeven, gentle moduli:
  https://www.texmacs.org/joris/chinese-macis/chinese-macis.html

### Existing Codebases And Libraries

- GEMMul8: https://github.com/RIKEN-RCCS/GEMMul8
- GEMMul8 numerical results:
  https://github.com/UCHINO-Yuki/GEMMul8_numerical_results
- ozIMMU: https://github.com/enp1s0/ozIMMU
- OzBLAS: https://github.com/RIKEN-RCCS/ozblas
- GRNS: https://github.com/kisupov/grns
- MPRES-BLAS: https://github.com/kisupov/mpres-blas
- M4RI: https://github.com/malb/m4ri
- FFLAS-FFPACK: https://linbox-team.github.io/fflas-ffpack/
- LinBox: https://linalg.org/linbox/linbox/
- FLINT `fmpz_mat`: https://flintlib.org/doc/fmpz_mat.html
- NTL: https://libntl.org/doc/tour-changes.html

### AMD And ROCm

- AMD Radeon RX 7900 XTX specs:
  https://www.amd.com/en/products/graphics/desktops/radeon/7000-series/amd-radeon-rx-7900xtx.html
- AMD Radeon RX 9070 XT specs:
  https://www.amd.com/en/products/graphics/desktops/radeon/9000-series/amd-radeon-rx-9070xt.html
- AMD RDNA3 WMMA guide:
  https://gpuopen.com/learn/wmma_on_rdna3/
- AMD RDNA4 matrix cores:
  https://gpuopen.com/learn/using_matrix_core_amd_rdna4/
- rocWMMA docs:
  https://rocm.docs.amd.com/projects/rocWMMA/en/latest/
- rocWMMA API reference:
  https://rocwmma.readthedocs.io/en/stable/api-reference/api-reference-guide.html
- hipBLASLt docs:
  https://rocmdocs.amd.com/projects/hipBLASLt/en/latest/index.html
- hipBLASLt data type support:
  https://rocm.docs.amd.com/projects/hipBLASLt/en/docs-7.2.1/reference/data-type-support.html
- hipBLASLt Stream-K:
  https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/how-to/how-to-use-streamk.html
- hipBLASLt extension reference:
  https://rocm.docs.amd.com/projects/hipBLASLt/en/docs-7.0.2/reference/ext-reference.html
- Composable Kernel grouped GEMM:
  https://rocm.docs.amd.com/projects/composable_kernel/en/docs-7.1.1/doxygen/html/structck__tile_1_1GroupedGemmKernel.html
- ROCm precision support:
  https://rocmdocs.amd.com/en/develop/reference/precision-support.html
- ROCm Radeon Linux compatibility:
  https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/native_linux/native_linux_compatibility.html
- ROCm Radeon limitations:
  https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/limitations/limitationsrad.html
- ROCm Radeon Triton:
  https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/native_linux/install-triton.html
- MLIR AMDGPU WMMA dialect:
  https://mlir.llvm.org/python-bindings/autoapi/mlir/dialects/amdgpu/index.html
- Clang AMDGPU builtins:
  https://clang.llvm.org/docs/AMDGPUBuiltinReference.html
- hipSPARSELt:
  https://rocm.docs.amd.com/projects/hipSPARSELt/en/docs-7.1.0/reference/supported-functions.html

### CUDA And Scheduling References

- NVIDIA Matrix Multiplication Background Guide:
  https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html
- cuBLAS documentation:
  https://docs.nvidia.com/cuda/cublas/
- CUTLASS:
  https://github.com/NVIDIA/cutlass
- CUTLASS efficient GEMM:
  https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/efficient_gemm.md
- CUTLASS GEMM API:
  https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/gemm_api_3x.md
- NVIDIA PTX ISA:
  https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
- Triton grouped GEMM:
  https://triton-lang.org/main/getting-started/tutorials/08-grouped-gemm.html
- PyTorch MoE grouped GEMM blog:
  https://pytorch.org/blog/accelerating-moes-with-a-triton-persistent-cache-aware-grouped-gemm-kernel/
- Stream-K:
  https://arxiv.org/abs/2301.03598
- Stream-K++:
  https://arxiv.org/abs/2408.11417
- TMA-adaptive FP8 grouped GEMM:
  https://arxiv.org/abs/2508.16584

### CPU SIMD And Integer Arithmetic

- AVX512 IFMA overview:
  https://en.wikichip.org/wiki/x86/avx512_ifma
- Intel AVX512 fast modular multiplication:
  https://builders.intel.com/solutionslibrary/intel-avx-512-fast-modular-multiplication-technique-technology-guide
- Intel HEXL:
  https://arxiv.org/abs/2103.16400
- Accelerating big integer arithmetic using Intel IFMA:
  https://cris.haifa.ac.il/en/publications/accelerating-big-integer-arithmetic-using-intel-ifma-extensions
- GMP carry propagation:
  https://gmplib.org/manual/Assembly-Carry-Propagation.html
- GMP Karatsuba:
  https://www.manpagez.com/info/gmp/gmp-5.0.5/gmp_88.php
- GMP Toom-3:
  https://www.manpagez.com/info/gmp/gmp-5.0.5/gmp_89.php
- GMP FFT multiplication:
  https://www.manpagez.com/info/gmp/gmp-5.0.5/gmp_92.php
- Harvey truncated multiplication:
  https://arxiv.org/abs/1703.00640
- Mulders short product:
  https://www.sciencedirect.com/science/article/pii/S0747717103001172

### Additional Low-Precision And RNS References

- Tensor-core FFT/NTT:
  https://experts.illinois.edu/en/publications/accelerating-fourier-and-number-theoretic-transforms-using-tensor/
- HadaCore FWHT:
  https://arxiv.org/abs/2412.08832
- RNS TPU concept:
  https://maitrix.com/technology/rns-tpu/
- High-performance computation in RNS using floating-point arithmetic:
  https://www.mdpi.com/2079-3197/9/2/9
- Exact sparse matrix-vector multiplication on GPUs:
  https://arxiv.org/abs/1004.3719

## 20. Final Architecture

Production bounded 64-bit path:

```text
int64 or uint64 source
  -> validated bounds metadata
  -> centered INT8 RNS packing using default modulus ladder
  -> persistent modulus-major residue matrices
  -> grouped INT8 matrix GEMM over required modulus prefixes
  -> fused INT32-to-residue reduction
  -> RNS output
  -> requested GPU fixed-limb CRT export to int64 or uint64
```

Production strict wraparound path:

```text
uint64 source
  -> base-256 byte limbs
  -> 36 grouped INT8 GEMMs
  -> Comba diagonal accumulation
  -> delayed carry propagation
  -> low 64-bit output
```

The central performance thesis is:

```text
For bounded exact 64-bit output, a persistent RNS8 matrix engine on RX 9070 XT
must reach at least 25 TOPS-equivalent for RNS output and 15 TOPS-equivalent
with int64 reconstruction on 8192 square GEMM. Anything below those gates means
the implementation has failed to amortize or fuse the non-GEMM work.
```
