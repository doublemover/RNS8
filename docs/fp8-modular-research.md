# Explicit FP8 modular research

The public `rns8_prepare_fp8_recipe` / `rns8_gemm_fp8_research` owner implements
two separately identified formats: CDNA3 `gfx942` E4M3 **FNUZ** and
RDNA4 `gfx1201` E4M3 **OCP**, with fixed exact integer encodings. It accepts finite ring/field moduli2..256 through ordinary finite
matrix ownership. The separate host integer split below has its own representation;
default integer dispatch is unchanged.

Every signed storage byte is split into two balanced radix16 digits. Four
matrix products accumulate separate FP32 vectors: gfx942 uses
`v_mfma_f32_16x16x32_fp8_fp8`; gfx1201 uses
`v_wmma_f32_16x16x16_fp8_fp8`. The
largest digit magnitude is8; K slabs of65536 bound every partial sum by2^22.
Each floating partial is converted to integer and reduced before weights
1/16/16/256 are applied, then one centered residue byte is stored. Method0 fuses recombination. Method1
materializes four unweighted INT32 product-residue planes and launches a separate
recombination kernel, providing a simple control for fusion. Its explicit retained
budget is4+16*M*N bytes, compared with4bytes for the fused method. Tails use
zero fragments; slab carries are reduced integers. No numeric-one scale byte
is guessed: each unscaled opcode uses its explicitly selected FNUZ or OCP
digit table.

`tools/fp8_modular_certificate.py` ties the four-product integer algebra,
selected FNUZ/OCP alphabet, K cap and named instruction model to source/build
identities.
The FNUZ bias8 format is separately named in the floating oracle; OCP bias7 is
not substituted. The conditional FP32 arithmetic model is not evidence of actual
internal MFMA rounding or lane mapping. Preparation requires the explicit
`RNS8_FP8_ASSUME_NAMED_INSTRUCTION_MODEL` research acknowledgement and probes the
actual compiled code object's capability. Missing opcode/target rejects.
Reported `target_qualified` and `performance_qualified` remain false.

Preparation retains its context/plan and four-byte capability/status allocation and optional materialized product planes.
Execution uses ordinary finite validation and reader/writer/workspace leases,
invalidates before possible writes, and publishes after successful completion.
A device status guard rejects nonfinite, fractional and out-of-model partials
before integer conversion; its readback is inside the complete timed call.
A detected contradiction disables that research owner and blocks publication.
This detects explicit model contradictions, not every possible integer-valued
internal rounding error.
An undrainable failure retains every borrowed resource and disables arithmetic;
destroy retries drain before releasing anything. A nonnull cleanup-only owner
from failed preparation must likewise be destroyed. Calls are synchronous and
shared handles require serialized host access.

Authored CPU fixtures check every source byte/digit pair and all moduli; GPU
fixtures cover complete Boost products, padding, aliases, every modulus2..256 at K33, plus moduli2/16/127/251/255/256
at K31/32/33/65535/65536/65537. The host certificates and exact digit/layout checks pass the
[October27 implementation validation](../metadata/implementation-closeout-20261027.json). Compile-only assembly contains the actual
MFMA and WMMA opcodes. Target tests skip on gfx1100; actual lane/accumulator
adversarial execution remains unqualified on both named targets.

## Matched capture source

`rns8-fp8-calibrate` compares public INT8, fused FP8 and materialized FP8 products with identical
finite contracts and padded host output. Cold, repeated-B and resident-AB use
separate cohorts. Setup, producer, export and cleanup are recorded; complete
Boost comparison stays outside production intervals. Three sustained bursts
require16calls and250ms, or explicitly record censoring at4096calls.

Committed cases are under `benchmarks/fp8/`. The collector defaults to writing a
plan, not executing it:

```text
python tools/collect_research_kernels.py --binary build/rns8-fp8-calibrate --cases temp/benchmark-cases/current/fp8/17x19x33-int8.json temp/benchmark-cases/current/fp8/17x19x33-fp8.json --output temp/fp8-plan
```

Explicit `--execute` starts a new collection when hardware/validation is
authorized. Counterbalanced independent processes bind artifact/build/runtime
and actual format; unsupported exits are not measurements. No captures or
speedups are asserted here. The separately implemented CDNA4 source contracts
below cover FP4/FP6/dynamic and mixed formats; live qualification remains open.

Instruction/format references: [LLVM builtins](https://clang.llvm.org/docs/AMDGPUBuiltinReference.html),
[AMD CDNA3 MFMA guide](https://rocm-handbook.amd.com/projects/amd-rocm-optimization-guide/en/develop/compiler-builtins/cdna/cdna3-dense-mfma-builtins.html),
[HIP FP8 formats](https://rocm.docs.amd.com/projects/HIP/en/docs-6.3.1/reference/fp8_numbers.html).

## RDNA4 and three-product source extension

Format2 binds the unscaled `gfx1201` WMMA E4M3 opcode and bias7 encoding;
format1 retains gfx942 MFMA FNUZ bias8. Preparation checks both the requested
format/target and the actual compiled capability. Their wave sizes, K fragments
and accumulator coordinates differ. A shared bijection check covers the new
wave32 map; host launch dimensions use the requested format, never host-pass
architecture macros. Neither format supplies implicit scales or an exact
encoding for arbitrary byte integers such as17.

Methods2/3 implement the separately certified three-product identity. For each
byte, h=sign(x)*ceil(abs(x)/16), l=x-16h and s=h+l. The exact E4M3 alphabets are
h in[-8,8], l in[-15,15], s in[-14,14]. Products Qh/Ql/Qs reconstruct as
240*Qh-15*Ql+16*Qs. Every floating accumulator resets at each65536-term slab;
per-product guards use the actual64/225/196 magnitude bounds, all below2^24.
Integer modular carries combine slabs. Method2 fuses the integer composition;
method3 retains three INT32 product-residue planes, costing4+12*M*N bytes.
Methods0/1 remain the matched four-product control. Model/build hashes bind
format, algebra and method, and no candidate joins AUTO.

The offline certificate tool accepts `--format 1|2 --method 0|1|2|3`. The public
calibrator and compact catalogue include the corresponding `fp8-three` and
`fp8-ocp[-three]` fused/materialized modes. Captures reject a wrong target,
format, method or retained byte count. No new measurement is claimed.

The fragment/opcode basis is AMD's [RDNA4 WMMA reference](https://rocm-handbook.amd.com/projects/amd-rocm-optimization-guide/en/latest/compiler-builtins/rdna/rdna4-dense-wmma-builtins.html).
The [AMD FP8 progression](https://rocm.blogs.amd.com/software-tools-optimization/rdna-fp8-gemm/README.html)
provides a bring-up reference; its restricted input validation does not qualify
RNS8's larger digit alphabets or accumulator model.

The RDNA4 FP8 source map follows AMD's separately named FP8 policy: each
lane group supplies eight contiguous K bytes, with eight contiguous accumulator
rows per lane group. This is distinct from the generic FP16 fragment diagram.
The reviewed [official FP8 source blob](https://api.github.com/repos/ROCm/rocm-blogs/git/blobs/add24ccf438a24f30094e4de98858f005aa503f8)
is identified by Git blob `add24ccf438a24f30094e4de98858f005aa503f8` (the blob
identity, not a commit revision). Compile-only assembly emits the named MFMA and
WMMA instructions; actual accumulator/lane behavior still requires those GPUs.


## CDNA4 FP4, FP6, dynamic scales and mixed formats

Formats3-10 of the same explicit research owner use the real
`__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4` opcode on `gfx950`.
The implementation follows the [CDNA4 ISA, sections6.7 and7.1.5](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf)
and [LLVM builtin interface](https://clang.llvm.org/docs/AMDGPUBuiltinReference.html).
The compile-time operand selectors are E4M3=0, E2M3=2, E3M2=3 and E2M1=4.
Each operand uses its own physical map: FP4/FP6 lane groups own32 contiguous K
values; FP8 groups own16 values followed by another16 at K+64. Mixed formats
therefore load different logical K coordinates into the same register-item index.
The source map follows ISA printed pages51-52; CPU bijections and explicit lane
anchors cover both layouts.

| Public format | A / B | Digit decomposition | Product count |
|---|---|---|---:|
| 3 | FP4 E2M1 / E2M1 | radix4 / radix4 | 16 |
| 4 | FP6 E2M3 / E2M3 | radix8 / radix8 | 9 |
| 5 | FP6 E3M2 / E3M2 | radix8 / radix8 | 9 |
| 6 | FP8 E4M3 / E4M3 | radix16 / radix16 | 4 |
| 7,8 | E4M3 / E2M1, or reversed | radix16 / radix4 | 8 |
| 9,10 | E2M3 / E2M1, or reversed | radix8 / radix4 | 12 |

Balanced digits cover all signed storage bytes, including -128. Four radix4
digits use magnitudes at most2, three radix8 digits at most4, and two radix16
digits at most8. Each operand is encoded directly as an exact dyadic value;
no lossy native-to-floating conversion is used. Methods0/1 select fused integer
recombination or retained INT32 product-residue planes. The conditional model
and actual code-object probe remain separate; unsupported targets reject.

Optional `a_scale_exponents`, `b_scale_exponents` and `scale_count` bind one
signed exponent per ceil(K/128) instruction block. Both arrays are copied at
preparation and included in the model identity. Each 128-term instruction uses
one scale for all its 32-element scale granules. E8M0 bias127 encodes numeric
one as127; the hardware scale register repeats that byte in all four quarters.
The physical digit encoding divides by the chosen power of two exactly, so
applying the instruction scale recovers the same logical digit. Admission
rejects any exponent that cannot represent the complete promised alphabet.
Allowed exponents are FP4 [-1,1], E2M3 [0,3], E3M2 [-2,4] and E4M3 [-5,9].

Every product accumulator starts at zero for each 128-term block. Its finite,
integral, magnitude-checked result is converted and reduced before applying
integer radix weights and merging with earlier blocks. No floating carry crosses
scale regimes. Tiny/large scale changes, opposite scale order, cancellation,
zero terms and a partial final granule therefore share a precisely stated
reset/conversion boundary. Under the explicit instruction model, every proper
partial has magnitude at most8192; that arithmetic bound alone does not certify
internal hardware rounding or scale application order.

Retained device bytes are4+2*ceil(K/128), plus4*product_count*M*N for method1.
An additional2*ceil(K/128) host bytes retain the scale source through safe cleanup.
The operation query reports this cost and safe_k128. Info exposes scale blocks,
mandatory reset boundaries, operand selectors and product count. Failed copies,
launches or drains use the same resource retention/publication policy as FP8;
no failed result becomes current. None of these formats joins AUTO.

The compact catalogue includes fixed/dynamic fused/materialized `mxfp*` modes.
Capture schema `rns8_public_fp8_microcase_v2` requires scale policy, scale schedule
hash, product count, reset count and complete retained-byte accounting. The
collector rejects mismatched target, mode, scales or omitted reset cost.
`tools/mxfp_contract.py` reuses the exact floating decoder to prove the named
alphabet/scale encodings and integer reconstruction model. Authored CPU and
HIP fixtures cover every byte, all supported exponent extremes, prime/composite
moduli, padded tails around32/128 and multi-block K65537, with Boost output
references. The CPU model/certificate checks pass the
[current consolidated batch](../metadata/remaining-code-validation-20261027.json);
gfx950 assembly emits the scaled f8f6f4 MFMA instruction. Actual gfx950 execution,
internal arithmetic adversaries and measured comparisons remain open.


## Exact native integer splitting

`rns8_query_integer_split_i64`, `rns8_split_i64` and `rns8_merge_i64` implement
an allocation-free host baseline over caller-owned buffers. The public
[format/descriptor](../include/rns8/integer_split.h) fixes ten low-first signed
radix128 planes: x=sum(d[s]*2^(7*s)). Each nonzero digit has the input sign,
with magnitude at most127; the highest digit has magnitude at most1.
Magnitude extraction uses unsigned arithmetic, including INT64_MIN. No native
integer passes through double, and values beyond2^53 retain every bit.

The descriptor binds rows, columns, native element stride, component byte row
stride and component byte plane stride. Zero component strides select compact
storage. Query returns exact accessible spans through the final logical cell,
component count, radix and layout version. The caller allocates and releases
all memory; no context, device, allocator callback or hidden lifetime is involved.
Split/merge touch logical cells alone, preserving row/plane/outer padding.
Address overflow, overlap, alignment, dimensions, ABI/format and insufficient
capacity reject before writes. Merge first validates every component cell:
-128, mixed-sign digits and an invalid top digit are malformed; an otherwise
canonical magnitude outside int64 is a range error. A bad final cell preserves
the complete old destination. Inputs must remain stable throughout the call.

This is the native integer-splitting contract owned by #45. Floating-input
FP64 emulation has a different numerical contract; the split API contains
signed INT8 digits and no FP8 bytes or scale metadata. The old ambiguous
reserved decomposition entrypoint is removed in the same cutover.

The installed strict C17 consumer exercises full-width values, padding, repeated
calls and capacity checks through either exported linkage:

```text
cmake -S examples/integer-split -B temp/integer-split-consumer -DCMAKE_PREFIX_PATH=/path/to/installed/RNS8 -DRNS8_SPLIT_USE_SHARED=ON
cmake --build temp/integer-split-consumer
ctest --test-dir temp/integer-split-consumer --output-on-failure
```

Use `RNS8_SPLIT_USE_SHARED=OFF` for the static target. The development package
ships the same source under `share/RNS8/examples/integer-split`. CPU fixtures
independently reconstruct every digit with Boost, cover signed byte values,
full-width/power-of-two/2^53 boundaries and deterministic arbitrary bit patterns,
and check failure preservation for malformed final cells and invalid spans.
The [full implementation validation](../metadata/full-implementation-validation-20261027.json)
passes all four new CPU-ASan regressions and both installed C17 linkage cases.
Strict GCC C17/C++17 linkage/execution also passes under Ubuntu24.04/WSL.
Floating instruction qualification retains its separate target-specific gates.
