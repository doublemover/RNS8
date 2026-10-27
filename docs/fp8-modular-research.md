# Explicit gfx942 FP8 modular research

The public `rns8_prepare_fp8_recipe` / `rns8_gemm_fp8_research` owner implements
one initial format: CDNA3 `gfx942` E4M3 **FNUZ**, with fixed exact integer
encodings. It accepts finite ring/field moduli2..256 through ordinary finite
matrix ownership. There is no Ozaki decomposition or default integer dispatch.

Every signed storage byte is split into two balanced radix16 digits. Four
`v_mfma_f32_16x16x32_fp8_fp8` products accumulate separate FP32 vectors. The
largest digit magnitude is8; K slabs of65536 bound every partial sum by2^22.
Each floating partial is converted to integer and reduced before weights
1/16/16/256 are applied, then one centered residue byte is stored. Method0 fuses recombination. Method1
materializes four unweighted INT32 product-residue planes and launches a separate
recombination kernel, providing a simple control for fusion. Its explicit retained
budget is4+16*M*N bytes, compared with4bytes for the fused method. Tails use
zero fragments; slab carries are reduced integers. No numeric-one scale byte
is guessed: this unscaled opcode uses the exact FNUZ digit table.

`tools/fp8_modular_certificate.py` ties the four-product integer algebra,
FNUZ alphabet, K cap and named instruction model to source/build identities.
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
at K31/32/33/65535/65536/65537. Device-only gfx942 syntax passes; these new tests
and actual gfx942 ISA/accumulator adversarial probes have not run.

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
speedups are asserted here. FP4, FP6, OCP FP8, dynamic scales and mixed-format
expressions require separate implementations/contracts, beyond this initial
[issue45](https://github.com/doublemover/RNS8/issues/45) form.

Instruction/format references: [LLVM builtins](https://clang.llvm.org/docs/AMDGPUBuiltinReference.html),
[AMD CDNA3 MFMA guide](https://rocm-handbook.amd.com/projects/amd-rocm-optimization-guide/en/develop/compiler-builtins/cdna/cdna3-dense-mfma-builtins.html),
[HIP FP8 formats](https://rocm.docs.amd.com/projects/HIP/en/docs-6.3.1/reference/fp8_numbers.html).
