# CDNA5 TDM and multicast source

This private direct-HIP experiment implements three data-movement forms over
the same raw native64 tiles: cooperative ordinary global-to-LDS loads,
two-stage tensor data movement (TDM), and two-/four-workgroup cluster multicast.
Both an integer-ALU control and an explicit wave32 INT8 WMMA16x16x64 consumer
are implemented, followed by fused modular byte stores. The private compute
selector requires its own actual code-object capability; no public native path
is replaced and no matrix throughput is claimed from compilation.

`backend_common/native_tdm.hpp` owns pure shape/span/resource admission and a
minimal 2D tensor descriptor. Dimensions, tile widths, 57-bit global addresses,
LDS offsets, alignment and 48-bit **element** strides are checked before use.
Sixteen-by-sixteen output tiles consume16x64A and64x16B native words, with two
32KiB-total shared buffers. Native conversion happens after transfer, preserving
full signed/unsigned64-bit source semantics. Modulus groups1/2/4 and prefixes1..20
are explicit; every64-term integer partial reduces before carry accumulation.

The TDM path zeroes tail storage, waits for preceding LDS writes, issues raw
tensor loads and waits for tensor completion before workgroup consumers read.
The next stage prefetches into the alternate buffer; barriers separate readers
from buffer reuse. Cluster participants issue matching requests and multicast
masks, including inactive groups in an incomplete final prefix cohort. A second
cluster mapping shares raw A between output-column neighbors and transfers each
neighbor's distinct B separately; padded inactive neighbors still participate. After
local readers finish, one wave per workgroup signals cluster barrier-3 and all
waves wait. No leader-only request is presented as multicast.

Compilation detects the actual gfx1250 tensor/wait builtins and cluster
attribute. Code-object capability flags distinguish TDM from cluster support.
Missing transfer/cluster/matrix capabilities return unsupported. LLVM23 and LLVM22 tensor-load builtin
arities are handled explicitly. LLVM23 accepted the active gfx1250 device body
in a device-only syntax check using the available HIP7.1 headers and no GPU
libraries. That does not establish link compatibility, an installable supported
CDNA5 stack or device correctness. The local ROCm/HIP installation is unchanged.

The raw admitted wrapper validates pointers, aliasing, alignment and the complete
resource envelope, clears status, launches, drains, reads status and returns
before a caller can publish output. Raw callers must keep buffers and preparation
state alive until successful drain; an uncertain completion never permits free.
No public matrix owner silently selects this experiment.

## Fixtures and measurements

A dedicated raw64 transfer litmus checks every word and zero tail before relying
on arithmetic. Authored GPU fixtures cover cluster1/2/4, signed/unsigned native
extremes, odd padded tails, partial prefix groups, K31/32/33/63/64/65/65537 and full Boost
residue products. CPU fixtures check descriptor units, fields and bounds and
complete output ownership and a unique wave32 operand/result coordinate model.
The matrix map follows the explicit gfx1250 CK subgroup/K-block facts and still
requires actual lane-mapping qualification; the GPU oracle checks all outputs. These new fixtures have not executed.

`rns8-tdm-calibrate` and `benchmarks/tdm/` supply ordinary/TDM/multicast controls
with identical input/output checksums, preparation accounting, status/completion
inside each timed call and full Boost/padding comparison outside it. Outputs
are residues: no native reconstruction uniqueness is claimed. Ideal raw-source read amplification is modeled separately from actual HBM bytes;
profiler stall, occupancy and spill fields remain null until measured. Three sustained
bursts and independent processes use the same contracts as the
[FP8 collector](fp8-modular-research.md#matched-capture-source).

```text
python tools/collect_research_kernels.py --binary build/rns8-tdm-calibrate --cases benchmarks/tdm/17x19x33-ordinary.json benchmarks/tdm/17x19x33-tdm.json benchmarks/tdm/17x19x33-multicast2.json benchmarks/tdm/17x19x33-multicast4.json --output temp/tdm-plan
```

This writes a plan only. Hardware litmus, emitted ISA, wait/barrier stress,
negative resource results, spills/occupancy and matched bandwidth/elapsed gains
remain [issue52](https://github.com/doublemover/RNS8/issues/52) qualification gates.
There are no new benchmark results or promoted cohorts.

The additional `*-wmma.json` and `*-wmma-neighbor.json` fixtures select actual
INT8 matrix consumers and both cluster mappings. Comparisons keep scalar and
matrix primitive cohorts separate.

References: [CK gfx1250 WMMA source](https://github.com/ROCm/composable_kernel/blob/f176bf5b118309cb475b20841daf3f17e5231017/include/ck/tensor_operation/gpu/warp/wmma_gemm.hpp),
[LLVM builtin declarations](https://github.com/llvm/llvm-project/blob/8c56646054e45b08ed6dee6fb23417f4754997ba/clang/include/clang/Basic/BuiltinsAMDGPU.td),
[AMD CDNA5 ISA](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna5-instruction-set-architecture.pdf),
[LLVM DMA operations](https://llvm.org/docs/AMDGPUDMAOperations.html),
[AMD TDM GEMM ladder](https://rocm.blogs.amd.com/software-tools-optimization/hipkittens-gemm-ladder/README.html).
