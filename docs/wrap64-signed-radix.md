# Carry-recoded wrap64 matrix candidate

The private direct-HIP entry `hip_direct_wrap_radix_wmma_device` admits compact
canonical byte-limb inputs/output for strict modulo 2^64 multiplication only.
It has two explicit candidates: signed carry-recode and native unsigned IU8.
Neither changes public dispatch, storage, or AUTO selection.

The signed view computes `(x + H) ^ H`, H=0x8080808080808080, once per loaded
cell per 16x16x16 tile. Unsigned addition propagates carries before bytes become
signed digits. Its weighted digits sum to x modulo 2^64. Exactly 36 pairs with
i+j<8 remain; higher pairs vanish in the ring. Both lanes in each half-wave
load the replicated RDNA3 fragment mapping. Each matrix instruction starts from
zero, uses 16 padded K terms, and disables clamp. The signed bound is 16*16384;
the unsigned control bound is 16*65025. Each exact INT32 result converts through
signed64 to unsigned64 before shifting by 8*(i+j); accumulation wraps only in
the intended unsigned ring. No INT32 diagonal sums are formed.

The implementation stages 256 full recoded A and B cells in 4096 shared bytes,
reusing them across all pairs. Input/output cell byte numbering is explicitly
little-endian through shifts; no alignment-sensitive native loads or signed
out-of-range casts are needed. Row/column/K tails are zero-masked, including
recode(0)=0. Output stores reconstruct canonical low64 bytes. The native unsigned
control uses identical tiles, staging, loop and output boundaries without recode.

Admission checks complete compact spans and rejects output overlap with either
input. Read-only A/B alias is legal. A four-byte device capability allocation
and readback precede output writes. Only gfx1100/1101/1102 with the actual IU8
builtin are admitted; unsupported targets cannot report successful computation.
The synchronous private call reports whether output writes may have begun;
a failed issued launch invalidates the output. No persistent recoded cache or
borrowed-stream contract is introduced.

`tools/data/wrap64_signed_radix_certificate.json` names the exact arithmetic
schedule and layout version. `arithmetic_certificate.py` checks the ring
construction, omitted terms, exact pair bounds and unsigned folding independently
of C++ arithmetic. The certificate never grants runtime dispatch eligibility.
Host tests cover shifted exhaustive 16-bit recodes, full-width adversarial and
random products, independent byte/U64/Boost oracles and broken recodes. GPU
fixtures cover tails, canaries, aliases, budgets and K around tile, 4096,
33025, 65536 and 131071 boundaries. Tests and compilation remain pending.

This first implementation deliberately folds after every 16-term MMA. It does
not claim a 65536-term accumulation optimization, ISA qualification, or a speedup.
The longer mathematical signed cap of 131071 and unsigned cap of 33025 are
certificate bounds, not current schedule lengths. Matched comparison against
direct-HIP v4 and corrected rocWMMA, ISA/resource receipts, and sustained
end-to-end benchmark evidence remain required before any measured integration.

The authored [matched calibration driver](wrap64-calibration.md) supplies
resident, roundtrip and cold-input controls with common canonical byte limbs,
hook-disabled production timing and complete outside references. Its long-K
cases explicitly omit rocWMMA above that existing control's K32768 cap.
The new driver, collector, case sets and fixtures have not run.

## Separate CDNA signed-INT8 MFMA implementation

`hip_direct_wrap_radix_mfma_device` is a distinct, opt-in private synchronous
candidate, not an RDNA fragment reinterpretation. It selects only an actual
compiled builtin on gfx90a, gfx942 or gfx950. Their instruction K widths are
16, 32 and 64 respectively; the signatures consume one I32, one I64 or four I32
packed operand registers per lane. The wave has 64 lanes and produces four I32
accumulators per lane. All instruction accumulators start at zero; each is
bounded by `K*16384 <= 1048576`, then folded separately into the U64 ring.
The zero-scratch native-U64 vector-ALU control uses one output per lane and
ordinary unsigned multiplication, without either byte convolution or an MMA.

`MfmaI8Layout<K>` specifies and compile-time checks bijective coordinates:
A `(lane%16, (lane/16)*(K/4)+byte)`, B with these axes exchanged, and C
`((lane/16)*4+reg, lane%16)`. Each full canonical cell is loaded and recoded once
per K instruction tile and retained in lane registers across all 36 pairs.
There is no explicit LDS or barrier in this kernel. This is a source contract,
not a measured register, spill, issued-instruction or occupancy result.

The distinct builtin signatures come from [Clang's AMDGPU builtin reference](https://clang.llvm.org/docs/AMDGPUBuiltinReference.html)
and AMD's [CDNA2](https://rocm-handbook.amd.com/projects/amd-rocm-optimization-guide/en/docs-1.0.0/compiler-builtins/cdna/cdna2-dense-mfma-builtins.html),
[CDNA3](https://rocm-handbook.amd.com/projects/amd-rocm-optimization-guide/en/develop/compiler-builtins/cdna/cdna3-dense-mfma-builtins.html)
and [CDNA4](https://rocm-handbook.amd.com/projects/amd-rocm-optimization-guide/en/docs-1.0.0/compiler-builtins/cdna/cdna4-dense-mfma-builtins.html)
references. CDNA2/3 coordinates were independently derived from the equations in
AMD's [matrix instruction calculator at 2ef9189](https://github.com/ROCm/amd_matrix_instruction_calculator/tree/2ef91896bcdc4d26624f952e5c905c787cd9bc9e).
The CDNA4 extension uses its documented 16x16, single-block, wave64 shape and
16 input bytes per lane; its physical mapping still requires the authored
one-hot/coordinate and exact-output hardware qualification. No calculator code
was copied or executed.

Three `tools/data/wrap64_signed_mfma_gfx*.json` certificates bind the ring
construction, exact instruction width, target/opcode and wave-coordinate model.
The checker keeps physical layout and execution `not_run`. Captures bind the
certificate's file SHA256 and layout version; changing either invalidates the
collector contract. Shared capability probes retain storage after an uncertain
drain instead of freeing memory potentially still in use.

The harness now includes native-U64 on both architecture families and CDNA
three-way controls against direct-v4. `repeated_b` retains canonical B, packs A
and exports C in every batch; it is **not** a persistent recoded-B cache. No
source-generation cache keys are invented for a cache that does not exist.
Device compilation, differential execution, emitted-ISA signedness/layout,
resource receipts, whole-call performance and any public promotion remain
unverified. The source implementation is not a profitable-cohort result.
