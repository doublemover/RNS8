# rocWMMA And AMDGPU Builtin Backend

Opt-in Windows `gfx1100` rocWMMA accelerator backend plus reserved AMDGPU
builtin hot-kernel path.

The rocWMMA backend is compiled only with `RNS8_ENABLE_ROCWMMA=ON`. It uses the
pinned repo-local rocWMMA headers and RNS8-owned HIP kernels to pack centered
signed residues into 16-aligned panels, execute `int8 x int8 -> int32` WMMA,
and fuse INT32 accumulators back to centered `int8_t` residues without global
INT32 scratch output.

The implemented rocWMMA coverage includes fixed-prefix bounded RNS plans,
adaptive per-tile bounded schedules, exact-wide RNS output, and finite u8. The
ISA gate requires the expected `v_wmma` instruction and rejects scalar
divide/remainder/reciprocal mnemonics plus unintended INT32 global stores.
The same object also contains an internal strict wrap64 byte-GEMM36 candidate:
it consumes compact byte-limb device buffers, uses signed WMMA plus high-bit
correction WMMA terms to recover unsigned byte products, and writes compact
byte-limb output for comparison against direct HIP and the CPU byte-pair oracle.
That candidate is not a public backend and is not AUTO-selected. Benchmark
fixtures currently record host wall-clock evidence only, so the backend keeps
`performance_validated=false` until reviewed captures prove a target-shape win.

Dependency discovery and primitive compile probes remain evidence only; they
do not enable rocWMMA by themselves. `RNS8_ENABLE_AMDGPU_BUILTINS` still fails
fast because no target-specific builtin correctness kernels have been added.
