# Exact INT8 instruction-chain probes

The internal `hip_direct_instruction_roof` calibration path runs native RDNA3
signed-byte WMMA with one, two, four or eight independent accumulators per wave.
One chain exposes a serial accumulator dependency; additional chains provide
instruction-level independence within a wave. A separate explicit mode applies
centered modulus-251 reduction after each WMMA instruction. It measures the
combined matrix/reducer sequence, not a standalone reducer throughput claim.

Each input array supplies one signed byte per chain, replicated into the actual
WMMA fragments at runtime. Every wave stores all 256 output cells of every chain.
There is no discarded accumulator or validation in the timed loop. Inputs include
full signed-byte extrema; at most 4096 iterations give K=65536 and a raw absolute
accumulator bound 2^30. The reduced sequence has at most 16*128^2+125 before each
reduction. Neither bound depends on cancellation or a small final result.

The pure query reports two chain-byte input arrays, full int32 output size,
wave-instruction count and integer operations (8192 per WMMA). The caller provides
all device storage and a separate four-byte capability word. A runtime capability
probe rejects unsupported targets before output writes; the kernel has no scalar
fallback. Borrowed buffers must remain alive until completion, including draining
failed work. This does not change production selection or reviewed caches.

The event label isolates the issued chain kernel from the preceding capability
probe. Host-call time includes that probe and synchronization and must be reported
separately. Single-wave event latency includes loop, setup and output stores;
ns per wave instruction is not a direct hardware cycle latency. Saturated tests
must vary resident waves and independent chains, record achieved occupancy and
inspect the emitted instruction chain and spills. Compiler unrolling, scheduling
or instruction elimination must be checked against ISA before interpreting any
roof. Output traffic is modeled, not an observed memory counter.

Authored CPU fixtures check counts, address/cap bounds and signed extrema. HIP
fixtures check every output cell against an independent integer recurrence,
iteration cap-1/cap, a partial four-wave block, all chain counts, both modes and
output guards. Compilation, execution, benchmark integration and disassembly
qualification remain pending. These sources establish no measured roof or speedup.
