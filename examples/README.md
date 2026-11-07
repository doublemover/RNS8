# Checked examples

These complete C++17 programs exercise the public API with status checks and
independently known results. `RNS8_BUILD_EXAMPLES=ON` builds them by default.
The [practical workload catalogue](workloads/README.md) adds an exactness
walkthrough, integer Gram/cross-products, GF(251) transforms, and matched
resident versus export/reimport chains with explicit GPU backend/device
selection, independent CPU comparators, range admission and measurement tools.
An additional experimental split-complex workload compares public 3M/4M with a
four-product Boost reference; its new source/fixtures await consolidated validation.
The original four workloads and Windows installed consumers have
[dated CPU/direct-HIP evidence](../docs/local-all-features-20261023.md). Updated
source/dependencies, the fifth workload, application performance and native Linux
installed qualification remain pending. The tiny programs below default to CPU reference contexts; the two sparse
programs additionally offer an explicit GPU invocation.

| Program | Target | Arithmetic and output |
|---|---|---|
| [Bounded signed product](bounded_i64_oneshot.cpp) | `rns8-example-bounded-i64-oneshot` | Exact i64 with an explicit bound |
| [Finite ring](finite_ring_u8_oneshot.cpp) | `rns8-example-finite-ring-u8-oneshot` | One explicit byte-sized modulus |
| [Strict wrap64](wrap64_u64_oneshot.cpp) | `rns8-example-wrap64-u64-oneshot` | Low 64 bits with byte-limb semantics |
| [Exact-wide export](exact_wide_limb_export.cpp) | `rns8-example-exact-wide-limb-export` | Fixed-width limbs from range-proven residues |
| [CPU continuation](downstream-cmake/exact_wide_cpu_continuation.cpp) | `rns8-example-exact-wide-cpu-continuation` | Prefix selection, budget rollback and matching-prefix export |
| [Finite-ring Sparse-A](downstream-cmake/sparse_finite_ring.cpp) | `rns8-example-sparse-finite-ring` | Explicit 2:4 compression/import, composite mod250 (CPU) or255 (explicit GPU) and independent scalar reference |
| [Bounded signed Sparse-A](downstream-cmake/sparse_bounded_i64.cpp) | `rns8-example-sparse-bounded-i64` | Per-plane centered default-basis bytes, bound4096 and independent exact signed reference |

Run a target's executable from the configured build directory. For example,
`build/windows-msvc-hip-debug/rns8-example-bounded-i64-oneshot.exe` runs the CPU
example even though that build also enables HIP. The
[installed-package consumer](downstream-cmake/README.md) explains static/shared
linkage and the continuation workflow without private repository headers.
The new structured Sparse-A examples also ship with that standalone installed
consumer and use only public headers/targets. With no arguments they explicitly
request CPU expansion. `--hip [--device 0..63]` requests only the builtin sparse
provider, checks the shared target/shape capability before allocating matrices,
and returns a visible77 skip for unsupported targets/builds, including gfx1100.
Supported source targets are gfx942/gfx1200/gfx1201; selected backend identity
and exact full outputs are checked, with no fallback. The optional GPU examples
and package registrations are source-only until the consolidated validation
batch. A generated capability/profile matrix and actual target-specific
arithmetic/ISA/preparation/reuse evidence remain open under
[#51](https://github.com/doublemover/RNS8/issues/51).
