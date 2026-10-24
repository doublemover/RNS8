# Checked examples

These complete C++17 programs exercise the public API with status checks and
independently known results. They currently select CPU reference contexts;
`RNS8_BUILD_EXAMPLES=ON` builds them by default. GPU benchmark entry commands
are in the [Windows](../docs/platform-windows.md) and
[Linux](../docs/platform-linux.md) guides. Practical GPU workload programs are
tracked separately in [#62](https://github.com/doublemover/RNS8/issues/62).

| Program | Target | Arithmetic and output |
|---|---|---|
| [Bounded signed product](bounded_i64_oneshot.cpp) | `rns8-example-bounded-i64-oneshot` | Exact i64 with an explicit bound |
| [Finite ring](finite_ring_u8_oneshot.cpp) | `rns8-example-finite-ring-u8-oneshot` | One explicit byte-sized modulus |
| [Strict wrap64](wrap64_u64_oneshot.cpp) | `rns8-example-wrap64-u64-oneshot` | Low 64 bits with byte-limb semantics |
| [Exact-wide export](exact_wide_limb_export.cpp) | `rns8-example-exact-wide-limb-export` | Fixed-width limbs from range-proven residues |
| [CPU continuation](downstream-cmake/exact_wide_cpu_continuation.cpp) | `rns8-example-exact-wide-cpu-continuation` | Prefix selection, budget rollback and matching-prefix export |

Run a target's executable from the configured build directory. For example,
`build/windows-msvc-hip-debug/rns8-example-bounded-i64-oneshot.exe` runs the CPU
example even though that build also enables HIP. The
[installed-package consumer](downstream-cmake/README.md) explains static/shared
linkage and the continuation workflow without private repository headers.
Explicit structured sparse examples and their acceptance remain owned by
[#51](https://github.com/doublemover/RNS8/issues/51).
