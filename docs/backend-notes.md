# Backend Notes

Backend status:

- CPU reference: implemented and tested.
- Direct HIP: implemented for device inspection, signed/unsigned residue
  conversion, persistent device-resident RNS matrix buffers, one-modulus
  correctness smoke, fused INT32-to-centered-residue reduction, and bounded
  i64/u64 GPU export through the supported prefix-20 bound.
  Public bounded GEMM can execute the direct HIP pack, RNS GEMM, and export
  path, with K split into blocks no larger than 65536 before centered residue
  reduction.
- hipBLASLt: not implemented.
- CK: not implemented.
- rocWMMA/AMDGPU builtins: not implemented.
- Wraparound byte-limb backend: CPU reference implemented for one-shot and
  persistent byte-limb matrix APIs. A private direct-HIP byte-limb Comba smoke
  kernel exists for correctness comparison only; public HIP wrap64 backend
  support, optimized byte GEMMs, and accelerator signedness corrections are not
  implemented.

Unsupported backends must return unsupported status. They must not expose stub
paths that appear to validate GPU behavior.

The future backend directories under `src/` are scaffold markers only. They
exist to keep ownership boundaries visible while preserving the rule that no
accelerator path counts until it has compiled kernels and exact CPU
differential validation.

Optional accelerator discovery is platform evidence, not backend enablement.
`tools/check_dependencies.py` and the `FindRNS8HIPBLASLT.cmake`,
`FindRNS8CK.cmake`, and `FindRNS8ROCWMMA.cmake` modules can report candidate
hipBLASLt, CK, and rocWMMA component files. These probes are shallow
header/library/tool discovery only. They do not compile kernels, link an
accelerator backend, run device capability checks, or satisfy correctness
requirements.

The direct HIP pack kernels copy logical host `int64_t` and `uint64_t` inputs
to a matrix-owned device upload buffer and write centered residues into
matrix-owned device residue storage. The direct HIP RNS GEMM path consumes those
device residues directly, launches one thread per output element per modulus,
and reduces each INT32 K-block sum to a centered residue in the kernel. For K
above 65536, it launches multiple block kernels and accumulates the centered
residue on device.

Bounded direct HIP export reconstructs i64/u64 outputs on device with a fixed
three-limb Garner kernel for prefixes up to `RNS8_MAX_SUPPORTED_PREFIX`, writes
a device status for range errors, and copies the compact output to the caller's
host layout. Signed export supports the full `int64_t` range, including
`INT64_MIN` when the bounded contract supplies magnitude `2^63`. CPU
Boost.Multiprecision CRT/Garner remains the reference and debug path. The direct
HIP kernels are intentionally inspectable and unoptimized; they are correctness
bring-up kernels, not performance evidence.

Exact-wide signed and unsigned semantics are supported as persistent RNS output
with `RNS8_BOUND_NONE`. They are not exported through the bounded i64/u64 APIs.
CPU export uses explicit little-endian limbs: signed output is fixed-width
two's-complement and unsigned output is fixed-width magnitude. HIP-resident RNS
matrices use host reconstruction for exact-wide limb export; GPU exact-wide
export remains unimplemented.

Strict wraparound `RNS8_WRAP_U64_MOD_2_64` is exposed through the explicit
`RNS8_BACKEND_WRAP64_BYTE_LIMB` CPU reference backend. It supports both
`rns8_gemm_wrap_u64_oneshot` and persistent byte-limb matrices via
`rns8_pack_u64`, `rns8_gemm_wrap_u64`, and `rns8_export_wrap_u64`. The backend
uses the byte-limb Comba reference, returns low-64-bit `uint64_t` output, does
not allocate RNS residue matrices, does not use CRT reconstruction, and rejects
bounds or prefixes in the descriptor.

The private `wrap64_hip_gemm_byte_limbs` path compiles a direct HIP kernel and
compares GPU byte-limb output against the CPU reference in the differential
suite. It is intentionally not wired into public context creation or benchmark
backend selection. It is a correctness smoke for byte-limb GPU arithmetic, not
the production 36 byte-GEMM accelerator path.
