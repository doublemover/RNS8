# Backend Notes

Backend status:

- CPU reference: implemented and tested.
- Direct HIP: implemented for device inspection, signed/unsigned residue
  conversion, persistent device-resident RNS matrix buffers, one-modulus
  correctness smoke, fused INT32-to-centered-residue reduction, and bounded
  i64/u64 GPU export for prefixes that fit the direct 128-bit Garner path.
  Public bounded GEMM can execute the direct HIP pack, RNS GEMM, and export
  path, with K split into blocks no larger than 65536 before centered residue
  reduction.
- hipBLASLt: not implemented.
- CK: not implemented.
- rocWMMA/AMDGPU builtins: not implemented.
- Wraparound byte-limb backend: CPU one-shot reference implemented; persistent
  byte-limb storage, HIP byte-limb kernels, and accelerator signedness
  corrections are not implemented.

Unsupported backends must return unsupported status. They must not expose stub
paths that appear to validate GPU behavior.

The future backend directories under `src/` are scaffold markers only. They
exist to keep ownership boundaries visible while preserving the rule that no
accelerator path counts until it has compiled kernels and exact CPU
differential validation.

The direct HIP pack kernels copy logical host `int64_t` and `uint64_t` inputs
to a matrix-owned device upload buffer and write centered residues into
matrix-owned device residue storage. The direct HIP RNS GEMM path consumes those
device residues directly, launches one thread per output element per modulus,
and reduces each INT32 K-block sum to a centered residue in the kernel. For K
above 65536, it launches multiple block kernels and accumulates the centered
residue on device.

Bounded direct HIP export reconstructs i64/u64 outputs on device with a compact
Garner kernel for prefixes up to 16, writes a device status for range errors,
and copies the compact output to the caller's host layout. CPU
Boost.Multiprecision CRT/Garner remains the reference and debug path. The direct
HIP kernels are intentionally inspectable and unoptimized; they are correctness
bring-up kernels, not performance evidence.

Exact-wide signed and unsigned semantics are supported as persistent RNS output
with `RNS8_BOUND_NONE`. They are not exported through the bounded i64/u64 APIs.
CPU export uses explicit little-endian limbs: signed output is fixed-width
two's-complement and unsigned output is fixed-width magnitude. HIP-resident RNS
matrices use host reconstruction for exact-wide limb export; GPU exact-wide
export remains unimplemented.

Strict wraparound `RNS8_WRAP_U64_MOD_2_64` is exposed only through
`rns8_gemm_wrap_u64_oneshot` on `RNS8_BACKEND_WRAP64_BYTE_LIMB`. That path uses
the byte-limb Comba reference and returns low-64-bit `uint64_t` output. It does
not allocate RNS matrices, does not use CRT reconstruction, and rejects bounds
or prefixes in the descriptor.
