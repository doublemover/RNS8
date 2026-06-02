# Backend Notes

Backend status:

- CPU reference: implemented and tested.
- Direct HIP: implemented for device inspection, signed/unsigned residue
  conversion, and one-modulus correctness smoke. Public bounded GEMM can
  execute the direct HIP pack and ring-GEMM path, with K split into blocks no
  larger than 65536 before centered residue reduction.
- hipBLASLt: not implemented.
- CK: not implemented.
- rocWMMA/AMDGPU builtins: not implemented.
- Wraparound byte-limb backend: not implemented.

Unsupported backends must return unsupported status. They must not expose stub
paths that appear to validate GPU behavior.

The direct HIP pack kernels convert host `int64_t` and `uint64_t` matrices into
centered residues and copy the current host-side residue storage back. The
direct HIP ring-GEMM kernel then copies host residues to device, launches one
thread per output element, reduces the INT32 sum to a centered residue, and
copies the result back. For K above 65536, it launches multiple block kernels
and accumulates the centered residue on device. This is intentionally
inspectable and unoptimized.
