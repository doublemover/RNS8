# Backend Notes

Backend status:

- CPU reference: implemented and tested.
- Direct HIP: implemented for device inspection and one-modulus correctness
  smoke. Public bounded GEMM can execute the direct HIP ring-GEMM path for
  K values at or below 65536.
- hipBLASLt: not implemented.
- CK: not implemented.
- rocWMMA/AMDGPU builtins: not implemented.
- Wraparound byte-limb backend: not implemented.

Unsupported backends must return unsupported status. They must not expose stub
paths that appear to validate GPU behavior.

The direct HIP kernel currently copies host residues to device, launches one
thread per output element, reduces the INT32 sum to a centered residue, and
copies the result back. This is intentionally inspectable and unoptimized.

