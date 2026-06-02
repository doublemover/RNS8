# rocWMMA And AMDGPU Builtin Backend

Reserved for future target-specific hot kernels using rocWMMA or AMDGPU
builtins where the active ROCm/HIP SDK supports them.

This is not a validated backend. Do not add stubs that report success without a
compiled kernel and exact CPU differential tests.

`RNS8_ENABLE_ROCWMMA` and `RNS8_ENABLE_AMDGPU_BUILTINS` must keep failing fast
until real correctness backends exist. rocWMMA component probes are evidence
only. AMDGPU builtins have no discovery-only readiness path; they need
target-specific exact kernels, CPU differentials, and ISA evidence before
enablement.
