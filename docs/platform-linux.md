# Linux ROCm Platform Notes

Linux ROCm remains the production, profiling, multi-GPU, and Instinct
validation path. The scaffold keeps Linux ROCm presets and toolchain variables
represented, but this slice was validated on Windows.

Expected Linux configure path:

```bash
cmake --preset linux-rocm-debug
cmake --build --preset linux-debug
ctest --preset linux-debug --output-on-failure
```

Linux-specific accelerator paths are intentionally not required for core
correctness:

- hipBLASLt INT8 GEMM remains a later feature-detected backend.
- CK grouped/fused kernels remain a later feature-detected backend.
- rocWMMA and AMDGPU builtins remain target-specific hot paths.

Before claiming Linux production readiness, run direct HIP parity tests on the
target ROCm release and actual supported Radeon or Instinct hardware.

