// Shared native/planner source inventory. Runtime and target detection remain required.
RNS8_GEMM_SOURCE_CONTRACT(1)
RNS8_GEMM_SOURCE_SEMANTIC(RNS8_BOUNDED_I64, "bounded-i64")
RNS8_GEMM_SOURCE_SEMANTIC(RNS8_BOUNDED_U64, "bounded-u64")
RNS8_GEMM_SOURCE_SEMANTIC(RNS8_EXACT_WIDE_SIGNED, "exact-wide-signed")
RNS8_GEMM_SOURCE_SEMANTIC(RNS8_EXACT_WIDE_UNSIGNED, "exact-wide-unsigned")
RNS8_GEMM_SOURCE_SEMANTIC(RNS8_WRAP_U64_MOD_2_64, "wrap-u64")
RNS8_GEMM_SOURCE_SEMANTIC(RNS8_FINITE_RING_U8, "finite-u8-ring")
RNS8_GEMM_SOURCE_SEMANTIC(RNS8_FINITE_FIELD_U8, "finite-u8-field")
// Flags follow the seven semantic records above; modulus bounds apply only to finite storage.
RNS8_GEMM_SOURCE_BACKEND(RNS8_BACKEND_AUTO, "auto", resolve_first, false, false, false, false,
                         false, false, false, 0, 0)
RNS8_GEMM_SOURCE_BACKEND(RNS8_BACKEND_CPU_REFERENCE, "cpu-reference", host_cpu, true, true, true,
                         true, false, true, true, 2, 256)
RNS8_GEMM_SOURCE_BACKEND(RNS8_BACKEND_HIP_DIRECT, "hip-direct", hip_target, true, true, true, true,
                         true, true, true, 2, 256)
RNS8_GEMM_SOURCE_BACKEND(RNS8_BACKEND_HIPBLASLT, "hipblaslt", feature_detected_hip, true, true,
                         true, true, false, true, true, 2, 256)
RNS8_GEMM_SOURCE_BACKEND(RNS8_BACKEND_CK, "ck", feature_detected_hip, true, false, true, true,
                         false, true, true, 2, 256)
RNS8_GEMM_SOURCE_BACKEND(RNS8_BACKEND_ROCWMMA, "rocwmma", feature_detected_hip, true, true, true,
                         true, false, true, true, 2, 256)
RNS8_GEMM_SOURCE_BACKEND(RNS8_BACKEND_WRAP64_BYTE_LIMB, "wrap64-byte-limb", host_cpu, false, false,
                         false, false, true, false, false, 0, 0)
RNS8_GEMM_SOURCE_BACKEND(RNS8_BACKEND_HIP_VECTOR_ALU_INT64, "hip-vector-alu-int64", hip_target,
                         true, true, false, false, false, false, false, 0, 0)
RNS8_GEMM_SOURCE_BACKEND(RNS8_BACKEND_AMDGPU_BUILTINS, "amdgpu-builtins", feature_detected_hip,
                         true, true, true, true, false, true, true, 2, 256)
RNS8_GEMM_SOURCE_BACKEND(RNS8_BACKEND_CPU_AMX, "cpu-amx-int8", feature_detected_cpu, true, true,
                         true, true, false, true, true, 2, 256)
