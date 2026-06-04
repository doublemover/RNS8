#ifndef RNS8_BACKEND_WRAP64_HIP_KERNEL_CONFIG_HPP
#define RNS8_BACKEND_WRAP64_HIP_KERNEL_CONFIG_HPP

#include <cstdint>

namespace rns8::detail {

constexpr int64_t kWrap64HipU32AccumulatorMaxK = 4096;
constexpr int64_t kWrap64HipVectorizedPackExportMinDimension = 128;

inline bool wrap64_hip_uses_u32_accumulator(int64_t k) {
  return k > 0 && k <= kWrap64HipU32AccumulatorMaxK;
}

inline const char* wrap64_hip_selected_kernel_for_k(int64_t k) {
  return wrap64_hip_uses_u32_accumulator(k) ? "direct_hip_wrap64_byte_gemm36_u32acc_tiled_2d_v4"
                                            : "direct_hip_wrap64_byte_gemm36_u64acc_tiled_2d_v4";
}

inline bool wrap64_hip_uses_vectorized_pack_export(int64_t rows, int64_t cols) {
  return rows >= kWrap64HipVectorizedPackExportMinDimension && cols >= kWrap64HipVectorizedPackExportMinDimension;
}

}  // namespace rns8::detail

#endif
