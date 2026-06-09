#ifndef RNS8_BACKEND_WRAP64_HIP_KERNEL_CONFIG_HPP
#define RNS8_BACKEND_WRAP64_HIP_KERNEL_CONFIG_HPP

#include <cstddef>
#include <cstdint>
#include <cstdlib>

namespace rns8::detail {

constexpr int64_t kWrap64HipU32AccumulatorMaxK = 4096;
constexpr int64_t kWrap64HipVectorizedPackExportMinDimension = 128;
constexpr int64_t kWrap64HipContiguous4PackExportCells = 4;
constexpr int64_t kWrap64HipColPairMinDimension = 256;

constexpr const char* kWrap64HipU32Kernel = "direct_hip_wrap64_byte_gemm36_u32acc_tiled_2d_v4";
constexpr const char* kWrap64HipU64Kernel = "direct_hip_wrap64_byte_gemm36_u64acc_tiled_2d_v4";
constexpr const char* kWrap64HipU32ColPairKernel =
    "direct_hip_wrap64_byte_gemm36_u32acc_colpair_2d_v5";

inline bool wrap64_hip_uses_u32_accumulator(int64_t k) {
  return k > 0 && k <= kWrap64HipU32AccumulatorMaxK;
}

inline bool wrap64_hip_colpair_experiment_enabled() {
#ifdef _WIN32
  char* value = nullptr;
  std::size_t length = 0;
  if (_dupenv_s(&value, &length, "RNS8_WRAP64_HIP_COLPAIR_EXPERIMENT") != 0 || value == nullptr) {
    return false;
  }
  const bool enabled = length > 0 && value[0] == '1';
  std::free(value);
  return enabled;
#else
  const char* value = std::getenv("RNS8_WRAP64_HIP_COLPAIR_EXPERIMENT");
  return value != nullptr && value[0] == '1';
#endif
}

inline bool wrap64_hip_uses_colpair_kernel(int64_t m, int64_t n, int64_t k) {
  return wrap64_hip_colpair_experiment_enabled() && wrap64_hip_uses_u32_accumulator(k) &&
         m >= kWrap64HipColPairMinDimension &&
         n >= kWrap64HipColPairMinDimension;
}

inline const char* wrap64_hip_gemm_event_label_for_shape(int64_t m, int64_t n, int64_t k) {
  return wrap64_hip_uses_colpair_kernel(m, n, k) ? "wrap64_byte_gemm36_colpair_2d_kernel"
                                                 : "wrap64_byte_gemm36_tiled_2d_kernel";
}

inline const char* wrap64_hip_selected_kernel_for_k(int64_t k) {
  return wrap64_hip_uses_u32_accumulator(k) ? kWrap64HipU32Kernel : kWrap64HipU64Kernel;
}

inline const char* wrap64_hip_selected_kernel_for_shape(int64_t m, int64_t n, int64_t k) {
  return wrap64_hip_uses_colpair_kernel(m, n, k) ? kWrap64HipU32ColPairKernel
                                                : wrap64_hip_selected_kernel_for_k(k);
}

inline bool wrap64_hip_uses_vectorized_pack_export(int64_t rows, int64_t cols) {
  return rows >= kWrap64HipVectorizedPackExportMinDimension && cols >= kWrap64HipVectorizedPackExportMinDimension;
}

inline bool wrap64_hip_uses_contiguous4_pack_export(int64_t rows, int64_t cols) {
  return rows > 0 && cols > 0 && rows <= INT64_MAX / cols &&
         rows * cols >= kWrap64HipContiguous4PackExportCells;
}

}  // namespace rns8::detail

#endif
