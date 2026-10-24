#pragma once
#include <cstdint>
#if defined(__HIPCC__)
#define RNS8_CONVERSION_HD __host__ __device__
#else
#define RNS8_CONVERSION_HD
#endif
namespace rns8::detail {
enum class NativeConversion : uint32_t { replicated = 0, cooperative = 1, encoded_lds = 2 };
// RDNA3 wave32 only. Each lane owns exactly one operand before exchanging
// packed words; input replication is intentional, output stores are unique.
RNS8_CONVERSION_HD constexpr bool native_lane_owns_a(uint32_t lane) { return lane < 16; }
RNS8_CONVERSION_HD constexpr uint32_t native_partner_lane(uint32_t lane) { return lane ^ 16u; }
RNS8_CONVERSION_HD constexpr uint32_t native_owned_axis(uint32_t lane) { return lane & 15u; }
// A is row/K-word, B is column/K-word (explicit transpose of native B).
// Padded odd word stride rotates bank rows; actual conflicts remain measured.
RNS8_CONVERSION_HD constexpr uint32_t native_encoded_stride(uint32_t depth) { return depth / 4 + 1; }
RNS8_CONVERSION_HD constexpr uint32_t native_encoded_word(uint32_t axis, uint32_t word, uint32_t depth) {
  return axis * native_encoded_stride(depth) + word;
}
inline constexpr const char* native_conversion_identity(NativeConversion mode) {
  return mode == NativeConversion::cooperative ? "rdna3_wave32_own_ab_half_xor16_packed_v1" :
         mode == NativeConversion::encoded_lds ? "rdna3_cta32x32_encode_once_ab_axis_kword_pad1_v1" :
         "rdna3_replicated_native_conversion_v1";
}
}
#undef RNS8_CONVERSION_HD
