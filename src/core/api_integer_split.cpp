#include "rns8/integer_split.h"

#include <cstddef>
#include <cstdint>
#include <limits>

namespace {
constexpr uint32_t digit_count = 10;
constexpr uint64_t sign_bit = uint64_t{1} << 63;

struct Layout {
  uint64_t rows, cols, native_ld, component_ld, plane_stride;
  uint64_t native_bytes, component_bytes;
};

bool multiply(uint64_t a, uint64_t b, uint64_t& result) noexcept {
  if (b && a > UINT64_MAX / b) return false;
  result = a * b;
  return true;
}
bool add(uint64_t a, uint64_t b, uint64_t& result) noexcept {
  if (a > UINT64_MAX - b) return false;
  result = a + b;
  return true;
}

rns8_status layout(const rns8_integer_split_i64_desc* d, Layout& out) noexcept {
  if (!d || d->struct_size != sizeof(*d) || d->abi_version != RNS8_ABI_VERSION ||
      d->format != RNS8_INTEGER_SPLIT_SIGNED_RADIX128 || d->rows <= 0 ||
      d->cols <= 0 || d->native_ld < d->cols || d->component_ld < 0 ||
      d->component_plane_stride < 0 || (d->component_ld && d->component_ld < d->cols))
    return RNS8_INVALID_ARGUMENT;
  out = {uint64_t(d->rows), uint64_t(d->cols), uint64_t(d->native_ld),
      uint64_t(d->component_ld ? d->component_ld : d->cols), 0, 0, 0};
  uint64_t plane = 0, native_elements = 0, component_tail = 0, planes = 0;
  if (!multiply(out.rows, out.component_ld, plane) ||
      !multiply(out.rows - 1, out.native_ld, native_elements) ||
      !add(native_elements, out.cols, native_elements) ||
      !multiply(native_elements, sizeof(int64_t), out.native_bytes))
    return RNS8_RANGE_ERROR;
  out.plane_stride = d->component_plane_stride ? uint64_t(d->component_plane_stride) : plane;
  if (out.plane_stride < plane) return RNS8_INVALID_ARGUMENT;
  if (!multiply(out.rows - 1, out.component_ld, component_tail) ||
      !add(component_tail, out.cols, component_tail) ||
      !multiply(digit_count - 1, out.plane_stride, planes) ||
      !add(planes, component_tail, out.component_bytes)) return RNS8_RANGE_ERROR;
  const auto limit = uint64_t(std::numeric_limits<std::ptrdiff_t>::max());
  if (out.native_bytes > limit || out.component_bytes > limit) return RNS8_RANGE_ERROR;
  return RNS8_SUCCESS;
}

rns8_status buffers(const Layout& d, const void* native, const void* components,
    uint64_t capacity) noexcept {
  if (!native || !components || reinterpret_cast<uintptr_t>(native) % alignof(int64_t))
    return RNS8_INVALID_ARGUMENT;
  const auto a = reinterpret_cast<uintptr_t>(native), b = reinterpret_cast<uintptr_t>(components);
  if (d.native_bytes > UINTPTR_MAX - a || d.component_bytes > UINTPTR_MAX - b)
    return RNS8_RANGE_ERROR;
  if (a < b + d.component_bytes && b < a + d.native_bytes) return RNS8_INVALID_ARGUMENT;
  if (capacity < d.component_bytes) return RNS8_WORKSPACE_TOO_SMALL;
  return RNS8_SUCCESS;
}

rns8_status decode(const int8_t* cell, uint64_t stride, int64_t& value) noexcept {
  uint64_t magnitude = 0;
  bool positive = false, negative = false;
  for (uint32_t s = 0; s < digit_count; ++s) {
    const int digit = cell[uint64_t(s) * stride];
    if (digit == -128) return RNS8_INVALID_ARGUMENT;
    positive |= digit > 0;
    negative |= digit < 0;
    const auto part = uint64_t(digit < 0 ? -digit : digit);
    if (s == digit_count - 1 && part > 1) return RNS8_INVALID_ARGUMENT;
    magnitude |= part << (7 * s);
  }
  if (positive && negative) return RNS8_INVALID_ARGUMENT;
  if (magnitude > (negative ? sign_bit : sign_bit - 1)) return RNS8_RANGE_ERROR;
  value = negative ? (magnitude == sign_bit ? INT64_MIN : -int64_t(magnitude)) : int64_t(magnitude);
  return RNS8_SUCCESS;
}
}

RNS8_API rns8_status rns8_query_integer_split_i64(const rns8_integer_split_i64_desc* d,
    rns8_integer_split_i64_info* out) {
  if (!out || out->struct_size != sizeof(*out) || out->abi_version != RNS8_ABI_VERSION)
    return RNS8_INVALID_ARGUMENT;
  Layout l{};
  const auto status = layout(d, l);
  if (status != RNS8_SUCCESS) return status;
  rns8_integer_split_i64_info result{};
  result.struct_size = sizeof(result);
  result.abi_version = RNS8_ABI_VERSION;
  result.format = RNS8_INTEGER_SPLIT_SIGNED_RADIX128;
  result.radix_bits = 7;
  result.component_count = digit_count;
  result.layout_version = 1;
  result.component_ld = int64_t(l.component_ld);
  result.component_plane_stride = int64_t(l.plane_stride);
  result.native_span_bytes = l.native_bytes;
  result.required_component_bytes = l.component_bytes;
  result.digit_min = -127;
  result.digit_max = 127;
  *out = result;
  return RNS8_SUCCESS;
}

RNS8_API rns8_status rns8_split_i64(const rns8_integer_split_i64_desc* d,
    const int64_t* native, int8_t* components, uint64_t capacity) {
  Layout l{};
  auto status = layout(d, l);
  if (status != RNS8_SUCCESS) return status;
  status = buffers(l, native, components, capacity);
  if (status != RNS8_SUCCESS) return status;
  for (uint64_t row = 0; row < l.rows; ++row) for (uint64_t col = 0; col < l.cols; ++col) {
    const auto value = native[row * l.native_ld + col];
    // Unsigned negation is defined even for the two's-complement minimum.
    uint64_t magnitude = value < 0 ? uint64_t{0} - uint64_t(value) : uint64_t(value);
    for (uint32_t s = 0; s < digit_count; ++s) {
      const auto digit = int(magnitude & 127);
      components[uint64_t(s) * l.plane_stride + row * l.component_ld + col] =
          int8_t(value < 0 ? -digit : digit);
      magnitude >>= 7;
    }
  }
  return RNS8_SUCCESS;
}

RNS8_API rns8_status rns8_merge_i64(const rns8_integer_split_i64_desc* d,
    const int8_t* components, uint64_t capacity, int64_t* native) {
  Layout l{};
  auto status = layout(d, l);
  if (status != RNS8_SUCCESS) return status;
  status = buffers(l, native, components, capacity);
  if (status != RNS8_SUCCESS) return status;
  // Validate every cell before the first write; no fallible work or allocation
  // follows admission. This also keeps a malformed final cell transactional.
  for (uint64_t row = 0; row < l.rows; ++row) for (uint64_t col = 0; col < l.cols; ++col) {
    int64_t value = 0;
    status = decode(components + row * l.component_ld + col, l.plane_stride, value);
    if (status != RNS8_SUCCESS) return status;
  }
  for (uint64_t row = 0; row < l.rows; ++row) for (uint64_t col = 0; col < l.cols; ++col)
    (void)decode(components + row * l.component_ld + col, l.plane_stride,
        native[row * l.native_ld + col]);
  return RNS8_SUCCESS;
}
