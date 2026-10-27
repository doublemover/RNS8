#include <catch2/catch_test_macros.hpp>
#include <boost/multiprecision/cpp_int.hpp>
#include "rns8/integer_split.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <vector>

namespace {
rns8_integer_split_i64_desc descriptor(int64_t rows, int64_t cols, int64_t ld) {
  rns8_integer_split_i64_desc d{};
  d.struct_size = sizeof(d); d.abi_version = RNS8_ABI_VERSION;
  d.format = RNS8_INTEGER_SPLIT_SIGNED_RADIX128;
  d.rows = rows; d.cols = cols; d.native_ld = ld;
  return d;
}
rns8_integer_split_i64_info query(const rns8_integer_split_i64_desc& d) {
  rns8_integer_split_i64_info info{};
  info.struct_size = sizeof(info); info.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_query_integer_split_i64(&d, &info) == RNS8_SUCCESS);
  return info;
}
}

TEST_CASE("signed radix128 split reconstructs full integer boundaries without floating conversion") {
  using boost::multiprecision::cpp_int;
  std::vector<int64_t> values{0, 1, -1, INT64_MIN, INT64_MAX,
      (int64_t{1} << 53) - 1, int64_t{1} << 53, (int64_t{1} << 53) + 1,
      -((int64_t{1} << 53) + 1), 127, 128, -127, -128};
  for (int bit = 0; bit < 63; ++bit) {
    const auto value = int64_t{1} << bit;
    values.insert(values.end(), {value, -value, value - 1, -(value - 1)});
  }
  // Every signed byte plus varied deterministic full-width magnitudes/signs.
  for (int value = -128; value <= 127; ++value) values.push_back(value);
  uint64_t random = 0x25a774a6b495d2e1;
  for (int i = 0; i < 1024; ++i) {
    random ^= random << 13; random ^= random >> 7; random ^= random << 17;
    int64_t value = 0; std::memcpy(&value, &random, sizeof(value)); values.push_back(value);
  }
  auto d = descriptor(1, int64_t(values.size()), int64_t(values.size()));
  const auto info = query(d);
  REQUIRE(info.component_count == 10); CHECK(info.radix_bits == 7);
  CHECK(info.digit_min == -127); CHECK(info.digit_max == 127); CHECK(info.layout_version == 1);
  std::vector<int8_t> bytes(size_t(info.required_component_bytes), -128);
  std::vector<int64_t> result(values.size(), 42);
  REQUIRE(rns8_split_i64(&d, values.data(), bytes.data(), bytes.size()) == RNS8_SUCCESS);
  for (size_t cell = 0; cell < values.size(); ++cell) {
    cpp_int exact = 0;
    for (unsigned s = 0; s < info.component_count; ++s) {
      const int digit = bytes[s * info.component_plane_stride + cell];
      REQUIRE(digit >= -127); REQUIRE(digit <= 127);
      CHECK((values[cell] < 0 ? digit <= 0 : digit >= 0));
      exact += cpp_int(digit) * (cpp_int(1) << (7 * s));
    }
    REQUIRE(exact == values[cell]);
  }
  REQUIRE(rns8_merge_i64(&d, bytes.data(), bytes.size(), result.data()) == RNS8_SUCCESS);
  CHECK(result == values);
}

TEST_CASE("integer split respects native component row plane and guard padding") {
  auto d = descriptor(3, 4, 7); d.component_ld = 6; d.component_plane_stride = 23;
  const auto info = query(d);
  CHECK(info.native_span_bytes == 18 * sizeof(int64_t));
  CHECK(info.required_component_bytes == 9 * 23 + 2 * 6 + 4);
  std::array<int64_t, 24> input{}, output{};
  input.fill(99); output.fill(77);
  for (size_t r = 0; r < 3; ++r) for (size_t c = 0; c < 4; ++c)
    input[1 + r * 7 + c] = (r + c) % 2 ? INT64_MIN + int64_t(r * 4 + c) : INT64_MAX - int64_t(r * 4 + c);
  std::vector<int8_t> bytes(size_t(info.required_component_bytes) + 11, -128);
  REQUIRE(rns8_split_i64(&d, input.data() + 1, bytes.data() + 3, info.required_component_bytes) == RNS8_SUCCESS);
  for (size_t i = 0; i < bytes.size(); ++i) {
    bool logical = false;
    for (size_t s = 0; s < 10; ++s) for (size_t r = 0; r < 3; ++r)
      logical |= i >= 3 + s * 23 + r * 6 && i < 3 + s * 23 + r * 6 + 4;
    if (!logical) CHECK(bytes[i] == -128);
  }
  REQUIRE(rns8_merge_i64(&d, bytes.data() + 3, info.required_component_bytes, output.data() + 1) == RNS8_SUCCESS);
  for (size_t i = 0; i < output.size(); ++i) {
    const bool logical = i >= 1 && (i - 1) / 7 < 3 && (i - 1) % 7 < 4;
    CHECK(output[i] == (logical ? input[i] : 77));
  }
}

TEST_CASE("integer split rejects invalid descriptors spans alias and short capacity transactionally") {
  const auto d = descriptor(2, 3, 5); const auto info = query(d);
  std::array<int64_t, 12> native{}; native.fill(123);
  std::vector<int8_t> bytes(size_t(info.required_component_bytes), 42);
  const auto unchanged = bytes;
  CHECK(rns8_split_i64(&d, native.data(), bytes.data(), bytes.size() - 1) == RNS8_WORKSPACE_TOO_SMALL);
  CHECK(bytes == unchanged);
  CHECK(rns8_merge_i64(&d, bytes.data(), bytes.size() - 1, native.data()) == RNS8_WORKSPACE_TOO_SMALL);
  CHECK(std::all_of(native.begin(), native.end(), [](auto x) { return x == 123; }));
  CHECK(rns8_split_i64(&d, native.data(), reinterpret_cast<int8_t*>(native.data()) + 1, bytes.size()) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8_merge_i64(&d, reinterpret_cast<const int8_t*>(native.data()) + 1, bytes.size(), native.data()) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8_split_i64(&d, nullptr, bytes.data(), bytes.size()) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8_split_i64(&d, reinterpret_cast<const int64_t*>(UINTPTR_MAX - 7), bytes.data(), bytes.size()) == RNS8_RANGE_ERROR);
  CHECK(rns8_split_i64(&d, reinterpret_cast<const int64_t*>(uintptr_t{1}), bytes.data(), bytes.size()) == RNS8_INVALID_ARGUMENT);
  for (unsigned mutation = 0; mutation < 11; ++mutation) {
    auto bad = d;
    switch (mutation) {
      case 0: --bad.struct_size; break;
      case 1: ++bad.abi_version; break;
      case 2: bad.format = rns8_integer_split_format(99); break;
      case 3: bad.rows = 0; break;
      case 4: bad.cols = -1; break;
      case 5: bad.native_ld = 2; break;
      case 6: bad.component_ld = 2; break;
      case 7: bad.component_plane_stride = 5; break;
      case 8: bad.component_ld = -1; break;
      case 9: bad.component_plane_stride = -1; break;
      case 10: bad.rows = INT64_MAX; break;
    }
    auto output = info; const auto before = output;
    CHECK(rns8_query_integer_split_i64(&bad, &output) == (mutation == 10 ? RNS8_RANGE_ERROR : RNS8_INVALID_ARGUMENT));
    CHECK(std::memcmp(&output, &before, sizeof(output)) == 0);
    CHECK(rns8_split_i64(&bad, native.data(), bytes.data(), bytes.size()) != RNS8_SUCCESS);
    CHECK(bytes == unchanged);
  }
}

TEST_CASE("integer merge rejects malformed final cells before changing any native output") {
  auto d = descriptor(2, 2, 3); const auto info = query(d);
  std::array<int64_t, 6> input{INT64_MIN, INT64_MAX, 88, -129, 127, 88}, output{};
  std::vector<int8_t> original(size_t(info.required_component_bytes));
  REQUIRE(rns8_split_i64(&d, input.data(), original.data(), original.size()) == RNS8_SUCCESS);
  for (unsigned mutation = 0; mutation < 5; ++mutation) {
    CAPTURE(mutation);
    auto bytes = original; output.fill(456);
    const size_t cell = 3; const size_t stride = size_t(info.component_plane_stride);
    for (unsigned s = 0; s < 10; ++s) bytes[s * stride + cell] = 0;
    if (mutation == 0) bytes[cell] = -128;
    if (mutation == 1) { bytes[cell] = 1; bytes[stride + cell] = -1; }
    if (mutation == 2) bytes[9 * stride + cell] = 2;
    if (mutation == 3) bytes[9 * stride + cell] = 1; // +2^63
    if (mutation == 4) { bytes[9 * stride + cell] = -1; bytes[cell] = -1; }
    CHECK(rns8_merge_i64(&d, bytes.data(), bytes.size(), output.data()) ==
        (mutation < 3 ? RNS8_INVALID_ARGUMENT : RNS8_RANGE_ERROR));
    CHECK(std::all_of(output.begin(), output.end(), [](auto x) { return x == 456; }));
  }
}
