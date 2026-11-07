#include "diagnostics/etl_sample_payload.hpp"
#include <catch2/catch_test_macros.hpp>
#include <array>
#include <cstring>

TEST_CASE("native ETW sampled-profile payload preserves widths and packed fields", "[cpu][trace]") {
  using namespace rns8::detail::trace;
  for (const uint32_t width : {4u, 8u}) {
    std::array<unsigned char, 17> storage{};
    auto* bytes = storage.data() + 1; // deliberately unaligned
    const uint64_t ip = width == 4 ? 0xf1234567u : 0xf123456789abcdefull;
    const uint32_t tid = 0xf0123456u, packed = 0xabfd1234u;
    std::memcpy(bytes, &ip, width);
    std::memcpy(bytes + width, &tid, 4);
    std::memcpy(bytes + width + 4, &packed, 4);
    EtwProfileSample value;
    REQUIRE(decode_etw_profile_sample(bytes, width + 8, 2, width, value));
    CHECK(value.instruction_pointer == ip);
    CHECK(value.thread_id == tid);
    CHECK(value.packed_count_flags_rank == packed);
  }
}

TEST_CASE("native ETW sampled-profile rejection preserves caller state without reading bytes",
          "[cpu][trace]") {
  using namespace rns8::detail::trace;
  std::array<unsigned char, 16> bytes{};
  EtwProfileSample value{123, 456, 789};
  CHECK_FALSE(decode_etw_profile_sample(nullptr, 16, 2, 8, value));
  for (const uint32_t version : {0u, 1u, 3u, UINT32_MAX})
    CHECK_FALSE(decode_etw_profile_sample(bytes.data(), bytes.size(), version, 8, value));
  for (const uint32_t width : {0u, 1u, 2u, 16u, UINT32_MAX})
    CHECK_FALSE(decode_etw_profile_sample(bytes.data(), bytes.size(), 2, width, value));
  for (const size_t size : {size_t(0), size_t(1), size_t(15), size_t(17), SIZE_MAX})
    CHECK_FALSE(decode_etw_profile_sample(bytes.data(), size, 2, 8, value));
  CHECK(value.instruction_pointer == 123);
  CHECK(value.thread_id == 456);
  CHECK(value.packed_count_flags_rank == 789);
}
