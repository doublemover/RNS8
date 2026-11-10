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

TEST_CASE("native ETW scheduler v2 payloads are unaligned bounded copies", "[cpu][trace]") {
  using namespace rns8::detail::trace;
  std::array<unsigned char, 25> bytes{};
  auto* raw = bytes.data() + 1;
  const uint32_t new_tid = 0xf1234567, old_tid = 0xe1234567, wait = 0xd1234567,
                 reserved = 0xc1234567;
  std::memcpy(raw, &new_tid, 4);
  std::memcpy(raw + 4, &old_tid, 4);
  for (size_t i = 0; i < 8; ++i)
    raw[8 + i] = static_cast<unsigned char>(128 + i);
  std::memcpy(raw + 16, &wait, 4);
  std::memcpy(raw + 20, &reserved, 4);
  EtwContextSwitch value;
  REQUIRE(decode_etw_context_switch(raw, 24, 2, value));
  CHECK(value.new_tid == new_tid);
  CHECK(value.old_tid == old_tid);
  CHECK(value.new_wait_time == wait);
  CHECK(value.reserved == reserved);
  CHECK(value.control[6] == 134);
  const auto unchanged = value;
  for (const size_t size : {size_t(0), size_t(23), size_t(25), SIZE_MAX})
    CHECK_FALSE(decode_etw_context_switch(raw, size, 2, value));
  CHECK_FALSE(decode_etw_context_switch(nullptr, 24, 2, value));
  for (const uint32_t version : {0u, 1u, 3u, UINT32_MAX})
    CHECK_FALSE(decode_etw_context_switch(raw, 24, version, value));
  CHECK(value.new_tid == unchanged.new_tid);
  CHECK(value.control == unchanged.control);
  CHECK(value.reserved == unchanged.reserved);

  EtwReadyThread ready;
  REQUIRE(decode_etw_ready_thread(raw, 8, 2, ready));
  CHECK(ready.tid == new_tid);
  CHECK(std::memcmp(ready.control.data(), &old_tid, 4) == 0);
  for (const size_t size : {size_t(0), size_t(7), size_t(9), SIZE_MAX})
    CHECK_FALSE(decode_etw_ready_thread(raw, size, 2, ready));
  CHECK_FALSE(decode_etw_ready_thread(nullptr, 8, 2, ready));
  for (const uint32_t version : {0u, 1u, 3u, UINT32_MAX})
    CHECK_FALSE(decode_etw_ready_thread(raw, 8, version, ready));
  CHECK(ready.tid == new_tid);
  CHECK(std::memcmp(ready.control.data(), &old_tid, 4) == 0);
}

TEST_CASE("native ETW heap v0 layouts preserve pointer domains and reject unsupported bytes",
          "[cpu][trace]") {
  using namespace rns8::detail::trace;
  for (const uint32_t width : {4u, 8u})
    for (const uint32_t opcode : {32u, 33u, 34u, 35u, 36u}) {
      const uint32_t count = opcode == 32 || opcode == 35 ? 1u
                             : opcode == 33               ? 3u
                             : opcode == 34               ? 5u
                                                          : 2u;
      const size_t size = count * width + (opcode == 35 ? 0u : 4u);
      std::array<unsigned char, 45> storage{};
      auto* raw = storage.data() + 1;
      for (uint32_t i = 0; i < count; ++i) {
        const uint64_t value =
            (width == 4 ? UINT64_C(0xf1234560) : UINT64_C(0xf123456789abcde0)) + i;
        std::memcpy(raw + i * width, &value, width);
      }
      const uint32_t tail = 0xfedcba98;
      if (opcode != 35)
        std::memcpy(raw + count * width, &tail, 4);
      EtwHeapEvent output;
      REQUIRE(decode_etw_heap_event(raw, size, 0, opcode, width, output));
      CHECK(output.word_count == count);
      for (uint32_t i = 0; i < count; ++i)
        CHECK(output.words[i] ==
              (width == 4 ? UINT64_C(0xf1234560) : UINT64_C(0xf123456789abcde0)) + i);
      CHECK(output.tail == (opcode == 35 ? 0u : tail));
      const auto previous = output;
      CHECK_FALSE(decode_etw_heap_event(nullptr, size, 0, opcode, width, output));
      CHECK_FALSE(decode_etw_heap_event(raw, size - 1, 0, opcode, width, output));
      CHECK_FALSE(decode_etw_heap_event(raw, size + 1, 0, opcode, width, output));
      CHECK_FALSE(decode_etw_heap_event(raw, SIZE_MAX, 0, opcode, width, output));
      for (const uint32_t version : {1u, 2u, UINT32_MAX})
        CHECK_FALSE(decode_etw_heap_event(raw, size, version, opcode, width, output));
      for (const uint32_t invalid : {0u, 31u, 37u, UINT32_MAX})
        CHECK_FALSE(decode_etw_heap_event(raw, size, 0, invalid, width, output));
      for (const uint32_t invalid : {0u, 1u, 16u, UINT32_MAX})
        CHECK_FALSE(decode_etw_heap_event(raw, size, 0, opcode, invalid, output));
      CHECK(output.words == previous.words);
      CHECK(output.word_count == previous.word_count);
      CHECK(output.tail == previous.tail);
    }
}

TEST_CASE(
    "native ETW StackWalk prefixes bound unaligned native frames and preserve rejected output",
    "[cpu][trace]") {
  using namespace rns8::detail::trace;
  for (const uint32_t width : {4u, 8u}) {
    std::array<unsigned char, 16 + 192 * 8 + 1> storage{};
    auto* raw = storage.data() + 1;
    const uint64_t stamp = UINT64_C(0xf123456789abcdef);
    const uint32_t pid = 0xf1234567, tid = 0xe1234567;
    std::memcpy(raw, &stamp, 8);
    std::memcpy(raw + 8, &pid, 4);
    std::memcpy(raw + 12, &tid, 4);
    const uint64_t ip = width == 4 ? UINT64_C(0xf1234567) : UINT64_C(0xf123456789abcdef);
    for (size_t i = 0; i < 192; ++i)
      std::memcpy(raw + 16 + i * width, &ip, width);
    for (const uint32_t version : {0u, 1u, 2u})
      for (const size_t count : {size_t(0), size_t(1), size_t(192)}) {
        EtwStackWalk output;
        REQUIRE(decode_etw_stack_walk(raw, 16 + count * width, version, width, output));
        CHECK(output.pid == pid);
        CHECK(output.tid == tid);
        CHECK(output.event_timestamp == stamp);
        CHECK(output.frame_count == count);
        if (count)
          CHECK(output.frames[count - 1] == ip);
        const auto previous = output;
        CHECK_FALSE(decode_etw_stack_walk(nullptr, 16, version, width, output));
        CHECK_FALSE(decode_etw_stack_walk(raw, 15, version, width, output));
        CHECK_FALSE(decode_etw_stack_walk(raw, 17, version, width, output));
        CHECK_FALSE(decode_etw_stack_walk(raw, 16 + 193 * width, version, width, output));
        CHECK_FALSE(decode_etw_stack_walk(raw, SIZE_MAX, version, width, output));
        CHECK_FALSE(decode_etw_stack_walk(raw, 16, 3, width, output));
        CHECK_FALSE(decode_etw_stack_walk(raw, 16, version, 16, output));
        CHECK(output.pid == previous.pid);
        CHECK(output.event_timestamp == previous.event_timestamp);
        CHECK(output.frame_count == previous.frame_count);
        CHECK(output.frames == previous.frames);
      }
  }
}
