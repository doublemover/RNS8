#pragma once

#include <cstddef>
#include <array>
#include <cstdint>
#include <cstring>

namespace rns8::detail::trace {

struct EtwProfileSample {
  uint64_t instruction_pointer = 0;
  uint32_t thread_id = 0, packed_count_flags_rank = 0;
};

struct EtwContextSwitch {
  uint32_t new_tid = 0, old_tid = 0;
  std::array<uint8_t, 8> control{};
  uint32_t new_wait_time = 0, reserved = 0;
};

struct EtwReadyThread {
  uint32_t tid = 0;
  std::array<uint8_t, 4> control{};
};

struct EtwHeapEvent {
  std::array<uint64_t, 5> words{};
  uint32_t word_count = 0, tail = 0;
};

struct EtwStackWalk {
  uint64_t event_timestamp = 0;
  uint32_t pid = 0, tid = 0, frame_count = 0;
  std::array<uint64_t, 192> frames{};
};

inline bool decode_etw_stack_walk(const void* bytes, size_t size, uint32_t version,
                                  uint32_t pointer_bytes, EtwStackWalk& output) noexcept {
  if (!bytes || version > 2 || (pointer_bytes != 4 && pointer_bytes != 8) || size < 16 ||
      (size - 16) % pointer_bytes || (size - 16) / pointer_bytes > 192)
    return false;
  EtwStackWalk value;
  const auto* raw = static_cast<const unsigned char*>(bytes);
  std::memcpy(&value.event_timestamp, raw, 8);
  std::memcpy(&value.pid, raw + 8, 4);
  std::memcpy(&value.tid, raw + 12, 4);
  value.frame_count = static_cast<uint32_t>((size - 16) / pointer_bytes);
  for (uint32_t i = 0; i < value.frame_count; ++i)
    std::memcpy(&value.frames[i], raw + 16 + i * pointer_bytes, pointer_bytes);
  output = value;
  return true;
}

// Pinned Microsoft HeapTrace v0 layouts. Other opcodes/versions are not
// allocation records merely because their payload has a similar byte length.
inline bool decode_etw_heap_event(const void* bytes, size_t size, uint32_t version, uint32_t opcode,
                                  uint32_t pointer_bytes, EtwHeapEvent& output) noexcept {
  if (!bytes || version != 0 || (pointer_bytes != 4 && pointer_bytes != 8))
    return false;
  const uint32_t count = opcode == 32 || opcode == 35 ? 1u
                         : opcode == 33               ? 3u
                         : opcode == 34               ? 5u
                         : opcode == 36               ? 2u
                                                      : 0u;
  if (!count || size != count * pointer_bytes + (opcode == 35 ? 0u : 4u))
    return false;
  EtwHeapEvent value;
  const auto* raw = static_cast<const unsigned char*>(bytes);
  value.word_count = count;
  for (uint32_t i = 0; i < count; ++i)
    std::memcpy(&value.words[i], raw + i * pointer_bytes, pointer_bytes);
  if (opcode != 35)
    std::memcpy(&value.tail, raw + count * pointer_bytes, 4);
  output = value;
  return true;
}

inline bool decode_etw_context_switch(const void* bytes, size_t size, uint32_t version,
                                      EtwContextSwitch& output) noexcept {
  if (!bytes || size != 24 || version != 2)
    return false;
  const auto* raw = static_cast<const unsigned char*>(bytes);
  EtwContextSwitch value;
  std::memcpy(&value.new_tid, raw, 4);
  std::memcpy(&value.old_tid, raw + 4, 4);
  std::memcpy(value.control.data(), raw + 8, 8);
  std::memcpy(&value.new_wait_time, raw + 16, 4);
  std::memcpy(&value.reserved, raw + 20, 4);
  output = value;
  return true;
}

inline bool decode_etw_ready_thread(const void* bytes, size_t size, uint32_t version,
                                    EtwReadyThread& output) noexcept {
  if (!bytes || size != 8 || version != 2)
    return false;
  const auto* raw = static_cast<const unsigned char*>(bytes);
  EtwReadyThread value;
  std::memcpy(&value.tid, raw, 4);
  std::memcpy(value.control.data(), raw + 4, 4);
  output = value;
  return true;
}

// Native little-endian Windows SampledProfile v2. Never reinterpret a packed
// or unaligned payload as a C++ struct, and preserve the caller on rejection.
inline bool decode_etw_profile_sample(const void* bytes, size_t size, uint32_t version,
                                      uint32_t pointer_bytes, EtwProfileSample& output) noexcept {
  if (!bytes || version != 2 || (pointer_bytes != 4 && pointer_bytes != 8) ||
      size != pointer_bytes + 8)
    return false;
  EtwProfileSample value;
  const auto* raw = static_cast<const unsigned char*>(bytes);
  std::memcpy(&value.instruction_pointer, raw, pointer_bytes);
  std::memcpy(&value.thread_id, raw + pointer_bytes, 4);
  std::memcpy(&value.packed_count_flags_rank, raw + pointer_bytes + 4, 4);
  output = value;
  return true;
}

} // namespace rns8::detail::trace
