#pragma once

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>
#include <system_error>

namespace rns8::detail::trace {

// The fixed suffix ties a native marker to its complete snapshot's array slot.
// Reserve the full uint64 width so success never depends on the current count.
inline bool format_marker_label(char* destination, std::size_t capacity, std::string_view phase,
                                uint64_t slot) noexcept {
  constexpr std::string_view suffix = "#slot=";
  constexpr std::size_t tail = suffix.size() + 20 + 1;
  if (!destination || phase.empty() || capacity < tail || phase.size() > capacity - tail)
    return false;
  std::memcpy(destination, phase.data(), phase.size());
  auto* next = destination + phase.size();
  std::memcpy(next, suffix.data(), suffix.size());
  next += suffix.size();
  const auto result = std::to_chars(next, destination + capacity - 1, slot);
  if (result.ec != std::errc{})
    return false;
  *result.ptr = '\0';
  return true;
}

} // namespace rns8::detail::trace
