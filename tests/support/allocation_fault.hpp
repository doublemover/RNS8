#ifndef RNS8_TESTS_ALLOCATION_FAULT_HPP
#define RNS8_TESTS_ALLOCATION_FAULT_HPP

#include <cstddef>
#include <limits>

namespace rns8::test::allocation_fault {

struct Profile {
  std::size_t calls = 0;
  std::size_t bytes = 0;
  bool triggered = false;
};

// The dedicated executable replaces global C++ allocation functions. Only the
// current thread and the explicit begin/end interval are counted or faulted.
void begin(std::size_t fail_at = std::numeric_limits<std::size_t>::max());
Profile end() noexcept;
std::size_t live_allocations() noexcept;

}  // namespace rns8::test::allocation_fault
#endif
