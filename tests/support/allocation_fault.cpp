#include "allocation_fault.hpp"

#include <cstdint>
#include <cstdlib>
#include <new>

namespace {

struct alignas(std::max_align_t) Header {
  void* base;
};

thread_local bool active = false;
thread_local std::size_t fail_index = std::numeric_limits<std::size_t>::max();
thread_local rns8::test::allocation_fault::Profile profile;
thread_local std::size_t live = 0;

void* allocate(std::size_t bytes, std::size_t alignment) {
  if (active) {
    const std::size_t index = profile.calls++;
    if (bytes > std::numeric_limits<std::size_t>::max() - profile.bytes) throw std::bad_alloc();
    profile.bytes += bytes;
    if (index == fail_index) {
      profile.triggered = true;
      throw std::bad_alloc();
    }
  }
  const std::size_t size = bytes == 0 ? 1 : bytes;
  if (alignment < alignof(Header)) alignment = alignof(Header);
  const std::size_t overhead = sizeof(Header) + alignment - 1;
  if (size > std::numeric_limits<std::size_t>::max() - overhead) throw std::bad_alloc();
  void* base = std::malloc(size + overhead);
  if (!base) throw std::bad_alloc();
  const auto start = reinterpret_cast<std::uintptr_t>(base) + sizeof(Header);
  const auto address = (start + alignment - 1) & ~(std::uintptr_t(alignment) - 1);
  ::new (reinterpret_cast<void*>(address - sizeof(Header))) Header{base};
  ++live;
  return reinterpret_cast<void*>(address);
}

void release(void* pointer) noexcept {
  if (!pointer) return;
  auto* header = reinterpret_cast<Header*>(static_cast<char*>(pointer) - sizeof(Header));
  void* base = header->base;
  --live;
  std::free(base);
}

}  // namespace

namespace rns8::test::allocation_fault {

void begin(std::size_t fail_at) {
  profile = {};
  fail_index = fail_at;
  active = true;
}

Profile end() noexcept {
  active = false;
  return profile;
}

std::size_t live_allocations() noexcept { return live; }

}  // namespace rns8::test::allocation_fault

void* operator new(std::size_t size) { return allocate(size, alignof(std::max_align_t)); }
void* operator new[](std::size_t size) { return allocate(size, alignof(std::max_align_t)); }
void operator delete(void* pointer) noexcept { release(pointer); }
void operator delete[](void* pointer) noexcept { release(pointer); }
void operator delete(void* pointer, std::size_t) noexcept { release(pointer); }
void operator delete[](void* pointer, std::size_t) noexcept { release(pointer); }

void* operator new(std::size_t size, const std::nothrow_t&) noexcept {
  try {
    return allocate(size, alignof(std::max_align_t));
  } catch (...) {
    return nullptr;
  }
}
void* operator new[](std::size_t size, const std::nothrow_t&) noexcept {
  try {
    return allocate(size, alignof(std::max_align_t));
  } catch (...) {
    return nullptr;
  }
}
void operator delete(void* pointer, const std::nothrow_t&) noexcept { release(pointer); }
void operator delete[](void* pointer, const std::nothrow_t&) noexcept { release(pointer); }

void* operator new(std::size_t size, std::align_val_t alignment) {
  return allocate(size, static_cast<std::size_t>(alignment));
}
void* operator new[](std::size_t size, std::align_val_t alignment) {
  return allocate(size, static_cast<std::size_t>(alignment));
}
void operator delete(void* pointer, std::align_val_t) noexcept { release(pointer); }
void operator delete[](void* pointer, std::align_val_t) noexcept { release(pointer); }
void operator delete(void* pointer, std::size_t, std::align_val_t) noexcept { release(pointer); }
void operator delete[](void* pointer, std::size_t, std::align_val_t) noexcept { release(pointer); }

void* operator new(std::size_t size, std::align_val_t alignment, const std::nothrow_t&) noexcept {
  try {
    return allocate(size, static_cast<std::size_t>(alignment));
  } catch (...) {
    return nullptr;
  }
}
void* operator new[](std::size_t size, std::align_val_t alignment, const std::nothrow_t&) noexcept {
  try {
    return allocate(size, static_cast<std::size_t>(alignment));
  } catch (...) {
    return nullptr;
  }
}
void operator delete(void* pointer, std::align_val_t, const std::nothrow_t&) noexcept { release(pointer); }
void operator delete[](void* pointer, std::align_val_t, const std::nothrow_t&) noexcept { release(pointer); }
