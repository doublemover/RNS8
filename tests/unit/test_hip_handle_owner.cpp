#include <array>
#include <catch2/catch_test_macros.hpp>
#include <stdexcept>

#include "backend_common/checked_arithmetic.hpp"
#include "core/hip_handle_owner.hpp"
#include "core/hip_resources.hpp"

namespace {
struct Handles {
  using handle_type = void*;
  using status_type = rns8_status;
  static constexpr auto success = RNS8_SUCCESS, out_of_memory = RNS8_INTERNAL_ERROR,
                        invalid_argument = RNS8_INVALID_ARGUMENT;
  static constexpr auto kind = rns8::detail::hip_deferred_allocation_kind::event;
  inline static int device = 7;
  inline static bool fail = false;
  inline static std::array<rns8::detail::hip_deferred_buffer_cleanup*, 8> deferred{};
  inline static unsigned count = 0, destroys = 0;
  inline static int destroyed_device = -1;
  static rns8_status current_device(int& out) noexcept {
    out = device;
    return RNS8_SUCCESS;
  }
  static rns8_status destroy(int source, void*) noexcept {
    ++destroys;
    destroyed_device = source;
    return fail ? RNS8_BACKEND_FAILURE : RNS8_SUCCESS;
  }
  static void defer(rns8::detail::hip_deferred_buffer_cleanup* record) noexcept {
    deferred[count++] = record;
  }
};
struct HandleEnvironment {
  HandleEnvironment() {
    Handles::fail = false;
    Handles::device = 7;
    Handles::count = Handles::destroys = 0;
  }
  ~HandleEnvironment() {
    for (unsigned i = 0; i < Handles::count; ++i) delete Handles::deferred[i];
    Handles::count = 0;
  }
};
using Owner = rns8::detail::basic_hip_handle<Handles>;
void* handle(uintptr_t value) { return reinterpret_cast<void*>(value); }
}  // namespace

TEST_CASE(
    "runtime handle replacement preserves failed destruction and "
    "original device") {
  HandleEnvironment environment;
  Owner first, second;
  REQUIRE(first.create([](void** out) noexcept {
    *out = handle(1);
    return RNS8_SUCCESS;
  }) == RNS8_SUCCESS);
  Handles::device = 9;
  REQUIRE(second.create([](void** out) noexcept {
    *out = handle(2);
    return RNS8_SUCCESS;
  }) == RNS8_SUCCESS);
  Handles::fail = true;
  CHECK(first.reset() == RNS8_BACKEND_FAILURE);
  CHECK(first.get() == handle(1));
  bool called = false;
  CHECK(first.create([&](void**) noexcept {
    called = true;
    return RNS8_SUCCESS;
  }) == RNS8_BACKEND_FAILURE);
  CHECK_FALSE(called);
  first = std::move(second);
  REQUIRE(Handles::count == 1);
  CHECK(Handles::deferred[0]->pointer == handle(1));
  CHECK(Handles::deferred[0]->device_id == 7);
  CHECK(first.get() == handle(2));
  CHECK_FALSE(second);
  Handles::fail = false;
  CHECK(first.reset() == RNS8_SUCCESS);
  CHECK(Handles::destroyed_device == 9);
  CHECK_FALSE(first);
}

TEST_CASE("runtime handle partial acquisition remains owned through failed cleanup") {
  HandleEnvironment environment;
  {
    Owner owner;
    REQUIRE(owner.reserve() == RNS8_SUCCESS);
    CHECK(owner.receive([](void** out) noexcept {
      *out = handle(3);
      return RNS8_BACKEND_FAILURE;
    }) == RNS8_BACKEND_FAILURE);
    REQUIRE(owner.get() == handle(3));
    Handles::fail = true;
  }
  REQUIRE(Handles::count == 1);
  CHECK(Handles::deferred[0]->pointer == handle(3));
  CHECK(Handles::deferred[0]->kind == Handles::kind);
  Handles::fail = false;
  CHECK(Handles::destroy(Handles::deferred[0]->device_id, Handles::deferred[0]->pointer) ==
        RNS8_SUCCESS);
}

TEST_CASE(
    "shared checked arithmetic preserves output on overflow and handles "
    "aliased results") {
  using namespace rns8::detail::checked;
  uint64_t result = 17;
  CHECK_FALSE(add(UINT64_MAX, 1, result));
  CHECK(result == 17);
  CHECK_FALSE(multiply(UINT64_MAX, 2, result));
  CHECK(result == 17);
  CHECK(add(result, UINT64_MAX - 17, result));
  CHECK(result == UINT64_MAX);
  CHECK(multiply(result, 0, result));
  CHECK(result == 0);
  CHECK(multiply(UINT64_MAX, 1, result));
  CHECK(result == UINT64_MAX);
}

TEST_CASE(
    "owned HIP capture ends on host exception and retains the captured "
    "operation") {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  using namespace rns8::detail;
  int count = 0;
  if (hipGetDeviceCount(&count) != hipSuccess || !count) SKIP("HIP device required");
  hip_scoped_device device(0);
  REQUIRE(device.status() == hipSuccess);
  hip_direct_device_buffer bytes;
  REQUIRE(bytes.allocate(0, 4) == RNS8_SUCCESS);
  hip_unique_stream stream;
  REQUIRE(stream.create_non_blocking() == hipSuccess);
  hip_unique_graph graph;
  REQUIRE(graph.reserve() == hipSuccess);
  try {
    hip_capture_scope capture(stream.get(), graph);
    REQUIRE(capture.begin() == hipSuccess);
    REQUIRE(hipMemsetAsync(bytes.get(), 42, 4, stream.get()) == hipSuccess);
    throw std::runtime_error("host unwinding");
  } catch (const std::runtime_error&) {
  }
  hipStreamCaptureStatus state = hipStreamCaptureStatusActive;
  REQUIRE(hipStreamIsCapturing(stream.get(), &state) == hipSuccess);
  CHECK(state == hipStreamCaptureStatusNone);
  REQUIRE(graph);
  hip_unique_graph_exec executable;
  REQUIRE(executable.instantiate(graph.get()) == hipSuccess);
  REQUIRE(hipGraphLaunch(executable.get(), stream.get()) == hipSuccess);
  REQUIRE(hipStreamSynchronize(stream.get()) == hipSuccess);
  std::array<unsigned char, 4> host{};
  REQUIRE(hipMemcpy(host.data(), bytes.get(), 4, hipMemcpyDeviceToHost) == hipSuccess);
  for (auto byte : host) CHECK(byte == 42);
#else
  SKIP("HIP not compiled");
#endif
}
