// Disposable detector sensitivity controls, never ordinary application tests.
#include "tool_common/build_identity.hpp"
#include "../support/noninteractive_errors.hpp"
#include "../support/thread_start_gate.hpp"
#include "../support/host_runtime_inventory.hpp"
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <thread>
#if defined(_OPENMP)
#include <omp.h>
#endif
#ifndef __has_feature
#define __has_feature(value) 0
#endif
#ifndef RNS8_HOST_CONTROL_LANE
#error "Detector controls require an explicit configured instrumentation lane"
#endif

#if defined(__linux__) && __has_feature(thread_sanitizer)
extern "C" void __tsan_init();
#elif defined(__linux__) && __has_feature(address_sanitizer)
extern "C" void __asan_init();
#endif

namespace {
constexpr bool address = __has_feature(address_sanitizer);
constexpr bool thread = __has_feature(thread_sanitizer);
int heap_overread() {
  auto values = std::make_unique<int[]>(8);
  volatile unsigned index = 8;           // Preserve a real instrumented runtime load.
  volatile int observed = values[index]; // rns8-detector-site:heap-overread
  std::cout << observed << '\n';
  return 0; // A zero exit is detector failure, not a successful negative control.
}
int signed_overflow() {
  volatile int left = std::numeric_limits<int>::max(), right = 1;
  volatile int observed = left + right; // rns8-detector-site:signed-overflow
  std::cout << observed << '\n';
  return 0;
}
int threads(bool synchronized) {
  rns8::test::ThreadStartGate gate(2);
  std::mutex mutex;
  volatile uint64_t value = 0;
  const auto run = [&] {
    gate.wait();
    for (unsigned i = 0; i < 64; ++i) {
      if (synchronized) {
        std::lock_guard<std::mutex> lock(mutex);
        ++value;
      } else
        ++value; // rns8-detector-site:race
    }
  };
  std::thread first(run), second(run);
  gate.release();
  first.join();
  second.join();
  std::cout << "{\"schema\":\"rns8_host_detector_valid_v1\",\"value\":" << value
            << ",\"expected\":128}\n";
  return synchronized && value != 128 ? 1 : 0;
}
int openmp(bool synchronized) {
#if defined(_OPENMP)
  volatile uint64_t value = 0;
  int workers = 0;
#pragma omp parallel num_threads(2) shared(value, workers)
  {
#pragma omp single
    workers = omp_get_num_threads();
#pragma omp barrier
    for (unsigned i = 0; i < 64; ++i) {
      if (synchronized) {
#pragma omp critical(rns8_detector_control)
        ++value;
      } else
        ++value; // rns8-detector-site:openmp-race
    }
  }
  if (workers != 2)
    return 77;
  std::cout << "{\"schema\":\"rns8_host_detector_valid_v1\",\"value\":" << value
            << ",\"expected\":128}\n";
  return synchronized && value != 128 ? 1 : 0;
#else
  (void)synchronized;
  return 77;
#endif
}

int runtime_inventory() {
  int workers = 0;
#if defined(_OPENMP)
  // Force lazy OpenMP/OMPT initialization before observing loaded images.
#pragma omp parallel num_threads(2) shared(workers)
  {
#pragma omp single
    workers = omp_get_num_threads();
  }
#endif
  auto document = rns8::test::host_runtime_inventory();
  document["openmp_workers"] = workers;
  document["symbol_owners"] = nlohmann::json::object();
#if defined(__linux__) && __has_feature(thread_sanitizer)
  document["symbol_owners"]["sanitizer"] =
      rns8::test::host_symbol_owner(reinterpret_cast<const void*>(&__tsan_init));
#elif defined(__linux__) && __has_feature(address_sanitizer)
  document["symbol_owners"]["sanitizer"] =
      rns8::test::host_symbol_owner(reinterpret_cast<const void*>(&__asan_init));
#endif
#if defined(__linux__) && defined(_OPENMP)
  document["symbol_owners"]["openmp"] =
      rns8::test::host_symbol_owner(reinterpret_cast<const void*>(&omp_get_num_threads));
#endif
  std::cout << document.dump() << '\n';
  return std::cout ? 0 : 2;
}
} // namespace

int main(int argc, char** argv) {
  const int identity = rns8::tools::emit_build_identity(argc, argv, std::cout);
  if (identity >= 0)
    return identity;
  rns8::test::configure_noninteractive_errors();
  if (argc != 2)
    return 2;
  if (!std::strcmp(argv[1], "--runtime-inventory")) {
    try {
      return runtime_inventory();
    } catch (const std::exception& error) {
      std::cerr << "host runtime inventory: " << error.what() << '\n';
      return 2;
    }
  }
  if (!std::strcmp(argv[1], "--capabilities")) {
    std::cout << "{\"schema\":\"rns8_host_detector_capabilities_v1\",\"lane\":\""
              << RNS8_HOST_CONTROL_LANE << "\",\"address_feature\":" << (address ? "true" : "false")
              << ",\"thread_feature\":" << (thread ? "true" : "false")
#if defined(_OPENMP)
              << ",\"openmp_compiled\":true"
#else
              << ",\"openmp_compiled\":false"
#endif
              << ",\"scope\":\"host_only_not_hip_device\"}\n";
    return 0;
  }
  if (!std::strcmp(argv[1], "valid"))
    return threads(true);
  if (!std::strcmp(argv[1], "heap-overread"))
    return address ? heap_overread() : 77;
  if (!std::strcmp(argv[1], "signed-overflow"))
    return !std::strcmp(RNS8_HOST_CONTROL_LANE, "address-undefined") ? signed_overflow() : 77;
  if (!std::strcmp(argv[1], "race"))
    return thread ? threads(false) : 77;
  if (!std::strcmp(argv[1], "openmp-race"))
    return thread ? openmp(false) : 77;
  if (!std::strcmp(argv[1], "openmp-valid"))
    return thread ? openmp(true) : 77;
  return 2;
}
