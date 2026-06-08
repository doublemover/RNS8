#include "core/internal.hpp"

#include <atomic>
#include <cstdint>
#include <limits>

#if defined(RNS8_CPU_PARALLEL_OPENMP) && RNS8_CPU_PARALLEL_OPENMP
#  include <omp.h>
#endif

namespace rns8::detail {

namespace {

constexpr uint64_t kDefaultCpuParallelThresholdOps = UINT64_C(1) << 20;

std::atomic<uint32_t> g_requested_threads{0};
std::atomic<uint64_t> g_threshold_ops{kDefaultCpuParallelThresholdOps};
std::atomic<bool> g_progress{false};

uint32_t clamped_positive_thread_count(int value) {
  return value > 0 ? static_cast<uint32_t>(value) : 0;
}

}  // namespace

void configure_cpu_parallel(uint32_t requested_threads, uint64_t threshold_ops, bool progress) {
  g_requested_threads.store(requested_threads, std::memory_order_relaxed);
  g_threshold_ops.store(threshold_ops, std::memory_order_relaxed);
  g_progress.store(progress, std::memory_order_relaxed);
#if defined(RNS8_CPU_PARALLEL_OPENMP) && RNS8_CPU_PARALLEL_OPENMP
  if (requested_threads > 0 && requested_threads <= static_cast<uint32_t>(std::numeric_limits<int>::max())) {
    omp_set_num_threads(static_cast<int>(requested_threads));
  }
#else
  (void)requested_threads;
#endif
}

bool cpu_parallel_runtime_available() {
#if defined(RNS8_CPU_PARALLEL_OPENMP) && RNS8_CPU_PARALLEL_OPENMP
  return true;
#else
  return false;
#endif
}

const char* cpu_parallel_runtime_name() {
#if defined(RNS8_CPU_PARALLEL_OPENMP) && RNS8_CPU_PARALLEL_OPENMP
  return "openmp";
#else
  return "serial";
#endif
}

uint32_t cpu_parallel_requested_threads() {
  return g_requested_threads.load(std::memory_order_relaxed);
}

uint32_t cpu_parallel_max_threads() {
#if defined(RNS8_CPU_PARALLEL_OPENMP) && RNS8_CPU_PARALLEL_OPENMP
  return clamped_positive_thread_count(omp_get_max_threads());
#else
  return 1;
#endif
}

uint32_t cpu_parallel_effective_threads() {
  const uint32_t requested = cpu_parallel_requested_threads();
  const uint32_t max_threads = cpu_parallel_max_threads();
  if (requested == 0) {
    return max_threads;
  }
  return requested < max_threads ? requested : max_threads;
}

bool cpu_parallel_enabled() {
  return cpu_parallel_runtime_available() && cpu_parallel_effective_threads() > 1;
}

uint64_t cpu_parallel_threshold_ops() {
  return g_threshold_ops.load(std::memory_order_relaxed);
}

bool cpu_parallel_progress_enabled() {
  return g_progress.load(std::memory_order_relaxed);
}

uint64_t cpu_parallel_saturating_mul(uint64_t lhs, uint64_t rhs) {
  if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
    return std::numeric_limits<uint64_t>::max();
  }
  return lhs * rhs;
}

uint64_t cpu_parallel_saturating_mul3(uint64_t lhs, uint64_t rhs, uint64_t rhs2) {
  return cpu_parallel_saturating_mul(cpu_parallel_saturating_mul(lhs, rhs), rhs2);
}

bool cpu_parallel_should_use(uint64_t work_estimate) {
  return cpu_parallel_enabled() && work_estimate >= cpu_parallel_threshold_ops();
}

}  // namespace rns8::detail
