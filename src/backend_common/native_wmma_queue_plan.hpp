#ifndef RNS8_NATIVE_WMMA_QUEUE_PLAN_HPP
#define RNS8_NATIVE_WMMA_QUEUE_PLAN_HPP
#include "native_wmma_queue.hpp"
#include "address_domain.hpp"
#include "rns8/status.h"
#include <algorithm>
#include <deque>
#include <new>
#include <vector>
namespace rns8::detail {
// Host-only descriptor builder. No ownership assumption relies on a GPU's
// execution order. Output/input overlap is rejected across the whole batch.
inline rns8_status build_native_wmma_queue(const std::vector<NativeWmmaTask>& tasks,
    uint32_t workers, uint64_t budget, std::vector<NativeWmmaWork>& output, uint64_t& bytes) {
  if (tasks.empty() || tasks.size() > 4096 || !workers || workers > 1024) return RNS8_INVALID_ARGUMENT;
  try {
    struct Span { uintptr_t begin, end; bool write; };
    std::vector<Span> spans;
    std::vector<uint32_t> counts, next(tasks.size());
    spans.reserve(tasks.size() * 3); counts.reserve(tasks.size());
    uint64_t total = 0;
    const auto span = [&](const void* pointer, uint64_t size, bool write) {
      const auto address = reinterpret_cast<uintptr_t>(pointer);
      if (!pointer || !size || size > UINTPTR_MAX || address > UINTPTR_MAX - size) return false;
      spans.push_back({address, address + static_cast<uintptr_t>(size), write});
      return true;
    };
    for (const auto& task : tasks) {
      if (!task.a || !task.b || !task.c || task.m <= 0 || task.n <= 0 || task.k <= 0 ||
          task.lda < task.k || task.ldb < task.n || task.ldc < task.n || !task.prefix || task.prefix > 20 ||
          (task.group != 1 && task.group != 2 && task.group != 4) || task.unsigned_inputs > 1)
        return RNS8_INVALID_ARGUMENT;
      if (!qualified_i32_matrix_address(task.m, task.k, task.lda, 1, 8) ||
          !qualified_i32_matrix_address(task.k, task.n, task.ldb, 1, 8) ||
          !qualified_i32_matrix_address(task.m, task.n, task.ldc, task.prefix)) return RNS8_UNSUPPORTED_BACKEND;
      // Bound accessed bytes, excluding unused final pitch. Split-K tasks can
      // start within a row; charging that row offset again at the end would
      // falsely extend the input span beyond its allocation.
      if (!span(task.a, (static_cast<uint64_t>(task.m - 1) * task.lda + task.k) * 8, false) ||
          !span(task.b, (static_cast<uint64_t>(task.k - 1) * task.ldb + task.n) * 8, false) ||
          !span(task.c, (static_cast<uint64_t>(task.m) * task.prefix - 1) * task.ldc + task.n, true)) return RNS8_INVALID_ARGUMENT;
      const uint64_t count = ((task.m + 15) / 16) * ((task.n + 15) / 16) *
                             ((task.prefix + task.group - 1) / task.group);
      if (count > UINT32_MAX - workers || total > UINT32_MAX - workers - count) return RNS8_RANGE_ERROR;
      counts.push_back(static_cast<uint32_t>(count)); total += count;
    }
    std::sort(spans.begin(), spans.end(), [](const Span& a, const Span& b) { return a.begin < b.begin; });
    uintptr_t read_end = 0, write_end = 0;
    for (const auto& item : spans) {
      if (item.begin < write_end || (item.write && item.begin < read_end)) return RNS8_INVALID_ARGUMENT;
      if (item.write) write_end = std::max(write_end, item.end);
      else read_end = std::max(read_end, item.end);
    }
    const uint64_t required = tasks.size() * sizeof(NativeWmmaTask) + total * sizeof(NativeWmmaWork) + sizeof(uint32_t);
    if (required > budget || required > SIZE_MAX) return RNS8_WORKSPACE_TOO_SMALL;
    std::vector<NativeWmmaWork> work;
    work.reserve(static_cast<std::size_t>(total));
    std::deque<uint32_t> active;
    for (uint32_t task = 0; task < tasks.size(); ++task) active.push_back(task);
    while (!active.empty()) {
      const auto id = active.front(); active.pop_front();
      const auto& task = tasks[id];
      const uint32_t ordinal = next[id]++;
      const uint32_t groups = (task.prefix + task.group - 1) / task.group;
      const uint32_t tile = ordinal / groups, cols = static_cast<uint32_t>((task.n + 15) / 16);
      work.push_back({id, tile / cols, tile % cols, (ordinal % groups) * task.group});
      if (next[id] < counts[id]) active.push_back(id);
    }
    output.swap(work); bytes = required;
    return RNS8_SUCCESS;
  } catch (const std::bad_alloc&) { return RNS8_INTERNAL_ERROR; }
}
}
#endif
