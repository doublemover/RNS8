#ifndef RNS8_NATIVE_WMMA_QUEUE_PLAN_HPP
#define RNS8_NATIVE_WMMA_QUEUE_PLAN_HPP
#include "native_wmma_queue.hpp"
#include "native_wmma_schedule.hpp"
#include "address_domain.hpp"
#include "encoded_b_layout.hpp"
#include "rns8/status.h"
#include <algorithm>
#include <new>
#include <vector>
namespace rns8::detail {
// Zero selects the heterogeneous kernel. The caller still performs full queue
// admission before using this launch hint; shapes, prefixes and pitches may vary.
inline NativeWmmaUniformHint native_wmma_uniform_hint(
    const std::vector<NativeWmmaTask>& tasks,
    NativeWmmaDispatchPolicy policy = NativeWmmaDispatchPolicy::native_uniform_only) {
  if (tasks.empty() || policy == NativeWmmaDispatchPolicy::generic_task_dispatch ||
      uint32_t(policy) > 2)
    return {};
  const auto& first = tasks.front();
  if ((first.group != 1 && first.group != 2 && first.group != 4) || first.unsigned_inputs > 1 ||
      (first.encoded_b && policy == NativeWmmaDispatchPolicy::native_uniform_only))
    return {};
  return std::all_of(tasks.begin(), tasks.end(),
                     [&](const NativeWmmaTask& task) {
                       return task.group == first.group &&
                              task.unsigned_inputs == first.unsigned_inputs &&
                              bool(task.encoded_b) == bool(first.encoded_b);
                     })
             ? NativeWmmaUniformHint{first.group, first.unsigned_inputs,
                                     uint32_t(bool(first.encoded_b))}
             : NativeWmmaUniformHint{};
}
// Host-only descriptor builder. No ownership assumption relies on a GPU's
// execution order. Output/input overlap is rejected across the whole batch.
inline rns8_status native_wmma_queue_counts(const std::vector<NativeWmmaTask>& tasks,
                                            uint32_t workers, uint32_t chunk,
                                            std::vector<uint32_t>& output_counts,
                                            uint32_t& output_total,
                                            uint64_t* peak_payload = nullptr) {
  if (tasks.empty() || tasks.size() > 4096 || !workers || workers > 1024 || !chunk || chunk > 32)
    return RNS8_INVALID_ARGUMENT;
  try {
    struct Span {
      uintptr_t begin, end;
      bool write;
    };
    std::vector<Span> spans;
    std::vector<uint32_t> counts;
    spans.reserve(tasks.size() * 4);
    counts.reserve(tasks.size());
    uint64_t total = 0;
    const auto span = [&](const void* pointer, uint64_t size, bool write) {
      const auto address = reinterpret_cast<uintptr_t>(pointer);
      if (!pointer || !size || size > UINTPTR_MAX || address > UINTPTR_MAX - size)
        return false;
      spans.push_back({address, address + static_cast<uintptr_t>(size), write});
      return true;
    };
    for (const auto& task : tasks) {
      if (!task.a || !task.b || !task.c || task.m <= 0 || task.n <= 0 || task.k <= 0 ||
          task.lda < task.k || task.ldb < task.n || task.ldc < task.n || !task.prefix ||
          task.prefix > 20 || (task.group != 1 && task.group != 2 && task.group != 4) ||
          task.unsigned_inputs > 1)
        return RNS8_INVALID_ARGUMENT;
      if (!qualified_i32_matrix_address(task.m, task.k, task.lda, 1, 8) ||
          !qualified_i32_matrix_address(task.k, task.n, task.ldb, 1, 8) ||
          !qualified_i32_matrix_address(task.m, task.n, task.ldc, task.prefix))
        return RNS8_UNSUPPORTED_BACKEND;
      // Bound accessed bytes, excluding unused final pitch. Split-K tasks can
      // start within a row; charging that row offset again at the end would
      // falsely extend the input span beyond its allocation.
      if (!span(task.a, (static_cast<uint64_t>(task.m - 1) * task.lda + task.k) * 8, false) ||
          !span(task.b, (static_cast<uint64_t>(task.k - 1) * task.ldb + task.n) * 8, false) ||
          !span(task.c, (static_cast<uint64_t>(task.m) * task.prefix - 1) * task.ldc + task.n,
                true))
        return RNS8_INVALID_ARGUMENT;
      if (task.encoded_b) {
        const auto encoded = encoded_b_layout(task.k, task.n, task.ldb, task.prefix, UINT64_MAX);
        if (encoded.status != RNS8_SUCCESS)
          return encoded.status;
        if (reinterpret_cast<uintptr_t>(task.encoded_b) % 4 ||
            task.encoded_k_words != encoded.k_words ||
            !span(task.encoded_b, encoded.total_bytes, false))
          return RNS8_INVALID_ARGUMENT;
      } else if (task.encoded_k_words)
        return RNS8_INVALID_ARGUMENT;
      const uint64_t count = ((task.m + 15) / 16) * ((task.n + 15) / 16) *
                             ((task.prefix + task.group - 1) / task.group);
      if (count > UINT32_MAX - uint64_t(workers) * chunk ||
          total > UINT32_MAX - uint64_t(workers) * chunk - count)
        return RNS8_RANGE_ERROR;
      counts.push_back(static_cast<uint32_t>(count));
      total += count;
    }
    std::sort(spans.begin(), spans.end(),
              [](const Span& a, const Span& b) { return a.begin < b.begin; });
    uintptr_t read_end = 0, write_end = 0;
    for (const auto& item : spans) {
      if (item.begin < write_end || (item.write && item.begin < read_end))
        return RNS8_INVALID_ARGUMENT;
      if (item.write)
        write_end = std::max(write_end, item.end);
      else
        read_end = std::max(read_end, item.end);
    }
    if (peak_payload)
      *peak_payload = spans.capacity() * sizeof(Span) + counts.capacity() * sizeof(uint32_t);
    output_counts.swap(counts);
    output_total = static_cast<uint32_t>(total);
    return RNS8_SUCCESS;
  } catch (const std::bad_alloc&) {
    return RNS8_INTERNAL_ERROR;
  }
}
inline rns8_status build_native_wmma_queue(const std::vector<NativeWmmaTask>& tasks,
                                           uint32_t workers, uint64_t budget,
                                           std::vector<NativeWmmaWork>& output, uint64_t& bytes,
                                           uint64_t* peak_payload = nullptr) {
  std::vector<uint32_t> counts;
  uint32_t total = 0;
  uint64_t admission_peak = 0;
  auto status = native_wmma_queue_counts(tasks, workers, 1, counts, total, &admission_peak);
  if (status != RNS8_SUCCESS)
    return status;
  try {
    std::vector<uint32_t> next(tasks.size());
    const uint64_t required =
        tasks.size() * sizeof(NativeWmmaTask) + total * sizeof(NativeWmmaWork) + sizeof(uint32_t);
    if (required > budget || required > SIZE_MAX)
      return RNS8_WORKSPACE_TOO_SMALL;
    std::vector<NativeWmmaWork> work;
    work.reserve(static_cast<std::size_t>(total));
    // Circular index links retain round-robin order without an opaque deque
    // allocator; every retained/planning payload capacity is now accountable.
    std::vector<uint32_t> links(tasks.size());
    for (uint32_t i = 0; i < tasks.size(); ++i)
      links[i] = (i + 1) % tasks.size();
    uint32_t head = 0, tail = uint32_t(tasks.size() - 1);
    while (true) {
      const auto id = head;
      const auto& task = tasks[id];
      const uint32_t ordinal = next[id]++;
      const uint32_t groups = (task.prefix + task.group - 1) / task.group;
      const uint32_t tile = ordinal / groups, cols = static_cast<uint32_t>((task.n + 15) / 16);
      work.push_back({id, tile / cols, tile % cols, (ordinal % groups) * task.group});
      if (next[id] < counts[id]) {
        tail = id;
        head = links[id];
      } else {
        if (head == tail)
          break;
        links[tail] = links[id];
        head = links[id];
      }
    }
    if (peak_payload)
      *peak_payload =
          std::max(admission_peak,
                   (counts.capacity() + next.capacity() + links.capacity()) * sizeof(uint32_t) +
                       work.capacity() * sizeof(NativeWmmaWork));
    output.swap(work);
    bytes = required;
    return RNS8_SUCCESS;
  } catch (const std::bad_alloc&) {
    return RNS8_INTERNAL_ERROR;
  }
}
struct NativeWmmaSchedulePlan {
  NativeWmmaScheduleHeader header{};
  std::vector<NativeWmmaWork> explicit_work;
  std::vector<NativeWmmaTaskRange> ranges;
  std::vector<NativeWmmaRoundPhase> phases;
  std::vector<uint32_t> rank_directory;
  uint64_t device_bytes = 0, peak_host_payload_bytes = 0;
  uint64_t transient_host_capacity_bytes() const {
    return explicit_work.capacity() * sizeof(NativeWmmaWork) +
           ranges.capacity() * sizeof(NativeWmmaTaskRange) +
           phases.capacity() * sizeof(NativeWmmaRoundPhase) +
           rank_directory.capacity() * sizeof(uint32_t);
  }
};

inline rns8_status build_native_wmma_schedule(const std::vector<NativeWmmaTask>& tasks,
                                              uint32_t workers, uint64_t budget,
                                              NativeWmmaScheduleConfig config,
                                              NativeWmmaSchedulePlan& output) {
  if (config.abi_version != 1 || uint32_t(config.mode) > 3 || !config.claim_chunk ||
      config.claim_chunk > 32 || uint32_t(config.dispatch_policy) > 2)
    return RNS8_INVALID_ARGUMENT;
  try {
    std::vector<uint32_t> counts;
    uint32_t total = 0;
    uint64_t admission_peak = 0;
    auto status = native_wmma_queue_counts(tasks, workers, config.claim_chunk, counts, total,
                                           &admission_peak);
    if (status != RNS8_SUCCESS)
      return status;
    NativeWmmaSchedulePlan plan;
    plan.peak_host_payload_bytes = admission_peak;
    plan.header = {
        1, uint32_t(config.mode), uint32_t(tasks.size()), total, 0, config.claim_chunk, 0, 0};
    if (config.mode == NativeWmmaScheduleMode::ranked_round_robin) {
      plan.header.abi_version = 2;
      plan.header.reserved0 = (uint32_t(tasks.size()) + 31) / 32;
      plan.header.reserved1 = 2 * plan.header.reserved0 + 1;
    }
    if (config.mode == NativeWmmaScheduleMode::explicit_round_robin) {
      uint64_t explicit_peak = 0;
      status = build_native_wmma_queue(tasks, workers, budget, plan.explicit_work,
                                       plan.device_bytes, &explicit_peak);
      plan.peak_host_payload_bytes = std::max(plan.peak_host_payload_bytes,
                                              explicit_peak + counts.capacity() * sizeof(uint32_t));
      if (status != RNS8_SUCCESS)
        return status;
    } else {
      plan.ranges.reserve(tasks.size());
      uint32_t offset = 0;
      for (uint32_t i = 0; i < tasks.size(); ++i) {
        plan.ranges.push_back({offset, offset + counts[i], uint32_t((tasks[i].n + 15) / 16),
                               (tasks[i].prefix + tasks[i].group - 1) / tasks[i].group});
        offset += counts[i];
      }
      if (config.mode == NativeWmmaScheduleMode::compact_round_robin ||
          config.mode == NativeWmmaScheduleMode::ranked_round_robin) {
        auto sorted = counts;
        std::sort(sorted.begin(), sorted.end());
        uint32_t round = 0, first = 0, active = uint32_t(tasks.size());
        plan.phases.reserve(tasks.size());
        for (uint32_t i = 0; i < sorted.size();) {
          const uint32_t end = sorted[i];
          plan.phases.push_back({first, round, end, active});
          first += uint32_t(uint64_t(end - round) * active); // admitted sum equals total
          round = end;
          while (i < sorted.size() && sorted[i] == end) {
            ++i;
            --active;
          }
        }
        if (first != total)
          return RNS8_INTERNAL_ERROR;
        if (config.mode == NativeWmmaScheduleMode::ranked_round_robin && plan.phases.size() > 1) {
          const uint32_t words = plan.header.reserved0, stride = plan.header.reserved1;
          const uint64_t directory_words = uint64_t(plan.phases.size() - 1) * stride;
          const uint64_t bytes = tasks.size() * sizeof(NativeWmmaTask) +
                                 plan.ranges.size() * sizeof(NativeWmmaTaskRange) +
                                 plan.phases.size() * sizeof(NativeWmmaRoundPhase) +
                                 directory_words * sizeof(uint32_t) + sizeof(uint32_t);
          if (bytes > budget || bytes > SIZE_MAX)
            return RNS8_WORKSPACE_TOO_SMALL;
          plan.rank_directory.resize(size_t(directory_words));
          for (size_t phase = 1; phase < plan.phases.size(); ++phase) {
            auto* table = plan.rank_directory.data() + (phase - 1) * stride;
            const auto round = plan.phases[phase].round_begin;
            for (uint32_t task = 0; task < tasks.size(); ++task)
              if (counts[task] > round)
                table[words + 1 + task / 32] |= 1u << (task % 32);
            for (uint32_t word = 0; word < words; ++word)
              table[word + 1] = table[word] + native_wmma_popcount32(table[words + 1 + word]);
            if (table[words] != plan.phases[phase].active_tasks)
              return RNS8_INTERNAL_ERROR;
          }
        }
        plan.peak_host_payload_bytes =
            std::max(plan.peak_host_payload_bytes,
                     plan.transient_host_capacity_bytes() +
                         (counts.capacity() + sorted.capacity()) * sizeof(uint32_t));
      }
      plan.peak_host_payload_bytes =
          std::max(plan.peak_host_payload_bytes,
                   plan.transient_host_capacity_bytes() + counts.capacity() * sizeof(uint32_t));
      plan.header.phase_count = uint32_t(plan.phases.size());
      plan.device_bytes = tasks.size() * sizeof(NativeWmmaTask) +
                          plan.ranges.size() * sizeof(NativeWmmaTaskRange) +
                          plan.phases.size() * sizeof(NativeWmmaRoundPhase) +
                          plan.rank_directory.size() * sizeof(uint32_t) + sizeof(uint32_t);
      if (plan.device_bytes > budget || plan.device_bytes > SIZE_MAX)
        return RNS8_WORKSPACE_TOO_SMALL;
    }
    output = std::move(plan);
    return RNS8_SUCCESS;
  } catch (const std::bad_alloc&) {
    return RNS8_INTERNAL_ERROR;
  }
}

// Serialization/admission boundary: reject malformed or reordered directories
// without expanding the work list. Equality to the canonical builder
// proves monotonicity, complete coverage, exact rasterization and fair phases.
inline rns8_status validate_native_wmma_schedule(const std::vector<NativeWmmaTask>& tasks,
                                                 uint32_t workers,
                                                 const NativeWmmaSchedulePlan& candidate) {
  const auto& h = candidate.header;
  if (h.mode == 0 || !native_wmma_schedule_header_valid(h))
    return RNS8_INVALID_ARGUMENT;
  NativeWmmaSchedulePlan expected;
  const auto status = build_native_wmma_schedule(
      tasks, workers, UINT64_MAX, {1, NativeWmmaScheduleMode(h.mode), h.claim_chunk}, expected);
  if (status != RNS8_SUCCESS)
    return status;
  if (h.abi_version != expected.header.abi_version || h.reserved0 != expected.header.reserved0 ||
      h.reserved1 != expected.header.reserved1 ||
      candidate.rank_directory != expected.rank_directory ||
      h.task_count != expected.header.task_count || h.work_count != expected.header.work_count ||
      h.phase_count != expected.header.phase_count ||
      candidate.ranges.size() != expected.ranges.size() ||
      candidate.phases.size() != expected.phases.size() || !candidate.explicit_work.empty() ||
      candidate.device_bytes != expected.device_bytes)
    return RNS8_INVALID_ARGUMENT;
  for (uint32_t i = 0; i < candidate.ranges.size(); ++i) {
    const auto a = candidate.ranges[i], b = expected.ranges[i];
    if (a.begin != b.begin || a.end != b.end || a.tile_cols != b.tile_cols || a.groups != b.groups)
      return RNS8_INVALID_ARGUMENT;
  }
  for (uint32_t i = 0; i < candidate.phases.size(); ++i) {
    const auto a = candidate.phases[i], b = expected.phases[i];
    if (a.first_work != b.first_work || a.round_begin != b.round_begin ||
        a.round_end != b.round_end || a.active_tasks != b.active_tasks)
      return RNS8_INVALID_ARGUMENT;
  }
  return RNS8_SUCCESS;
}

} // namespace rns8::detail
#endif
