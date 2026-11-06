#if defined(RNS8_ENABLE_ROCTX) && RNS8_ENABLE_ROCTX && !defined(_GNU_SOURCE)
#define _GNU_SOURCE 1
#endif
#include "diagnostics/trace_ranges.hpp"
#include <ostream>

#if defined(RNS8_ENABLE_ROCTX) && RNS8_ENABLE_ROCTX
#include <rocprofiler-sdk-roctx/roctx.h>
#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <nlohmann/json.hpp>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

namespace rns8::detail::trace {
namespace {
constexpr uint32_t capacity = 4096;
struct Stamp {
  uint64_t ns = 0;
  bool available = false;
};
Stamp monotonic_stamp() noexcept {
  timespec value{};
  if (clock_gettime(CLOCK_MONOTONIC, &value) != 0 || value.tv_sec < 0 || value.tv_nsec < 0 ||
      value.tv_nsec >= 1000000000 ||
      uint64_t(value.tv_sec) > (UINT64_MAX - uint64_t(value.tv_nsec)) / 1000000000)
    return {};
  return {uint64_t(value.tv_sec) * 1000000000 + uint64_t(value.tv_nsec), true};
}
uint64_t native_thread() noexcept {
  const auto value = syscall(SYS_gettid);
  return value > 0 ? uint64_t(value) : 0;
}
struct Record {
  uint64_t range = 0;
  Parent parent{};
  Tag tag{};
  Phase phase = Phase::public_call;
  uint64_t process = 0, start_thread = 0, stop_thread = 0;
  Stamp start_before, start_after, stop_before, stop_after;
  std::atomic<bool> ready{false};
  std::atomic<uint32_t> outcome{0}; // zero means no host-side stop observed yet
};
std::array<Record, capacity> records;
std::atomic<uint64_t> claimed{0};
std::atomic<uint64_t> operation_sequence{1};
thread_local Tag tag{};
thread_local Parent parent_range{};
struct Provider {
  decltype(&roctxRangeStartA) start = nullptr;
  decltype(&roctxRangeStop) stop = nullptr;
  const char* status = "disabled_at_runtime";
  const char* module_path = nullptr;
};
template <class Function> Function symbol(void* module, const char* name) noexcept {
  void* raw = dlsym(module, name);
  Function result{};
  static_assert(sizeof(result) == sizeof(raw), "POSIX function pointer size");
  std::memcpy(&result, &raw, sizeof(result));
  return result;
}
const Provider& provider() noexcept {
  static const Provider value = []() noexcept {
    Provider result;
    const char* requested = std::getenv("RNS8_PROFILE_RANGES");
    if (!requested || std::strcmp(requested, "1") != 0)
      return result;
    result.status = "requested_provider_unavailable";
    void* module = dlopen("librocprofiler-sdk-roctx.so", RTLD_NOW | RTLD_LOCAL);
    if (!module)
      return result;
    result.start = symbol<decltype(result.start)>(module, "roctxRangeStartA");
    result.stop = symbol<decltype(result.stop)>(module, "roctxRangeStop");
    if (!result.start || !result.stop) {
      result.start = nullptr;
      result.stop = nullptr;
      dlclose(module);
      return result;
    }
    // Keep the provider loaded for process lifetime: asynchronous stops and
    // static destruction must never call through an unloaded function pointer.
    result.status = "active";
    Dl_info identity{};
    void* address = nullptr;
    static_assert(sizeof(address) == sizeof(result.start), "POSIX function pointer size");
    std::memcpy(&address, &result.start, sizeof(address));
    if (dladdr(address, &identity) && identity.dli_fname)
      result.module_path = identity.dli_fname;
    return result;
  }();
  return value;
}
const char* label(Phase phase) noexcept {
  switch (phase) {
  case Phase::preparation:
    return "rns8.preparation";
  case Phase::source_pack:
    return "rns8.source_pack";
  case Phase::proof:
    return "rns8.proof";
  case Phase::dispatch:
    return "rns8.dispatch_enqueue";
  case Phase::reduction:
    return "rns8.reduction";
  case Phase::export_result:
    return "rns8.export";
  case Phase::completion:
    return "rns8.completion";
  case Phase::publication:
    return "rns8.publication";
  case Phase::oracle:
    return "rns8.oracle";
  case Phase::cleanup:
    return "rns8.cleanup";
  case Phase::public_call:
    return "rns8.public_call";
  }
  return "rns8.unknown";
}
} // namespace

const char* provider_status() noexcept {
  return provider().status;
}
uint64_t next_operation() noexcept {
  return provider().start ? operation_sequence.fetch_add(1, std::memory_order_relaxed) : 0;
}
Tag current_tag() noexcept {
  return tag;
}
Tag exchange_tag(Tag next) noexcept {
  return std::exchange(tag, next);
}
Parent current_parent() noexcept {
  return parent_range;
}
Parent exchange_parent(Parent next) noexcept {
  return std::exchange(parent_range, next);
}
Token begin(Phase phase, Tag metadata, Parent parent) noexcept {
  const auto& api = provider();
  if (!api.start)
    return {};
  Token result;
  const auto start_before = monotonic_stamp();
  result.range = api.start(label(phase));
  const auto start_after = monotonic_stamp();
  result.active = true; // Range ID zero is valid; availability has its own bit.
  const uint64_t index = claimed.fetch_add(1, std::memory_order_relaxed);
  if (index < capacity) {
    result.slot = static_cast<uint32_t>(index);
    auto& record = records[result.slot];
    record.range = result.range;
    record.parent = parent;
    record.tag = metadata;
    record.phase = phase;
    record.process = uint64_t(getpid());
    record.start_thread = native_thread();
    record.start_before = start_before;
    record.start_after = start_after;
    record.ready.store(true, std::memory_order_release);
  }
  return result;
}
void finish(Token& token, Outcome outcome) noexcept {
  if (!token.active)
    return;
  const auto stop_before = monotonic_stamp();
  provider().stop(token.range);
  const auto stop_after = monotonic_stamp();
  if (token.slot < capacity) {
    auto& record = records[token.slot];
    record.stop_thread = native_thread();
    record.stop_before = stop_before;
    record.stop_after = stop_after;
    record.outcome.store(static_cast<uint32_t>(outcome) + 1, std::memory_order_release);
  }
  token.active = false;
}
void write_snapshot(std::ostream& out) {
  const auto flags = out.flags();
  out << std::dec;
  const uint64_t count = claimed.load(std::memory_order_acquire);
  out << "{\"schema\":\"rns8_profiler_ranges_v1\",\"provider_status\":\"" << provider_status()
      << "\",\"capacity\":" << capacity << ",\"claimed\":" << count
      << ",\"dropped\":" << (count > capacity ? count - capacity : 0)
      << ",\"scope\":\"host_ranges_not_inferred_GPU_dependencies\",\"host_clock\":{"
         "\"schema\":\"rns8_linux_host_range_clock_v1\",\"clock_id\":"
      << int(CLOCK_MONOTONIC) << ','
      << "\"name\":\"CLOCK_MONOTONIC\",\"unit\":\"nanoseconds\","
         "\"timestamp_scope\":\"brackets_around_native_ROCTx_calls\"},\"records\":[";
  bool comma = false;
  uint32_t unpublished = 0;
  for (uint32_t i = 0; i < capacity && i < count; ++i) {
    const auto& record = records[i];
    if (!record.ready.load(std::memory_order_acquire)) {
      ++unpublished;
      continue;
    }
    if (comma)
      out << ',';
    comma = true;
    const auto stop_state = record.outcome.load(std::memory_order_acquire);
    out << "{\"range_id\":" << record.range << ",\"parent_range_id\":";
    if (record.parent.available)
      out << record.parent.range;
    else
      out << "null";
    out << ",\"operation_id\":" << record.tag.operation
        << ",\"iteration_id\":" << record.tag.iteration
        << ",\"operation_namespace\":" << static_cast<uint32_t>(record.tag.operation_namespace)
        << ",\"role\":" << static_cast<uint32_t>(record.tag.role)
        << ",\"phase\":" << static_cast<uint32_t>(record.phase) << ",\"stop_state\":" << stop_state
        << ",\"process_id\":" << record.process << ",\"start_thread_id\":" << record.start_thread
        << ",\"stop_thread_id\":";
    if (stop_state)
      out << record.stop_thread;
    else
      out << "null";
    const auto stamp = [&](const char* name, const Stamp* value) {
      out << ",\"" << name << "\":";
      if (value && value->available)
        out << value->ns;
      else
        out << "null";
    };
    stamp("start_before_ns", &record.start_before);
    stamp("start_after_ns", &record.start_after);
    // Stop fields have one writer and are read only after its release store;
    // an in-progress snapshot must not race with cross-thread completion.
    stamp("stop_before_ns", stop_state ? &record.stop_before : nullptr);
    stamp("stop_after_ns", stop_state ? &record.stop_after : nullptr);
    out << '}';
  }
  out << "],\"unpublished\":" << unpublished << ",\"loaded_provider\":"
      << (provider().module_path ? nlohmann::json(provider().module_path).dump() : "null") << '}';
  out.flags(flags);
}
} // namespace rns8::detail::trace
#else
namespace rns8::detail::trace {
const char* provider_status() noexcept {
  return "disabled_at_build";
}
void write_snapshot(std::ostream& out) {
  out << "{\"schema\":\"rns8_profiler_ranges_v1\",\"provider_status\":\"disabled_at_build\","
         "\"capacity\":0,\"claimed\":0,\"dropped\":0,\"records\":[]}";
}
} // namespace rns8::detail::trace
#endif
