// Read finalized files only. OpenTrace's native timestamp conversion is retained;
// no logger is opened by name and no recording/service configuration is changed.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <evntrace.h>
#include <evntcons.h>
#include <tdh.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <climits>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <map>
#include <new>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <nlohmann/json.hpp>
#include "../src/diagnostics/etl_sample_payload.hpp"

#ifndef RNS8_ETL_INSPECT_SOURCE_SHA256
#error "CMake must bind the ETL inspector to its compiled source"
#endif

namespace {
using json = nlohmann::json;
constexpr uint64_t max_bytes = uint64_t(512) << 20;
constexpr uint64_t max_events = 1000000;
constexpr size_t max_process_events = 16384;
constexpr GUID process_provider = {
    0x3d6fa8d0, 0xfe05, 0x11d0, {0x9d, 0xda, 0x00, 0xc0, 0x4f, 0xd7, 0xba, 0x7c}};
constexpr GUID image_provider = {
    0x2cb15d1d, 0x5fc1, 0x11d2, {0xab, 0xe1, 0x00, 0xa0, 0xc9, 0x11, 0xf5, 0x18}};
constexpr GUID marker_provider = {
    0xac42f789, 0x71b2, 0x4a0b, {0xa4, 0x5c, 0x5c, 0x66, 0xf4, 0x1a, 0xb8, 0x13}};
constexpr GUID thread_provider = {
    0x3d6fa8d1, 0xfe05, 0x11d0, {0x9d, 0xda, 0x00, 0xc0, 0x4f, 0xd7, 0xba, 0x7c}};
constexpr GUID profile_provider = {
    0xce1dbfb4, 0x137e, 0x4da6, {0x87, 0xb0, 0x3f, 0x59, 0xaa, 0x10, 0x2c, 0xbc}};
constexpr GUID heap_provider = {
    0x222962ab, 0x6180, 0x4b88, {0xa8, 0x25, 0x34, 0x6b, 0x75, 0xf2, 0xa2, 0x4a}};
constexpr GUID stack_provider = {
    0xdef2fe46, 0x7bd6, 0x4b80, {0xbd, 0x94, 0xf5, 0x7f, 0xe2, 0x0d, 0x0c, 0xe3}};
enum class EventFamily : uint32_t { process, image, thread };
enum class PointFamily { sample, schedule, heap };

struct PointSource {
  EVENT_HEADER header{};
  ETW_BUFFER_CONTEXT context{};
  std::array<BYTE, 44> payload{};
  size_t payload_size = 0;
  uint64_t record_index = 0;
  PointFamily family = PointFamily::sample;
  size_t position = 0;
};

struct Property {
  USHORT type = 0;
};
struct Schema {
  bool valid = false;
  std::vector<BYTE> bytes;
  std::map<std::wstring, Property> fields;
};
struct State {
  uint64_t events = 0;
  uint64_t undecoded_process_events = 0;
  uint64_t process_events = 0;
  uint64_t image_events = 0;
  uint64_t undecoded_image_events = 0;
  size_t image_text_bytes = 0;
  uint64_t marker_events = 0, undecoded_marker_events = 0;
  uint64_t thread_events = 0, undecoded_thread_events = 0;
  uint64_t sample_walk_events = 0, sample_events = 0, undecoded_sample_events = 0,
           excluded_sample_events = 0;
  uint64_t schedule_events = 0, undecoded_schedule_events = 0, excluded_schedule_events = 0;
  uint64_t heap_events = 0, undecoded_heap_events = 0, excluded_heap_events = 0,
           unsupported_heap_events = 0;
  uint32_t selected_pid = 0;
  bool stopped = false;
  bool sample_pass = false;
  bool raw_pass = false;
  bool bind_raw_points = false;
  uint64_t raw_walk_events = 0, stack_events = 0, undecoded_stack_events = 0,
           excluded_stack_events = 0, unsupported_stack_events = 0;
  size_t stack_frames = 0;
  const char* error = nullptr;
  json selected = json::array();
  json selected_images = json::array();
  json selected_markers = json::array();
  json selected_threads = json::array(), selected_samples = json::array();
  json selected_schedule = json::array();
  json selected_heap = json::array();
  json selected_stacks = json::array();
  std::vector<PointSource> point_sources;
  size_t raw_next = 0;
  std::map<uint32_t, std::pair<uint64_t, uint64_t>> sample_thread_windows;
  std::map<std::tuple<EventFamily, USHORT, UCHAR, UCHAR, USHORT, bool>, Schema> schemas;
};

[[noreturn]] void fail(const char* message) {
  throw std::runtime_error(message);
}

struct DecoderBudgetExceeded {
  const char* message;
};

std::wstring bounded_name(const std::vector<BYTE>& bytes, ULONG offset) {
  if (offset % sizeof(wchar_t) || offset >= bytes.size())
    fail("TDH property name offset is outside schema");
  const size_t limit = std::min<size_t>((bytes.size() - offset) / sizeof(wchar_t), 128);
  std::wstring result;
  for (size_t i = 0; i < limit; ++i) {
    wchar_t value = 0;
    std::memcpy(&value, bytes.data() + offset + i * sizeof(wchar_t), sizeof(value));
    if (!value)
      return result;
    result.push_back(value);
  }
  fail("TDH property name is unterminated or exceeds budget");
}

const Schema& schema(EVENT_RECORD* event, State& state, EventFamily family) {
  const auto& descriptor = event->EventHeader.EventDescriptor;
  const auto key =
      std::make_tuple(family, descriptor.Id, descriptor.Version, descriptor.Opcode, descriptor.Task,
                      (event->EventHeader.Flags & EVENT_HEADER_FLAG_32_BIT_HEADER) != 0);
  const auto found = state.schemas.find(key);
  if (found != state.schemas.end()) {
    if (!found->second.valid)
      fail("cached native ETL TDH schema is unsupported");
    return found->second;
  }
  if (state.schemas.size() >= 32)
    throw DecoderBudgetExceeded{"TDH schema budget exhausted"};
  Schema& value = state.schemas.emplace(key, Schema{}).first->second;
  ULONG size = 0;
  if (TdhGetEventInformation(event, 0, nullptr, nullptr, &size) != ERROR_INSUFFICIENT_BUFFER ||
      size < offsetof(TRACE_EVENT_INFO, EventPropertyInfoArray))
    fail("native ETL TDH schema unavailable");
  if (size > (1u << 20))
    throw DecoderBudgetExceeded{"native ETL TDH schema byte budget exhausted"};
  value.bytes.resize(size);
  auto* info = reinterpret_cast<TRACE_EVENT_INFO*>(value.bytes.data());
  if (TdhGetEventInformation(event, 0, nullptr, info, &size) != ERROR_SUCCESS ||
      size > value.bytes.size())
    fail("native ETL TDH schema read failed");
  if (info->TopLevelPropertyCount > info->PropertyCount || info->PropertyCount > 128 ||
      offsetof(TRACE_EVENT_INFO, EventPropertyInfoArray) +
              size_t(info->PropertyCount) * sizeof(EVENT_PROPERTY_INFO) >
          size)
    fail("native TDH property directory is incomplete");
  for (ULONG i = 0; i < info->TopLevelPropertyCount; ++i) {
    const auto& field = info->EventPropertyInfoArray[i];
    const std::wstring name = bounded_name(value.bytes, field.NameOffset);
    const bool selected =
        name == L"ProcessId" ||
        (family == EventFamily::image
             ? (name == L"ImageBase" || name == L"ImageSize" || name == L"ImageChecksum" ||
                name == L"TimeDateStamp" || name == L"FileName")
         : family == EventFamily::thread
             ? name == L"TThreadId"
             : (name == L"ParentId" || name == L"UniqueProcessKey" || name == L"ImageFileName"));
    if (!selected)
      continue;
    const auto unsupported =
        PropertyStruct | PropertyParamCount | PropertyWBEMXmlFragment | PropertyHasCustomSchema;
    if (field.Flags & unsupported || field.count != 1)
      fail("native ETL scalar property is not a supported scalar");
    if (!value.fields.emplace(name, Property{field.nonStructType.InType}).second)
      fail("native ETL scalar schema has duplicate fields");
  }
  value.valid = true;
  return value;
}

std::vector<BYTE> property(EVENT_RECORD* event, const Schema& schema, const wchar_t* name,
                           USHORT expected, ULONG maximum) {
  const auto field = schema.fields.find(name);
  if (field == schema.fields.end() || field->second.type != expected)
    fail("native ETL scalar property type is unsupported");
  PROPERTY_DATA_DESCRIPTOR descriptor{};
  descriptor.PropertyName = reinterpret_cast<ULONGLONG>(name);
  descriptor.ArrayIndex = ULONG_MAX;
  ULONG size = 0;
  if (TdhGetPropertySize(event, 0, nullptr, 1, &descriptor, &size) != ERROR_SUCCESS || !size ||
      size > maximum)
    fail("native ETL scalar property size is invalid");
  std::vector<BYTE> bytes(size);
  if (TdhGetProperty(event, 0, nullptr, 1, &descriptor, size, bytes.data()) != ERROR_SUCCESS)
    fail("native ETL scalar property read failed");
  return bytes;
}

uint32_t uint32_property(EVENT_RECORD* event, const Schema& schema, const wchar_t* name) {
  const auto bytes = property(event, schema, name, TDH_INTYPE_UINT32, 4);
  if (bytes.size() != sizeof(uint32_t))
    fail("native ETL scalar property is not uint32");
  uint32_t result = 0;
  std::memcpy(&result, bytes.data(), sizeof(result));
  return result;
}

uint64_t pointer_property(EVENT_RECORD* event, const Schema& fields, const wchar_t* name,
                          size_t width) {
  const auto bytes = property(event, fields, name, TDH_INTYPE_POINTER, 8);
  if (bytes.size() != width)
    fail("native ETL pointer field width differs from its recorded event header");
  uint64_t value = 0;
  std::memcpy(&value, bytes.data(), width);
  return value;
}

std::string unicode_image(const std::vector<BYTE>& bytes) {
  if (bytes.size() % sizeof(wchar_t))
    fail("native image path has a partial UTF16 unit");
  std::vector<wchar_t> wide(bytes.size() / sizeof(wchar_t));
  std::memcpy(wide.data(), bytes.data(), bytes.size());
  const auto end = std::find(wide.begin(), wide.end(), wchar_t(0));
  if (end == wide.end() || end == wide.begin() ||
      std::any_of(wide.begin(), end, [](wchar_t c) { return c < 32; }))
    fail("native image path is empty, unterminated or contains controls");
  const int count = static_cast<int>(end - wide.begin());
  const int required = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, wide.data(), count,
                                           nullptr, 0, nullptr, nullptr);
  if (required <= 0 || required > 16384)
    fail("native image path has invalid or oversized UTF16");
  std::string result(static_cast<size_t>(required), '\0');
  if (WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, wide.data(), count, result.data(),
                          required, nullptr, nullptr) != required)
    fail("native image path UTF8 conversion failed");
  return result;
}

json ascii_image(const std::vector<BYTE>& bytes) {
  const auto end = std::find(bytes.begin(), bytes.end(), BYTE(0));
  if (end == bytes.end())
    fail("native process image is not terminated");
  auto begin = bytes.begin();
  // TDH identifies an ANSI string, not the recorder's code page. Do not split
  // an unverified multibyte character containing a slash byte into a basename.
  if (std::any_of(begin, end, [](BYTE value) { return value < 32 || value > 126; }))
    return nullptr;
  for (auto it = begin; it != end; ++it)
    if (*it == '\\' || *it == '/')
      begin = it + 1;
  if (begin == end)
    return nullptr;
  return std::string(begin, end);
}

bool selected_thread_at(const State& state, uint32_t tid, uint64_t time) {
  const auto window = state.sample_thread_windows.find(tid);
  return window != state.sample_thread_windows.end() && window->second.first <= time &&
         time < window->second.second;
}

size_t selected_observations(const State& state) {
  return state.selected_heap.size() + state.selected_samples.size() +
         state.selected_schedule.size();
}

void remember_point(EVENT_RECORD* event, State& state, PointFamily family, size_t position) {
  if (!state.bind_raw_points)
    return;
  if (event->UserDataLength > 44 ||
      (!state.point_sources.empty() &&
       state.point_sources.back().record_index >= state.sample_walk_events))
    fail("selected native event payload/order exceeds replay contract");
  PointSource value;
  value.header = event->EventHeader;
  value.header.TimeStamp.QuadPart = 0;
  value.context = event->BufferContext;
  value.payload_size = event->UserDataLength;
  std::memcpy(value.payload.data(), event->UserData, value.payload_size);
  value.record_index = state.sample_walk_events;
  value.family = family;
  value.position = position;
  state.point_sources.push_back(std::move(value));
}

void raw_event(EVENT_RECORD* event, State& state) noexcept {
  if (++state.raw_walk_events > max_events) {
    state.stopped = true;
    state.error = "ETL raw event walk budget exhausted";
    return;
  }
  try {
    if (state.raw_next < state.point_sources.size() &&
        state.point_sources[state.raw_next].record_index == state.raw_walk_events) {
      auto header = event->EventHeader;
      header.TimeStamp.QuadPart = 0;
      auto& saved = state.point_sources[state.raw_next++];
      if (std::memcmp(&header, &saved.header, sizeof(header)) ||
          std::memcmp(&event->BufferContext, &saved.context, sizeof(saved.context)) ||
          event->UserDataLength != saved.payload_size || !event->UserData ||
          std::memcmp(event->UserData, saved.payload.data(), saved.payload_size) ||
          event->EventHeader.TimeStamp.QuadPart < 0)
        fail("raw/native converted event ordinal or header/payload identity differs");
      auto& rows = saved.family == PointFamily::sample ? state.selected_samples
                   : saved.family == PointFamily::heap ? state.selected_heap
                                                       : state.selected_schedule;
      rows.at(saved.position)["raw_time_qpc"] = event->EventHeader.TimeStamp.QuadPart;
    }
    if (!IsEqualGUID(event->EventHeader.ProviderId, stack_provider))
      return;
    ++state.stack_events;
    if (event->EventHeader.EventDescriptor.Opcode != 32) {
      ++state.unsupported_stack_events;
      return;
    }
    const bool is32 = (event->EventHeader.Flags & EVENT_HEADER_FLAG_32_BIT_HEADER) != 0;
    const bool is64 = (event->EventHeader.Flags & EVENT_HEADER_FLAG_64_BIT_HEADER) != 0;
    const uint32_t width = is32 ? 4u : 8u;
    rns8::detail::trace::EtwStackWalk payload;
    if (is32 == is64 || !rns8::detail::trace::decode_etw_stack_walk(
                            event->UserData, event->UserDataLength,
                            event->EventHeader.EventDescriptor.Version, width, payload)) {
      ++state.undecoded_stack_events;
      return;
    }
    if (payload.pid != state.selected_pid) {
      ++state.excluded_stack_events;
      return;
    }
    if (state.selected_stacks.size() >= 8192 || payload.frame_count > 262144 - state.stack_frames)
      fail("selected native stack record/frame budget exhausted");
    state.stack_frames += payload.frame_count;
    json frames = json::array();
    for (uint32_t i = 0; i < payload.frame_count; ++i)
      frames.push_back(payload.frames[i]);
    state.selected_stacks.push_back(
        json{{"record_index", state.raw_walk_events},
             {"wire_contract", "windows_kernel_StackWalk_prefix_versions_0_to_2"},
             {"event_version", event->EventHeader.EventDescriptor.Version},
             {"event_time_qpc", payload.event_timestamp},
             {"pid", payload.pid},
             {"tid", payload.tid},
             {"pointer_bytes", width},
             {"frames", std::move(frames)}});
  } catch (const std::exception&) {
    state.stopped = true;
    state.error = "native raw event/stack decode failed identity or resource admission";
  } catch (...) {
    state.stopped = true;
    state.error = "native raw event/stack decode allocation failed";
  }
}

void heap_event(EVENT_RECORD* event, State& state) noexcept {
  ++state.heap_events;
  const auto& header = event->EventHeader;
  if (header.TimeStamp.QuadPart < 0) {
    ++state.undecoded_heap_events;
    return;
  }
  const auto stamp = uint64_t(header.TimeStamp.QuadPart);
  if (header.ProcessId != state.selected_pid &&
      !(header.ProcessId == UINT32_MAX && selected_thread_at(state, header.ThreadId, stamp))) {
    ++state.excluded_heap_events;
    return;
  }
  const auto opcode = header.EventDescriptor.Opcode;
  if (opcode < 32 || opcode > 36) {
    ++state.unsupported_heap_events;
    return;
  }
  const bool is32 = (header.Flags & EVENT_HEADER_FLAG_32_BIT_HEADER) != 0;
  const bool is64 = (header.Flags & EVENT_HEADER_FLAG_64_BIT_HEADER) != 0;
  const uint32_t width = is32 ? 4u : 8u;
  rns8::detail::trace::EtwHeapEvent payload;
  if (is32 == is64 || !rns8::detail::trace::decode_etw_heap_event(
                          event->UserData, event->UserDataLength, header.EventDescriptor.Version,
                          opcode, width, payload)) {
    ++state.undecoded_heap_events;
    return;
  }
  if (selected_observations(state) >= 65536) {
    state.stopped = true;
    state.error = "selected ETL CPU/scheduler/heap projection budget exhausted";
    return;
  }
  try {
    json words = json::array();
    for (uint32_t i = 0; i < payload.word_count; ++i)
      words.push_back(payload.words[i]);
    state.selected_heap.push_back(
        json{{"wire_contract", "windows_kernel_HeapTrace_v0"},
             {"record_index", state.sample_walk_events},
             {"time_filetime_100ns", stamp},
             {"native_header_pid", header.ProcessId},
             {"native_header_tid", header.ThreadId},
             {"opcode", opcode},
             {"pointer_bytes", width},
             {"payload_words", std::move(words)},
             {"tail", opcode == 35 ? json(nullptr) : json(payload.tail)}});
    remember_point(event, state, PointFamily::heap, state.selected_heap.size() - 1);
  } catch (...) {
    state.stopped = true;
    state.error = "ETL heap decoder allocation failed";
  }
}

void schedule_event(EVENT_RECORD* event, State& state) noexcept {
  ++state.schedule_events;
  const auto& header = event->EventHeader;
  const bool context_switch = header.EventDescriptor.Opcode == 36;
  rns8::detail::trace::EtwContextSwitch switched;
  rns8::detail::trace::EtwReadyThread ready;
  if (header.TimeStamp.QuadPart < 0 ||
      (context_switch
           ? !rns8::detail::trace::decode_etw_context_switch(
                 event->UserData, event->UserDataLength, header.EventDescriptor.Version, switched)
           : !rns8::detail::trace::decode_etw_ready_thread(
                 event->UserData, event->UserDataLength, header.EventDescriptor.Version, ready))) {
    ++state.undecoded_schedule_events;
    return;
  }
  const auto stamp = uint64_t(header.TimeStamp.QuadPart);
  if (context_switch ? !selected_thread_at(state, switched.old_tid, stamp) &&
                           !selected_thread_at(state, switched.new_tid, stamp)
                     : !selected_thread_at(state, ready.tid, stamp)) {
    ++state.excluded_schedule_events;
    return;
  }
  if (selected_observations(state) >= 65536) {
    state.stopped = true;
    state.error = "selected ETL CPU/schedule projection budget exhausted";
    return;
  }
  try {
    const bool processor_index = (header.Flags & EVENT_HEADER_FLAG_PROCESSOR_INDEX) != 0;
    json row{{"record_index", state.sample_walk_events},
             {"time_filetime_100ns", stamp},
             {"native_header_pid", header.ProcessId},
             {"native_header_tid", header.ThreadId},
             {"processor_index_mode", processor_index},
             {"processor", processor_index ? event->BufferContext.ProcessorIndex
                                           : event->BufferContext.ProcessorNumber}};
    if (context_switch) {
      row["wire_contract"] = "windows_kernel_CSwitch_v2";
      row["old_tid"] = switched.old_tid;
      row["new_tid"] = switched.new_tid;
      row["control_bytes"] = switched.control;
      row["new_wait_time"] = switched.new_wait_time;
      row["reserved"] = switched.reserved;
    } else {
      row["wire_contract"] = "windows_kernel_ReadyThread_v2";
      row["tid"] = ready.tid;
      row["control_bytes"] = ready.control;
    }
    state.selected_schedule.push_back(std::move(row));
    remember_point(event, state, PointFamily::schedule, state.selected_schedule.size() - 1);
  } catch (...) {
    state.stopped = true;
    state.error = "ETL scheduler decoder allocation failed";
  }
}

void sample_event(EVENT_RECORD* event, State& state) noexcept {
  if (++state.sample_walk_events > max_events) {
    state.stopped = true;
    state.error = "ETL sample walk event budget exhausted";
    return;
  }
  if (IsEqualGUID(event->EventHeader.ProviderId, heap_provider)) {
    heap_event(event, state);
    return;
  }
  if (IsEqualGUID(event->EventHeader.ProviderId, thread_provider) &&
      (event->EventHeader.EventDescriptor.Opcode == 36 ||
       event->EventHeader.EventDescriptor.Opcode == 50)) {
    schedule_event(event, state);
    return;
  }
  if (!IsEqualGUID(event->EventHeader.ProviderId, profile_provider) ||
      event->EventHeader.EventDescriptor.Opcode != 46)
    return;
  ++state.sample_events;
  const bool is32 = (event->EventHeader.Flags & EVENT_HEADER_FLAG_32_BIT_HEADER) != 0;
  const bool is64 = (event->EventHeader.Flags & EVENT_HEADER_FLAG_64_BIT_HEADER) != 0;
  const size_t width = is32 ? 4u : 8u;
  // Pinned Microsoft SampledProfile v2: pointer, TID, uint16 count,
  // DPC/ISR/priority byte and rank byte. Preserve the packed word as well.
  rns8::detail::trace::EtwProfileSample sample;
  if (is32 == is64 ||
      !rns8::detail::trace::decode_etw_profile_sample(event->UserData, event->UserDataLength,
                                                      event->EventHeader.EventDescriptor.Version,
                                                      static_cast<uint32_t>(width), sample) ||
      event->EventHeader.TimeStamp.QuadPart < 0) {
    ++state.undecoded_sample_events;
    return;
  }
  const uint64_t ip = sample.instruction_pointer;
  const uint32_t tid = sample.thread_id, packed = sample.packed_count_flags_rank;
  const uint32_t pid = event->EventHeader.ProcessId;
  const uint64_t stamp = uint64_t(event->EventHeader.TimeStamp.QuadPart);
  const bool unknown_pid_selected = pid == UINT32_MAX && selected_thread_at(state, tid, stamp);
  if (pid != state.selected_pid && !unknown_pid_selected) {
    ++state.excluded_sample_events;
    return;
  }
  if (!tid) {
    ++state.undecoded_sample_events;
    return;
  }
  if (selected_observations(state) >= 65536) {
    state.stopped = true;
    state.error = "selected ETL sample budget exhausted";
    return;
  }
  const bool processor_index = (event->EventHeader.Flags & EVENT_HEADER_FLAG_PROCESSOR_INDEX) != 0;
  try {
    state.selected_samples.push_back(
        json{{"tid", tid},
             {"native_header_pid", pid},
             {"native_header_tid", event->EventHeader.ThreadId},
             {"time_filetime_100ns", stamp},
             {"instruction_pointer", ip},
             {"pointer_bytes", width},
             {"packed_count_flags_rank", packed},
             {"count", packed & 0xffff},
             {"sample_flags", (packed >> 16) & 0xff},
             {"rank", packed >> 24},
             {"processor", processor_index ? event->BufferContext.ProcessorIndex
                                           : event->BufferContext.ProcessorNumber},
             {"processor_index_mode", processor_index},
             {"record_index", state.sample_walk_events},
             {"wire_contract", "windows_kernel_SampledProfile_v2"}});
    remember_point(event, state, PointFamily::sample, state.selected_samples.size() - 1);
  } catch (...) {
    state.stopped = true;
    state.error = "ETL sample decoder allocation failed";
  }
}

void WINAPI event_callback(EVENT_RECORD* event) noexcept {
  auto& state = *static_cast<State*>(event->UserContext);
  if (state.stopped)
    return;
  if (state.raw_pass) {
    raw_event(event, state);
    return;
  }
  if (state.sample_pass) {
    sample_event(event, state);
    return;
  }
  if (++state.events > max_events) {
    state.stopped = true;
    state.error = "ETL event budget exhausted";
    return;
  }
  if (IsEqualGUID(event->EventHeader.ProviderId, marker_provider)) {
    ++state.marker_events;
    if (event->EventHeader.ProcessId != state.selected_pid)
      return;
    const auto& descriptor = event->EventHeader.EventDescriptor;
    if (descriptor.Version != 1 || (descriptor.Id != 1 && descriptor.Id != 2) ||
        descriptor.Opcode != descriptor.Id || descriptor.Level != 4 || descriptor.Keyword != 1 ||
        descriptor.Task || descriptor.Channel || event->UserDataLength != sizeof(uint64_t) ||
        !event->UserData || !event->EventHeader.ThreadId ||
        event->EventHeader.TimeStamp.QuadPart < 0) {
      ++state.undecoded_marker_events;
      return;
    }
    if (state.selected_markers.size() >= 8192) {
      state.stopped = true;
      state.error = "selected ETL marker event budget exhausted";
      return;
    }
    // No TDH/manifest guess: this exact private v1 payload is one uint64.
    uint64_t slot = 0;
    std::memcpy(&slot, event->UserData, sizeof(slot));
    try {
      state.selected_markers.push_back(
          json{{"pid", state.selected_pid},
               {"tid", event->EventHeader.ThreadId},
               {"slot", slot},
               {"event_id", descriptor.Id},
               {"time_filetime_100ns", event->EventHeader.TimeStamp.QuadPart},
               {"record_index", state.events},
               {"wire_contract", "rns8_etw_slot_v1"}});
    } catch (...) {
      state.stopped = true;
      state.error = "ETL marker decoder allocation failed";
    }
    return;
  }
  const bool image_event = IsEqualGUID(event->EventHeader.ProviderId, image_provider);
  const bool thread_event = IsEqualGUID(event->EventHeader.ProviderId, thread_provider);
  if (!image_event && !thread_event &&
      !IsEqualGUID(event->EventHeader.ProviderId, process_provider))
    return;
  const UCHAR opcode = event->EventHeader.EventDescriptor.Opcode;
  if (opcode != 2 && opcode != 3 && opcode != 4 &&
      !(image_event    ? opcode == 10
        : thread_event ? opcode == 1
                       : opcode == 1 || opcode == 39))
    return;
  if (image_event)
    ++state.image_events;
  else if (thread_event)
    ++state.thread_events;
  else
    ++state.process_events;
  try {
    const Schema& fields = schema(event, state,
                                  image_event    ? EventFamily::image
                                  : thread_event ? EventFamily::thread
                                                 : EventFamily::process);
    const uint32_t pid = uint32_property(event, fields, L"ProcessId");
    if (pid != state.selected_pid)
      return;
    const bool is32 = (event->EventHeader.Flags & EVENT_HEADER_FLAG_32_BIT_HEADER) != 0;
    const bool is64 = (event->EventHeader.Flags & EVENT_HEADER_FLAG_64_BIT_HEADER) != 0;
    if (is32 == is64)
      fail("native ETL event has no unique recorded pointer width");
    const size_t width = is32 ? 4u : 8u;
    if (event->EventHeader.TimeStamp.QuadPart < 0)
      fail("native ETL converted timestamp is invalid");
    if (thread_event) {
      if (state.selected_threads.size() >= max_process_events)
        throw DecoderBudgetExceeded{"selected ETL thread event budget exhausted"};
      const uint32_t tid = uint32_property(event, fields, L"TThreadId");
      if (!tid)
        fail("native ETL thread ID is zero");
      state.selected_threads.push_back(
          json{{"pid", pid},
               {"tid", tid},
               {"opcode", opcode},
               {"time_filetime_100ns", event->EventHeader.TimeStamp.QuadPart},
               {"event_version", event->EventHeader.EventDescriptor.Version},
               {"pointer_bytes", width},
               {"record_index", state.events}});
      return;
    }
    if (image_event) {
      if (state.selected_images.size() >= 4096)
        throw DecoderBudgetExceeded{"selected ETL image event budget exhausted"};
      const uint64_t base = pointer_property(event, fields, L"ImageBase", width);
      const uint64_t size = pointer_property(event, fields, L"ImageSize", width);
      if (!base || !size || size > std::numeric_limits<uint64_t>::max() - base)
        fail("native image span is empty or overflows");
      if (width == 4 && size > (uint64_t(1) << 32) - base)
        fail("native image span exceeds its recorded pointer domain");
      auto name =
          unicode_image(property(event, fields, L"FileName", TDH_INTYPE_UNICODESTRING, 8192));
      if (name.size() > (1u << 20) - state.image_text_bytes)
        throw DecoderBudgetExceeded{"selected ETL image path byte budget exhausted"};
      state.image_text_bytes += name.size();
      state.selected_images.push_back(
          json{{"pid", pid},
               {"opcode", opcode},
               {"time_filetime_100ns", event->EventHeader.TimeStamp.QuadPart},
               {"image_base", base},
               {"image_size", size},
               {"image_checksum", uint32_property(event, fields, L"ImageChecksum")},
               {"image_timestamp", uint32_property(event, fields, L"TimeDateStamp")},
               {"image_path_utf8", std::move(name)},
               {"event_version", event->EventHeader.EventDescriptor.Version},
               {"pointer_bytes", width}});
      return;
    }
    if (state.selected.size() >= max_process_events)
      throw DecoderBudgetExceeded{"selected ETL process event budget exhausted"};
    const uint64_t key = pointer_property(event, fields, L"UniqueProcessKey", width);
    if (!key)
      fail("native process key or converted timestamp is invalid");
    const auto image = property(event, fields, L"ImageFileName", TDH_INTYPE_ANSISTRING, 4096);
    state.selected.push_back(json{{"pid", pid},
                                  {"parent_pid", uint32_property(event, fields, L"ParentId")},
                                  {"process_key", key},
                                  {"opcode", opcode},
                                  {"time_filetime_100ns", event->EventHeader.TimeStamp.QuadPart},
                                  {"image_basename_ascii", ascii_image(image)},
                                  {"event_version", event->EventHeader.EventDescriptor.Version},
                                  {"pointer_bytes", width}});
  } catch (const DecoderBudgetExceeded& budget) {
    // Do not scan the rest of a large file after an enforceable resource limit.
    // The buffer callback stops ProcessTrace; no completed metadata is emitted.
    state.stopped = true;
    state.error = budget.message;
  } catch (const std::bad_alloc&) {
    state.stopped = true;
    state.error = "ETL native decoder allocation failed";
  } catch (...) {
    // An undecodable process event can hide PID reuse, including a record whose
    // PID property itself was unavailable. Keep the whole identity scope partial.
    if (image_event)
      ++state.undecoded_image_events;
    else if (thread_event)
      ++state.undecoded_thread_events;
    else
      ++state.undecoded_process_events;
  }
}

ULONG WINAPI buffer_callback(EVENT_TRACE_LOGFILEW* logfile) noexcept {
  return static_cast<State*>(logfile->Context)->stopped ? FALSE : TRUE;
}

struct TraceHandle {
  TRACEHANDLE value = INVALID_PROCESSTRACE_HANDLE;
  ~TraceHandle() {
    if (value != INVALID_PROCESSTRACE_HANDLE)
      CloseTrace(value);
  }
  void close() {
    const TRACEHANDLE current = value;
    value = INVALID_PROCESSTRACE_HANDLE;
    if (CloseTrace(current) != ERROR_SUCCESS)
      fail("native ETL read handle did not close successfully");
  }
};

TRACE_LOGFILE_HEADER decode_log(std::wstring& filename, State& state) {
  EVENT_TRACE_LOGFILEW logfile{};
  logfile.LogFileName = filename.data();
  logfile.ProcessTraceMode = PROCESS_TRACE_MODE_EVENT_RECORD;
  if (state.raw_pass)
    logfile.ProcessTraceMode |= PROCESS_TRACE_MODE_RAW_TIMESTAMP;
  logfile.Context = &state;
  logfile.EventRecordCallback = event_callback;
  logfile.BufferCallback = buffer_callback;
  TraceHandle trace;
  trace.value = OpenTraceW(&logfile);
  if (trace.value == INVALID_PROCESSTRACE_HANDLE)
    fail("native OpenTrace file decode failed");
  if (!logfile.LogfileHeader.EndTime.QuadPart || logfile.LogfileHeader.StartTime.QuadPart < 0 ||
      logfile.LogfileHeader.EndTime.QuadPart < logfile.LogfileHeader.StartTime.QuadPart ||
      logfile.LogfileHeader.PointerSize != sizeof(void*))
    fail("finalized native-width ETL header required");
  const ULONG status = ProcessTrace(&trace.value, 1, nullptr, nullptr);
  if (status != ERROR_SUCCESS || state.stopped)
    fail(state.error ? state.error : "native ProcessTrace did not finish the complete file");
  trace.close();
  return logfile.LogfileHeader;
}

bool same_native_header(const TRACE_LOGFILE_HEADER& left, const TRACE_LOGFILE_HEADER& right) {
  return left.StartTime.QuadPart == right.StartTime.QuadPart &&
         left.EndTime.QuadPart == right.EndTime.QuadPart &&
         left.PerfFreq.QuadPart == right.PerfFreq.QuadPart &&
         left.ReservedFlags == right.ReservedFlags && left.PointerSize == right.PointerSize &&
         left.EventsLost == right.EventsLost && left.BuffersLost == right.BuffersLost;
}

void select_sample_windows(State& state) {
  struct Bounds {
    uint32_t starts = 0, stops = 0;
    uint64_t start = 0, stop = 0;
  };
  std::map<uint32_t, Bounds> bounds;
  for (const auto& row : state.selected_threads) {
    auto& value = bounds[row.at("tid").get<uint32_t>()];
    const auto opcode = row.at("opcode").get<uint32_t>();
    if (opcode == 1) {
      ++value.starts;
      value.start = row.at("time_filetime_100ns").get<uint64_t>();
    } else if (opcode == 2) {
      ++value.stops;
      value.stop = row.at("time_filetime_100ns").get<uint64_t>();
    }
  }
  for (const auto& entry : bounds)
    if (entry.second.starts == 1 && entry.second.stops == 1 &&
        entry.second.start < entry.second.stop)
      state.sample_thread_windows.emplace(entry.first,
                                          std::make_pair(entry.second.start, entry.second.stop));
}

uint32_t parse_pid(const wchar_t* text) {
  uint64_t result = 0;
  if (!*text)
    fail("positive uint32 PID required");
  for (; *text; ++text) {
    if (*text < L'0' || *text > L'9' || result > (UINT32_MAX - uint32_t(*text - L'0')) / 10)
      fail("positive uint32 PID required");
    result = result * 10 + uint32_t(*text - L'0');
  }
  if (!result)
    fail("positive uint32 PID required");
  return uint32_t(result);
}
} // namespace

int wmain(int argc, wchar_t** argv) {
  try {
    if (argc == 2 && std::wstring(argv[1]) == L"--help") {
      std::cout << "rns8-etl-inspect --input finalized.etl --pid positive_uint32\n";
      return 0;
    }
    if (argc == 2 && std::wstring(argv[1]) == L"--build-identity") {
#ifdef _MSC_FULL_VER
      constexpr uint64_t compiler_msc = _MSC_FULL_VER;
#else
      constexpr uint64_t compiler_msc = 0;
#endif
      std::cout << json{{"schema", "rns8_etl_inspector_identity_v1"},
                        {"process_id", GetCurrentProcessId()},
                        {"source_sha256", RNS8_ETL_INSPECT_SOURCE_SHA256},
                        {"pointer_bytes", sizeof(void*)},
                        {"native_sdk_compiler_msc", compiler_msc}}
                       .dump()
                << '\n';
      return 0;
    }
    if (argc != 5 || std::wstring(argv[1]) != L"--input" || std::wstring(argv[3]) != L"--pid")
      fail("usage: rns8-etl-inspect --input finalized.etl --pid positive_uint32");
    State state;
    state.selected_pid = parse_pid(argv[4]);
    const std::filesystem::path path = std::filesystem::canonical(argv[2]);
    if (!std::filesystem::is_regular_file(path))
      fail("regular finalized ETL required");
    const auto before = std::filesystem::file_size(path);
    const auto modified = std::filesystem::last_write_time(path);
    if (!before || before > max_bytes)
      fail("ETL file byte budget exceeded");
    std::wstring filename = path.wstring();
    const auto header = decode_log(filename, state);
    state.bind_raw_points = header.ReservedFlags == 1 && header.PerfFreq.QuadPart > 0;
    select_sample_windows(state);
    state.sample_pass = true;
    const auto sample_header = decode_log(filename, state);
    if (state.sample_walk_events != state.events || !same_native_header(header, sample_header))
      fail("native ETL metadata and sample walks disagree");
    if (state.bind_raw_points && !state.point_sources.empty()) {
      state.raw_pass = true;
      const auto raw_header = decode_log(filename, state);
      if (state.raw_walk_events != state.events || !same_native_header(header, raw_header) ||
          state.raw_next != state.point_sources.size())
        fail("native raw/converted walks or source event inventory disagree");
    }
    if (std::filesystem::file_size(path) != before ||
        std::filesystem::last_write_time(path) != modified)
      fail("ETL file changed during native decoding");
    json result{
        {"schema", "rns8_native_etl_process_metadata_v1"},
        {"reader_source_sha256", RNS8_ETL_INSPECT_SOURCE_SHA256},
        {"decoder_pid", GetCurrentProcessId()},
        {"selected_pid", state.selected_pid},
        {"native_process_trace_status", ERROR_SUCCESS},
        {"complete_event_walk", true},
        {"file_bytes", before},
        {"observed_events", state.events},
        {"observed_process_events", state.process_events},
        {"undecoded_process_events", state.undecoded_process_events},
        {"process_events", std::move(state.selected)},
        {"observed_image_events", state.image_events},
        {"undecoded_image_events", state.undecoded_image_events},
        {"image_events", std::move(state.selected_images)},
        {"observed_marker_events", state.marker_events},
        {"undecoded_marker_events", state.undecoded_marker_events},
        {"marker_events", std::move(state.selected_markers)},
        {"observed_thread_events", state.thread_events},
        {"undecoded_thread_events", state.undecoded_thread_events},
        {"thread_events", std::move(state.selected_threads)},
        {"sample_walk_events", state.sample_walk_events},
        {"observed_sample_events", state.sample_events},
        {"undecoded_sample_events", state.undecoded_sample_events},
        {"excluded_sample_events", state.excluded_sample_events},
        {"sample_events", std::move(state.selected_samples)},
        {"sample_filter",
         "known_selected_header_PID_or_unique_selected_thread_Start_End_for_unknown_header_PID"},
        {"sample_record_namespace", "second_native_event_walk"},
        {"complete_selected_sample_coverage_proven", false},
        {"observed_schedule_events", state.schedule_events},
        {"undecoded_schedule_events", state.undecoded_schedule_events},
        {"excluded_schedule_events", state.excluded_schedule_events},
        {"schedule_events", std::move(state.selected_schedule)},
        {"schedule_filter", "unique_selected_thread_Start_End_window"},
        {"schedule_record_namespace", "second_native_event_walk"},
        {"observed_heap_events", state.heap_events},
        {"undecoded_heap_events", state.undecoded_heap_events},
        {"excluded_heap_events", state.excluded_heap_events},
        {"unsupported_heap_events", state.unsupported_heap_events},
        {"heap_events", std::move(state.selected_heap)},
        {"heap_filter",
         "known_selected_header_PID_or_unique_selected_thread_Start_End_for_unknown_header_PID"},
        {"heap_record_namespace", "second_native_event_walk"},
        {"raw_binding_state", state.raw_pass
                                  ? "raw_QPC_bound_by_native_ordinal_header_context_payload_replay"
                              : state.bind_raw_points ? "no_selected_native_points"
                                                      : "unsupported_non_QPC_clock"},
        {"raw_walk_events", state.raw_walk_events},
        {"raw_source_point_count", state.raw_pass ? state.point_sources.size() : 0},
        {"raw_record_namespace", "third_native_event_walk"},
        {"observed_stack_events", state.stack_events},
        {"undecoded_stack_events", state.undecoded_stack_events},
        {"excluded_stack_events", state.excluded_stack_events},
        {"unsupported_stack_events", state.unsupported_stack_events},
        {"stack_events", std::move(state.selected_stacks)},
        {"stack_frames", state.stack_frames},
        {"header",
         {{"pointer_bytes", header.PointerSize},
          {"clock_type", header.ReservedFlags},
          {"perf_frequency", header.PerfFreq.QuadPart},
          {"start_filetime_100ns", header.StartTime.QuadPart},
          {"end_filetime_100ns", header.EndTime.QuadPart},
          {"events_lost", header.EventsLost},
          {"buffers_lost", header.BuffersLost},
          {"buffers_written", header.BuffersWritten}}},
        {"event_timestamp_domain", "native_ProcessTrace_converted_FILETIME_100ns"},
        {"complete_loss_free_acquisition_proven", false},
        {"image_path_or_loaded_PDB_verified", false},
        {"production_latency", false},
        {"promotion_allowed", false}};
    std::cout << result.dump() << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "rns8-etl-inspect: " << error.what() << '\n';
    return 2;
  }
}
