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
#include <vector>
#include <nlohmann/json.hpp>

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
  uint32_t selected_pid = 0;
  bool stopped = false;
  const char* error = nullptr;
  json selected = json::array();
  std::map<uint32_t, Schema> schemas;
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

const Schema& schema(EVENT_RECORD* event, State& state) {
  const auto& descriptor = event->EventHeader.EventDescriptor;
  const uint32_t key = uint32_t(descriptor.Version) | (uint32_t(descriptor.Opcode) << 8) |
                       (uint32_t(event->EventHeader.Flags & EVENT_HEADER_FLAG_32_BIT_HEADER) << 16);
  const auto found = state.schemas.find(key);
  if (found != state.schemas.end()) {
    if (!found->second.valid)
      fail("cached native process TDH schema is unsupported");
    return found->second;
  }
  if (state.schemas.size() >= 32)
    throw DecoderBudgetExceeded{"TDH process schema budget exhausted"};
  Schema& value = state.schemas.emplace(key, Schema{}).first->second;
  ULONG size = 0;
  if (TdhGetEventInformation(event, 0, nullptr, nullptr, &size) != ERROR_INSUFFICIENT_BUFFER ||
      size < offsetof(TRACE_EVENT_INFO, EventPropertyInfoArray))
    fail("native process TDH schema unavailable");
  if (size > (1u << 20))
    throw DecoderBudgetExceeded{"native process TDH schema byte budget exhausted"};
  value.bytes.resize(size);
  auto* info = reinterpret_cast<TRACE_EVENT_INFO*>(value.bytes.data());
  if (TdhGetEventInformation(event, 0, nullptr, info, &size) != ERROR_SUCCESS ||
      size > value.bytes.size())
    fail("native process TDH schema read failed");
  if (info->TopLevelPropertyCount > info->PropertyCount || info->PropertyCount > 128 ||
      offsetof(TRACE_EVENT_INFO, EventPropertyInfoArray) +
              size_t(info->PropertyCount) * sizeof(EVENT_PROPERTY_INFO) >
          size)
    fail("native TDH property directory is incomplete");
  for (ULONG i = 0; i < info->TopLevelPropertyCount; ++i) {
    const auto& field = info->EventPropertyInfoArray[i];
    const std::wstring name = bounded_name(value.bytes, field.NameOffset);
    if (name != L"ProcessId" && name != L"ParentId" && name != L"UniqueProcessKey" &&
        name != L"ImageFileName")
      continue;
    const auto unsupported =
        PropertyStruct | PropertyParamCount | PropertyWBEMXmlFragment | PropertyHasCustomSchema;
    if (field.Flags & unsupported || field.count != 1)
      fail("native process identity property is not a supported scalar");
    if (!value.fields.emplace(name, Property{field.nonStructType.InType}).second)
      fail("native process identity schema has duplicate fields");
  }
  value.valid = true;
  return value;
}

std::vector<BYTE> property(EVENT_RECORD* event, const Schema& schema, const wchar_t* name,
                           USHORT expected, ULONG maximum) {
  const auto field = schema.fields.find(name);
  if (field == schema.fields.end() || field->second.type != expected)
    fail("native process identity property type is unsupported");
  PROPERTY_DATA_DESCRIPTOR descriptor{};
  descriptor.PropertyName = reinterpret_cast<ULONGLONG>(name);
  descriptor.ArrayIndex = ULONG_MAX;
  ULONG size = 0;
  if (TdhGetPropertySize(event, 0, nullptr, 1, &descriptor, &size) != ERROR_SUCCESS || !size ||
      size > maximum)
    fail("native process identity property size is invalid");
  std::vector<BYTE> bytes(size);
  if (TdhGetProperty(event, 0, nullptr, 1, &descriptor, size, bytes.data()) != ERROR_SUCCESS)
    fail("native process identity property read failed");
  return bytes;
}

uint32_t pid_property(EVENT_RECORD* event, const Schema& schema, const wchar_t* name) {
  const auto bytes = property(event, schema, name, TDH_INTYPE_UINT32, 4);
  if (bytes.size() != sizeof(uint32_t))
    fail("native process PID property is not uint32");
  uint32_t result = 0;
  std::memcpy(&result, bytes.data(), sizeof(result));
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

void WINAPI event_callback(EVENT_RECORD* event) noexcept {
  auto& state = *static_cast<State*>(event->UserContext);
  if (state.stopped)
    return;
  if (++state.events > max_events) {
    state.stopped = true;
    state.error = "ETL event budget exhausted";
    return;
  }
  if (!IsEqualGUID(event->EventHeader.ProviderId, process_provider))
    return;
  const UCHAR opcode = event->EventHeader.EventDescriptor.Opcode;
  if (opcode != 1 && opcode != 2 && opcode != 3 && opcode != 4 && opcode != 39)
    return;
  ++state.process_events;
  try {
    const Schema& fields = schema(event, state);
    const uint32_t pid = pid_property(event, fields, L"ProcessId");
    if (pid != state.selected_pid)
      return;
    if (state.selected.size() >= max_process_events)
      throw DecoderBudgetExceeded{"selected ETL process event budget exhausted"};
    const bool is32 = (event->EventHeader.Flags & EVENT_HEADER_FLAG_32_BIT_HEADER) != 0;
    const bool is64 = (event->EventHeader.Flags & EVENT_HEADER_FLAG_64_BIT_HEADER) != 0;
    if (is32 == is64)
      fail("native process event has no unique recorded pointer width");
    const auto key_bytes = property(event, fields, L"UniqueProcessKey", TDH_INTYPE_POINTER, 8);
    if (key_bytes.size() != (is32 ? 4u : 8u))
      fail("native process key width differs from its recorded event header");
    uint64_t key = 0;
    std::memcpy(&key, key_bytes.data(), key_bytes.size());
    if (!key || event->EventHeader.TimeStamp.QuadPart < 0)
      fail("native process key or converted timestamp is invalid");
    const auto image = property(event, fields, L"ImageFileName", TDH_INTYPE_ANSISTRING, 4096);
    state.selected.push_back(json{{"pid", pid},
                                  {"parent_pid", pid_property(event, fields, L"ParentId")},
                                  {"process_key", key},
                                  {"opcode", opcode},
                                  {"time_filetime_100ns", event->EventHeader.TimeStamp.QuadPart},
                                  {"image_basename_ascii", ascii_image(image)},
                                  {"event_version", event->EventHeader.EventDescriptor.Version},
                                  {"pointer_bytes", key_bytes.size()}});
  } catch (const DecoderBudgetExceeded& budget) {
    // Do not scan the rest of a large file after an enforceable resource limit.
    // The buffer callback stops ProcessTrace; no completed metadata is emitted.
    state.stopped = true;
    state.error = budget.message;
  } catch (const std::bad_alloc&) {
    state.stopped = true;
    state.error = "ETL process decoder allocation failed";
  } catch (...) {
    // An undecodable process event can hide PID reuse, including a record whose
    // PID property itself was unavailable. Keep the whole identity scope partial.
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
    EVENT_TRACE_LOGFILEW logfile{};
    logfile.LogFileName = filename.data();
    logfile.ProcessTraceMode = PROCESS_TRACE_MODE_EVENT_RECORD;
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
    if (std::filesystem::file_size(path) != before ||
        std::filesystem::last_write_time(path) != modified)
      fail("ETL file changed during native decoding");
    const auto& header = logfile.LogfileHeader;
    json result{{"schema", "rns8_native_etl_process_metadata_v1"},
                {"reader_source_sha256", RNS8_ETL_INSPECT_SOURCE_SHA256},
                {"decoder_pid", GetCurrentProcessId()},
                {"selected_pid", state.selected_pid},
                {"native_process_trace_status", status},
                {"complete_event_walk", true},
                {"file_bytes", before},
                {"observed_events", state.events},
                {"observed_process_events", state.process_events},
                {"undecoded_process_events", state.undecoded_process_events},
                {"process_events", std::move(state.selected)},
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
