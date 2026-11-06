#include "../tests/fuzz/arithmetic_replay.hpp"
#include "backend_common/sha256.hpp"
#include "tool_common/build_identity.hpp"
#include "tool_common/cli_integer.hpp"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <string_view>

namespace {
using nlohmann::json;
using namespace rns8_fuzz;
using namespace rns8_fuzz::replay_support;
json report;
Observation observed;
void publish(const char* state) {
  report["state"] = state;
  report["gemm_calls"] = observed.gemm_calls;
  report["full_output_comparisons"] = observed.comparisons;
  report["device"] = observed.device_observed
                         ? json{{"name", observed.device.name},
                                {"target", observed.device.gcn_arch},
                                {"ordinal", observed.device.device_id},
                                {"hip_available", observed.device.hip_available},
                                {"runtime_version", observed.device.hip_runtime_version},
                                {"driver_version", observed.device.hip_driver_version}}
                         : json(nullptr);
  report["plan"] =
      observed.plan_observed
          ? json{{"backend", observed.plan.backend},
                 {"kernel", observed.plan.selected_kernel},
                 {"accelerator", observed.plan.accelerator_library},
                 {"accelerator_version", observed.plan.accelerator_version},
                 {"compiled_kernel_available", observed.plan.compiled_kernel_available},
                 {"k_block_size", observed.plan.accumulator_k_block_size},
                 {"k_block_cap", observed.plan.accumulator_k_block_cap}}
          : json(nullptr);
  std::cout << report.dump() << '\n' << std::flush;
}
[[noreturn]] void native_failure(const Failure& failure) {
  const std::string_view phase(failure.phase);
  const bool unsupported =
      std::string_view(failure.kind) == "status" &&
      (phase == "create context" || phase == "create plan") &&
      (failure.actual == RNS8_UNSUPPORTED_BACKEND || failure.actual == RNS8_UNSUPPORTED_OS ||
       failure.actual == RNS8_UNSUPPORTED_ARCH);
  report["failure"] = {{"kind", failure.kind},
                       {"phase", failure.phase},
                       {"actual_status", failure.actual},
                       {"expected_status", failure.expected},
                       {"storage_element", failure.storage_element},
                       {"actual_value", failure.actual_value},
                       {"expected_value", failure.expected_value}};
  // One case per owned child. Preserve the first failure without unwinding
  // public owners or issuing further cleanup calls after uncertain device work.
  report["cleanup"] = "process_exit_without_further_library_calls";
  publish(unsupported ? "unsupported" : "failed");
  std::_Exit(unsupported ? 77 : 1);
}
std::vector<uint8_t> read_case(const char* path) {
  if (!std::filesystem::is_regular_file(path))
    throw std::invalid_argument("case must be a regular file");
  std::ifstream stream(path, std::ios::binary);
  if (!stream)
    throw std::invalid_argument("cannot open arithmetic case");
  std::vector<uint8_t> bytes(max_case_bytes + 1);
  stream.read(reinterpret_cast<char*>(bytes.data()), std::streamsize(bytes.size()));
  const auto count = stream.gcount();
  if (stream.bad() || count < 0 || uint64_t(count) > max_case_bytes)
    throw std::invalid_argument("arithmetic case read/byte budget failed");
  bytes.resize(size_t(count));
  return bytes;
}
std::string digest(const std::vector<uint8_t>& bytes) {
  return rns8::detail::sha256(
      std::string_view(reinterpret_cast<const char*>(bytes.data()), bytes.size()));
}
bool run_name(const char* text) {
  const std::string_view value(text);
  return !value.empty() && value.size() <= 128 &&
         std::all_of(value.begin(), value.end(), [](char c) {
           return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
                  c == '-' || c == '_';
         });
}
} // namespace
int main(int argc, char** argv) {
  const int identity = rns8::tools::emit_build_identity(argc, argv, std::cout);
  if (identity >= 0)
    return identity;
  try {
    const char* usage =
        "usage: rns8-arithmetic-replay --case FILE --backend NAME --device ORDINAL --run-id ID";
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
      std::cout
          << usage
          << "\nBackends: cpu-reference, hip-direct, hipblaslt, ck, rocwmma, wrap64-byte-limb, hip-vector-alu-int64, amdgpu-builtins, cpu-amx. CPU device=-1; GPU device>=0; wrap selects explicitly by device. One bounded case; no AUTO.\n";
      return 0;
    }
    if (argc != 9)
      throw std::invalid_argument(usage);
    const char *path = nullptr, *backend_name = nullptr, *device_text = nullptr, *run_id = nullptr;
    for (int i = 1; i < argc; i += 2) {
      const std::string_view key(argv[i]);
      const char** slot = key == "--case"      ? &path
                          : key == "--backend" ? &backend_name
                          : key == "--device"  ? &device_text
                          : key == "--run-id"  ? &run_id
                                               : nullptr;
      if (!slot || *slot)
        throw std::invalid_argument(usage);
      *slot = argv[i + 1];
    }
    int device = -2;
    if (!path || !backend_name || !device_text || !run_id || !run_name(run_id) ||
        !rns8::cli::parse_device(device_text, device))
      throw std::invalid_argument(usage);
    const char* names[] = {
        "cpu-reference",    "hip-direct",           "hipblaslt",       "ck",     "rocwmma",
        "wrap64-byte-limb", "hip-vector-alu-int64", "amdgpu-builtins", "cpu-amx"};
    unsigned backend_number = 0;
    for (unsigned i = 0; i < std::size(names); ++i)
      if (std::string_view(names[i]) == backend_name)
        backend_number = i + 1;
    if (!backend_number || ((backend_number == 1 || backend_number == 9) ? device != -1
                            : backend_number == 6                        ? false
                                                                         : device < 0))
      throw std::invalid_argument("explicit backend/device contract rejected");
    const auto bytes = read_case(path);
    const auto hash = digest(bytes);
    report = {{"schema", "rns8_arithmetic_replay_v1"},
              {"run_id", run_id},
              {"process_id", rns8::tools::process_id()},
              {"case_sha256", hash},
              {"case_bytes", bytes.size()},
              {"requested_backend", backend_name},
              {"requested_backend_value", backend_number},
              {"requested_device", device},
              {"promotion_eligible", false},
              {"production_latency", false},
              {"exact_reference_passed", false},
              {"cleanup", "not_started"},
              {"failure", nullptr}};
    failure_handler = native_failure;
#ifdef RNS8_BUILD_MANIFEST_JSON
    report["build_manifest"] = json::parse(RNS8_BUILD_MANIFEST_JSON);
#else
    throw std::runtime_error("compiled executable manifest required");
#endif
    auto library = abi<rns8_public_abi_info>();
    check(rns8_get_public_abi_info(&library), RNS8_SUCCESS, "library identity");
    if (!library.build_manifest_json || !library.declaration_sha256)
      throw std::runtime_error("compiled library identity required");
    report["library_build_manifest"] = json::parse(library.build_manifest_json);
    report["library_abi_declaration_sha256"] = library.declaration_sha256;
    ArithmeticCase value;
    if (!decode(bytes.data(), bytes.size(), value)) {
      publish("malformed");
      return 65;
    }
    report["work"] = {{"semantic", value.semantic},
                      {"m", value.m},
                      {"n", value.n},
                      {"k", value.k},
                      {"prefix", value.prefix},
                      {"modulus", value.modulus},
                      {"operations", value.operations},
                      {"padding_a", value.padding_a},
                      {"padding_b", value.padding_b},
                      {"padding_c", value.padding_c}};
    const auto expected = oracle(value);
    if (!expected.admitted) {
      publish("rejected_envelope");
      return 65;
    }
    failure_handler = native_failure;
    replay(value, expected, static_cast<rns8_backend_kind>(backend_number), device, &observed);
    failure_handler = nullptr;
    report["cleanup"] = "checked_public_destruction_passed";
    if (read_case(path) != bytes)
      throw std::runtime_error("case changed during replay");
    report["exact_reference_passed"] = true;
    publish("passed");
    return 0;
  } catch (const std::exception& error) {
    failure_handler = nullptr;
    if (report.is_object()) {
      report["infrastructure_error"] = error.what();
      publish("infrastructure_error");
    } else
      std::cerr << error.what() << '\n';
    return 2;
  }
}
