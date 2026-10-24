#include "backend_hip_direct/hip_backend.hpp"
#include "core/hip_resources.hpp"
#include "core/internal.hpp"
#include <charconv>
#include <chrono>
#include <iostream>
#include <nlohmann/json.hpp>
#include <random>
#include <stdexcept>

namespace {
using namespace rns8::detail;
using boost::multiprecision::cpp_int;
void check(rns8_status status) {
  if (status != RNS8_SUCCESS)
    throw std::runtime_error(rns8_status_string(status));
}
uint32_t number(const char *text, uint32_t low, uint32_t high) {
  uint32_t value = 0;
  const auto end = text + std::char_traits<char>::length(text);
  const auto parsed = std::from_chars(text, end, value);
  if (parsed.ec != std::errc{} || parsed.ptr != end || value < low ||
      value > high)
    throw std::invalid_argument("numeric argument outside supported range");
  return value;
}
struct StatusStorage {
  void *pointer = nullptr;
  std::size_t bytes = 0;
  ~StatusStorage() {
    if (pointer)
      (void)hip_direct_free(0, pointer);
  }
};
} // namespace
int main(int argc, char **argv) {
  try {
    if (argc != 7)
      throw std::invalid_argument(
          "usage: rns8-crt-calibrate <garner|direct|balanced|tile_balanced|tile_convert_balanced|tile_convert_garner> <cells "
          "1..1048576> <prefix 1..20> <repeats 1..10000> <warmups 1..1000> "
          "<unique-run-id>");
    const std::string candidate = argv[1], run_id = argv[6];
    if ((candidate != "garner" && candidate != "direct" &&
         candidate != "balanced" && candidate != "tile_balanced" &&
         candidate != "tile_convert_balanced" && candidate != "tile_convert_garner") ||
        run_id.empty())
      throw std::invalid_argument("candidate and unique run ID required");
    const auto cells = number(argv[2], 1, 1048576),
               prefix = number(argv[3], 1, 20),
               repeats = number(argv[4], 1, 10000),
               warmups = number(argv[5], 1, 1000);
    rns8_device_info device{};
    check(hip_direct_probe(0, device));
    std::mt19937_64 rng(3903);
    const cpp_int product = modulus_product(prefix);
    std::vector<int8_t> residues(std::size_t{cells} * prefix);
    std::vector<uint64_t> expected(std::size_t{cells} * 3 + 3, UINT64_MAX),
        observed(expected.size());
    for (uint32_t cell = 0; cell < cells; ++cell) {
      cpp_int value =
          (cpp_int(rng()) + (cpp_int(rng()) << 64) + (cpp_int(rng()) << 128)) %
          product;
      if (cell == 0)
        value = 0;
      if (cell == 1)
        value = product - 1;
      if (cell == 2)
        value = product / 2;
      for (uint32_t p = 0; p < prefix; ++p)
        residues[std::size_t{p} * cells + cell] =
            centered_residue(value, kDefaultModuli[p]);
      for (int limb = 0; limb < 3; ++limb) {
        expected[std::size_t{cell} * 3 + limb] =
            static_cast<uint64_t>(value & UINT64_MAX);
        value >>= 64;
      }
    }
    const bool tile_input = candidate.rfind("tile_", 0) == 0;
    const uint64_t cols = cells % 256 == 0 ? 256 : cells, rows = cells / cols;
    ResidueLayout plain, tiled;
    if (!make_residue_layout(ResidueLayoutKind::modulus_major, rows, cols, cols, prefix, 0, 0, plain) ||
        !make_residue_layout(ResidueLayoutKind::tile_major, rows, cols, cols, prefix, 8, 16, tiled))
      throw std::runtime_error("CRT layout construction failed");
    std::vector<int8_t> tile_residues;
    if (tile_input) {
      tile_residues.assign(tiled.bytes, 99);
      if (!convert_residue_layout_host(residues.data(), plain, tile_residues.data(), tiled))
        throw std::runtime_error("CRT input layout conversion failed");
    }
    hip_direct_device_buffer input, output, tile_storage;
    if (candidate != "tile_balanced") check(input.allocate(0, residues.size()));
    check(output.allocate(0, expected.size() * 8));
    if (candidate != "tile_balanced")
      check(hip_direct_copy_host_to_device(0, input.get(), residues.data(), residues.size()));
    std::fill(observed.begin(), observed.end(), UINT64_MAX);
    check(hip_direct_copy_host_to_device(0, output.get(), observed.data(),
                                         observed.size() * 8));
    if (tile_input) {
      check(tile_storage.allocate(0, tiled.bytes));
      check(hip_direct_copy_host_to_device(0,tile_storage.get(),tile_residues.data(),tile_residues.size()));
    }
    StatusStorage status;
    const auto execute = [&] {
      if (candidate == "tile_balanced")
        return hip_direct_export_canonical_balanced_crt_layout_device(0,tile_storage.get(),tiled,output.get());
      if (tile_input)
        check(hip_direct_convert_residue_layout(0,tile_storage.get(),tiled,input.get(),plain));
      if (candidate == "direct")
        return hip_direct_export_canonical_direct_crt_device(
            0, input.get(), output.get(), cells, prefix);
      if (candidate == "balanced" || candidate == "tile_convert_balanced")
        return hip_direct_export_canonical_balanced_crt_device(
            0, input.get(), output.get(), cells, prefix);
      return hip_direct_export_exact_wide_unsigned_limbs_to_device(
          0, input.get(), output.get(), &status.pointer, &status.bytes, 1,
          cells, prefix, 3);
    };
    const auto verify = [&] {
      check(hip_direct_copy_device_to_host(0, observed.data(), output.get(),
                                           observed.size() * 8));
      if (observed != expected)
        throw std::runtime_error(
            "CRT output or tail canary differs from Boost reference");
    };
    for (uint32_t i = 0; i < warmups; ++i) {
      check(execute());
      verify();
    }
    nlohmann::json runs = nlohmann::json::array();
    hip_direct_timing_set_enabled(true);
    for (uint32_t i = 0; i < repeats; ++i) {
      hip_direct_timing_reset();
      const auto start = std::chrono::steady_clock::now();
      check(execute());
      const double host_ns = std::chrono::duration<double, std::nano>(
                                 std::chrono::steady_clock::now() - start)
                                 .count();
      hip_direct_timing_flush_pending_events();
      const auto events = hip_direct_timing_snapshot();
      nlohmann::json event_json = nlohmann::json::array();
      for (const auto &event : events)
        event_json.push_back(
            {{"label", event.label},
             {"elapsed_ns", event.microseconds > 0
                                ? nlohmann::json(event.microseconds * 1000)
                                : nlohmann::json(nullptr)}});
      verify();
      runs.push_back({{"host_call_ns", host_ns > 0 ? nlohmann::json(host_ns)
                                                   : nlohmann::json(nullptr)},
                      {"events", event_json},
                      {"exact_reference_passed", true}});
    }
    hip_direct_timing_set_enabled(false);
    std::cout << nlohmann::json(
                     {{"schema", "rns8_crt_microcase_v1"},
                      {"run_id", run_id},
                      {"candidate", candidate},
                      {"cells", cells},
                      {"prefix", prefix},
                      {"seed", 3903},
                      {"warmups", warmups},
                      {"repeats", repeats},
                      {"output_contract", "canonical_three_limbs_device"},
                      {"cache_state", "repeated_same_buffers"},
                      {"device", device.name},
                      {"target", device.gcn_arch},
                      {"runtime_version", device.hip_runtime_version},
                      {"build_manifest",
                       nlohmann::json::parse(RNS8_BUILD_MANIFEST_JSON)},
                      {"source_bytes_modeled", std::size_t{cells} * prefix},
                      {"source_layout", tile_input ? "tile_major_8x16_v1" : "modulus_major_v1"},
                      {"source_physical_bytes", tile_input ? tiled.bytes : plain.bytes},
                      {"logical_rows", rows}, {"logical_cols", cols},
                      {"conversion_inside_call", tile_input && candidate != "tile_balanced"},
                      {"conversion_scratch_bytes", tile_input && candidate != "tile_balanced" ? plain.bytes : 0},
                      {"output_bytes_modeled", std::size_t{cells} * 24},
                      {"actual_hbm_bytes", nullptr},
                      {"timing_resolution_ns", nullptr},
                      {"isa_verified", false},
                      {"promotion_eligible", false},
                      {"samples", runs},
                      {"timing_scope",
                       "host call includes synchronization and "
                       "candidate-specific status work; events retain separate "
                       "labels; verification copy excluded"}})
                     .dump(2)
              << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
