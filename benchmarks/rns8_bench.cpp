#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "rns8/rns8.h"

namespace {

struct Args {
  int64_t m = 64;
  int64_t n = 64;
  int64_t k = 64;
  uint32_t repeats = 5;
  uint64_t seed = 1;
};

int64_t parse_i64(const char* text) {
  return std::strtoll(text, nullptr, 10);
}

Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--m" && i + 1 < argc) args.m = parse_i64(argv[++i]);
    else if (arg == "--n" && i + 1 < argc) args.n = parse_i64(argv[++i]);
    else if (arg == "--k" && i + 1 < argc) args.k = parse_i64(argv[++i]);
    else if (arg == "--repeats" && i + 1 < argc) args.repeats = static_cast<uint32_t>(parse_i64(argv[++i]));
    else if (arg == "--seed" && i + 1 < argc) args.seed = static_cast<uint64_t>(parse_i64(argv[++i]));
    else if (arg == "--help") {
      std::cout << "usage: rns8-bench [--m M] [--n N] [--k K] [--repeats R] [--seed S]\n";
      std::exit(0);
    } else {
      std::cerr << "unknown argument: " << arg << "\n";
      std::exit(2);
    }
  }
  return args;
}

uint64_t millis_since(std::chrono::steady_clock::time_point start, std::chrono::steady_clock::time_point end) {
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
}

rns8_matrix_desc matrix_desc(int64_t rows, int64_t cols) {
  rns8_matrix_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.rows = rows;
  desc.cols = cols;
  desc.logical_ld = cols;
  desc.semantics = RNS8_BOUNDED_I64;
  desc.logical_layout = RNS8_LAYOUT_ROW_MAJOR;
  desc.bound_kind = RNS8_BOUND_GLOBAL_MAX_ABS;
  desc.tile_m = 128;
  desc.tile_n = 128;
  desc.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
  return desc;
}

void fail_status(const char* label, rns8_status status) {
  std::cerr << label << ": " << rns8_status_string(status) << "\n";
  std::exit(1);
}

}  // namespace

int main(int argc, char** argv) {
  const Args args = parse_args(argc, argv);
  if (args.m <= 0 || args.n <= 0 || args.k <= 0 || args.repeats == 0) {
    std::cerr << "invalid shape or repeat count\n";
    return 2;
  }

  std::mt19937_64 rng(args.seed);
  std::uniform_int_distribution<int64_t> dist(-16, 16);
  std::vector<int64_t> A(static_cast<std::size_t>(args.m * args.k));
  std::vector<int64_t> B(static_cast<std::size_t>(args.k * args.n));
  std::vector<int64_t> C(static_cast<std::size_t>(args.m * args.n));
  for (auto& value : A) value = dist(rng);
  for (auto& value : B) value = dist(rng);

  rns8_context_options options{};
  options.struct_size = sizeof(options);
  options.abi_version = RNS8_ABI_VERSION;
  options.requested_backend = RNS8_BACKEND_CPU_REFERENCE;
  rns8_context* ctx = nullptr;
  rns8_status status = rns8_create_context(-1, &options, &ctx);
  if (status != RNS8_SUCCESS) {
    std::cerr << "rns8_create_context: " << rns8_status_string(status) << "\n";
    return 1;
  }

  rns8_gemm_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.semantics = RNS8_BOUNDED_I64;
  desc.bound_kind = RNS8_BOUND_GLOBAL_MAX_ABS;
  desc.requested_backend = RNS8_BACKEND_CPU_REFERENCE;
  desc.m = args.m;
  desc.n = args.n;
  desc.k = args.k;
  desc.bound = static_cast<uint64_t>(args.k * 16 * 16);

  const auto plan_start = std::chrono::steady_clock::now();
  rns8_plan* plan = nullptr;
  status = rns8_create_plan(ctx, &desc, &plan);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_plan", status);
  rns8_workspace* workspace = nullptr;
  status = rns8_create_workspace(ctx, plan, &workspace);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_workspace", status);
  const auto plan_end = std::chrono::steady_clock::now();

  rns8_matrix* a_matrix = nullptr;
  rns8_matrix* b_matrix = nullptr;
  rns8_matrix* c_matrix = nullptr;
  const auto alloc_start = std::chrono::steady_clock::now();
  auto a_desc = matrix_desc(args.m, args.k);
  auto b_desc = matrix_desc(args.k, args.n);
  auto c_desc = matrix_desc(args.m, args.n);
  status = rns8_create_matrix(ctx, &a_desc, &a_matrix);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(A)", status);
  status = rns8_create_matrix(ctx, &b_desc, &b_matrix);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(B)", status);
  status = rns8_create_matrix(ctx, &c_desc, &c_matrix);
  if (status != RNS8_SUCCESS) fail_status("rns8_create_matrix(C)", status);
  const auto alloc_end = std::chrono::steady_clock::now();

  uint64_t pack_us = 0;
  uint64_t gemm_us = 0;
  uint64_t export_us = 0;
  for (uint32_t r = 0; r < args.repeats; ++r) {
    const auto pack_start = std::chrono::steady_clock::now();
    status = rns8_pack_i64(ctx, a_matrix, A.data(), args.k, r + 1);
    if (status != RNS8_SUCCESS) fail_status("rns8_pack_i64(A)", status);
    status = rns8_pack_i64(ctx, b_matrix, B.data(), args.n, r + 1);
    if (status != RNS8_SUCCESS) fail_status("rns8_pack_i64(B)", status);
    const auto pack_end = std::chrono::steady_clock::now();

    const auto gemm_start = std::chrono::steady_clock::now();
    status = rns8_gemm_rns(ctx, plan, a_matrix, b_matrix, c_matrix, workspace);
    if (status != RNS8_SUCCESS) fail_status("rns8_gemm_rns", status);
    const auto gemm_end = std::chrono::steady_clock::now();

    const auto export_start = std::chrono::steady_clock::now();
    status = rns8_export_i64(ctx, plan, c_matrix, C.data(), args.n);
    if (status != RNS8_SUCCESS) fail_status("rns8_export_i64", status);
    const auto export_end = std::chrono::steady_clock::now();

    pack_us += millis_since(pack_start, pack_end);
    gemm_us += millis_since(gemm_start, gemm_end);
    export_us += millis_since(export_start, export_end);
  }

  rns8_destroy_matrix(c_matrix);
  rns8_destroy_matrix(b_matrix);
  rns8_destroy_matrix(a_matrix);
  rns8_destroy_workspace(workspace);
  rns8_destroy_plan(plan);
  rns8_destroy_context(ctx);

  const double repeats = static_cast<double>(args.repeats);
  const double avg_pack_us = static_cast<double>(pack_us) / repeats;
  const double avg_gemm_us = static_cast<double>(gemm_us) / repeats;
  const double avg_export_us = static_cast<double>(export_us) / repeats;
  const double avg_total_us = avg_pack_us + avg_gemm_us + avg_export_us;
  std::cout << "{\n";
  std::cout << "  \"benchmark\": \"cpu_reference_bounded_i64\",\n";
  std::cout << "  \"backend\": \"cpu-reference\",\n";
  std::cout << "  \"semantics\": \"bounded_i64\",\n";
  std::cout << "  \"m\": " << args.m << ",\n";
  std::cout << "  \"n\": " << args.n << ",\n";
  std::cout << "  \"k\": " << args.k << ",\n";
  std::cout << "  \"prefix\": " << RNS8_DEFAULT_BOUNDED_PREFIX << ",\n";
  std::cout << "  \"seed\": " << args.seed << ",\n";
  std::cout << "  \"repeats\": " << args.repeats << ",\n";
  std::cout << "  \"timing_source\": \"std::chrono::steady_clock\",\n";
  std::cout << "  \"plan_us\": " << millis_since(plan_start, plan_end) << ",\n";
  std::cout << "  \"matrix_alloc_us\": " << millis_since(alloc_start, alloc_end) << ",\n";
  std::cout << "  \"avg_pack_us\": " << avg_pack_us << ",\n";
  std::cout << "  \"avg_per_modulus_gemm_us\": " << avg_gemm_us << ",\n";
  std::cout << "  \"avg_crt_export_us\": " << avg_export_us << ",\n";
  std::cout << "  \"avg_end_to_end_us\": " << avg_total_us << "\n";
  std::cout << "}\n";
  return 0;
}
