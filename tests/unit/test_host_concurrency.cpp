#include <catch2/catch_test_macros.hpp>
#include "core/internal.hpp"
#include <array>
#include "../support/thread_start_gate.hpp"
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
void require(rns8_status status) {
  if (status != RNS8_SUCCESS)
    throw std::runtime_error(rns8_status_string(status));
}
struct CpuCall {
  rns8_context* context = nullptr;
  rns8_plan* plan = nullptr;
  rns8_workspace* workspace = nullptr;
  rns8_matrix *a = nullptr, *b = nullptr, *c = nullptr;
  rns8_status close() {
    for (auto** matrix : {&c, &b, &a})
      if (*matrix) {
        const auto status = rns8_destroy_matrix(*matrix);
        if (status != RNS8_SUCCESS)
          return status;
        *matrix = nullptr;
      }
    if (workspace) {
      const auto status = rns8_destroy_workspace(workspace);
      if (status != RNS8_SUCCESS)
        return status;
      workspace = nullptr;
    }
    if (plan) {
      const auto status = rns8_destroy_plan(plan);
      if (status != RNS8_SUCCESS)
        return status;
      plan = nullptr;
    }
    if (context) {
      const auto status = rns8_destroy_context(context);
      if (status != RNS8_SUCCESS)
        return status;
      context = nullptr;
    }
    return RNS8_SUCCESS;
  }
  ~CpuCall() { (void)close(); }
  void initialize() {
    rns8_context_options options{};
    options.struct_size = sizeof(options);
    options.abi_version = RNS8_ABI_VERSION;
    options.requested_backend = RNS8_BACKEND_CPU_REFERENCE;
    require(rns8_create_context(-1, &options, &context));
    rns8_gemm_desc desc{};
    desc.struct_size = sizeof(desc);
    desc.abi_version = RNS8_ABI_VERSION;
    desc.semantics = RNS8_EXACT_WIDE_SIGNED;
    desc.bound_kind = RNS8_BOUND_NONE;
    desc.requested_backend = RNS8_BACKEND_CPU_REFERENCE;
    desc.m = desc.n = desc.k = 2;
    desc.max_prefix = 20;
    desc.flags = RNS8_PLAN_FORCE_FIXED_PREFIX | RNS8_PLAN_EXACT_VALUE_BOUND;
    require(rns8_create_plan(context, &desc, &plan));
    require(rns8_create_workspace(context, plan, &workspace));
    rns8_matrix_desc md{};
    md.struct_size = sizeof(md);
    md.abi_version = RNS8_ABI_VERSION;
    md.semantics = desc.semantics;
    md.bound_kind = RNS8_BOUND_NONE;
    md.rows = md.cols = md.logical_ld = 2;
    md.max_prefix = 20;
    md.logical_layout = RNS8_LAYOUT_ROW_MAJOR;
    for (auto** matrix : {&a, &b, &c})
      require(rns8_create_matrix(context, &md, matrix));
  }
  void execute(int64_t value) {
    const int64_t av[]{value, -2, 3, 4 + value}, bv[]{5, 6, 7, 8};
    require(rns8_pack_i64(context, a, av, 2, 0));
    require(rns8_pack_i64(context, b, bv, 2, 0));
    const auto before = c->content_generation;
    require(rns8_gemm_rns(context, plan, a, b, c, workspace));
    if (c->content_generation <= before || !c->host_residues_current)
      throw std::runtime_error("concurrent public product did not publish a new host generation");
    uint64_t output[12]{};
    require(rns8_export_exact_wide_signed_limbs(context, plan, c, output, 2, 3));
    using boost::multiprecision::cpp_int;
    for (unsigned r = 0; r < 2; ++r)
      for (unsigned col = 0; col < 2; ++col) {
        const unsigned cell = (r * 2 + col) * 3;
        cpp_int observed = cpp_int(output[cell]) + (cpp_int(output[cell + 1]) << 64) +
                           (cpp_int(output[cell + 2]) << 128);
        if (output[cell + 2] >> 63)
          observed -= cpp_int(1) << 192;
        const cpp_int expected =
            cpp_int(av[r * 2]) * bv[col] + cpp_int(av[r * 2 + 1]) * bv[2 + col];
        if (observed != expected)
          throw std::runtime_error("concurrent public exact value mismatch");
      }
  }
};
} // namespace

TEST_CASE("independent CPU owners overlap with exact results and unique matrix identities",
          "[cpu][concurrency][sanitizer]") {
  rns8::test::ThreadStartGate gate(4);
  std::array<std::thread, 4> threads;
  std::array<std::string, 4> errors;
  std::array<std::vector<uint64_t>, 4> identities;
  for (unsigned i = 0; i < threads.size(); ++i)
    threads[i] = std::thread([&, i] {
      gate.wait();
      try {
        for (unsigned round = 0; round < 8; ++round) {
          CpuCall owner;
          owner.initialize();
          for (const auto* matrix : {owner.a, owner.b, owner.c})
            identities[i].push_back(matrix->matrix_instance_id);
          owner.execute(1 + int64_t(i + round));
          owner.execute(-3 - int64_t(i + round));
          if (owner.close() != RNS8_SUCCESS)
            throw std::runtime_error("independent CPU cleanup failed");
        }
      } catch (const std::exception& error) {
        errors[i] = error.what();
      }
    });
  gate.release();
  for (auto& thread : threads)
    thread.join();
  std::set<uint64_t> unique;
  for (unsigned i = 0; i < threads.size(); ++i) {
    INFO(errors[i]);
    REQUIRE(errors[i].empty());
    REQUIRE(identities[i].size() == 24);
    for (const auto identity : identities[i]) {
      CHECK(identity != 0);
      CHECK(unique.insert(identity).second);
    }
  }
}

TEST_CASE("shared CPU handles obey their external serialization contract across host threads",
          "[cpu][concurrency][sanitizer]") {
  CpuCall owner;
  owner.initialize();
  rns8::test::ThreadStartGate gate(4);
  std::mutex external;
  std::array<std::thread, 4> threads;
  std::array<std::string, 4> errors;
  for (unsigned i = 0; i < threads.size(); ++i)
    threads[i] = std::thread([&, i] {
      gate.wait();
      try {
        for (unsigned round = 0; round < 8; ++round) {
          std::lock_guard<std::mutex> lock(external);
          owner.execute(1 + int64_t(i + round));
        }
      } catch (const std::exception& error) {
        errors[i] = error.what();
      }
    });
  gate.release();
  for (auto& thread : threads)
    thread.join();
  for (const auto& error : errors) {
    INFO(error);
    REQUIRE(error.empty());
  }
  CHECK(owner.close() == RNS8_SUCCESS);
}
