#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <rns8/rns8.hpp>
#include <stdexcept>

#include "noninteractive_errors.hpp"

namespace {
using Limbs = std::array<uint64_t, 3>;  // Least-significant 64-bit limb first.

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

rns8_gemm_desc scalar_gemm(rns8_semantics semantics, uint32_t ceiling) {
  rns8_gemm_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.semantics = semantics;
  desc.bound_kind = RNS8_BOUND_NONE;
  desc.requested_backend = RNS8_BACKEND_CPU_REFERENCE;
  desc.m = desc.n = desc.k = 1;
  desc.max_prefix = ceiling;
  return desc;
}

rns8_matrix_desc scalar_matrix(rns8_semantics semantics, uint32_t ceiling) {
  rns8_matrix_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.semantics = semantics;
  desc.bound_kind = RNS8_BOUND_NONE;
  desc.rows = desc.cols = desc.logical_ld = 1;
  desc.logical_layout = RNS8_LAYOUT_ROW_MAJOR;
  desc.max_prefix = ceiling;
  return desc;
}

rns8_status export_status(rns8::Context& ctx, const rns8::Plan& plan, const rns8::Matrix& matrix,
                          rns8_semantics semantics, Limbs& limbs) {
  return semantics == RNS8_EXACT_WIDE_SIGNED
             ? rns8_export_exact_wide_signed_limbs(ctx.get(), plan.get(), matrix.get(), limbs.data(), 1, 3)
             : rns8_export_exact_wide_unsigned_limbs(ctx.get(), plan.get(), matrix.get(), limbs.data(), 1, 3);
}

Limbs export_limbs(rns8::Context& ctx, const rns8::Plan& plan, const rns8::Matrix& matrix,
                   rns8_semantics semantics) {
  Limbs result{};
  rns8::check(export_status(ctx, plan, matrix, semantics, result));
  return result;
}

void check_storage(const rns8::Matrix& matrix, const rns8_matrix_storage_info& before) {
  const auto after = matrix.storage_info();
  require(after.matrix_instance_id == before.matrix_instance_id &&
              after.source_version == before.source_version && after.max_prefix == before.max_prefix &&
              after.host_residue_bytes == before.host_residue_bytes &&
              after.host_native_bytes == before.host_native_bytes &&
              after.host_residues_current == before.host_residues_current &&
              after.host_native_current == before.host_native_current,
          "resident identity/version/storage/currentness changed after budget rejection");
}

void continuation(rns8_semantics semantics) {
  constexpr uint32_t ceiling = RNS8_MAX_SUPPORTED_PREFIX;
  const bool signed_values = semantics == RNS8_EXACT_WIDE_SIGNED;
  const uint32_t expected_prefix = signed_values ? 19 : 20;  // Current default modulus ladder.
  rns8::Context ctx(-1, RNS8_BACKEND_CPU_REFERENCE);
  auto desc = scalar_gemm(semantics, ceiling);
  rns8::Plan plan(ctx, desc);
  rns8::Workspace workspace(ctx, plan);
  const uint32_t floor = plan.schedule_info().min_selected_prefix;
  require(floor < expected_prefix && expected_prefix <= ceiling, "example needs room above the plan floor");

  const auto matrix_desc = scalar_matrix(semantics, ceiling);
  rns8::Matrix native(ctx, matrix_desc), scale(ctx, matrix_desc);
  rns8::Matrix a(ctx, matrix_desc), b(ctx, matrix_desc), c(ctx, matrix_desc);
  if (signed_values) {
    const int64_t input = INT64_MIN, multiplier = 1024, sentinel = 42;
    rns8::check(rns8_pack_i64(ctx.get(), native.get(), &input, 1, 11));
    rns8::check(rns8_pack_i64(ctx.get(), scale.get(), &multiplier, 1, 12));
    rns8::check(rns8_pack_i64(ctx.get(), c.get(), &sentinel, 1, 99));
  } else {
    const uint64_t input = UINT64_MAX, multiplier = 1024, sentinel = 42;
    rns8::check(rns8_pack_u64(ctx.get(), native.get(), &input, 1, 11));
    rns8::check(rns8_pack_u64(ctx.get(), scale.get(), &multiplier, 1, 12));
    rns8::check(rns8_pack_u64(ctx.get(), c.get(), &sentinel, 1, 99));
  }
  // Public GEMM creates values wider than 64 bits, initialized at the plan floor.
  rns8::check(rns8_gemm_rns(ctx.get(), plan.get(), native.get(), scale.get(), a.get(), workspace.get()));
  rns8::check(rns8_gemm_rns(ctx.get(), plan.get(), native.get(), scale.get(), b.get(), workspace.get()));
  const Limbs wide_value = signed_values ? Limbs{0, UINT64_C(0xfffffffffffffe00), UINT64_MAX}
                                         : Limbs{UINT64_C(0xfffffffffffffc00), 1023, 0};
  const Limbs expected_product = signed_values
                                     ? Limbs{0, 0, UINT64_C(1) << 18}  // (-2^73)^2 = 2^146.
                                     : Limbs{UINT64_C(1) << 20, UINT64_C(0xffffffffffe00000), 1048575};
  require(export_limbs(ctx, plan, a, semantics) == wide_value &&
              export_limbs(ctx, plan, b, semantics) == wide_value,
          "native-to-wide input mismatch");
  const auto before_a = a.storage_info(), before_b = b.storage_info(), before_c = c.storage_info();

  // Ordinary GEMM preserves its original range contract; continuation is opt-in.
  require(
      rns8_gemm_rns(ctx.get(), plan.get(), a.get(), b.get(), c.get(), workspace.get()) == RNS8_RANGE_ERROR,
      "ordinary GEMM unexpectedly admitted the wider product");

  // Two distinct scalar input suffixes plus every selected output plane.
  // General payload: sum(input_cells * missing_planes) + output_cells * selected.
  // Range summaries, CRT scratch and row scratch are outside this payload limit.
  const uint64_t budget = 2 * uint64_t(expected_prefix - floor) + expected_prefix;
  for (unsigned repeat = 0; repeat < 2; ++repeat) {
    uint32_t selected = 777;
    const auto status = rns8_gemm_exact_wide_cpu_auto(ctx.get(), plan.get(), a.get(), b.get(), c.get(),
                                                      workspace.get(), budget - 1, &selected);
    require(status == RNS8_WORKSPACE_TOO_SMALL && selected == 777,
            "budget error or selected-prefix rollback");
    require(export_limbs(ctx, plan, a, semantics) == wide_value &&
                export_limbs(ctx, plan, b, semantics) == wide_value &&
                export_limbs(ctx, plan, c, semantics) == Limbs{42, 0, 0},
            "input/output values changed after rejected transaction");
    check_storage(a, before_a);
    check_storage(b, before_b);
    check_storage(c, before_c);
  }

  const uint32_t selected = rns8::gemm_exact_wide_cpu_auto(ctx, plan, a, b, c, workspace, budget);
  require(selected == expected_prefix, "unexpected selected prefix");
  require(a.storage_info().source_version == before_a.source_version &&
              b.storage_info().source_version == before_b.source_version,
          "lifting changed input versions");

  // The original plan was not rewritten. Inadequate-prefix export rejects and
  // preserves the caller's buffer; use a matching fixed-prefix export plan.
  Limbs rejected{999, 999, 999};
  require(export_status(ctx, plan, c, semantics, rejected) == RNS8_RANGE_ERROR &&
              rejected == Limbs{999, 999, 999},
          "inadequate original export did not preserve its destination");
  auto export_desc = desc;
  export_desc.max_prefix = selected;
  export_desc.flags = RNS8_PLAN_FORCE_FIXED_PREFIX;
  rns8::Plan exporter(ctx, export_desc);
  require(export_limbs(ctx, exporter, c, semantics) == expected_product, "wide product limb mismatch");
  require(export_limbs(ctx, exporter, a, semantics) == wide_value &&
              export_limbs(ctx, exporter, b, semantics) == wide_value,
          "lifting changed input integers");

  // Input suffixes are now initialized: a repeated call needs only output bytes.
  const uint64_t reuse_budget = selected;
  require(rns8::gemm_exact_wide_cpu_auto(ctx, plan, a, b, c, workspace, reuse_budget) == selected &&
              export_limbs(ctx, exporter, c, semantics) == expected_product &&
              plan.schedule_info().min_selected_prefix == floor,
          "resident reuse or original plan preservation failed");
  std::cout << (signed_values ? "signed" : "unsigned") << ": floor=" << floor << " selected=" << selected
            << " budget=" << budget << " rejected_budget=" << budget - 1 << " reuse_budget=" << reuse_budget
            << " rollback=PASS original_export=RANGE_ERROR limbs_le64=0x" << std::hex << expected_product[0]
            << ",0x" << expected_product[1] << ",0x" << expected_product[2] << std::dec << '\n';
}
}  // namespace

int main() {
  rns8_example::configure_noninteractive_errors();
  try {
    continuation(RNS8_EXACT_WIDE_SIGNED);
    continuation(RNS8_EXACT_WIDE_UNSIGNED);
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "CPU exact-wide continuation failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
