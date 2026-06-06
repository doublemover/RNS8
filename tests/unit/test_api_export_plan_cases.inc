namespace {

rns8_plan export_plan_fixture(rns8_backend_kind backend, rns8_semantics semantics) {
  rns8_plan plan{};
  plan.backend = backend;
  plan.desc.semantics = semantics;
  plan.desc.m = 16;
  plan.desc.n = 16;
  plan.desc.k = 16;
  plan.desc.bound_kind = RNS8_BOUND_GLOBAL_MAX_ABS;
  plan.desc.bound = 16;
  plan.prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
  return plan;
}

void add_export_plan_tile(rns8_plan& plan, bool zero_output) {
  rns8_plan_tile_schedule_entry entry{};
  entry.row_offset = 0;
  entry.col_offset = 0;
  entry.row_extent = plan.desc.m;
  entry.col_extent = plan.desc.n;
  entry.required_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
  entry.selected_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
  entry.flags = zero_output ? RNS8_TILE_SCHEDULE_ZERO_OUTPUT : 0;
  plan.tile_schedule.push_back(entry);
  plan.tile_bounds.push_back(plan.desc.bound);
}

}  // namespace

namespace rns8_api = rns8::detail::api;

TEST_CASE("internal export reconstruction plan exposes selector metadata") {
  auto cpu_i64 = export_plan_fixture(RNS8_BACKEND_CPU_REFERENCE, RNS8_BOUNDED_I64);
  const auto cpu_i64_export = rns8_api::make_export_plan(cpu_i64, RNS8_BOUNDED_I64);
  CHECK(cpu_i64_export.output_layout == rns8_api::export_output_layout::scalar_i64);
  CHECK(cpu_i64_export.status_policy == rns8_api::export_status_policy::range_checked_status_buffer);
  CHECK(cpu_i64_export.d2h_policy == rns8_api::export_d2h_policy::host_ld_padded);
  CHECK_FALSE(cpu_i64_export.requires_hip_tile_metadata);
  CHECK_FALSE(cpu_i64_export.all_zero_tiled_output);
  CHECK(std::string(cpu_i64_export.selected_export_kernel) == "cpu_reference_export_i64");
  CHECK(std::string(rns8_api::export_output_layout_name(cpu_i64_export.output_layout)) == "scalar_i64");
  CHECK(std::string(rns8_api::export_status_policy_name(cpu_i64_export.status_policy)) ==
        "range_checked_status_buffer");
  CHECK(std::string(rns8_api::export_d2h_policy_name(cpu_i64_export.d2h_policy)) == "host_ld_padded");

  auto vector_i64 = export_plan_fixture(RNS8_BACKEND_HIP_VECTOR_ALU_INT64, RNS8_BOUNDED_I64);
  const auto vector_i64_export = rns8_api::make_export_plan(vector_i64, RNS8_BOUNDED_I64);
  CHECK(vector_i64_export.output_layout == rns8_api::export_output_layout::scalar_i64);
  CHECK(std::string(vector_i64_export.selected_export_kernel) == "vector_alu_output_d2h");
  CHECK_FALSE(vector_i64_export.requires_hip_tile_metadata);

  auto hip_i64 = export_plan_fixture(RNS8_BACKEND_HIP_DIRECT, RNS8_BOUNDED_I64);
  add_export_plan_tile(hip_i64, false);
  const auto hip_i64_export = rns8_api::make_export_plan(hip_i64, RNS8_BOUNDED_I64);
  CHECK(hip_i64_export.output_layout == rns8_api::export_output_layout::scalar_i64);
  CHECK(std::string(hip_i64_export.selected_export_kernel) == "hip_direct_export_i64_tiled_device");
  CHECK(hip_i64_export.requires_hip_tile_metadata);
  CHECK_FALSE(hip_i64_export.all_zero_tiled_output);

  auto hip_zero_i64 = export_plan_fixture(RNS8_BACKEND_HIP_DIRECT, RNS8_BOUNDED_I64);
  add_export_plan_tile(hip_zero_i64, true);
  const auto hip_zero_i64_export = rns8_api::make_export_plan(hip_zero_i64, RNS8_BOUNDED_I64);
  CHECK(std::string(hip_zero_i64_export.selected_export_kernel) == "hip_direct_export_i64_tiled_device");
  CHECK_FALSE(hip_zero_i64_export.requires_hip_tile_metadata);
  CHECK(hip_zero_i64_export.all_zero_tiled_output);

  auto cpu_u64 = export_plan_fixture(RNS8_BACKEND_CPU_REFERENCE, RNS8_BOUNDED_U64);
  const auto cpu_u64_export = rns8_api::make_export_plan(cpu_u64, RNS8_BOUNDED_U64);
  CHECK(cpu_u64_export.output_layout == rns8_api::export_output_layout::scalar_u64);
  CHECK(std::string(cpu_u64_export.selected_export_kernel) == "cpu_reference_export_u64");

  auto finite_cpu = export_plan_fixture(RNS8_BACKEND_CPU_REFERENCE, RNS8_FINITE_FIELD_U8);
  const auto finite_cpu_export = rns8_api::make_export_plan(finite_cpu, RNS8_FINITE_FIELD_U8);
  CHECK(finite_cpu_export.output_layout == rns8_api::export_output_layout::finite_u8);
  CHECK(finite_cpu_export.status_policy == rns8_api::export_status_policy::none);
  CHECK(std::string(finite_cpu_export.selected_export_kernel) == "finite_reference_export_u8");
  CHECK(std::string(rns8_api::export_output_layout_name(finite_cpu_export.output_layout)) == "finite_u8");

  auto finite_hip = export_plan_fixture(RNS8_BACKEND_HIP_DIRECT, RNS8_FINITE_RING_U8);
  const auto finite_hip_export = rns8_api::make_export_plan(finite_hip, RNS8_FINITE_RING_U8);
  CHECK(finite_hip_export.output_layout == rns8_api::export_output_layout::finite_u8);
  CHECK(std::string(finite_hip_export.selected_export_kernel) == "hip_direct_export_finite_u8_device");

  auto wrap_cpu = export_plan_fixture(RNS8_BACKEND_WRAP64_BYTE_LIMB, RNS8_WRAP_U64_MOD_2_64);
  const auto wrap_cpu_export = rns8_api::make_export_plan(wrap_cpu, RNS8_WRAP_U64_MOD_2_64);
  CHECK(wrap_cpu_export.output_layout == rns8_api::export_output_layout::scalar_u64);
  CHECK(wrap_cpu_export.status_policy == rns8_api::export_status_policy::none);
  CHECK(std::string(wrap_cpu_export.selected_export_kernel) == "wrap64_reference_export_u64");

  auto wrap_hip = export_plan_fixture(RNS8_BACKEND_HIP_DIRECT, RNS8_WRAP_U64_MOD_2_64);
  const auto wrap_hip_export = rns8_api::make_export_plan(wrap_hip, RNS8_WRAP_U64_MOD_2_64);
  CHECK(wrap_hip_export.output_layout == rns8_api::export_output_layout::scalar_u64);
  CHECK(std::string(wrap_hip_export.selected_export_kernel) == "wrap64_hip_export_u64_device");

  auto exact_signed_cpu = export_plan_fixture(RNS8_BACKEND_CPU_REFERENCE, RNS8_EXACT_WIDE_SIGNED);
  const auto exact_signed_cpu_export =
      rns8_api::make_export_plan(exact_signed_cpu, RNS8_EXACT_WIDE_SIGNED, 3);
  CHECK(exact_signed_cpu_export.output_layout == rns8_api::export_output_layout::fixed_u64_limbs);
  CHECK(exact_signed_cpu_export.status_policy == rns8_api::export_status_policy::range_checked_status_buffer);
  CHECK(exact_signed_cpu_export.limb_count == 3);
  CHECK(std::string(exact_signed_cpu_export.selected_export_kernel) ==
        "cpu_reference_export_exact_wide_signed_limbs");
  CHECK(std::string(rns8_api::export_output_layout_name(exact_signed_cpu_export.output_layout)) ==
        "fixed_u64_limbs");

  auto exact_unsigned_hip = export_plan_fixture(RNS8_BACKEND_HIP_DIRECT, RNS8_EXACT_WIDE_UNSIGNED);
  const auto exact_unsigned_hip_export =
      rns8_api::make_export_plan(exact_unsigned_hip, RNS8_EXACT_WIDE_UNSIGNED, 4);
  CHECK(exact_unsigned_hip_export.output_layout == rns8_api::export_output_layout::fixed_u64_limbs);
  CHECK(exact_unsigned_hip_export.status_policy == rns8_api::export_status_policy::range_checked_status_buffer);
  CHECK(exact_unsigned_hip_export.limb_count == 4);
  CHECK(std::string(exact_unsigned_hip_export.selected_export_kernel) ==
        "hip_direct_export_exact_wide_unsigned_limbs_device");
}
