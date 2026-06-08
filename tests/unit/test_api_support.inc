
rns8_context* create_cpu_context() {
  rns8_context_options options{};
  options.struct_size = sizeof(options);
  options.abi_version = RNS8_ABI_VERSION;
  options.requested_backend = RNS8_BACKEND_CPU_REFERENCE;
  rns8_context* ctx = nullptr;
  REQUIRE(rns8_create_context(-1, &options, &ctx) == RNS8_SUCCESS);
  return ctx;
}

rns8_context* create_wrap_context() {
  rns8_context_options options{};
  options.struct_size = sizeof(options);
  options.abi_version = RNS8_ABI_VERSION;
  options.requested_backend = RNS8_BACKEND_WRAP64_BYTE_LIMB;
  rns8_context* ctx = nullptr;
  REQUIRE(rns8_create_context(-1, &options, &ctx) == RNS8_SUCCESS);
  return ctx;
}

rns8::detail::PlanLoweringDescription lowering_for_plan(rns8_plan* plan) {
  rns8_plan_backend_info backend{};
  backend.struct_size = sizeof(backend);
  backend.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_plan_backend_info(plan, &backend) == RNS8_SUCCESS);

  rns8_plan_packing_info packing{};
  packing.struct_size = sizeof(packing);
  packing.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_plan_packing_info(plan, &packing) == RNS8_SUCCESS);

  rns8_plan_schedule_info schedule{};
  schedule.struct_size = sizeof(schedule);
  schedule.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_plan_schedule_info(plan, &schedule) == RNS8_SUCCESS);

  return rns8::detail::describe_plan_lowering(backend, packing, schedule);
}

void set_autotune_cache_path_for_test(const std::filesystem::path& path) {
#if defined(_WIN32)
  _putenv_s("RNS8_AUTOTUNE_CACHE_PATH", path.string().c_str());
#else
  setenv("RNS8_AUTOTUNE_CACHE_PATH", path.string().c_str(), 1);
#endif
}

void clear_autotune_cache_path_for_test() {
#if defined(_WIN32)
  _putenv_s("RNS8_AUTOTUNE_CACHE_PATH", "");
#else
  unsetenv("RNS8_AUTOTUNE_CACHE_PATH");
#endif
}

struct ScopedAutotuneCachePath {
  explicit ScopedAutotuneCachePath(const std::filesystem::path& path) { set_autotune_cache_path_for_test(path); }
  ~ScopedAutotuneCachePath() { clear_autotune_cache_path_for_test(); }
};

std::filesystem::path unique_cache_fixture_path(const char* stem) {
  const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
  return std::filesystem::temp_directory_path() / (std::string(stem) + "-" + std::to_string(tick) + ".json");
}

std::string reviewed_release_status_for_target(const char* target_id) {
  const std::string target = target_id ? target_id : "";
  if (target == "gfx1100") {
    return "reviewed_release_same_contract_fastest_windows_gfx1100";
  }
  if (target == "gfx90a" || target == "gfx942" || target == "gfx950") {
    return "reviewed_release_same_contract_fastest_linux_" + target;
  }
  if (target.rfind("gfx", 0) == 0) {
    return "reviewed_release_same_contract_fastest_target_" + target;
  }
  return "reviewed_release_unknown_target";
}

rns8_gemm_desc exact_desc(rns8_semantics semantics, int64_t m, int64_t n, int64_t k) {
  rns8_gemm_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.semantics = semantics;
  desc.bound_kind = RNS8_BOUND_NONE;
  desc.requested_backend = RNS8_BACKEND_CPU_REFERENCE;
  desc.m = m;
  desc.n = n;
  desc.k = k;
  desc.bound = 0;
  desc.max_prefix = RNS8_MAX_SUPPORTED_PREFIX;
  return desc;
}

rns8_gemm_desc bounded_gemm_desc(rns8_semantics semantics, int64_t m, int64_t n, int64_t k) {
  rns8_gemm_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.semantics = semantics;
  desc.bound_kind = semantics == RNS8_BOUNDED_I64 ? RNS8_BOUND_GLOBAL_MAX_ABS : RNS8_BOUND_GLOBAL_MAX_UNSIGNED;
  desc.requested_backend = RNS8_BACKEND_CPU_REFERENCE;
  desc.m = m;
  desc.n = n;
  desc.k = k;
  desc.bound = static_cast<uint64_t>(k > 0 ? k : 0);
  desc.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
  return desc;
}

rns8_gemm_desc finite_gemm_desc(rns8_semantics semantics, int64_t m, int64_t n, int64_t k, uint32_t modulus) {
  rns8_gemm_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.semantics = semantics;
  desc.bound_kind = RNS8_BOUND_NONE;
  desc.requested_backend = RNS8_BACKEND_CPU_REFERENCE;
  desc.m = m;
  desc.n = n;
  desc.k = k;
  desc.finite_modulus = modulus;
  return desc;
}

rns8_gemm_desc wrap_gemm_desc(int64_t m, int64_t n, int64_t k) {
  rns8_gemm_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.semantics = RNS8_WRAP_U64_MOD_2_64;
  desc.bound_kind = RNS8_BOUND_NONE;
  desc.requested_backend = RNS8_BACKEND_WRAP64_BYTE_LIMB;
  desc.m = m;
  desc.n = n;
  desc.k = k;
  return desc;
}

rns8_matrix_desc exact_matrix_desc(int64_t rows, int64_t cols, rns8_semantics semantics) {
  rns8_matrix_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.rows = rows;
  desc.cols = cols;
  desc.logical_ld = cols;
  desc.semantics = semantics;
  desc.logical_layout = RNS8_LAYOUT_ROW_MAJOR;
  desc.bound_kind = RNS8_BOUND_NONE;
  desc.max_prefix = RNS8_MAX_SUPPORTED_PREFIX;
  return desc;
}

rns8_matrix_desc bounded_matrix_desc(int64_t rows, int64_t cols, rns8_semantics semantics) {
  rns8_matrix_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.rows = rows;
  desc.cols = cols;
  desc.logical_ld = cols;
  desc.semantics = semantics;
  desc.logical_layout = RNS8_LAYOUT_ROW_MAJOR;
  desc.bound_kind = semantics == RNS8_BOUNDED_I64 ? RNS8_BOUND_GLOBAL_MAX_ABS : RNS8_BOUND_GLOBAL_MAX_UNSIGNED;
  desc.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
  return desc;
}

rns8_matrix_desc finite_matrix_desc(int64_t rows, int64_t cols, rns8_semantics semantics) {
  rns8_matrix_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.rows = rows;
  desc.cols = cols;
  desc.logical_ld = cols;
  desc.semantics = semantics;
  desc.logical_layout = RNS8_LAYOUT_ROW_MAJOR;
  desc.bound_kind = RNS8_BOUND_NONE;
  desc.max_prefix = 0;
  return desc;
}

rns8_matrix_desc wrap_matrix_desc(int64_t rows, int64_t cols) {
  rns8_matrix_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.rows = rows;
  desc.cols = cols;
  desc.logical_ld = cols;
  desc.semantics = RNS8_WRAP_U64_MOD_2_64;
  desc.logical_layout = RNS8_LAYOUT_ROW_MAJOR;
  desc.bound_kind = RNS8_BOUND_NONE;
  desc.max_prefix = 0;
  return desc;
}

rns8_matrix_storage_info matrix_storage_info(const rns8_matrix* matrix) {
  rns8_matrix_storage_info info{};
  info.struct_size = sizeof(info);
  info.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_matrix_storage_info(matrix, &info) == RNS8_SUCCESS);
  return info;
}

rns8_resident_lifetime_info resident_lifetime_info(
    const rns8_matrix* matrix,
    const rns8_plan* plan,
    const rns8_workspace* workspace,
    rns8_resident_matrix_role role) {
  rns8_resident_lifetime_info info{};
  info.struct_size = sizeof(info);
  info.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_resident_lifetime_info(matrix, plan, workspace, role, &info) == RNS8_SUCCESS);
  return info;
}

rns8_prepack_cache_key_info prepack_cache_key_info(
    const rns8_plan* plan,
    const rns8_matrix* matrix,
    rns8_operand_role operand_role) {
  rns8_prepack_cache_key_info info{};
  info.struct_size = sizeof(info);
  info.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_prepack_cache_key_info(plan, matrix, operand_role, &info) == RNS8_SUCCESS);
  return info;
}

