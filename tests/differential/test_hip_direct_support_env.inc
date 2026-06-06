
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
extern "C" int rns8_hip_direct_ring_gemm_i8_device(
    const int8_t* d_a,
    const int8_t* d_b,
    int8_t* d_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int modulus,
    uint32_t modulus_reciprocal,
    int modulus_index,
    int selected_prefix,
    int safe_k_block);
#endif

uint32_t reciprocal_for_modulus(uint16_t modulus) {
  return static_cast<uint32_t>((uint64_t{1} << 32u) / static_cast<uint32_t>(modulus));
}

bool hip_available() {
  if (!rns8::detail::hip_direct_compiled()) {
    return false;
  }
  rns8_device_info info{};
  info.struct_size = sizeof(info);
  info.abi_version = RNS8_ABI_VERSION;
  return rns8::detail::hip_direct_probe(0, info) == RNS8_SUCCESS;
}

std::string env_value_for_test(const char* name, bool* present) {
  if (present) {
    *present = false;
  }
#ifdef _WIN32
  char* buffer = nullptr;
  std::size_t length = 0;
  if (_dupenv_s(&buffer, &length, name) != 0 || !buffer) {
    return {};
  }
  std::string value(buffer, length > 0 ? length - 1 : 0);
  std::free(buffer);
  if (present) {
    *present = true;
  }
  return value;
#else
  const char* value = std::getenv(name);
  if (!value) {
    return {};
  }
  if (present) {
    *present = true;
  }
  return value;
#endif
}

void set_env_for_test(const char* name, const char* value) {
#ifdef _WIN32
  _putenv_s(name, value);
#else
  setenv(name, value, 1);
#endif
}

void clear_env_for_test(const char* name) {
#ifdef _WIN32
  _putenv_s(name, "");
#else
  unsetenv(name);
#endif
}

struct ScopedEnvVar {
  explicit ScopedEnvVar(const char* env_name, const char* value) : name(env_name) {
    original = env_value_for_test(name.c_str(), &had_original);
    set_env_for_test(name.c_str(), value);
  }

  ~ScopedEnvVar() {
    if (had_original) {
      set_env_for_test(name.c_str(), original.c_str());
    } else {
      clear_env_for_test(name.c_str());
    }
  }

  std::string name;
  std::string original;
  bool had_original = false;
};

rns8_context* create_context(rns8_backend_kind backend) {
  rns8_context_options options{};
  options.struct_size = sizeof(options);
  options.abi_version = RNS8_ABI_VERSION;
  options.requested_backend = backend;
  rns8_context* ctx = nullptr;
  REQUIRE(rns8_create_context(backend == RNS8_BACKEND_HIP_DIRECT ? 0 : -1, &options, &ctx) == RNS8_SUCCESS);
  return ctx;
}

rns8_gemm_desc signed_desc(int64_t m, int64_t n, int64_t k, uint64_t bound, rns8_backend_kind backend) {
  rns8_gemm_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.semantics = RNS8_BOUNDED_I64;
  desc.bound_kind = RNS8_BOUND_GLOBAL_MAX_ABS;
  desc.requested_backend = backend;
  desc.m = m;
  desc.n = n;
  desc.k = k;
  desc.bound = bound;
  return desc;
}

rns8_gemm_desc unsigned_desc(int64_t m, int64_t n, int64_t k, uint64_t bound, rns8_backend_kind backend) {
  rns8_gemm_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.semantics = RNS8_BOUNDED_U64;
  desc.bound_kind = RNS8_BOUND_GLOBAL_MAX_UNSIGNED;
  desc.requested_backend = backend;
  desc.m = m;
  desc.n = n;
  desc.k = k;
  desc.bound = bound;
  return desc;
}

rns8_gemm_desc per_tile_signed_desc(
    int64_t m,
    int64_t n,
    int64_t k,
    const std::vector<uint64_t>& bounds,
    rns8_backend_kind backend) {
  rns8_gemm_desc desc = signed_desc(m, n, k, 0, backend);
  desc.bound_kind = RNS8_BOUND_PER_TILE_MAX_ABS;
  desc.tile_m = 64;
  desc.tile_n = 64;
  desc.tile_bounds = bounds.data();
  desc.tile_bounds_count = bounds.size();
  return desc;
}

rns8_gemm_desc per_tile_unsigned_desc(
    int64_t m,
    int64_t n,
    int64_t k,
    const std::vector<uint64_t>& bounds,
    rns8_backend_kind backend) {
  rns8_gemm_desc desc = unsigned_desc(m, n, k, 0, backend);
  desc.bound_kind = RNS8_BOUND_PER_TILE_MAX_UNSIGNED;
  desc.tile_m = 64;
  desc.tile_n = 64;
  desc.tile_bounds = bounds.data();
  desc.tile_bounds_count = bounds.size();
  return desc;
}

rns8_gemm_desc exact_signed_desc(int64_t m, int64_t n, int64_t k, rns8_backend_kind backend) {
  rns8_gemm_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.semantics = RNS8_EXACT_WIDE_SIGNED;
  desc.bound_kind = RNS8_BOUND_NONE;
  desc.requested_backend = backend;
  desc.m = m;
  desc.n = n;
  desc.k = k;
  desc.max_prefix = RNS8_MAX_SUPPORTED_PREFIX;
  return desc;
}

rns8_gemm_desc exact_unsigned_desc(int64_t m, int64_t n, int64_t k, rns8_backend_kind backend) {
  rns8_gemm_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.semantics = RNS8_EXACT_WIDE_UNSIGNED;
  desc.bound_kind = RNS8_BOUND_NONE;
  desc.requested_backend = backend;
  desc.m = m;
  desc.n = n;
  desc.k = k;
  desc.max_prefix = RNS8_MAX_SUPPORTED_PREFIX;
  return desc;
}

rns8_gemm_desc wrap_desc(int64_t m, int64_t n, int64_t k, rns8_backend_kind backend) {
  rns8_gemm_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.semantics = RNS8_WRAP_U64_MOD_2_64;
  desc.bound_kind = RNS8_BOUND_NONE;
  desc.requested_backend = backend;
  desc.m = m;
  desc.n = n;
  desc.k = k;
  return desc;
}

rns8_gemm_desc finite_desc(
    int64_t m,
    int64_t n,
    int64_t k,
    rns8_semantics semantics,
    rns8_backend_kind backend,
    uint16_t modulus = 0) {
  rns8_gemm_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.semantics = semantics;
  desc.bound_kind = RNS8_BOUND_NONE;
  desc.requested_backend = backend;
  desc.m = m;
  desc.n = n;
  desc.k = k;
  desc.max_prefix = 0;
  desc.finite_modulus =
      modulus != 0 ? modulus : (semantics == RNS8_FINITE_FIELD_U8 ? uint16_t{251} : uint16_t{255});
  return desc;
}

rns8_matrix_desc matrix_desc(int64_t rows, int64_t cols, rns8_semantics semantics, rns8_bound_kind bound_kind) {
  rns8_matrix_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.rows = rows;
  desc.cols = cols;
  desc.logical_ld = cols;
  desc.semantics = semantics;
  desc.logical_layout = RNS8_LAYOUT_ROW_MAJOR;
  desc.bound_kind = bound_kind;
  desc.tile_m = 128;
  desc.tile_n = 128;
  if (semantics == RNS8_EXACT_WIDE_SIGNED || semantics == RNS8_EXACT_WIDE_UNSIGNED) {
    desc.max_prefix = RNS8_MAX_SUPPORTED_PREFIX;
  } else if (semantics == RNS8_WRAP_U64_MOD_2_64 || semantics == RNS8_FINITE_RING_U8 ||
             semantics == RNS8_FINITE_FIELD_U8) {
    desc.max_prefix = 0;
  } else {
    desc.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
  }
  return desc;
}

