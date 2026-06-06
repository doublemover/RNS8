bool parse_backend(const std::string& value, rns8_backend_kind& out) {
  if (value == "auto") {
    out = RNS8_BACKEND_AUTO;
    return true;
  }
  if (value == "cpu" || value == "cpu-reference") {
    out = RNS8_BACKEND_CPU_REFERENCE;
    return true;
  }
  if (value == "hip-direct") {
    out = RNS8_BACKEND_HIP_DIRECT;
    return true;
  }
  if (value == "hip-vector-alu-int64" || value == "vector-alu-int64") {
    out = RNS8_BACKEND_HIP_VECTOR_ALU_INT64;
    return true;
  }
  if (value == "wrap64-byte-limb") {
    out = RNS8_BACKEND_WRAP64_BYTE_LIMB;
    return true;
  }
  if (value == "hipblaslt") {
    out = RNS8_BACKEND_HIPBLASLT;
    return true;
  }
  if (value == "ck") {
    out = RNS8_BACKEND_CK;
    return true;
  }
  if (value == "rocwmma") {
    out = RNS8_BACKEND_ROCWMMA;
    return true;
  }
  return false;
}

bool evidence_only_accelerator_backend(rns8_backend_kind backend) {
  return backend == RNS8_BACKEND_HIPBLASLT || backend == RNS8_BACKEND_CK || backend == RNS8_BACKEND_ROCWMMA;
}

void print_usage(std::ostream& out) {
  out << "usage: rns8-inspect [--backend auto|cpu-reference|hip-direct|hip-vector-alu-int64|wrap64-byte-limb|hipblaslt|ck|rocwmma]"
      << " [--device N] [--json] [--autotune-key KEY] [--show-autotune-cache] [--selector-shadow]\n";
}

const char* backend_name(rns8_backend_kind backend) {
  switch (backend) {
    case RNS8_BACKEND_AUTO:
      return "auto";
    case RNS8_BACKEND_CPU_REFERENCE:
      return "cpu-reference";
    case RNS8_BACKEND_HIP_DIRECT:
      return "hip-direct";
    case RNS8_BACKEND_HIP_VECTOR_ALU_INT64:
      return "hip-vector-alu-int64";
    case RNS8_BACKEND_HIPBLASLT:
      return "hipblaslt";
    case RNS8_BACKEND_CK:
      return "ck";
    case RNS8_BACKEND_ROCWMMA:
      return "rocwmma";
    case RNS8_BACKEND_WRAP64_BYTE_LIMB:
      return "wrap64-byte-limb";
  }
  return "unknown";
}

const char* semantics_name(rns8_semantics semantics) {
  switch (semantics) {
    case RNS8_BOUNDED_I64:
      return "bounded_i64";
    case RNS8_BOUNDED_U64:
      return "bounded_u64";
    case RNS8_EXACT_WIDE_SIGNED:
      return "exact_wide_signed";
    case RNS8_EXACT_WIDE_UNSIGNED:
      return "exact_wide_unsigned";
    case RNS8_WRAP_U64_MOD_2_64:
      return "wrap_u64_mod_2_64";
    case RNS8_FINITE_RING_U8:
      return "finite_ring_u8";
    case RNS8_FINITE_FIELD_U8:
      return "finite_field_u8";
  }
  return "unknown";
}

const char* output_domain_name(rns8_output_domain domain) {
  switch (domain) {
    case RNS8_OUTPUT_DOMAIN_RNS_RESIDUE:
      return "rns_residue_current";
    case RNS8_OUTPUT_DOMAIN_NATIVE_I64_U64:
      return "native_i64_u64_current";
    case RNS8_OUTPUT_DOMAIN_FINITE_U8:
      return "finite_u8_current";
    case RNS8_OUTPUT_DOMAIN_WRAP64_BYTE_LIMB:
      return "wrap64_byte_limb_current";
  }
  return "unknown";
}

std::string json_escape(const char* input) {
  std::string escaped;
  if (!input) {
    return escaped;
  }
  for (const unsigned char ch : std::string(input)) {
    switch (ch) {
      case '\\':
        escaped += "\\\\";
        break;
      case '"':
        escaped += "\\\"";
        break;
      case '\n':
        escaped += "\\n";
        break;
      case '\r':
        escaped += "\\r";
        break;
      case '\t':
        escaped += "\\t";
        break;
      default:
        escaped.push_back(static_cast<char>(ch));
        break;
    }
  }
  return escaped;
}

std::string json_escape(const std::string& input) {
  return json_escape(input.c_str());
}

bool parse_key_u32_field(const std::string& key, const std::string& name, uint32_t& out) {
  const std::string prefix = name + "=";
  std::size_t begin = 0;
  while (begin <= key.size()) {
    const std::size_t end = key.find(';', begin);
    const std::string field = key.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
    if (field.rfind(prefix, 0) == 0) {
      try {
        const unsigned long parsed = std::stoul(field.substr(prefix.size()));
        if (parsed > std::numeric_limits<uint32_t>::max()) {
          return false;
        }
        out = static_cast<uint32_t>(parsed);
        return true;
      } catch (...) {
        return false;
      }
    }
    if (end == std::string::npos) {
      break;
    }
    begin = end + 1;
  }
  return false;
}

bool fill_plan_desc_from_autotune_entry(
    const rns8::detail::AutotuneCacheEntry& entry,
    rns8_backend_kind backend,
    rns8_gemm_desc& desc) {
  if (entry.selected_backend != backend_name(backend) || entry.m <= 0 || entry.n <= 0 || entry.k <= 0) {
    return false;
  }

  rns8_semantics semantics = RNS8_BOUNDED_I64;
  rns8_bound_kind bound_kind = RNS8_BOUND_NONE;
  uint64_t bound = 0;
  uint32_t max_prefix = 0;
  uint32_t finite_modulus = 0;
  if (entry.semantic_contract == "bounded_i64") {
    semantics = RNS8_BOUNDED_I64;
    bound_kind = RNS8_BOUND_GLOBAL_MAX_ABS;
    bound = 127;
    max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
  } else if (entry.semantic_contract == "bounded_u64") {
    semantics = RNS8_BOUNDED_U64;
    bound_kind = RNS8_BOUND_GLOBAL_MAX_UNSIGNED;
    bound = 255;
    max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
  } else if (entry.semantic_contract == "exact_wide_signed") {
    semantics = RNS8_EXACT_WIDE_SIGNED;
    bound_kind = RNS8_BOUND_NONE;
    max_prefix = RNS8_MAX_SUPPORTED_PREFIX;
  } else if (entry.semantic_contract == "exact_wide_unsigned") {
    semantics = RNS8_EXACT_WIDE_UNSIGNED;
    bound_kind = RNS8_BOUND_NONE;
    max_prefix = RNS8_MAX_SUPPORTED_PREFIX;
  } else if (entry.semantic_contract == "finite_ring_u8") {
    semantics = RNS8_FINITE_RING_U8;
    bound_kind = RNS8_BOUND_NONE;
    max_prefix = 0;
    finite_modulus = entry.finite_modulus;
  } else if (entry.semantic_contract == "finite_field_u8") {
    semantics = RNS8_FINITE_FIELD_U8;
    bound_kind = RNS8_BOUND_NONE;
    max_prefix = 0;
    finite_modulus = entry.finite_modulus;
  } else {
    return false;
  }

  uint32_t parsed_prefix = max_prefix;
  if (parse_key_u32_field(entry.key, "prefix", parsed_prefix)) {
    max_prefix = parsed_prefix;
  }

  desc = {};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.semantics = semantics;
  desc.bound_kind = bound_kind;
  desc.requested_backend = backend;
  desc.m = entry.m;
  desc.n = entry.n;
  desc.k = entry.k;
  desc.bound = bound;
  desc.max_prefix = max_prefix;
  desc.finite_modulus = finite_modulus;
  desc.tile_m = entry.tile_m;
  desc.tile_n = entry.tile_n;
  return true;
}

rns8::detail::AutotuneRuntimeIdentity runtime_identity_with_plan_version(
    rns8_context* ctx,
    rns8_backend_kind backend,
    const rns8::detail::AutotuneCacheSnapshot& snapshot,
    const std::string& autotune_key,
    rns8::detail::AutotuneRuntimeIdentity runtime,
    rns8_plan_packing_info* plan_packing,
    bool* plan_packing_available,
    rns8::detail::PlanLoweringDescription* plan_lowering,
    bool* plan_lowering_available) {
  if (plan_packing_available) {
    *plan_packing_available = false;
  }
  if (plan_lowering_available) {
    *plan_lowering_available = false;
  }
  const auto* hit = rns8::detail::find_exact_autotune_entry(snapshot, autotune_key);
  if (!hit) {
    return runtime;
  }

  rns8_gemm_desc desc{};
  if (!fill_plan_desc_from_autotune_entry(*hit, backend, desc)) {
    return runtime;
  }

  rns8_plan* plan = nullptr;
  if (rns8_create_plan(ctx, &desc, &plan) != RNS8_SUCCESS || !plan) {
    return runtime;
  }

  rns8_plan_backend_info plan_info{};
  plan_info.struct_size = sizeof(plan_info);
  plan_info.abi_version = RNS8_ABI_VERSION;
  const rns8_status backend_status = rns8_get_plan_backend_info(plan, &plan_info);
  rns8_plan_packing_info local_packing{};
  local_packing.struct_size = sizeof(local_packing);
  local_packing.abi_version = RNS8_ABI_VERSION;
  const rns8_status packing_status = rns8_get_plan_packing_info(plan, &local_packing);
  rns8_plan_schedule_info schedule{};
  schedule.struct_size = sizeof(schedule);
  schedule.abi_version = RNS8_ABI_VERSION;
  const rns8_status schedule_status = rns8_get_plan_schedule_info(plan, &schedule);
  if (plan_packing) {
    *plan_packing = local_packing;
    if (packing_status == RNS8_SUCCESS && plan_packing_available) {
      *plan_packing_available = true;
    }
  }
  if (backend_status == RNS8_SUCCESS && packing_status == RNS8_SUCCESS && schedule_status == RNS8_SUCCESS &&
      plan_lowering) {
    *plan_lowering = rns8::detail::describe_plan_lowering(plan_info, local_packing, schedule);
    if (plan_lowering_available) {
      *plan_lowering_available = true;
    }
  }
  rns8_destroy_plan(plan);
  if (backend_status == RNS8_SUCCESS && plan_info.accelerator_version[0] != '\0') {
    runtime.hip_sdk_or_library_version = plan_info.accelerator_version;
  }
  return runtime;
}

