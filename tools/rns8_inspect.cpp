#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>

#include "core/autotune_cache.hpp"
#include "rns8/rns8.h"

namespace {

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
  if (value == "rocwmma" || value == "wmma") {
    out = RNS8_BACKEND_WMMA;
    return true;
  }
  return false;
}

bool evidence_only_accelerator_backend(rns8_backend_kind backend) {
  return backend == RNS8_BACKEND_HIPBLASLT || backend == RNS8_BACKEND_CK || backend == RNS8_BACKEND_WMMA;
}

void print_usage(std::ostream& out) {
  out << "usage: rns8-inspect [--backend auto|cpu-reference|hip-direct|hip-vector-alu-int64|wrap64-byte-limb|hipblaslt|ck|rocwmma]"
      << " [--device N] [--json] [--autotune-key KEY] [--show-autotune-cache]\n";
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
    case RNS8_BACKEND_WMMA:
      return "wmma";
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
    bool* plan_packing_available) {
  if (plan_packing_available) {
    *plan_packing_available = false;
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
  const rns8_status status = rns8_get_plan_backend_info(plan, &plan_info);
  if (plan_packing) {
    plan_packing->struct_size = sizeof(*plan_packing);
    plan_packing->abi_version = RNS8_ABI_VERSION;
    if (rns8_get_plan_packing_info(plan, plan_packing) == RNS8_SUCCESS && plan_packing_available) {
      *plan_packing_available = true;
    }
  }
  rns8_destroy_plan(plan);
  if (status == RNS8_SUCCESS && plan_info.accelerator_version[0] != '\0') {
    runtime.hip_sdk_or_library_version = plan_info.accelerator_version;
  }
  return runtime;
}

void print_capability_text(const rns8_backend_capability_info& capability) {
  std::cout << "capability_status: " << capability.status << "\n";
  std::cout << "selected_kernel:   " << capability.selected_kernel << "\n";
  std::cout << "accelerator:       " << capability.is_accelerator << "\n";
  std::cout << "correctness:       " << capability.is_correctness_backend << "\n";
  std::cout << "matrix_engine:     " << capability.is_matrix_engine_backend << "\n";
  std::cout << "compiled_kernel:   " << capability.compiled_kernel_available << "\n";
  std::cout << "exact_validated:   " << capability.exact_differential_validated << "\n";
  std::cout << "perf_validated:    " << capability.performance_validated << "\n";
  std::cout << "library:           " << capability.library_name << "\n";
  std::cout << "enable_flag:       " << capability.enable_flag << "\n";
  std::cout << "epilogue_mode:     " << capability.epilogue_mode << "\n";
  std::cout << "workspace_mode:    " << capability.workspace_mode << "\n";
  std::cout << "isa_evidence:      " << capability.isa_evidence << "\n";
  std::cout << "capability_detail: " << capability.detail << "\n";
}

void print_capability_json(const rns8_backend_capability_info& capability, bool trailing_comma) {
  std::cout << "  \"capability\": {\n";
  std::cout << "    \"backend\": \"" << backend_name(capability.backend) << "\",\n";
  std::cout << "    \"backend_name\": \"" << json_escape(capability.backend_name) << "\",\n";
  std::cout << "    \"status\": \"" << json_escape(capability.status) << "\",\n";
  std::cout << "    \"detail\": \"" << json_escape(capability.detail) << "\",\n";
  std::cout << "    \"selected_kernel\": \"" << json_escape(capability.selected_kernel) << "\",\n";
  std::cout << "    \"accelerator_backend\": " << (capability.is_accelerator ? "true" : "false") << ",\n";
  std::cout << "    \"correctness_backend\": " << (capability.is_correctness_backend ? "true" : "false") << ",\n";
  std::cout << "    \"matrix_engine_backend\": " << (capability.is_matrix_engine_backend ? "true" : "false") << ",\n";
  std::cout << "    \"requires_feature_detection\": "
            << (capability.requires_feature_detection ? "true" : "false") << ",\n";
  std::cout << "    \"enable_flag_fail_fast\": "
            << (capability.enable_flag_fail_fast ? "true" : "false") << ",\n";
  std::cout << "    \"candidate_evidence_only\": "
            << (capability.candidate_evidence_only ? "true" : "false") << ",\n";
  std::cout << "    \"compiled_kernel_available\": "
            << (capability.compiled_kernel_available ? "true" : "false") << ",\n";
  std::cout << "    \"exact_differential_validated\": "
            << (capability.exact_differential_validated ? "true" : "false") << ",\n";
  std::cout << "    \"performance_validated\": " << (capability.performance_validated ? "true" : "false") << ",\n";
  std::cout << "    \"library_name\": \"" << json_escape(capability.library_name) << "\",\n";
  std::cout << "    \"library_version\": \"" << json_escape(capability.library_version) << "\",\n";
  std::cout << "    \"enable_flag\": \"" << json_escape(capability.enable_flag) << "\",\n";
  std::cout << "    \"epilogue_mode\": \"" << json_escape(capability.epilogue_mode) << "\",\n";
  std::cout << "    \"workspace_mode\": \"" << json_escape(capability.workspace_mode) << "\",\n";
  std::cout << "    \"isa_evidence\": \"" << json_escape(capability.isa_evidence) << "\"\n";
  std::cout << "  }" << (trailing_comma ? "," : "") << "\n";
}

void print_plan_packing_text(const rns8_plan_packing_info& packing) {
  std::cout << "autotune_plan_backend:            " << backend_name(packing.backend) << "\n";
  std::cout << "autotune_plan_semantics:          " << semantics_name(packing.semantics) << "\n";
  std::cout << "autotune_plan_resident_inputs:    " << packing.uses_resident_matrix_inputs << "\n";
  std::cout << "autotune_plan_transient_pack:     " << packing.uses_transient_pack_workspace << "\n";
  std::cout << "autotune_plan_matrix_engine_pack: " << packing.uses_matrix_engine_pack_layout << "\n";
  std::cout << "autotune_plan_reusable_prepack:   " << packing.reusable_prepack_cache_available << "\n";
  std::cout << "autotune_plan_production_prepack: " << packing.production_prepack_cache_available << "\n";
  std::cout << "autotune_plan_input_domain:       " << output_domain_name(packing.input_domain) << "\n";
  std::cout << "autotune_plan_output_domain:      " << output_domain_name(packing.output_domain) << "\n";
  std::cout << "autotune_plan_next_op_flags:      " << packing.next_op_flags << "\n";
  std::cout << "autotune_plan_total_pack_bytes:   " << packing.total_transient_workspace_bytes << "\n";
  std::cout << "autotune_plan_prepack_scope:      " << packing.prepack_cache_scope << "\n";
  std::cout << "autotune_plan_next_op_hint:       " << packing.next_op_hint << "\n";
}

void print_plan_packing_json(const rns8_plan_packing_info& packing, bool trailing_comma) {
  std::cout << "    \"plan_packing\": {\n";
  std::cout << "      \"backend\": \"" << backend_name(packing.backend) << "\",\n";
  std::cout << "      \"semantics\": \"" << semantics_name(packing.semantics) << "\",\n";
  std::cout << "      \"uses_resident_matrix_inputs\": "
            << (packing.uses_resident_matrix_inputs ? "true" : "false") << ",\n";
  std::cout << "      \"uses_transient_pack_workspace\": "
            << (packing.uses_transient_pack_workspace ? "true" : "false") << ",\n";
  std::cout << "      \"uses_matrix_engine_pack_layout\": "
            << (packing.uses_matrix_engine_pack_layout ? "true" : "false") << ",\n";
  std::cout << "      \"reusable_prepack_cache_available\": "
            << (packing.reusable_prepack_cache_available ? "true" : "false") << ",\n";
  std::cout << "      \"production_prepack_cache_available\": "
            << (packing.production_prepack_cache_available ? "true" : "false") << ",\n";
  std::cout << "      \"input_domain\": \"" << json_escape(output_domain_name(packing.input_domain)) << "\",\n";
  std::cout << "      \"output_domain\": \"" << json_escape(output_domain_name(packing.output_domain)) << "\",\n";
  std::cout << "      \"input_domain_name\": \"" << json_escape(packing.input_domain_name) << "\",\n";
  std::cout << "      \"output_domain_name\": \"" << json_escape(packing.output_domain_name) << "\",\n";
  std::cout << "      \"output_host_current\": " << (packing.output_host_current ? "true" : "false") << ",\n";
  std::cout << "      \"output_device_current\": " << (packing.output_device_current ? "true" : "false") << ",\n";
  std::cout << "      \"next_op_flags\": " << packing.next_op_flags << ",\n";
  std::cout << "      \"next_op_hint\": \"" << json_escape(packing.next_op_hint) << "\",\n";
  std::cout << "      \"a_pack_workspace_bytes\": " << packing.a_pack_workspace_bytes << ",\n";
  std::cout << "      \"b_pack_workspace_bytes\": " << packing.b_pack_workspace_bytes << ",\n";
  std::cout << "      \"accumulator_workspace_bytes\": " << packing.accumulator_workspace_bytes << ",\n";
  std::cout << "      \"library_workspace_bytes\": " << packing.library_workspace_bytes << ",\n";
  std::cout << "      \"total_transient_workspace_bytes\": " << packing.total_transient_workspace_bytes << ",\n";
  std::cout << "      \"a_layout_version\": \"" << json_escape(packing.a_layout_version) << "\",\n";
  std::cout << "      \"b_layout_version\": \"" << json_escape(packing.b_layout_version) << "\",\n";
  std::cout << "      \"output_layout_version\": \"" << json_escape(packing.output_layout_version) << "\",\n";
  std::cout << "      \"prepack_cache_scope\": \"" << json_escape(packing.prepack_cache_scope) << "\",\n";
  std::cout << "      \"detail\": \"" << json_escape(packing.detail) << "\"\n";
  std::cout << "    }" << (trailing_comma ? "," : "") << "\n";
}

rns8::detail::AutotuneRuntimeIdentity autotune_runtime_identity(
    const rns8_device_info& info,
    const rns8_backend_capability_info& capability) {
  rns8::detail::AutotuneRuntimeIdentity runtime{};
  const std::string target = info.gcn_arch;
  if (!target.empty() && target != "none") {
    runtime.target_id = target;
  } else if (!info.hip_available) {
    runtime.target_id = "cpu";
  }
  if (capability.library_version[0] != '\0' &&
      std::string(capability.library_version) != "runtime_queried_in_context") {
    runtime.hip_sdk_or_library_version = capability.library_version;
  }
  return runtime;
}

void print_autotune_text(
    const rns8::detail::AutotuneCacheSnapshot& snapshot,
    const std::string& autotune_key,
    const std::string& selected_backend,
    const rns8::detail::AutotuneRuntimeIdentity& runtime,
    const rns8_plan_packing_info* plan_packing,
    bool show_entries) {
  std::cout << "autotune_cache_path:   " << snapshot.path.string() << "\n";
  std::cout << "autotune_cache_loaded: " << (snapshot.loaded ? 1 : 0) << "\n";
  std::cout << "autotune_cache_exists: " << (snapshot.exists ? 1 : 0) << "\n";
  std::cout << "autotune_entry_count:  " << snapshot.entries.size() << "\n";
  if (!snapshot.error.empty()) {
    std::cout << "autotune_cache_error:  " << snapshot.error << "\n";
  }
  if (!autotune_key.empty()) {
    const auto* hit = rns8::detail::find_exact_autotune_entry(snapshot, autotune_key);
    std::cout << "autotune_key:          " << autotune_key << "\n";
    std::cout << "autotune_exact_hit:    " << (hit ? 1 : 0) << "\n";
    std::cout << "autotune_runtime_target:  " << runtime.target_id << "\n";
    std::cout << "autotune_runtime_version: " << runtime.hip_sdk_or_library_version << "\n";
    std::cout << "selection_rationale:   "
              << rns8::detail::autotune_selection_rationale(snapshot, autotune_key, selected_backend, runtime)
              << "\n";
    if (hit) {
      std::cout << "autotune_backend:      " << hit->selected_backend << "\n";
      std::cout << "autotune_kernel:       " << hit->selected_kernel << "\n";
      std::cout << "autotune_median_e2e:   " << hit->measured_median_end_to_end_us << "\n";
    }
    if (plan_packing) {
      print_plan_packing_text(*plan_packing);
    }
  }
  if (show_entries) {
    for (const auto& entry : snapshot.entries) {
      std::cout << "autotune_entry:        " << entry.selected_backend << " " << entry.selected_kernel
                << " " << entry.key << "\n";
    }
  }
}

void print_autotune_json(
    const rns8::detail::AutotuneCacheSnapshot& snapshot,
    const std::string& autotune_key,
    const std::string& selected_backend,
    const rns8::detail::AutotuneRuntimeIdentity& runtime,
    const rns8_plan_packing_info* plan_packing,
    bool show_entries) {
  const auto* hit = rns8::detail::find_exact_autotune_entry(snapshot, autotune_key);
  std::cout << "  \"autotune_cache\": {\n";
  std::cout << "    \"path\": \"" << json_escape(snapshot.path.string()) << "\",\n";
  std::cout << "    \"loaded\": " << (snapshot.loaded ? "true" : "false") << ",\n";
  std::cout << "    \"exists\": " << (snapshot.exists ? "true" : "false") << ",\n";
  std::cout << "    \"schema_version\": " << snapshot.schema_version << ",\n";
  std::cout << "    \"entry_count\": " << snapshot.entries.size() << ",\n";
  std::cout << "    \"error\": \"" << json_escape(snapshot.error) << "\",\n";
  std::cout << "    \"queried_key\": \"" << json_escape(autotune_key) << "\",\n";
  std::cout << "    \"exact_hit\": " << (hit ? "true" : "false") << ",\n";
  std::cout << "    \"runtime_target_id\": \"" << json_escape(runtime.target_id) << "\",\n";
  std::cout << "    \"runtime_version\": \"" << json_escape(runtime.hip_sdk_or_library_version) << "\",\n";
  std::cout << "    \"selection_rationale\": \""
            << json_escape(rns8::detail::autotune_selection_rationale(snapshot, autotune_key, selected_backend, runtime))
            << "\"";
  if (plan_packing) {
    std::cout << ",\n";
    print_plan_packing_json(*plan_packing, hit != nullptr || show_entries);
  }
  if (hit) {
    if (!plan_packing) {
      std::cout << ",\n";
    }
    std::cout << "    \"entry\": {\n";
    std::cout << "      \"selected_backend\": \"" << json_escape(hit->selected_backend) << "\",\n";
    std::cout << "      \"selected_kernel\": \"" << json_escape(hit->selected_kernel) << "\",\n";
    std::cout << "      \"target_id\": \"" << json_escape(hit->target_id) << "\",\n";
    std::cout << "      \"validation_status\": \"" << json_escape(hit->validation_status) << "\",\n";
    std::cout << "      \"performance_validated\": " << (hit->performance_validated ? "true" : "false") << ",\n";
    std::cout << "      \"measured_median_end_to_end_us\": " << hit->measured_median_end_to_end_us << "\n";
    std::cout << "    }";
  }
  if (show_entries) {
    std::cout << ",\n";
    std::cout << "    \"entries\": [";
    for (std::size_t i = 0; i < snapshot.entries.size(); ++i) {
      const auto& entry = snapshot.entries[i];
      std::cout << (i == 0 ? "\n" : ",\n");
      std::cout << "      {\"key\": \"" << json_escape(entry.key) << "\", \"selected_backend\": \""
                << json_escape(entry.selected_backend) << "\", \"selected_kernel\": \""
                << json_escape(entry.selected_kernel) << "\"}";
    }
    if (!snapshot.entries.empty()) {
      std::cout << "\n    ";
    }
    std::cout << "]";
  }
  std::cout << "\n";
  std::cout << "  }\n";
}

void print_text(
    const rns8_device_info& info,
    const rns8_backend_capability_info& capability,
    const rns8::detail::AutotuneCacheSnapshot* snapshot,
    const std::string& autotune_key,
    const rns8::detail::AutotuneRuntimeIdentity& runtime,
    const rns8_plan_packing_info* plan_packing,
    bool show_autotune_cache) {
  std::cout << "RNS8 inspect\n";
  std::cout << "backend:       " << backend_name(info.backend) << "\n";
  std::cout << "device_id:     " << info.device_id << "\n";
  std::cout << "name:          " << info.name << "\n";
  std::cout << "gcn_arch:      " << info.gcn_arch << "\n";
  std::cout << "hip_available: " << info.hip_available << "\n";
  std::cout << "hip_runtime:   " << info.hip_runtime_version << "\n";
  std::cout << "hip_driver:    " << info.hip_driver_version << "\n";
  std::cout << "global_mem:    " << info.global_mem_bytes << "\n";
  std::cout << "detail:        " << info.detail << "\n";
  print_capability_text(capability);
  if (snapshot) {
    print_autotune_text(
        *snapshot,
        autotune_key,
        backend_name(info.backend),
        runtime,
        plan_packing,
        show_autotune_cache);
  }
}

void print_json(
    const rns8_device_info& info,
    const rns8_backend_capability_info& capability,
    const rns8::detail::AutotuneCacheSnapshot* snapshot,
    const std::string& autotune_key,
    const rns8::detail::AutotuneRuntimeIdentity& runtime,
    const rns8_plan_packing_info* plan_packing,
    bool show_autotune_cache) {
  std::cout << "{\n";
  std::cout << "  \"backend\": \"" << backend_name(info.backend) << "\",\n";
  std::cout << "  \"device_id\": " << info.device_id << ",\n";
  std::cout << "  \"name\": \"" << json_escape(info.name) << "\",\n";
  std::cout << "  \"gcn_arch\": \"" << json_escape(info.gcn_arch) << "\",\n";
  std::cout << "  \"hip_available\": " << info.hip_available << ",\n";
  std::cout << "  \"hip_runtime_version\": " << info.hip_runtime_version << ",\n";
  std::cout << "  \"hip_driver_version\": " << info.hip_driver_version << ",\n";
  std::cout << "  \"global_mem_bytes\": " << info.global_mem_bytes << ",\n";
  std::cout << "  \"detail\": \"" << json_escape(info.detail) << "\",\n";
  print_capability_json(capability, snapshot != nullptr);
  if (snapshot) {
    print_autotune_json(
        *snapshot,
        autotune_key,
        backend_name(info.backend),
        runtime,
        plan_packing,
        show_autotune_cache);
  }
  std::cout << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
  bool json = false;
  bool show_autotune_cache = false;
  int device_id = -1;
  rns8_backend_kind backend = RNS8_BACKEND_CPU_REFERENCE;
  std::string autotune_key;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--json") {
      json = true;
    } else if (arg == "--show-autotune-cache") {
      show_autotune_cache = true;
    } else if (arg == "--autotune-key" && i + 1 < argc) {
      autotune_key = argv[++i];
    } else if (arg == "--backend" && i + 1 < argc) {
      const std::string value = argv[++i];
      if (!parse_backend(value, backend)) {
        std::cerr << "invalid backend string: " << value
                  << " (unknown names are not routed to auto; choose an explicit listed backend)\n";
        print_usage(std::cerr);
        return 2;
      }
    } else if (arg == "--device" && i + 1 < argc) {
      device_id = std::atoi(argv[++i]);
    } else if (arg == "--help") {
      print_usage(std::cout);
      return 0;
    } else {
      std::cerr << "unknown argument: " << arg << "\n";
      return 2;
    }
  }

  rns8_context_options options{};
  options.struct_size = sizeof(options);
  options.abi_version = RNS8_ABI_VERSION;
  options.requested_backend = backend;

  rns8_status status = RNS8_SUCCESS;
  rns8_backend_capability_info requested_capability{};
  requested_capability.struct_size = sizeof(requested_capability);
  requested_capability.abi_version = RNS8_ABI_VERSION;
  status = rns8_get_backend_capability_info(backend, &requested_capability);
  if (status != RNS8_SUCCESS) {
    std::cerr << "rns8_get_backend_capability_info(" << backend_name(backend) << "): "
              << rns8_status_string(status) << "\n";
    return 1;
  }

  rns8_context* ctx = nullptr;
  status = rns8_create_context(device_id, &options, &ctx);
  if (status != RNS8_SUCCESS) {
    std::cerr << "rns8_create_context(" << backend_name(backend) << "): " << rns8_status_string(status) << "\n";
    std::cerr << "capability_status: " << requested_capability.status << "\n";
    std::cerr << "capability_detail: " << requested_capability.detail << "\n";
    if (requested_capability.enable_flag[0] != '\0') {
      std::cerr << "enable_flag: " << requested_capability.enable_flag << "\n";
    }
    if (status == RNS8_UNSUPPORTED_BACKEND && evidence_only_accelerator_backend(backend)) {
      std::cerr << "requested accelerator is evidence-only; enable flags fail fast until a real exact "
                   "correctness backend exists\n";
    } else if (status == RNS8_UNSUPPORTED_BACKEND) {
      std::cerr << "backend is not available for this context; RNS8_BACKEND_AUTO does not route across "
                   "bounded, exact-wide, or wrap64 semantic backends\n";
    }
    return 1;
  }

  rns8_device_info info{};
  info.struct_size = sizeof(info);
  info.abi_version = RNS8_ABI_VERSION;
  status = rns8_get_device_info(ctx, &info);
  if (status != RNS8_SUCCESS) {
    rns8_destroy_context(ctx);
    std::cerr << "rns8_get_device_info: " << rns8_status_string(status) << "\n";
    return 1;
  }

  rns8_backend_capability_info selected_capability{};
  selected_capability.struct_size = sizeof(selected_capability);
  selected_capability.abi_version = RNS8_ABI_VERSION;
  status = rns8_get_backend_capability_info(info.backend, &selected_capability);
  if (status != RNS8_SUCCESS) {
    rns8_destroy_context(ctx);
    std::cerr << "rns8_get_backend_capability_info(" << backend_name(info.backend) << "): "
              << rns8_status_string(status) << "\n";
    return 1;
  }

  rns8::detail::AutotuneCacheSnapshot snapshot{};
  const bool inspect_autotune = show_autotune_cache || !autotune_key.empty();
  if (inspect_autotune) {
    snapshot = rns8::detail::read_autotune_cache();
  }

  auto runtime = autotune_runtime_identity(info, selected_capability);
  rns8_plan_packing_info autotune_plan_packing{};
  bool autotune_plan_packing_available = false;
  if (inspect_autotune && !autotune_key.empty()) {
    runtime = runtime_identity_with_plan_version(
        ctx,
        info.backend,
        snapshot,
        autotune_key,
        runtime,
        &autotune_plan_packing,
        &autotune_plan_packing_available);
  }
  const rns8_plan_packing_info* plan_packing =
      autotune_plan_packing_available ? &autotune_plan_packing : nullptr;

  if (json) {
    print_json(
        info,
        selected_capability,
        inspect_autotune ? &snapshot : nullptr,
        autotune_key,
        runtime,
        plan_packing,
        show_autotune_cache);
  } else {
    print_text(
        info,
        selected_capability,
        inspect_autotune ? &snapshot : nullptr,
        autotune_key,
        runtime,
        plan_packing,
        show_autotune_cache);
  }
  rns8_destroy_context(ctx);
  return 0;
}
