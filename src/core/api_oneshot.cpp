#include "core/api_internal.hpp"

namespace rns8::detail::api {

resident_oneshot_state::~resident_oneshot_state() {
  rns8_destroy_workspace(workspace);
  rns8_destroy_matrix(C);
  rns8_destroy_matrix(B);
  rns8_destroy_matrix(A);
  rns8_destroy_plan(plan);
}

bool uses_rns_storage(rns8_semantics semantics) {
  return semantics == RNS8_BOUNDED_I64 || semantics == RNS8_BOUNDED_U64 ||
         semantics == RNS8_EXACT_WIDE_SIGNED || semantics == RNS8_EXACT_WIDE_UNSIGNED;
}

bool uses_finite_storage(rns8_semantics semantics) {
  return semantics == RNS8_FINITE_RING_U8 || semantics == RNS8_FINITE_FIELD_U8;
}

rns8_matrix_desc make_matrix_desc(
    int64_t rows,
    int64_t cols,
    rns8_semantics semantics,
    rns8_bound_kind bound_kind,
    uint32_t prefix,
    uint32_t tile_m,
    uint32_t tile_n) {
  rns8_matrix_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.rows = rows;
  desc.cols = cols;
  desc.logical_ld = cols;
  desc.semantics = semantics;
  desc.logical_layout = RNS8_LAYOUT_ROW_MAJOR;
  desc.bound_kind = bound_kind;
  desc.tile_m = tile_m;
  desc.tile_n = tile_n;
  desc.max_prefix = prefix;
  return desc;
}

bool valid_matrix_access(int64_t rows, int64_t cols, int64_t ld) {
  if (rows <= 0 || cols <= 0 || ld < cols) {
    return false;
  }
  return rows <= std::numeric_limits<int64_t>::max() / ld;
}

bool valid_api_tile_size(uint32_t value) {
  return value == 0 || ((value >= 64 && value <= 512) && (value & (value - 1u)) == 0);
}

bool finite_backend_supports(rns8_backend_kind backend) {
  return backend == RNS8_BACKEND_CPU_REFERENCE || backend == RNS8_BACKEND_HIP_DIRECT ||
         backend == RNS8_BACKEND_HIPBLASLT || backend == RNS8_BACKEND_CK || backend == RNS8_BACKEND_WMMA;
}

rns8_status validate_finite_u8_oneshot_contract(
    const rns8_context& ctx,
    const rns8_gemm_desc& desc,
    rns8_semantics expected_semantics,
    uint16_t modulus,
    int64_t lda,
    int64_t ldb,
    int64_t ldc) {
  if (!rns8::detail::valid_abi(desc.struct_size, desc.abi_version, sizeof(desc)) ||
      desc.semantics != expected_semantics) {
    return RNS8_INVALID_ARGUMENT;
  }
  const bool valid_modulus = rns8::detail::valid_finite_modulus_for_semantics(expected_semantics, modulus);
  if (!valid_modulus || desc.m <= 0 || desc.n <= 0 || desc.k <= 0 || desc.bound_kind != RNS8_BOUND_NONE ||
      desc.bound != 0 || desc.max_prefix != 0 || desc.finite_modulus != modulus || desc.flags != 0 ||
      desc.tile_bounds || desc.tile_bounds_count != 0 || !valid_api_tile_size(desc.tile_m) ||
      !valid_api_tile_size(desc.tile_n)) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (!valid_matrix_access(desc.m, desc.k, lda) || !valid_matrix_access(desc.k, desc.n, ldb) ||
      !valid_matrix_access(desc.m, desc.n, ldc)) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_backend_kind requested = effective_backend(desc.requested_backend, ctx.backend);
  if (requested != ctx.backend || !finite_backend_supports(requested)) {
    return RNS8_UNSUPPORTED_BACKEND;
  }
  return RNS8_SUCCESS;
}


rns8_status create_resident_oneshot_state(
    rns8_context* ctx,
    const rns8_gemm_desc& desc,
    resident_oneshot_state& state) {
  rns8_status status = rns8_create_plan(ctx, &desc, &state.plan);
  if (status != RNS8_SUCCESS) {
    return status;
  }

  const rns8_matrix_desc a_desc =
      make_matrix_desc(desc.m, desc.k, desc.semantics, desc.bound_kind, state.plan->prefix,
                       state.plan->desc.tile_m, state.plan->desc.tile_n);
  const rns8_matrix_desc b_desc =
      make_matrix_desc(desc.k, desc.n, desc.semantics, desc.bound_kind, state.plan->prefix,
                       state.plan->desc.tile_m, state.plan->desc.tile_n);
  const rns8_matrix_desc c_desc =
      make_matrix_desc(desc.m, desc.n, desc.semantics, desc.bound_kind, state.plan->prefix,
                       state.plan->desc.tile_m, state.plan->desc.tile_n);

  status = rns8_create_matrix(ctx, &a_desc, &state.A);
  if (status == RNS8_SUCCESS) status = rns8_create_matrix(ctx, &b_desc, &state.B);
  if (status == RNS8_SUCCESS) status = rns8_create_matrix(ctx, &c_desc, &state.C);
  if (status == RNS8_SUCCESS) status = rns8_create_workspace(ctx, state.plan, &state.workspace);
  return status;
}

rns8_status finite_u8_oneshot_resident(
    rns8_context* ctx,
    const rns8_gemm_desc& desc,
    uint16_t modulus,
    const uint8_t* A,
    int64_t lda,
    const uint8_t* B,
    int64_t ldb,
    uint8_t* C,
    int64_t ldc) {
  resident_oneshot_state state;
  rns8_status status = create_resident_oneshot_state(ctx, desc, state);
  if (status == RNS8_SUCCESS) status = rns8_pack_finite_u8(ctx, state.A, modulus, A, lda, 1);
  if (status == RNS8_SUCCESS) status = rns8_pack_finite_u8(ctx, state.B, modulus, B, ldb, 2);
  if (status == RNS8_SUCCESS) {
    status = rns8_gemm_finite_u8(ctx, state.plan, modulus, state.A, state.B, state.C, state.workspace);
  }
  if (status == RNS8_SUCCESS) status = rns8_export_finite_u8(ctx, state.plan, modulus, state.C, C, ldc);
  return status;
}

bool valid_limb_export_access(int64_t rows, int64_t cols, int64_t ld, uint32_t limb_count) {
  if (!valid_matrix_access(rows, cols, ld) || limb_count == 0 || limb_count > 32) {
    return false;
  }
  const auto max = std::numeric_limits<int64_t>::max();
  if (ld > max / static_cast<int64_t>(limb_count)) {
    return false;
  }
  const int64_t limb_ld = ld * static_cast<int64_t>(limb_count);
  return rows <= max / limb_ld;
}

rns8_status validate_typed_oneshot_contract(
    const rns8_context& ctx,
    const rns8_gemm_desc& desc,
    rns8_semantics expected_semantics,
    int64_t lda,
    int64_t ldb,
    int64_t ldc) {
  if (!rns8::detail::valid_abi(desc.struct_size, desc.abi_version, sizeof(desc)) ||
      desc.semantics != expected_semantics) {
    return RNS8_INVALID_ARGUMENT;
  }
  const uint32_t prefix =
      desc.max_prefix == 0 ? rns8::detail::default_prefix_for_semantics(desc.semantics) : desc.max_prefix;
  const rns8_status validation = rns8::detail::validate_gemm_desc(desc, prefix);
  if (validation != RNS8_SUCCESS) {
    return validation;
  }
  if (!valid_matrix_access(desc.m, desc.k, lda) || !valid_matrix_access(desc.k, desc.n, ldb) ||
      !valid_matrix_access(desc.m, desc.n, ldc)) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_backend_kind requested = effective_backend(desc.requested_backend, ctx.backend);
  if (requested != ctx.backend || !backend_supports_semantics(requested, desc.semantics)) {
    return RNS8_UNSUPPORTED_BACKEND;
  }
  return RNS8_SUCCESS;
}

uint64_t ceil_div_i64_u32(int64_t value, uint32_t divisor) {
  const auto unsigned_value = static_cast<uint64_t>(value);
  return (unsigned_value + static_cast<uint64_t>(divisor) - 1u) / static_cast<uint64_t>(divisor);
}

boost::multiprecision::cpp_int schedule_required_range(const rns8_gemm_desc& desc) {
  using boost::multiprecision::cpp_int;
  switch (desc.semantics) {
    case RNS8_BOUNDED_I64:
      return cpp_int(2) * cpp_int(desc.bound);
    case RNS8_BOUNDED_U64:
      return cpp_int(desc.bound);
    case RNS8_EXACT_WIDE_SIGNED:
      return cpp_int(desc.k) * (cpp_int(1) << 127u);
    case RNS8_EXACT_WIDE_UNSIGNED:
      return cpp_int(desc.k) * (cpp_int(1) << 128u);
    case RNS8_WRAP_U64_MOD_2_64:
    case RNS8_FINITE_RING_U8:
    case RNS8_FINITE_FIELD_U8:
      return 0;
  }
  return 0;
}

}  // namespace rns8::detail::api

using namespace rns8::detail::api;

rns8_status rns8_gemm_i64_oneshot(
    rns8_context* ctx,
    const rns8_gemm_desc* desc,
    const int64_t* A,
    int64_t lda,
    const int64_t* B,
    int64_t ldb,
    int64_t* C,
    int64_t ldc) {
  return guard_api([&]() -> rns8_status {
    if (!ctx || !desc || !A || !B || !C) {
      return RNS8_INVALID_ARGUMENT;
    }
    const rns8_status preflight =
        validate_typed_oneshot_contract(*ctx, *desc, RNS8_BOUNDED_I64, lda, ldb, ldc);
    if (preflight != RNS8_SUCCESS) {
      return preflight;
    }

    resident_oneshot_state state;
    rns8_status status = create_resident_oneshot_state(ctx, *desc, state);
    if (status == RNS8_SUCCESS) status = rns8_pack_i64(ctx, state.A, A, lda, 1);
    if (status == RNS8_SUCCESS) status = rns8_pack_i64(ctx, state.B, B, ldb, 1);
    if (status == RNS8_SUCCESS) status = rns8_gemm_rns(ctx, state.plan, state.A, state.B, state.C, state.workspace);
    if (status == RNS8_SUCCESS) status = rns8_export_i64(ctx, state.plan, state.C, C, ldc);
    return status;
  });
}

rns8_status rns8_gemm_u64_oneshot(
    rns8_context* ctx,
    const rns8_gemm_desc* desc,
    const uint64_t* A,
    int64_t lda,
    const uint64_t* B,
    int64_t ldb,
    uint64_t* C,
    int64_t ldc) {
  return guard_api([&]() -> rns8_status {
    if (!ctx || !desc || !A || !B || !C) {
      return RNS8_INVALID_ARGUMENT;
    }
    const rns8_status preflight =
        validate_typed_oneshot_contract(*ctx, *desc, RNS8_BOUNDED_U64, lda, ldb, ldc);
    if (preflight != RNS8_SUCCESS) {
      return preflight;
    }

    resident_oneshot_state state;
    rns8_status status = create_resident_oneshot_state(ctx, *desc, state);
    if (status == RNS8_SUCCESS) status = rns8_pack_u64(ctx, state.A, A, lda, 1);
    if (status == RNS8_SUCCESS) status = rns8_pack_u64(ctx, state.B, B, ldb, 1);
    if (status == RNS8_SUCCESS) status = rns8_gemm_rns(ctx, state.plan, state.A, state.B, state.C, state.workspace);
    if (status == RNS8_SUCCESS) status = rns8_export_u64(ctx, state.plan, state.C, C, ldc);
    return status;
  });
}

rns8_status rns8_gemm_wrap_u64_oneshot(
    rns8_context* ctx,
    const rns8_gemm_desc* desc,
    const uint64_t* A,
    int64_t lda,
    const uint64_t* B,
    int64_t ldb,
    uint64_t* C,
    int64_t ldc) {
  return guard_api([&]() -> rns8_status {
    if (!ctx || !desc || !A || !B || !C) {
      return RNS8_INVALID_ARGUMENT;
    }
    const rns8_status preflight =
        validate_typed_oneshot_contract(*ctx, *desc, RNS8_WRAP_U64_MOD_2_64, lda, ldb, ldc);
    if (preflight != RNS8_SUCCESS) {
      return preflight;
    }

    resident_oneshot_state state;
    rns8_status status = create_resident_oneshot_state(ctx, *desc, state);
    if (status == RNS8_SUCCESS) status = rns8_pack_u64(ctx, state.A, A, lda, 1);
    if (status == RNS8_SUCCESS) status = rns8_pack_u64(ctx, state.B, B, ldb, 1);
    if (status == RNS8_SUCCESS) {
      status = rns8_gemm_wrap_u64(ctx, state.plan, state.A, state.B, state.C, state.workspace);
    }
    if (status == RNS8_SUCCESS) status = rns8_export_wrap_u64(ctx, state.plan, state.C, C, ldc);
    return status;
  });
}

rns8_status rns8_gemm_finite_ring_u8_oneshot(
    rns8_context* ctx,
    const rns8_gemm_desc* desc,
    uint16_t modulus,
    const uint8_t* A,
    int64_t lda,
    const uint8_t* B,
    int64_t ldb,
    uint8_t* C,
    int64_t ldc) {
  return guard_api([&]() -> rns8_status {
    if (!ctx || !desc || !A || !B || !C) {
      return RNS8_INVALID_ARGUMENT;
    }
    const rns8_status preflight =
        validate_finite_u8_oneshot_contract(*ctx, *desc, RNS8_FINITE_RING_U8, modulus, lda, ldb, ldc);
    if (preflight != RNS8_SUCCESS) {
      return preflight;
    }
    return finite_u8_oneshot_resident(ctx, *desc, modulus, A, lda, B, ldb, C, ldc);
  });
}

rns8_status rns8_gemm_finite_field_u8_oneshot(
    rns8_context* ctx,
    const rns8_gemm_desc* desc,
    uint16_t modulus,
    const uint8_t* A,
    int64_t lda,
    const uint8_t* B,
    int64_t ldb,
    uint8_t* C,
    int64_t ldc) {
  return guard_api([&]() -> rns8_status {
    if (!ctx || !desc || !A || !B || !C) {
      return RNS8_INVALID_ARGUMENT;
    }
    const rns8_status preflight =
        validate_finite_u8_oneshot_contract(*ctx, *desc, RNS8_FINITE_FIELD_U8, modulus, lda, ldb, ldc);
    if (preflight != RNS8_SUCCESS) {
      return preflight;
    }
    return finite_u8_oneshot_resident(ctx, *desc, modulus, A, lda, B, ldb, C, ldc);
  });
}
