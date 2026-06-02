#include "core/internal.hpp"

#include <algorithm>
#include <limits>
#include <new>

#include "backend_hip_direct/hip_backend.hpp"

namespace {

template <typename Fn>
rns8_status guard_api(Fn&& fn) {
  try {
    return fn();
  } catch (const std::bad_alloc&) {
    return RNS8_INTERNAL_ERROR;
  } catch (...) {
    return RNS8_INTERNAL_ERROR;
  }
}

rns8_backend_kind effective_backend(rns8_backend_kind requested, rns8_backend_kind fallback) {
  return requested == RNS8_BACKEND_AUTO ? fallback : requested;
}

bool backend_supports_semantics(rns8_backend_kind backend, rns8_semantics semantics) {
  if (semantics == RNS8_WRAP_U64_MOD_2_64) {
    return backend == RNS8_BACKEND_WRAP64_BYTE_LIMB;
  }
  if (backend == RNS8_BACKEND_WRAP64_BYTE_LIMB) {
    return false;
  }
  return true;
}

rns8_matrix_desc make_matrix_desc(
    int64_t rows,
    int64_t cols,
    rns8_semantics semantics,
    rns8_bound_kind bound_kind,
    uint32_t prefix) {
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
  desc.max_prefix = prefix;
  return desc;
}

bool valid_matrix_access(int64_t rows, int64_t cols, int64_t ld) {
  if (rows <= 0 || cols <= 0 || ld < cols) {
    return false;
  }
  return rows <= std::numeric_limits<int64_t>::max() / ld;
}

std::vector<int8_t> gather_cell_residues(const rns8_matrix& matrix, int64_t row, int64_t col, uint32_t prefix) {
  std::vector<int8_t> residues(prefix);
  for (uint32_t p = 0; p < prefix; ++p) {
    residues[p] = matrix.residues[rns8::detail::residue_index(matrix, p, row, col)];
  }
  return residues;
}

rns8_status free_hip_matrix_storage(rns8_matrix& matrix) {
  rns8_status status = RNS8_SUCCESS;
  if (matrix.hip_upload_buffer) {
    status = rns8::detail::hip_direct_free(matrix.hip_device_id, matrix.hip_upload_buffer);
    matrix.hip_upload_buffer = nullptr;
    matrix.hip_upload_bytes = 0;
  }
  if (matrix.hip_status_buffer) {
    const rns8_status free_status = rns8::detail::hip_direct_free(matrix.hip_device_id, matrix.hip_status_buffer);
    if (status == RNS8_SUCCESS) {
      status = free_status;
    }
    matrix.hip_status_buffer = nullptr;
    matrix.hip_status_bytes = 0;
  }
  if (matrix.hip_export_buffer) {
    const rns8_status free_status = rns8::detail::hip_direct_free(matrix.hip_device_id, matrix.hip_export_buffer);
    if (status == RNS8_SUCCESS) {
      status = free_status;
    }
    matrix.hip_export_buffer = nullptr;
    matrix.hip_export_bytes = 0;
  }
  if (matrix.hip_residues) {
    const rns8_status free_status = rns8::detail::hip_direct_free(matrix.hip_device_id, matrix.hip_residues);
    if (status == RNS8_SUCCESS) {
      status = free_status;
    }
    matrix.hip_residues = nullptr;
    matrix.hip_residue_bytes = 0;
  }
  matrix.device_residues_current = false;
  return status;
}

rns8_status allocate_hip_matrix_storage(rns8_context& ctx, rns8_matrix& matrix) {
  if (matrix.residues.empty()) {
    return RNS8_INVALID_ARGUMENT;
  }
  matrix.hip_device_id = ctx.device_id;
  matrix.hip_residue_bytes = matrix.residues.size() * sizeof(int8_t);
  rns8_status status = rns8::detail::hip_direct_allocate(ctx.device_id, matrix.hip_residue_bytes, &matrix.hip_residues);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  status = rns8::detail::hip_direct_zero(ctx.device_id, matrix.hip_residues, matrix.hip_residue_bytes);
  if (status != RNS8_SUCCESS) {
    (void)free_hip_matrix_storage(matrix);
    return status;
  }
  matrix.host_residues_current = true;
  matrix.device_residues_current = true;
  return RNS8_SUCCESS;
}

rns8_status ensure_device_residues_current(rns8_matrix& matrix) {
  if (matrix.backend != RNS8_BACKEND_HIP_DIRECT) {
    return RNS8_SUCCESS;
  }
  if (matrix.device_residues_current) {
    return RNS8_SUCCESS;
  }
  if (!matrix.host_residues_current || !matrix.hip_residues || matrix.hip_residue_bytes == 0) {
    return RNS8_INTERNAL_ERROR;
  }
  const rns8_status status = rns8::detail::hip_direct_copy_host_to_device(
      matrix.hip_device_id, matrix.hip_residues, matrix.residues.data(), matrix.hip_residue_bytes);
  if (status == RNS8_SUCCESS) {
    matrix.device_residues_current = true;
  }
  return status;
}

rns8_status ensure_host_residues_current(const rns8_matrix& const_matrix) {
  auto& matrix = const_cast<rns8_matrix&>(const_matrix);
  if (matrix.backend != RNS8_BACKEND_HIP_DIRECT) {
    return RNS8_SUCCESS;
  }
  if (matrix.host_residues_current) {
    return RNS8_SUCCESS;
  }
  if (!matrix.device_residues_current || !matrix.hip_residues || matrix.hip_residue_bytes == 0) {
    return RNS8_INTERNAL_ERROR;
  }
  const rns8_status status = rns8::detail::hip_direct_copy_device_to_host(
      matrix.hip_device_id, matrix.residues.data(), matrix.hip_residues, matrix.hip_residue_bytes);
  if (status == RNS8_SUCCESS) {
    matrix.host_residues_current = true;
  }
  return status;
}

}  // namespace

rns8_status rns8_create_context(int device_id, const rns8_context_options* options, rns8_context** out) {
  return guard_api([&]() -> rns8_status {
    if (!out) {
      return RNS8_INVALID_ARGUMENT;
    }
    *out = nullptr;

    rns8_backend_kind requested = RNS8_BACKEND_CPU_REFERENCE;
    if (options) {
      if (!rns8::detail::valid_abi(options->struct_size, options->abi_version, sizeof(*options))) {
        return RNS8_INVALID_ARGUMENT;
      }
      requested = effective_backend(options->requested_backend, RNS8_BACKEND_CPU_REFERENCE);
    }

    auto* ctx = new (std::nothrow) rns8_context();
    if (!ctx) {
      return RNS8_INTERNAL_ERROR;
    }

    if (requested == RNS8_BACKEND_CPU_REFERENCE) {
      ctx->backend = RNS8_BACKEND_CPU_REFERENCE;
      ctx->device_id = -1;
      ctx->device_info.struct_size = sizeof(ctx->device_info);
      ctx->device_info.abi_version = RNS8_ABI_VERSION;
      rns8::detail::fill_cpu_device_info(ctx->device_info);
      *out = ctx;
      return RNS8_SUCCESS;
    }

    if (requested == RNS8_BACKEND_HIP_DIRECT) {
      ctx->backend = RNS8_BACKEND_HIP_DIRECT;
      ctx->device_id = device_id < 0 ? 0 : device_id;
      ctx->device_info.struct_size = sizeof(ctx->device_info);
      ctx->device_info.abi_version = RNS8_ABI_VERSION;
      const rns8_status status = rns8::detail::hip_direct_probe(ctx->device_id, ctx->device_info);
      if (status != RNS8_SUCCESS) {
        delete ctx;
        return status;
      }
      *out = ctx;
      return RNS8_SUCCESS;
    }

    if (requested == RNS8_BACKEND_WRAP64_BYTE_LIMB) {
      ctx->backend = RNS8_BACKEND_WRAP64_BYTE_LIMB;
      ctx->device_id = -1;
      ctx->device_info.struct_size = sizeof(ctx->device_info);
      ctx->device_info.abi_version = RNS8_ABI_VERSION;
      rns8::detail::fill_wrap64_device_info(ctx->device_info);
      *out = ctx;
      return RNS8_SUCCESS;
    }

    delete ctx;
    return RNS8_UNSUPPORTED_BACKEND;
  });
}

rns8_status rns8_destroy_context(rns8_context* ctx) {
  delete ctx;
  return RNS8_SUCCESS;
}

rns8_status rns8_get_device_info(rns8_context* ctx, rns8_device_info* out) {
  return guard_api([&]() -> rns8_status {
    if (!ctx || !out || !rns8::detail::valid_abi(out->struct_size, out->abi_version, sizeof(*out))) {
      return RNS8_INVALID_ARGUMENT;
    }
    const uint64_t struct_size = out->struct_size;
    const uint32_t abi_version = out->abi_version;
    *out = ctx->device_info;
    out->struct_size = struct_size;
    out->abi_version = abi_version;
    return RNS8_SUCCESS;
  });
}

rns8_status rns8_create_plan(rns8_context* ctx, const rns8_gemm_desc* desc, rns8_plan** out) {
  return guard_api([&]() -> rns8_status {
    if (!ctx || !desc || !out) {
      return RNS8_INVALID_ARGUMENT;
    }
    *out = nullptr;
    if (!rns8::detail::valid_abi(desc->struct_size, desc->abi_version, sizeof(*desc))) {
      return RNS8_INVALID_ARGUMENT;
    }
    const uint32_t prefix =
        desc->max_prefix == 0 ? rns8::detail::default_prefix_for_semantics(desc->semantics) : desc->max_prefix;
    const rns8_status validation = rns8::detail::validate_gemm_desc(*desc, prefix);
    if (validation != RNS8_SUCCESS) {
      return validation;
    }

    const rns8_backend_kind requested = effective_backend(desc->requested_backend, ctx->backend);
    if (requested != ctx->backend) {
      return RNS8_UNSUPPORTED_BACKEND;
    }
    if (requested != RNS8_BACKEND_CPU_REFERENCE && requested != RNS8_BACKEND_HIP_DIRECT &&
        requested != RNS8_BACKEND_WRAP64_BYTE_LIMB) {
      return RNS8_UNSUPPORTED_BACKEND;
    }
    if (!backend_supports_semantics(requested, desc->semantics)) {
      return RNS8_UNSUPPORTED_BACKEND;
    }

    auto* plan = new (std::nothrow) rns8_plan();
    if (!plan) {
      return RNS8_INTERNAL_ERROR;
    }
    plan->desc = *desc;
    plan->desc.max_prefix = prefix;
    if (plan->desc.tile_m == 0) {
      plan->desc.tile_m = 128;
    }
    if (plan->desc.tile_n == 0) {
      plan->desc.tile_n = 128;
    }
    plan->prefix = prefix;
    plan->modulus_product = prefix == 0 ? 0 : rns8::detail::modulus_product(prefix);
    plan->backend = requested;
    *out = plan;
    return RNS8_SUCCESS;
  });
}

rns8_status rns8_destroy_plan(rns8_plan* plan) {
  delete plan;
  return RNS8_SUCCESS;
}

rns8_status rns8_create_workspace(rns8_context* ctx, const rns8_plan* plan, rns8_workspace** out) {
  return guard_api([&]() -> rns8_status {
    if (!ctx || !plan || !out) {
      return RNS8_INVALID_ARGUMENT;
    }
    *out = nullptr;
    if (ctx->backend != plan->backend) {
      return RNS8_INVALID_ARGUMENT;
    }
    auto* workspace = new (std::nothrow) rns8_workspace();
    if (!workspace) {
      return RNS8_INTERNAL_ERROR;
    }
    workspace->m = plan->desc.m;
    workspace->n = plan->desc.n;
    workspace->k = plan->desc.k;
    workspace->prefix = plan->prefix;
    workspace->backend = ctx->backend;
    workspace->hip_device_id = ctx->backend == RNS8_BACKEND_HIP_DIRECT ? ctx->device_id : -1;
    *out = workspace;
    return RNS8_SUCCESS;
  });
}

rns8_status rns8_destroy_workspace(rns8_workspace* workspace) {
  if (workspace && workspace->hip_scratch) {
    const rns8_status status = rns8::detail::hip_direct_free(workspace->hip_device_id, workspace->hip_scratch);
    delete workspace;
    return status;
  }
  delete workspace;
  return RNS8_SUCCESS;
}

rns8_status rns8_create_matrix(rns8_context* ctx, const rns8_matrix_desc* desc, rns8_matrix** out) {
  return guard_api([&]() -> rns8_status {
    if (!ctx || !desc || !out) {
      return RNS8_INVALID_ARGUMENT;
    }
    *out = nullptr;
    if (!rns8::detail::valid_abi(desc->struct_size, desc->abi_version, sizeof(*desc))) {
      return RNS8_INVALID_ARGUMENT;
    }
    const uint32_t prefix =
        desc->max_prefix == 0 ? rns8::detail::default_prefix_for_semantics(desc->semantics) : desc->max_prefix;
    const rns8_status validation = rns8::detail::validate_matrix_desc(*desc, prefix);
    if (validation != RNS8_SUCCESS) {
      return validation;
    }
    if (!backend_supports_semantics(ctx->backend, desc->semantics)) {
      return RNS8_UNSUPPORTED_BACKEND;
    }
    auto* matrix = new (std::nothrow) rns8_matrix();
    if (!matrix) {
      return RNS8_INTERNAL_ERROR;
    }
    matrix->desc = *desc;
    matrix->backend = ctx->backend;
    matrix->desc.max_prefix = prefix;
    if (matrix->desc.logical_ld == 0) {
      matrix->desc.logical_ld = matrix->desc.cols;
    }
    if (matrix->desc.tile_m == 0) {
      matrix->desc.tile_m = 128;
    }
    if (matrix->desc.tile_n == 0) {
      matrix->desc.tile_n = 128;
    }
    matrix->prefix = prefix;
    if (matrix->desc.semantics == RNS8_WRAP_U64_MOD_2_64) {
      const auto elements = static_cast<std::size_t>(desc->rows) * static_cast<std::size_t>(desc->cols) * 8u;
      matrix->byte_limbs.assign(elements, 0);
      matrix->host_residues_current = false;
      matrix->device_residues_current = false;
    } else {
      const auto elements = static_cast<std::size_t>(prefix) * static_cast<std::size_t>(desc->rows) *
                            static_cast<std::size_t>(desc->cols);
      matrix->residues.assign(elements, 0);
    }
    if (ctx->backend == RNS8_BACKEND_HIP_DIRECT) {
      const rns8_status status = allocate_hip_matrix_storage(*ctx, *matrix);
      if (status != RNS8_SUCCESS) {
        delete matrix;
        return status;
      }
    }
    *out = matrix;
    return RNS8_SUCCESS;
  });
}

rns8_status rns8_destroy_matrix(rns8_matrix* matrix) {
  if (matrix) {
    const rns8_status status = free_hip_matrix_storage(*matrix);
    delete matrix;
    return status;
  }
  delete matrix;
  return RNS8_SUCCESS;
}

rns8_status rns8_pack_i64(
    rns8_context* ctx,
    rns8_matrix* matrix,
    const int64_t* src,
    int64_t ld,
    uint64_t source_version) {
  return guard_api([&]() -> rns8_status {
    if (!ctx || !matrix || !src || ld < matrix->desc.cols) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (matrix->desc.semantics != RNS8_BOUNDED_I64 && matrix->desc.semantics != RNS8_EXACT_WIDE_SIGNED) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (ctx->backend != matrix->backend) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (ctx->backend == RNS8_BACKEND_HIP_DIRECT) {
      const rns8_status status = rns8::detail::hip_direct_pack_i64_device(
          ctx->device_id,
          src,
          &matrix->hip_upload_buffer,
          &matrix->hip_upload_bytes,
          matrix->hip_residues,
          matrix->desc.rows,
          matrix->desc.cols,
          ld,
          matrix->prefix);
      if (status != RNS8_SUCCESS) {
        return status;
      }
      matrix->device_residues_current = true;
      matrix->host_residues_current = false;
    } else {
      rns8::detail::pack_i64_matrix(*matrix, src, ld);
      matrix->host_residues_current = true;
      matrix->device_residues_current = false;
    }
    matrix->source_version = source_version;
    return RNS8_SUCCESS;
  });
}

rns8_status rns8_pack_u64(
    rns8_context* ctx,
    rns8_matrix* matrix,
    const uint64_t* src,
    int64_t ld,
    uint64_t source_version) {
  return guard_api([&]() -> rns8_status {
    if (!ctx || !matrix || !src || !valid_matrix_access(matrix->desc.rows, matrix->desc.cols, ld)) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (matrix->desc.semantics != RNS8_BOUNDED_U64 && matrix->desc.semantics != RNS8_EXACT_WIDE_UNSIGNED &&
        matrix->desc.semantics != RNS8_WRAP_U64_MOD_2_64) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (ctx->backend != matrix->backend) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (matrix->desc.semantics == RNS8_WRAP_U64_MOD_2_64) {
      if (ctx->backend != RNS8_BACKEND_WRAP64_BYTE_LIMB) {
        return RNS8_UNSUPPORTED_BACKEND;
      }
      rns8::detail::pack_wrap_u64_matrix(*matrix, src, ld);
      matrix->host_residues_current = false;
      matrix->device_residues_current = false;
    } else if (ctx->backend == RNS8_BACKEND_HIP_DIRECT) {
      const rns8_status status = rns8::detail::hip_direct_pack_u64_device(
          ctx->device_id,
          src,
          &matrix->hip_upload_buffer,
          &matrix->hip_upload_bytes,
          matrix->hip_residues,
          matrix->desc.rows,
          matrix->desc.cols,
          ld,
          matrix->prefix);
      if (status != RNS8_SUCCESS) {
        return status;
      }
      matrix->device_residues_current = true;
      matrix->host_residues_current = false;
    } else {
      rns8::detail::pack_u64_matrix(*matrix, src, ld);
      matrix->host_residues_current = true;
      matrix->device_residues_current = false;
    }
    matrix->source_version = source_version;
    return RNS8_SUCCESS;
  });
}

rns8_status rns8_gemm_rns(
    rns8_context* ctx,
    const rns8_plan* plan,
    const rns8_matrix* A,
    const rns8_matrix* B,
    rns8_matrix* C,
    rns8_workspace* workspace) {
  return guard_api([&]() -> rns8_status {
    if (!ctx || !plan || !A || !B || !C || !workspace) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (workspace->m != plan->desc.m || workspace->n != plan->desc.n || workspace->k != plan->desc.k ||
        workspace->prefix != plan->prefix) {
      return RNS8_WORKSPACE_TOO_SMALL;
    }
    if (ctx->backend != plan->backend || workspace->backend != plan->backend || A->backend != plan->backend ||
        B->backend != plan->backend || C->backend != plan->backend) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (plan->desc.semantics == RNS8_WRAP_U64_MOD_2_64) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (A->desc.semantics != plan->desc.semantics || B->desc.semantics != plan->desc.semantics ||
        C->desc.semantics != plan->desc.semantics) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (plan->backend == RNS8_BACKEND_CPU_REFERENCE) {
      return rns8::detail::cpu_gemm_rns(*plan, *A, *B, *C);
    }
    if (plan->backend == RNS8_BACKEND_HIP_DIRECT) {
      if (A->desc.rows != plan->desc.m || A->desc.cols != plan->desc.k || B->desc.rows != plan->desc.k ||
          B->desc.cols != plan->desc.n || C->desc.rows != plan->desc.m || C->desc.cols != plan->desc.n) {
        return RNS8_INVALID_ARGUMENT;
      }
      rns8_status status = ensure_device_residues_current(const_cast<rns8_matrix&>(*A));
      if (status != RNS8_SUCCESS) {
        return status;
      }
      status = ensure_device_residues_current(const_cast<rns8_matrix&>(*B));
      if (status != RNS8_SUCCESS) {
        return status;
      }
      status = rns8::detail::hip_direct_gemm_rns_device(
          ctx->device_id,
          A->hip_residues,
          B->hip_residues,
          C->hip_residues,
          plan->desc.m,
          plan->desc.n,
          plan->desc.k,
          A->desc.cols,
          B->desc.cols,
          C->desc.cols,
          plan->prefix);
      if (status != RNS8_SUCCESS) {
        return status;
      }
      C->device_residues_current = true;
      C->host_residues_current = false;
      return RNS8_SUCCESS;
    }
    return RNS8_UNSUPPORTED_BACKEND;
  });
}

rns8_status rns8_gemm_wrap_u64(
    rns8_context* ctx,
    const rns8_plan* plan,
    const rns8_matrix* A,
    const rns8_matrix* B,
    rns8_matrix* C,
    rns8_workspace* workspace) {
  return guard_api([&]() -> rns8_status {
    if (!ctx || !plan || !A || !B || !C || !workspace) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (plan->desc.semantics != RNS8_WRAP_U64_MOD_2_64 || plan->prefix != 0) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (ctx->backend != RNS8_BACKEND_WRAP64_BYTE_LIMB || plan->backend != RNS8_BACKEND_WRAP64_BYTE_LIMB ||
        workspace->backend != RNS8_BACKEND_WRAP64_BYTE_LIMB || A->backend != RNS8_BACKEND_WRAP64_BYTE_LIMB ||
        B->backend != RNS8_BACKEND_WRAP64_BYTE_LIMB || C->backend != RNS8_BACKEND_WRAP64_BYTE_LIMB) {
      return RNS8_UNSUPPORTED_BACKEND;
    }
    if (workspace->m != plan->desc.m || workspace->n != plan->desc.n || workspace->k != plan->desc.k ||
        workspace->prefix != 0) {
      return RNS8_WORKSPACE_TOO_SMALL;
    }
    if (A->desc.semantics != RNS8_WRAP_U64_MOD_2_64 || B->desc.semantics != RNS8_WRAP_U64_MOD_2_64 ||
        C->desc.semantics != RNS8_WRAP_U64_MOD_2_64) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (A->desc.rows != plan->desc.m || A->desc.cols != plan->desc.k || B->desc.rows != plan->desc.k ||
        B->desc.cols != plan->desc.n || C->desc.rows != plan->desc.m || C->desc.cols != plan->desc.n) {
      return RNS8_INVALID_ARGUMENT;
    }
    return rns8::detail::cpu_gemm_wrap_u64(*plan, *A, *B, *C);
  });
}

rns8_status rns8_export_i64(rns8_context* ctx, const rns8_plan* plan, const rns8_matrix* C, int64_t* dst, int64_t ld) {
  return guard_api([&]() -> rns8_status {
    if (!ctx || !plan || !C || !dst || ld < plan->desc.n || C->desc.semantics != RNS8_BOUNDED_I64) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (C->desc.rows != plan->desc.m || C->desc.cols != plan->desc.n) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (ctx->backend != plan->backend || C->backend != plan->backend) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (plan->backend == RNS8_BACKEND_HIP_DIRECT) {
      const rns8_status status = rns8::detail::hip_direct_export_i64_device(
          ctx->device_id,
          C->hip_residues,
          &const_cast<rns8_matrix*>(C)->hip_export_buffer,
          &const_cast<rns8_matrix*>(C)->hip_export_bytes,
          &const_cast<rns8_matrix*>(C)->hip_status_buffer,
          &const_cast<rns8_matrix*>(C)->hip_status_bytes,
          plan->desc.m,
          plan->desc.n,
          plan->prefix,
          plan->desc.bound,
          dst,
          ld);
      return status;
    }
    const rns8_status sync_status = ensure_host_residues_current(*C);
    if (sync_status != RNS8_SUCCESS) {
      return sync_status;
    }
    for (int64_t row = 0; row < plan->desc.m; ++row) {
      for (int64_t col = 0; col < plan->desc.n; ++col) {
        int64_t value = 0;
        const std::vector<int8_t> residues = gather_cell_residues(*C, row, col, plan->prefix);
        const rns8_status status = rns8::detail::reconstruct_signed(residues, plan->prefix, plan->desc.bound, value);
        if (status != RNS8_SUCCESS) {
          return status;
        }
        dst[row * ld + col] = value;
      }
    }
    return RNS8_SUCCESS;
  });
}

rns8_status rns8_export_u64(
    rns8_context* ctx,
    const rns8_plan* plan,
    const rns8_matrix* C,
    uint64_t* dst,
    int64_t ld) {
  return guard_api([&]() -> rns8_status {
    if (!ctx || !plan || !C || !dst || ld < plan->desc.n || C->desc.semantics != RNS8_BOUNDED_U64) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (C->desc.rows != plan->desc.m || C->desc.cols != plan->desc.n) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (ctx->backend != plan->backend || C->backend != plan->backend) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (plan->backend == RNS8_BACKEND_HIP_DIRECT) {
      const rns8_status status = rns8::detail::hip_direct_export_u64_device(
          ctx->device_id,
          C->hip_residues,
          &const_cast<rns8_matrix*>(C)->hip_export_buffer,
          &const_cast<rns8_matrix*>(C)->hip_export_bytes,
          &const_cast<rns8_matrix*>(C)->hip_status_buffer,
          &const_cast<rns8_matrix*>(C)->hip_status_bytes,
          plan->desc.m,
          plan->desc.n,
          plan->prefix,
          plan->desc.bound,
          dst,
          ld);
      return status;
    }
    const rns8_status sync_status = ensure_host_residues_current(*C);
    if (sync_status != RNS8_SUCCESS) {
      return sync_status;
    }
    for (int64_t row = 0; row < plan->desc.m; ++row) {
      for (int64_t col = 0; col < plan->desc.n; ++col) {
        uint64_t value = 0;
        const std::vector<int8_t> residues = gather_cell_residues(*C, row, col, plan->prefix);
        const rns8_status status = rns8::detail::reconstruct_unsigned(residues, plan->prefix, plan->desc.bound, value);
        if (status != RNS8_SUCCESS) {
          return status;
        }
        dst[row * ld + col] = value;
      }
    }
    return RNS8_SUCCESS;
  });
}

rns8_status rns8_export_wrap_u64(
    rns8_context* ctx,
    const rns8_plan* plan,
    const rns8_matrix* C,
    uint64_t* dst,
    int64_t ld) {
  return guard_api([&]() -> rns8_status {
    if (!ctx || !plan || !C || !dst || !valid_matrix_access(plan->desc.m, plan->desc.n, ld)) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (plan->desc.semantics != RNS8_WRAP_U64_MOD_2_64 || C->desc.semantics != RNS8_WRAP_U64_MOD_2_64 ||
        plan->prefix != 0) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (C->desc.rows != plan->desc.m || C->desc.cols != plan->desc.n) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (ctx->backend != RNS8_BACKEND_WRAP64_BYTE_LIMB || plan->backend != RNS8_BACKEND_WRAP64_BYTE_LIMB ||
        C->backend != RNS8_BACKEND_WRAP64_BYTE_LIMB) {
      return RNS8_UNSUPPORTED_BACKEND;
    }
    for (int64_t row = 0; row < plan->desc.m; ++row) {
      for (int64_t col = 0; col < plan->desc.n; ++col) {
        dst[row * ld + col] = rns8::detail::wrap_u64_matrix_cell(*C, row, col);
      }
    }
    return RNS8_SUCCESS;
  });
}

rns8_status rns8_export_exact_wide_signed_limbs(
    rns8_context* ctx,
    const rns8_plan* plan,
    const rns8_matrix* C,
    uint64_t* dst,
    int64_t ld,
    uint32_t limb_count) {
  return guard_api([&]() -> rns8_status {
    if (!ctx || !plan || !C || !dst || ld < plan->desc.n || limb_count == 0 ||
        C->desc.semantics != RNS8_EXACT_WIDE_SIGNED) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (C->desc.rows != plan->desc.m || C->desc.cols != plan->desc.n || plan->desc.semantics != C->desc.semantics) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (ctx->backend != plan->backend || C->backend != plan->backend) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (plan->backend == RNS8_BACKEND_HIP_DIRECT) {
      rns8_status status = ensure_device_residues_current(const_cast<rns8_matrix&>(*C));
      if (status != RNS8_SUCCESS) {
        return status;
      }
      return rns8::detail::hip_direct_export_exact_wide_signed_limbs_device(
          ctx->device_id,
          C->hip_residues,
          &const_cast<rns8_matrix*>(C)->hip_export_buffer,
          &const_cast<rns8_matrix*>(C)->hip_export_bytes,
          &const_cast<rns8_matrix*>(C)->hip_status_buffer,
          &const_cast<rns8_matrix*>(C)->hip_status_bytes,
          plan->desc.m,
          plan->desc.n,
          plan->prefix,
          dst,
          ld,
          limb_count);
    }
    const rns8_status sync_status = ensure_host_residues_current(*C);
    if (sync_status != RNS8_SUCCESS) {
      return sync_status;
    }
    for (int64_t row = 0; row < plan->desc.m; ++row) {
      for (int64_t col = 0; col < plan->desc.n; ++col) {
        const std::vector<int8_t> residues = gather_cell_residues(*C, row, col, plan->prefix);
        const rns8_status status = rns8::detail::export_exact_wide_signed_limbs(
            residues, plan->prefix, dst + (row * ld + col) * limb_count, limb_count);
        if (status != RNS8_SUCCESS) {
          return status;
        }
      }
    }
    return RNS8_SUCCESS;
  });
}

rns8_status rns8_export_exact_wide_unsigned_limbs(
    rns8_context* ctx,
    const rns8_plan* plan,
    const rns8_matrix* C,
    uint64_t* dst,
    int64_t ld,
    uint32_t limb_count) {
  return guard_api([&]() -> rns8_status {
    if (!ctx || !plan || !C || !dst || ld < plan->desc.n || limb_count == 0 ||
        C->desc.semantics != RNS8_EXACT_WIDE_UNSIGNED) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (C->desc.rows != plan->desc.m || C->desc.cols != plan->desc.n || plan->desc.semantics != C->desc.semantics) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (ctx->backend != plan->backend || C->backend != plan->backend) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (plan->backend == RNS8_BACKEND_HIP_DIRECT) {
      rns8_status status = ensure_device_residues_current(const_cast<rns8_matrix&>(*C));
      if (status != RNS8_SUCCESS) {
        return status;
      }
      return rns8::detail::hip_direct_export_exact_wide_unsigned_limbs_device(
          ctx->device_id,
          C->hip_residues,
          &const_cast<rns8_matrix*>(C)->hip_export_buffer,
          &const_cast<rns8_matrix*>(C)->hip_export_bytes,
          &const_cast<rns8_matrix*>(C)->hip_status_buffer,
          &const_cast<rns8_matrix*>(C)->hip_status_bytes,
          plan->desc.m,
          plan->desc.n,
          plan->prefix,
          dst,
          ld,
          limb_count);
    }
    const rns8_status sync_status = ensure_host_residues_current(*C);
    if (sync_status != RNS8_SUCCESS) {
      return sync_status;
    }
    for (int64_t row = 0; row < plan->desc.m; ++row) {
      for (int64_t col = 0; col < plan->desc.n; ++col) {
        const std::vector<int8_t> residues = gather_cell_residues(*C, row, col, plan->prefix);
        const rns8_status status = rns8::detail::export_exact_wide_unsigned_limbs(
            residues, plan->prefix, dst + (row * ld + col) * limb_count, limb_count);
        if (status != RNS8_SUCCESS) {
          return status;
        }
      }
    }
    return RNS8_SUCCESS;
  });
}

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
    if (!ctx || !desc || !A || !B || !C || desc->semantics != RNS8_BOUNDED_I64) {
      return RNS8_INVALID_ARGUMENT;
    }

    rns8_plan* plan = nullptr;
    rns8_status status = rns8_create_plan(ctx, desc, &plan);
    if (status != RNS8_SUCCESS) {
      return status;
    }

    rns8_matrix* a_matrix = nullptr;
    rns8_matrix* b_matrix = nullptr;
    rns8_matrix* c_matrix = nullptr;
    rns8_workspace* workspace = nullptr;
    const rns8_matrix_desc a_desc =
        make_matrix_desc(desc->m, desc->k, desc->semantics, desc->bound_kind, plan->prefix);
    const rns8_matrix_desc b_desc =
        make_matrix_desc(desc->k, desc->n, desc->semantics, desc->bound_kind, plan->prefix);
    const rns8_matrix_desc c_desc =
        make_matrix_desc(desc->m, desc->n, desc->semantics, desc->bound_kind, plan->prefix);

    status = rns8_create_matrix(ctx, &a_desc, &a_matrix);
    if (status == RNS8_SUCCESS) status = rns8_create_matrix(ctx, &b_desc, &b_matrix);
    if (status == RNS8_SUCCESS) status = rns8_create_matrix(ctx, &c_desc, &c_matrix);
    if (status == RNS8_SUCCESS) status = rns8_create_workspace(ctx, plan, &workspace);
    if (status == RNS8_SUCCESS) status = rns8_pack_i64(ctx, a_matrix, A, lda, 1);
    if (status == RNS8_SUCCESS) status = rns8_pack_i64(ctx, b_matrix, B, ldb, 1);
    if (status == RNS8_SUCCESS) status = rns8_gemm_rns(ctx, plan, a_matrix, b_matrix, c_matrix, workspace);
    if (status == RNS8_SUCCESS) status = rns8_export_i64(ctx, plan, c_matrix, C, ldc);

    rns8_destroy_workspace(workspace);
    rns8_destroy_matrix(c_matrix);
    rns8_destroy_matrix(b_matrix);
    rns8_destroy_matrix(a_matrix);
    rns8_destroy_plan(plan);
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
    if (!ctx || !desc || !A || !B || !C || desc->semantics != RNS8_BOUNDED_U64) {
      return RNS8_INVALID_ARGUMENT;
    }

    rns8_plan* plan = nullptr;
    rns8_status status = rns8_create_plan(ctx, desc, &plan);
    if (status != RNS8_SUCCESS) {
      return status;
    }

    rns8_matrix* a_matrix = nullptr;
    rns8_matrix* b_matrix = nullptr;
    rns8_matrix* c_matrix = nullptr;
    rns8_workspace* workspace = nullptr;
    const rns8_matrix_desc a_desc =
        make_matrix_desc(desc->m, desc->k, desc->semantics, desc->bound_kind, plan->prefix);
    const rns8_matrix_desc b_desc =
        make_matrix_desc(desc->k, desc->n, desc->semantics, desc->bound_kind, plan->prefix);
    const rns8_matrix_desc c_desc =
        make_matrix_desc(desc->m, desc->n, desc->semantics, desc->bound_kind, plan->prefix);

    status = rns8_create_matrix(ctx, &a_desc, &a_matrix);
    if (status == RNS8_SUCCESS) status = rns8_create_matrix(ctx, &b_desc, &b_matrix);
    if (status == RNS8_SUCCESS) status = rns8_create_matrix(ctx, &c_desc, &c_matrix);
    if (status == RNS8_SUCCESS) status = rns8_create_workspace(ctx, plan, &workspace);
    if (status == RNS8_SUCCESS) status = rns8_pack_u64(ctx, a_matrix, A, lda, 1);
    if (status == RNS8_SUCCESS) status = rns8_pack_u64(ctx, b_matrix, B, ldb, 1);
    if (status == RNS8_SUCCESS) status = rns8_gemm_rns(ctx, plan, a_matrix, b_matrix, c_matrix, workspace);
    if (status == RNS8_SUCCESS) status = rns8_export_u64(ctx, plan, c_matrix, C, ldc);

    rns8_destroy_workspace(workspace);
    rns8_destroy_matrix(c_matrix);
    rns8_destroy_matrix(b_matrix);
    rns8_destroy_matrix(a_matrix);
    rns8_destroy_plan(plan);
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
    if (!ctx || !desc || !A || !B || !C || !rns8::detail::valid_abi(desc->struct_size, desc->abi_version, sizeof(*desc))) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (desc->semantics != RNS8_WRAP_U64_MOD_2_64) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (ctx->backend != RNS8_BACKEND_WRAP64_BYTE_LIMB) {
      return RNS8_UNSUPPORTED_BACKEND;
    }
    const rns8_backend_kind requested = effective_backend(desc->requested_backend, ctx->backend);
    if (requested != RNS8_BACKEND_WRAP64_BYTE_LIMB) {
      return RNS8_UNSUPPORTED_BACKEND;
    }
    if (desc->bound_kind != RNS8_BOUND_NONE || desc->bound != 0 || desc->max_prefix != 0) {
      return RNS8_UNSUPPORTED_BACKEND;
    }
    if (desc->tile_m != 0 && (desc->tile_m < 64 || desc->tile_m > 512)) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (desc->tile_n != 0 && (desc->tile_n < 64 || desc->tile_n > 512)) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (!valid_matrix_access(desc->m, desc->k, lda) || !valid_matrix_access(desc->k, desc->n, ldb) ||
        !valid_matrix_access(desc->m, desc->n, ldc)) {
      return RNS8_INVALID_ARGUMENT;
    }

    for (int64_t row = 0; row < desc->m; ++row) {
      for (int64_t col = 0; col < desc->n; ++col) {
        C[row * ldc + col] =
            rns8::detail::wrap64_byte_limb_gemm_cell(A, lda, B, ldb, row, col, desc->k);
      }
    }
    return RNS8_SUCCESS;
  });
}
