#include "core/api_internal.hpp"

using namespace rns8::detail::api;

rns8_status rns8_create_context(int device_id, const rns8_context_options* options, rns8_context** out) {
  return guard_api([&]() -> rns8_status {
    if (!out) {
      return RNS8_INVALID_ARGUMENT;
    }
    *out = nullptr;

    rns8_backend_kind requested = RNS8_BACKEND_CPU_REFERENCE;
    bool requested_auto = false;
    if (options) {
      if (!rns8::detail::valid_abi(options->struct_size, options->abi_version, sizeof(*options))) {
        return RNS8_INVALID_ARGUMENT;
      }
      if (options->flags != 0) {
        return RNS8_INVALID_ARGUMENT;
      }
      requested_auto = options->requested_backend == RNS8_BACKEND_AUTO;
      requested = requested_auto ? RNS8_BACKEND_AUTO : options->requested_backend;
    }

    auto* ctx = new (std::nothrow) rns8_context();
    if (!ctx) {
      return RNS8_INTERNAL_ERROR;
    }

    if (requested_auto) {
      ctx->auto_backend_selection = true;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
      ctx->backend = RNS8_BACKEND_HIP_DIRECT;
      ctx->device_id = device_id < 0 ? 0 : device_id;
      ctx->device_info.struct_size = sizeof(ctx->device_info);
      ctx->device_info.abi_version = RNS8_ABI_VERSION;
      const rns8_status hip_status = rns8::detail::hip_direct_probe(ctx->device_id, ctx->device_info);
      if (hip_status == RNS8_SUCCESS) {
        *out = ctx;
        return RNS8_SUCCESS;
      }
#endif
      ctx->backend = RNS8_BACKEND_CPU_REFERENCE;
      ctx->device_id = -1;
      ctx->device_info.struct_size = sizeof(ctx->device_info);
      ctx->device_info.abi_version = RNS8_ABI_VERSION;
      rns8::detail::fill_cpu_device_info(ctx->device_info);
      *out = ctx;
      return RNS8_SUCCESS;
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

    if (requested == RNS8_BACKEND_HIP_VECTOR_ALU_INT64) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
      ctx->backend = RNS8_BACKEND_HIP_VECTOR_ALU_INT64;
      ctx->device_id = device_id < 0 ? 0 : device_id;
      ctx->device_info.struct_size = sizeof(ctx->device_info);
      ctx->device_info.abi_version = RNS8_ABI_VERSION;
      const rns8_status status = rns8::detail::vector_alu_probe(ctx->device_id, ctx->device_info);
      if (status != RNS8_SUCCESS) {
        delete ctx;
        return status;
      }
      ctx->device_info.backend = RNS8_BACKEND_HIP_VECTOR_ALU_INT64;
      *out = ctx;
      return RNS8_SUCCESS;
#else
      delete ctx;
      return RNS8_UNSUPPORTED_BACKEND;
#endif
    }

    if (requested == RNS8_BACKEND_HIPBLASLT) {
#if defined(RNS8_ENABLE_HIPBLASLT) && RNS8_ENABLE_HIPBLASLT
      ctx->backend = RNS8_BACKEND_HIPBLASLT;
      ctx->device_id = device_id < 0 ? 0 : device_id;
      ctx->device_info.struct_size = sizeof(ctx->device_info);
      ctx->device_info.abi_version = RNS8_ABI_VERSION;
      const rns8_status status = rns8::detail::hipblaslt_create_context(
          ctx->device_id, ctx->device_info, &ctx->hipblaslt_handle, ctx->hipblaslt_library_version);
      if (status != RNS8_SUCCESS) {
        delete ctx;
        return status;
      }
      *out = ctx;
      return RNS8_SUCCESS;
#else
      delete ctx;
      return RNS8_UNSUPPORTED_BACKEND;
#endif
    }

    if (requested == RNS8_BACKEND_CK) {
#if defined(RNS8_ENABLE_CK) && RNS8_ENABLE_CK
      ctx->backend = RNS8_BACKEND_CK;
      ctx->device_id = device_id < 0 ? 0 : device_id;
      ctx->device_info.struct_size = sizeof(ctx->device_info);
      ctx->device_info.abi_version = RNS8_ABI_VERSION;
      const rns8_status status = rns8::detail::ck_probe(ctx->device_id, ctx->device_info);
      if (status != RNS8_SUCCESS) {
        delete ctx;
        return status;
      }
      *out = ctx;
      return RNS8_SUCCESS;
#else
      delete ctx;
      return RNS8_UNSUPPORTED_BACKEND;
#endif
    }

    if (requested == RNS8_BACKEND_ROCWMMA) {
#if defined(RNS8_ENABLE_ROCWMMA) && RNS8_ENABLE_ROCWMMA
      ctx->backend = RNS8_BACKEND_ROCWMMA;
      ctx->device_id = device_id < 0 ? 0 : device_id;
      ctx->device_info.struct_size = sizeof(ctx->device_info);
      ctx->device_info.abi_version = RNS8_ABI_VERSION;
      const rns8_status status = rns8::detail::wmma_probe(ctx->device_id, ctx->device_info);
      if (status != RNS8_SUCCESS) {
        delete ctx;
        return status;
      }
      *out = ctx;
      return RNS8_SUCCESS;
#else
      delete ctx;
      return RNS8_UNSUPPORTED_BACKEND;
#endif
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
  if (ctx && ctx->hipblaslt_handle) {
#if defined(RNS8_ENABLE_HIPBLASLT) && RNS8_ENABLE_HIPBLASLT
    const rns8_status status = rns8::detail::hipblaslt_destroy_context(ctx->device_id, ctx->hipblaslt_handle);
    ctx->hipblaslt_handle = nullptr;
    delete ctx;
    return status;
#else
    delete ctx;
    return RNS8_UNSUPPORTED_BACKEND;
#endif
  }
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
