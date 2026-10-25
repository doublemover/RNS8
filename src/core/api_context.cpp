#include "core/api_internal.hpp"

using namespace rns8::detail::api;

namespace {

int default_visible_hip_context_device(int requested_device_id) {
  return requested_device_id < 0 ? 0 : requested_device_id;
}

}  // namespace

namespace rns8::detail::api {
rns8_status begin_synchronous_lease(rns8_context& ctx,const rns8_matrix& a,const rns8_matrix& b,rns8_matrix& c,rns8_workspace& workspace,size_t& token) {
  if(a.in_flight_writer || b.in_flight_writer || c.in_flight_writer || c.in_flight_readers || workspace.in_flight)return RNS8_BUSY;
  if(a.in_flight_readers==UINT64_MAX || b.in_flight_readers==UINT64_MAX)return RNS8_RANGE_ERROR;
  token=ctx.failed_work_leases.size();ctx.failed_work_leases.push_back({&a,&b,&c,&workspace});
  ++a.in_flight_readers;if(&a!=&b)++b.in_flight_readers;c.in_flight_writer=true;workspace.in_flight=true;return RNS8_SUCCESS;
}
bool finish_synchronous_lease(rns8_context& ctx,size_t token,rns8_status status,bool written) noexcept {
  if(token>=ctx.failed_work_leases.size() || !ctx.failed_work_leases[token].a)return true;
  auto& lease=ctx.failed_work_leases[token];
  if(status!=RNS8_SUCCESS && written){lease.c->in_flight_writer=false;invalidate_output_currentness(*lease.c);lease.c->in_flight_writer=true;}
  if((status==RNS8_BACKEND_FAILURE || status==RNS8_INTERNAL_ERROR) && rns8::detail::hip_direct_synchronize(ctx.device_id)!=RNS8_SUCCESS){ctx.hip_failure_poisoned=true;return false;}
  --lease.a->in_flight_readers;if(lease.a!=lease.b)--lease.b->in_flight_readers;lease.c->in_flight_writer=false;lease.workspace->in_flight=false;lease={};
  while(!ctx.failed_work_leases.empty() && !ctx.failed_work_leases.back().a)ctx.failed_work_leases.pop_back();return true;
}
}
rns8_status rns8_drain_failed_work(rns8_context* ctx) {
  if(!ctx)return RNS8_INVALID_ARGUMENT;
  if(ctx->device_id<0)return ctx->failed_work_leases.empty()?RNS8_SUCCESS:RNS8_UNSUPPORTED_BACKEND;
  const auto status=rns8::detail::hip_direct_synchronize(ctx->device_id);if(status!=RNS8_SUCCESS)return status;
  for(size_t i=0;i<ctx->failed_work_leases.size();++i)finish_synchronous_lease(*ctx,i,RNS8_SUCCESS,false);
  // A successful drain releases memory, not the failed context's execution authority.
  return RNS8_SUCCESS;
}

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
      ctx->device_id = default_visible_hip_context_device(device_id);
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
      ctx->device_id = default_visible_hip_context_device(device_id);
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
      ctx->device_id = default_visible_hip_context_device(device_id);
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
      ctx->device_id = default_visible_hip_context_device(device_id);
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
      ctx->device_id = default_visible_hip_context_device(device_id);
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
      ctx->device_id = default_visible_hip_context_device(device_id);
      ctx->device_info.struct_size = sizeof(ctx->device_info);
      ctx->device_info.abi_version = RNS8_ABI_VERSION;
      const rns8_status status = rns8::detail::rocwmma_probe(ctx->device_id, ctx->device_info);
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

    if (requested == RNS8_BACKEND_AMDGPU_BUILTINS) {
#if defined(RNS8_ENABLE_AMDGPU_BUILTINS) && RNS8_ENABLE_AMDGPU_BUILTINS && \
    defined(RNS8_AMDGPU_BUILTIN_KERNELS_AVAILABLE) && RNS8_AMDGPU_BUILTIN_KERNELS_AVAILABLE
      ctx->backend = RNS8_BACKEND_AMDGPU_BUILTINS;
      ctx->device_id = default_visible_hip_context_device(device_id);
      ctx->device_info.struct_size = sizeof(ctx->device_info);
      ctx->device_info.abi_version = RNS8_ABI_VERSION;
      const rns8_status status = rns8::detail::amdgpu_builtins_probe(ctx->device_id, ctx->device_info);
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
  if(ctx && !ctx->failed_work_leases.empty()){const auto status=rns8_drain_failed_work(ctx);if(status!=RNS8_SUCCESS)return status;}
  if (ctx && ctx->prepared_references) return RNS8_BUSY;
  if (ctx) {
    const auto status = drain_failed_ragged_cleanup(*ctx);
    if (status != RNS8_SUCCESS) return status;
  }
  if (ctx) {
    const auto status = drain_failed_native_cleanup(*ctx);
    if (status != RNS8_SUCCESS) return status;
  }
  while (ctx && ctx->failed_matrix_cleanup) {
    auto* matrix = ctx->failed_matrix_cleanup;
    const auto status = free_hip_matrix_storage(*matrix);
    if (status != RNS8_SUCCESS) return status;
    ctx->failed_matrix_cleanup = matrix->failed_cleanup_next;
    delete matrix;
  }
  if (ctx && ctx->device_id>=0) {
    const auto status=rns8::detail::hip_direct_collect_deferred_buffers(ctx->device_id);
    if(status!=RNS8_SUCCESS) return status;
  }
  if (ctx && ctx->hipblaslt_handle) {
#if defined(RNS8_ENABLE_HIPBLASLT) && RNS8_ENABLE_HIPBLASLT
    const rns8_status status = rns8::detail::hipblaslt_destroy_context(ctx->device_id, ctx->hipblaslt_handle);
    if (status != RNS8_SUCCESS) return status;
    ctx->hipblaslt_handle = nullptr;
    delete ctx;
    return status;
#else
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
