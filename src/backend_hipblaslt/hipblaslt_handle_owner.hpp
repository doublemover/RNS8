#pragma once
#include "hipblaslt_backend.hpp"
#include "hipblaslt_headers.hpp"
#include "core/hip_runtime_handles.hpp"

namespace rns8::detail {
enum class HipblasLtResource { context,operation,layout,preference };
template<class Handle,auto Destroy,HipblasLtResource Resource>
struct hipblaslt_handle_operations {
  using handle_type=Handle;
  using status_type=hipblasStatus_t;
  static constexpr auto success=HIPBLAS_STATUS_SUCCESS,out_of_memory=HIPBLAS_STATUS_ALLOC_FAILED,
    invalid_argument=HIPBLAS_STATUS_INVALID_VALUE;
  static constexpr auto kind=hip_deferred_allocation_kind::library_handle;
  static status_type current_device(int& device) noexcept {
    return hipGetDevice(&device)==hipSuccess?success:HIPBLAS_STATUS_INTERNAL_ERROR;
  }
  static status_type destroy(int device,Handle handle) noexcept {
    hip_scoped_device selected(device);
    if(selected.status()!=hipSuccess)return HIPBLAS_STATUS_INTERNAL_ERROR;
    if constexpr(Resource==HipblasLtResource::preference)++hipblaslt_metadata_state::metadata_counters.preference_destroy_calls;
    else if constexpr(Resource!=HipblasLtResource::context)++hipblaslt_metadata_state::metadata_counters.descriptor_destroy_calls;
    return Destroy(handle);
  }
  static void defer(hip_deferred_buffer_cleanup* record) noexcept {hip_direct_defer_buffer_cleanup(record);}
};
using hipblaslt_unique_context=basic_hip_handle<hipblaslt_handle_operations<hipblasLtHandle_t,hipblasLtDestroy,HipblasLtResource::context>>;
using hipblaslt_unique_operation=basic_hip_handle<hipblaslt_handle_operations<hipblasLtMatmulDesc_t,hipblasLtMatmulDescDestroy,HipblasLtResource::operation>>;
using hipblaslt_unique_layout=basic_hip_handle<hipblaslt_handle_operations<hipblasLtMatrixLayout_t,hipblasLtMatrixLayoutDestroy,HipblasLtResource::layout>>;
using hipblaslt_unique_preference=basic_hip_handle<hipblaslt_handle_operations<hipblasLtMatmulPreference_t,hipblasLtMatmulPreferenceDestroy,HipblasLtResource::preference>>;
}
