#pragma once
#include "operation_support.hpp"
#include "address_domain.hpp"
namespace rns8::detail {
struct ExactLiftWorkspaceView {int* status=nullptr;int8_t* suffix=nullptr;uint64_t suffix_bytes=0,required_bytes=0;};
// Status-first layout is independent of odd suffix sizes. The enclosing owner
// supplies int alignment and retains the complete region through failed work.
inline rns8_status exact_lift_workspace_view(void* scratch,uint64_t capacity,
    const void* residues,int64_t cells,uint32_t source,uint32_t target,ExactLiftWorkspaceView& out) {
  const auto support=query_exact_lift_support(cells,source,target,UINT64_MAX);
  if(support.status!=RNS8_SUCCESS)return support.status;
  if(!scratch || !residues || reinterpret_cast<uintptr_t>(scratch)%alignof(int))return RNS8_INVALID_ARGUMENT;
  if(capacity<support.scratch_bytes)return RNS8_WORKSPACE_TOO_SMALL;
  if(!nonoverlapping_byte_ranges(scratch,support.scratch_bytes,residues,uint64_t(cells)*target))return RNS8_INVALID_ARGUMENT;
  out={static_cast<int*>(scratch),static_cast<int8_t*>(scratch)+sizeof(int),support.payload_bytes,support.scratch_bytes};
  return RNS8_SUCCESS;
}
}
