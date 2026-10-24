#pragma once
#include "address_domain.hpp"
#include "rns8/rns8.h"
namespace rns8::detail {
inline bool native_device_limb_type(rns8_native_device_type type) {
  return type==RNS8_NATIVE_DEVICE_SIGNED_LIMBS || type==RNS8_NATIVE_DEVICE_UNSIGNED_LIMBS;
}
inline bool native_device_signed_type(rns8_native_device_type type) {
  return type==RNS8_NATIVE_DEVICE_I64 || type==RNS8_NATIVE_DEVICE_SIGNED_LIMBS;
}
inline rns8_status query_native_device_contract(const rns8_native_device_desc& d,rns8_native_device_resources& out) {
  if(d.struct_size!=sizeof(d) || d.abi_version!=RNS8_ABI_VERSION) return RNS8_INVALID_ARGUMENT;
  AddressDomain view,compact;
  const bool limbs=native_device_limb_type(d.element_type);
  const uint64_t element_bytes=limbs?uint64_t(d.limb_count)*8:8;
  if((d.element_type!=RNS8_NATIVE_DEVICE_I64 && d.element_type!=RNS8_NATIVE_DEVICE_U64 && !limbs) ||
     (limbs ? (!d.limb_count || d.limb_count>32 || d.access!=RNS8_NATIVE_DEVICE_WRITE ||
               d.policy!=RNS8_NATIVE_DEVICE_BORROW) : d.limb_count!=0) ||
     (d.access!=RNS8_NATIVE_DEVICE_READ && d.access!=RNS8_NATIVE_DEVICE_WRITE) ||
     !d.allocation_base || !d.owner_identity || !d.generation || !d.acquire || !d.release ||
     d.device_id<0 || (d.policy!=RNS8_NATIVE_DEVICE_BORROW && d.policy!=RNS8_NATIVE_DEVICE_COPY_OWNED) ||
     (d.policy==RNS8_NATIVE_DEVICE_COPY_OWNED && d.access!=RNS8_NATIVE_DEVICE_READ) ||
     (d.policy==RNS8_NATIVE_DEVICE_BORROW && d.snapshot_budget_bytes!=0) || d.producer_complete!=1 || d.rows<=0 || d.cols<=0 || d.ld<d.cols ||
     d.byte_offset%8 || reinterpret_cast<uintptr_t>(d.allocation_base)%8 ||
     !checked_host_plane_address(d.rows,d.cols,d.ld,1,element_bytes,view) ||
     !checked_host_plane_address(d.rows,d.cols,d.cols,1,element_bytes,compact) ||
     d.byte_offset>d.allocation_bytes || view.byte_extent>d.allocation_bytes-d.byte_offset ||
     d.allocation_bytes>UINTPTR_MAX-reinterpret_cast<uintptr_t>(d.allocation_base) ||
     compact.byte_extent>uint64_t(PTRDIFF_MAX)-4) return RNS8_INVALID_ARGUMENT;
  if(!qualified_i32_matrix_address(d.rows,d.cols,d.ld,1,element_bytes)) return RNS8_UNSUPPORTED_BACKEND;
  if(d.policy==RNS8_NATIVE_DEVICE_COPY_OWNED && d.snapshot_budget_bytes<compact.byte_extent)
    return RNS8_WORKSPACE_TOO_SMALL;
  out={};out.struct_size=sizeof(out);out.abi_version=RNS8_ABI_VERSION;
  out.view_span_bytes=view.byte_extent;out.compact_native_bytes=compact.byte_extent;
  out.export_scratch_bytes=compact.byte_extent+4;out.verified_import_scratch_bytes=limbs?0:8;
  out.snapshot_bytes=d.policy==RNS8_NATIVE_DEVICE_COPY_OWNED?compact.byte_extent:0;return RNS8_SUCCESS;
}
}
