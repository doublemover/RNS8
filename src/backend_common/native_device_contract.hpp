#pragma once
#include "address_domain.hpp"
#include "rns8/rns8.h"
namespace rns8::detail {
inline rns8_status query_native_device_contract(const rns8_native_device_desc& d,rns8_native_device_resources& out) {
  AddressDomain view,compact;
  if(d.struct_size!=sizeof(d) || d.abi_version!=RNS8_ABI_VERSION ||
     (d.element_type!=RNS8_NATIVE_DEVICE_I64 && d.element_type!=RNS8_NATIVE_DEVICE_U64) ||
     (d.access!=RNS8_NATIVE_DEVICE_READ && d.access!=RNS8_NATIVE_DEVICE_WRITE) ||
     !d.allocation_base || !d.owner_identity || !d.generation || !d.acquire || !d.release ||
     d.device_id<0 || d.policy!=RNS8_NATIVE_DEVICE_BORROW || d.producer_complete!=1 || d.rows<=0 || d.cols<=0 || d.ld<d.cols ||
     d.byte_offset%8 || reinterpret_cast<uintptr_t>(d.allocation_base)%8 ||
     !checked_host_plane_address(d.rows,d.cols,d.ld,1,8,view) ||
     !checked_host_plane_address(d.rows,d.cols,d.cols,1,8,compact) ||
     d.byte_offset>d.allocation_bytes || view.byte_extent>d.allocation_bytes-d.byte_offset ||
     d.allocation_bytes>UINTPTR_MAX-reinterpret_cast<uintptr_t>(d.allocation_base) ||
     compact.byte_extent>uint64_t(PTRDIFF_MAX)-4) return RNS8_INVALID_ARGUMENT;
  if(!qualified_i32_matrix_address(d.rows,d.cols,d.ld,1,8)) return RNS8_UNSUPPORTED_BACKEND;
  out={};out.struct_size=sizeof(out);out.abi_version=RNS8_ABI_VERSION;
  out.view_span_bytes=view.byte_extent;out.compact_native_bytes=compact.byte_extent;
  out.export_scratch_bytes=compact.byte_extent+4;out.verified_import_scratch_bytes=8;return RNS8_SUCCESS;
}
}
