#pragma once
#include "native_device_contract.hpp"
namespace rns8::detail {
struct PinnedHostResources {uint64_t pinned=0,compact=0,element_bytes=0;};
inline rns8_status query_pinned_host_contract(const rns8_pinned_host_desc& d,PinnedHostResources& out) noexcept {
  if(d.struct_size!=sizeof(d) || d.abi_version!=RNS8_ABI_VERSION || d.flags || d.reserved ||
      d.rows<=0 || d.cols<=0 || d.ld<d.cols || (d.access!=RNS8_NATIVE_DEVICE_READ && d.access!=RNS8_NATIVE_DEVICE_WRITE))return RNS8_INVALID_ARGUMENT;
  const bool limbs=native_device_limb_type(d.element_type);
  if((!limbs && d.element_type!=RNS8_NATIVE_DEVICE_I64 && d.element_type!=RNS8_NATIVE_DEVICE_U64) ||
      (limbs?(!d.limb_count || d.limb_count>32 || d.access!=RNS8_NATIVE_DEVICE_WRITE):d.limb_count!=0))return RNS8_INVALID_ARGUMENT;
  const uint64_t width=limbs?uint64_t(d.limb_count)*8:8;
  // Allocate the complete padded rectangle, including final-row padding.
  if(uint64_t(d.rows)>uint64_t(PTRDIFF_MAX)/uint64_t(d.ld)/width)return RNS8_RANGE_ERROR;
  if(!qualified_i32_matrix_address(d.rows,d.cols,d.cols,1,width))return RNS8_UNSUPPORTED_BACKEND;
  out={uint64_t(d.rows)*uint64_t(d.ld)*width,uint64_t(d.rows)*uint64_t(d.cols)*width,width};return RNS8_SUCCESS;
}
inline bool pinned_host_transfer_bytes(const PinnedHostResources& r,uint32_t prefix,bool importing,bool retain,
    uint64_t& device,uint64_t& pinned,uint64_t& total) noexcept {
  if(!prefix || prefix>20 || !r.element_bytes || !r.compact || r.compact%r.element_bytes ||
      r.compact>r.pinned || r.pinned>SIZE_MAX-4 || (retain && !importing))return false;
  const uint64_t cells=r.compact/r.element_bytes;
  if(importing ? cells>(SIZE_MAX-16)/prefix : r.compact>SIZE_MAX-4)return false;
  const uint64_t conversion=importing?cells*prefix+16:r.compact+4;
  if(r.compact>SIZE_MAX-conversion)return false;
  const uint64_t device_bytes=r.compact+conversion,pinned_bytes=r.pinned+4;
  if(device_bytes>SIZE_MAX-pinned_bytes || (retain && cells>(SIZE_MAX-device_bytes-pinned_bytes)/8))return false;
  device=device_bytes;pinned=pinned_bytes;total=device+pinned+(retain?cells*8:0);return true;
}
}
