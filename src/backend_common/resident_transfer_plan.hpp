#pragma once
#include "address_domain.hpp"
#include "operation_support.hpp"

namespace rns8::detail {
struct ResidentTransferLayout {
  rns8_status status=RNS8_INVALID_ARGUMENT;
  uint64_t device_bytes=0,pinned_bytes=0,payload_bytes=0;
};
inline ResidentTransferLayout resident_import_layout(int64_t rows,int64_t cols,uint32_t prefix) {
  ResidentTransferLayout out;AddressDomain payload;
  if(!prefix || prefix>20 || !checked_host_plane_address(rows,cols,cols,prefix,1,payload))return out;
  if(payload.byte_extent>uint64_t(PTRDIFF_MAX)-16) {out.status=RNS8_RANGE_ERROR;return out;}
  out.status=RNS8_SUCCESS;out.payload_bytes=payload.byte_extent;
  out.device_bytes=out.payload_bytes+16;out.pinned_bytes=4;return out;
}
inline ResidentTransferLayout resident_lift_layout(int64_t cells,uint32_t source,uint32_t target) {
  ResidentTransferLayout out;
  if(source==target && source && source<=20 && cells>0) {out.status=RNS8_SUCCESS;return out;}
  const auto support=query_exact_lift_support(cells,source,target,UINT64_MAX);
  out.status=support.status;
  if(out.status==RNS8_SUCCESS) {
    out.device_bytes=support.scratch_bytes;out.pinned_bytes=4;out.payload_bytes=support.payload_bytes;
  }
  return out;
}
inline ResidentTransferLayout resident_export_layout(int64_t rows,int64_t cols,uint32_t limbs) {
  ResidentTransferLayout out;AddressDomain payload;
  if(!limbs || limbs>32 || !checked_host_plane_address(rows,cols,cols,1,uint64_t(limbs)*8,payload))return out;
  if(payload.byte_extent>uint64_t(PTRDIFF_MAX)-8) {out.status=RNS8_RANGE_ERROR;return out;}
  out.status=RNS8_SUCCESS;out.payload_bytes=payload.byte_extent;
  out.device_bytes=out.payload_bytes+4;
  // Status first, then an aligned compact limb array. Only observation copies
  // this owned payload to the caller, after every graph status has passed.
  out.pinned_bytes=out.payload_bytes+8;return out;
}
}
