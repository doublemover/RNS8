#pragma once
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include "rns8/rns8.h"
#include "balanced_crt.hpp"
#include "address_domain.hpp"

#if defined(__HIPCC__)
#define RNS8_CHECKED_CRT_HD __host__ __device__
#else
#define RNS8_CHECKED_CRT_HD
#endif

namespace rns8::detail::checked_crt {
using balanced_crt::Wide;
struct Decode {
  Wide product{}, half{}, bound{};
  uint32_t limbs=0, mode=0, signed_source=0;
};

// Inputs are canonical complete-root values. Subgroup residues are never
// centered or interpreted as signed integer partial answers.
RNS8_CHECKED_CRT_HD inline bool decode(Wide root,const Decode& contract,uint64_t* output) {
  using namespace balanced_crt;
  if(!contract.limbs || contract.limbs>32 || contract.mode>2 || compare(root,contract.product)>=0)return false;
  const bool negative=contract.signed_source && compare(root,contract.half)>0;
  const auto magnitude=negative?subtract(contract.product,root):root;
  if(compare(magnitude,contract.bound)>0)return false;
  const bool signed_output=contract.mode==0;
  const auto value=contract.mode==2?root:magnitude;
  const bool output_negative=signed_output && negative;
  if(contract.mode==1 && negative)return false;
  if(contract.limbs<3) {
    Wide limit{};
    const unsigned width=64*contract.limbs-(signed_output?1:0);
    limit.limb[width/64]=uint64_t{1}<<(width%64);
    const int relation=compare(value,limit);
    if(relation>0 || (relation==0 && !output_negative))return false;
  }
  const auto encoded=output_negative?subtract(Wide{},value):value;
  for(uint32_t i=0;i<contract.limbs;++i)
    output[i]=i<3?encoded.limb[i]:(output_negative?UINT64_MAX:0);
  return true;
}

struct Resources {
  rns8_status status=RNS8_INVALID_ARGUMENT;
  uint64_t host_bytes=0,device_bytes=0,root_bytes=0,output_bytes=0,group_bytes=0,total_bytes=0;
};
// Payload reservation, not allocator bookkeeping or achieved physical memory.
// Every GPU policy uses the same canonical-root -> checked integer boundary.
inline Resources resources(int64_t cells,uint32_t prefix,uint32_t limbs,uint32_t algorithm,bool gpu,uint64_t budget) {
  Resources out;
  if(cells<=0 || !prefix || prefix>20 || !limbs || limbs>32 || algorithm>4)return out;
  if(gpu && !qualified_i32_matrix_address(1,cells,cells,prefix,24)) {
    out.status=RNS8_UNSUPPORTED_BACKEND;return out;
  }
  const auto count=uint64_t(cells),limit=uint64_t(PTRDIFF_MAX);
  if(count>limit/(8*uint64_t(limbs)) || count>limit/prefix ||
      (gpu && (cells>INT32_MAX-255 || count>limit/24))) {
    out.status=RNS8_RANGE_ERROR;return out;
  }
  out.output_bytes=count*limbs*8;
  if(algorithm==4) {
    if(count>limit/(((prefix+7)/8)*8)) {out.status=RNS8_RANGE_ERROR;return out;}
    out.group_bytes=count*((prefix+7)/8)*8;
  }
  out.host_bytes=out.output_bytes;
  if(gpu) {
    out.root_bytes=count*24;
    out.device_bytes=out.root_bytes;
    for(auto bytes:{out.output_bytes,out.group_bytes,uint64_t{4}}) {
      if(bytes>limit-out.device_bytes) {out.status=RNS8_RANGE_ERROR;return out;}
      out.device_bytes+=bytes;
    }
  } else {
    if(out.group_bytes>limit-out.host_bytes) {out.status=RNS8_RANGE_ERROR;return out;}
    out.host_bytes+=out.group_bytes;
  }
  if(out.device_bytes>limit-out.host_bytes) {out.status=RNS8_RANGE_ERROR;return out;}
  out.total_bytes=out.host_bytes+out.device_bytes;
  out.status=out.total_bytes>budget?RNS8_WORKSPACE_TOO_SMALL:RNS8_SUCCESS;
  return out;
}
} // namespace rns8::detail::checked_crt
#undef RNS8_CHECKED_CRT_HD
