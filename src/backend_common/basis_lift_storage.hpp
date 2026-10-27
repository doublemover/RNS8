#pragma once
#include "basis_packet.hpp"
#include "rns8/status.h"
#include <cstdint>
namespace rns8::detail {
struct BasisLiftStorage {
  rns8_status status=RNS8_INVALID_ARGUMENT;
  uint64_t payload_bytes=0,flag_offset=0,packet_offset=0,scratch_bytes=0;
};
inline BasisLiftStorage basis_lift_storage(int64_t cells,uint32_t source,uint32_t target) {
  BasisLiftStorage out;
  if(cells<=0 || !source || source>=target || target>20)return out;
  const auto limit=uint64_t(PTRDIFF_MAX);
  if(cells>INT32_MAX-256 || uint64_t(cells)>(limit-259)/target){out.status=RNS8_RANGE_ERROR;return out;}
  out.payload_bytes=uint64_t(cells)*target;out.flag_offset=(out.payload_bytes+3)&~UINT64_C(3);
  out.packet_offset=(out.flag_offset+4+255)&~UINT64_C(255);
  const auto packet_bytes=(sizeof(BasisPacket)+255)&~UINT64_C(255);
  if(packet_bytes>limit-out.packet_offset){out.status=RNS8_RANGE_ERROR;return out;}
  out.scratch_bytes=out.packet_offset+packet_bytes;out.status=RNS8_SUCCESS;return out;
}
}
