#pragma once
#include "balanced_crt.hpp"

#if defined(__HIPCC__)
#define RNS8_GROUP_CRT_HD __host__ __device__
#else
#define RNS8_GROUP_CRT_HD
#endif
namespace rns8::detail::group_crt {
// Each leaf is an unsigned canonical residue modulo its own product. It is
// never a signed integer partial answer. Eight byte moduli have product <2^64
// in the default basis; the complete root remains below 2^160.
struct Group {
  uint64_t product=1;
  uint32_t channel_mask=0,count=0;
  uint16_t moduli[8]{},inverses[8]{};
  uint8_t channels[8]{};
};
struct Plan {
  Group groups[4]{};
  balanced_crt::Merge merges[3]{};
  uint32_t prefix=0,group_count=0,merge_count=0;
};
#if !defined(__HIP_DEVICE_COMPILE__)
// Partition assigns every active channel exactly once to a nonempty group.
// Builders validate the complete basis, leaf widths and every merge inverse.
Plan make_plan(uint32_t prefix,const uint8_t* partition,uint32_t group_count);
const Plan& plan(uint32_t prefix);
#endif

RNS8_GROUP_CRT_HD inline uint64_t reconstruct_group(const int8_t* residues,
    uint64_t cells,uint64_t cell,const Group& group) {
  uint64_t value=0,product=1;
  for(uint32_t i=0;i<group.count;++i) {
    const uint32_t modulus=group.moduli[i];
    const int centered=residues[uint64_t(group.channels[i])*cells+cell];
    const uint32_t residue=centered<0?uint32_t(centered+int(modulus)):uint32_t(centered);
    const auto current=uint32_t(value%modulus);
    const auto coefficient=((residue+modulus-current)%modulus)*group.inverses[i]%modulus;
    // 0<=value<product and coefficient<modulus imply the result is below
    // the next prefix product. The checked eight-channel leaf fits uint64.
    value+=product*coefficient;
    product*=modulus;
  }
  return value;
}
RNS8_GROUP_CRT_HD inline balanced_crt::Wide merge_groups(const uint64_t* groups,
    const Plan& plan) {
  balanced_crt::Wide values[4]{};
  for(uint32_t g=0;g<plan.group_count;++g)values[g].limb[0]=groups[g];
  for(uint32_t i=0;i<plan.merge_count;++i) {
    const auto& merge=plan.merges[i];
    values[merge.left]=balanced_crt::merge(values[merge.left],values[merge.right],merge);
  }
  return values[0];
}
// Enumerate rotations and their reversals. Stable logical group IDs select
// scratch slots and tree positions independently of physical producer order.
RNS8_GROUP_CRT_HD inline uint32_t producer_group(uint32_t physical,uint32_t order,uint32_t count) {
  const auto rotated=(physical+order%count)%count;
  return order>=count?count-1-rotated:rotated;
}
} // namespace rns8::detail::group_crt
#undef RNS8_GROUP_CRT_HD
