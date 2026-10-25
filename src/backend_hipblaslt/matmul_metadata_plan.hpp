#ifndef RNS8_MATMUL_METADATA_PLAN_HPP
#define RNS8_MATMUL_METADATA_PLAN_HPP
#include <array>
#include <cstddef>
#include <cstdint>
namespace rns8::detail {
// All operation/layout attributes are fixed by ABI1: column-major signed-i8
// inputs, i32 compute/scale/output, N/N transforms, alpha1/beta0, no epilogue.
// Input generations/pointers/moduli are deliberately not metadata attributes.
struct HipblasLtMetadataKey {
  int device_id=-1,m=0,n=0,k=0,scratch_ld=0;
  std::size_t workspace_bytes=0;
  uint32_t attribute_abi=1;
  bool operator==(const HipblasLtMetadataKey& other) const {
    return device_id==other.device_id && m==other.m && n==other.n && k==other.k &&
        scratch_ld==other.scratch_ld && workspace_bytes==other.workspace_bytes && attribute_abi==other.attribute_abi;
  }
};
// A fixed-M/N whole GEMM has at most full-slab and tail padded geometries.
// Publish only after successful descriptor construction. No heap, global raw
// descriptors, cross-context sharing, or value/materialization cache.
struct HipblasLtCallMetadataDirectory {
  std::array<HipblasLtMetadataKey,2> keys{};
  std::array<bool,2> ready{};
  int find(const HipblasLtMetadataKey& key) const {
    for(unsigned i=0;i<2;++i)if(ready[i]&&keys[i]==key)return int(i);return -1;
  }
  int free_slot() const {for(unsigned i=0;i<2;++i)if(!ready[i])return int(i);return -1;}
  bool publish(unsigned slot,const HipblasLtMetadataKey& key) {
    if(slot>=2 || ready[slot] || key.attribute_abi!=1 || key.device_id<0 || key.m<=0 || key.n<=0 || key.k<=0 || key.scratch_ld<key.n)return false;
    keys[slot]=key;ready[slot]=true;return true;
  }
};
enum class HipblasLtMetadataPolicy : uint32_t { per_block_control=0, per_call=1 };
struct HipblasLtMetadataCounters {
  uint64_t descriptor_create_calls=0,descriptor_destroy_calls=0,attribute_set_calls=0;
  uint64_t preference_create_calls=0,preference_destroy_calls=0,preference_attribute_calls=0;
  uint64_t heuristic_calls=0,algorithm_cache_hits=0,descriptor_set_hits=0,descriptor_sets_created=0;
  uint64_t matmul_block_calls=0,maximum_live_sets=0;
};
}
#endif
