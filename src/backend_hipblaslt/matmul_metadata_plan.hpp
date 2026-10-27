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
    for(unsigned i=0;i<2;++i) {
      if(ready[i]&&keys[i]==key)return int(i);
    }
    return -1;
  }
  int free_slot() const {for(unsigned i=0;i<2;++i)if(!ready[i])return int(i);return -1;}
  bool publish(unsigned slot,const HipblasLtMetadataKey& key) {
    if(slot>=2 || ready[slot] || key.attribute_abi!=1 || key.device_id<0 || key.m<=0 || key.n<=0 || key.k<=0 || key.scratch_ld<key.n)return false;
    keys[slot]=key;ready[slot]=true;return true;
  }
};
enum class HipblasLtMetadataPolicy : uint32_t { per_block_control=0, per_call=1 };
enum class HipblasLtPrepackPolicy : uint32_t { none=0,a=1,b=2,both=3 };
inline bool hipblaslt_prepack_a_enabled(HipblasLtPrepackPolicy policy) {
  return (static_cast<uint32_t>(policy)&1)!=0;
}
inline bool hipblaslt_prepack_b_enabled(HipblasLtPrepackPolicy policy) {
  return (static_cast<uint32_t>(policy)&2)!=0;
}
struct HipblasLtPrepackCounters {
  uint64_t a_retained_builds=0,b_retained_builds=0,a_retained_hits=0,b_retained_hits=0;
  uint64_t a_retained_planes_packed=0,b_retained_planes_packed=0;
  uint64_t a_transient_planes_packed=0,b_transient_planes_packed=0;
};
struct HipblasLtMetadataCounters {
  uint64_t descriptor_create_calls=0,descriptor_destroy_calls=0,attribute_set_calls=0;
  uint64_t preference_create_calls=0,preference_destroy_calls=0,preference_attribute_calls=0;
  uint64_t heuristic_calls=0,algorithm_cache_hits=0,descriptor_set_hits=0,descriptor_sets_created=0;
  uint64_t matmul_block_calls=0,maximum_live_sets=0;
};

struct HipblasLtAlgorithmObservation {
  HipblasLtMetadataKey key{};
  int algorithm_index=-1;
  uint64_t calls=0;
};
// An opt-in benchmark observation, not another algorithm/descriptor cache.
// Keep indices scoped to the actual library/build/device and padded geometry.
struct HipblasLtAlgorithmInventory {
  std::array<HipblasLtAlgorithmObservation,16> entries{};
  uint32_t size=0;
  bool available=false,complete=true;
  uint64_t index_queries=0;
  void record(const HipblasLtMetadataKey& key,int index) noexcept {
    if(index<0)complete=false;
    for(uint32_t i=0;i<size;++i) {
      auto& entry=entries[i];
      if(entry.key==key && entry.algorithm_index==index) {
        if(entry.calls==UINT64_MAX)complete=false;
        else ++entry.calls;
        return;
      }
    }
    if(size==entries.size()){complete=false;return;}
    entries[size++]={key,index,1};
  }
};
}
#endif
