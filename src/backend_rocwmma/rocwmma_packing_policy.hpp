#pragma once
#include <cstdint>
namespace rns8::detail {
// Research controls: keep the existing packed path as the default until paired
// measurements qualify a change. Policy is passed explicitly across the HIP ABI.
enum class RocwmmaPackingPolicy:uint32_t {packed=0,direct_a=1,reuse=2,direct_a_reuse=3};
struct RocwmmaPackingStats {
  uint64_t a_pack_launches=0,b_pack_launches=0,direct_a_blocks=0,a_panel_hits=0,b_panel_hits=0;
};
inline bool rocwmma_direct_a_eligible(uintptr_t address,int rows,int k,int lda) {
  // Whole M/K tiles only: a masked output cannot make an out-of-bounds fragment
  // load safe. Each row and slab start must preserve the vector alignment.
  return address && !(address%16) && rows>0 && !(rows%16) && k>0 && !(k%16) && lda>=k && !(lda%16);
}
struct RocwmmaPanelKey {
  uintptr_t source=0;int extent=0,valid_k=0,padded_k=0,ld=0,k_offset=0;
  bool operator==(const RocwmmaPanelKey& x) const {
    return source==x.source && extent==x.extent && valid_k==x.valid_k && padded_k==x.padded_k && ld==x.ld && k_offset==x.k_offset;
  }
};
#if !defined(__HIPCC__)
RocwmmaPackingPolicy rocwmma_packing_policy();
bool rocwmma_set_packing_policy(RocwmmaPackingPolicy);
void rocwmma_reset_packing_stats();
RocwmmaPackingStats rocwmma_packing_stats();
#endif
}
