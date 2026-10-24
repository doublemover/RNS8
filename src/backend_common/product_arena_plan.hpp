#pragma once
#include "address_domain.hpp"
#include "rns8/status.h"
#include <algorithm>
#include <vector>

namespace rns8::detail {
struct ProductShape { int64_t rows=0,cols=0; };
// Value IDs address external inputs first, then preceding node outputs.
struct ProductNode { uint32_t a=0,b=0; };
struct ProductArenaValue {
  ProductShape shape;
  uint64_t offset=0,bytes=0;
  uint32_t last_use=0;
};
struct ProductArenaPlan {
  rns8_status status=RNS8_INVALID_ARGUMENT;
  uint64_t arena_bytes=0,unreused_bytes=0;
  std::vector<ProductArenaValue> values;
};
inline ProductArenaPlan plan_product_arena(const std::vector<ProductShape>& inputs,
    const std::vector<ProductNode>& nodes,uint32_t prefix,uint64_t budget,
    const std::vector<uint32_t>& output_capacities = {},
    const std::vector<uint32_t>& retained_outputs = {}) {
  ProductArenaPlan plan;
  if(inputs.empty() || inputs.size()>4096 || nodes.empty() || nodes.size()>4096 ||
      !prefix || prefix>20 || (!output_capacities.empty() && output_capacities.size()!=nodes.size())) return plan;
  for(const auto& shape:inputs) {
    AddressDomain address;
    if(shape.rows<=0 || shape.cols<=0 || !checked_host_plane_address(
        shape.rows,shape.cols,shape.cols,prefix,1,address)) return plan;
    plan.values.push_back({shape,0,address.byte_extent,0});
  }
  const auto count=static_cast<uint32_t>(inputs.size());
  for(uint32_t i=0;i<nodes.size();++i) {
    const auto node=nodes[i];
    if(node.a>=count+i || node.b>=count+i) return plan;
    const auto a=plan.values[node.a].shape,b=plan.values[node.b].shape;
    if(a.cols!=b.rows) return plan;
    const auto capacity=output_capacities.empty()?prefix:output_capacities[i];
    if(!capacity || capacity>prefix) return plan;
    AddressDomain address;
    if(!checked_host_plane_address(a.rows,b.cols,b.cols,capacity,1,address)) {
      plan.status=RNS8_RANGE_ERROR; return plan;
    }
    plan.values[node.a].last_use=std::max(plan.values[node.a].last_use,i);
    plan.values[node.b].last_use=std::max(plan.values[node.b].last_use,i);
    plan.values.push_back({{a.rows,b.cols},0,address.byte_extent,i});
  }
  if(retained_outputs.empty()) plan.values.back().last_use=static_cast<uint32_t>(nodes.size());
  else {
    if(retained_outputs.size()>nodes.size()) return plan;
    std::vector<bool> seen(nodes.size(),false);
    for(auto id:retained_outputs) {
      if(id<count || id>=plan.values.size() || seen[id-count]) return plan;
      seen[id-count]=true;plan.values[id].last_use=static_cast<uint32_t>(nodes.size());
    }
  }
  struct Slot { uint64_t offset,capacity; uint32_t last_use; };
  std::vector<Slot> slots;
  for(uint32_t i=0;i<nodes.size();++i) {
    auto& value=plan.values[count+i];
    if(value.bytes>SIZE_MAX-255 || plan.unreused_bytes>SIZE_MAX-value.bytes) {
      plan.status=RNS8_RANGE_ERROR; return plan;
    }
    plan.unreused_bytes+=value.bytes;
    const uint64_t aligned=(value.bytes+255)&~uint64_t{255};
    size_t selected=slots.size();
    for(size_t j=0;j<slots.size();++j)
      if(slots[j].last_use<i && slots[j].capacity>=aligned &&
          (selected==slots.size() || slots[j].capacity<slots[selected].capacity)) selected=j;
    if(selected==slots.size()) {
      const auto object_limit=static_cast<uint64_t>(PTRDIFF_MAX);
      if(aligned>object_limit || plan.arena_bytes>object_limit-aligned) {
        plan.status=RNS8_RANGE_ERROR; return plan;
      }
      if(aligned>SIZE_MAX-plan.arena_bytes || aligned>budget || plan.arena_bytes>budget-aligned) {
        plan.status=RNS8_WORKSPACE_TOO_SMALL; return plan;
      }
      value.offset=plan.arena_bytes;
      slots.push_back({value.offset,aligned,value.last_use});
      plan.arena_bytes+=aligned;
    } else {
      value.offset=slots[selected].offset;
      slots[selected].last_use=value.last_use;
    }
  }
  plan.status=RNS8_SUCCESS;
  return plan;
}
} // namespace rns8::detail
