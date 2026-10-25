#pragma once
#include "product_prefix_cost.hpp"
#include "residue_layout_gemm.hpp"
#include <map>
#include <tuple>

namespace rns8::detail {
struct ProductPackedCosts {
  static constexpr size_t gemm_index(uint32_t capacity,uint32_t compute){return capacity*21+compute;}
  static constexpr size_t pack_index(uint32_t capacity,uint32_t old_prefix,uint32_t new_prefix){return (capacity*21+old_prefix)*21+new_prefix;}
  std::array<uint64_t,441> gemm;
  // Actual conversion of [old_prefix,new_prefix), including launch overhead.
  std::array<uint64_t,9261> pack;
  ProductPackedCosts(){gemm.fill(UINT64_MAX);pack.fill(UINT64_MAX);}
};
struct ProductMaterializationProfile {
  ProductCostProfile uniform;
  std::vector<ProductPackedCosts> packed;
  uint32_t cost_schema_version=2;
  uint32_t max_frontier=256;
};
struct ProductPackedValue {
  ResidueLayout layout;
  uint64_t offset=0;
  uint32_t first=UINT32_MAX,last=0;
  bool used=false;
};
struct ProductMaterializationSelection {
  rns8_status status=RNS8_UNSUPPORTED_BACKEND;
  ProductArenaPlan arena;
  std::vector<ProductPackedValue> packed;
  std::vector<uint32_t> prefixes,capacities;
  std::vector<uint8_t> packed_b;
  uint64_t estimated_ns=UINT64_MAX,lift_scratch_bytes=0,packed_payload_bytes=0;
  uint32_t evaluated=0;
  bool exhaustive=false,chain_dynamic_programming=false,cost_optimal=false;
  uint64_t chain_cost_lower_bound=UINT64_MAX;
};
inline uint64_t product_cost_add(uint64_t a,uint64_t b) {
  return a==UINT64_MAX || b==UINT64_MAX || a>=UINT64_MAX-b?UINT64_MAX:a+b;
}

// Evaluate an actual sequential schedule. Packing is shared by logical B value,
// grows only when another consumer needs more planes, and is priced exactly at
// the stage performing that growth. Uniform and packed storage share one arena.
inline ProductMaterializationSelection evaluate_product_materializations(
    const std::vector<ProductShape>& inputs,const std::vector<ProductNode>& nodes,
    uint32_t prefix,const std::vector<uint32_t>& chosen,const std::vector<uint8_t>& packing,
    uint64_t budget,bool gpu,const ProductMaterializationProfile& profile,
    const std::vector<uint32_t>& outputs={}) {
  ProductMaterializationSelection result;
  result.prefixes=chosen;result.capacities=chosen;result.packed_b=packing;result.estimated_ns=0;
  result.arena=plan_product_arena(inputs,nodes,prefix,UINT64_MAX,{},outputs);
  if(result.arena.status!=RNS8_SUCCESS){result.status=result.arena.status;return result;}
  const auto input_count=static_cast<uint32_t>(inputs.size());
  const size_t count=result.arena.values.size();
  if(profile.cost_schema_version!=2 || chosen.size()!=nodes.size() || packing.size()!=nodes.size() ||
      profile.uniform.nodes.size()!=nodes.size() || profile.packed.size()!=nodes.size()) {
    result.status=RNS8_INVALID_ARGUMENT;return result;
  }
  result.packed.resize(count);
  std::vector<uint32_t> current(count),packed_prefix(count),packed_capacity(count),raw_last(count);
  // Tile offsets use the retained allocation's final capacity, including at
  // earlier consumers that compute fewer planes. Price that physical layout.
  for(size_t i=0;i<nodes.size();++i) {
    if(!chosen[i] || chosen[i]>prefix || packing[i]>1){result.status=RNS8_INVALID_ARGUMENT;return result;}
    if(packing[i])packed_capacity[nodes[i].b]=std::max(packed_capacity[nodes[i].b],chosen[i]);
  }
  std::fill_n(current.begin(),input_count,prefix);
  for(uint32_t i=0;i<nodes.size();++i)raw_last[input_count+i]=i;
  const auto add=[&](uint64_t ns) {
    result.estimated_ns=product_cost_add(result.estimated_ns,ns);
    return result.estimated_ns!=UINT64_MAX;
  };
  const auto demand=[&](uint32_t id,uint32_t target,uint32_t reader) {
    if(id<input_count)return true;
    raw_last[id]=std::max(raw_last[id],reader);
    const auto producer=id-input_count;
    result.capacities[producer]=std::max(result.capacities[producer],target);
    if(current[id]>=target)return true;
    if(!current[id] || !add(profile.uniform.nodes[producer].lift[current[id]*21+target]))return false;
    current[id]=target;return true;
  };
  for(uint32_t i=0;i<nodes.size();++i) {
    const auto node=nodes[i];const auto p=chosen[i];
    if(!p || p>prefix || packing[i]>1){result.status=RNS8_INVALID_ARGUMENT;return result;}
    if(!demand(node.a,p,i))return result;
    if(packing[i]) {
      auto& packed=result.packed[node.b];packed.used=true;
      packed.first=std::min(packed.first,i);packed.last=i;
      if(packed_prefix[node.b]<p) {
        if(!demand(node.b,p,i) || !add(profile.packed[i].pack[ProductPackedCosts::pack_index(packed_capacity[node.b],packed_prefix[node.b],p)]))return result;
        packed_prefix[node.b]=p;
      }
      if(!add(profile.packed[i].gemm[ProductPackedCosts::gemm_index(packed_capacity[node.b],p)]))return result;
    } else if(!demand(node.b,p,i) || !add(profile.uniform.nodes[i].gemm[p]))return result;
    current[input_count+i]=p;
  }
  const auto observed=outputs.empty()?std::vector<uint32_t>{static_cast<uint32_t>(count-1)}:outputs;
  for(auto id:observed)if(!demand(id,prefix,static_cast<uint32_t>(nodes.size())))return result;
  if(gpu)for(uint32_t i=0;i<nodes.size();++i)if(result.capacities[i]>chosen[i]) {
    const auto shape=result.arena.values[input_count+i].shape;
    const auto support=profile.uniform.experimental_basis
        ?basis_lift_storage(shape.rows*shape.cols,chosen[i],result.capacities[i])
        :[&] {const auto q=query_exact_lift_support(shape.rows*shape.cols,chosen[i],result.capacities[i],UINT64_MAX);
            BasisLiftStorage s;s.status=q.status;s.scratch_bytes=q.scratch_bytes;return s;}();
    if(support.status!=RNS8_SUCCESS)return result;
    result.lift_scratch_bytes=std::max(result.lift_scratch_bytes,support.scratch_bytes);
  }
  if(result.lift_scratch_bytes>budget){result.status=RNS8_WORKSPACE_TOO_SMALL;return result;}
  struct Region {uint32_t birth,last,id;uint64_t bytes;bool packed;};
  std::vector<Region> regions;
  for(uint32_t i=0;i<nodes.size();++i) {
    auto& value=result.arena.values[input_count+i];AddressDomain domain;
    if(!checked_host_plane_address(value.shape.rows,value.shape.cols,value.shape.cols,result.capacities[i],1,domain)) {
      result.status=RNS8_RANGE_ERROR;return result;
    }
    value.bytes=domain.byte_extent;value.last_use=raw_last[input_count+i];
    regions.push_back({i,value.last_use,input_count+i,value.bytes,false});
  }
  for(uint32_t id=0;id<count;++id)if(result.packed[id].used) {
    auto& value=result.packed[id];const auto shape=result.arena.values[id].shape;
    if(!make_residue_layout(ResidueLayoutKind::tile_major,shape.rows,shape.cols,shape.cols,
          packed_prefix[id],16,16,value.layout,profile.uniform.basis) ||
        (gpu && value.layout.bytes>INT32_MAX))return result;
    if(result.packed_payload_bytes>UINT64_MAX-value.layout.bytes){result.status=RNS8_RANGE_ERROR;return result;}
    result.packed_payload_bytes+=value.layout.bytes;
    regions.push_back({value.first,value.last,id,value.layout.bytes,true});
  }
  std::sort(regions.begin(),regions.end(),[](const Region& a,const Region& b) {
    return std::tie(a.birth,a.packed,a.id)<std::tie(b.birth,b.packed,b.id);
  });
  struct Slot {uint64_t offset,capacity;uint32_t last;};
  std::vector<Slot> slots;result.arena.arena_bytes=0;result.arena.unreused_bytes=0;
  const uint64_t available=budget-result.lift_scratch_bytes;
  for(const auto& region:regions) {
    if(region.bytes>uint64_t(PTRDIFF_MAX)-255 || result.arena.unreused_bytes>UINT64_MAX-region.bytes) {
      result.status=RNS8_RANGE_ERROR;return result;
    }
    result.arena.unreused_bytes+=region.bytes;const uint64_t aligned=(region.bytes+255)&~uint64_t{255};
    size_t selected=slots.size();
    for(size_t j=0;j<slots.size();++j)
      if(slots[j].last<region.birth && slots[j].capacity>=aligned &&
          (selected==slots.size() || slots[j].capacity<slots[selected].capacity))selected=j;
    if(selected==slots.size()) {
      if(aligned>uint64_t(PTRDIFF_MAX)-result.arena.arena_bytes){result.status=RNS8_RANGE_ERROR;return result;}
      if(aligned>available || result.arena.arena_bytes>available-aligned) {
        result.status=RNS8_WORKSPACE_TOO_SMALL;return result;
      }
      slots.push_back({result.arena.arena_bytes,aligned,region.last});result.arena.arena_bytes+=aligned;
    } else slots[selected].last=region.last;
    if(region.packed)result.packed[region.id].offset=slots[selected].offset;
    else result.arena.values[region.id].offset=slots[selected].offset;
  }
  result.arena.status=result.status=RNS8_SUCCESS;return result;
}

struct ProductMaterializationChainPaths {
  struct Path {uint64_t cost=0;std::vector<uint32_t> prefixes,packed_prefix,packed_capacity;std::vector<uint8_t> packing;};
  bool applicable=false,truncated=false,unknown_cost=false;
  uint64_t lower_bound=UINT64_MAX;
  std::vector<Path> paths;
};
// Cost DP for strict chains, keyed by the previous raw prefix and every shared
// external packed prefix and retained physical capacity. Memory is admitted by the real arena evaluator after
// this cost pass; dropping a cost state never proves memory infeasibility.
inline ProductMaterializationChainPaths product_materialization_chain_paths(size_t input_count,
    const std::vector<ProductNode>& nodes,uint32_t prefix,const std::vector<uint32_t>& minimum,
    const ProductMaterializationProfile& profile) {
  ProductMaterializationChainPaths result;
  if(!input_count || input_count>4096 || nodes.empty() || nodes.size()>64 || !prefix || prefix>20 ||
      profile.cost_schema_version!=2 || !profile.max_frontier || profile.max_frontier>4096 || minimum.size()!=nodes.size() ||
      profile.uniform.nodes.size()!=nodes.size() || profile.packed.size()!=nodes.size())return result;
  for(size_t i=0;i<nodes.size();++i)
    if(!minimum[i] || minimum[i]>prefix || nodes[i].a>=input_count+i || nodes[i].b>=input_count+i)return result;
  for(size_t i=1;i<nodes.size();++i) {
    const uint32_t previous=static_cast<uint32_t>(input_count+i-1);
    if(nodes[i].a!=previous && nodes[i].b!=previous)return result;
    for(auto id:{nodes[i].a,nodes[i].b})if(id>=input_count && id!=previous)return result;
  }
  result.applicable=true;
  std::vector<uint32_t> external_slot(input_count,UINT32_MAX);uint32_t external_count=0;
  for(const auto node:nodes)if(node.b<input_count && external_slot[node.b]==UINT32_MAX)
    external_slot[node.b]=external_count++;
  using Path=ProductMaterializationChainPaths::Path;
  std::vector<size_t> last_use(external_count);
  for(size_t i=0;i<nodes.size();++i)if(nodes[i].b<input_count)last_use[external_slot[nodes[i].b]]=i;
  std::vector<Path> current(1);current[0].packed_prefix.resize(external_count);current[0].packed_capacity.resize(external_count);
  for(size_t i=0;i<nodes.size();++i) {
    std::map<std::vector<uint32_t>,Path> next;
    for(const auto& state:current)for(uint32_t p=minimum[i];p<=prefix;++p)for(uint8_t packing=0;packing<2;++packing) {
      const auto id=nodes[i].b;
      const bool external=id<input_count;
      const auto slot=external?external_slot[id]:0;
      const uint32_t first_capacity=packing && external && !state.packed_capacity[slot]?p:
          packing && external?state.packed_capacity[slot]:p;
      const uint32_t last_capacity=packing && external && !state.packed_capacity[slot]?prefix:first_capacity;
      for(uint32_t capacity=first_capacity;capacity<=last_capacity;++capacity) {
      if(packing && capacity<p)continue;
      auto candidate=state;
      if(i && state.prefixes.back()<p)
        candidate.cost=product_cost_add(candidate.cost,profile.uniform.nodes[i-1].lift[state.prefixes.back()*21+p]);
      if(packing) {
        const uint32_t old=external?state.packed_prefix[slot]:0;
        if(old<p)candidate.cost=product_cost_add(candidate.cost,profile.packed[i].pack[ProductPackedCosts::pack_index(capacity,old,p)]);
        if(external){candidate.packed_prefix[slot]=std::max(old,p);candidate.packed_capacity[slot]=capacity;}
      }
      // The evaluator retains exactly the largest packed demand. A speculative
      // larger capacity is legal only if a later consumer actually demands it.
      bool realizable=true;
      for(uint32_t s=0;s<external_count;++s)if(last_use[s]<=i && candidate.packed_prefix[s]!=candidate.packed_capacity[s]){realizable=false;break;}
      if(!realizable)continue;
      candidate.cost=product_cost_add(candidate.cost,packing?profile.packed[i].gemm[ProductPackedCosts::gemm_index(capacity,p)]:profile.uniform.nodes[i].gemm[p]);
      if(candidate.cost==UINT64_MAX){result.unknown_cost=true;continue;}
      candidate.prefixes.push_back(p);candidate.packing.push_back(packing);
      // Dead external caches cannot affect subsequent cost. Coalesce them so
      // bounded frontiers retain useful alternatives instead of dead history.
      for(uint32_t s=0;s<external_count;++s)if(last_use[s]<=i)
        candidate.packed_prefix[s]=candidate.packed_capacity[s]=0;
      auto key=candidate.packed_prefix;key.insert(key.end(),candidate.packed_capacity.begin(),candidate.packed_capacity.end());key.push_back(p);
      auto found=next.find(key);
      if(found==next.end() || std::tie(candidate.cost,candidate.prefixes,candidate.packing)<
          std::tie(found->second.cost,found->second.prefixes,found->second.packing))next[key]=std::move(candidate);
      }
    }
    current.clear();for(auto& entry:next)current.push_back(std::move(entry.second));
    std::sort(current.begin(),current.end(),[](const Path& a,const Path& b) {
      return std::tie(a.cost,a.prefixes,a.packing)<std::tie(b.cost,b.prefixes,b.packing);
    });
    if(current.size()>profile.max_frontier){current.resize(profile.max_frontier);result.truncated=true;}
  }
  for(auto& path:current) {
    if(path.prefixes.back()<prefix)path.cost=product_cost_add(path.cost,
        profile.uniform.nodes.back().lift[path.prefixes.back()*21+prefix]);
    if(path.cost!=UINT64_MAX)result.paths.push_back(std::move(path));
    else result.unknown_cost=true;
  }
  std::sort(result.paths.begin(),result.paths.end(),[](const Path& a,const Path& b) {
    return std::tie(a.cost,a.prefixes,a.packing)<std::tie(b.cost,b.prefixes,b.packing);
  });
  if(!result.truncated && !result.unknown_cost && !result.paths.empty())result.lower_bound=result.paths.front().cost;
  return result;
}

inline ProductMaterializationSelection select_product_materializations(
    const std::vector<ProductShape>& inputs,const std::vector<ProductNode>& nodes,uint32_t prefix,
    const std::vector<uint32_t>& minimum,uint64_t budget,bool gpu,
    const ProductMaterializationProfile& profile,const std::vector<uint32_t>& outputs={}) {
  ProductMaterializationSelection best;const auto& costs=profile.uniform;
  if(profile.cost_schema_version!=2 || costs.version!=2 || !valid_ordered_basis(costs.basis) || !prefix || prefix>costs.basis.count ||
      inputs.empty() || nodes.empty() || nodes.size()>64 || minimum.size()!=nodes.size() ||
      costs.nodes.size()!=nodes.size() || profile.packed.size()!=nodes.size() ||
      costs.target.empty() || costs.provenance.empty() || costs.max_evaluations<4 || costs.max_evaluations>65536 ||
      !profile.max_frontier || profile.max_frontier>4096 ||
      costs.graph_key!=product_graph_key(inputs,nodes,prefix,outputs)) {
    best.status=RNS8_INVALID_ARGUMENT;return best;
  }
  const auto shape=plan_product_arena(inputs,nodes,prefix,UINT64_MAX,{},outputs);
  if(shape.status!=RNS8_SUCCESS){best.status=shape.status;return best;}
  uint64_t combinations=1;
  for(auto p:minimum) {
    if(!p || p>prefix){best.status=RNS8_INVALID_ARGUMENT;return best;}
    const uint64_t choices=2*(prefix-p+1);
    combinations=combinations>costs.max_evaluations/choices?uint64_t(costs.max_evaluations)+1:combinations*choices;
  }
  uint32_t evaluated=0;bool unknown=false,budget_rejected=false;
  const auto evaluate=[&](const std::vector<uint32_t>& chosen,const std::vector<uint8_t>& packed) {
    ++evaluated;
    auto candidate=evaluate_product_materializations(inputs,nodes,prefix,chosen,packed,budget,gpu,profile,outputs);
    if(candidate.status!=RNS8_SUCCESS) {
      unknown=unknown || candidate.status!=RNS8_WORKSPACE_TOO_SMALL;
      budget_rejected=budget_rejected || candidate.status==RNS8_WORKSPACE_TOO_SMALL;return;
    }
    const auto peak=candidate.arena.arena_bytes+candidate.lift_scratch_bytes;
    const auto best_peak=best.arena.arena_bytes+best.lift_scratch_bytes;
    if(best.status!=RNS8_SUCCESS || std::tie(candidate.estimated_ns,peak,candidate.prefixes,candidate.packed_b)<
        std::tie(best.estimated_ns,best_peak,best.prefixes,best.packed_b))best=std::move(candidate);
  };
  ProductMaterializationChainPaths chain;
  if(combinations<=costs.max_evaluations) {
    auto chosen=minimum;std::vector<uint8_t> packed(nodes.size());
    std::function<void(size_t)> visit=[&](size_t i) {
      if(i==nodes.size()){evaluate(chosen,packed);return;}
      for(chosen[i]=minimum[i];chosen[i]<=prefix;++chosen[i])for(packed[i]=0;packed[i]<2;++packed[i])visit(i+1);
    };visit(0);best.exhaustive=true;
  } else {
    if(outputs.empty() || (outputs.size()==1 && outputs[0]==inputs.size()+nodes.size()-1))
      chain=product_materialization_chain_paths(inputs.size(),nodes,prefix,minimum,profile);
    // Keep uniform/full and minimum seeds even if an unconstrained cost optimum
    // is infeasible under the retained-memory budget.
    for(const auto& p:{minimum,std::vector<uint32_t>(nodes.size(),prefix)})
      for(uint8_t pack=0;pack<2;++pack)if(evaluated<costs.max_evaluations)evaluate(p,std::vector<uint8_t>(nodes.size(),pack));
    for(const auto& path:chain.paths)if(evaluated<costs.max_evaluations)evaluate(path.prefixes,path.packing);
    for(unsigned pass=0;pass<8 && best.status==RNS8_SUCCESS && evaluated<costs.max_evaluations;++pass) {
      const auto previous=std::make_pair(best.prefixes,best.packed_b);
      for(size_t i=0;i<nodes.size() && evaluated<costs.max_evaluations;++i)
        for(uint32_t p=minimum[i];p<=prefix && evaluated<costs.max_evaluations;++p)
          for(uint8_t pack=0;pack<2 && evaluated<costs.max_evaluations;++pack) {
            auto chosen=best.prefixes;auto packing=best.packed_b;chosen[i]=p;packing[i]=pack;evaluate(chosen,packing);
          }
      if(previous==std::make_pair(best.prefixes,best.packed_b))break;
    }
  }
  if(best.status!=RNS8_SUCCESS && best.exhaustive && !unknown && budget_rejected)best.status=RNS8_WORKSPACE_TOO_SMALL;
  best.evaluated=evaluated;best.chain_dynamic_programming=chain.applicable;
  best.chain_cost_lower_bound=chain.lower_bound;
  best.cost_optimal=best.status==RNS8_SUCCESS && ((best.exhaustive && !unknown) ||
      (chain.lower_bound!=UINT64_MAX && best.estimated_ns==chain.lower_bound));
  return best;
}
} // namespace rns8::detail
