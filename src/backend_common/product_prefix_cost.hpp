#pragma once
#include "product_arena_plan.hpp"
#include "operation_support.hpp"
#include "rns8/rns8.h"
#include <array>
#include <functional>
#include <string>
namespace rns8::detail {
// Costs are explicit caller estimates in integer nanoseconds, never evidence.
// UINT64_MAX means unknown, not free. Tables are bound to a graph and target;
// no values in this profile can lower the independent minimum exact prefixes.
struct ProductNodeCosts {
  std::array<uint64_t,21> gemm;
  std::array<uint64_t,441> lift;
  ProductNodeCosts() { gemm.fill(UINT64_MAX); lift.fill(UINT64_MAX); }
};
struct ProductCostProfile {
  uint32_t version=1, max_evaluations=4096;
  uint64_t graph_key=0;
  rns8_backend_kind backend=RNS8_BACKEND_CPU_REFERENCE;
  rns8_semantics semantics=RNS8_EXACT_WIDE_SIGNED;
  std::string target, provenance;
  std::vector<ProductNodeCosts> nodes;
};
inline uint64_t product_graph_key(const std::vector<ProductShape>& inputs,
    const std::vector<ProductNode>& nodes,uint32_t prefix) {
  uint64_t hash=UINT64_C(14695981039346656037);
  const auto add=[&](uint64_t value) { for(unsigned i=0;i<8;++i) { hash^=(value>>(8*i))&255; hash*=UINT64_C(1099511628211); } };
  add(1);add(prefix);add(inputs.size());add(nodes.size());
  for(const auto& input:inputs) {add(uint64_t(input.rows));add(uint64_t(input.cols));}
  for(const auto& node:nodes) {add(node.a);add(node.b);}
  return hash;
}
struct ProductPrefixSelection {
  rns8_status status=RNS8_UNSUPPORTED_BACKEND;
  uint64_t estimated_ns=UINT64_MAX,peak_bytes=0,lift_scratch_bytes=0;
  uint32_t evaluated=0;
  bool exhaustive=false;
  std::vector<uint32_t> prefixes,capacities;
};
inline ProductPrefixSelection select_product_prefixes(const std::vector<ProductShape>& inputs,
    const std::vector<ProductNode>& nodes,uint32_t prefix,const std::vector<uint32_t>& minimum,
    uint64_t budget,bool gpu,const ProductCostProfile& costs) {
  ProductPrefixSelection best;
  if(costs.version!=1 || costs.provenance.empty() || costs.target.empty() ||
      costs.max_evaluations<2 || costs.max_evaluations>65536 || nodes.size()>64 ||
      costs.nodes.size()!=nodes.size() || minimum.size()!=nodes.size() ||
      costs.graph_key!=product_graph_key(inputs,nodes,prefix)) {
    best.status=RNS8_INVALID_ARGUMENT;return best;
  }
  const auto shapes=plan_product_arena(inputs,nodes,prefix,UINT64_MAX);
  if(shapes.status!=RNS8_SUCCESS) {best.status=shapes.status;return best;}
  uint64_t combinations=1;
  for(auto p:minimum) {
    if(!p || p>prefix) {best.status=RNS8_INVALID_ARGUMENT;return best;}
    const uint64_t choices=prefix-p+1;
    combinations=combinations>costs.max_evaluations/choices?uint64_t(costs.max_evaluations)+1:combinations*choices;
  }
  uint32_t evaluated=0;
  const auto evaluate=[&](const std::vector<uint32_t>& chosen) {
    ++evaluated;
    ProductPrefixSelection candidate;
    candidate.prefixes=chosen;candidate.capacities=chosen;candidate.estimated_ns=0;
    std::vector<uint32_t> current(shapes.values.size(),prefix);
    const auto add=[&](uint64_t ns) {
      if(ns==UINT64_MAX || candidate.estimated_ns>=UINT64_MAX-ns) return false;
      candidate.estimated_ns+=ns;return true;
    };
    const auto demand=[&](uint32_t id,uint32_t target) {
      if(id<inputs.size() || current[id]>=target) return true;
      const auto node=id-inputs.size();
      if(!add(costs.nodes[node].lift[current[id]*21+target])) return false;
      candidate.capacities[node]=std::max(candidate.capacities[node],target);
      if(gpu) {
        const auto shape=shapes.values[id].shape;
        const auto support=query_exact_lift_support(shape.rows*shape.cols,current[id],target,UINT64_MAX);
        if(support.status!=RNS8_SUCCESS) return false;
        candidate.lift_scratch_bytes=std::max(candidate.lift_scratch_bytes,support.scratch_bytes);
      }
      current[id]=target;return true;
    };
    for(size_t i=0;i<nodes.size();++i) {
      if(!demand(nodes[i].a,chosen[i]) || !demand(nodes[i].b,chosen[i]) || !add(costs.nodes[i].gemm[chosen[i]])) return;
      // A consumer may require less than its producer computed; retained
      // capacity must still contain every produced plane.
      for(auto id:{nodes[i].a,nodes[i].b}) if(id>=inputs.size())
        candidate.capacities[id-inputs.size()]=std::max(candidate.capacities[id-inputs.size()],chosen[i]);
      current[inputs.size()+i]=chosen[i];
    }
    if(!demand(uint32_t(shapes.values.size()-1),prefix)) return;
    candidate.capacities.back()=prefix;
    // Match the runtime owner's conservative reservation for a value that may
    // grow in several steps. Incremental lift cost remains charged per demand.
    if(gpu) for(size_t i=0;i<nodes.size();++i) if(candidate.capacities[i]>chosen[i]) {
      const auto shape=shapes.values[inputs.size()+i].shape;
      const auto support=query_exact_lift_support(shape.rows*shape.cols,chosen[i],candidate.capacities[i],UINT64_MAX);
      if(support.status!=RNS8_SUCCESS) return;
      candidate.lift_scratch_bytes=std::max(candidate.lift_scratch_bytes,support.scratch_bytes);
    }
    if(candidate.lift_scratch_bytes>budget) return;
    const auto arena=plan_product_arena(inputs,nodes,prefix,budget-candidate.lift_scratch_bytes,candidate.capacities);
    if(arena.status!=RNS8_SUCCESS) return;
    candidate.peak_bytes=arena.arena_bytes+candidate.lift_scratch_bytes;
    candidate.status=RNS8_SUCCESS;
    if(best.status!=RNS8_SUCCESS || candidate.estimated_ns<best.estimated_ns ||
        (candidate.estimated_ns==best.estimated_ns && (candidate.peak_bytes<best.peak_bytes ||
         (candidate.peak_bytes==best.peak_bytes && candidate.prefixes<best.prefixes)))) best=std::move(candidate);
  };
  if(combinations<=costs.max_evaluations) {
    auto chosen=minimum;
    std::function<void(size_t)> enumerate=[&](size_t index) {
      if(index==nodes.size()) {evaluate(chosen);return;}
      for(chosen[index]=minimum[index];chosen[index]<=prefix;++chosen[index]) enumerate(index+1);
    };
    enumerate(0);best.exhaustive=true;
  } else {
    evaluate(minimum);evaluate(std::vector<uint32_t>(nodes.size(),prefix));
    // Bounded deterministic coordinate descent, not a global optimality claim.
    // A failed seed does not prove the feasible set empty; return unsupported.
    for(unsigned pass=0;pass<8 && best.status==RNS8_SUCCESS && evaluated<costs.max_evaluations;++pass) {
      const auto previous=best.prefixes;
      for(size_t i=0;i<nodes.size() && evaluated<costs.max_evaluations;++i)
        for(uint32_t p=minimum[i];p<=prefix && evaluated<costs.max_evaluations;++p) {
          auto chosen=best.prefixes;chosen[i]=p;evaluate(chosen);
        }
      if(best.prefixes==previous) break;
    }
  }
  best.evaluated=evaluated;
  return best;
}
}
