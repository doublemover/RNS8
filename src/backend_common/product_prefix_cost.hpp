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
  // v2 lift estimates price retained suffix/status scratch on HIP (including
  // reset/check/staged copy), or inline host extension on CPU; not one-shot malloc.
  uint32_t version=2, max_evaluations=4096;
  uint64_t graph_key=0;
  rns8_backend_kind backend=RNS8_BACKEND_CPU_REFERENCE;
  rns8_semantics semantics=RNS8_EXACT_WIDE_SIGNED;
  std::string target, provenance;
  std::vector<ProductNodeCosts> nodes;
};
inline uint64_t product_graph_key(const std::vector<ProductShape>& inputs,
    const std::vector<ProductNode>& nodes,uint32_t prefix,
    const std::vector<uint32_t>& outputs = {}) {
  uint64_t hash=UINT64_C(14695981039346656037);
  const auto add=[&](uint64_t value) { for(unsigned i=0;i<8;++i) { hash^=(value>>(8*i))&255; hash*=UINT64_C(1099511628211); } };
  add(2);add(prefix);add(inputs.size());add(nodes.size());
  for(const auto& input:inputs) {add(uint64_t(input.rows));add(uint64_t(input.cols));}
  for(const auto& node:nodes) {add(node.a);add(node.b);}
  add(outputs.empty()?1:outputs.size());
  if(outputs.empty()) add(inputs.size()+nodes.size()-1);
  else for(auto id:outputs) add(id);
  return hash;
}
struct ProductPrefixSelection {
  rns8_status status=RNS8_UNSUPPORTED_BACKEND;
  uint64_t estimated_ns=UINT64_MAX,peak_bytes=0,lift_scratch_bytes=0;
  uint32_t evaluated=0;
  bool exhaustive=false,chain_dynamic_programming=false,cost_optimal=false;
  uint64_t chain_cost_lower_bound=UINT64_MAX;
  std::vector<uint32_t> prefixes,capacities;
};
struct ProductChainCostPath {
  bool applicable=false;
  uint64_t estimated_ns=UINT64_MAX;
  std::vector<uint32_t> prefixes;
};
// Exact cost-only DP for a strict left/right chain. Resource admission remains
// the real arena/lift query; a rejected unconstrained optimum proves no memory
// infeasibility claim and cannot authorize an over-budget execution.
inline ProductChainCostPath product_chain_cost_path(size_t input_count,
    const std::vector<ProductNode>& nodes,uint32_t prefix,const std::vector<uint32_t>& minimum,
    const ProductCostProfile& costs) {
  ProductChainCostPath result;
  if(nodes.empty()) return result;
  for(size_t i=1;i<nodes.size();++i) {
    const uint32_t previous=uint32_t(input_count+i-1);
    if(nodes[i].a!=previous && nodes[i].b!=previous) return result;
    for(auto operand:{nodes[i].a,nodes[i].b})
      if(operand>=input_count && operand!=previous) return result;
  }
  result.applicable=true;
  struct State {uint64_t ns=UINT64_MAX;std::vector<uint32_t> path;};
  std::array<State,21> current;
  const auto append_cost=[](uint64_t a,uint64_t b) {
    return a==UINT64_MAX || b==UINT64_MAX || a>=UINT64_MAX-b?UINT64_MAX:a+b;
  };
  for(uint32_t p=minimum[0];p<=prefix;++p) {
    current[p].ns=costs.nodes[0].gemm[p];
    if(current[p].ns!=UINT64_MAX) current[p].path={p};
  }
  for(size_t i=1;i<nodes.size();++i) {
    std::array<State,21> next;
    for(uint32_t q=minimum[i];q<=prefix;++q) for(uint32_t p=minimum[i-1];p<=prefix;++p) {
      const auto lifted=append_cost(current[p].ns,p<q?costs.nodes[i-1].lift[p*21+q]:0);
      const auto ns=append_cost(lifted,costs.nodes[i].gemm[q]);if(ns==UINT64_MAX) continue;
      auto path=current[p].path;path.push_back(q);
      if(ns<next[q].ns || (ns==next[q].ns && path<next[q].path)) next[q]={ns,std::move(path)};
    }
    current=std::move(next);
  }
  for(uint32_t p=minimum.back();p<=prefix;++p) {
    const auto ns=append_cost(current[p].ns,p<prefix?costs.nodes.back().lift[p*21+prefix]:0);
    if(ns<result.estimated_ns || (ns!=UINT64_MAX && ns==result.estimated_ns && current[p].path<result.prefixes)) {
      result.estimated_ns=ns;result.prefixes=current[p].path;
    }
  }
  return result;
}
inline ProductPrefixSelection select_product_prefixes(const std::vector<ProductShape>& inputs,
    const std::vector<ProductNode>& nodes,uint32_t prefix,const std::vector<uint32_t>& minimum,
    uint64_t budget,bool gpu,const ProductCostProfile& costs,
    const std::vector<uint32_t>& outputs = {}) {
  ProductPrefixSelection best;
  if(costs.version!=2 || costs.provenance.empty() || costs.target.empty() ||
      costs.max_evaluations<2 || costs.max_evaluations>65536 || nodes.size()>64 ||
      costs.nodes.size()!=nodes.size() || minimum.size()!=nodes.size() ||
      costs.graph_key!=product_graph_key(inputs,nodes,prefix,outputs)) {
    best.status=RNS8_INVALID_ARGUMENT;return best;
  }
  const auto shapes=plan_product_arena(inputs,nodes,prefix,UINT64_MAX,{},outputs);
  if(shapes.status!=RNS8_SUCCESS) {best.status=shapes.status;return best;}
  uint64_t combinations=1;
  for(auto p:minimum) {
    if(!p || p>prefix) {best.status=RNS8_INVALID_ARGUMENT;return best;}
    const uint64_t choices=prefix-p+1;
    combinations=combinations>costs.max_evaluations/choices?uint64_t(costs.max_evaluations)+1:combinations*choices;
  }
  const std::vector<uint32_t> observed=outputs.empty()
      ? std::vector<uint32_t>{static_cast<uint32_t>(shapes.values.size()-1)} : outputs;
  // The chain DP includes one final export. Multiple retained outputs use the
  // full demand evaluator so final lifting and liveness cannot be omitted.
  const auto chain=observed.size()==1 && observed[0]==shapes.values.size()-1
      ? product_chain_cost_path(inputs.size(),nodes,prefix,minimum,costs) : ProductChainCostPath{};
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
    for(auto id:observed) {
      if(!demand(id,prefix)) return;
      candidate.capacities[id-inputs.size()]=prefix;
    }
    // Match the runtime owner's conservative reservation for a value that may
    // grow in several steps. Incremental lift cost remains charged per demand.
    if(gpu) for(size_t i=0;i<nodes.size();++i) if(candidate.capacities[i]>chosen[i]) {
      const auto shape=shapes.values[inputs.size()+i].shape;
      const auto support=query_exact_lift_support(shape.rows*shape.cols,chosen[i],candidate.capacities[i],UINT64_MAX);
      if(support.status!=RNS8_SUCCESS) return;
      candidate.lift_scratch_bytes=std::max(candidate.lift_scratch_bytes,support.scratch_bytes);
    }
    if(candidate.lift_scratch_bytes>budget) return;
    const auto arena=plan_product_arena(inputs,nodes,prefix,budget-candidate.lift_scratch_bytes,candidate.capacities,observed);
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
    if(chain.estimated_ns!=UINT64_MAX) evaluate(chain.prefixes);
    if(evaluated<costs.max_evaluations) evaluate(minimum);
    if(evaluated<costs.max_evaluations) evaluate(std::vector<uint32_t>(nodes.size(),prefix));
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
  best.chain_dynamic_programming=chain.applicable;
  best.chain_cost_lower_bound=chain.estimated_ns;
  best.cost_optimal=best.status==RNS8_SUCCESS && (best.exhaustive ||
      (chain.estimated_ns!=UINT64_MAX && best.estimated_ns==chain.estimated_ns));
  return best;
}
}
