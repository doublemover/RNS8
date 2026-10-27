#pragma once
#include "backend_common/product_arena_plan.hpp"
#include "backend_common/product_prefix_cost.hpp"
#include "backend_common/product_materialization_plan.hpp"
#include "rns8/rns8.h"
#include <memory>
#include <vector>
namespace rns8::detail {
// Internal synchronous executor: ordinary exact-wide semantics, one common
// input/output prefix and modulus-major layout, fixed topological order, staged
// final publication. Optional minimum-prefix products lift on consumer demand.
// Optional internal cost profiles select prefixes without changing admissibility.
class RuntimeProductRecipe {
 public:
  ~RuntimeProductRecipe();
  RuntimeProductRecipe(const RuntimeProductRecipe&)=delete;
  RuntimeProductRecipe& operator=(const RuntimeProductRecipe&)=delete;
  static rns8_status create(rns8_context*,const std::vector<const rns8_matrix*>&,
      const std::vector<ProductNode>&,uint64_t max_arena_bytes,
      std::unique_ptr<RuntimeProductRecipe>& out,bool minimize_compute_prefix = false,
      const ProductCostProfile* costs = nullptr,
      const std::vector<uint32_t>& outputs = {},
      const ProductMaterializationProfile* materializations = nullptr);
  rns8_status execute(rns8_matrix* output);
  rns8_status execute(const std::vector<rns8_matrix*>& outputs);
  rns8_status execute(rns8_matrix* const* outputs,size_t count);
  rns8_status close();
  rns8_status rebind(const std::vector<const rns8_matrix*>& inputs);
  const ProductArenaPlan& allocation() const;
  uint64_t lift_scratch_bytes() const;
  const ProductPrefixSelection& cost_selection() const;
  const ProductMaterializationSelection& materialization_selection() const;
  uint64_t completed_packing_calls() const;
  uint64_t completed_replays() const;
  uint64_t reused_packing_calls() const;
  uint64_t expected_replays() const;
  const ReplayCostPolicy& replay_policy() const;
  uint32_t materialization_b_value(uint32_t node) const;
  const std::vector<uint32_t>& compute_prefixes() const;
  const BasisPacket& basis_packet() const;
  uint32_t output_prefix() const;
 private:
  struct Impl;
  explicit RuntimeProductRecipe(std::unique_ptr<Impl>);
  std::unique_ptr<Impl> impl_;
};
} // namespace rns8::detail
