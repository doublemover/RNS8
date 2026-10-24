#pragma once
#include "backend_common/product_arena_plan.hpp"
#include "backend_common/product_prefix_cost.hpp"
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
      const ProductCostProfile* costs = nullptr);
  rns8_status execute(rns8_matrix* output);
  rns8_status close();
  rns8_status rebind(const std::vector<const rns8_matrix*>& inputs);
  const ProductArenaPlan& allocation() const;
  uint64_t lift_scratch_bytes() const;
  const ProductPrefixSelection& cost_selection() const;
  const std::vector<uint32_t>& compute_prefixes() const;
 private:
  struct Impl;
  explicit RuntimeProductRecipe(std::unique_ptr<Impl>);
  std::unique_ptr<Impl> impl_;
};
} // namespace rns8::detail
