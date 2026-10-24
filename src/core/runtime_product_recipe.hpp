#pragma once
#include "backend_common/product_arena_plan.hpp"
#include "rns8/rns8.h"
#include <memory>
#include <vector>
namespace rns8::detail {
// Internal synchronous executor: ordinary exact-wide semantics, one common
// prefix/modulus-major layout, fixed topological order, staged final publication.
// This is an executable arena adapter, not a public precision/layout planner.
class RuntimeProductRecipe {
 public:
  ~RuntimeProductRecipe();
  RuntimeProductRecipe(const RuntimeProductRecipe&)=delete;
  RuntimeProductRecipe& operator=(const RuntimeProductRecipe&)=delete;
  static rns8_status create(rns8_context*,const std::vector<const rns8_matrix*>&,
      const std::vector<ProductNode>&,uint64_t max_arena_bytes,
      std::unique_ptr<RuntimeProductRecipe>& out);
  rns8_status execute(rns8_matrix* output);
  const ProductArenaPlan& allocation() const;
 private:
  struct Impl;
  explicit RuntimeProductRecipe(std::unique_ptr<Impl>);
  std::unique_ptr<Impl> impl_;
};
} // namespace rns8::detail
