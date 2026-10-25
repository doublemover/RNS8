#pragma once
#include "rns8/rns8.h"
namespace rns8::detail {
// Read-only current-owner proof and physically retained payload. Source identity
// is checked again on every recipe replay, not inferred from a stale snapshot.
rns8_status ragged_recipe_binding_info(rns8_context*,const rns8_ragged_materialization*,
    const rns8_matrix*& source,uint64_t& retained_bytes,uint64_t& stage_capacity);
rns8_status retain_ragged_recipe_binding(rns8_ragged_materialization*);
void release_ragged_recipe_binding(rns8_ragged_materialization*) noexcept;
}
