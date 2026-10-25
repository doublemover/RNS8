#include "backend_hipblaslt/hipblaslt_backend.hpp"

namespace rns8::detail {
namespace hipblaslt_metadata_state {
thread_local HipblasLtMetadataPolicy metadata_policy = HipblasLtMetadataPolicy::per_call;
thread_local HipblasLtMetadataCounters metadata_counters;
thread_local bool algorithm_observations_enabled=false;
thread_local HipblasLtAlgorithmInventory algorithm_inventory;
thread_local HipblasLtPrepackPolicy prepack_policy=HipblasLtPrepackPolicy::both;
thread_local HipblasLtPrepackCounters prepack_counters;
}

// Host bookkeeping is available to every benchmark build. Only the enabled
// backend records real descriptor activity; disabled backends remain unsupported.
void hipblaslt_metadata_set_policy(HipblasLtMetadataPolicy policy) {
  hipblaslt_metadata_state::metadata_policy = policy;
}
HipblasLtMetadataPolicy hipblaslt_metadata_policy() {
  return hipblaslt_metadata_state::metadata_policy;
}
void hipblaslt_metadata_counters_reset() {
  hipblaslt_metadata_state::metadata_counters = {};
  hipblaslt_metadata_state::algorithm_inventory = {};
}
HipblasLtMetadataCounters hipblaslt_metadata_counters_snapshot() {
  return hipblaslt_metadata_state::metadata_counters;
}
void hipblaslt_algorithm_observations_set_enabled(bool enabled) {
  hipblaslt_metadata_state::algorithm_observations_enabled=enabled;
  hipblaslt_metadata_state::algorithm_inventory={};
}
HipblasLtAlgorithmInventory hipblaslt_algorithm_observations_snapshot() {
  auto result=hipblaslt_metadata_state::algorithm_inventory;
#if defined(RNS8_HIPBLASLT_ALGO_INDEX_AVAILABLE) && RNS8_HIPBLASLT_ALGO_INDEX_AVAILABLE
  result.available=true;
#endif
  result.complete=result.complete && result.available && hipblaslt_metadata_state::algorithm_observations_enabled;
  return result;
}
rns8_status hipblaslt_prepack_set_policy(HipblasLtPrepackPolicy policy) {
  if(static_cast<uint32_t>(policy)>3) return RNS8_INVALID_ARGUMENT;
  hipblaslt_metadata_state::prepack_policy=policy;return RNS8_SUCCESS;
}
HipblasLtPrepackPolicy hipblaslt_prepack_policy() {
  return hipblaslt_metadata_state::prepack_policy;
}
void hipblaslt_prepack_counters_reset() {
  hipblaslt_metadata_state::prepack_counters={};
}
HipblasLtPrepackCounters hipblaslt_prepack_counters_snapshot() {
  return hipblaslt_metadata_state::prepack_counters;
}
}  // namespace rns8::detail
