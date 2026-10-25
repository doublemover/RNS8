#include "backend_hipblaslt/hipblaslt_backend.hpp"

namespace rns8::detail {
namespace hipblaslt_metadata_state {
thread_local HipblasLtMetadataPolicy metadata_policy = HipblasLtMetadataPolicy::per_call;
thread_local HipblasLtMetadataCounters metadata_counters;
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
}
HipblasLtMetadataCounters hipblaslt_metadata_counters_snapshot() {
  return hipblaslt_metadata_state::metadata_counters;
}
}  // namespace rns8::detail
