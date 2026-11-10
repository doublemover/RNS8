#pragma once
#include "core/host_control_observations.hpp"
#include <nlohmann/json.hpp>

namespace rns8::bench {
inline constexpr const char* host_proof_phase_contract =
    "calling_thread_steady_ns_inclusive_scopes_nonadditive_v1";
// Serialize only after observed/timed calls. Construction is never a hot-loop callback.
inline nlohmann::json host_control_json(const detail::HostControlObservations& value) {
  return {{"device_gets", value.device_gets},
          {"device_sets", value.device_sets},
          {"stream_device_queries", value.stream_device_queries},
          {"stream_capture_queries", value.stream_capture_queries},
          {"event_creates", value.event_creates},
          {"event_destroys", value.event_destroys},
          {"event_records", value.event_records},
          {"event_queries", value.event_queries},
          {"event_waits", value.event_waits},
          {"submission_proofs", value.submission_proofs},
          {"proof_builds", value.proof_builds},
          {"proof_reuses", value.proof_reuses},
          {"range_proof_calls", value.range_proof_calls},
          {"range_proof_host_ns", value.range_proof_host_ns},
          {"proof_identity_host_ns", value.proof_identity_host_ns},
          {"proof_metadata_host_ns", value.proof_metadata_host_ns},
          {"graph_preflight_host_ns", value.graph_preflight_host_ns},
          {"graph_launches", value.graph_launches}};
}
} // namespace rns8::bench
