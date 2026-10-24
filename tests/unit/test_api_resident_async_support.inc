#include "core/hip_resources.hpp"
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
static void exercise_resident_async_submission(
    rns8_resident_graph *graph, rns8_matrix *a, rns8_matrix *b, rns8_matrix *c,
    rns8_matrix *d, rns8_matrix *e, rns8_workspace *workspace,
    rns8_graph_execution_info &info, bool replay = false) {
  REQUIRE(rns8_prepare_resident_graph_hip(graph) == RNS8_SUCCESS);
  rns8::detail::hip_unique_stream stream;
  REQUIRE(stream.create_non_blocking() == hipSuccess);
  const auto cg = c->content_generation, dg = d->content_generation,
             eg = e->content_generation;
  const auto submit =
      replay ? rns8_replay_resident_graph_hip : rns8_submit_resident_graph_hip;
  if (replay) {
    REQUIRE(rns8_capture_resident_graph_hip(graph) == RNS8_SUCCESS);
    REQUIRE(rns8_capture_resident_graph_hip(graph) == RNS8_SUCCESS);
    REQUIRE(c->content_generation == cg);
    REQUIRE(d->content_generation == dg);
    REQUIRE(e->content_generation == eg);
    REQUIRE_FALSE(workspace->in_flight);
    rns8::detail::hip_direct_device_buffer replacement;
    REQUIRE(replacement.allocate(0, c->hip_residue_bytes) == RNS8_SUCCESS);
    struct Restore {
      void *&target;
      void *prior;
      ~Restore() { target = prior; }
    } restore{c->hip_residues, c->hip_residues};
    c->hip_residues = replacement.get();
    rns8_graph_completion_token rejected{77, 88};
    REQUIRE(submit(graph, stream.get(), &rejected) == RNS8_UNSUPPORTED_BACKEND);
    REQUIRE(rejected.graph_id == 77);
    REQUIRE(rejected.epoch == 88);
  }
  rns8_graph_completion_token token{77, 88};
  a->in_flight_readers = UINT64_MAX;
  REQUIRE(submit(graph, stream.get(), &token) == RNS8_RANGE_ERROR);
  a->in_flight_readers = 0;
  REQUIRE(token.graph_id == 77);
  REQUIRE(token.epoch == 88);
  REQUIRE(c->content_generation == cg);
  REQUIRE(d->content_generation == dg);
  REQUIRE(hipStreamBeginCapture(stream.get(),
                                hipStreamCaptureModeThreadLocal) == hipSuccess);
  REQUIRE(submit(graph, stream.get(), &token) == RNS8_UNSUPPORTED_BACKEND);
  hipGraph_t empty = nullptr;
  REQUIRE(hipStreamEndCapture(stream.get(), &empty) == hipSuccess);
  if (empty)
    REQUIRE(hipGraphDestroy(empty) == hipSuccess);
  REQUIRE_FALSE(workspace->in_flight);
  REQUIRE(a->in_flight_readers == 0);
  REQUIRE(b->in_flight_readers == 0);
  for (auto* output : {c, d, e}) REQUIRE_FALSE(output->in_flight_writer);
  for (auto special : {hipStreamLegacy, hipStreamPerThread}) {
    REQUIRE(submit(graph, special, &token) == RNS8_INVALID_ARGUMENT);
    REQUIRE(token.epoch == 88);
  }
  REQUIRE(submit(graph, stream.get(), &token) == RNS8_SUCCESS);
  REQUIRE(rns8_cancel_resident_graph_hip(graph, token) == RNS8_SUCCESS);
  REQUIRE(rns8_cancel_resident_graph_hip(graph, token) == RNS8_SUCCESS);
  REQUIRE(rns8_destroy_resident_graph(graph) == RNS8_BUSY);
  REQUIRE(workspace->in_flight);
  REQUIRE(rns8_wait_resident_graph_hip(graph, token, &info) == RNS8_SUCCESS);
  REQUIRE(rns8_wait_resident_graph_hip_for(graph, token, 0, &info) == RNS8_SUCCESS);
  REQUIRE(info.operation_status == RNS8_CANCELED);
  REQUIRE(info.completed_nodes == 0);
  for (auto* output : {c, d, e}) {
    REQUIRE_FALSE(output->device_residues_current);
    REQUIRE_FALSE(output->in_flight_writer);
  }
  REQUIRE(c->content_generation == cg);
  REQUIRE(d->content_generation == dg);
  REQUIRE(e->content_generation == eg);
  rns8_async_failure_info failure{};
  failure.struct_size = sizeof(failure); failure.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_resident_graph_async_failure(graph, token, &failure) == RNS8_SUCCESS);
  REQUIRE(failure.stage == RNS8_ASYNC_FAILURE_NONE);
  REQUIRE(failure.failed_node == UINT32_MAX);
  rns8::detail::hip_direct_allocation_counters_reset();
  REQUIRE(submit(graph, stream.get(), &token) == RNS8_SUCCESS);
  REQUIRE(a->in_flight_readers == 1);
  REQUIRE(b->in_flight_readers == 1);
  REQUIRE(c->in_flight_writer);
  REQUIRE(d->in_flight_writer);
  REQUIRE(e->in_flight_writer);
  REQUIRE_FALSE(c->device_residues_current);
  REQUIRE_FALSE(d->device_residues_current);
  REQUIRE_FALSE(e->device_residues_current);
  REQUIRE(c->content_generation == cg);
  REQUIRE(d->content_generation == dg);
  REQUIRE(e->content_generation == eg);
  REQUIRE(workspace->in_flight);
  REQUIRE(rns8_destroy_resident_graph(graph) == RNS8_BUSY);
  REQUIRE(rns8_refresh_resident_graph_inputs(graph) == RNS8_BUSY);
  REQUIRE(rns8_execute_resident_graph(graph, &info) == RNS8_BUSY);
  REQUIRE(rns8_destroy_workspace(workspace) == RNS8_BUSY);
  auto unused = token;
  REQUIRE(submit(graph, stream.get(), &unused) == RNS8_BUSY);
  REQUIRE(unused.epoch == token.epoch);
  auto stale = token;
  --stale.epoch;
  REQUIRE(rns8_query_resident_graph_hip(graph, stale, &info) ==
          RNS8_INVALID_ARGUMENT);
  auto wrong = token;
  ++wrong.graph_id;
  REQUIRE(rns8_wait_resident_graph_hip(graph, wrong, &info) ==
          RNS8_INVALID_ARGUMENT);
  REQUIRE(rns8_wait_resident_graph_hip(graph, token, &info) == RNS8_SUCCESS);
  REQUIRE(info.operation_status == RNS8_SUCCESS);
  REQUIRE(info.epoch == token.epoch);
  REQUIRE(info.failed_node == UINT32_MAX);
  REQUIRE(c->content_generation == cg + 1);
  REQUIRE(d->content_generation == dg + 1);
  REQUIRE(e->content_generation == eg + 1);
  REQUIRE(a->in_flight_readers == 0);
  REQUIRE(b->in_flight_readers == 0);
  REQUIRE_FALSE(workspace->in_flight);
  REQUIRE(
      rns8::detail::hip_direct_allocation_counters_snapshot().allocate_calls ==
      0);
  REQUIRE(rns8_query_resident_graph_hip(graph, token, &info) == RNS8_SUCCESS);
}
#endif
