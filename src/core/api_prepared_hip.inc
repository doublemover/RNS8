namespace {
void finish_prepared_hip(rns8_prepared_execution &e, rns8_status operation) {
  --e.a->in_flight_readers;
  if (e.a != e.b)
    --e.b->in_flight_readers;
  e.c->in_flight_writer = false;
  e.workspace->in_flight = false;
  if (operation == RNS8_SUCCESS) {
    mark_output_device_residues_current(*e.c);
    rns8::detail::commit_exact_range(*e.c, *e.plan, e.pending_bound,
                                     &e.pending_ranges);
  }
  e.completion = {};
  e.completion.operation_status = operation;
  e.completion.a_instance = e.a_instance;
  e.completion.a_generation = e.a_generation;
  e.completion.b_instance = e.b_instance;
  e.completion.b_generation = e.b_generation;
  e.completion.c_instance = e.c->matrix_instance_id;
  e.completion.c_generation = e.c->content_generation;
  e.completion.completed = 1;
  e.running = false;
  e.pending_stream = nullptr;
}
rns8_status observe_prepared_hip(rns8_prepared_execution *e,
                                 rns8_completion_token token,
                                 rns8_completion_info *out, bool wait) {
  return guard_api([&]() -> rns8_status {
    if (!e || !out ||
        !rns8::detail::valid_abi(out->struct_size, out->abi_version,
                                 sizeof(*out)) ||
        !e->hip_submission_prepared || token.prepared_id != e->id ||
        !token.epoch || token.epoch != e->epoch)
      return RNS8_INVALID_ARGUMENT;
    if (!e->running)
      return rns8_get_prepared_completion(e, token, out);
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
    if (!e->pending_stream || hipSetDevice(e->ctx->device_id) != hipSuccess)
      return RNS8_BACKEND_FAILURE;
    const auto error = wait ? hipEventSynchronize(e->completion_event.get())
                            : hipEventQuery(e->completion_event.get());
    if (error == hipErrorNotReady)
      return RNS8_BUSY;
    if (error != hipSuccess) {
      // On an event error, drain this stream before releasing reserved buffers.
      (void)hipStreamSynchronize(static_cast<hipStream_t>(e->pending_stream));
    }
    finish_prepared_hip(*e, error == hipSuccess ? RNS8_SUCCESS
                                                : RNS8_BACKEND_FAILURE);
    return rns8_get_prepared_completion(e, token, out);
#else
    (void)wait;
    return RNS8_UNSUPPORTED_BACKEND;
#endif
  });
}
} // namespace
rns8_status rns8_prepare_hip_submission(rns8_prepared_execution *e) {
  return guard_api([&]() -> rns8_status {
    if (!e)
      return RNS8_INVALID_ARGUMENT;
    if (e->running)
      return RNS8_BUSY;
    if (e->plan->backend != RNS8_BACKEND_HIP_DIRECT ||
        !rns8::detail::exact_wide_semantics(e->plan->desc.semantics) ||
        !e->plan->tile_schedule.empty())
      return RNS8_UNSUPPORTED_BACKEND;
    const auto status = validate_prepared(*e);
    if (status != RNS8_SUCCESS)
      return status;
    if (!inputs_match(*e))
      return RNS8_INVALID_ARGUMENT;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
    if (hipSetDevice(e->ctx->device_id) != hipSuccess)
      return RNS8_BACKEND_FAILURE;
    if (!e->completion_event && e->completion_event.create() != hipSuccess)
      return RNS8_BACKEND_FAILURE;
    e->hip_submission_prepared = true;
    return RNS8_SUCCESS;
#else
    return RNS8_UNSUPPORTED_BACKEND;
#endif
  });
}
rns8_status rns8_submit_prepared_hip(rns8_prepared_execution *e, void *stream,
                                     rns8_completion_token *token) {
  return guard_api([&]() -> rns8_status {
    if (!e || !stream || !token)
      return RNS8_INVALID_ARGUMENT;
    if (e->running)
      return RNS8_BUSY;
    if (!e->hip_submission_prepared)
      return RNS8_UNSUPPORTED_BACKEND;
    if (e->epoch == UINT64_MAX || e->a->in_flight_readers == UINT64_MAX ||
        e->b->in_flight_readers == UINT64_MAX)
      return RNS8_RANGE_ERROR;
    auto status = validate_prepared(*e);
    if (status != RNS8_SUCCESS)
      return status;
    if (!inputs_match(*e))
      return RNS8_INVALID_ARGUMENT;
    boost::multiprecision::cpp_int bound;
    rns8_exact_matrix_ranges ranges;
    status =
        rns8::detail::exact_gemm_range(*e->plan, *e->a, *e->b, bound, &ranges);
    if (status != RNS8_SUCCESS)
      return status;
    if (!rns8::detail::qualified_i32_gemm_address(
            e->plan->desc.m, e->plan->desc.n, e->plan->desc.k, e->a->desc.cols,
            e->b->desc.cols, e->c->desc.cols, e->plan->prefix))
      return RNS8_UNSUPPORTED_BACKEND;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
    if (hipSetDevice(e->ctx->device_id) != hipSuccess)
      return RNS8_BACKEND_FAILURE;
    hipDevice_t stream_device = -1;
    if (hipStreamGetDevice(static_cast<hipStream_t>(stream), &stream_device) !=
            hipSuccess ||
        stream_device != e->ctx->device_id)
      return RNS8_INVALID_ARGUMENT;
    hipStreamCaptureStatus capture = hipStreamCaptureStatusNone;
    if (hipStreamIsCapturing(static_cast<hipStream_t>(stream), &capture) !=
        hipSuccess)
      return RNS8_BACKEND_FAILURE;
    if (capture != hipStreamCaptureStatusNone)
      return RNS8_UNSUPPORTED_BACKEND;
    e->pending_bound.swap(bound);
    e->pending_ranges = std::move(ranges);
    invalidate_output_currentness(*e->c);
    ++e->a->in_flight_readers;
    if (e->a != e->b)
      ++e->b->in_flight_readers;
    e->c->in_flight_writer = true;
    e->workspace->in_flight = true;
    e->running = true;
    e->pending_stream = stream;
    ++e->epoch;
    *token = {e->id, e->epoch};
    status = rns8::detail::hip_direct_enqueue_rns_no_publish(
        e->plan, e->a, e->b, e->c, stream);
    if (status == RNS8_SUCCESS &&
        hipEventRecord(e->completion_event.get(),
                       static_cast<hipStream_t>(stream)) != hipSuccess)
      status = RNS8_BACKEND_FAILURE;
    if (status != RNS8_SUCCESS) {
      (void)hipStreamSynchronize(static_cast<hipStream_t>(stream));
      finish_prepared_hip(*e, status);
    }
    return status;
#else
    return RNS8_UNSUPPORTED_BACKEND;
#endif
  });
}
rns8_status rns8_query_prepared_hip(rns8_prepared_execution *e,
                                    rns8_completion_token token,
                                    rns8_completion_info *out) {
  return observe_prepared_hip(e, token, out, false);
}
rns8_status rns8_wait_prepared_hip(rns8_prepared_execution *e,
                                   rns8_completion_token token,
                                   rns8_completion_info *out) {
  return observe_prepared_hip(e, token, out, true);
}
