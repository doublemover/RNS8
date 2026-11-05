#include "backend_common/abi_output_header.hpp"

namespace {
rns8_status prepare_narrow_domains(rns8_context& ctx, const rns8_matrix& a, const rns8_matrix& b,
                                   const void* native_a, const void* native_b,
                                   rns8_workspace& workspace) {
  using namespace rns8::detail;
  const rns8_matrix* inputs[] = {&a, &b};
  const void* native[] = {native_a, native_b};
  NativeSummaryIdentity desired[2];
  const auto increment = [](uint64_t& count) {
    if (count != UINT64_MAX)
      ++count;
  };
  for (unsigned side = 0; side < 2; ++side) {
    desired[side] = native_summary_identity(ctx, *inputs[side], native[side]);
    if (!desired[side].instance || !desired[side].generation)
      return RNS8_INVALID_ARGUMENT;
  }
  // Retire both changed keys before any scan can fail. An unchanged side keeps
  // its own certificate; a failed overwrite cannot revive an older generation.
  for (unsigned side = 0; side < 2; ++side)
    if (!workspace.narrow_domains[side].matches(desired[side]))
      workspace.narrow_domains[side] = {};
  for (unsigned side = 0; side < 2; ++side) {
    auto& certificate = workspace.narrow_domains[side];
    if (certificate.matches(desired[side])) {
      increment(workspace.narrow_reuses[side]);
      continue;
    }
    if (inputs[side]->native_domain.matches(desired[side])) {
      certificate = inputs[side]->native_domain;
      increment(workspace.narrow_host_proofs[side]);
      continue;
    }
    const auto& peer = workspace.narrow_domains[1 - side];
    if (peer.matches(desired[side])) {
      certificate = peer;
      increment(workspace.narrow_reuses[side]);
      continue;
    }
    auto& scratch = workspace.narrow_scan_owner;
    constexpr size_t bytes = 2 * sizeof(uint64_t);
    if (scratch.pending())
      return RNS8_BACKEND_FAILURE;
    if (scratch && (scratch.bytes() != bytes || scratch.device_id() != ctx.device_id))
      return RNS8_INVALID_ARGUMENT;
    if (!scratch) {
      const auto allocated = scratch.allocate(ctx.device_id, bytes);
      if (allocated != RNS8_SUCCESS)
        return allocated;
    }
    const auto checkpoint =
        synchronous_dispatch_checkpoint(ctx, side == 0 ? "narrow_domain_a" : "narrow_domain_b");
    if (checkpoint != RNS8_SUCCESS)
      return checkpoint;
    NativeDomainRange range;
    const auto scanned = hip_direct_scan_native_domain(
        ctx.device_id, native[side], desired[side].rows, desired[side].cols, desired[side].ld,
        desired[side].unsigned_inputs, scratch, range);
    if (scanned != RNS8_SUCCESS)
      return scanned;
    certificate = {desired[side], range};
    increment(workspace.narrow_device_scans[side]);
  }
  return RNS8_SUCCESS;
}
} // namespace

rns8_status rns8_get_narrow_summary_info(const rns8_workspace* workspace,
                                         rns8_narrow_summary_info* out) {
  if (!workspace || !rns8::detail::exact_output_header(out) || out->reserved)
    return RNS8_INVALID_ARGUMENT;
  if (workspace->in_flight)
    return RNS8_BUSY;
  if (workspace->backend != RNS8_BACKEND_HIP_DIRECT)
    return RNS8_UNSUPPORTED_BACKEND;
  rns8_narrow_summary_info info{};
  info.struct_size = sizeof(info);
  info.abi_version = RNS8_ABI_VERSION;
  info.retained_scan_bytes = workspace->narrow_scan_owner.bytes();
  info.retained_product_bytes = workspace->narrow_product_owner.bytes();
  const auto& capability = workspace->narrow_capability_observation;
  const bool observed =
      capability.provider == &rns8::detail::hip_direct_query_native_wmma_capability &&
      capability.device_id == workspace->hip_device_id && capability.target[0];
  info.wmma_capability_observed = observed;
  info.wmma_capability_supported = observed && capability.supported;
  info.a_instance = workspace->narrow_domains[0].identity.instance;
  info.a_generation = workspace->narrow_domains[0].identity.generation;
  info.b_instance = workspace->narrow_domains[1].identity.instance;
  info.b_generation = workspace->narrow_domains[1].identity.generation;
  info.a_device_scans = workspace->narrow_device_scans[0];
  info.b_device_scans = workspace->narrow_device_scans[1];
  info.a_reuses = workspace->narrow_reuses[0];
  info.b_reuses = workspace->narrow_reuses[1];
  info.a_host_proofs = workspace->narrow_host_proofs[0];
  info.b_host_proofs = workspace->narrow_host_proofs[1];
  *out = info;
  return RNS8_SUCCESS;
}
