namespace {
rns8_status prepare_adaptive_summaries(rns8_context& ctx, const rns8_matrix& a,
    const rns8_matrix& b, const void* native_a, const void* native_b,
    rns8::detail::NativeBoundMethod method, uint32_t panel_k,
    const rns8::detail::AdaptiveNativeSupport& support, rns8_workspace& workspace) {
  using namespace rns8::detail;
  auto& storage = workspace.adaptive_summary_owner;
  const uint64_t bytes = support.discovery_bytes - 43*sizeof(uint32_t);
  const uint64_t rows = (uint64_t(a.desc.rows)+15)/16;
  const uint64_t cols = (uint64_t(b.desc.cols)+15)/16;
  const uint64_t a_bytes = bytes/(rows+cols)*rows;
  if (storage.pending()) return RNS8_BACKEND_FAILURE;
  if (storage.bytes() != bytes) {
    workspace.adaptive_summary_identity[0] = {};
    workspace.adaptive_summary_identity[1] = {};
    const auto status = storage.allocate(ctx.device_id,size_t(bytes));
    if (status != RNS8_SUCCESS) return status;
  }
  const rns8_matrix* inputs[] = {&a,&b};
  const void* native[] = {native_a,native_b};
  const uint64_t side_bytes[] = {a_bytes,bytes-a_bytes};
  const uint64_t offsets[] = {0,a_bytes};
  for (unsigned side=0; side<2; ++side) {
    const auto& input = *inputs[side];
    NativeSummaryIdentity desired;
    desired.instance=input.matrix_instance_id; desired.generation=input.content_generation;
    desired.native=native[side]; desired.rows=input.desc.rows; desired.cols=input.desc.cols;
    desired.ld=input.desc.cols; desired.device=ctx.device_id; desired.method=method;
    desired.panel_k=method==NativeBoundMethod::panel_l1 ? panel_k : 0;
    desired.unsigned_inputs=input.desc.semantics==RNS8_EXACT_WIDE_UNSIGNED;
    desired.row_tiles=side==0;
    auto& identity=workspace.adaptive_summary_identity[side];
    if (identity==desired) {
      auto& count=workspace.adaptive_summary_reuses[side];
      if (count!=UINT64_MAX) ++count;
      continue;
    }
    // Never let a failed overwrite retain its predecessor's certificate.
    identity={}; storage.mark_pending();
    const auto status=hip_direct_scan_native_summary(ctx.device_id,native[side],input.desc.rows,input.desc.cols,
        input.desc.cols,desired.unsigned_inputs,side==0,method,panel_k,
        static_cast<uint8_t*>(storage.get())+offsets[side],side_bytes[side]);
    if (status!=RNS8_SUCCESS) return status;
    storage.mark_completed(); identity=desired;
    auto& count=workspace.adaptive_summary_scans[side];
    if (count!=UINT64_MAX) ++count;
  }
  return RNS8_SUCCESS;
}
} // namespace

rns8_status rns8_get_adaptive_summary_info(const rns8_workspace* workspace,rns8_adaptive_summary_info* out) {
  if (!workspace || !out || !rns8::detail::valid_abi(out->struct_size,out->abi_version,sizeof(*out)))
    return RNS8_INVALID_ARGUMENT;
  if (workspace->in_flight) return RNS8_BUSY;
  rns8_adaptive_summary_info info{};
  info.struct_size=sizeof(info); info.abi_version=RNS8_ABI_VERSION;
  info.retained_bytes=workspace->adaptive_summary_owner.bytes();
  info.a_instance=workspace->adaptive_summary_identity[0].instance;
  info.a_generation=workspace->adaptive_summary_identity[0].generation;
  info.b_instance=workspace->adaptive_summary_identity[1].instance;
  info.b_generation=workspace->adaptive_summary_identity[1].generation;
  info.a_scans=workspace->adaptive_summary_scans[0]; info.b_scans=workspace->adaptive_summary_scans[1];
  info.a_reuses=workspace->adaptive_summary_reuses[0]; info.b_reuses=workspace->adaptive_summary_reuses[1];
  *out=info; return RNS8_SUCCESS;
}
