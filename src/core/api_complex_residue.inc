namespace rns8::detail::api {
namespace {
// Synchronous API calls require external host-thread exclusion, as elsewhere in
// this API. Reservations also exclude outstanding asynchronous owners.
struct ComplexWriteReservation {
  std::array<const rns8_matrix *, 4> inputs;
  rns8_matrix &real;
  rns8_matrix &imag;
  rns8_workspace &workspace;
  bool commit_started = false, published = false;
  ComplexWriteReservation(std::array<const rns8_matrix *, 4> values, rns8_matrix &r, rns8_matrix &i,
                          rns8_workspace &w)
      : inputs(values), real(r), imag(i), workspace(w) {
    for (std::size_t n = 0; n < inputs.size(); ++n)
      if (std::find(inputs.begin(), inputs.begin() + n, inputs[n]) == inputs.begin() + n)
        ++inputs[n]->in_flight_readers;
    real.in_flight_writer = imag.in_flight_writer = workspace.in_flight = true;
  }
  ~ComplexWriteReservation() {
    for (std::size_t n = 0; n < inputs.size(); ++n)
      if (std::find(inputs.begin(), inputs.begin() + n, inputs[n]) == inputs.begin() + n)
        --inputs[n]->in_flight_readers;
    real.in_flight_writer = imag.in_flight_writer = workspace.in_flight = false;
    if (commit_started && !published) {
      invalidate_output_currentness(real);
      invalidate_output_currentness(imag);
    }
  }
};

void cpu_complex_residue_product(const ComplexResiduePair<const int8_t> &a,
                                 const ComplexResiduePair<const int8_t> &b, int8_t *real,
                                 int8_t *imag, ComplexProductMethod method) {
  const auto m = a.real.rows, n = b.real.cols, k = a.real.cols;
  for (uint32_t p = 0; p < a.real.basis.prefix; ++p) {
    const int64_t modulus = rns8_default_modulus(p);
    const auto centered = [modulus](int64_t value) {
      value %= modulus;
      if (value < 0)
        value += modulus;
      if (value >= (modulus + 1) / 2)
        value -= modulus;
      return value;
    };
    for (int64_t row = 0; row < m; ++row)
      for (int64_t col = 0; col < n; ++col) {
        int64_t p0 = 0, p1 = 0, p2 = 0, p3 = 0;
        for (int64_t q = 0; q < k; ++q) {
          const int64_t ar = a.real.data[(int64_t(p) * m + row) * a.real.ld + q];
          const int64_t ai = a.imag.data[(int64_t(p) * m + row) * a.imag.ld + q];
          const int64_t br = b.real.data[(int64_t(p) * k + q) * b.real.ld + col];
          const int64_t bi = b.imag.data[(int64_t(p) * k + q) * b.imag.ld + col];
          p0 = centered(p0 + ar * br);
          p1 = centered(p1 + ai * bi);
          if (method == ComplexProductMethod::three_real)
            p2 = centered(p2 + centered(ar + ai) * centered(br + bi));
          else {
            p2 = centered(p2 + ar * bi);
            p3 = centered(p3 + ai * br);
          }
        }
        const auto offset = (int64_t(p) * m + row) * n + col;
        real[offset] = static_cast<int8_t>(centered(p0 - p1));
        imag[offset] = static_cast<int8_t>(
            centered(method == ComplexProductMethod::three_real ? p2 - p0 - p1 : p2 + p3));
      }
  }
}
} // namespace

rns8_status complex_exact_signed_gemm(rns8_context *ctx, const rns8_plan *plan,
                                      ComplexMatrixPair<const rns8_matrix> a,
                                      ComplexMatrixPair<const rns8_matrix> b,
                                      ComplexMatrixPair<rns8_matrix> c, ComplexProductMethod method,
                                      uint64_t max_scratch_bytes, rns8_workspace *workspace) {
  return guard_api([&]() -> rns8_status {
    if (!ctx || !plan || !workspace || !a.real || !a.imag || !b.real || !b.imag || !c.real ||
        !c.imag || c.real == c.imag || plan->desc.semantics != RNS8_EXACT_WIDE_SIGNED)
      return RNS8_INVALID_ARGUMENT;
    const std::array<const rns8_matrix *, 4> inputs{a.real, a.imag, b.real, b.imag};
    for (const auto *input : inputs)
      if (input == c.real || input == c.imag)
        return RNS8_INVALID_ARGUMENT;
    if (plan->backend != RNS8_BACKEND_CPU_REFERENCE && plan->backend != RNS8_BACKEND_HIP_DIRECT)
      return RNS8_UNSUPPORTED_BACKEND;
    auto status = validate_plan_context_workspace(*ctx, *plan, *workspace);
    if (status != RNS8_SUCCESS)
      return status;
    status = validate_rns_gemm_operands(*ctx, *plan, *a.real, *b.real, *c.real, true);
    if (status != RNS8_SUCCESS)
      return status;
    status = validate_rns_gemm_operands(*ctx, *plan, *a.imag, *b.imag, *c.imag, true);
    if (status != RNS8_SUCCESS)
      return status;
    for (const auto *input : inputs) {
      if (input->in_flight_readers == UINT64_MAX || input->in_flight_writer)
        return RNS8_BUSY;
      if (input->exact_range_prefix < plan->prefix)
        return RNS8_RANGE_ERROR;
    }
    if (c.real->generation_exhausted || c.imag->generation_exhausted)
      return RNS8_RANGE_ERROR;
    const bool hip = plan->backend == RNS8_BACKEND_HIP_DIRECT;
    const ComplexResidueBasis basis{ComplexResidueDomain::wide_signed, 1, plan->prefix, 0};
    const auto view = [&](const rns8_matrix *matrix) {
      return ComplexResidueView<const int8_t>{
          hip ? static_cast<const int8_t *>(matrix->hip_residues) : matrix->residues.data(),
          matrix->desc.rows, matrix->desc.cols, matrix->desc.cols, basis};
    };
    const auto output = [&](rns8_matrix *matrix) {
      return ComplexResidueView<int8_t>{
          hip ? static_cast<int8_t *>(matrix->hip_residues) : matrix->residues.data(),
          matrix->desc.rows, matrix->desc.cols, matrix->desc.cols, basis};
    };
    const ComplexResiduePair<const int8_t> av{view(a.real), view(a.imag)},
        bv{view(b.real), view(b.imag)};
    const ComplexResiduePair<int8_t> cv{output(c.real), output(c.imag)};
    const ComplexMagnitudeBounds bounds{a.real->exact_max_magnitude, a.imag->exact_max_magnitude,
                                        b.real->exact_max_magnitude, b.imag->exact_max_magnitude};
    auto admitted = plan_complex_residue_product(av, bv, cv, bounds, method, max_scratch_bytes);
    if (admitted.status != RNS8_SUCCESS)
      return admitted.status;
    // Finish all CPU scratch/proof allocation before taking ownership or writing.
    std::vector<int8_t> stage;
    if (!hip)
      stage.resize(static_cast<std::size_t>(admitted.product_bytes * 2));
    ComplexWriteReservation write(inputs, *c.real, *c.imag, *workspace);
    if (hip) {
      status = hip_direct_gemm_complex_residues(ctx->device_id, av, bv, cv, bounds, method,
                                                max_scratch_bytes, write.commit_started);
      if (status != RNS8_SUCCESS)
        return status;
    } else {
      cpu_complex_residue_product(av, bv, stage.data(), stage.data() + admitted.product_bytes,
                                  method);
      write.commit_started = true;
      std::copy_n(stage.data(), static_cast<std::size_t>(admitted.product_bytes), cv.real.data);
      std::copy_n(stage.data() + admitted.product_bytes,
                  static_cast<std::size_t>(admitted.product_bytes), cv.imag.data);
    }
    // No allocation or fallible device operation between the two publications.
    // External host-thread exclusion makes this one logical synchronous commit.
    if (hip) {
      mark_output_device_residues_current(*c.real);
      mark_output_device_residues_current(*c.imag);
    } else {
      mark_output_host_residues_current(*c.real);
      mark_output_host_residues_current(*c.imag);
    }
    commit_exact_range(*c.real, *plan, admitted.real_bound);
    commit_exact_range(*c.imag, *plan, admitted.imag_bound);
    write.published = true;
    return RNS8_SUCCESS;
  });
}
} // namespace rns8::detail::api
