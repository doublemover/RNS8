namespace {
struct HostIntegerContract {
  ReconstructionContract arithmetic;
  bool zero = false;
};
struct HostIntegerWorker {
  cpp_int value;
  uint64_t first_failure = UINT64_MAX;
  rns8_status status = RNS8_SUCCESS;
};

template <class Scalar>
rns8_status export_host_integer(const rns8_context& ctx, const rns8_plan& plan,
                                const rns8_matrix& matrix, Scalar* destination, int64_t ld,
                                IntegerExportKind kind, uint32_t limbs) {
  const bool wide =
      kind == IntegerExportKind::wide_signed || kind == IntegerExportKind::wide_unsigned;
  const bool tiled = !wide && !plan.tile_schedule.empty();
  std::vector<HostIntegerContract> contracts(tiled ? plan.tile_schedule.size() : 1);
  for (size_t i = 0; i < contracts.size(); ++i) {
    auto& contract = contracts[i];
    const auto* entry = tiled ? &plan.tile_schedule[i] : nullptr;
    contract.zero = entry && (entry->flags & RNS8_TILE_SCHEDULE_ZERO_OUTPUT);
    if (contract.zero)
      continue;
    const uint32_t prefix = entry ? entry->selected_prefix : plan.prefix;
    const uint64_t bound = tiled ? plan.tile_bounds[i] : plan.desc.bound;
    const auto status =
        prepare_reconstruction_contract(kind, prefix, bound, limbs, contract.arithmetic);
    if (status != RNS8_SUCCESS)
      return status;
  }
  const int64_t cells = plan.desc.m * plan.desc.n;
  const size_t plane_stride = size_t(matrix.desc.rows) * size_t(matrix.desc.cols);
  std::vector<Scalar> staged(size_t(cells) * limbs, 0);
  uint32_t threads = 1;
#if defined(RNS8_CPU_PARALLEL_OPENMP) && RNS8_CPU_PARALLEL_OPENMP
  const uint64_t work =
      cpu_parallel_saturating_mul3(plan.desc.m, plan.desc.n, uint64_t(plan.prefix) * limbs);
  if (!omp_in_parallel() && cpu_parallel_should_use(work))
    threads = uint32_t(std::min<uint64_t>(cells, cpu_parallel_effective_threads()));
#endif
  std::vector<HostIntegerWorker> workers(threads);
  // Initialize immutable tables and reusable multiprecision storage outside
  // OpenMP. Remaining Boost failures are caught on their originating worker.
  const auto& scratch_domain =
      wide ? contracts[0].arithmetic.ring_limit : reconstruction_product(plan.prefix);
  for (auto& worker : workers) {
    worker.value = scratch_domain;
    worker.value = 0;
  }
  const auto* hooks = ctx.cpu_export_test_hooks;
  const auto export_cell = [&](int64_t cell, uint32_t thread) noexcept {
    auto& worker = workers[thread];
    // Static scheduling visits each worker's cells in increasing order. Its
    // first error is therefore enough for deterministic row-major selection.
    if (worker.status != RNS8_SUCCESS)
      return;
    try {
      if (hooks && hooks->cell)
        hooks->cell(hooks->state, cell, thread);
      const int64_t row = cell / plan.desc.n, col = cell % plan.desc.n;
      const auto& contract = contracts[tiled ? size_t(tile_index_for_cell(plan, row, col)) : 0];
      if (contract.zero)
        return;
      const auto status = reconstruct_strided(matrix.residues.data() + size_t(cell), plane_stride,
                                              contract.arithmetic, worker.value);
      if (status != RNS8_SUCCESS) {
        if (uint64_t(cell) < worker.first_failure) {
          worker.first_failure = uint64_t(cell);
          worker.status = status;
        }
        return;
      }
      if constexpr (std::is_same_v<Scalar, uint64_t>) {
        if (wide)
          store_reconstructed_limbs(worker.value, contract.arithmetic,
                                    staged.data() + size_t(cell) * limbs);
        else
          staged[size_t(cell)] = static_cast<uint64_t>(worker.value);
      } else
        staged[size_t(cell)] = static_cast<int64_t>(worker.value);
    } catch (...) {
      if (uint64_t(cell) < worker.first_failure) {
        worker.first_failure = uint64_t(cell);
        worker.status = RNS8_INTERNAL_ERROR;
      }
    }
  };
#if defined(RNS8_CPU_PARALLEL_OPENMP) && RNS8_CPU_PARALLEL_OPENMP
#pragma omp parallel for schedule(static) num_threads(threads) if (threads > 1)
  for (int64_t cell = 0; cell < cells; ++cell)
    export_cell(cell, uint32_t(omp_get_thread_num()));
#else
  for (int64_t cell = 0; cell < cells; ++cell)
    export_cell(cell, 0);
#endif
  const HostIntegerWorker* failure = nullptr;
  for (const auto& worker : workers)
    if (worker.status != RNS8_SUCCESS &&
        (!failure || worker.first_failure < failure->first_failure))
      failure = &worker;
  if (failure)
    return failure->status;
  // Publish only after every cell passed; caller row padding is never touched.
  for (int64_t row = 0; row < plan.desc.m; ++row)
    std::copy_n(staged.data() + size_t(row * plan.desc.n) * limbs, size_t(plan.desc.n) * limbs,
                destination + size_t(row * ld) * limbs);
  return RNS8_SUCCESS;
}
} // namespace
