namespace {
int calibrate_complex(const std::string &candidate, uint32_t m, uint32_t n, uint32_t k,
                      uint32_t prefix, uint32_t group, bool unsigned_inputs, uint32_t repeats,
                      uint32_t warmups, const std::string &run_id) {
  using Wide = boost::multiprecision::int256_t;
  if (group != 1 || unsigned_inputs || prefix != 20 || run_id.empty() ||
      uint64_t(m) * n * k > 1073741824)
    throw std::invalid_argument(
        "complex calibration requires signed inputs, prefix20, group1 and bounded reference work");
  rns8_device_info device{};
  check(hip_direct_probe(0, device));
  const auto method = candidate == "complex_3m" ? ComplexProductMethod::three_real
                                                : ComplexProductMethod::four_real;
  const std::size_t ac = std::size_t(m) * k, bc = std::size_t(k) * n, cc = std::size_t(m) * n;
  std::array<std::vector<int64_t>, 4> native;
  std::mt19937_64 rng(3135);
  for (std::size_t j = 0; j < native.size(); ++j) {
    native[j].resize(j < 2 ? ac : bc);
    for (auto &value : native[j]) {
      const uint64_t bits = rng();
      std::memcpy(&value, &bits, sizeof(value));
    }
    native[j][0] = j % 2 ? INT64_MAX : INT64_MIN;
  }
  std::array<std::vector<int8_t>, 2> expected;
  for (auto &output : expected)
    output.assign(cc * prefix + 8, 99);
  {
    std::array<std::vector<Wide>, 4> values;
    for (std::size_t j = 0; j < values.size(); ++j) {
      values[j].resize(native[j].size());
      for (std::size_t q = 0; q < values[j].size(); ++q) {
        // Transpose B components for contiguous independent-reference dot products.
        if (j < 2)
          values[j][q] = native[j][q];
        else
          values[j][(q % n) * k + q / n] = native[j][q];
      }
    }
    for (uint32_t row = 0; row < m; ++row)
      for (uint32_t col = 0; col < n; ++col) {
        Wide real = 0, imag = 0;
        cpp_int audit_r = 0, audit_i = 0;
        for (uint32_t q = 0; q < k; ++q) {
          const auto aindex = std::size_t(row) * k + q, bindex = std::size_t(col) * k + q;
          real += values[0][aindex] * values[2][bindex] - values[1][aindex] * values[3][bindex];
          imag += values[0][aindex] * values[3][bindex] + values[1][aindex] * values[2][bindex];
          if (col == 0 || col + 1 == n) {
            const auto native_b = std::size_t(q) * n + col;
            audit_r += cpp_int(native[0][aindex]) * native[2][native_b] -
                       cpp_int(native[1][aindex]) * native[3][native_b];
            audit_i += cpp_int(native[0][aindex]) * native[3][native_b] +
                       cpp_int(native[1][aindex]) * native[2][native_b];
          }
        }
        if ((col == 0 || col + 1 == n) && (cpp_int(real) != audit_r || cpp_int(imag) != audit_i))
          throw std::runtime_error(
              "complex fixed-width reference differs from independent cpp_int audit");
        for (uint32_t p = 0; p < prefix; ++p) {
          const int modulus = rns8_default_modulus(p);
          const auto centered = [modulus](const Wide &value) {
            int result = (value % modulus).convert_to<int>();
            if (result < 0)
              result += modulus;
            if (result >= (modulus + 1) / 2)
              result -= modulus;
            return static_cast<int8_t>(result);
          };
          expected[0][std::size_t(p) * cc + std::size_t(row) * n + col] = centered(real);
          expected[1][std::size_t(p) * cc + std::size_t(row) * n + col] = centered(imag);
        }
      }
  }
  std::array<hip_direct_device_buffer, 4> inputs, residues;
  std::array<hip_direct_device_buffer, 2> outputs;
  const auto input_start = std::chrono::steady_clock::now();
  for (std::size_t j = 0; j < inputs.size(); ++j) {
    check(inputs[j].allocate(0, native[j].size() * 8));
    check(residues[j].allocate(0, native[j].size() * prefix));
    check(
        hip_direct_copy_host_to_device(0, inputs[j].get(), native[j].data(), native[j].size() * 8));
  }
  for (std::size_t j = 0; j < outputs.size(); ++j)
    check(outputs[j].allocate(0, expected[j].size()));
  hip_direct_timing_set_enabled(false);
  pack_pair(inputs[0].get(), inputs[2].get(), residues[0].get(), residues[2].get(), m, n, k, prefix,
            false);
  pack_pair(inputs[1].get(), inputs[3].get(), residues[1].get(), residues[3].get(), m, n, k, prefix,
            false);
  const double input_prepare_ns =
      std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - input_start)
          .count();
  const ComplexResidueBasis basis{ComplexResidueDomain::wide_signed, 1, prefix, 0};
  const ComplexResiduePair<const int8_t> a{
      {static_cast<const int8_t *>(residues[0].get()), m, k, k, basis},
      {static_cast<const int8_t *>(residues[1].get()), m, k, k, basis}};
  const ComplexResiduePair<const int8_t> b{
      {static_cast<const int8_t *>(residues[2].get()), k, n, n, basis},
      {static_cast<const int8_t *>(residues[3].get()), k, n, n, basis}};
  const ComplexResiduePair<int8_t> c{{static_cast<int8_t *>(outputs[0].get()), m, n, n, basis},
                                     {static_cast<int8_t *>(outputs[1].get()), m, n, n, basis}};
  const cpp_int magnitude = cpp_int(1) << 63;
  const ComplexMagnitudeBounds bounds{magnitude, magnitude, magnitude, magnitude};
  const auto plan = plan_complex_residue_product(a, b, c, bounds, method, UINT64_MAX);
  check(plan.status);
  const auto prepare = [&] {
    const std::vector<int8_t> poison(expected[0].size(), 99);
    for (auto &output : outputs)
      check(hip_direct_copy_host_to_device(0, output.get(), poison.data(), poison.size()));
  };
  const auto execute = [&] {
    bool written = false;
    check(
        hip_direct_gemm_complex_residues(0, a, b, c, bounds, method, plan.scratch_bytes, written));
    if (!written)
      throw std::runtime_error("complex operation did not reach paired commit");
  };
  const auto verify = [&] {
    std::vector<int8_t> observed(expected[0].size());
    for (std::size_t j = 0; j < outputs.size(); ++j) {
      check(hip_direct_copy_device_to_host(0, observed.data(), outputs[j].get(), observed.size()));
      if (observed != expected[j])
        throw std::runtime_error("complex full residue/reference/canary mismatch");
    }
  };
  for (uint32_t i = 0; i < warmups; ++i) {
    prepare();
    execute();
    verify();
  }
  nlohmann::json samples = nlohmann::json::array();
  for (uint32_t i = 0; i < repeats; ++i) {
    hip_direct_timing_set_enabled(true);
    prepare();
    hip_direct_timing_reset();
    const auto start = std::chrono::steady_clock::now();
    execute();
    const double elapsed =
        std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start).count();
    hip_direct_timing_flush_pending_events();
    nlohmann::json events = nlohmann::json::array();
    for (const auto &event : hip_direct_timing_snapshot())
      events.push_back(
          {{"label", event.label},
           {"elapsed_ns", event.microseconds > 0 ? nlohmann::json(event.microseconds * 1000)
                                                 : nlohmann::json(nullptr)}});
    verify();
    hip_direct_timing_set_enabled(false);
    prepare();
    const auto production = std::chrono::steady_clock::now();
    execute();
    const double production_ns =
        std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - production)
            .count();
    verify();
    samples.push_back({{"host_call_ns", elapsed},
                       {"uninstrumented_host_call_ns", production_ns},
                       {"exact_reference_passed", true},
                       {"uninstrumented_exact_reference_passed", true},
                       {"events", events}});
  }
  const auto sustained = sustained_calls(prepare, execute, verify);
  std::cout
      << nlohmann::json{{"schema", "rns8_native_microcase_v2"},
                        {"candidate", candidate},
                        {"run_id", run_id},
                        {"m", m},
                        {"n", n},
                        {"k", k},
                        {"prefix", prefix},
                        {"group", group},
                        {"unsigned_inputs", false},
                        {"splits", 1u},
                        {"requested_workers", 0u},
                        {"effective_workers", 0u},
                        {"queue_descriptors", 0u},
                        {"queue_kernel_selection", "not_applicable"},
                        {"seed", 3135},
                        {"repeats", repeats},
                        {"warmups", warmups},
                        {"dataset", "split_complex_full_signed64_mt19937_3135_extrema_v1"},
                        {"input_domain", "resident_split_complex_centered_residues"},
                        {"output_contract", "paired_exact_signed_residues_no_integer_export"},
                        {"real_products", plan.products},
                        {"scratch_peak_bytes", plan.scratch_bytes},
                        {"scratch_accounting", "explicit_queried_capacity_not_measured_peak_or_HBM"},
                        {"scratch_layout_version", ComplexResidueScratch::layout_version},
                        {"reused_internal_stage_bytes", plan.reused_stage_capacity_bytes()},
                        {"stage_ownership", "p0_p1_after_cell_local_product_loads_public_paired_commit_unchanged"},
                        {"native_input_storage_bytes", 2 * (ac + bc) * 8},
                        {"residue_input_storage_bytes", 2 * (ac + bc) * prefix},
                        {"residue_output_bytes", 2 * cc * prefix},
                        {"input_prepare_ns", input_prepare_ns},
                        {"input_preparation_scope", "native upload, four input packs and common "
                                                    "allocations; outside resident repeated calls"},
                        {"reference_method",
                         "boost_int256_four_products_all_outputs_with_cpp_int_boundary_columns"},
                        {"component_bound", plan.real_bound.str()},
                        {"device", device.name},
                        {"target", device.gcn_arch},
                        {"runtime_version", device.hip_runtime_version},
                        {"build_manifest", nlohmann::json::parse(RNS8_BUILD_MANIFEST_JSON)},
                        {"actual_hbm_bytes", nullptr},
                        {"timing_resolution_ns", nullptr},
                        {"isa_verified", false},
                        {"promotion_eligible", false},
                        {"samples", samples},
                        {"sustained", sustained},
                        {"timing_scope",
                         "resident split inputs to both residue outputs; admission, scratch "
                         "allocation/free, transforms, all products, recombination, paired commit "
                         "and synchronization included; native upload/packing, integer export and "
                         "verification excluded"},
                        {"cache_state",
                         "same resident inputs; no transformed panel cache; outputs poisoned "
                         "outside paired calls and once before sustained burst"}}
             .dump(2)
      << '\n';
  return 0;
}
} // namespace
