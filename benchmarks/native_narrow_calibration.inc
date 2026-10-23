// Included after the shared argument, exact-value and HIP helpers.
namespace {
int calibrate_narrow(const std::string& candidate, uint32_t m, uint32_t n,
                     uint32_t k, uint32_t prefix, uint32_t group,
                     bool unsigned_inputs, uint32_t repeats, uint32_t warmups,
                     const std::string& run_id) {
  if (prefix != 1 || group != 1 || k > 32768 || run_id.empty() ||
      uint64_t{m} * n * k > 16777216 ||
      (candidate == "narrow_scalar" && unsigned_inputs))
    throw std::invalid_argument("narrow cases require prefix/group=1, K<=32768; scalar is signed only");
  rns8_device_info device{};
  check(hip_direct_probe(0, device));
  std::mt19937_64 rng(3135);
  std::vector<uint64_t> a(std::size_t{m} * k), b(std::size_t{k} * n);
  for (auto* operand : {&a, &b}) {
    for (auto& x : *operand) {
      const int64_t value = int64_t(rng() & 255) - (unsigned_inputs ? 0 : 128);
      x = static_cast<uint64_t>(value);
    }
    operand->front() = unsigned_inputs ? 255 : static_cast<uint64_t>(-128);
  }
  const uint64_t bound = uint64_t{k} * (unsigned_inputs ? 255 * 255 : 128 * 128);
  // Native output and eight uint64 canaries; no residue storage or CRT here.
  const uint64_t poison = UINT64_C(0x6363636363636363);
  std::vector<uint64_t> expected(std::size_t{m} * n + 8, poison);
  for (uint32_t row = 0; row < m; ++row)
    for (uint32_t col = 0; col < n; ++col) {
      cpp_int sum = 0;
      for (uint32_t q = 0; q < k; ++q)
        sum += native_value(a[row * k + q], unsigned_inputs) *
               native_value(b[q * n + col], unsigned_inputs);
      if (sum < -cpp_int(bound) || sum > cpp_int(bound))
        throw std::runtime_error("dataset bound construction failed");
      expected[std::size_t{row} * n + col] = unsigned_inputs
          ? sum.convert_to<uint64_t>() : static_cast<uint64_t>(sum.convert_to<int64_t>());
    }
  hip_direct_device_buffer da, db, dc, status;
  check(da.allocate(0, a.size() * 8)); check(db.allocate(0, b.size() * 8));
  check(dc.allocate(0, expected.size() * 8)); check(status.allocate(0, sizeof(uint32_t)));
  check(hip_direct_copy_host_to_device(0, da.get(), a.data(), a.size() * 8));
  check(hip_direct_copy_host_to_device(0, db.get(), b.data(), b.size() * 8));
  const std::vector<uint64_t> poisoned(expected.size(), poison);
  std::vector<uint64_t> observed(expected.size());
  const auto prepare = [&] {
    check(hip_direct_copy_host_to_device(0, dc.get(), poisoned.data(), poisoned.size() * 8));
  };
  const auto execute = [&] {
    uint64_t observed_bound = 0;
    if (candidate == "narrow_vector") {
      // Status reset and readback belong to this candidate's complete checked call.
      check(hip_direct_zero(0, status.get(), sizeof(uint32_t)));
      check(unsigned_inputs ? vector_alu_gemm_u64_device(0, da.get(), db.get(), dc.get(), status.get(), m, n, k)
                            : vector_alu_gemm_i64_device(0, da.get(), db.get(), dc.get(), status.get(), m, n, k));
      uint32_t failures = 0;
      check(hip_direct_copy_device_to_host(0, &failures, status.get(), sizeof(failures)));
      if (failures) throw std::runtime_error("vector native overflow status");
    } else {
      check(candidate == "narrow_wmma"
        ? hip_direct_gemm_native_byte_wmma_checked_device(0, da.get(), db.get(), dc.get(), m, n, k, k, n, n, unsigned_inputs, &observed_bound)
        : hip_direct_gemm_i64_narrow_checked_device(0, static_cast<const int64_t*>(da.get()), static_cast<const int64_t*>(db.get()), static_cast<int64_t*>(dc.get()), m, n, k, k, n, n, &observed_bound));
      if (observed_bound != bound) throw std::runtime_error("device scan bound mismatch");
    }
  };
  const auto verify = [&] {
    check(hip_direct_copy_device_to_host(0, observed.data(), dc.get(), observed.size() * 8));
    if (observed != expected) throw std::runtime_error("narrow native reference/canary mismatch");
  };
  for (uint32_t i = 0; i < warmups; ++i) { prepare(); execute(); verify(); }
  nlohmann::json samples = nlohmann::json::array();
  hip_direct_timing_set_enabled(true);
  for (uint32_t i = 0; i < repeats; ++i) {
    prepare(); hip_direct_timing_reset();
    const auto start = std::chrono::steady_clock::now();
    execute();
    const double host_ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start).count();
    hip_direct_timing_flush_pending_events();
    nlohmann::json events = nlohmann::json::array();
    for (const auto& event : hip_direct_timing_snapshot())
      events.push_back({{"label", event.label}, {"elapsed_ns", event.microseconds > 0
          ? nlohmann::json(event.microseconds * 1000) : nlohmann::json(nullptr)}});
    verify();
    samples.push_back({{"host_call_ns", host_ns > 0 ? nlohmann::json(host_ns) : nlohmann::json(nullptr)},
                       {"events", events}, {"exact_reference_passed", true}});
  }
  hip_direct_timing_set_enabled(false);
  std::cout << nlohmann::json{
      {"schema", "rns8_native_microcase_v1"}, {"run_id", run_id}, {"candidate", candidate},
      {"m", m}, {"n", n}, {"k", k}, {"prefix", prefix}, {"group", group},
      {"unsigned_inputs", unsigned_inputs}, {"seed", 3135}, {"repeats", repeats}, {"warmups", warmups},
      {"dataset", "byte_domain_mt19937_64_seed3135_forced_maxima_v1"},
      {"input_domain", "resident_native_64_byte_values"}, {"output_contract", "bounded_exact_native_64"},
      {"abs_partial_sum_bound", bound}, {"device", device.name}, {"target", device.gcn_arch},
      {"runtime_version", device.hip_runtime_version}, {"build_manifest", nlohmann::json::parse(RNS8_BUILD_MANIFEST_JSON)},
      {"native_input_storage_bytes", (a.size() + b.size()) * 8}, {"native_output_bytes", std::size_t{m} * n * 8},
      {"reserved_status_bytes", sizeof(uint32_t)}, {"actual_hbm_bytes", nullptr}, {"timing_resolution_ns", nullptr},
      {"isa_verified", false}, {"promotion_eligible", false}, {"samples", samples},
      {"timing_scope", "resident native inputs; complete checked host call includes device range scans or vector status reset/readback, allocations and synchronization; upload, output poisoning and Boost verification excluded"},
      {"cache_state", "repeated inputs; destination reinitialized outside every timed call"}}.dump(2) << '\n';
  return 0;
}
} // namespace
