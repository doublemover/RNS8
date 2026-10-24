template<class T>
void fill_exact_wide_benchmark_input(const Args& args, std::vector<T>& values, std::mt19937_64& rng) {
  unsigned exponent = 0;
  bool mixed = false;
  switch (args.input_profile) {
    case InputProfile::WideTwoLimbConstant: exponent = 40; break;
    case InputProfile::WideThreeLimbConstant: exponent = 62; break;
    case InputProfile::WideTwoLimbMixed: exponent = 40; mixed = true; break;
    case InputProfile::WideThreeLimbMixed: exponent = 62; mixed = true; break;
    default: break;
  }
  rns8::bench::fill_exact_wide_magnitude(values, rng, exponent, mixed);
}

template<class T>
void admit_exact_wide_benchmark_input(const Args& args, const std::vector<T>& a, const std::vector<T>& b,
                                      uint32_t prefix, BenchmarkResult& result) {
  if (args.residue_chain_length != 1) return; // chained magnitude is a separate contract
  const auto start = std::chrono::steady_clock::now();
  std::vector<uint16_t> moduli;
  for (uint32_t p = 0; p < prefix; ++p) moduli.push_back(rns8_default_modulus(p));
  rns8::bench::admit_exact_wide_fixture(a, b, args.k, moduli, args.exact_wide_limb_count, result.exact_wide_magnitude);
  result.exact_wide_magnitude.admission_us += elapsed_us(start, std::chrono::steady_clock::now());
}

template<class T>
void verify_exact_wide_benchmark_output(const Args& args, const std::vector<T>& a, const std::vector<T>& b,
                                        const std::vector<uint64_t>& c, BenchmarkResult& result) {
  if (args.residue_chain_length != 1) return;
  const auto start = std::chrono::steady_clock::now();
  rns8::bench::verify_exact_wide_fixture(a, b, c, args.m, args.n, args.k, output_logical_ld(args),
                                        args.exact_wide_limb_count, result.exact_wide_magnitude);
  result.exact_wide_magnitude.oracle_us += elapsed_us(start, std::chrono::steady_clock::now());
}

#include "rns8_bench_exact_wide_batch_lanes.inc"
#include "rns8_bench_exact_wide_persistent_lanes.inc"
