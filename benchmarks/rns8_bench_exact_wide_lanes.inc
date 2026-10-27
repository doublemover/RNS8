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
  if(!args.reference_cache_directory.empty() || args.verify_native_output) {
    using namespace rns8::bench;using boost::multiprecision::cpp_int;
    const bool sign=args.semantics==BenchSemantics::ExactWideSigned;
    const uint32_t limbs=args.exact_wide_limb_count;const uint64_t ld=output_logical_ld(args);
    result.native_reference=verify_integer_reference(args.m,args.n,args.k,ld,a,b,c,
        {sign?ReferenceSemantics::wide_signed:ReferenceSemantics::wide_unsigned,0,limbs},
        args.reference_cache_directory,args.populate_reference_cache);
    auto& stats=result.exact_wide_magnitude;
    const cpp_int storage=cpp_int(1)<<(64*limbs);
    for(int64_t row=0;row<args.m;++row) {
      for(int64_t col=0;col<args.n;++col) {
        cpp_int value=0;for(uint32_t word=0;word<limbs;++word)value+=cpp_int(c[(row*ld+col)*limbs+word])<<(word*64);
        if(sign && (c[(row*ld+col)*limbs+limbs-1]>>63))value-=storage;
        const auto bits=magnitude_bits(value);stats.output_min_magnitude_bits=std::min(stats.output_min_magnitude_bits,bits);
        stats.output_max_magnitude_bits=std::max(stats.output_max_magnitude_bits,bits);stats.negative_outputs+=value<0;
        ++stats.checked_outputs;stats.checked_limbs+=limbs;
      }
      for(uint64_t col=args.n;col<ld && (row*ld+col+1)*limbs<=c.size();++col)
        for(uint32_t word=0;word<limbs;++word)if(c[(row*ld+col)*limbs+word]!=UINT64_C(0x5a5a5a5a5a5a5a5a))
          throw std::runtime_error("exact-wide export changed output padding");
    }
    stats.oracle_complete=true;stats.oracle_us+=elapsed_us(start,std::chrono::steady_clock::now());return;
  }
  rns8::bench::verify_exact_wide_fixture(a, b, c, args.m, args.n, args.k, output_logical_ld(args),
                                        args.exact_wide_limb_count, result.exact_wide_magnitude);
  result.exact_wide_magnitude.oracle_us += elapsed_us(start, std::chrono::steady_clock::now());
}

#include "rns8_bench_exact_wide_batch_lanes.inc"
#include "rns8_bench_exact_wide_persistent_lanes.inc"
