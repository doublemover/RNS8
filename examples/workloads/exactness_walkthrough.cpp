#include "workload_baselines.hpp"
#include "workload_engine.hpp"
#include <cfenv>

using namespace rns8_workload;
namespace {
struct RoundingGuard {
  int previous = std::fegetround();
  RoundingGuard() { require(previous != -1 && std::fesetround(FE_TONEAREST) == 0, "cannot establish FE_TONEAREST"); }
  ~RoundingGuard() { std::fesetround(previous); }
};
Json witness(const Options& common, const char* name, const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
  Options options = common; options.m = options.n = 1; options.k = static_cast<uint32_t>(a.size()); options.wide = true;
  Dataset data; data.a = a; data.b = b; data.recipe = name;
  const auto reference_start = Clock::now();
  const Words expected = integer_words(exact_product(data, options), 3);
  const double reference_ns = since(reference_start);
  const cpp_int exact = decode(expected, 0, 3);
  const auto setup = Clock::now(); DenseEngine candidate(data, options, 0); const double setup_ns = since(setup);
  Phases phases; const auto call = Clock::now(); candidate.execute(phases, 1); const double call_ns = since(call);
  const auto verification_start = Clock::now(); require(candidate.words == expected, "walkthrough exact result mismatch");
  const double verification_ns = since(verification_start);
  Json limbs = Json::array(); for (uint64_t word : candidate.words) limbs.push_back("0x" + hex64(word));
  int64_t checked = 0; bool fits = true;
  uint64_t wrapped = 0;
  for (std::size_t q = 0; q < a.size(); ++q) {
    int64_t product = 0, next = 0;
    if (fits && (!checked_product(a[q], b[q], product) || !checked_sum(checked, product, next))) fits = false;
    if (fits) checked = next;
    wrapped += static_cast<uint64_t>(a[q]) * static_cast<uint64_t>(b[q]);
  }
  // Volatile storage forces actual binary64 arithmetic rather than a constant
  // integer expression folded by the compiler. The rounding mode is recorded.
  volatile double floating = 0;
  for (std::size_t q = 0; q < a.size(); ++q) {
    volatile double left = static_cast<double>(a[q]), right = static_cast<double>(b[q]);
    volatile double product = left * right; floating = floating + product;
  }
  const double observed = floating;
  const cpp_int floating_integer(observed);
  Json row = {{"name", name}, {"exact_decimal", exact.str()}, {"little_endian_u64_limbs", limbs},
              {"checksum_hex", hex64(checksum(expected))}, {"exact_reference_passed", true},
              {"checked_i64", {{"status", fits ? "exact" : "insufficient_range"}, {"value", fits ? Json(checked) : Json(nullptr)},
                               {"rule", "checked product and ordered partial sums; no signed overflow executed"}}},
              {"wrap64_low_bits_hex", hex64(wrapped)},
              {"wrap64_contract", "unsigned low64 bits only; not a signed exact-integer contract"},
              {"binary64", {{"integer_value", floating_integer.str()}, {"integer_difference_from_exact", cpp_int(floating_integer - exact).str()},
                            {"rounding_mode", std::fegetround()}, {"method", "runtime binary64 conversion, multiply and ordered addition; exact integer value decoded from resulting double"}}},
              {"reference_ns", reference_ns}, {"setup_ns", setup_ns}, {"application_ns", call_ns},
              {"verification_ns", verification_ns}, {"phases_ns", phases.values}, {"candidate", candidate.metadata}};
#if defined(__SIZEOF_INT128__)
  Native128 sum = 0;
  for (std::size_t q = 0; q < a.size(); ++q) sum += Native128(a[q]) * Native128(b[q]);
  std::vector<Native128> values{sum}; Words native_words(3); encode_into(values, 3, native_words);
  require(native_words == expected, "genuine128 walkthrough differs");
  row["native128"] = {{"status", "exact"}, {"type", "compiler __int128"}};
#else
  row["native128"] = {{"status", "unavailable"}, {"reason", "this compiler has no genuine native signed128; Boost fixed-width types are not native128 baselines"}};
#endif
  // Fixed export width is independent of the selected residue information range.
  uint64_t too_narrow = UINT64_C(0x9c63b128452a7e91);
  const auto status = rns8_export_exact_wide_signed_limbs(candidate.context.get(), candidate.plan->get(), candidate.c->get(), &too_narrow, 1, 1);
  const bool final_fits_i64 = exact >= INT64_MIN && exact <= INT64_MAX;
  require(status == (final_fits_i64 ? RNS8_SUCCESS : RNS8_RANGE_ERROR), "one-limb export has wrong range status");
  if (!final_fits_i64) require(too_narrow == UINT64_C(0x9c63b128452a7e91), "failed narrow export changed destination");
  row["one_limb_export_status"] = rns8_status_string(status);
  row["selected_prefix"] = candidate.prefix;
  return row;
}
Json wider_than_128(const Options& options) {
  rns8::Context context(options.device, options.backend);
  const auto identity = device_info(context, options);
  const auto descriptor = gemm_desc(options, RNS8_EXACT_WIDE_SIGNED, 1, 1, 1, 0, true);
  rns8::Plan plan(context, descriptor); rns8::Workspace workspace(context, plan);
  const auto md = matrix_desc(1, 1, RNS8_EXACT_WIDE_SIGNED, RNS8_BOUND_NONE, 20);
  rns8::Matrix left(context, md), right(context, md), product(context, md), scale(context, md), result(context, md);
  const int64_t minimum = INT64_MIN, eight = 8, excessive = INT64_C(1) << 30;
  rns8::check(rns8_pack_i64(context.get(), left.get(), &minimum, 1, 1));
  rns8::check(rns8_pack_i64(context.get(), right.get(), &minimum, 1, 1));
  rns8::check(rns8_pack_i64(context.get(), scale.get(), &eight, 1, 1));
  rns8::check(rns8_gemm_rns(context.get(), plan.get(), left.get(), right.get(), product.get(), workspace.get()));
  rns8::check(rns8_gemm_rns(context.get(), plan.get(), product.get(), scale.get(), result.get(), workspace.get()));
  Words words(3);
  rns8::check(rns8_export_exact_wide_signed_limbs(context.get(), plan.get(), result.get(), words.data(), 1, 3));
  const cpp_int expected = cpp_int(1) << 129;
  require(decode(words, 0, 3) == expected, "above128 example differs from exact power of two");
  rns8::check(rns8_pack_i64(context.get(), scale.get(), &excessive, 1, 2));
  const auto rejected = rns8_gemm_rns(context.get(), plan.get(), product.get(), scale.get(), result.get(), workspace.get());
  require(rejected == RNS8_RANGE_ERROR, "out-of-basis 2^156 product was not cleanly rejected");
  Words preserved(3);
  rns8::check(rns8_export_exact_wide_signed_limbs(context.get(), plan.get(), result.get(), preserved.data(), 1, 3));
  require(preserved == words, "preflight range rejection changed previous output");
  return {{"name", "above128_resident_power"}, {"formula", "(INT64_MIN*INT64_MIN)*8=2^129"},
          {"exact_decimal", expected.str()}, {"selected_prefix", 20}, {"range_bits", rns8_prefix_range_bits(20)},
          {"checksum_hex", hex64(checksum(words))}, {"exact_reference_passed", true},
          {"native128", {{"status", "insufficient_range"}, {"reason", "2^129 exceeds even unsigned128; no native128 arithmetic attempted"}}},
          {"out_of_range_formula", "(INT64_MIN*INT64_MIN)*2^30=2^156"},
          {"out_of_range_status", rns8_status_string(rejected)}, {"previous_output_preserved", true}, {"device", identity}};
}
}
int main(int argc, char** argv) {
  return program(argc, argv, [](const Options& common) {
    reject_options(common, {"--m", "--n", "--k", "--bits", "--seed", "--memory-mib", "--gram", "--wide", "--pattern", "--baseline-max-products", "--sustained-ms"});
    require(common.size != "large", "exactness walkthrough is deliberately tiny; --size large is irrelevant");
    require((!common.supplied.count("--repeats") || common.repeats == 1)
        && (!common.supplied.count("--warmups") || common.warmups == 0),
        "walkthrough uses one observation per fixed witness; --repeats must be1 and --warmups0");
    RoundingGuard rounding;
    Options effective = common; effective.m = effective.n = 1; effective.k = 2;
    effective.repeats = 1; effective.warmups = 0;
    Json report = base_report(effective, "exactness_walkthrough");
    report["task"] = "An integer above signed64: 2^70-15=1180591620717411303409. Exact arithmetic preserves the15 that binary64 rounds away in this deliberately chosen witness.";
    report["semantics"] = "exact_wide_signed";
    report["output_contract"] = "signed decimal and three little-endian u64 limbs; educational negative controls are not performance baselines";
    report["witnesses"] = Json::array();
    report["witnesses"].push_back(witness(common, "above_i64_v1", {INT64_C(1) << 40, -3}, {INT64_C(1) << 30, 5}));
    report["witnesses"].push_back(witness(common, "signed_cancellation_v1", {INT64_C(1) << 40, -(INT64_C(1) << 40)}, {INT64_C(1) << 30, INT64_C(1) << 30}));
    report["witnesses"].push_back(witness(common, "zero_v1", {0, 0}, {INT64_MIN, INT64_MAX}));
    report["witnesses"].push_back(witness(common, "full_width_v1", {INT64_MIN, INT64_MAX}, {1, 1}));
    report["witnesses"].push_back(wider_than_128(common));
    Words aggregate;
    for (const auto& row : report["witnesses"]) {
      const auto encoded = integer_words({cpp_int(row["exact_decimal"].get<std::string>())}, 3);
      aggregate.insert(aggregate.end(), encoded.begin(), encoded.end());
    }
    report["checksum_u64"] = checksum(aggregate); report["checksum_hex"] = hex64(checksum(aggregate));
    report["dataset_recipe"] = "exactness_five_witnesses_v1";
    report["checksum_contract"] = "fnv1a64_bytes_v1 over all five signed192 witness outputs in report order";
    report["reference_method"] = "Boost cpp_int and known powers-of-two; all exact outputs compared, export-range rejection preserves destination/value";
    report["timing_note"] = "tiny walkthrough observations only; no speedup estimate; negative controls are not timed baselines";
    print_report(common, report);
  });
}
