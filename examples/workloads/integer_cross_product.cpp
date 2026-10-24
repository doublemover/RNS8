#include "workload_baselines.hpp"
#include "workload_engine.hpp"
#include "workload_report.hpp"

using namespace rns8_workload;
int main(int argc, char** argv) {
  return program(argc, argv, [](const Options& options) {
    memory_admission(options);
    Dataset data = make_data(options);
    // This envelope bounds every partial, including after the explicit mutation.
    const cpp_int bound = cpp_int(options.k) * maximum_after_mutation(data) * (options.gram ? maximum_after_change(data.b) : maximum(data.b));
    if (!options.wide) require(bound <= INT64_MAX, "bounded output requires K*amax*bmax<=INT64_MAX; use --wide for a larger exact result");
    cpp_int basis = 1;
    for (uint32_t p = 0; p < RNS8_MAX_SUPPORTED_PREFIX; ++p) basis *= rns8_default_modulus(p);
    require(basis > 2 * bound, "twenty-plane represented range is insufficient");
    const uint32_t limbs = options.wide ? 3 : 1;
    const auto reference_start = Clock::now();
    const Words expected = integer_words(exact_product(data, options), limbs);
    const double reference_ns = since(reference_start);
    Json report = base_report(options, options.gram ? "integer_gram" : "integer_cross_product");
    report["task"] = options.gram
        ? "Integer Gram matrix X^T X: K observations, M=N integer features; output units are products of feature units."
        : "Integer cross-product AB: M observations of K integer features times K-by-N integer coefficients; exact transformed integer observations.";
    report["formula"] = options.gram ? "C=X^T*X" : "C=A*B";
    report["semantics"] = options.wide ? "exact_wide_signed" : "bounded_i64";
    report["output_contract"] = options.wide ? "row_major_signed_three_u64_limbs_per_cell" : "row_major_signed_i64";
    report["output_limbs_per_cell"] = limbs;
    report["abs_partial_sum_bound"] = bound.str();
    report["bound_method"] = "K * max_abs_A_including_mutated_A0 * max_abs_B; no final-cancellation assumption";
    report["dataset_recipe"] = data.recipe;
    report["input_checksum_u64"] = checksum(dataset_words(data));
    report["reference_ns"] = reference_ns;
    report["output_preview_decimal"] = preview(expected, limbs);
    report["useful_integer_products_per_call"] = uint64_t(options.m) * options.n * options.k;
    report["work_definition"] = "M*N*K mathematical integer products, counted once; residue planes are implementation work";
    report["estimated_host_comparator_envelope_bytes"] = options.memory->reserved;
    report["lifecycle"] = "common prepared row-major inputs; RNS8 pack called with stable version1 for unchanged reuse, fresh A version2 after actual mutation; every call recomputes GEMM and exports full output";
    std::vector<Variant> variants;
    Json skipped = Json::array();
    add_integer_baselines(variants, skipped, data, options, bound, limbs);
    variants.push_back(dense_variant(data, options, bound));
    run_variants(report, options, data, variants, expected, [&] {
      mutate_input(data);
      if (options.gram) data.b[0] = data.a[0];
      report["changed_input_checksum_u64"] = checksum(dataset_words(data));
      return integer_words(exact_product(data, options), limbs);
    }, skipped);
    print_report(options, report);
  });
}
