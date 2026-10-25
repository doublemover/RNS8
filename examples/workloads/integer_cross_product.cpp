#include "workload_baselines.hpp"
#include "workload_engine.hpp"
#include "workload_report.hpp"
#include "workload_optional_references.hpp"

using namespace rns8_workload;
int main(int argc, char** argv) {
  return program(argc, argv, [](const Options& options) {
    if(options.gram_reuse && options.backend!=RNS8_BACKEND_CPU_REFERENCE && options.backend!=RNS8_BACKEND_HIP_DIRECT)
      throw rns8::Error(RNS8_UNSUPPORTED_BACKEND);
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
    report["optional_reference_cohort"]=1;
    report["optional_references"]=optional_reference_identity();
    add_optional_integer_baselines(variants,skipped,data,options,limbs);
    report["estimated_host_comparator_envelope_bytes"]=options.memory->reserved;
    if(options.gram_reuse) {
      report["gram_batch"]=gram_batch_contract(options);
      report["gram_prepared_owner_cohort"]=1;report["gram_ablation_cohort"]=1;
      report["bound_method"]="K * max_abs_X_including_changed_X0 squared; no cancellation assumption";
      report["useful_integer_products_per_call"]=uint64_t(options.m)*options.n*options.k*options.gram_reuse;
      report["work_definition"]="R complete identical Gram products per application batch; dense equivalent mathematical products, counted once each";
      report["lifecycle"]="packed inputs and public dense preparation retained per generation; R complete products per batch; explicit each/once reconstruction; generation2 repacks and refreshes prepared dense inputs";
      for(auto& variant:variants) {
        const auto execute=variant.execute;const uint32_t reuse=options.gram_reuse;
        variant.execute=[execute,reuse](Phases& phases,uint64_t version) {for(uint32_t product=0;product<reuse;++product) execute(phases,version);};
        variant.metadata["gram_batch"]=gram_batch_contract(options);
        variant.metadata["gram_batch"]["implementation"]="cpu_native_comparator";
        variant.metadata["gram_batch"]["public_api"]=nullptr;
        variant.metadata["gram_batch"]["rns_exports_per_batch"]=0;
        variant.metadata["gram_batch"]["native_serializations_per_batch"]=reuse;
        variant.metadata["gram_batch"]["prepared_gram_api"]=false;
        variant.metadata["gram_batch"]["job_list_allocated_per_call"]=false;
        variant.metadata["gram_batch"]["capability_probed_per_call"]=false;
      }
    }
    variants.push_back(dense_variant(data, options, bound));
    if (options.gram && (options.backend==RNS8_BACKEND_CPU_REFERENCE || options.backend==RNS8_BACKEND_HIP_DIRECT)) {
      variants.push_back(gram_variant(data,options,bound,RNS8_GRAM_SCALAR));
      if(options.gram_reuse) {
            variants.push_back(gram_variant(data,options,bound,RNS8_GRAM_SCALAR,true));
            variants.push_back(gram_variant(data,options,bound,RNS8_GRAM_SCALAR,true,RNS8_GRAM_RETAINED_NATIVE_I64));
            variants.push_back(gram_variant(data,options,bound,RNS8_GRAM_SCALAR,true,RNS8_GRAM_RETAINED_TRANSPOSE));
            variants.push_back(gram_variant(data,options,bound,RNS8_GRAM_SCALAR,true,RNS8_GRAM_RESIDENT_RESIDUES,true));
            variants.push_back(gram_variant(data,options,bound,RNS8_GRAM_SCALAR_SEPARATE_MIRROR,true));
          }
      if (options.backend==RNS8_BACKEND_HIP_DIRECT) {
        rns8::Context probe(options.device,options.backend);
        const std::string target=probe.device_info().gcn_arch;
        if (target.rfind("gfx1100",0)==0 || target.rfind("gfx1101",0)==0 || target.rfind("gfx1102",0)==0) {
          variants.push_back(gram_variant(data,options,bound,RNS8_GRAM_WMMA));
          if(options.gram_reuse) {
            variants.push_back(gram_variant(data,options,bound,RNS8_GRAM_WMMA,true));
            variants.push_back(gram_variant(data,options,bound,RNS8_GRAM_WMMA,true,RNS8_GRAM_RETAINED_NATIVE_I64));
            variants.push_back(gram_variant(data,options,bound,RNS8_GRAM_WMMA,true,RNS8_GRAM_RETAINED_TRANSPOSE));
            variants.push_back(gram_variant(data,options,bound,RNS8_GRAM_WMMA,true,RNS8_GRAM_RESIDENT_RESIDUES,true));
            variants.push_back(gram_variant(data,options,bound,RNS8_GRAM_WMMA_SEPARATE_MIRROR,true));
          }
        }
        else {
          skipped.push_back({{"name","rns8_single_source_gram_wmma"},{"status","unsupported"},{"reason","requires RDNA3 IU8 target"}});
          if(options.gram_reuse) skipped.push_back({{"name","rns8_prepared_single_source_gram_wmma"},{"status","unsupported"},{"reason","requires RDNA3 IU8 target"}});
        }
      }
      report["gram_independent_integer_products_per_call"]=uint64_t(options.n)*(options.n+1)/2*options.k*std::max(options.gram_reuse,1u);
      report["gram_dense_equivalent_integer_products_per_call"]=uint64_t(options.n)*options.n*options.k*std::max(options.gram_reuse,1u);
    }
    run_variants(report, options, data, variants, expected, [&] {
      mutate_input(data);
      if (options.gram) data.b[0] = data.a[0];
      report["changed_input_checksum_u64"] = checksum(dataset_words(data));
      return integer_words(exact_product(data, options), limbs);
    }, skipped);
    print_report(options, report);
  });
}
