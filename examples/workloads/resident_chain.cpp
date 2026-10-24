#include "workload_baselines.hpp"
#include "workload_engine.hpp"
#include "workload_report.hpp"
#include "workload_optional_references.hpp"

using namespace rns8_workload;
namespace {
struct ChainEngine {
  const Dataset& data;
  const Options& options;
  bool roundtrip;
  rns8::Context context;
  std::unique_ptr<rns8::Plan> first_plan, final_plan;
  std::unique_ptr<rns8::Workspace> first_workspace, final_workspace;
  std::unique_ptr<rns8::Matrix> a, b, transform, intermediate, imported, result;
  Words words, roundtrip_words;
  std::vector<int64_t> roundtrip_native;
  Json metadata;
  ChainEngine(const Dataset& input, const Options& config, bool export_reimport)
      : data(input), options(config), roundtrip(export_reimport), context(config.device, config.backend),
        words(std::size_t(config.m) * config.n * 3), roundtrip_words(roundtrip ? std::size_t(config.m) * config.n : 0),
        roundtrip_native(roundtrip_words.size()) {
    metadata["device"] = device_info(context, options);
    first_plan = std::make_unique<rns8::Plan>(context, gemm_desc(options, RNS8_EXACT_WIDE_SIGNED, options.m, options.n, options.k, 0, true));
    final_plan = std::make_unique<rns8::Plan>(context, gemm_desc(options, RNS8_EXACT_WIDE_SIGNED, options.m, options.n, options.n, 0, true));
    require(first_plan->backend_info().backend == config.backend && final_plan->backend_info().backend == config.backend,
            "chain plan changed explicit backend");
    const uint64_t workspace_bytes = first_plan->backend_info().workspace_required_bytes + final_plan->backend_info().workspace_required_bytes;
    const uint64_t cells = uint64_t(options.m) * options.k + uint64_t(options.k) * options.n + uint64_t(options.n) * options.n +
        uint64_t(options.m) * options.n * (roundtrip ? 3 : 2);
    const uint64_t modeled_bytes = cells * 128 + workspace_bytes;
    options.memory->reserve(modeled_bytes, options.memory_mib * 1024 * 1024, "retained chain candidate storage/workspace");
    const uint64_t global = context.device_info().global_mem_bytes;
    require(!global || modeled_bytes <= global, "chain envelope exceeds total GPU memory; free memory remains runtime checked");
    first_workspace = std::make_unique<rns8::Workspace>(context, *first_plan);
    final_workspace = std::make_unique<rns8::Workspace>(context, *final_plan);
    const auto matrix = [&](uint32_t rows, uint32_t cols) {
      return std::make_unique<rns8::Matrix>(context, matrix_desc(rows, cols, RNS8_EXACT_WIDE_SIGNED, RNS8_BOUND_NONE, 20));
    };
    a = matrix(options.m, options.k); b = matrix(options.k, options.n); transform = matrix(options.n, options.n);
    intermediate = matrix(options.m, options.n); result = matrix(options.m, options.n);
    if (roundtrip) imported = matrix(options.m, options.n);
    metadata["first_plan"] = plan_info(*first_plan); metadata["final_plan"] = plan_info(*final_plan);
    metadata["storage"] = {{"a", storage_info(*a)}, {"b", storage_info(*b)}, {"transform", storage_info(*transform)},
                            {"intermediate", storage_info(*intermediate)}, {"result", storage_info(*result)}};
    if (roundtrip) metadata["storage"]["imported"] = storage_info(*imported);
    metadata["modeled_device_storage_envelope_bytes"] = modeled_bytes;
    metadata["roundtrip_import_contract"] = roundtrip
        ? "AB proven signed64; one-limb signed export then public pack_i64(source_version0) forces actual reimport each call"
        : "AB stays in the same resident exact-wide matrix; no intermediate export/import";
  }
  void execute(Phases& phases, uint64_t version) {
    phases.measure("input_pack_and_upload_ns", [&] {
      rns8::check(rns8_pack_i64(context.get(), a->get(), data.a.data(), options.k, version));
      rns8::check(rns8_pack_i64(context.get(), b->get(), data.b.data(), options.n, 1));
      rns8::check(rns8_pack_i64(context.get(), transform->get(), data.t.data(), options.n, 1));
    });
    phases.measure("first_gemm_and_required_sync_ns", [&] {
      rns8::check(rns8_gemm_rns(context.get(), first_plan->get(), a->get(), b->get(), intermediate->get(), first_workspace->get()));
    });
    if (roundtrip) {
      phases.measure("intermediate_reconstruction_download_ns", [&] {
        rns8::check(rns8_export_exact_wide_signed_limbs(context.get(), first_plan->get(), intermediate->get(), roundtrip_words.data(), options.n, 1));
        for (std::size_t cell = 0; cell < roundtrip_words.size(); ++cell)
          std::memcpy(&roundtrip_native[cell], &roundtrip_words[cell], sizeof(int64_t));
      });
      phases.measure("intermediate_repack_upload_ns", [&] {
        rns8::check(rns8_pack_i64(context.get(), imported->get(), roundtrip_native.data(), options.n, 0));
      });
    }
    phases.measure("final_gemm_and_required_sync_ns", [&] {
      rns8::check(rns8_gemm_rns(context.get(), final_plan->get(), roundtrip ? imported->get() : intermediate->get(),
                              transform->get(), result->get(), final_workspace->get()));
    });
    phases.measure("final_reconstruction_download_ns", [&] {
      rns8::check(rns8_export_exact_wide_signed_limbs(context.get(), final_plan->get(), result->get(), words.data(), options.n, 3));
    });
  }
};
Variant chain_variant(const Dataset& data, const Options& options, bool roundtrip) {
  const auto start = Clock::now(); auto state = std::make_shared<ChainEngine>(data, options, roundtrip);
  Variant variant;
  variant.name = roundtrip ? "rns8_export_reimport_chain" : "rns8_residue_resident_chain";
  variant.setup_ns = since(start); variant.metadata = state->metadata;
  variant.execute = [state](Phases& phases, uint64_t version) { state->execute(phases, version); };
  variant.output = [state]() -> const Words& { return state->words; };
  return variant;
}
}
int main(int argc, char** argv) {
  return program(argc, argv, [](const Options& options) {
    reject_options(options, {"--wide", "--gram"});
    memory_admission(options, true);
    Dataset data = make_data(options, true);
    const cpp_int first_bound = cpp_int(options.k) * maximum_after_mutation(data) * maximum(data.b);
    const cpp_int final_bound = cpp_int(options.n) * first_bound * maximum(data.t);
    require(first_bound <= INT64_MAX,
            "public export/reimport comparison requires AB to fit signed64; reduce --bits/K (arbitrary-width import is not a public API)");
    cpp_int basis = 1; for (uint32_t p = 0; p < 20; ++p) basis *= rns8_default_modulus(p);
    require(2 * std::max(first_bound, final_bound) < basis, "chain intermediate/final envelope exceeds twenty-plane unique range");
    const auto start = Clock::now(); const Words expected = integer_words(exact_product(data, options, true), 3);
    const double reference_ns = since(start);
    Json report = base_report(options, "resident_integer_chain");
    report["task"] = "Repeated exact integer transforms Y=(AB)T: an intermediate feature matrix feeds a second transform, then the final integer result is exported.";
    report["formula"] = "U=A*B; Y=U*T";
    report["semantics"] = "exact_wide_signed";
    report["output_contract"] = "row_major_signed_three_u64_limbs_per_cell";
    report["output_limbs_per_cell"] = 3;
    report["bounds"] = {{"first_absolute_partial_sum", first_bound.str()}, {"final_absolute_partial_sum", final_bound.str()},
                          {"first_fits_signed64", true}, {"selected_prefix", 20}, {"strict_signed_uniqueness", "P20>2*max(first_bound,final_bound)"}};
    report["abs_partial_sum_bound"] = std::max(first_bound, final_bound).str();
    report["dataset_recipe"] = data.recipe; report["input_checksum_u64"] = checksum(dataset_words(data));
    report["reference_ns"] = reference_ns; report["output_preview_decimal"] = preview(expected, 3);
    report["useful_integer_products_per_call"] = uint64_t(options.m) * options.n * options.k + uint64_t(options.m) * options.n * options.n;
    report["lifecycle"] = "same AB and UT arithmetic and final three-limb host output; initial A/B/T pack/setup charged, unchanged version1 reuse, A mutation version2; no asynchronous/private import/result-cache path";
    std::vector<Variant> variants;
    Json skipped = Json::array();
    // A zero T cannot waive safety of the still-executed AB stage.
    add_integer_baselines(variants, skipped, data, options, std::max(first_bound, final_bound), 3, true);
    report["optional_reference_cohort"]=1;
    report["optional_references"]=optional_reference_identity();
    add_optional_integer_baselines(variants,skipped,data,options,3,true);
    variants.push_back(chain_variant(data, options, false));
    variants.push_back(chain_variant(data, options, true));
    run_variants(report, options, data, variants, expected, [&] {
      mutate_input(data); report["changed_input_checksum_u64"] = checksum(dataset_words(data));
      return integer_words(exact_product(data, options, true), 3);
    }, skipped);
    print_report(options, report);
  });
}
