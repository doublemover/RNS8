#include "workload_complex.hpp"
#include "workload_report.hpp"

using namespace rns8_workload;
namespace {
struct ComplexEngine {
  const ComplexData& data;
  const Options& options;
  rns8_complex_product_method method;
  uint64_t scratch_cap;
  rns8::Context context;
  std::unique_ptr<rns8::Plan> plan;
  std::unique_ptr<rns8::Workspace> workspace;
  std::array<std::unique_ptr<rns8::Matrix>, 6> matrices;
  Words words;
  Json metadata;
  ComplexEngine(const ComplexData& input, const Options& config, rns8_complex_product_method selected,
                uint64_t scratch_bytes)
      : data(input), options(config), method(selected), scratch_cap(scratch_bytes), context(config.device, config.backend),
        words(std::size_t(config.m) * config.n * 6) {
    metadata["device"] = device_info(context, options);
    plan = std::make_unique<rns8::Plan>(context,
        gemm_desc(options, RNS8_EXACT_WIDE_SIGNED, options.m, options.n, options.k, 0, true));
    require(plan->backend_info().backend == config.backend, "complex plan changed explicit backend");
    const uint64_t cells = 2 * (uint64_t(options.m) * options.k + uint64_t(options.k) * options.n +
                               uint64_t(options.m) * options.n);
    const uint64_t retained = cells * 128 + plan->backend_info().workspace_required_bytes;
    options.memory->reserve(retained, options.memory_mib * 1024 * 1024, "retained split-complex matrix/workspace state");
    const uint64_t global = context.device_info().global_mem_bytes;
    require(!global || options.memory->reserved <= global,
            "aggregate complex envelope exceeds total GPU memory; free memory remains runtime checked");
    workspace = std::make_unique<rns8::Workspace>(context, *plan);
    for (std::size_t index = 0; index < matrices.size(); ++index) {
      const uint32_t rows = index < 2 || index >= 4 ? options.m : options.k;
      const uint32_t cols = index < 2 ? options.k : options.n;
      matrices[index] = std::make_unique<rns8::Matrix>(context,
          matrix_desc(rows, cols, RNS8_EXACT_WIDE_SIGNED, RNS8_BOUND_NONE, 20));
    }
    metadata["plan"] = plan_info(*plan);
    metadata["public_api"] = "rns8_gemm_complex_exact_signed";
    metadata["method"] = method == RNS8_COMPLEX_THREE_REAL_PRODUCTS ? "3m" : "4m";
    metadata["real_product_count"] = method == RNS8_COMPLEX_THREE_REAL_PRODUCTS ? 3 : 4;
    metadata["scratch_budget_bytes"] = scratch_cap;
    metadata["retained_storage_envelope_bytes"] = retained;
    metadata["scratch_scope"] = "public call transient payload cap; allocation/release and synchronization charged on every call; no transformed input cache";
    metadata["storage"] = Json::array();
    for (const auto& matrix : matrices) metadata["storage"].push_back(storage_info(*matrix));
  }
  void execute(Phases& phases, uint64_t version) {
    phases.measure("four_input_pack_and_upload_ns", [&] {
      for (std::size_t index = 0; index < data.values.size(); ++index)
        rns8::check(rns8_pack_i64(context.get(), matrices[index]->get(), data.values[index].data(),
                                index < 2 ? options.k : options.n, index == 0 ? version : 1));
    });
    phases.measure("complex_composition_scratch_allocation_compute_sync_release_ns", [&] {
      rns8::check(rns8_gemm_complex_exact_signed(context.get(), plan->get(),
          matrices[0]->get(), matrices[1]->get(), matrices[2]->get(), matrices[3]->get(),
          matrices[4]->get(), matrices[5]->get(), method, scratch_cap, workspace->get()));
    });
    phases.measure("both_components_reconstruction_download_ns", [&] {
      const auto component_words = std::size_t(options.m) * options.n * 3;
      rns8::check(rns8_export_exact_wide_signed_limbs(context.get(), plan->get(), matrices[4]->get(),
                                                   words.data(), options.n, 3));
      rns8::check(rns8_export_exact_wide_signed_limbs(context.get(), plan->get(), matrices[5]->get(),
                                                   words.data() + component_words, options.n, 3));
    });
  }
};
Variant complex_variant(const ComplexData& data, const Options& options,
                        rns8_complex_product_method method, uint64_t scratch) {
  const auto start = Clock::now(); auto state = std::make_shared<ComplexEngine>(data, options, method, scratch);
  Variant result;
  result.name = method == RNS8_COMPLEX_THREE_REAL_PRODUCTS ? "rns8_complex_3m" : "rns8_complex_4m";
  result.setup_ns = since(start); result.metadata = state->metadata;
  result.execute = [state](Phases& phases, uint64_t version) { state->execute(phases, version); };
  result.output = [state]() -> const Words& { return state->words; };
  return result;
}
}
int main(int argc, char** argv) {
  return program(argc, argv, [](const Options& options) {
    reject_options(options, {"--wide", "--gram"});
    if (options.backend != RNS8_BACKEND_CPU_REFERENCE && options.backend != RNS8_BACKEND_HIP_DIRECT)
      throw rns8::Error(RNS8_UNSUPPORTED_BACKEND);
    const uint64_t mk = uint64_t(options.m) * options.k, kn = uint64_t(options.k) * options.n;
    const uint64_t mn = uint64_t(options.m) * options.n, budget = options.memory_mib * 1024 * 1024;
    // All four inputs, both reference components and simultaneously retained
    // baseline states. Below 2^138 at allowed shapes/bits, cpp_int needs at most
    // three significant64 limbs;1024 bytes/cell leaves capacity/serialization
    // room for naive+blocked+native comparators and old/new exact references.
    options.memory->reserve(2 * (mk + kn + mn) * 1024, budget, "aggregate complex host comparators/references");
    // Only one variant executes at a time. Reserve this shared transient cap
    // once, then charge both candidates' retained matrices/workspaces separately.
    // It conservatively exceeds20*(MK+KN+5MN) for3M and20*6MN for4M.
    const uint64_t scratch_cap = 128 * (mk + kn + 6 * mn);
    options.memory->reserve(scratch_cap, budget, "shared complex transient scratch cap");
    require(options.size == "large" || 4 * uint64_t(options.m) * options.n * options.k <= 4000000,
            "larger four-product reference work requires explicit --size large");
    ComplexData data = make_complex_data(options);
    const cpp_int bound = complex_final_bound(data, options);
    cpp_int basis = 1; for (uint32_t p = 0; p < 20; ++p) basis *= rns8_default_modulus(p);
    require(2 * bound < basis, "complex final component envelope exceeds twenty-plane signed uniqueness");
    const auto reference_start = Clock::now(); const Words expected = complex_reference(data, options);
    const double reference_ns = since(reference_start);
    Json report = base_report(options, "split_complex_integer_product");
    report["task"] = "Exact split-complex integer transform with explicit3M or4M composition and identical fully exported real/imaginary outputs.";
    report["formula"] = "Cr=Ar*Br-Ai*Bi; Ci=Ar*Bi+Ai*Br (no conjugation)";
    report["semantics"] = "exact_wide_signed_split_complex";
    report["output_contract"] = "real_row_major_signed_three_u64_limbs_then_imag_same";
    report["output_limbs_per_component_cell"] = 3;
    report["dataset_recipe"] = data.recipe; report["input_checksum_u64"] = checksum(complex_input_words(data));
    report["bounds"] = {{"component_absolute_partial_sum", bound.str()}, {"selected_prefix", 20},
                         {"strict_signed_uniqueness", "P20>2*max(real_bound,imag_bound), including generation2"}};
    report["abs_partial_sum_bound"] = bound.str(); report["reference_ns"] = reference_ns;
    report["output_preview_decimal"] = {{"real", preview(Words(expected.begin(), expected.begin() + mn * 3), 3)},
                                        {"imag", preview(Words(expected.begin() + mn * 3, expected.end()), 3)}};
    report["useful_integer_products_per_call"] = 4 * uint64_t(options.m) * options.n * options.k;
    report["work_definition"] = "four-product mathematical reference, irrespective of selected3M/4M implementation";
    report["lifecycle"] = "four native signed inputs packed each call, same source version1 reuse, Ar mutation version2; scratch allocation/composition/synchronization and both signed192 exports included; no asynchronous API or transformed cache";
    std::vector<Variant> variants; Json skipped = Json::array();
    add_complex_baselines(variants, skipped, data, options, bound);
    variants.push_back(complex_variant(data, options, RNS8_COMPLEX_THREE_REAL_PRODUCTS, scratch_cap));
    variants.push_back(complex_variant(data, options, RNS8_COMPLEX_FOUR_REAL_PRODUCTS, scratch_cap));
    run_variants(report, options, data, variants, expected, [&] {
      mutate_complex_input(data); report["changed_input_checksum_u64"] = checksum(complex_input_words(data));
      return complex_reference(data, options);
    }, skipped);
    report["checksum_contract"] = "fnv1a64_bytes_v1; all real row-major signed192 cells then all imaginary cells; least-significant u64 limb then little-endian bytes";
    print_report(options, report);
  });
}
