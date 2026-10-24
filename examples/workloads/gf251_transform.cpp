#include "workload_baselines.hpp"
#include "workload_report.hpp"

using namespace rns8_workload;
namespace {
struct FieldBaseline {
  const std::vector<uint8_t>& a;
  const std::vector<uint8_t>& b;
  const Options& options;
  bool blocked;
  std::vector<uint32_t> partial;
  Words words;
  FieldBaseline(const std::vector<uint8_t>& lhs, const std::vector<uint8_t>& rhs, const Options& config, bool tiled)
      : a(lhs), b(rhs), options(config), blocked(tiled), partial(std::size_t(config.m) * config.n), words(partial.size()) {}
  void execute(Phases& phases, uint64_t) {
    phases.measure("modular_product_ns", [&] {
      if (!blocked) {
        std::fill(words.begin(), words.end(), 0);
        for (uint32_t row = 0; row < options.m; ++row) for (uint32_t col = 0; col < options.n; ++col)
          for (uint32_t q = 0; q < options.k; ++q) {
            auto& value = words[std::size_t(row) * options.n + col];
            value = (value + uint32_t(a[std::size_t(row) * options.k + q]) * b[std::size_t(q) * options.n + col]) % 251;
          }
        return;
      }
      std::fill(partial.begin(), partial.end(), 0);
      // Every unsigned partial <=250+256*250^2=16,000,250<2^32.
      for (uint32_t slab = 0; slab < options.k; slab += 256) {
        for (uint32_t ii = 0; ii < options.m; ii += 16) for (uint32_t jj = 0; jj < options.n; jj += 32)
          for (uint32_t row = ii; row < std::min(options.m, ii + 16); ++row)
            for (uint32_t q = slab; q < std::min(options.k, slab + 256); ++q) {
              const uint32_t av = a[std::size_t(row) * options.k + q];
              for (uint32_t col = jj; col < std::min(options.n, jj + 32); ++col)
                partial[std::size_t(row) * options.n + col] += av * b[std::size_t(q) * options.n + col];
            }
        for (auto& value : partial) value %= 251;
      }
      for (std::size_t cell = 0; cell < words.size(); ++cell) words[cell] = partial[cell];
    });
  }
};
struct FieldEngine {
  const std::vector<uint8_t>& input_a;
  const std::vector<uint8_t>& input_b;
  const Options& options;
  rns8::Context context;
  std::unique_ptr<rns8::Plan> plan;
  std::unique_ptr<rns8::Workspace> workspace;
  std::unique_ptr<rns8::Matrix> a, b, c;
  std::vector<uint8_t> output;
  Words words;
  Json metadata;
  FieldEngine(const std::vector<uint8_t>& lhs, const std::vector<uint8_t>& rhs, const Options& config)
      : input_a(lhs), input_b(rhs), options(config), context(config.device, config.backend),
        output(std::size_t(config.m) * config.n), words(output.size()) {
    metadata["device"] = device_info(context, options);
    plan = std::make_unique<rns8::Plan>(context, gemm_desc(config, RNS8_FINITE_FIELD_U8, config.m, config.n, config.k, 0));
    const uint64_t envelope = (uint64_t(config.m) * config.k + uint64_t(config.k) * config.n
        + uint64_t(config.m) * config.n) * 128 + plan->backend_info().workspace_required_bytes;
    options.memory->reserve(envelope, options.memory_mib * 1024 * 1024, "retained finite candidate storage/workspace");
    require(!context.device_info().global_mem_bytes || envelope <= context.device_info().global_mem_bytes,
            "finite plan exceeds total device memory; free memory remains runtime checked");
    workspace = std::make_unique<rns8::Workspace>(context, *plan);
    a = std::make_unique<rns8::Matrix>(context, matrix_desc(config.m, config.k, RNS8_FINITE_FIELD_U8, RNS8_BOUND_NONE, 0));
    b = std::make_unique<rns8::Matrix>(context, matrix_desc(config.k, config.n, RNS8_FINITE_FIELD_U8, RNS8_BOUND_NONE, 0));
    c = std::make_unique<rns8::Matrix>(context, matrix_desc(config.m, config.n, RNS8_FINITE_FIELD_U8, RNS8_BOUND_NONE, 0));
    metadata["plan"] = plan_info(*plan);
    require(plan->backend_info().backend == config.backend, "finite plan changed explicit backend");
    metadata["storage"] = {{"a", storage_info_value(*a)}, {"b", storage_info_value(*b)}, {"c", storage_info_value(*c)}};
  }
  static Json storage_info_value(const rns8::Matrix& matrix) {
    const auto value = matrix.storage_info();
    return {{"host_residue_bytes", value.host_residue_bytes}, {"device_residue_bytes", value.device_residue_bytes}};
  }
  void execute(Phases& phases, uint64_t version) {
    phases.measure("pack_and_upload_ns", [&] {
      rns8::check(rns8_pack_finite_u8(context.get(), a->get(), 251, input_a.data(), options.k, version));
      rns8::check(rns8_pack_finite_u8(context.get(), b->get(), 251, input_b.data(), options.n, 1));
    });
    phases.measure("gemm_reduce_and_required_sync_ns", [&] {
      rns8::check(rns8_gemm_finite_u8(context.get(), plan->get(), 251, a->get(), b->get(), c->get(), workspace->get()));
    });
    phases.measure("canonical_export_and_download_ns", [&] {
      rns8::check(rns8_export_finite_u8(context.get(), plan->get(), 251, c->get(), output.data(), options.n));
      for (std::size_t cell = 0; cell < output.size(); ++cell) words[cell] = output[cell];
    });
  }
};
}
int main(int argc, char** argv) {
  return program(argc, argv, [](const Options& options) {
    reject_options(options, {"--wide", "--gram", "--bits", "--baseline-max-products"});
    memory_admission(options);
    const auto preparation_start = Clock::now();
    Dataset data;
    data.recipe = "splitmix64_mod251_v1";
    std::vector<uint8_t> a(std::size_t(options.m) * options.k), b(std::size_t(options.k) * options.n);
    uint64_t state = options.seed;
    for (auto* operand : {&a, &b}) for (std::size_t index = 0; index < operand->size(); ++index) {
      const uint64_t value = next_word(state);
      (*operand)[index] = options.pattern == "zero" || (options.pattern == "sparse" && index % 8) ? 0 : static_cast<uint8_t>(value % 251);
    }
    if (options.size == "tiny" && options.m == 2 && options.n == 2 && options.k == 3 && options.pattern == "seeded"
        && !options.supplied.count("--seed")) {
      a = {250, 2, 3, 4, 5, 6}; b = {7, 8, 9, 10, 11, 12}; data.recipe = "tiny_gf251_v1";
    }
    data.a.assign(a.begin(), a.end()); data.b.assign(b.begin(), b.end());
    data.preparation_ns = since(preparation_start);
    const auto reference = [&] {
      auto integers = exact_product(data, options);
      Words result(integers.size());
      for (std::size_t cell = 0; cell < result.size(); ++cell) result[cell] = (integers[cell] % 251).convert_to<uint64_t>();
      return result;
    };
    const auto reference_start = Clock::now(); const Words expected = reference(); const double reference_ns = since(reference_start);
    Json report = base_report(options, "gf251_linear_transform");
    report["task"] = "Batched GF(251) linear transform: each column is a data block; rows select integer coefficient combinations modulo prime251.";
    report["formula"] = "C[i,j] = sum_q A[i,q]*B[q,j] mod251";
    report["semantics"] = "finite_u8_field"; report["finite_modulus"] = 251;
    report["output_contract"] = "row_major_canonical_GF251_values_serialized_as_one_u64_each";
    report["output_limbs_per_cell"] = 1;
    report["abs_partial_sum_bound"] = cpp_int(cpp_int(options.k) * 250 * 250).str();
    report["blocked_slab_k"] = 256; report["blocked_partial_max"] = 16000250;
    report["field_note"] = "GF(251) prime-field arithmetic; neither GF(2^8) nor the modulo256 integer ring";
    report["dataset_recipe"] = data.recipe; report["input_checksum_u64"] = checksum(dataset_words(data));
    report["reference_ns"] = reference_ns; report["output_preview_decimal"] = preview(expected, 1);
    report["useful_integer_products_per_call"] = uint64_t(options.m) * options.n * options.k;
    report["lifecycle"] = "retained canonical byte inputs; stable source version1, real A mutation with version2; full product/export each call";
    std::vector<Variant> variants;
    variants.push_back(state_variant<FieldBaseline>("naive_reduce_every_product", a, b, options, false));
    variants.push_back(state_variant<FieldBaseline>("blocked_u32_deferred_reduce", a, b, options, true));
    const auto setup = Clock::now(); auto candidate = std::make_shared<FieldEngine>(a, b, options);
    Variant gpu;
    gpu.name = "rns8_" + std::string(backend_name(options.backend)); gpu.setup_ns = since(setup); gpu.metadata = candidate->metadata;
    gpu.execute = [candidate](Phases& phases, uint64_t version) { candidate->execute(phases, version); };
    gpu.output = [candidate]() -> const Words& { return candidate->words; }; variants.push_back(std::move(gpu)); candidate.reset();
    run_variants(report, options, data, variants, expected, [&] {
      a[0] = static_cast<uint8_t>((uint32_t(a[0]) + 1) % 251); data.a[0] = a[0];
      report["changed_input_checksum_u64"] = checksum(dataset_words(data)); return reference();
    });
    report["reference_method"] = "independent cpp_int full integer dots followed by mod251; every canonical output compared";
    print_report(options, report);
  });
}
