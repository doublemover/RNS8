// Same public workload and controls, plus a separately audited native diagnostic batch.
#include "native_diagnostic_observation.hpp"
#include "../examples/workloads/workload_integer_cross_product.hpp"

int main(int argc, char** argv) {
  using namespace rns8_workload;
  return program(argc, argv, [](const Options& options) {
    require(options.gram && options.gram_reuse, "Gram diagnostics require --gram and --gram-reuse");
    if (options.backend != RNS8_BACKEND_HIP_DIRECT)
      throw rns8::Error(RNS8_UNSUPPORTED_BACKEND);
    integer_cross_product(options, [&](rns8_context* ctx, const std::function<void()>& run) {
      auto value = rns8_benchmark::observe_native_batch(ctx, run);
      value["schema"] = "rns8_native_application_batch_v1";
      value["scope"] = "one_separate_public_application_batch";
      value["observed_batches"] = 1;
      value["gram_products"] = options.gram_reuse;
      value["rns_exports"] = options.gram_export == "each" ? options.gram_reuse : 1;
      value["inside_timed_samples"] = false;
      value["inside_sustained_interval"] = false;
      value["event_time_scope"] = "per_label_sums_may_overlap_not_additive_end_to_end";
      return value;
    });
  });
}
