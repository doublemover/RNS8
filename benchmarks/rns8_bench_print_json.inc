void print_json(
    const Args& args,
    const rns8_device_info& info,
    const BenchmarkResult& result,
    uint64_t bound,
    const std::string& cmdline) {
  std::cout << std::setprecision(17);
#include "rns8_bench_print_json_context.inc"
#include "rns8_bench_print_json_header_backend.inc"
#include "rns8_bench_print_json_shape_schedule.inc"
#include "rns8_bench_print_json_timing_metadata.inc"
#include "rns8_bench_print_json_timings.inc"
}
