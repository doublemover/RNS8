#include "backend_hip_direct/hip_backend.hpp"
#include "backend_hip_direct/native_wmma_prepared.hpp"
#include "backend_rocwmma/rocwmma_backend.hpp"
#include "backend_vector_alu/vector_alu_backend.hpp"
#include "core/hip_resources.hpp"
#include "core/internal.hpp"
#include <charconv>
#include <chrono>
#include <cstring>
#include <iostream>
#include <nlohmann/json.hpp>
#include <random>
#include <stdexcept>

#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
extern "C" int rns8_hip_direct_pack_i64_device(const int64_t*, int8_t*, int, int, int, int);
extern "C" int rns8_hip_direct_pack_u64_device(const uint64_t*, int8_t*, int, int, int, int);
#endif

namespace {
using namespace rns8::detail;
using boost::multiprecision::cpp_int;
void check(rns8_status status) {
  if (status != RNS8_SUCCESS) throw std::runtime_error(rns8_status_string(status));
}
uint32_t number(const char* text, uint32_t low, uint32_t high) {
  uint32_t value = 0;
  const auto end = text + std::strlen(text);
  const auto result = std::from_chars(text, end, value);
  if (result.ec != std::errc{} || result.ptr != end || value < low || value > high)
    throw std::invalid_argument("numeric argument outside supported calibration domain");
  return value;
}
cpp_int native_value(uint64_t bits, bool unsigned_inputs) {
  if (unsigned_inputs) return cpp_int(bits);
  int64_t value;
  std::memcpy(&value, &bits, sizeof(value));
  return cpp_int(value);
}
void pack_pair(const void* a, const void* b, void* ra, void* rb,
               int m, int n, int k, int prefix, bool unsigned_inputs) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  const auto hip_check = [](hipError_t status) {
    if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status));
  };
  hip_unique_event start, stop;
  const bool timed = hip_direct_timing_enabled();
  if (timed) {
    hip_check(start.create()); hip_check(stop.create());
    hip_check(hipEventRecord(start.get(), nullptr));
  }
  if (unsigned_inputs) {
    hip_check(static_cast<hipError_t>(rns8_hip_direct_pack_u64_device(static_cast<const uint64_t*>(a), static_cast<int8_t*>(ra), m,k,k,prefix)));
    hip_check(static_cast<hipError_t>(rns8_hip_direct_pack_u64_device(static_cast<const uint64_t*>(b), static_cast<int8_t*>(rb), k,n,n,prefix)));
  } else {
    hip_check(static_cast<hipError_t>(rns8_hip_direct_pack_i64_device(static_cast<const int64_t*>(a), static_cast<int8_t*>(ra), m,k,k,prefix)));
    hip_check(static_cast<hipError_t>(rns8_hip_direct_pack_i64_device(static_cast<const int64_t*>(b), static_cast<int8_t*>(rb), k,n,n,prefix)));
  }
  if (!timed) {
    hip_check(hipDeviceSynchronize());
    return;
  }
  hip_check(hipEventRecord(stop.get(), nullptr));
  hip_check(hipEventSynchronize(stop.get()));
  float milliseconds = 0;
  hip_check(hipEventElapsedTime(&milliseconds, start.get(), stop.get()));
  hip_direct_timing_record_sample("native_calibration_pack_pair", milliseconds * 1000.0);
#else
  (void)a; (void)b; (void)ra; (void)rb; (void)m; (void)n; (void)k; (void)prefix; (void)unsigned_inputs;
  throw std::runtime_error("HIP not compiled");
#endif
}
template<class Prepare, class Execute, class Verify>
nlohmann::json sustained_calls(Prepare prepare, Execute execute, Verify verify) {
  hip_direct_timing_set_enabled(false);
  prepare();
  uint32_t calls = 0;
  double elapsed_ns = 0;
  const auto start = std::chrono::steady_clock::now();
  do {
    execute();
    ++calls;
    elapsed_ns = std::chrono::duration<double, std::nano>(
        std::chrono::steady_clock::now() - start).count();
  } while (calls < 4096 && (calls < 16 || elapsed_ns < 250000000.0));
  verify();
  return {{"calls", calls}, {"elapsed_ns", elapsed_ns}, {"mean_call_ns", elapsed_ns / calls},
          {"final_exact_reference_passed", true},
          {"scope", "same resident inputs and output; synchronous checked calls; no timing hooks, poison or reference readback between calls; final output and canaries verified"}};
}
}

#include "native_narrow_calibration.inc"

int main(int argc, char** argv) {
  try {
    const char* usage =
        "usage: rns8-native-calibrate <pack_pair|packed_gemm|pack_gemm|wmma|grid_prepared|queue|queue_prepared|split_k|"
        "narrow_wmma|narrow_scalar|narrow_vector|narrow_rns|epilogue_direct|epilogue_rocwmma|epilogue_unfused> "
        "M N K prefix group unsigned repeats warmups unique-run-id [workers]\n"
        "workers: queue/queue_prepared/split_k only, 1..1024 (default 32); launch clamps to descriptor count.\n"
        "M,N:1..1024 K:1..65537 prefix:1..20 group:1,2,4 unsigned:0,1; M*N*K<=1073741824.\n"
        "repeats:1..10000 warmups:1..1000; group=1 outside wmma/grid_prepared/queue/queue_prepared/split_k.\n"
        "narrow cases: prefix=group=1 K<=32768; narrow_scalar requires unsigned=0.";
    if (argc == 2 && std::string(argv[1]) == "--help") { std::cout << usage << '\n'; return 0; }
    if (argc != 11 && argc != 12) throw std::invalid_argument(usage);
    // Preserve the original ten arguments; optional workers follows run identity.
    const std::string candidate = argv[1], run_id = argv[10];
    const auto m = number(argv[2],1,1024), n = number(argv[3],1,1024), k = number(argv[4],1,65537),
               prefix = number(argv[5],1,20), group = number(argv[6],1,4), unsigned_inputs = number(argv[7],0,1),
               repeats = number(argv[8],1,10000), warmups = number(argv[9],1,1000);
    const bool queued = candidate == "queue" || candidate == "queue_prepared" || candidate == "split_k";
    if (argc == 12 && !queued)
      throw std::invalid_argument("workers applies only to queue/queue_prepared/split_k candidates");
    const uint32_t requested_workers = queued ? (argc == 12 ? number(argv[11], 1, 1024) : 32u) : 0u;
    const bool prepared = candidate == "queue_prepared" || candidate == "grid_prepared";
    const bool native = candidate == "wmma" || queued || prepared;
    if (candidate == "narrow_wmma" || candidate == "narrow_scalar" || candidate == "narrow_vector" || candidate == "narrow_rns")
      return calibrate_narrow(candidate, m, n, k, prefix, group, unsigned_inputs, repeats, warmups, run_id);
    const bool epilogue = candidate == "epilogue_direct" || candidate == "epilogue_rocwmma" || candidate == "epilogue_unfused";
    const int64_t alpha = unsigned_inputs ? INT64_MAX : INT64_MIN, beta = INT64_MAX;
    if ((!native && !epilogue && candidate != "pack_pair" && candidate != "packed_gemm" && candidate != "pack_gemm") ||
        run_id.empty() || (group != 1 && group != 2 && group != 4) || (!native && group != 1) ||
        uint64_t{m} * n * k > 1073741824) throw std::invalid_argument("invalid candidate/group or reference-work limit");
    const uint32_t splits = candidate == "split_k" ? std::min(k, 3u) : 1u;
    const uint32_t queue_descriptors = queued ? ((m + 15) / 16) * ((n + 15) / 16) *
        ((prefix + group - 1) / group) * splits : 0u;
    const uint32_t effective_workers = std::min(requested_workers, queue_descriptors);
    const std::string queue_kernel_selection = queued ? "homogeneous_group_" + std::to_string(group) +
        (unsigned_inputs ? "_unsigned" : "_signed") : "not_applicable";
    rns8_device_info device{};
    check(hip_direct_probe(0, device));
    std::mt19937_64 rng(3135);
    std::vector<uint64_t> a(std::size_t{m}*k), b(std::size_t{k}*n);
    for (auto& x : a) x = rng();
    for (auto& x : b) x = rng();
    a.front() = uint64_t{1} << 63; b.front() = UINT64_MAX;
    std::vector<uint64_t> d(epilogue ? std::size_t{m} * n : 0);
    for (auto& x : d) x = rng();
    std::vector<int8_t> expected_d(d.size() * prefix);
    std::vector<int8_t> expected_a(a.size()*prefix), expected_b(b.size()*prefix),
                        expected(std::size_t{m}*n*prefix + 64, 99);
    for (uint32_t p = 0; p < prefix; ++p) {
      for (std::size_t i = 0; i < a.size(); ++i) expected_a[p*a.size()+i] = centered_residue(native_value(a[i],unsigned_inputs),kDefaultModuli[p]);
      for (std::size_t i = 0; i < b.size(); ++i) expected_b[p*b.size()+i] = centered_residue(native_value(b[i],unsigned_inputs),kDefaultModuli[p]);
    }
    for (uint32_t p = 0; p < prefix; ++p)
      for (std::size_t i = 0; i < d.size(); ++i)
        expected_d[p*d.size()+i] = centered_residue(native_value(d[i],unsigned_inputs),kDefaultModuli[p]);
    if (candidate != "pack_pair") {
      // K <= 65537 and |input| < 2^64 bound every dot product below 2^145.
      // Even the full-width alpha/beta expression is below 2^209. Fixed Boost
      // 256-bit arithmetic is therefore exact, with no per-product allocation.
      // Transpose the reference B only; GPU inputs and timed work are unchanged.
      using boost::multiprecision::int256_t;
      std::vector<int256_t> reference_a(a.size()), reference_bt(b.size());
      for (std::size_t i = 0; i < a.size(); ++i) reference_a[i] = int256_t(native_value(a[i], unsigned_inputs));
      for (uint32_t col = 0; col < n; ++col)
        for (uint32_t q = 0; q < k; ++q)
          reference_bt[std::size_t{col} * k + q] = int256_t(native_value(b[std::size_t{q} * n + col], unsigned_inputs));
      for (uint32_t row = 0; row < m; ++row) for (uint32_t col = 0; col < n; ++col) {
        int256_t fixed_sum = 0;
        for (uint32_t q = 0; q < k; ++q)
          fixed_sum += reference_a[std::size_t{row} * k + q] * reference_bt[std::size_t{col} * k + q];
        cpp_int sum = fixed_sum;
        // Independent unbounded audit of both edge columns in every row.
        if (col == 0 || col + 1 == n) {
          cpp_int audit = 0;
          for (uint32_t q = 0; q < k; ++q)
            audit += native_value(a[row*k+q], unsigned_inputs) * native_value(b[q*n+col], unsigned_inputs);
          if (audit != sum) throw std::runtime_error("fixed-width reference differs from unbounded audit");
        }
        if (epilogue) sum = cpp_int(alpha) * sum + cpp_int(beta) * native_value(d[row*n+col],unsigned_inputs);
        for (uint32_t p = 0; p < prefix; ++p) expected[(std::size_t{p}*m+row)*n+col] = centered_residue(sum,kDefaultModuli[p]);
      }
    }
    hip_direct_device_buffer da, db, ra, rb, rd, intermediate, dc;
    check(da.allocate(0,a.size()*8)); check(db.allocate(0,b.size()*8));
    check(ra.allocate(0,expected_a.size())); check(rb.allocate(0,expected_b.size())); check(dc.allocate(0,expected.size()));
    check(hip_direct_copy_host_to_device(0,da.get(),a.data(),a.size()*8));
    check(hip_direct_copy_host_to_device(0,db.get(),b.data(),b.size()*8));
    if (epilogue) {
      check(rd.allocate(0,expected_d.size()));
      check(hip_direct_copy_host_to_device(0,rd.get(),expected_d.data(),expected_d.size()));
      // Reserve identical scratch for all three epilogue cases; only unfused
      // writes it. This keeps allocation footprint out of the timing comparison.
      check(intermediate.allocate(0,std::size_t{m}*n*prefix));
    }
    pack_pair(da.get(),db.get(),ra.get(),rb.get(),m,n,k,prefix,unsigned_inputs);
    const auto compare = [&](void* pointer, const std::vector<int8_t>& reference) {
      std::vector<int8_t> observed(reference.size());
      check(hip_direct_copy_device_to_host(0,observed.data(),pointer,observed.size()));
      if (observed != reference) throw std::runtime_error("native calibration differs from exact reference or canary");
    };
    compare(ra.get(),expected_a); compare(rb.get(),expected_b);
    std::vector<NativeWmmaTask> tasks{{da.get(),db.get(),static_cast<int8_t*>(dc.get()),m,n,k,k,n,n,prefix,group,unsigned_inputs}};
    NativeWmmaPreparedQueue prepared_queue;
    double queue_prepare_ns = 0;
    if (prepared) {
      hip_direct_timing_set_enabled(false);
      const auto start = std::chrono::steady_clock::now();
      check(prepared_queue.prepare(0, tasks, requested_workers ? requested_workers : 32, UINT64_MAX));
      queue_prepare_ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start).count();
    }
    const auto execute = [&]() {
      if (candidate == "pack_pair" || candidate == "pack_gemm") pack_pair(da.get(),db.get(),ra.get(),rb.get(),m,n,k,prefix,unsigned_inputs);
      if (candidate == "pack_pair") return;
      bool written = false;
      if (candidate == "epilogue_direct") check(hip_direct_gemm_rns_axpby_device(0,ra.get(),rb.get(),rd.get(),dc.get(),m,n,k,k,n,n,n,prefix,alpha,beta));
      else if (candidate == "epilogue_rocwmma") check(rocwmma_exact_axpby_device(0,ra.get(),rb.get(),rd.get(),dc.get(),m,n,k,k,n,n,n,prefix,alpha,beta));
      else if (candidate == "epilogue_unfused") {
        check(hip_direct_gemm_rns_device(0,ra.get(),rb.get(),intermediate.get(),m,n,k,k,n,n,prefix));
        check(hip_direct_axpby_residues_device(0,intermediate.get(),rd.get(),dc.get(),m,n,n,n,n,prefix,alpha,beta));
      }
      else if (candidate == "wmma") check(hip_direct_gemm_native_wmma_grouped_device(0,da.get(),db.get(),static_cast<int8_t*>(dc.get()),m,n,k,k,n,n,prefix,group,unsigned_inputs));
      else if (candidate == "queue") check(hip_direct_gemm_native_wmma_queue(0,tasks,requested_workers,UINT64_MAX,written));
      else if (candidate == "queue_prepared") check(prepared_queue.execute(written));
      else if (candidate == "grid_prepared") check(prepared_queue.execute_ordinary(written));
      else if (candidate == "split_k") check(hip_direct_gemm_native_wmma_split_k_device(0,da.get(),db.get(),static_cast<int8_t*>(dc.get()),m,n,k,k,n,n,prefix,group,unsigned_inputs,splits,requested_workers,UINT64_MAX,written));
      else check(hip_direct_gemm_rns_device(0,ra.get(),rb.get(),dc.get(),m,n,k,k,n,n,prefix));
    };
    const auto prepare = [&]() {
      std::vector<int8_t> poison(expected.size(),99);
      check(hip_direct_copy_host_to_device(0,dc.get(),poison.data(),poison.size()));
      if (epilogue) check(hip_direct_zero(0,intermediate.get(),std::size_t{m}*n*prefix));
      if (candidate == "pack_pair" || candidate == "pack_gemm") {
        check(hip_direct_zero(0,ra.get(),expected_a.size())); check(hip_direct_zero(0,rb.get(),expected_b.size()));
      }
    };
    const auto verify = [&]() {
      if (candidate == "pack_pair" || candidate == "pack_gemm") { compare(ra.get(),expected_a); compare(rb.get(),expected_b); }
      if (candidate != "pack_pair") compare(dc.get(),expected);
    };
    for (uint32_t i=0;i<warmups;++i) { prepare(); execute(); verify(); }
    nlohmann::json samples = nlohmann::json::array();
    hip_direct_timing_set_enabled(true);
    for (uint32_t i=0;i<repeats;++i) {
      prepare(); hip_direct_timing_reset();
      const auto start = std::chrono::steady_clock::now();
      execute();
      const double host_ns = std::chrono::duration<double,std::nano>(std::chrono::steady_clock::now()-start).count();
      hip_direct_timing_flush_pending_events();
      nlohmann::json events = nlohmann::json::array();
      for (const auto& event : hip_direct_timing_snapshot()) events.push_back({{"label",event.label},
          {"elapsed_ns",event.microseconds>0 ? nlohmann::json(event.microseconds*1000) : nlohmann::json(nullptr)}});
      verify();
      hip_direct_timing_set_enabled(false);
      prepare();
      const auto production_start = std::chrono::steady_clock::now();
      execute();
      const double production_ns = std::chrono::duration<double,std::nano>(
          std::chrono::steady_clock::now()-production_start).count();
      verify();
      hip_direct_timing_set_enabled(true);
      samples.push_back({{"host_call_ns",host_ns>0 ? nlohmann::json(host_ns) : nlohmann::json(nullptr)},
                         {"uninstrumented_host_call_ns",production_ns},
                         {"uninstrumented_exact_reference_passed",true},
                         {"events",events},{"exact_reference_passed",true}});
    }
    hip_direct_timing_set_enabled(false);
    const auto sustained = sustained_calls(prepare, execute, verify);
    std::cout << nlohmann::json{{"schema","rns8_native_microcase_v2"},{"run_id",run_id},{"candidate",candidate},
        {"sustained",sustained},
        {"queue_prepare_ns",queue_prepare_ns},
        {"queue_preparation_scope",prepared ? "one host preparation including allocation, capability check and descriptor upload; excluded from steady calls" : "not_applicable"},
        {"queue_descriptor_storage_bytes",queued ? std::size_t(splits)*sizeof(NativeWmmaTask)+std::size_t(queue_descriptors)*sizeof(NativeWmmaWork)+sizeof(uint32_t) : prepared ? sizeof(NativeWmmaTask) + std::size_t((m+15)/16)*((n+15)/16)*((prefix+group-1)/group)*sizeof(NativeWmmaWork)+sizeof(uint32_t) : 0},
        {"m",m},{"n",n},{"k",k},{"prefix",prefix},{"group",group},{"unsigned_inputs",bool(unsigned_inputs)},
        {"splits",splits},{"requested_workers",requested_workers},{"effective_workers",effective_workers},
        {"queue_descriptors",queue_descriptors},{"queue_kernel_selection",queue_kernel_selection},{"seed",3135},{"repeats",repeats},{"warmups",warmups},
        {"input_domain",(candidate=="packed_gemm" || epilogue) ? "resident_centered_residues" : "resident_native_64"},
        {"output_contract",candidate=="pack_pair" ? "operand_residue_planes" : epilogue ? "integer_alpha_ab_plus_beta_d_residues_no_integer_export" : "product_residues_default_basis_no_integer_export"},
        {"expression",epilogue ? nlohmann::json{{"alpha",alpha},{"beta",beta},{"coefficients","original_integers"},{"d_dataset","seed_3135_after_a_b"}}
                                : nlohmann::json(nullptr)},
        {"reserved_intermediate_bytes",epilogue ? std::size_t{m}*n*prefix : 0},
        {"materialized_intermediate_bytes",candidate=="epilogue_unfused" ? std::size_t{m}*n*prefix : 0},
        {"device",device.name},{"target",device.gcn_arch},{"runtime_version",device.hip_runtime_version},
        {"build_manifest",nlohmann::json::parse(RNS8_BUILD_MANIFEST_JSON)},
        {"reference_method",candidate=="pack_pair" ? "boost_cpp_int_operand_residues" :
            "boost_int256_full_product_with_cpp_int_row_edge_audits_and_epilogue"},
        {"reference_operand_storage_bytes",candidate=="pack_pair" ? std::size_t{0} :
            (a.size()+b.size())*sizeof(boost::multiprecision::int256_t)},
        {"native_input_storage_bytes",(a.size()+b.size())*8},{"residue_input_storage_bytes",expected_a.size()+expected_b.size()+expected_d.size()},
        {"residue_output_bytes",std::size_t{m}*n*prefix},{"actual_hbm_bytes",nullptr},{"timing_resolution_ns",nullptr},
        {"isa_verified",false},{"promotion_eligible",false},{"samples",samples},
        {"timing_scope",prepared ? "resident inputs and prepared descriptors; host calls include synchronous execution (cursor reset for queue only); one-time queue_prepare_ns reported separately; initial upload, poison and Boost verification excluded" : "resident inputs; host calls include per-call allocations/capability/sync; labeled HIP events; initial upload, poison and Boost verification excluded"},
        {"cache_state","repeated inputs; destination reinitialized outside every timed call"}}.dump(2) << '\n';
    return 0;
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
