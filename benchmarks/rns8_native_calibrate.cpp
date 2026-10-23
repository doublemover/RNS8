#include "backend_hip_direct/hip_backend.hpp"
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
}

#include "native_narrow_calibration.inc"

int main(int argc, char** argv) {
  try {
    if (argc != 11) throw std::invalid_argument(
        "usage: rns8-native-calibrate <pack_pair|packed_gemm|pack_gemm|wmma|queue|split_k|epilogue_direct|epilogue_rocwmma|epilogue_unfused> M N K prefix group unsigned repeats warmups unique-run-id");
    // argc includes the executable: ten arguments, ending in the run identity.
    const std::string candidate = argv[1], run_id = argv[10];
    const auto m = number(argv[2],1,1024), n = number(argv[3],1,1024), k = number(argv[4],1,65537),
               prefix = number(argv[5],1,20), group = number(argv[6],1,4), unsigned_inputs = number(argv[7],0,1),
               repeats = number(argv[8],1,10000), warmups = number(argv[9],1,1000);
    const bool native = candidate == "wmma" || candidate == "queue" || candidate == "split_k";
    if (candidate == "narrow_wmma" || candidate == "narrow_scalar" || candidate == "narrow_vector" || candidate == "narrow_rns")
      return calibrate_narrow(candidate, m, n, k, prefix, group, unsigned_inputs, repeats, warmups, run_id);
    const bool epilogue = candidate == "epilogue_direct" || candidate == "epilogue_rocwmma" || candidate == "epilogue_unfused";
    const int64_t alpha = unsigned_inputs ? INT64_MAX : INT64_MIN, beta = INT64_MAX;
    if ((!native && !epilogue && candidate != "pack_pair" && candidate != "packed_gemm" && candidate != "pack_gemm") ||
        run_id.empty() || (group != 1 && group != 2 && group != 4) || (!native && group != 1) ||
        uint64_t{m} * n * k > 1073741824) throw std::invalid_argument("invalid candidate/group or reference-work limit");
    const uint32_t splits = std::min(k, 3u), workers = 32;
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
    if (candidate != "pack_pair")
      for (uint32_t row = 0; row < m; ++row) for (uint32_t col = 0; col < n; ++col) {
        cpp_int sum = 0;
        for (uint32_t q = 0; q < k; ++q) sum += native_value(a[row*k+q],unsigned_inputs)*native_value(b[q*n+col],unsigned_inputs);
        if (epilogue) sum = cpp_int(alpha) * sum + cpp_int(beta) * native_value(d[row*n+col],unsigned_inputs);
        for (uint32_t p = 0; p < prefix; ++p) expected[(std::size_t{p}*m+row)*n+col] = centered_residue(sum,kDefaultModuli[p]);
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
      else if (candidate == "queue") check(hip_direct_gemm_native_wmma_queue(0,tasks,workers,UINT64_MAX,written));
      else if (candidate == "split_k") check(hip_direct_gemm_native_wmma_split_k_device(0,da.get(),db.get(),static_cast<int8_t*>(dc.get()),m,n,k,k,n,n,prefix,group,unsigned_inputs,splits,workers,UINT64_MAX,written));
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
    std::cout << nlohmann::json{{"schema","rns8_native_microcase_v1"},{"run_id",run_id},{"candidate",candidate},
        {"m",m},{"n",n},{"k",k},{"prefix",prefix},{"group",group},{"unsigned_inputs",bool(unsigned_inputs)},
        {"splits",candidate=="split_k" ? splits : 1u},{"workers",workers},{"seed",3135},{"repeats",repeats},{"warmups",warmups},
        {"input_domain",(candidate=="packed_gemm" || epilogue) ? "resident_centered_residues" : "resident_native_64"},
        {"output_contract",candidate=="pack_pair" ? "operand_residue_planes" : epilogue ? "integer_alpha_ab_plus_beta_d_residues_no_integer_export" : "product_residues_default_basis_no_integer_export"},
        {"expression",epilogue ? nlohmann::json{{"alpha",alpha},{"beta",beta},{"coefficients","original_integers"},{"d_dataset","seed_3135_after_a_b"}}
                                : nlohmann::json(nullptr)},
        {"reserved_intermediate_bytes",epilogue ? std::size_t{m}*n*prefix : 0},
        {"materialized_intermediate_bytes",candidate=="epilogue_unfused" ? std::size_t{m}*n*prefix : 0},
        {"device",device.name},{"target",device.gcn_arch},{"runtime_version",device.hip_runtime_version},
        {"build_manifest",nlohmann::json::parse(RNS8_BUILD_MANIFEST_JSON)},
        {"native_input_storage_bytes",(a.size()+b.size())*8},{"residue_input_storage_bytes",expected_a.size()+expected_b.size()+expected_d.size()},
        {"residue_output_bytes",std::size_t{m}*n*prefix},{"actual_hbm_bytes",nullptr},{"timing_resolution_ns",nullptr},
        {"isa_verified",false},{"promotion_eligible",false},{"samples",samples},
        {"timing_scope","resident inputs; host calls include per-call allocations/capability/sync; labeled HIP events; initial upload, poison and Boost verification excluded"},
        {"cache_state","repeated inputs; destination reinitialized outside every timed call"}}.dump(2) << '\n';
    return 0;
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
