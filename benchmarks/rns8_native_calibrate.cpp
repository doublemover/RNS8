#include "backend_hip_direct/hip_backend.hpp"
#include "backend_hip_direct/native_wmma_prepared.hpp"
#include "backend_hip_direct/native_tiled_prepared.hpp"
#include "backend_hip_direct/encoded_b_backend.hpp"
#include "backend_common/complex_residue_plan.hpp"
#include "backend_rocwmma/rocwmma_backend.hpp"
#include "backend_vector_alu/vector_alu_backend.hpp"
#include "core/hip_resources.hpp"
#include "core/internal.hpp"
#include "epilogue_coefficients.hpp"
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
#include "complex_residue_calibration.inc"

int main(int argc, char** argv) {
  try {
    const char* usage =
        "usage: rns8-native-calibrate <pack_pair|packed_gemm|pack_gemm|wmma|grid_prepared|queue|queue_prepared|split_k|adaptive_grid|adaptive_compact|complex_3m|complex_4m|"
        "conversion_cold|conversion_prepared|conversion_pack_gemm|tiled_register|tiled_lds|encoded_b_cold|encoded_b_warm|narrow_wmma|narrow_scalar|narrow_vector|narrow_rns|epilogue_direct|epilogue_rocwmma|epilogue_unfused> "
        "M N K prefix group unsigned repeats warmups unique-run-id [mode-specific controls]\n"
        "workers: queue/queue_prepared/split_k only, 1..1024 (default 32); launch clamps to descriptor count.\n"
        "adaptive_grid: append axis_l1; adaptive_compact: append workers axis_l1 (workers required, 1..65535, no count clamp).\n"
        "conversion modes: append depth buffers waves replicated|cooperative|encoded_lds reuse_count stable|reverse_k; cold includes one preparation per R-call batch, prepared reports setup separately.\n"
        "tiled modes: append k_depth lds_buffers; tiled_register requires16 0; tiled_lds requires16|32 and1|2; LDS budget65536 bytes/block.\n"
        "encoded_b_cold/warm: append required reuse_count1..256; wmma accepts the same suffix (default1). Each timed execute is R GEMMs, one final verification. Cold includes allocate/prepare/free; warm reports one setup separately.\n"
        "M,N:1..1024 K:1..65537 prefix:1..20 group:1,2,4 unsigned:0,1; M*N*K<=1073741824.\n"
        "repeats:1..10000 warmups:1..1000; adaptive cases require prefix20 and group1/2/4.\n"
        "group=1 outside wmma/encoded_b/grid_prepared/queue/queue_prepared/split_k/adaptive/tiled cases.\n"
        "narrow cases: prefix=group=1 K<=32768; narrow_scalar requires unsigned=0.\n"
        "epilogue cases: optional extreme|zero|copy|unit|general|channel-zero coefficient profile (default extreme).\n"
        "complex cases: prefix=20, group=1, unsigned=0; both output residue components, no integer export.";
    if (argc == 2 && std::string(argv[1]) == "--help") { std::cout << usage << '\n'; return 0; }
    if (argc < 11 || argc > 17) throw std::invalid_argument(usage);
    // Preserve the original ten arguments; optional workers follows run identity.
    const std::string candidate = argv[1], run_id = argv[10];
    const auto m = number(argv[2],1,1024), n = number(argv[3],1,1024), k = number(argv[4],1,65537),
               prefix = number(argv[5],1,20), group = number(argv[6],1,4), unsigned_inputs = number(argv[7],0,1),
               repeats = number(argv[8],1,10000), warmups = number(argv[9],1,1000);
    const bool epilogue = candidate == "epilogue_direct" || candidate == "epilogue_rocwmma" || candidate == "epilogue_unfused";
    const bool conversion = candidate == "conversion_cold" || candidate == "conversion_prepared" || candidate == "conversion_pack_gemm";
    const bool conversion_prepared = candidate == "conversion_prepared";
    const bool conversion_pack = candidate == "conversion_pack_gemm";
    if(conversion && argc!=17)throw std::invalid_argument("conversion cases require six explicit controls after run identity");
    NativeTiledConfig conversion_config;
    std::string conversion_name="replicated",conversion_input="stable";
    if(conversion) {
      conversion_config.depth=number(argv[11],16,32);conversion_config.buffers=number(argv[12],0,2);
      conversion_config.waves=number(argv[13],1,4);conversion_name=argv[14];conversion_input=argv[16];
      if(conversion_name=="cooperative")conversion_config.conversion=NativeConversion::cooperative;
      else if(conversion_name=="encoded_lds")conversion_config.conversion=NativeConversion::encoded_lds;
      else if(conversion_name!="replicated")throw std::invalid_argument("unknown conversion ownership");
      const auto resources=query_native_tiled_resources(conversion_config.depth,conversion_config.buffers,65536,
          conversion_config.conversion,group,conversion_config.waves);
      if(resources.status!=RNS8_SUCCESS || (conversion_input!="stable" && conversion_input!="reverse_k") ||
          (conversion_pack && (conversion_name!="replicated" || conversion_config.depth!=16 || conversion_config.buffers || conversion_config.waves!=4)))
        throw std::invalid_argument("unsupported conversion geometry/storage/input control");
    }
    const bool queued = candidate == "queue" || candidate == "queue_prepared" || candidate == "split_k";
    const bool adaptive = candidate == "adaptive_grid" || candidate == "adaptive_compact";
    const bool compact = candidate == "adaptive_compact";
    const bool tiled = candidate == "tiled_register" || candidate == "tiled_lds";
    const bool encoded = candidate == "encoded_b_cold" || candidate == "encoded_b_warm";
    const bool encoded_warm = candidate == "encoded_b_warm";
    const bool reuse_batch = encoded || candidate == "wmma" || conversion;
    if (encoded && argc != 12)
      throw std::invalid_argument("encoded_b cases require explicit reuse_count1..256 after run identity");
    if (adaptive && ((compact ? argc != 13 : argc != 12) || prefix != 20 ||
                     std::string(argv[compact ? 12 : 11]) != "axis_l1"))
      throw std::invalid_argument("adaptive cases require prefix20 and explicit axis_l1; compact requires workers1..65535 before method");
    if (tiled && argc != 13)
      throw std::invalid_argument("tiled cases require explicit k_depth lds_buffers after run identity");
    if (!conversion && !adaptive && !tiled && (argc > 12 || (argc == 12 && !queued && !reuse_batch && !epilogue)))
      throw std::invalid_argument("workers applies only to queue/queue_prepared/split_k or explicit adaptive_compact cases");
    const uint32_t reuse_count = conversion ? number(argv[15],1,256) : reuse_batch && argc == 12 ? number(argv[11],1,256) : 1;
    const uint32_t tiled_depth = tiled ? number(argv[11],16,32) : 0;
    const uint32_t tiled_buffers = tiled ? number(argv[12],0,2) : 0;
    const uint64_t tiled_lds_budget = 65536;
    NativeTiledResources tiled_resources;
    if (tiled) {
      tiled_resources = query_native_tiled_resources(tiled_depth,tiled_buffers,tiled_lds_budget);
      if (tiled_resources.status != RNS8_SUCCESS ||
          (candidate == "tiled_register" ? tiled_buffers != 0 : tiled_buffers == 0))
        throw std::invalid_argument("tiled_register requires depth16/buffers0; tiled_lds requires depth16|32/buffers1|2 within64KiB LDS");
    }
    const uint32_t requested_workers = compact ? number(argv[11], 1, 65535) :
        queued ? (argc == 12 ? number(argv[11], 1, 1024) : 32u) : 0u;
    const bool prepared = candidate == "queue_prepared" || candidate == "grid_prepared";
    const bool native = conversion || candidate == "wmma" || encoded || queued || prepared || adaptive || tiled;
    if (candidate == "complex_3m" || candidate == "complex_4m")
      return calibrate_complex(candidate, m, n, k, prefix, group, unsigned_inputs, repeats, warmups, run_id);
    if (candidate == "narrow_wmma" || candidate == "narrow_scalar" || candidate == "narrow_vector" || candidate == "narrow_rns")
      return calibrate_narrow(candidate, m, n, k, prefix, group, unsigned_inputs, repeats, warmups, run_id);

    const std::string coefficient_profile=epilogue && argc==12?argv[11]:"extreme";
    int64_t alpha=0,beta=0;
    if(!rns8::bench::epilogue_coefficients(coefficient_profile,unsigned_inputs,alpha,beta))throw std::invalid_argument("unknown epilogue coefficient profile");
    if ((!native && !epilogue && candidate != "pack_pair" && candidate != "packed_gemm" && candidate != "pack_gemm") ||
        run_id.empty() || (group != 1 && group != 2 && group != 4) || (!native && group != 1) ||
        uint64_t{m} * n * k > 1073741824) throw std::invalid_argument("invalid candidate/group or reference-work limit");
    const uint32_t splits = candidate == "split_k" ? std::min(k, 3u) : 1u;
    const uint32_t queue_descriptors = queued ? ((m + 15) / 16) * ((n + 15) / 16) *
        ((prefix + group - 1) / group) * splits : 0u;
    const uint32_t effective_workers = compact ? requested_workers : std::min(requested_workers, queue_descriptors);
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
    const bool reserve_residue_inputs=!conversion || conversion_pack;
    if(reserve_residue_inputs) {check(ra.allocate(0,expected_a.size()));check(rb.allocate(0,expected_b.size()));}
    check(dc.allocate(0,expected.size()));
    check(hip_direct_copy_host_to_device(0,da.get(),a.data(),a.size()*8));
    check(hip_direct_copy_host_to_device(0,db.get(),b.data(),b.size()*8));
    if (epilogue) {
      check(rd.allocate(0,expected_d.size()));
      check(hip_direct_copy_host_to_device(0,rd.get(),expected_d.data(),expected_d.size()));
      // Reserve identical scratch for all three epilogue cases; only unfused
      // writes it. This keeps allocation footprint out of the timing comparison.
      check(intermediate.allocate(0,std::size_t{m}*n*prefix));
    }
    if(reserve_residue_inputs)pack_pair(da.get(),db.get(),ra.get(),rb.get(),m,n,k,prefix,unsigned_inputs);
    const auto compare = [&](void* pointer, const std::vector<int8_t>& reference) {
      std::vector<int8_t> observed(reference.size());
      check(hip_direct_copy_device_to_host(0,observed.data(),pointer,observed.size()));
      if (observed != reference) throw std::runtime_error("native calibration differs from exact reference or canary");
    };
    if(reserve_residue_inputs){compare(ra.get(),expected_a);compare(rb.get(),expected_b);}
    std::vector<NativeWmmaTask> tasks{{da.get(),db.get(),static_cast<int8_t*>(dc.get()),m,n,k,k,n,n,prefix,group,unsigned_inputs}};
    NativeWmmaPreparedQueue prepared_queue;
    NativeTiledPrepared conversion_owner;
    double conversion_prepare_ns=0;
    if(conversion_prepared) {
      hip_direct_timing_set_enabled(false);
      const auto start=std::chrono::steady_clock::now();
      check(conversion_owner.prepare(0,tasks.front(),conversion_config));
      conversion_prepare_ns=std::chrono::duration<double,std::nano>(std::chrono::steady_clock::now()-start).count();
    }
    auto reverse_a=a,reverse_b=b;
    if(conversion && conversion_input=="reverse_k") {
      for(uint32_t row=0;row<m;++row)for(uint32_t q=0;q<k;++q)reverse_a[row*k+q]=a[row*k+k-1-q];
      for(uint32_t q=0;q<k;++q)for(uint32_t col=0;col<n;++col)reverse_b[q*n+col]=b[(k-1-q)*n+col];
    }
    bool reversed_inputs=false;
    EncodedBLayout encoded_layout;
    hip_direct_device_buffer encoded_storage;
    double encoded_prepare_ns = 0;
    if (encoded) {
      encoded_layout = encoded_b_layout(k,n,n,prefix,UINT64_MAX);
      check(encoded_layout.status);
      if (encoded_warm) {
        hip_direct_timing_set_enabled(false);
        const auto start=std::chrono::steady_clock::now();
        check(encoded_storage.allocate(0,encoded_layout.total_bytes));
        check(hip_prepare_encoded_b(0,db.get(),encoded_storage.get(),k,n,n,prefix,
                                    unsigned_inputs,encoded_layout.total_bytes));
        encoded_prepare_ns=std::chrono::duration<double,std::nano>(std::chrono::steady_clock::now()-start).count();
      }
    }
    double queue_prepare_ns = 0;
    if (prepared) {
      hip_direct_timing_set_enabled(false);
      const auto start = std::chrono::steady_clock::now();
      check(prepared_queue.prepare(0, tasks, requested_workers ? requested_workers : 32, UINT64_MAX));
      queue_prepare_ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start).count();
    }
    const auto execute = [&]() {
      if(conversion) {
        bool written=false;
        NativeTiledPrepared cold;
        if(!conversion_prepared && !conversion_pack)check(cold.prepare(0,tasks.front(),conversion_config));
        for(uint32_t reuse=0;reuse<reuse_count;++reuse) {
          if(conversion_pack) {
            pack_pair(da.get(),db.get(),ra.get(),rb.get(),m,n,k,prefix,unsigned_inputs);
            check(hip_direct_gemm_rns_device(0,ra.get(),rb.get(),dc.get(),m,n,k,k,n,n,prefix));
          } else {
            check(conversion_prepared ? conversion_owner.execute(written) : cold.execute(written));
            if(!written)throw std::runtime_error("prepared conversion succeeded without committing output");
          }
        }
        return;
      }
      if (candidate == "pack_pair" || candidate == "pack_gemm") pack_pair(da.get(),db.get(),ra.get(),rb.get(),m,n,k,prefix,unsigned_inputs);
      if (candidate == "pack_pair") return;
      bool written = false;
      if (candidate == "epilogue_direct") check(hip_direct_gemm_rns_axpby_device(0,ra.get(),rb.get(),rd.get(),dc.get(),m,n,k,k,n,n,n,prefix,alpha,beta));
      else if (candidate == "epilogue_rocwmma") check(rocwmma_exact_axpby_device(0,ra.get(),rb.get(),rd.get(),dc.get(),m,n,k,k,n,n,n,prefix,alpha,beta));
      else if (candidate == "epilogue_unfused") {
        check(hip_direct_gemm_rns_device(0,ra.get(),rb.get(),intermediate.get(),m,n,k,k,n,n,prefix));
        check(hip_direct_axpby_residues_device(0,intermediate.get(),rd.get(),dc.get(),m,n,n,n,n,prefix,alpha,beta));
      }
      else if (candidate == "wmma") {
        for (uint32_t reuse=0;reuse<reuse_count;++reuse)
          check(hip_direct_gemm_native_wmma_grouped_device(0,da.get(),db.get(),static_cast<int8_t*>(dc.get()),m,n,k,k,n,n,prefix,group,unsigned_inputs));
      }
      else if (encoded) {
        hip_direct_device_buffer cold_storage;
        void* storage=encoded_storage.get();
        if (!encoded_warm) {
          check(cold_storage.allocate(0,encoded_layout.total_bytes));
          storage=cold_storage.get();
          check(hip_prepare_encoded_b(0,db.get(),storage,k,n,n,prefix,unsigned_inputs,encoded_layout.total_bytes));
        }
        for (uint32_t reuse=0;reuse<reuse_count;++reuse) {
          check(hip_gemm_encoded_b(0,da.get(),storage,dc.get(),m,n,k,k,n,prefix,group,unsigned_inputs,written));
          if (!written) throw std::runtime_error("encoded B operation succeeded without committing output");
        }
        // Release stays inside the cold batch's host interval, including status.
        if (!encoded_warm) check(cold_storage.reset());
      }
      else if (tiled) {
        check(hip_direct_gemm_native_tiled_device(0,da.get(),db.get(),static_cast<int8_t*>(dc.get()),
            m,n,k,k,n,n,prefix,group,unsigned_inputs,tiled_depth,tiled_buffers,tiled_lds_budget,written));
        if (!written) throw std::runtime_error("tiled operation returned success without committing output");
      }
      else if (candidate == "queue") check(hip_direct_gemm_native_wmma_queue(0,tasks,requested_workers,UINT64_MAX,written));
      else if (candidate == "queue_prepared") check(prepared_queue.execute(written));
      else if (candidate == "grid_prepared") check(prepared_queue.execute_ordinary(written));
      else if (candidate == "split_k") check(hip_direct_gemm_native_wmma_split_k_device(0,da.get(),db.get(),static_cast<int8_t*>(dc.get()),m,n,k,k,n,n,prefix,group,unsigned_inputs,splits,requested_workers,UINT64_MAX,written));
      else if (adaptive) {
        check(hip_direct_gemm_adaptive_native_device(0, da.get(), db.get(), static_cast<int8_t*>(dc.get()),
            m, n, k, k, n, n, unsigned_inputs, prefix, group, UINT64_MAX, written,
            NativeBoundMethod::axis_l1, 256, compact ? requested_workers : 0));
        if (!written) throw std::runtime_error("adaptive operation returned success without committing output");
      }
      else check(hip_direct_gemm_rns_device(0,ra.get(),rb.get(),dc.get(),m,n,k,k,n,n,prefix));
    };
    const auto prepare = [&]() {
      if(conversion && conversion_input=="reverse_k") {
        reversed_inputs=!reversed_inputs;
        const auto& next_a=reversed_inputs?reverse_a:a;
        const auto& next_b=reversed_inputs?reverse_b:b;
        check(hip_direct_copy_host_to_device(0,da.get(),next_a.data(),next_a.size()*8));
        check(hip_direct_copy_host_to_device(0,db.get(),next_b.data(),next_b.size()*8));
      }
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
    auto sustained = sustained_calls(prepare, execute, verify);
    if (reuse_batch) {
      const uint64_t gemm_calls=sustained["calls"].get<uint64_t>()*reuse_count;
      sustained.update({{"gemm_calls",gemm_calls},
          {"mean_gemm_ns",sustained["elapsed_ns"].get<double>()/gemm_calls},
          {"verified_outputs_per_batch",1}});
    }
    nlohmann::json report{{"schema","rns8_native_microcase_v2"},{"run_id",run_id},{"candidate",candidate},
        {"sustained",sustained},
        {"queue_prepare_ns",queue_prepare_ns},
        {"queue_preparation_scope",prepared ? "one host preparation including allocation, capability check and descriptor upload; excluded from steady calls" : "not_applicable"},
        {"queue_descriptor_storage_bytes",queued ? std::size_t(splits)*sizeof(NativeWmmaTask)+std::size_t(queue_descriptors)*sizeof(NativeWmmaWork)+sizeof(uint32_t) : prepared ? sizeof(NativeWmmaTask) + std::size_t((m+15)/16)*((n+15)/16)*((prefix+group-1)/group)*sizeof(NativeWmmaWork)+sizeof(uint32_t) : 0},
        {"m",m},{"n",n},{"k",k},{"prefix",prefix},{"group",group},{"unsigned_inputs",bool(unsigned_inputs)},
        {"splits",splits},{"requested_workers",requested_workers},{"effective_workers",effective_workers},
        {"queue_descriptors",queue_descriptors},{"queue_kernel_selection",queue_kernel_selection},{"seed",3135},{"repeats",repeats},{"warmups",warmups},
        {"input_domain",(candidate=="packed_gemm" || epilogue) ? "resident_centered_residues" : "resident_native_64"},
        {"output_contract",candidate=="pack_pair" ? "operand_residue_planes" : epilogue ? "integer_alpha_ab_plus_beta_d_residues_no_integer_export" : "product_residues_default_basis_no_integer_export"},
        {"coefficient_profile",epilogue ? nlohmann::json(coefficient_profile) : nlohmann::json(nullptr)},
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
        {"native_input_storage_bytes",(a.size()+b.size())*8},{"residue_input_storage_bytes",reserve_residue_inputs ? expected_a.size()+expected_b.size()+expected_d.size() : 0},
        {"residue_output_bytes",std::size_t{m}*n*prefix},{"actual_hbm_bytes",nullptr},{"timing_resolution_ns",nullptr},
        {"isa_verified",false},{"promotion_eligible",false},{"samples",samples},
        {"timing_scope",prepared ? "resident inputs and prepared descriptors; host calls include synchronous execution (cursor reset for queue only); one-time queue_prepare_ns reported separately; initial upload, poison and Boost verification excluded" : "resident inputs; host calls include per-call allocations/capability/sync; labeled HIP events; initial upload, poison and Boost verification excluded"},
        {"cache_state","repeated inputs; destination reinitialized outside every timed call"}};
    if (reuse_batch && !conversion) {
      report.update({{"reuse_count",reuse_count},{"batch_gemm_calls",reuse_count},
          {"reuse_batch_boundary","R_synchronous_calls_same_native_A_B_one_final_verified_output"},
          {"reuse_comparison_boundary","private_raw_residue_no_public_matrix_proof_or_generation"},
          {"reuse_dataset","full_width_native64_mt19937_64_seed3135_forced_extrema_v1"},
          {"encoded_b_layout",encoded ? "modulus_n_k_packed_word32_v1" : "none"},
          {"encoded_b_k_words",encoded ? encoded_layout.k_words : 0},
          {"encoded_b_cache_logical_bytes",encoded ? encoded_layout.total_bytes : 0},
          {"encoded_b_preparation_in_batch",encoded && !encoded_warm},
          {"encoded_b_preparation_scope",encoded_warm ? "one_preparation_outside_batches_cached_consume" :
              encoded ? "per_batch_allocate_probe_encode_consume_release" : "none_direct_native_grouped_each_call"},
          {"encoded_b_prepare_ns",encoded_warm ? nlohmann::json(encoded_prepare_ns) : nlohmann::json(nullptr)},
          {"encoded_b_resident_cache_bytes",encoded_warm ? encoded_layout.total_bytes : 0},
          {"encoded_b_transient_cache_bytes",encoded && !encoded_warm ? encoded_layout.total_bytes : 0},
          {"encoded_b_same_B_immutable",true},{"encoded_b_public_ownership_measured",false},
          {"encoded_b_kernel_selection",std::string(encoded ? "encoded_b_grouped_g" : "native_grouped_g")+
              std::to_string(group)+(unsigned_inputs ? "_unsigned" : "_signed")}});
      report["timing_scope"]=encoded_warm ?
          "R synchronous private cached GEMMs per batch; one allocation/probe/B conversion reported separately; no upload, poison or verification in interval" : encoded ?
          "one allocation/probe/B conversion, R synchronous private cached GEMMs and cache release per batch; no upload, poison or verification in interval" :
          "R synchronous grouped native GEMMs per batch, including ordinary per-call capability/allocation policy; no upload, poison or verification in interval";
      report["cache_state"]="same immutable native A/B; C poisoned before sampled batches and once before sustained burst";
    }
    if(conversion) {
      const auto resources=query_native_tiled_resources(conversion_config.depth,conversion_config.buffers,65536,
          conversion_config.conversion,group,conversion_config.waves);
      report.update({{"k_depth",conversion_config.depth},{"lds_buffers",conversion_config.buffers},
          {"waves",conversion_config.waves},{"conversion",conversion_name},{"input_change",conversion_input},
          {"reuse_count",reuse_count},{"batch_gemm_calls",reuse_count},
          {"conversion_identity",native_conversion_identity(conversion_config.conversion)},
          {"conversion_prepare_ns",conversion_prepared?nlohmann::json(conversion_prepare_ns):nlohmann::json(nullptr)},
          {"conversion_preparation_in_batch",!conversion_prepared && !conversion_pack},
          {"conversion_prepared_capability_bytes",conversion_pack?0:4},
          {"conversion_public_ownership_measured",false},
          {"conversion_raw_lds_bytes",conversion_pack?0:resources.raw_lds_bytes},
          {"conversion_encoded_lds_bytes",conversion_pack?0:resources.encoded_lds_bytes},
          {"conversion_total_lds_bytes",conversion_pack?0:resources.total_lds_bytes},
          {"conversion_source_counts_measured",false},
          {"conversion_output_tile_rows",conversion_pack?0:resources.output_rows},
          {"conversion_output_tile_cols",conversion_pack?0:resources.output_cols},
          {"conversion_grid_workgroups",conversion_pack?uint64_t(0):((uint64_t(m)+resources.output_rows-1)/resources.output_rows)*
              ((uint64_t(n)+resources.output_cols-1)/resources.output_cols)*((prefix+group-1)/group)},
          {"conversion_scalars_per_interior_panel",conversion_pack?0:resources.conversions_per_panel},
          {"conversion_raw_scalars_per_interior_panel",conversion_pack?0:resources.raw_operand_scalars_per_panel},
          {"conversion_accumulator_i32_per_lane",conversion_pack?0:resources.accumulator_i32_per_lane},
          {"conversion_fragment_words_per_lane",conversion_pack?0:resources.fragment_words_per_lane},
          {"conversion_lds_word_layout",conversion_config.conversion==NativeConversion::encoded_lds ? "A_row_Kword_B_column_Kword_odd_padded_stride" : "not_applicable"},
          {"conversion_dataset","full_width_native64_mt19937_64_seed3135_forced_extrema_v1"},
          {"conversion_input_upload_scope",conversion_input=="reverse_k" ? "alternate_common_K_reversal_before_each_sample_and_once_before_sustained_burst_outside_interval" : "initial_upload_only"}});
      report["timing_scope"]=conversion_pack ? "R resident-native pack-pair plus direct GEMM calls; no upload, poison or verification in interval" :
          conversion_prepared ? "R prepared synchronous conversion GEMMs; one measured preparation reported separately; no upload, poison or verification in interval" :
          "one capability/allocation preparation, R synchronous conversion GEMMs and owner release per batch; no upload, poison or verification in interval";
    }
    if (tiled) {
      report.update({{"k_depth",tiled_depth},{"lds_buffers",tiled_buffers},
          {"tiled_output_rows",tiled_resources.output_rows},{"tiled_output_cols",tiled_resources.output_cols},
          {"tiled_waves",tiled_resources.waves},{"tiled_raw_lds_bytes",tiled_resources.raw_lds_bytes},
          {"tiled_lds_budget_bytes",tiled_lds_budget},{"tiled_resource_query_is_device_qualification",false},
          {"tiled_dataset","full_width_native64_mt19937_64_seed3135_forced_extrema_v1"},
          {"tiled_storage",tiled_buffers ? "raw_native_padded_lds_encoded_fragments_in_registers" : "direct_native_register_loads"},
          {"tiled_prefetch",tiled_buffers == 2 ? "next_disjoint_buffer_loaded_before_current_arithmetic" :
              tiled_buffers == 1 ? "single_buffer_load_barrier_reuse" : "direct_native_register_loads"},
          {"tiled_kernel_selection",std::string("native32x32_w4_g") + std::to_string(group) +
              (unsigned_inputs ? "_unsigned_d" : "_signed_d") + std::to_string(tiled_depth) +
              "_b" + std::to_string(tiled_buffers)},
          {"tiled_control_scope","four_wave_32x32_same_output_residency_reference_and_slab_cap"}});
    }
    if (adaptive) {
      const auto support = query_adaptive_native_support(m,n,k,k,n,n,prefix,group,
          NativeBoundMethod::axis_l1,256,compact ? requested_workers : 0,UINT64_MAX);
      check(support.status);
      report.update({{"adaptive_peak_scratch_bytes", support.peak_scratch_bytes},
          {"adaptive_static_support_version", support.version},
          {"adaptive_static_support_is_device_qualification", false}});
      report.update({{"bound_method", "axis_l1"}, {"adaptive_launch", compact ? "compact_workers" : "dense_grid"},
          {"adaptive_dataset", "full_width_native64_mt19937_64_seed3135_forced_extrema_v1"},
          {"adaptive_dense_group_upper_bound", uint64_t((m+15)/16)*((n+15)/16)*((prefix+group-1)/group)},
          {"adaptive_actual_compact_groups", nullptr}, {"adaptive_compact_count_source", "device_offsets_not_read_back"},
          {"adaptive_worker_policy", compact ? "exact_requested_worker_launch_no_count_clamp" : "dense_grid_control"},
          {"adaptive_kernel_selection", std::string(compact ? "compact_group_" : "grid_group_") + std::to_string(group) +
              (unsigned_inputs ? "_unsigned_axis_l1" : "_signed_axis_l1")},
          {"adaptive_output_scope", "fresh bounds and per-tile prefix; exact suffix completes every capacity20 output plane"}});
      report["timing_scope"] = "resident full-width native inputs; per-call capability, allocation, axis_l1 discovery/compaction, grouped WMMA, exact suffix, staged commit and synchronization; no compact-count readback; upload, poison and complete Boost verification excluded";
    }
    std::cout << report.dump(2) << '\n';
    return 0;
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
