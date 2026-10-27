// Explicit physical capacity gate. Never part of an automatic benchmark sweep.
#include "stream_capacity_case.hpp"
#include <rns8/moduli.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__linux__)
#include <unistd.h>
#endif

namespace {
using namespace rns8::bench::capacity;
using json=nlohmann::json;
using Clock=std::chrono::steady_clock;
using boost::multiprecision::cpp_int;
constexpr int64_t canary=INT64_C(0x5a5a5a5a5a5a5a5a);
bool cleanup_failed=false;
struct Unrun:std::runtime_error {using std::runtime_error::runtime_error;};
void require(bool value,const char* text){if(!value)throw std::runtime_error(text);}
void check(rns8_status status){if(status!=RNS8_SUCCESS)throw std::runtime_error(rns8_status_string(status));}
template<class T,rns8_status(*Destroy)(T*)>struct Delete {
  void operator()(T* value)const noexcept {if(value && Destroy(value)!=RNS8_SUCCESS)cleanup_failed=true;}
};
using Context=std::unique_ptr<rns8_context,Delete<rns8_context,rns8_destroy_context>>;
using Stream=std::unique_ptr<rns8_stream_gemm_plan,Delete<rns8_stream_gemm_plan,rns8_destroy_stream_gemm>>;
uint64_t number(const char* text,uint64_t maximum) {
  uint64_t value=0;const auto end=text+std::strlen(text);const auto parsed=std::from_chars(text,end,value);
  require(parsed.ec==std::errc{} && parsed.ptr==end && value<=maximum,"invalid decimal budget/device");return value;
}
uint64_t available_host_bytes() {
#if defined(_WIN32)
  MEMORYSTATUSEX info{};info.dwLength=sizeof(info);
  if(!GlobalMemoryStatusEx(&info))throw Unrun("available host RAM query failed");
  return info.ullAvailPhys;
#elif defined(__linux__)
  const auto pages=sysconf(_SC_AVPHYS_PAGES),page_size=sysconf(_SC_PAGESIZE);
  if(pages<=0 || page_size<=0 || uint64_t(pages)>UINT64_MAX/uint64_t(page_size))throw Unrun("available host RAM query failed");
  return uint64_t(pages)*uint64_t(page_size);
#else
  throw Unrun("this platform has no qualified host-RAM admission query");
#endif
}
json resources(const rns8_stream_gemm_resources& r) {
  return {{"device_bytes",r.device_bytes},{"host_bytes",r.host_bytes},{"pinned_host_bytes",r.pinned_host_bytes},
      {"ordinary_host_bytes",r.ordinary_host_bytes},{"control_reservation_bytes",r.control_reservation_bytes},
      {"source_a_span_bytes",r.source_a_span_bytes},{"source_b_span_bytes",r.source_b_span_bytes},
      {"destination_span_bytes",r.destination_span_bytes},{"output_tiles",r.output_tiles},
      {"k_panels_per_tile",r.k_panels_per_tile},{"stream_count",r.stream_count},{"event_count",r.event_count}};
}
rns8_stream_gemm_resources query(const rns8_stream_gemm_desc& d) {
  rns8_stream_gemm_resources r{};r.struct_size=sizeof(r);r.abi_version=RNS8_ABI_VERSION;
  check(rns8_query_stream_gemm(&d,&r));return r;
}
rns8_allocation_observation observe(const Context& ctx) {
  rns8_allocation_observation value{};value.struct_size=sizeof(value);value.abi_version=RNS8_ABI_VERSION;
  check(rns8_get_allocation_observation(ctx.get(),&value));return value;
}
void help() {
  std::cout<<"rns8-stream-capacity-qualify DEVICE serial|overlap CALLER_BYTES DEVICE_BYTES LIBRARY_HOST_BYTES RUN_ID\n"
      "Explicit >32GiB native input and >36GiB full-A RNS capacity gate; no default GPU run.\n"
      "Fixed signed M=65537,N=3,K=65537,P9, panels256x4x4096; complete periodic Boost oracle.\n"
      "Suggested budgets: caller42949672960 device67108864 library-host67108864 bytes.\n"
      "Requires available host RAM for caller+library-host+8GiB reserve and a device smaller than full A.\n"
      "Runs cold, unchanged-generation warm, then changed-A-generation calls. Capacity evidence, no crossover/promotion.\n"
      "--describe queries both schedules without host allocation, device probing or execution.\n";
}
json run(int argc,char** argv) {
  require(argc==7,"expected six positional arguments; use --help");
  const int device_id=int(number(argv[1],63));const std::string mode=argv[2],run_id=argv[6];
  require(mode=="serial" || mode=="overlap","schedule must be serial or overlap");
  require(!run_id.empty() && run_id.size()<=128 && run_id.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")==std::string::npos,"invalid run ID");
  const auto caller_budget=number(argv[3],uint64_t{128}<<30),device_budget=number(argv[4],uint64_t{1}<<30),host_budget=number(argv[5],uint64_t{1}<<30);
  if(caller_bytes>uint64_t(std::numeric_limits<std::ptrdiff_t>::max()) || a_bytes>uint64_t(std::numeric_limits<size_t>::max()))
    throw Unrun("fixed caller layout exceeds this host's object/address domain");
  if(caller_budget<caller_bytes)throw Unrun("explicit caller budget cannot fit the fixed physical dataset");
  const uint64_t available=available_host_bytes();
  if(available<caller_bytes+host_budget+host_reserve_bytes)throw Unrun("insufficient available host RAM plus 8GiB reserve");
  const auto d=problem(mode=="serial"?RNS8_STREAM_GEMM_SERIAL:RNS8_STREAM_GEMM_OVERLAPPED,device_budget,host_budget);
  const auto prepare_start=Clock::now();const auto r=query(d);
  json moduli=json::array();cpp_int modulus_product=1;uint32_t minimum_prefix=0;
  for(uint32_t p=0;p<prefix;++p) {
    const auto modulus=rns8_default_modulus(p);moduli.push_back(modulus);modulus_product*=modulus;
    if(!minimum_prefix && modulus_product>cpp_int(bound)*2)minimum_prefix=p+1;
  }
  require(minimum_prefix>0,"fixed basis does not admit the signed complete-expression bound");
  const uint64_t minimum_full_a_bytes=uint64_t(m)*k*minimum_prefix;
  rns8_context_options options{};options.struct_size=sizeof(options);options.abi_version=RNS8_ABI_VERSION;options.requested_backend=RNS8_BACKEND_HIP_DIRECT;
  rns8_context* raw=nullptr;const auto status=rns8_create_context(device_id,&options,&raw);
  if(status==RNS8_UNSUPPORTED_BACKEND || status==RNS8_UNSUPPORTED_ARCH || status==RNS8_UNSUPPORTED_OS)throw Unrun(rns8_status_string(status));
  check(status);Context ctx(raw);
  rns8_device_info device{};device.struct_size=sizeof(device);device.abi_version=RNS8_ABI_VERSION;check(rns8_get_device_info(ctx.get(),&device));
  require(device.backend==RNS8_BACKEND_HIP_DIRECT && device.device_id==device_id && device.hip_available && device.global_mem_bytes>0,"missing actual HIP target identity");
  if(a_bytes<=device.global_mem_bytes || minimum_full_a_bytes<=device.global_mem_bytes)throw Unrun("this fixed dataset/minimum proved basis does not exceed actual total device memory");
  const auto before=observe(ctx);uint64_t epoch=0;check(rns8_reset_allocation_peaks(ctx.get(),&epoch));
  rns8_stream_gemm_plan* created=nullptr;check(rns8_create_stream_gemm(ctx.get(),&d,&created));Stream owner(created);
  const auto setup_ns=std::chrono::duration<double,std::nano>(Clock::now()-prepare_start).count();
  // Allocate complete caller buffers only after bounded GPU/library preparation.
  const auto fill_start=Clock::now();
  std::unique_ptr<int64_t[]> a;
  std::vector<int64_t> b,output,gold,patterns;
  try {
    a.reset(new int64_t[size_t(a_bytes/8)]);b.assign(size_t(b_bytes/8),canary);
    output.assign(size_t(c_bytes/8),canary);gold.resize(size_t(oracle_bytes/8));patterns.assign(size_t(pattern_bytes/8),canary);
  }catch(const std::bad_alloc&) {throw Unrun("caller input/reference allocation unavailable before execution");}
  const uint64_t actual_caller_capacity=a_bytes+8*uint64_t(b.capacity()+output.capacity()+gold.capacity()+patterns.capacity());
  require(actual_caller_capacity<=caller_budget,"actual caller vector capacities exceed the explicit budget");
  for(int64_t row=0;row<15;++row)for(int64_t q=0;q<k;++q)patterns[size_t(row*lda+q)]=native_base+(3*row+7*q)%15-7;
  for(int64_t row=0;row<m;++row)std::copy_n(patterns.data()+size_t((row%15)*lda),size_t(lda),a.get()+size_t(row*lda));
  for(int64_t q=0;q<k;++q)for(int64_t col=0;col<n;++col)b[size_t(q*ldb+col)]=(col%2?1:-1)*(native_base+(11*q+5*col+3)%15-7);
  std::array<int64_t,45> oracle{};
  for(int64_t row=0;row<15;++row)for(int64_t col=0;col<n;++col)oracle[size_t(row*n+col)]=reference(row,col);
  for(int64_t row=0;row<m;++row)for(int64_t col=0;col<n;++col)gold[size_t(row*n+col)]=oracle[size_t((row%15)*n+col)];
  const auto fill_ns=std::chrono::duration<double,std::nano>(Clock::now()-fill_start).count();
  require(a[0]!=a[size_t((m-1)*lda)],"capacity witness must not have identical first/last rows");
  rns8_stream_gemm_result last{};last.struct_size=sizeof(last);last.abi_version=RNS8_ABI_VERSION;
  json phases=json::array();
  for(unsigned call=0;call<3;++call) {
    const uint64_t generation=call==2?2:1;
    if(call==2){++a[0];for(int64_t col=0;col<n;++col)gold[size_t(col)]+=b[size_t(col)];}
    std::fill(output.begin(),output.end(),canary);
    const auto start=Clock::now();check(rns8_execute_stream_gemm(owner.get(),a.get(),b.data(),output.data(),101,generation,202,1,&last));
    const double elapsed=std::chrono::duration<double,std::nano>(Clock::now()-start).count();
    require(last.status==RNS8_SUCCESS && last.epoch==call+1 && last.output_started==1 && last.output_valid==1 && !last.quarantined &&
        last.completed_tiles==r.output_tiles && last.completed_panels==r.output_tiles*r.k_panels_per_tile,"incomplete streaming coverage/epoch");
    cpp_int checksum=0;
    for(int64_t row=0;row<m;++row) {
      for(int64_t col=0;col<ldc;++col)require(output[size_t(row*ldc+col)]==(col<n?gold[size_t(row*n+col)]:canary),"full exact output/padding mismatch");
      for(int64_t q=k;q<lda;++q)require(a[size_t(row*lda+q)]==canary,"input A padding changed");
      for(int64_t col=0;col<n;++col)checksum+=cpp_int(output[size_t(row*ldc+col)])*(row*n+col+1);
    }
    for(int64_t q=0;q<k;++q)for(int64_t col=n;col<ldb;++col)require(b[size_t(q*ldb+col)]==canary,"input B padding changed");
    const auto peak=observe(ctx);
    require(peak.device.peak_bytes>=before.device.live_bytes && peak.pinned_host.peak_bytes>=before.pinned_host.live_bytes,"invalid allocation peak baseline");
    const uint64_t device_peak=peak.device.peak_bytes-before.device.live_bytes,pinned_peak=peak.pinned_host.peak_bytes-before.pinned_host.live_bytes;
    require(device_peak>=r.device_bytes && pinned_peak>=r.pinned_host_bytes && device_peak<=device_budget &&
        pinned_peak<=host_budget && r.ordinary_host_bytes<=host_budget-pinned_peak,"observed requested payload does not reconcile with query/budgets");
    phases.push_back({{"source_a_generation",generation},{"epoch",last.epoch},{"completed_host_call_ns",elapsed},
        {"checked_outputs",uint64_t(m)*n},{"all_output_and_padding_exact",true},{"weighted_checksum",checksum.str()},
        {"completed_tiles",last.completed_tiles},{"completed_panels",last.completed_panels},
        {"h2d_payload_bytes",last.h2d_bytes},{"d2h_payload_bytes",last.d2h_bytes},
        {"h2d_transfers",last.h2d_transfers},{"d2h_transfers",last.d2h_transfers},
        {"observed_requested_device_peak_delta",device_peak},{"observed_requested_pinned_peak_delta",pinned_peak}});
  }
  const auto release_start=Clock::now();const auto destroyed=rns8_destroy_stream_gemm(owner.get());
  if(destroyed==RNS8_SUCCESS) {
    owner.release();
  }
  check(destroyed);
  const double release_ns=std::chrono::duration<double,std::nano>(Clock::now()-release_start).count();const auto retired=observe(ctx);
  require(retired.device.live_bytes==before.device.live_bytes && retired.pinned_host.live_bytes==before.pinned_host.live_bytes &&
      retired.device.live_allocations==before.device.live_allocations && retired.pinned_host.live_allocations==before.pinned_host.live_allocations,"stream owner payload did not retire to baseline");
  return {{"schema","rns8_stream_physical_capacity_v1"},{"run_id",run_id},{"mode",mode},{"target",device.gcn_arch},
      {"device",device.name},{"device_id",device_id},{"hip_runtime_version",device.hip_runtime_version},{"hip_driver_version",device.hip_driver_version},
      {"actual_total_device_bytes",device.global_mem_bytes},{"m",m},{"n",n},{"k",k},{"prefix",prefix},
      {"semantic_contract","bounded_i64_explicit_k_amax_bmax"},{"max_abs_a",maximum},{"max_abs_b",maximum},
      {"full_expression_abs_bound",bound},{"ordered_moduli",moduli},{"basis_product_decimal",modulus_product.str()},
      {"minimum_proven_prefix",minimum_prefix},{"minimum_full_a_residue_bytes",minimum_full_a_bytes},
      {"caller_payload_bytes",caller_bytes},{"actual_caller_capacity_bytes",actual_caller_capacity},
      {"caller_budget_bytes",caller_budget},{"available_host_bytes_before",available},
      {"host_reserve_bytes",host_reserve_bytes},{"a_native_allocation_bytes",a_bytes},{"full_a_residue_bytes",whole_a_residue_bytes},
      {"device_budget_bytes",device_budget},{"library_host_budget_bytes",host_budget},{"queried_resources",resources(r)},
      {"setup_ns",setup_ns},{"setup_plus_first_completed_call_ns",setup_ns+phases[0]["completed_host_call_ns"].get<double>()},
      {"owner_release_ns",release_ns},
      {"input_generation_and_oracle_ns",fill_ns},{"dataset","affine_native_base2p20_mod15_signed_columns_v1"},
      {"reference","independent_boost_period_sum_all_outputs_plus_output_input_padding"},{"phases",phases},
      {"all_owned_device_and_pinned_payload_retired",true},{"build_manifest",json::parse(RNS8_BUILD_MANIFEST_JSON)},
      {"timer","steady_clock_completed_public_host_call"},{"independent_oracle_inside_timer",false},
      {"nominal_clock_tick_ns",1e9*double(Clock::period::num)/double(Clock::period::den)},
      {"timing_includes","required_full_input_bound_scan_host_pack_upload_compute_merge_export_and_sync"},
      {"scope","physical_capacity_three_calls_not_a_crossover_or_achieved_overlap_measurement"},
      {"allocation_observation_scope","process_device_RNS8_owned_requested_capacities_excluding_driver_allocator_padding_external_buffers_and_ordinary_host_RSS"},
      {"caller_capacity_scope","native_inputs_output_period_table_and_full_reference_output_vector_capacities_excluding_allocator_bookkeeping_and_fixed_stack"},
      {"kernel_events",nullptr},{"achieved_overlap_trace",nullptr},{"process_rss_measured",false},{"promotion_eligible",false}};
}
} // namespace

int main(int argc,char** argv) {
  try {
    if(argc==2 && std::string(argv[1])=="--help"){help();return 0;}
    if(argc==2 && std::string(argv[1])=="--describe") {
      const auto serial=query(problem(RNS8_STREAM_GEMM_SERIAL,uint64_t{64}<<20,uint64_t{64}<<20));
      const auto overlap=query(problem(RNS8_STREAM_GEMM_OVERLAPPED,uint64_t{64}<<20,uint64_t{64}<<20));
      std::cout<<json({{"schema","rns8_stream_capacity_description_v1"},{"caller_bytes",caller_bytes},
          {"a_native_bytes",a_bytes},{"full_a_residue_bytes",whole_a_residue_bytes},
          {"serial",resources(serial)},{"overlap",resources(overlap)},{"gpu_execution",false}}).dump()<<'\n';return 0;
    }
    auto result=run(argc,argv);require(!cleanup_failed,"handle cleanup failed");
    result["command_line"]=json::array();for(int i=0;i<argc;++i)result["command_line"].push_back(argv[i]);
    std::cout<<result.dump()<<'\n';return 0;
  }catch(const Unrun& e){std::cerr<<"capacity gate unrun: "<<e.what()<<'\n';return cleanup_failed?1:77;}
  catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
