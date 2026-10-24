#include "backend_hip_direct/hip_backend.hpp"
#include "backend_common/instruction_roof_plan.hpp"
#include "core/internal.hpp"
#include <array>
#include <charconv>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <vector>
namespace {
using namespace rns8::detail;
using Clock=std::chrono::steady_clock;
using Json=nlohmann::json;
struct Unsupported:std::runtime_error {using std::runtime_error::runtime_error;};
void check(rns8_status status) {
  if(status==RNS8_UNSUPPORTED_BACKEND) throw Unsupported(rns8_status_string(status));
  if(status!=RNS8_SUCCESS) throw std::runtime_error(rns8_status_string(status));
}
uint64_t number(const char* text,uint64_t low,uint64_t high) {
  uint64_t value=0;const auto end=text+std::strlen(text);const auto parsed=std::from_chars(text,end,value);
  if(parsed.ec!=std::errc{} || parsed.ptr!=end || value<low || value>high) throw std::invalid_argument("numeric argument outside admitted domain");
  return value;
}
template<class Integer> std::string checksum(const std::vector<Integer>& values) {
  uint64_t hash=UINT64_C(14695981039346656037);
  for(auto value:values) for(unsigned byte=0;byte<sizeof(Integer);++byte)
    hash=(hash^uint8_t(uint64_t(value)>>(byte*8)))*UINT64_C(1099511628211);
  std::ostringstream out;out<<std::hex<<std::setfill('0')<<std::setw(16)<<hash;return out.str();
}
struct Buffers {
  int device;void *a=nullptr,*b=nullptr,*output=nullptr,*capability=nullptr;
  explicit Buffers(int selected):device(selected) {}
  Buffers(const Buffers&)=delete;
  // Borrowed kernels may have failed after submission. Never free uncertain work.
  ~Buffers() noexcept {
    if(!a && !b && !output && !capability) return;
    try {if(hip_direct_synchronize(device)!=RNS8_SUCCESS) return;
      for(auto pointer:{capability,output,b,a}) if(pointer) (void)hip_direct_free(device,pointer);
    } catch(...) {} // Process failure retains uncertain allocations until exit.
  }
  void release() {
    for(auto** pointer:{&capability,&output,&b,&a}) if(*pointer) {check(hip_direct_free(device,*pointer));*pointer=nullptr;}
  }
};
}
int main(int argc,char** argv) {
 try {
  const char* usage="usage: rns8-instruction-roof-calibrate DEVICE raw|reduce251 CHAINS ITERATIONS WAVES REPEATS WARMUPS RUN_ID\nDEVICE0..63; chains1/2/4/8; iterations1..4096; wave jobs1..4096; repeats1..10000 warmups1..1000. Explicit RDNA3 WMMA only; no fallback. Event ns per modeled wave instruction is NOT hardware cycles or exact single-instruction latency.";
  if(argc==2 && std::string(argv[1])=="--help") {std::cout<<usage<<'\n';return 0;}
  if(argc!=9) throw std::invalid_argument(usage);
  const int device=int(number(argv[1],0,63));const std::string mode=argv[2],run_id=argv[8];
  InstructionRoofProblem problem;problem.chains=uint32_t(number(argv[3],1,8));problem.iterations=uint32_t(number(argv[4],1,4096));
  problem.waves=uint32_t(number(argv[5],1,4096));problem.reduce_each_iteration=mode=="reduce251";
  const uint32_t repeats=uint32_t(number(argv[6],1,10000)),warmups=uint32_t(number(argv[7],1,1000));
  const auto resources=query_instruction_roof(problem);
  if((mode!="raw" && mode!="reduce251") || resources.status!=RNS8_SUCCESS || run_id.empty() || run_id.size()>128 ||
      run_id.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")!=std::string::npos)
    throw std::invalid_argument("invalid mode/chains/run identity");
  const uint64_t allocation_bytes=resources.output_bytes+64;
  const uint64_t envelope=allocation_bytes*4+6*problem.chains+8+16384;
  constexpr uint64_t budget=UINT64_C(256)*1024*1024;
  if(envelope>budget) throw std::invalid_argument("logical live payload exceeds256MiB");
  const auto oracle_start=Clock::now();
  const std::array<int8_t,8> left{-128,127,-128,1,0,-1,63,-64},right{-128,127,127,-128,127,-1,-64,63};
  std::vector<int8_t> a(left.begin(),left.begin()+problem.chains),b(right.begin(),right.begin()+problem.chains),audit_a(a.size()),audit_b(b.size());
  constexpr int32_t poison_value=0x12345678;
  std::vector<int32_t> gold(allocation_bytes/4,poison_value),observed(gold),poison(gold);
  for(uint32_t chain=0;chain<problem.chains;++chain) {
    int64_t value=0;
    for(uint32_t iteration=0;iteration<problem.iterations;++iteration) {
      // An independent scalar dot: the runtime fragment contains 16 equal terms.
      for(int k=0;k<16;++k) value+=int64_t(a[chain])*b[chain];
      if(problem.reduce_each_iteration) {value%=251;if(value<0) value+=251;if(value>=126) value-=251;}
    }
    if(value<INT32_MIN || value>INT32_MAX) throw std::runtime_error("reference exceeds certified INT32 domain");
    for(uint32_t wave=0;wave<problem.waves;++wave) for(uint32_t cell=0;cell<256;++cell)
      gold[(uint64_t(wave)*problem.chains+chain)*256+cell]=int32_t(value);
  }
  const double oracle_ns=std::chrono::duration<double,std::nano>(Clock::now()-oracle_start).count();
  rns8_device_info info{};check(hip_direct_probe(device,info));hip_direct_timing_set_enabled(false);
  Buffers buffers(device);const auto setup_start=Clock::now();
  check(hip_direct_allocate(device,a.size(),&buffers.a));check(hip_direct_allocate(device,b.size(),&buffers.b));
  check(hip_direct_allocate(device,allocation_bytes,&buffers.output));check(hip_direct_allocate(device,4,&buffers.capability));
  check(hip_direct_copy_host_to_device(device,buffers.a,a.data(),a.size()));check(hip_direct_copy_host_to_device(device,buffers.b,b.data(),b.size()));
  const double setup_ns=std::chrono::duration<double,std::nano>(Clock::now()-setup_start).count();
  const auto prepare=[&] {check(hip_direct_copy_host_to_device(device,buffers.output,poison.data(),allocation_bytes));};
  const auto execute=[&] {check(hip_direct_instruction_roof(device,buffers.a,buffers.b,buffers.output,buffers.capability,problem));};
  const auto verify=[&] {
    check(hip_direct_copy_device_to_host(device,observed.data(),buffers.output,allocation_bytes));
    check(hip_direct_copy_device_to_host(device,audit_a.data(),buffers.a,a.size()));check(hip_direct_copy_device_to_host(device,audit_b.data(),buffers.b,b.size()));
    if(observed!=gold || audit_a!=a || audit_b!=b) throw std::runtime_error("full scalar recurrence/output canary/input audit failed");
  };
  for(uint32_t i=0;i<warmups;++i) {prepare();execute();verify();}
  const std::string label=problem.reduce_each_iteration?"instruction_roof_wmma_reduce251":"instruction_roof_wmma_raw";
  Json samples=Json::array();
  for(uint32_t repeat=0;repeat<repeats;++repeat) {
    prepare();hip_direct_timing_reset();hip_direct_timing_set_enabled(true);auto start=Clock::now();execute();
    const double instrumented=std::chrono::duration<double,std::nano>(Clock::now()-start).count();
    hip_direct_timing_flush_pending_events();Json events=Json::array(),capability_events=Json::array();Json kernel_ns=nullptr;std::string event_state="unavailable";
    for(const auto& event:hip_direct_timing_snapshot()) {
      if(event.label=="residue_d2h_sync") {
        if(!capability_events.empty()) throw std::runtime_error("duplicate capability transfer timing event");
        capability_events.push_back({{"label",event.label},{"elapsed_ns",event.microseconds>0?Json(event.microseconds*1000):Json(nullptr)}});
        continue;
      }
      if(event.label!=label || !events.empty()) throw std::runtime_error("unexpected instruction kernel timing events");
      kernel_ns=event.microseconds>0?Json(event.microseconds*1000):Json(nullptr);
      event_state=event.microseconds>0?"measured":"below_resolution";events.push_back({{"label",event.label},{"elapsed_ns",kernel_ns}});
    }
    hip_direct_timing_set_enabled(false);verify();prepare();start=Clock::now();execute();
    const double production=std::chrono::duration<double,std::nano>(Clock::now()-start).count();verify();
    samples.push_back({{"host_call_ns",instrumented},{"uninstrumented_host_call_ns",production},{"events",events},{"capability_events",capability_events},
      {"kernel_event_ns",kernel_ns},{"kernel_event_state",event_state},
      {"event_ns_per_modeled_wave_instruction",kernel_ns.is_null()?Json(nullptr):Json(kernel_ns.get<double>()/resources.instructions)},
      {"exact_reference_passed",true},{"uninstrumented_exact_reference_passed",true}});
  }
  prepare();uint32_t calls=0;double elapsed=0;auto start=Clock::now();
  do {execute();++calls;elapsed=std::chrono::duration<double,std::nano>(Clock::now()-start).count();}
  while(calls<4096 && (calls<16 || elapsed<250000000));verify();
  Json report={{"schema","rns8_instruction_roof_microcase_v1"},{"run_id",run_id},{"mode",mode},{"chains",problem.chains},{"iterations",problem.iterations},{"waves",problem.waves},
    {"device",device},{"device_requested",device},{"device_actual",device},{"backend_requested","hip-direct"},{"backend_actual","hip-direct"},
    {"device_name",info.name},{"target",info.gcn_arch},{"runtime_version",info.hip_runtime_version},{"build_manifest",Json::parse(RNS8_BUILD_MANIFEST_JSON)},
    {"seed",0},{"dataset_recipe","fixed_signed_runtime_chain_extrema_v1"},{"a_values",a},{"b_values",b},{"a_checksum",checksum(a)},{"b_checksum",checksum(b)},{"output_checksum",checksum(gold)},
    {"repeats",repeats},{"warmups",warmups},{"input_contract","one_runtime_signed_byte_per_chain_replicated_into_uniform_16x16_fragments"},
    {"output_contract","wave_chain_row_major_int32_cells_then16_int32_canaries"},{"reference_method","int64_full_16_term_recurrence_every_iteration_all_cells"},
    {"semantic_contract",problem.reduce_each_iteration?"signed_int8_wmma_then_centered_mod251_each_iteration":"exact_signed_int8_wmma_int32_accumulation"},
    {"k_equivalent",uint64_t(problem.iterations)*16},{"maximum_accumulator_magnitude",resources.maximum_accumulator_magnitude},
    {"modeled_wave_instructions",resources.instructions},{"modeled_integer_operations",resources.integer_operations},{"modeled_output_cells",resources.output_bytes/4},
    {"operation_count_contract","two_operations_per_int8_mac_16cubed_per_wave_instruction_excludes_reduction_probe_and_stores"},
    {"accumulator_bound_scope",problem.reduce_each_iteration?"before_each_reduction":"entire_raw_chain"},
    {"modeled_output_store_bytes",resources.output_bytes},{"input_bytes_each",resources.input_bytes},{"output_allocation_bytes",allocation_bytes},{"capability_bytes",4},{"canary_bytes",64},
    {"logical_payload_envelope_bytes",envelope},{"memory_budget_bytes",budget},{"oracle_and_dataset_ns",oracle_ns},{"resident_setup_ns",setup_ns},
    {"threads_per_block",128},{"waves_per_block",4},{"grid_blocks",(problem.waves+3)/4},{"warp_width",32},
    {"kernel_event_label",label},{"timer","std_chrono_steady_clock_nanoseconds"},{"host_timing_scope","resident_checked_call_including_capability_probe_status_transfer_and_required_sync"},
    {"event_timing_scope","chain_kernel_only_including_fragment_setup_loop_reduction_if_requested_and_all_output_stores"},
    {"normalization_contract","kernel_event_ns_divided_by_modeled_wave_instructions_not_hardware_cycles_or_isolated_instruction_latency"},
    {"actual_issued_instructions",nullptr},{"observed_resident_waves",nullptr},{"achieved_occupancy",nullptr},{"hardware_clock_hz",nullptr},{"hardware_cycle_latency",nullptr},
    {"compiler_private_scratch_bytes",nullptr},{"actual_hbm_bytes",nullptr},{"timing_resolution_ns",nullptr},{"isa_sha256",nullptr},{"isa_verified",false},{"promotion_eligible",false},
    {"production_timing_hooks_enabled",false},{"samples",samples},{"sustained",{{"calls",calls},{"elapsed_ns",elapsed},{"mean_call_ns",elapsed/calls},
      {"minimum_calls",16},{"maximum_calls",4096},{"target_ns",250000000},{"independent_audit_inside_interval",false},{"final_exact_reference_passed",true}}}};
  const auto cleanup_start=Clock::now();buffers.release();report["resource_release_ns"]=std::chrono::duration<double,std::nano>(Clock::now()-cleanup_start).count();
  std::cout<<report.dump(2)<<'\n';return 0;
 } catch(const Unsupported& error) {hip_direct_timing_set_enabled(false);std::cerr<<error.what()<<'\n';return 77;}
 catch(const std::exception& error) {hip_direct_timing_set_enabled(false);std::cerr<<error.what()<<'\n';return 1;}
}
