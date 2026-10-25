// Private raw-pointer scheduling experiment, deliberately not a public cache.
#include "backend_hip_direct/native_wmma_prepared.hpp"
#include "backend_hip_direct/encoded_b_backend.hpp"
#include "backend_common/native_wmma_queue_plan.hpp"
#include "core/internal.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
using namespace rns8::detail;
using json=nlohmann::json;
using boost::multiprecision::cpp_int;
using Clock=std::chrono::steady_clock;
namespace {
void need(bool ok,const char* why){if(!ok)throw std::invalid_argument(why);}
void check(rns8_status s){if(s!=RNS8_SUCCESS)throw std::runtime_error(rns8_status_string(s));}
uint64_t ns(Clock::time_point t){return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-t).count());}
uint32_t number(const json& j,const char* key,uint32_t lo,uint32_t hi) {
  need(j.contains(key)&&j.at(key).is_number_integer(),"missing integer control");
  const auto n=j.at(key).get<int64_t>();need(n>=lo&&uint64_t(n)<=hi,"control outside calibration domain");return uint32_t(n);
}
cpp_int value(uint64_t bits,bool uns){if(uns)return cpp_int(bits);int64_t s;std::memcpy(&s,&bits,8);return cpp_int(s);}
struct TaskStorage {
  hip_direct_device_buffer a,b,c,encoded;
  std::vector<uint64_t> host_a,host_b;
  std::vector<int8_t> expected,actual;
  NativeWmmaTask task;
};
json run(const json& input,const std::string& mode,const std::string& run_id) {
  need(input.size()==10,"unknown case controls");
  need(input.at("schema")=="native_queue_case_v1","unknown case ABI");
  need(mode=="explicit"||mode=="task-major"||mode=="fair"||mode=="ordinary","unknown schedule");
  need(!run_id.empty()&&run_id.size()<=128&&run_id.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")==std::string::npos,"invalid run ID");
  const auto workers=number(input,"workers",1,1024),chunk=number(input,"chunk",1,32);
  const auto reuse=number(input,"reuse",1,256),repeats=number(input,"repeats",3,1000),warmups=number(input,"warmups",1,1000);
  const uint64_t budget=number(input,"metadata_budget_mib",1,1024)*UINT64_C(1048576);
  const bool encoded=input.at("encoded_b").get<bool>(),mutate_b=input.at("mutate_b").get<bool>();
  const auto specs=input.at("tasks");need(specs.is_array()&&!specs.empty()&&specs.size()<=4096,"task limit");
  uint64_t reference_work=0,payload=0;
  for(const auto& spec:specs) {
    need(spec.size()==6,"unknown task controls");
    const uint64_t m=number(spec,"m",1,1024),n=number(spec,"n",1,1024),k=number(spec,"k",1,65537),p=number(spec,"prefix",1,20);
    const auto g=number(spec,"group",1,4);need(g!=3,"group must be1,2,4");(void)spec.at("unsigned").get<bool>();
    reference_work+=m*n*k;payload+=8*(m*(k+1)+k*(n+2))+(p+1)*m*(n+3)+uint64_t(encoded)*p*n*((k+15)/16)*16+4;
  }
  need(reference_work<=UINT64_C(134217728)&&payload<=UINT64_C(536870912),"reference or payload envelope exceeds bounded experiment");
  rns8_device_info device{};check(hip_direct_probe(0,device));
  hip_direct_timing_set_enabled(false);
  std::vector<TaskStorage> storage(specs.size());std::vector<NativeWmmaTask> tasks;
  std::mt19937_64 rng(3838);uint64_t checksum=UINT64_C(14695981039346656037);
  const auto setup_start=Clock::now();
  for(size_t i=0;i<specs.size();++i) {
    auto& s=storage[i];const auto& spec=specs[i];auto& t=s.task;
    t.m=spec.at("m");t.n=spec.at("n");t.k=spec.at("k");t.prefix=spec.at("prefix");t.group=spec.at("group");t.unsigned_inputs=spec.at("unsigned").get<bool>();
    t.lda=t.k+1;t.ldb=t.n+2;t.ldc=t.n+3;
    s.host_a.resize(t.m*t.lda);s.host_b.resize(t.k*t.ldb);s.expected.resize((t.prefix+1)*t.m*t.ldc,99);s.actual=s.expected;
    for(auto& x:s.host_a)x=rng();for(auto& x:s.host_b)x=rng();s.host_a[0]=uint64_t{1}<<63;s.host_b[0]=UINT64_MAX;
    for(auto x:s.host_a)checksum=(checksum^x)*UINT64_C(1099511628211);
    for(auto x:s.host_b)checksum=(checksum^x)*UINT64_C(1099511628211);
    check(s.a.allocate(0,s.host_a.size()*8));check(s.b.allocate(0,s.host_b.size()*8));check(s.c.allocate(0,s.actual.size()));
    check(hip_direct_copy_host_to_device(0,s.a.get(),s.host_a.data(),s.host_a.size()*8));
    check(hip_direct_copy_host_to_device(0,s.b.get(),s.host_b.data(),s.host_b.size()*8));
    check(hip_direct_copy_host_to_device(0,s.c.get(),s.actual.data(),s.actual.size()));
    t.a=s.a.get();t.b=s.b.get();t.c=static_cast<int8_t*>(s.c.get());
    if(encoded) {
      auto layout=encoded_b_layout(t.k,t.n,t.ldb,t.prefix,UINT64_MAX);check(layout.status);
      check(s.encoded.allocate(0,layout.total_bytes));
      check(hip_prepare_encoded_b(0,t.b,s.encoded.get(),t.k,t.n,t.ldb,t.prefix,t.unsigned_inputs,layout.total_bytes));
      t.encoded_b=static_cast<const uint32_t*>(s.encoded.get());t.encoded_k_words=layout.k_words;
    }
    tasks.push_back(t);
  }
  const auto input_setup_ns=ns(setup_start);
  uint64_t oracle_ns=0,checked_cells=0;
  const auto reference=[&](){const auto start=Clock::now();for(auto& s:storage) {
    const auto& t=s.task;
    for(int64_t row=0;row<t.m;++row)for(int64_t col=0;col<t.n;++col) {
      cpp_int sum=0;for(int64_t q=0;q<t.k;++q)sum+=value(s.host_a[row*t.lda+q],t.unsigned_inputs)*value(s.host_b[q*t.ldb+col],t.unsigned_inputs);
      for(uint32_t p=0;p<t.prefix;++p)s.expected[(p*t.m+row)*t.ldc+col]=centered_residue(sum,kDefaultModuli[p]);
    }
  }oracle_ns+=ns(start);};
  reference();
  const auto audit=[&](){for(auto& s:storage){check(hip_direct_copy_device_to_host(0,s.actual.data(),s.c.get(),s.actual.size()));need(s.actual==s.expected,"exact residue/padding/extra-plane mismatch");checked_cells+=s.actual.size();}};
  NativeWmmaScheduleConfig config{1,mode=="task-major"?NativeWmmaScheduleMode::task_major:mode=="fair"?NativeWmmaScheduleMode::compact_round_robin:NativeWmmaScheduleMode::explicit_round_robin,chunk};
  uint64_t build_ns=0,upload_bytes=0,plan_capacity=0;uint32_t work=0;
  {NativeWmmaSchedulePlan plan;auto start=Clock::now();check(build_native_wmma_schedule(tasks,workers,budget,config,plan));build_ns=ns(start);upload_bytes=plan.device_bytes-4;plan_capacity=plan.transient_host_capacity_bytes();work=plan.header.work_count;}
  NativeWmmaPreparedQueue queue;auto start=Clock::now();check(queue.prepare(0,tasks,workers,budget,config));const auto prepare_ns=ns(start);
  const auto invoke=[&](){for(uint32_t i=0;i<reuse;++i){bool touched=false;check(mode=="ordinary"?queue.execute_ordinary(touched):queue.execute(touched));need(touched,"successful call did not write output");}};
  start=Clock::now();invoke();const auto first_ns=ns(start);audit();
  uint64_t generation=1;
  const auto rebuild=[&]()->uint64_t {
    if(!mutate_b)return 0;
    ++generation;
    for(auto& s:storage)for(auto& x:s.host_b)x^=UINT64_C(0x9e3779b97f4a7c15)+generation;
    reference(); // independent full reference remains outside all measured spans
    auto begin=Clock::now();for(auto& s:storage){const auto& t=s.task;
      check(hip_direct_copy_host_to_device(0,s.b.get(),s.host_b.data(),s.host_b.size()*8));
      if(encoded)check(hip_prepare_encoded_b(0,t.b,s.encoded.get(),t.k,t.n,t.ldb,t.prefix,t.unsigned_inputs,s.encoded.bytes()));
    }return ns(begin);
  };
  for(uint32_t i=0;i<warmups;++i){rebuild();invoke();audit();}
  json samples=json::array();std::vector<uint64_t> timings;
  for(uint32_t i=0;i<repeats;++i){const auto rebuild_ns=rebuild();hip_direct_allocation_counters_reset();start=Clock::now();invoke();const auto elapsed=ns(start);const auto counters=hip_direct_allocation_counters_snapshot();
    need(!counters.allocate_calls&&!counters.free_calls&&!counters.h2d_calls&&!counters.d2h_calls,"prepared execute unexpectedly allocated or transferred host payload");audit();timings.push_back(elapsed);
    samples.push_back({{"batch_ns",elapsed},{"source_generation",generation},{"upload_encode_rebuild_ns",rebuild_ns},{"rebuild_plus_batch_ns",rebuild_ns+elapsed},{"exact_reference_passed",true}});}
  // A separate diagnostic program adds completion atomics and a device epoch.
  // It never supplies production timing; ordinary grids have no such kernel.
  json completion=nullptr;
  if(mode!="ordinary") {
    NativeWmmaCompletionTelemetry receipt;bool touched=false;check(queue.execute_completion(receipt,touched));audit();
    std::vector<double> micros;for(auto tick:receipt.completion_ticks)micros.push_back(double(tick)*1000/receipt.wall_clock_rate_khz);
    auto ordered=micros;std::sort(ordered.begin(),ordered.end());
    completion={{"completed_work",receipt.completed_work},{"completion_ticks",receipt.completion_ticks},{"wall_clock_rate_khz",receipt.wall_clock_rate_khz},
      {"completion_us",micros},{"p50_task_completion_us",ordered[(ordered.size()-1)/2]},{"p95_task_completion_us",ordered[(ordered.size()*95+99)/100-1]},
      {"max_task_completion_us",ordered.back()},{"additional_device_bytes",receipt.diagnostic_device_bytes},
      {"scope","separate_general_dispatch_diagnostic_submission_from_prior_kernel_epoch_not_production_latency"}};
  }
  std::sort(timings.begin(),timings.end());const auto median=timings[timings.size()/2];
  json events=json::array(); // explicit event instrumented companion, not host production sample
  hip_direct_timing_set_enabled(true);hip_direct_timing_reset();invoke();hip_direct_timing_set_enabled(false);audit();
  for(const auto& event:hip_direct_timing_snapshot())events.push_back({{"phase",event.label},{"microseconds",event.microseconds}});
  return {{"schema","rns8_native_queue_microcase_v1"},{"promotion_eligible",false},{"run_id",run_id},{"case",input},{"schedule",mode},
    {"schedule_identity",mode=="ordinary"?"ordinary_grids_same_preparation":native_wmma_schedule_identity(config.mode)},
    {"build_manifest",json::parse(RNS8_BUILD_MANIFEST_JSON)},{"target",device.gcn_arch},{"device_name",device.name},{"hip_runtime_version",device.hip_runtime_version},{"hip_driver_version",device.hip_driver_version},
    {"input_recipe","fullwidth_seed3838_boundary_first_mutate_xor_generation_v1"},{"initial_input_checksum",std::to_string(checksum)},
    {"metadata",{{"device_retained_payload_bytes",queue.device_metadata_bytes()},{"device_preparation_peak_payload_bytes",queue.preparation_peak_device_metadata_bytes()},
      {"host_retained_capacity_bytes",queue.retained_host_metadata_bytes()},{"host_upload_stage_materialized_payload_bytes",queue.preparation_materialized_host_payload_bytes()},
      {"host_plan_capacity_bytes",plan_capacity},{"host_preparation_peak_payload_bytes",queue.preparation_peak_host_payload_bytes()},{"host_heap_peak_bytes",nullptr},{"descriptor_upload_payload_bytes",upload_bytes},
      {"exclusions","allocator_bookkeeping_object_headers_and_RSS_not_measured; device_payload_not_driver_residency"}}},
    {"input_setup_ns",input_setup_ns},{"cpu_schedule_construction_ns",build_ns},{"prepared_setup_ns",prepare_ns},{"first_batch_ns",first_ns},
    {"setup_plus_first_batch_ns",input_setup_ns+prepare_ns+first_ns},{"repeats",repeats},{"warmups",warmups},{"samples",samples},
    {"host_instrumentation_state","hooks_disabled"},{"completion_boundary","all_centered_residue_outputs_device_resident_default_stream_synchronized"},
    {"work_assignments_per_call",work},{"useful_tile_channel_groups_per_second",double(work)*reuse*1e9/median},{"median_batch_ns",median},
    {"completion_diagnostic",completion},{"gpu_event_diagnostic",events},{"oracle_ns",oracle_ns},{"checked_residue_and_canary_bytes",checked_cells},
    {"exact_reference_passed",true},{"oracle_inside_production_interval",false},{"generation_owner","private_harness_exclusive_raw_pointer_and_synchronous_encoded_rebuild_not_public_matrix_cache"}};
}
}
int main(int argc,char** argv) {
  try {
    if(argc==2&&std::string(argv[1])=="--help"){std::cout<<"rns8-queue-calibrate CASE.json explicit|task-major|fair|ordinary UNIQUE_RUN_ID\n";return 0;}
    need(argc==4,"expected CASE.json MODE RUN_ID");std::ifstream file(argv[1]);need(bool(file),"cannot open case");json input;file>>input;
    std::cout<<run(input,argv[2],argv[3]).dump()<<'\n';return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
