#include "backend_common/system_roof_plan.hpp"
#include "core/hip_resources.hpp"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <thread>

namespace {
namespace s=rns8::detail::system_roof;
using Json=nlohmann::json;using Clock=std::chrono::steady_clock;
double ns(Clock::time_point start){return std::chrono::duration<double,std::nano>(Clock::now()-start).count();}
uint64_t integer(const Json& j,const char* key,uint64_t lo,uint64_t hi) {
  if(!j.at(key).is_number_unsigned())throw std::invalid_argument(key);
  const auto value=j.at(key).get<uint64_t>();if(value<lo || value>hi)throw std::invalid_argument(key);return value;
}
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
extern "C" int rns8_hip_system_roof(uint32_t,const uint32_t*,uint32_t*,uint64_t,uint32_t,uint32_t,uint32_t,void*);
extern "C" int rns8_hip_system_roof_displace(uint32_t*,uint64_t,void*);
struct Unsupported:std::runtime_error {using std::runtime_error::runtime_error;};
void check(hipError_t code) {
  if(code==hipErrorNotSupported)throw Unsupported(hipGetErrorString(code));
  if(code!=hipSuccess)throw std::runtime_error(hipGetErrorString(code));
}
void check(rns8_status code){if(code!=RNS8_SUCCESS)throw std::runtime_error(rns8_status_string(code));}
struct Owner {
  struct Sample {double host_ns=0,event_ns=0;};
  int device=0;s::Problem problem;s::Resources resources;
  rns8::detail::hip_direct_device_buffer input,output,displacement;
  rns8::detail::hip_pinned_host_buffer host_input,host_output;
  hipStream_t stream=nullptr;
  hipEvent_t start=nullptr,stop=nullptr,marker=nullptr;
  hipGraph_t graph=nullptr;hipGraphExec_t executable=nullptr;
  uint64_t displacement_calls=0,calls=0;
  uint64_t output_sum=0,source_sum=0;
  uint32_t* a(){return static_cast<uint32_t*>(input.get());}
  uint32_t* b(){return static_cast<uint32_t*>(output.get());}
  void initialize() {
    check(hipSetDevice(device));check(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));
    const auto bytes=size_t(problem.words*4);
    check(input.allocate(device,bytes));check(output.allocate(device,bytes+64));
    check(host_input.ensure(device,bytes,true));check(host_output.ensure(device,size_t(resources.pinned_bytes)-bytes,true));
    for(uint64_t i=0;i<problem.words;++i)static_cast<uint32_t*>(host_input.get())[i]=s::seed(i);
    check(hipMemcpyAsync(a(),host_input.get(),bytes,hipMemcpyHostToDevice,stream));
    check(hipMemsetAsync(b(),0xcc,bytes+64,stream));
    if(problem.displacement_words) {
      check(displacement.allocate(device,size_t(problem.displacement_words*4)));
      check(hipMemsetAsync(displacement.get(),0,size_t(problem.displacement_words*4),stream));
    }
    check(hipEventCreate(&start));check(hipEventCreate(&stop));check(hipEventCreateWithFlags(&marker,hipEventDisableTiming));
    check(hipStreamSynchronize(stream));
  }
  void kernel(uint32_t mode) {
    check(hipError_t(rns8_hip_system_roof(mode,a(),b(),problem.words,problem.blocks,problem.iterations,problem.stride,stream)));
  }
  double capture() {
    const auto started=Clock::now();
    check(hipStreamBeginCapture(stream,hipStreamCaptureModeThreadLocal));kernel(3);
    check(hipStreamEndCapture(stream,&graph));check(hipGraphInstantiate(&executable,graph,nullptr,nullptr,0));return ns(started);
  }
  Sample execute(bool timed) {
    const auto started=Clock::now();
    if(problem.displacement_words) {
      // A separate, observable read/write sweep; never infer measured HBM or
      // guaranteed cache eviction from its requested size.
      check(hipError_t(rns8_hip_system_roof_displace(static_cast<uint32_t*>(displacement.get()),problem.displacement_words,stream)));
      ++displacement_calls;
    }
    if(timed)check(hipEventRecord(start,stream));
    if(problem.mode==s::Mode::event) {
      check(hipEventRecord(marker,stream));check(hipStreamWaitEvent(stream,marker,0));
    } else if(problem.mode==s::Mode::descriptor) {
      check(hipMemcpyAsync(a(),host_input.get(),size_t(problem.words*4),hipMemcpyHostToDevice,stream));
    } else if(problem.mode==s::Mode::graph)check(hipGraphLaunch(executable,stream));
    else kernel(uint32_t(problem.mode));
    if(timed)check(hipEventRecord(stop,stream));
    check(hipStreamSynchronize(stream));++calls;
    Sample result;result.host_ns=ns(started);
    if(timed){float milliseconds=0;check(hipEventElapsedTime(&milliseconds,start,stop));if(milliseconds>0)result.event_ns=double(milliseconds)*1e6;}
    return result;
  }
  void verify() {
    const auto bytes=size_t(problem.words*4);
    auto* observed=static_cast<uint32_t*>(host_output.get());
    check(hipMemcpyAsync(observed,b(),bytes+64,hipMemcpyDeviceToHost,stream));check(hipStreamSynchronize(stream));
    for(uint64_t i=problem.words;i<problem.words+16;++i)
      if(observed[i]!=UINT32_C(0xcccccccc))throw std::runtime_error("output guard changed");
    if(problem.mode==s::Mode::event || problem.mode==s::Mode::descriptor)
      for(uint64_t i=0;i<problem.words;++i)if(observed[i]!=UINT32_C(0xcccccccc))throw std::runtime_error("unwritten output changed");
    output_sum=0;
    if(problem.mode!=s::Mode::event) {
      if(problem.mode==s::Mode::descriptor) {
        check(hipMemcpyAsync(observed,a(),bytes,hipMemcpyDeviceToHost,stream));check(hipStreamSynchronize(stream));
      }
      if(problem.mode==s::Mode::lds_read || problem.mode==s::Mode::lds_read_write) {
        for(uint32_t block=0;block<problem.blocks;++block) {
          const auto expected=s::reference_block(problem,block);
          for(uint32_t lane=0;lane<128;++lane)if(observed[uint64_t(block)*128+lane]!=expected[lane])throw std::runtime_error("LDS exact oracle");
        }
      } else for(uint64_t i=0;i<problem.words;++i) {
        const auto expected=s::seed(i)+uint32_t(problem.mode==s::Mode::launch || problem.mode==s::Mode::graph);
        if(observed[i]!=expected)throw std::runtime_error("complete output oracle");
      }
      for(uint64_t i=0;i<problem.words;++i)output_sum+=observed[i];
    } else if(hipEventQuery(marker)!=hipSuccess)throw std::runtime_error("event marker not complete");
    // Read every source cell as well; poisoned or accidentally overwritten
    // sources cannot be hidden by a final output checksum.
    check(hipMemcpyAsync(observed,a(),bytes,hipMemcpyDeviceToHost,stream));check(hipStreamSynchronize(stream));
    source_sum=0;
    for(uint64_t i=0;i<problem.words;++i) {
      if(observed[i]!=s::seed(i))throw std::runtime_error("immutable source oracle");source_sum+=observed[i];
    }
  }
  void verify_displacement() {
    auto* observed=static_cast<uint32_t*>(host_output.get());
    const auto expected=displacement_calls%2?UINT32_C(0xa5a5a5a5):0;
    const auto capacity=host_output.capacity()/4;
    for(uint64_t offset=0;offset<problem.displacement_words;offset+=capacity) {
      const auto count=std::min<uint64_t>(capacity,problem.displacement_words-offset);
      check(hipMemcpyAsync(observed,static_cast<const uint32_t*>(displacement.get())+offset,size_t(count*4),hipMemcpyDeviceToHost,stream));
      check(hipStreamSynchronize(stream));
      for(uint64_t i=0;i<count;++i)if(observed[i]!=expected)throw std::runtime_error("displacement oracle");
    }
  }
  bool close() noexcept {
    if(hipSetDevice(device)!=hipSuccess)return false;
    if(stream && hipStreamSynchronize(stream)!=hipSuccess)return false;
    if(executable){if(hipGraphExecDestroy(executable)!=hipSuccess)return false;executable=nullptr;}
    if(graph){if(hipGraphDestroy(graph)!=hipSuccess)return false;graph=nullptr;}
    for(auto* e:{&start,&stop,&marker})if(*e){if(hipEventDestroy(*e)!=hipSuccess)return false;*e=nullptr;}
    if(stream){if(hipStreamDestroy(stream)!=hipSuccess)return false;stream=nullptr;}
    if(input.reset()!=RNS8_SUCCESS || output.reset()!=RNS8_SUCCESS || displacement.reset()!=RNS8_SUCCESS ||
        host_input.reset()!=hipSuccess || host_output.reset()!=hipSuccess)return false;
    return true;
  }
};
#endif
}
int main(int argc,char** argv) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  std::unique_ptr<Owner> owner;
#endif
  try {
    if(argc==2 && std::string(argv[1])=="--help") {
      std::cout<<"rns8-system-roof-calibrate CASE.json RUN_ID; bounded global/LDS, launch, event, pinned descriptor upload and owned graph probes. No automatic promotion.\n";return 0;
    }
    if(argc!=3)throw std::invalid_argument("CASE.json and unique run id required");
    std::ifstream input(argv[1]);const auto c=Json::parse(input);
    if(c.size()!=11 || c.at("schema")!="rns8_system_roof_case_v1")throw std::invalid_argument("case ABI");
    s::Problem p;
    if(!s::parse(c.at("mode").get<std::string>(),p.mode))throw std::invalid_argument("mode");
    const auto device=integer(c,"device",0,63);p.words=integer(c,"words",1,uint64_t{1}<<29);
    p.blocks=uint32_t(integer(c,"blocks",1,4096));p.iterations=uint32_t(integer(c,"iterations",1,4096));
    p.stride=uint32_t(integer(c,"stride",1,32));p.displacement_words=integer(c,"displacement_words",0,uint64_t{1}<<29);
    p.budget_bytes=integer(c,"budget_bytes",1,uint64_t{1}<<40);
    const auto warmups=integer(c,"warmups",1,100),repeats=integer(c,"repeats",3,100);
    const auto resources=s::query(p);if(!resources.admitted)throw std::invalid_argument("resource or work admission");
    const std::string run_id=argv[2];
    if(run_id.empty() || run_id.size()>128 || run_id.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")!=std::string::npos)
      throw std::invalid_argument("run id");
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
    int count=0;const auto found=hipGetDeviceCount(&count);
    if(found!=hipSuccess || int(device)>=count){std::cerr<<"selected HIP device unavailable\n";return 77;}
    hipDeviceProp_t prop{};check(hipGetDeviceProperties(&prop,int(device)));
    const auto setup_start=Clock::now();owner=std::make_unique<Owner>();owner->device=int(device);owner->problem=p;owner->resources=resources;owner->initialize();
    const double setup=ns(setup_start);Json capture=nullptr;
    if(p.mode==s::Mode::graph)capture=owner->capture();
    const double first=owner->execute(false).host_ns;owner->verify();
    for(uint64_t i=0;i<warmups;++i){owner->execute(false);owner->verify();}
    Json samples=Json::array(),bursts=Json::array();
    for(uint64_t i=0;i<repeats;++i) {
      const auto instrumented=owner->execute(true);owner->verify();const auto production=owner->execute(false);owner->verify();
      samples.push_back({{"instrumented_host_ns",instrumented.host_ns},{"production_host_ns",production.host_ns},
          {"device_event_ns",instrumented.event_ns>0?Json(instrumented.event_ns):Json(nullptr)},{"exact_reference_passed",true}});
    }
    for(int burst=0;burst<3;++burst) {
      uint32_t calls=0;double elapsed=0;const auto started=Clock::now();
      do {owner->execute(false);++calls;elapsed=ns(started);}while(calls<4096 && (calls<16 || elapsed<250000000));
      owner->verify();bursts.push_back({{"calls",calls},{"elapsed_ns",elapsed},{"duration_target_met",calls>=16 && elapsed>=250000000},
          {"call_cap_reached",calls==4096},{"exact_reference_passed",true}});
    }
    owner->verify_displacement();int runtime=0,driver=0;check(hipRuntimeGetVersion(&runtime));check(hipDriverGetVersion(&driver));
    char bdf[64]{};check(hipDeviceGetPCIBusId(bdf,sizeof(bdf),int(device)));
    std::string uuid;static constexpr char hex[]="0123456789abcdef";
    for(unsigned char byte:prop.uuid.bytes){uuid+=hex[byte>>4];uuid+=hex[byte&15];}
    const char* host_platform=
#if defined(_WIN32)
        "native-windows";
#elif defined(__linux__)
        "native-linux";
#else
        "other";
#endif
    Json manifest=nullptr;
#ifdef RNS8_BUILD_MANIFEST_JSON
    manifest=Json::parse(RNS8_BUILD_MANIFEST_JSON);
#endif
    Json report={{"schema","rns8_system_roof_microcase_v1"},{"case",c},{"run_id",run_id},{"build_manifest",manifest},
      {"target",prop.gcnArchName},{"device_name",prop.name},{"device",device},{"runtime_version",runtime},{"driver_version",driver},
      {"host_platform",host_platform},{"device_uuid",uuid},{"device_bdf",bdf},{"host_cpu_model",nullptr},
      {"hardware_threads",std::thread::hardware_concurrency()},{"setup_ns",setup},{"first_ns",first},{"capture_instantiate_ns",capture},{"samples",samples},{"sustained",bursts},
      {"calls",owner->calls},{"displacement_calls",owner->displacement_calls},{"device_payload_bytes",resources.device_bytes},
      {"pinned_payload_bytes",resources.pinned_bytes},{"admitted_payload_bytes",resources.total_bytes},{"lds_bytes_per_block",resources.lds_bytes_per_block},
      {"modeled_unique_source_bytes",resources.unique_source_bytes},{"modeled_global_read_bytes",resources.issued_global_read_bytes},
      {"modeled_global_write_bytes",resources.issued_global_write_bytes},{"modeled_lds_read_bytes",resources.issued_lds_read_bytes},
      {"modeled_lds_write_bytes",resources.issued_lds_write_bytes},{"modeled_displacement_bytes",resources.displacement_bytes},
      {"actual_hbm_bytes",nullptr},{"actual_lds_transactions",nullptr},{"physical_cache_eviction_proven",false},{"clock_power_policy","inherited_unobserved"},
      {"lds_bank_mapping","unknown_until_named_target_instruction_review"},{"timer","host_steady_clock_nanoseconds_and_same_stream_HIP_events"},
      {"host_scope","complete_borrowed_probe_on_owned_stream_including_requested_displacement_and_stream_completion"},
      {"event_scope","probe_only_excludes_displacement_includes_recorded_operation_and_required_LDS_barriers"},
      {"arithmetic","uint32_mod_2_32_probe_only"},{"oracle_inside_interval",false},{"exact_reference_passed",true},
      {"production_timing_hooks_enabled",false},{"promotion_eligible",false},{"allocator_overhead_known",false},
      {"input_seed",2891336453u},{"input_multiplier",747796405u},{"descriptor_buffer_kind","portable_pinned_owned"},
      {"source_u64_sum",owner->source_sum},{"output_u64_sum",p.mode==s::Mode::event?Json(nullptr):Json(owner->output_sum)},
      {"checksum_contract","sum_of_all_canonical_uint32_values_mod_2_64_not_a_correctness_substitute"},
      {"isa_verified",false},{"isa_sha256",nullptr}};
    if(!owner->close()){(void)owner.release();throw std::runtime_error("cleanup retained until process exit");}
    std::cout<<report.dump()<<'\n';return 0;
#else
    (void)device;std::cerr<<"HIP support not compiled\n";return 77;
#endif
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  } catch(const Unsupported& error) {
    if(owner && !owner->close()){(void)owner.release();std::cerr<<error.what()<<"; cleanup retained until process exit\n";return 1;}
    std::cerr<<error.what()<<'\n';return 77;
#endif
  } catch(const std::exception& error) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
    // Keep all DMA endpoints, events, graphs and host staging together if drain
    // or release fails. The isolated collector process is the cleanup boundary.
    if(owner && !owner->close())(void)owner.release();
#endif
    std::cerr<<error.what()<<'\n';return 1;
  }
}
