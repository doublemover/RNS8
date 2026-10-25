#include <rns8/rns8.h>
#include "core/hip_resources.hpp"
#include <boost/multiprecision/cpp_int.hpp>
#include <nlohmann/json.hpp>
#include <array>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
namespace {
using json=nlohmann::json;using Clock=std::chrono::steady_clock;using boost::multiprecision::cpp_int;
constexpr uint64_t canary=UINT64_C(0x5a5a5a5a5a5a5a5a);
bool cleanup_failed=false;
struct Unsupported:std::runtime_error{using std::runtime_error::runtime_error;};
void need(bool value,const char* message){if(!value)throw std::runtime_error(message);}
void check(rns8_status s) {
  if(s==RNS8_UNSUPPORTED_BACKEND || s==RNS8_UNSUPPORTED_ARCH || s==RNS8_UNSUPPORTED_OS)throw Unsupported(rns8_status_string(s));
  if(s!=RNS8_SUCCESS)throw std::runtime_error(rns8_status_string(s));
}
uint64_t elapsed(Clock::time_point a,Clock::time_point b){return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(b-a).count());}
uint64_t number(const json& c,const char* key,uint64_t low,uint64_t high) {
  need(c.at(key).is_number_unsigned() || c.at(key).is_number_integer(),"integer control required");
  const auto n=c.at(key).get<int64_t>();need(n>=0 && uint64_t(n)>=low && uint64_t(n)<=high,"integer control out of range");return uint64_t(n);
}
struct Case {json original;uint32_t device,m,n,k,repeats,warmups,seed;uint64_t budget;bool unsign,pinned,stream,graph,full;std::string mode;};
uint64_t payload_envelope(const Case& c) {
  const uint64_t ac=uint64_t(c.m)*c.k,bc=uint64_t(c.k)*c.n,cc=uint64_t(c.m)*c.n;
  const uint64_t host=8*(uint64_t(c.m)*(c.k+3)+uint64_t(c.k)*(c.n+3)+uint64_t(c.m)*(c.n+3)*3);
  return 2*host+128*cc+20*(ac+bc+cc)+16*(ac+bc)+48*cc+20*(ac+bc)+65536;
}
Case parse(const json& c) {
  need(c.is_object() && c.size()==12 && c.at("schema")=="host_transfer_case_v1" && c.at("unsigned_inputs").is_boolean(),"case ABI");
  Case x{};x.original=c;x.device=number(c,"device",0,63);x.m=number(c,"m",1,512);x.n=number(c,"n",1,512);x.k=number(c,"k",1,65537);
  x.repeats=number(c,"repeats",3,100);x.warmups=number(c,"warmups",1,100);x.seed=number(c,"seed",0,UINT32_MAX);
  x.budget=number(c,"max_payload_bytes",8,UINT64_C(512)*1024*1024);x.unsign=c.at("unsigned_inputs");x.mode=c.at("mode");
  const std::string dataset=c.at("dataset");x.full=dataset=="fullwidth";
  need(dataset=="small" || x.full,"dataset");
  need(x.mode=="pageable-sync" || x.mode=="pinned-sync" || x.mode=="pinned-stream" || x.mode=="pinned-graph","mode");
  x.pinned=x.mode!="pageable-sync";x.stream=x.mode=="pinned-stream" || x.mode=="pinned-graph";x.graph=x.mode=="pinned-graph";
  need(uint64_t(x.m)*x.n*x.k<=134217728,"bounded reference work exceeded");
  need(payload_envelope(x)<=x.budget,"preflight payload envelope exceeds hard budget");return x;
}
cpp_int integer(uint64_t bits,bool unsign){cpp_int value=bits;if(!unsign && bits>>63)value-=cpp_int(1)<<64;return value;}
uint64_t value(uint64_t i,const Case& c) {
  const uint64_t x=(i*37+c.seed)%15;
  if(!c.full)return c.unsign?x:uint64_t(int64_t(x)-7);
  return c.unsign?(x%3==0?UINT64_MAX:x%3==1?UINT64_C(1)<<63:x):
      (x%3==0?uint64_t(INT64_MIN):x%3==1?uint64_t(INT64_MAX):uint64_t(-int64_t(x)));
}
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
void hipcheck(hipError_t status){if(status!=hipSuccess)throw std::runtime_error(hipGetErrorString(status));}
struct Owners {
  rns8_context* ctx=nullptr;rns8_plan* plan=nullptr;rns8_workspace* workspace=nullptr;
  std::array<rns8_matrix*,3> matrices{};std::array<rns8_pinned_host_buffer*,3> pins{};
  std::array<rns8_prepared_execution*,4> operations{};std::array<hipStream_t,4> streams{};
  rns8_status close() noexcept {
    for(auto stream:streams)if(stream && hipStreamSynchronize(stream)!=hipSuccess)return RNS8_BACKEND_FAILURE;
    for(int i=3;i>=0;--i)if(operations[i]){auto s=rns8_destroy_prepared_execution(operations[i]);if(s!=RNS8_SUCCESS)return s;operations[i]=nullptr;}
    for(auto& pin:pins)if(pin){auto s=rns8_destroy_pinned_host_buffer(pin);if(s!=RNS8_SUCCESS)return s;pin=nullptr;}
    for(auto& matrix:matrices)if(matrix){auto s=rns8_destroy_matrix(matrix);if(s!=RNS8_SUCCESS)return s;matrix=nullptr;}
    if(workspace){auto s=rns8_destroy_workspace(workspace);if(s!=RNS8_SUCCESS)return s;workspace=nullptr;}
    if(plan){auto s=rns8_destroy_plan(plan);if(s!=RNS8_SUCCESS)return s;plan=nullptr;}
    for(auto& stream:streams)if(stream){if(hipStreamDestroy(stream)!=hipSuccess)return RNS8_BACKEND_FAILURE;stream=nullptr;}
    if(ctx){auto s=rns8_destroy_context(ctx);if(s!=RNS8_SUCCESS)return s;ctx=nullptr;}return RNS8_SUCCESS;
  }
  ~Owners(){if(close()!=RNS8_SUCCESS)cleanup_failed=true;}
};
struct ReadWindow {
  rns8_pinned_host_buffer* buffer;const void* data=nullptr;uint64_t bytes=0;
  explicit ReadWindow(rns8_pinned_host_buffer* b):buffer(b){check(rns8_begin_pinned_host_read(buffer,&data,&bytes));}
  ~ReadWindow(){if(rns8_end_pinned_host_read(buffer)!=RNS8_SUCCESS)cleanup_failed=true;}
};
void fill(rns8_pinned_host_buffer* buffer,const std::vector<uint64_t>& values,uint64_t generation) {
  void* data=nullptr;uint64_t bytes=0;check(rns8_begin_pinned_host_write(buffer,&data,&bytes));
  need(bytes==values.size()*8,"pinned rectangle capacity");std::memcpy(data,values.data(),size_t(bytes));
  check(rns8_end_pinned_host_write(buffer,generation));
}
json run(const Case& c,const std::string& run_id) {
  const uint64_t ac=uint64_t(c.m)*c.k,bc=uint64_t(c.k)*c.n,cc=uint64_t(c.m)*c.n;
  const uint64_t lda=c.k+3,ldb=c.n+3,ldc=c.n+3;
  const uint64_t host_rectangles=8*(uint64_t(c.m)*lda+uint64_t(c.k)*ldb+uint64_t(c.m)*ldc*3);
  // Includes both routes' possible upload/export/transfer payloads plus bounded
  // reference storage. Driver graphs/handles and allocator overhead stay opaque.
  uint64_t envelope=payload_envelope(c);
  need(envelope<=c.budget,"preflight payload envelope exceeds hard budget");
  std::vector<uint64_t> a(uint64_t(c.m)*lda,canary),b(uint64_t(c.k)*ldb,canary),output(uint64_t(c.m)*ldc*3,canary);
  const uint64_t caller_capacity=8*(uint64_t(a.capacity())+b.capacity()+output.capacity());
  need(caller_capacity>=host_rectangles && caller_capacity-host_rectangles<=c.budget-envelope,"caller capacity exceeds preflight allowance");
  envelope+=caller_capacity-host_rectangles;
  for(unsigned r=0;r<c.m;++r)for(unsigned q=0;q<c.k;++q)a[r*lda+q]=value(uint64_t(r)*c.k+q,c);
  for(unsigned q=0;q<c.k;++q)for(unsigned col=0;col<c.n;++col)b[q*ldb+col]=value(uint64_t(q)*c.n+col+11,c);
  std::vector<cpp_int> gold(cc);
  const auto oracle=[&](){for(unsigned r=0;r<c.m;++r)for(unsigned col=0;col<c.n;++col) {
    auto& sum=gold[uint64_t(r)*c.n+col];sum=0;
    for(unsigned q=0;q<c.k;++q)sum+=integer(a[r*lda+q],c.unsign)*integer(b[q*ldb+col],c.unsign);
  }};
  oracle();Owners owner;hipcheck(hipSetDevice(c.device));rns8::detail::hip_direct_timing_set_enabled(false);
#if defined(_WIN32)
  need(_putenv_s("RNS8_HIP_PINNED_EXPORT_STAGING","0")==0,"ordinary export staging control");
#else
  need(setenv("RNS8_HIP_PINNED_EXPORT_STAGING","0",1)==0,"ordinary export staging control");
#endif
  const auto setup_start=Clock::now();rns8_context_options options{};options.struct_size=sizeof(options);options.abi_version=RNS8_ABI_VERSION;options.requested_backend=RNS8_BACKEND_HIP_DIRECT;
  check(rns8_create_context(c.device,&options,&owner.ctx));
  rns8_gemm_desc gd{};gd.struct_size=sizeof(gd);gd.abi_version=RNS8_ABI_VERSION;gd.requested_backend=options.requested_backend;
  gd.semantics=c.unsign?RNS8_EXACT_WIDE_UNSIGNED:RNS8_EXACT_WIDE_SIGNED;gd.bound_kind=RNS8_BOUND_NONE;
  gd.m=c.m;gd.n=c.n;gd.k=c.k;gd.max_prefix=20;gd.flags=RNS8_PLAN_FORCE_FIXED_PREFIX;
  check(rns8_create_plan(owner.ctx,&gd,&owner.plan));
  rns8_plan_backend_info plan_info{};plan_info.struct_size=sizeof(plan_info);plan_info.abi_version=RNS8_ABI_VERSION;
  check(rns8_get_plan_backend_info(owner.plan,&plan_info));need(plan_info.workspace_required_bytes<=c.budget-envelope,"workspace exceeds remaining budget");
  check(rns8_create_workspace(owner.ctx,owner.plan,&owner.workspace));
  for(unsigned i=0;i<3;++i) {
    rns8_matrix_desc md{};md.struct_size=sizeof(md);md.abi_version=RNS8_ABI_VERSION;md.semantics=gd.semantics;md.bound_kind=RNS8_BOUND_NONE;
    md.rows=i==1?c.k:c.m;md.cols=i==0?c.k:c.n;md.logical_ld=md.cols;md.logical_layout=RNS8_LAYOUT_ROW_MAJOR;md.max_prefix=20;
    check(rns8_create_matrix(owner.ctx,&md,&owner.matrices[i]));
  }
  const auto pack=[&](){for(unsigned i=0;i<2;++i) {
    const auto& source=i?b:a;const auto ld=i?ldb:lda;
    check(c.unsign?rns8_pack_u64(owner.ctx,owner.matrices[i],source.data(),ld,0):
        rns8_pack_i64(owner.ctx,owner.matrices[i],reinterpret_cast<const int64_t*>(source.data()),ld,0));
  }};
  pack();check(rns8_gemm_rns(owner.ctx,owner.plan,owner.matrices[0],owner.matrices[1],owner.matrices[2],owner.workspace));
  uint64_t capture_ns=0;
  if(c.pinned) {
    for(unsigned i=0;i<3;++i) {
      rns8_pinned_host_desc d{};d.struct_size=sizeof(d);d.abi_version=RNS8_ABI_VERSION;
      d.element_type=i<2?(c.unsign?RNS8_NATIVE_DEVICE_U64:RNS8_NATIVE_DEVICE_I64):
          (c.unsign?RNS8_NATIVE_DEVICE_UNSIGNED_LIMBS:RNS8_NATIVE_DEVICE_SIGNED_LIMBS);
      d.access=i<2?RNS8_NATIVE_DEVICE_READ:RNS8_NATIVE_DEVICE_WRITE;d.rows=i==1?c.k:c.m;d.cols=i==0?c.k:c.n;
      d.ld=i==0?lda:i==1?ldb:ldc;d.limb_count=i==2?3:0;uint64_t bytes=0;check(rns8_query_pinned_host_bytes(&d,&bytes));
      check(rns8_create_pinned_host_buffer_hip(owner.ctx,&d,bytes,&owner.pins[i]));fill(owner.pins[i],i==0?a:i==1?b:output,1);
    }
    rns8_native_transfer_options transfer{};transfer.struct_size=sizeof(transfer);transfer.abi_version=RNS8_ABI_VERSION;
    transfer.import_values=1;transfer.verify_input_bound=0;transfer.max_total_scratch_bytes=c.budget;
    for(unsigned i=0;i<2;++i)check(rns8_prepare_host_transfer_hip(owner.ctx,owner.matrices[i],owner.pins[i],&transfer,&owner.operations[i]));
    // Freeze compute/export under the same full-native proof carried by later
    // imports; an initially small host-scan proof cannot bound that envelope.
    for(unsigned i=0;i<2;++i){rns8_completion_token token{};check(rns8_execute_prepared(owner.operations[i],&token));}
    check(rns8_gemm_rns(owner.ctx,owner.plan,owner.matrices[0],owner.matrices[1],owner.matrices[2],owner.workspace));
    check(rns8_prepare_gemm(owner.ctx,owner.plan,owner.matrices[0],owner.matrices[1],owner.matrices[2],owner.workspace,&owner.operations[2]));
    transfer.import_values=0;check(rns8_prepare_host_transfer_hip(owner.ctx,owner.matrices[2],owner.pins[2],&transfer,&owner.operations[3]));
    if(c.stream)for(auto& stream:owner.streams)hipcheck(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));
    if(c.graph) {const auto start=Clock::now();for(auto* operation:owner.operations)check(rns8_capture_prepared_hip_graph(operation));capture_ns=elapsed(start,Clock::now());}
  }
  rns8_device_info device{};device.struct_size=sizeof(device);device.abi_version=RNS8_ABI_VERSION;check(rns8_get_device_info(owner.ctx,&device));
  const uint64_t setup_ns=elapsed(setup_start,Clock::now());uint64_t calls=0;
  const auto wait=[&](rns8_completion_token token) {
    rns8_completion_info info{};info.struct_size=sizeof(info);info.abi_version=RNS8_ABI_VERSION;
    check(rns8_wait_prepared_hip(owner.operations[3],token,&info));check(info.operation_status);need(info.output_is_current,"unpublished pinned output");
  };
  const auto invoke=[&]() {
    const auto start=Clock::now();std::array<rns8_completion_token,4> tokens{};
    if(!c.pinned) {
      pack();check(rns8_gemm_rns(owner.ctx,owner.plan,owner.matrices[0],owner.matrices[1],owner.matrices[2],owner.workspace));
      check(c.unsign?rns8_export_exact_wide_unsigned_limbs(owner.ctx,owner.plan,owner.matrices[2],output.data(),ldc,3):
          rns8_export_exact_wide_signed_limbs(owner.ctx,owner.plan,owner.matrices[2],output.data(),ldc,3));
    } else if(!c.stream) {
      for(unsigned i=0;i<4;++i){check(rns8_refresh_prepared_inputs(owner.operations[i]));check(rns8_execute_prepared(owner.operations[i],&tokens[i]));}
    } else {
      check(rns8_refresh_prepared_inputs(owner.operations[3]));
      for(unsigned i=0;i<2;++i) {
        check(rns8_refresh_prepared_inputs(owner.operations[i]));
        check(c.graph?rns8_submit_prepared_hip_graph(owner.operations[i],owner.streams[i],&tokens[i]):
            rns8_submit_prepared_hip(owner.operations[i],owner.streams[i],&tokens[i]));
      }
      check(rns8_bind_prepared_dependencies(owner.operations[2],owner.operations.data(),tokens.data(),2));
      check(c.graph?rns8_submit_prepared_hip_graph_pipeline(owner.operations[2],owner.streams[2],&tokens[2]):
          rns8_submit_prepared_hip_pipeline(owner.operations[2],owner.streams[2],&tokens[2]));
      check(rns8_bind_prepared_dependency(owner.operations[3],owner.operations[2],tokens[2]));
      check(c.graph?rns8_submit_prepared_hip_graph_pipeline(owner.operations[3],owner.streams[3],&tokens[3]):
          rns8_submit_prepared_hip_pipeline(owner.operations[3],owner.streams[3],&tokens[3]));
    }
    const auto enqueued=Clock::now();if(c.pinned)wait(tokens[3]);const auto end=Clock::now();++calls;
    const auto enqueue=elapsed(start,enqueued),observe=elapsed(enqueued,end),total=elapsed(start,end);need(total>0,"censored complete-call duration");
    return json{{"enqueue_ns",enqueue},{"observation_ns",observe},{"total_ns",total},{"phase_censored",enqueue==0 || observe==0}};
  };
  const auto audit_words=[&](const uint64_t* words) {
    for(unsigned r=0;r<c.m;++r)for(unsigned col=0;col<ldc;++col) {
      const auto at=(uint64_t(r)*ldc+col)*3;
      if(col>=c.n){for(unsigned l=0;l<3;++l)need(words[at+l]==canary,"output padding");continue;}
      cpp_int actual=cpp_int(words[at])+(cpp_int(words[at+1])<<64)+(cpp_int(words[at+2])<<128);
      if(!c.unsign && words[at+2]>>63)actual-=cpp_int(1)<<192;need(actual==gold[uint64_t(r)*c.n+col],"full Boost product");
    }
  };
  uint64_t audits=0;
  const auto audit=[&](){
    if(c.pinned){ReadWindow window(owner.pins[2]);need(window.bytes==output.size()*8,"output pinned span");audit_words(static_cast<const uint64_t*>(window.data));}
    else audit_words(output.data());
    for(unsigned i=0;i<2;++i) {
      const auto& values=i?b:a;const uint64_t rows=i?c.k:c.m,cols=i?c.n:c.k,ld=i?ldb:lda;
      for(uint64_t r=0;r<rows;++r)for(uint64_t col=cols;col<ld;++col)need(values[r*ld+col]==canary,"native input padding");
      if(c.pinned){ReadWindow window(owner.pins[i]);const auto* words=static_cast<const uint64_t*>(window.data);
        for(uint64_t r=0;r<rows;++r)for(uint64_t col=cols;col<ld;++col)need(words[r*ld+col]==canary,"pinned input padding");}
    }
    ++audits;
  };
  json phases=json::array();
  for(unsigned generation=1;generation<=2;++generation) {
    uint64_t rebind_ns=0;if(generation==2) {
      a[0]=0;oracle();const auto start=Clock::now();if(c.pinned){fill(owner.pins[0],a,2);fill(owner.pins[1],b,2);}rebind_ns=elapsed(start,Clock::now());
    }
    const auto before=calls,before_audits=audits;auto first=invoke();audit();for(unsigned i=0;i<c.warmups;++i)invoke();audit();
    json samples=json::array();for(unsigned i=0;i<c.repeats;++i){samples.push_back(invoke());audit();}
    json sustained=json::array();for(unsigned i=0;i<3;++i) {
      uint64_t count=0,time=0;const auto start=Clock::now();
      do{invoke();++count;time=elapsed(start,Clock::now());}while(count<4096 && (count<16 || time<250000000));audit();
      sustained.push_back({{"calls",count},{"elapsed_ns",time},{"mean_call_ns",double(time)/count},{"final_exact_reference_passed",true}});
    }
    phases.push_back({{"generation",generation},{"binding_action",generation==1?"initial_inputs":"changed_A0_same_handles"},
        {"rebind_ns",rebind_ns},{"first_call",first},{"samples",samples},{"sustained",sustained},
        {"production_calls",calls-before},{"independent_audits",audits-before_audits},{"full_output_and_padding_passed",true}});
  }
  json allocations=json::array(),transfers=json::array();uint64_t host_bytes=caller_capacity,device_bytes=plan_info.workspace_required_bytes,pinned_bytes=0;
  for(auto* matrix:owner.matrices) {
    rns8_matrix_allocation_info i{};i.struct_size=sizeof(i);i.abi_version=RNS8_ABI_VERSION;check(rns8_get_matrix_allocation_info(matrix,&i));
    host_bytes+=i.host_payload_capacity_bytes;device_bytes+=i.device_total_capacity_bytes;
    allocations.push_back({{"host_bytes",i.host_payload_capacity_bytes},{"device_bytes",i.device_total_capacity_bytes}});
  }
  if(c.pinned)for(unsigned i:{0u,1u,3u}) {
    rns8_prepared_transfer_info t{};t.struct_size=sizeof(t);t.abi_version=RNS8_ABI_VERSION;check(rns8_get_prepared_transfer_info(owner.operations[i],&t));
    device_bytes+=t.retained_device_bytes;pinned_bytes+=t.retained_pinned_host_bytes;
    transfers.push_back({{"operation",t.operation},{"device_bytes",t.retained_device_bytes},{"pinned_bytes",t.retained_pinned_host_bytes},
        {"retained_native_bytes",t.retained_native_matrix_bytes},{"verified_input_bound",t.verified_input_bound}});
  }
  need(host_bytes+device_bytes+pinned_bytes+128*cc<=c.budget,"queried payload/reference envelope exceeded");
  json report={{"schema","rns8_host_transfer_microcase_v1"},{"case",c.original},{"run_id",run_id},{"build_manifest",json::parse(RNS8_BUILD_MANIFEST_JSON)},
      {"target",device.gcn_arch},{"device",device.name},{"device_actual",device.device_id},{"runtime_version",device.hip_runtime_version},
      {"output_boundary","readable_host_padded_three_limb_result"},{"prefix",20},{"input_bound_policy",c.pinned?"full_native64_envelope":"engine_host_value_scan"},
      {"verify_input_bound",false},{"graph_requested",c.graph},{"graph_executed",c.graph},{"prepared_owners",c.pinned?4:0},{"pinned_buffers",c.pinned?3:0},
      {"borrowed_streams",c.stream?4:0},{"setup_ns",setup_ns},{"capture_ns",capture_ns},{"timing_source","steady_clock_complete_host_observed_call_ns"},
      {"setup_bootstrap_import_calls",c.pinned?2:0},{"setup_bootstrap_gemm_calls",c.pinned?2:1},
      {"setup_scope","construction_initial_proof_materialization_and_optional_capture"},
      {"phase_scope","submission_then_terminal_observation_sync_submission_includes_completion"},{"production_timing_hooks_enabled",false},
      {"oracle_inside_interval",false},{"caller_initialization_inside_interval",false},{"observed_overlap",nullptr},
      {"ordinary_export_pinned_staging","disabled_explicit_control"},
      {"matrix_allocations",allocations},{"transfer_allocations",transfers},{"plan_workspace_bytes",plan_info.workspace_required_bytes},
      {"ordinary_host_payload_bytes",host_bytes},{"pinned_host_payload_bytes",pinned_bytes},{"device_payload_bytes",device_bytes},
      {"caller_host_payload_capacity_bytes",caller_capacity},
      {"reference_reservation_bytes",128*cc},{"preflight_payload_envelope_bytes",envelope+plan_info.workspace_required_bytes},
      {"opaque_runtime_setup_unknown",true},{"capacity_scope","queried_retained_payloads_not_measured_peak_or_driver_allocations"},
      {"production_calls",calls},{"phases",phases},{"exact_reference_passed",true},{"promotion_eligible",false}};
  const auto close_start=Clock::now();check(owner.close());report["cleanup_ns"]=elapsed(close_start,Clock::now());return report;
}
#endif
}
int main(int argc,char** argv) {
  try {
    if(argc==2 && std::string(argv[1])=="--help") {
      std::cout<<"rns8-host-transfer-calibrate CASE.json unique-run-id\nMatched pageable-sync|pinned-sync|pinned-stream|pinned-graph, native64 host inputs to padded host three-limb output. Independent audits outside timers.\n";return 0;
    }
    need(argc==3,"expected CASE.json unique-run-id");json input;std::ifstream(argv[1])>>input;const auto c=parse(input);
    const std::string id=argv[2];need(!id.empty() && id.size()<=128 && id.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")==std::string::npos,"run identity");
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
    auto report=run(c,id);need(!cleanup_failed,"resource cleanup failed");std::cout<<report.dump()<<'\n';return 0;
#else
    (void)c;throw Unsupported("compiled HIP backend required");
#endif
  }catch(const Unsupported& e){std::cerr<<e.what()<<'\n';return 77;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
