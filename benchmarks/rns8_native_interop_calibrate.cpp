#include <rns8/rns8.h>
#include <boost/multiprecision/cpp_int.hpp>
#include <nlohmann/json.hpp>
#include <array>
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
#include <hip/hip_runtime_api.h>
extern "C" int rns8_interop_application_produce(void*,void*,void*,uint64_t,uint64_t,uint64_t,uint32_t,bool,bool,void*);
extern "C" int rns8_interop_application_consume(const void*,void*,uint64_t,void*);
#endif
namespace {
using json=nlohmann::json;using Clock=std::chrono::steady_clock;using boost::multiprecision::cpp_int;
void check(rns8_status s){if(s!=RNS8_SUCCESS)throw std::runtime_error(rns8_status_string(s));}
double us(Clock::time_point a,Clock::time_point b){return std::chrono::duration<double,std::micro>(b-a).count();}
uint64_t number(const json& c,const char* key,uint64_t lo,uint64_t hi){if(!c.at(key).is_number_integer())throw std::invalid_argument(key);auto x=c.at(key).get<int64_t>();if(x<0 || uint64_t(x)<lo || uint64_t(x)>hi)throw std::invalid_argument(key);return uint64_t(x);}
struct ExternalOwner{unsigned references=0,acquires=0,releases=0;};
rns8_status acquire(void* p){auto& o=*static_cast<ExternalOwner*>(p);++o.references;++o.acquires;return RNS8_SUCCESS;}
void release(void* p){auto& o=*static_cast<ExternalOwner*>(p);--o.references;++o.releases;}
rns8_allocation_observation observe(rns8_context* ctx) {
  rns8_allocation_observation out{};out.struct_size=sizeof(out);out.abi_version=RNS8_ABI_VERSION;
  check(rns8_get_allocation_observation(ctx,&out));return out;
}
json allocation_domain(const rns8_allocation_domain_observation& d) {
  return {{"live_bytes",d.live_bytes},{"peak_bytes",d.peak_bytes},{"live_allocations",d.live_allocations},
      {"peak_allocations",d.peak_allocations},{"allocation_calls",d.allocation_calls},
      {"free_calls",d.free_calls},{"allocated_bytes",d.allocated_bytes}};
}
json allocation_observation(const rns8_allocation_observation& o) {
  return {{"epoch",o.epoch},{"device",allocation_domain(o.device)},{"pinned_host",allocation_domain(o.pinned_host)}};
}
bool same_live_payload(const rns8_allocation_domain_observation& a,const rns8_allocation_domain_observation& b) {
  return a.live_bytes==b.live_bytes && a.live_allocations==b.live_allocations;
}
}
int main(int argc,char** argv){try {
  if(argc!=3)throw std::invalid_argument("usage: rns8-native-interop-calibrate CASE.json unique-run-id");json c;std::ifstream(argv[1])>>c;
  if(c.size()!=12 || c.at("schema")!="native_interop_case_v1" || !c.at("unsigned_inputs").is_boolean())throw std::invalid_argument("case ABI");
  const uint32_t m=number(c,"m",1,512),n=number(c,"n",1,512),k=number(c,"k",1,65537),p=number(c,"prefix",1,20),
      repeats=number(c,"repeats",3,100),warmups=number(c,"warmups",1,100),seed=number(c,"seed",0,UINT32_MAX);
  const uint64_t budget=number(c,"max_payload_bytes",8,UINT64_C(1)<<40);const bool u=c.at("unsigned_inputs");
  const std::string mode=c.at("mode"),dataset=c.at("dataset");const bool full=dataset=="fullwidth",device=mode=="device-stream" || mode=="device-graph",graph=mode=="device-graph",resident=mode=="resident-control";
  if((!device && !resident && mode!="host-roundtrip") || (dataset!="small" && !full) || uint64_t(m)*n*k>134217728)throw std::invalid_argument("route/work");
  const uint64_t ac=uint64_t(m)*k,bc=uint64_t(k)*n,cc=uint64_t(m)*(n+3)*3,input_bytes=(ac+bc)*8,output_bytes=cc*8;
  const uint64_t device_payload=(ac+bc+uint64_t(m)*n)*p+input_bytes+output_bytes+8+(ac+bc)*p+32+uint64_t(m)*n*24+4;
  const uint64_t host_payload=input_bytes+2*output_bytes+12;
  if(device_payload+host_payload>budget || (ac+bc)*8>INT32_MAX || cc*8>INT32_MAX || ac*p>INT32_MAX || bc*p>INT32_MAX)throw std::invalid_argument("payload/address budget");
  const uint64_t envelope=full?(u?UINT64_MAX:UINT64_C(1)<<63):(u?14:7);
  cpp_int product=1;const unsigned moduli[]={256,255,253,251,247,239,233,229,227,223,217,211,199,197,193,191,181,179,173,167};for(unsigned i=0;i<p;++i)product*=moduli[i];
  if(product<=cpp_int(k)*envelope*envelope*(u?1:2))throw std::invalid_argument("strict product range");
#if !defined(RNS8_ENABLE_HIP) || !RNS8_ENABLE_HIP
  throw std::runtime_error("native interop benchmark requires a compiled HIP backend");
#else
  const auto hipcheck=[](hipError_t s){if(s!=hipSuccess)throw std::runtime_error(hipGetErrorString(s));};
  auto value=[&](uint64_t i){const uint64_t x=(i*37+seed)%15;if(!full)return u?x:uint64_t(int64_t(x)-7);return u?(x%3==0?UINT64_MAX:x%3==1?(UINT64_C(1)<<63):x):(x%3==0?uint64_t(INT64_MIN):x%3==1?uint64_t(INT64_MAX):uint64_t(-int64_t(x)));};
  auto integer=[&](uint64_t x){cpp_int v=x;if(!u && (x>>63))v-=cpp_int(1)<<64;return v;};
  std::vector<uint64_t> ha(ac),hb(bc),hc(cc,UINT64_C(0x5a5a5a5a5a5a5a5a)),validation(cc);
  for(uint64_t i=0;i<ac;++i)ha[i]=value(i);for(uint64_t i=0;i<bc;++i)hb[i]=value(i+11);
  std::vector<cpp_int> gold(uint64_t(m)*n);for(unsigned r=0;r<m;++r)for(unsigned col=0;col<n;++col)for(unsigned q=0;q<k;++q)gold[uint64_t(r)*n+col]+=integer(ha[uint64_t(r)*k+q])*integer(hb[uint64_t(q)*n+col]);
  // Gold construction and every full validation readback are outside timings in ALL modes.
  const auto setup_start=Clock::now();hipcheck(hipSetDevice(0));std::array<void*,4> allocations{};
  hipcheck(hipMalloc(&allocations[0],ac*8));hipcheck(hipMalloc(&allocations[1],bc*8));hipcheck(hipMalloc(&allocations[2],output_bytes));hipcheck(hipMalloc(&allocations[3],8));
  std::array<hipStream_t,5> streams{};for(unsigned i=0;i<streams.size();++i)if(device || i==0 || i==4)hipcheck(hipStreamCreateWithFlags(&streams[i],hipStreamNonBlocking));
  hipcheck(static_cast<hipError_t>(rns8_interop_application_produce(allocations[0],allocations[1],allocations[2],ac,bc,cc,seed,u,full,streams[0])));hipcheck(hipStreamSynchronize(streams[0]));
  rns8_context* ctx=nullptr;rns8_plan* plan=nullptr;rns8_workspace* work=nullptr;std::array<rns8_matrix*,3> matrices{};std::array<rns8_native_device_lease*,3> leases{};std::array<ExternalOwner,3> owners{};std::array<rns8_prepared_execution*,4> operations{};
  rns8_context_options co{};co.struct_size=sizeof(co);co.abi_version=RNS8_ABI_VERSION;co.requested_backend=RNS8_BACKEND_HIP_DIRECT;check(rns8_create_context(0,&co,&ctx));
  uint64_t allocation_epoch=0;check(rns8_reset_allocation_peaks(ctx,&allocation_epoch));
  const auto owned_before=observe(ctx);
  rns8_gemm_desc gd{};gd.struct_size=sizeof(gd);gd.abi_version=RNS8_ABI_VERSION;gd.requested_backend=co.requested_backend;gd.semantics=u?RNS8_EXACT_WIDE_UNSIGNED:RNS8_EXACT_WIDE_SIGNED;gd.bound_kind=RNS8_BOUND_NONE;gd.m=m;gd.n=n;gd.k=k;gd.max_prefix=p;gd.flags=RNS8_PLAN_FORCE_FIXED_PREFIX|RNS8_PLAN_EXACT_VALUE_BOUND;
  check(rns8_create_plan(ctx,&gd,&plan));check(rns8_create_workspace(ctx,plan,&work));
  for(unsigned i=0;i<3;++i){const uint32_t rows=i==1?k:m,cols=i==0?k:n;rns8_matrix_desc md{};md.struct_size=sizeof(md);md.abi_version=RNS8_ABI_VERSION;md.semantics=gd.semantics;md.bound_kind=RNS8_BOUND_NONE;md.rows=rows;md.cols=cols;md.logical_ld=cols;md.logical_layout=RNS8_LAYOUT_ROW_MAJOR;md.max_prefix=p;check(rns8_create_matrix(ctx,&md,&matrices[i]));
    if(device || (resident && i==2)){rns8_native_device_desc d{};d.struct_size=sizeof(d);d.abi_version=RNS8_ABI_VERSION;d.element_type=i<2?(u?RNS8_NATIVE_DEVICE_U64:RNS8_NATIVE_DEVICE_I64):(u?RNS8_NATIVE_DEVICE_UNSIGNED_LIMBS:RNS8_NATIVE_DEVICE_SIGNED_LIMBS);d.limb_count=i<2?0:3;d.access=i<2?RNS8_NATIVE_DEVICE_READ:RNS8_NATIVE_DEVICE_WRITE;d.allocation_base=allocations[i];d.allocation_bytes=i==0?ac*8:i==1?bc*8:output_bytes;d.rows=rows;d.cols=cols;d.ld=i<2?cols:n+3;d.device_id=0;d.producer_complete=1;d.policy=RNS8_NATIVE_DEVICE_BORROW;d.owner_identity=i+1;d.generation=1;d.owner=&owners[i];d.acquire=acquire;d.release=release;check(rns8_create_native_device_lease(ctx,&d,&leases[i]));}}
  uint64_t generation=1;
  auto pack=[&](){if(u){check(rns8_pack_u64(ctx,matrices[0],ha.data(),k,generation));check(rns8_pack_u64(ctx,matrices[1],hb.data(),n,generation));}else{check(rns8_pack_i64(ctx,matrices[0],reinterpret_cast<const int64_t*>(ha.data()),k,generation));check(rns8_pack_i64(ctx,matrices[1],reinterpret_cast<const int64_t*>(hb.data()),n,generation));}};
  rns8_native_transfer_options o{};o.struct_size=sizeof(o);o.abi_version=RNS8_ABI_VERSION;o.import_values=1;o.verify_input_bound=1;o.max_input_magnitude=envelope;o.max_total_scratch_bytes=budget;
  // Establish only this route's input facts; do not allocate unused host-pack buffers.
  if(device){for(unsigned i=0;i<2;++i){check(rns8_prepare_native_transfer_hip(ctx,matrices[i],leases[i],&o,&operations[i]));rns8_completion_token initial{};check(rns8_execute_prepared(operations[i],&initial));}}
  else pack();
  check(rns8_prepare_gemm(ctx,plan,matrices[0],matrices[1],matrices[2],work,&operations[2]));
  // Native export preparation freezes a proven output envelope, requiring one setup product.
  if(device || resident){rns8_completion_token initial{};check(rns8_execute_prepared(operations[2],&initial));o.import_values=0;o.verify_input_bound=0;o.max_input_magnitude=0;check(rns8_prepare_native_transfer_hip(ctx,matrices[2],leases[2],&o,&operations[3]));}
  // Stream submission needs an explicit completion-event preparation too;
  // graph capture performs that preparation internally.
  if(device)for(auto* e:operations)check(graph?rns8_capture_prepared_hip_graph(e):rns8_prepare_hip_submission(e));
  rns8_device_info di{};di.struct_size=sizeof(di);di.abi_version=RNS8_ABI_VERSION;check(rns8_get_device_info(ctx,&di));rns8_plan_backend_info pi{};pi.struct_size=sizeof(pi);pi.abi_version=RNS8_ABI_VERSION;check(rns8_get_plan_backend_info(plan,&pi));
  const double setup=us(setup_start,Clock::now());
  const auto owned_after_setup=observe(ctx);
  auto wait=[&](unsigned i,rns8_completion_token t){rns8_completion_info info{};info.struct_size=sizeof(info);info.abi_version=RNS8_ABI_VERSION;check(rns8_wait_prepared_hip(operations[i],t,&info));check(info.operation_status);if(!info.output_is_current)throw std::runtime_error("unpublished result");};
  auto invoke=[&](){const auto start=Clock::now();++generation;for(auto* lease:leases)if(lease)check(rns8_begin_native_device_write(lease));
    hipcheck(static_cast<hipError_t>(rns8_interop_application_produce(allocations[0],allocations[1],allocations[2],ac,bc,cc,seed,u,full,streams[0])));
    for(auto* lease:leases)if(lease)check(rns8_end_native_device_write_hip(lease,generation,streams[0]));const auto produced=Clock::now();
    std::array<rns8_completion_token,4> tokens{};
    if(device){check(rns8_refresh_prepared_inputs(operations[3]));for(unsigned i=0;i<2;++i){check(rns8_refresh_prepared_inputs(operations[i]));check(graph?rns8_submit_prepared_hip_graph(operations[i],streams[i+1],&tokens[i]):rns8_submit_prepared_hip(operations[i],streams[i+1],&tokens[i]));}
      check(rns8_bind_prepared_dependencies(operations[2],operations.data(),tokens.data(),2));
      check(graph?rns8_submit_prepared_hip_graph_pipeline(operations[2],streams[3],&tokens[2]):rns8_submit_prepared_hip_pipeline(operations[2],streams[3],&tokens[2]));
      check(rns8_bind_prepared_dependency(operations[3],operations[2],tokens[2]));
      check(graph?rns8_submit_prepared_hip_graph_pipeline(operations[3],streams[4],&tokens[3]):rns8_submit_prepared_hip_pipeline(operations[3],streams[4],&tokens[3]));
    }else if(!resident){hipcheck(hipStreamSynchronize(streams[0]));hipcheck(hipMemcpy(ha.data(),allocations[0],ac*8,hipMemcpyDeviceToHost));hipcheck(hipMemcpy(hb.data(),allocations[1],bc*8,hipMemcpyDeviceToHost));pack();check(rns8_refresh_prepared_inputs(operations[2]));check(rns8_execute_prepared(operations[2],&tokens[2]));}
    else{check(rns8_refresh_prepared_inputs(operations[2]));check(rns8_execute_prepared(operations[2],&tokens[2]));}
    const auto computed=Clock::now();
    if(device){wait(3,tokens[3]);}
    else if(resident){check(rns8_refresh_prepared_inputs(operations[3]));check(rns8_submit_prepared_hip(operations[3],streams[4],&tokens[3]));wait(3,tokens[3]);}
    else{if(u)check(rns8_export_exact_wide_unsigned_limbs(ctx,plan,matrices[2],hc.data(),n+3,3));else check(rns8_export_exact_wide_signed_limbs(ctx,plan,matrices[2],hc.data(),n+3,3));hipcheck(hipMemcpy(allocations[2],hc.data(),output_bytes,hipMemcpyHostToDevice));}
    const auto exported=Clock::now();hipcheck(hipMemsetAsync(allocations[3],0,8,streams[4]));hipcheck(static_cast<hipError_t>(rns8_interop_application_consume(allocations[2],allocations[3],cc,streams[4])));hipcheck(hipStreamSynchronize(streams[4]));const auto end=Clock::now();
    return json{{"producer_enqueue_us",us(start,produced)},{"input_compute_us",us(produced,computed)},{"export_observation_us",us(computed,exported)},{"consumer_us",us(exported,end)},{"total_us",us(start,end)}};};
  auto audit=[&](){hipcheck(hipMemcpy(validation.data(),allocations[2],output_bytes,hipMemcpyDeviceToHost));uint64_t expected_digest=0;for(uint64_t i=0;i<cc;++i)expected_digest^=validation[i]^(i*UINT64_C(0x9e3779b97f4a7c15));uint64_t actual_digest=0;hipcheck(hipMemcpy(&actual_digest,allocations[3],8,hipMemcpyDeviceToHost));if(actual_digest!=expected_digest)throw std::runtime_error("external consumer");
    for(unsigned r=0;r<m;++r)for(unsigned col=0;col<n+3;++col){const auto index=(uint64_t(r)*(n+3)+col)*3;if(col>=n){for(unsigned l=0;l<3;++l)if(validation[index+l]!=UINT64_C(0x5a5a5a5a5a5a5a5a))throw std::runtime_error("padding");continue;}
      cpp_int got=cpp_int(validation[index])+(cpp_int(validation[index+1])<<64)+(cpp_int(validation[index+2])<<128);if(!u && validation[index+2]>>63)got-=cpp_int(1)<<192;if(got!=gold[uint64_t(r)*n+col])throw std::runtime_error("full independent oracle");}};
  auto first=invoke();audit();for(unsigned i=0;i<warmups;++i)invoke();json samples=json::array();for(unsigned i=0;i<repeats;++i){samples.push_back(invoke());audit();}
  json sustained=json::array();for(unsigned i=0;i<3;++i){unsigned count=0;const auto start=Clock::now();do{invoke();++count;}while(count<4096 && (count<16 || us(start,Clock::now())<250000));const double elapsed=us(start,Clock::now());audit();sustained.push_back({{"calls",count},{"total_us",elapsed},{"per_call_us",elapsed/count}});}
  const auto owned_after_calls=observe(ctx);
  for(auto* e:operations)if(e)check(rns8_destroy_prepared_execution(e));for(auto* lease:leases)if(lease)check(rns8_destroy_native_device_lease(lease));
  const unsigned participating_owners=device?3:resident?1:0;
  for(unsigned i=0;i<owners.size();++i){const auto& owner=owners[i];const unsigned expected=device || (resident && i==2)?1:0;if(owner.references || owner.acquires!=expected || owner.releases!=expected)throw std::runtime_error("owner lifetime");}
  for(auto* matrix:matrices)check(rns8_destroy_matrix(matrix));check(rns8_destroy_workspace(work));check(rns8_destroy_plan(plan));
  const auto owned_after_cleanup=observe(ctx);
  if(owned_before.epoch!=allocation_epoch || owned_after_setup.epoch!=allocation_epoch || owned_after_calls.epoch!=allocation_epoch ||
      owned_after_cleanup.epoch!=allocation_epoch || !same_live_payload(owned_before.device,owned_after_cleanup.device) ||
      !same_live_payload(owned_before.pinned_host,owned_after_cleanup.pinned_host))throw std::runtime_error("owned payload did not retire to baseline");
  check(rns8_destroy_context(ctx));for(auto s:streams)if(s)hipcheck(hipStreamDestroy(s));for(auto a:allocations)hipcheck(hipFree(a));
  std::cout<<json({{"schema","rns8_native_interop_microcase_v2"},{"case",c},{"run_id",argv[2]},{"build_manifest",json::parse(RNS8_BUILD_MANIFEST_JSON)},{"target",di.gcn_arch},{"device",di.name},{"runtime_version",di.hip_runtime_version},{"selected_compute_kernel",pi.selected_kernel},
    {"output_boundary","external_producer_to_external_consumer_padded_three_limbs"},{"dataset_replayed_unchanged",true},{"external_generations_advanced",true},{"resident_control_excludes_input_binding",resident},
    {"host_payload_bytes_per_call",device||resident?0:2*input_bytes+uint64_t(m)*n*24+output_bytes},{"compact_status_bytes_per_call",device?12:resident?4:4},{"consumer_validation_bytes",8},{"validation_payload_bytes",output_bytes},
    {"preflight_device_payload_estimate_bytes",device_payload},{"plan_workspace_bytes",pi.workspace_required_bytes},{"preflight_host_payload_estimate_bytes",host_payload},{"memory_scope","conservative_all_route_logical_preflight_excludes_Boost_library_heap_driver_opaque_events_graphs_workspace_separate"},{"launch_count",nullptr},{"launch_count_scope","not_inferred_from_api_calls"},
    {"participating_external_owners",participating_owners},{"owner_acquire_callbacks",participating_owners},{"owner_release_callbacks",participating_owners},
    {"observed_allocation_scope","RNS8_owned_requested_device_and_pinned_payload_excludes_app_allocations_host_heap_allocator_and_driver"},
    {"owned_before",allocation_observation(owned_before)},{"owned_after_setup",allocation_observation(owned_after_setup)},
    {"owned_after_calls",allocation_observation(owned_after_calls)},{"owned_after_cleanup",allocation_observation(owned_after_cleanup)},
    {"all_owned_payload_retired",true},{"external_application_device_requested_bytes",input_bytes+output_bytes+8},
    {"ordinary_host_native_vector_capacity_bytes",8*uint64_t(ha.capacity()+hb.capacity()+hc.capacity()+validation.capacity())},
    {"ordinary_host_capacity_scope","four_native_vectors_only_excludes_Boost_objects_and_dynamic_limbs_library_heap_and_RSS"},
    {"timing_source","steady_clock_complete_host_observed_transaction"},{"phase_scope","enqueue_and_observation_not_disjoint_device_kernel_durations"},{"setup_us",setup},{"first_call",first},{"samples",samples},{"sustained",sustained},{"exact_reference_passed",true},{"owner_callbacks_exactly_once",true},{"promotion_eligible",false}}).dump(2)<<'\n';return 0;
#endif
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
