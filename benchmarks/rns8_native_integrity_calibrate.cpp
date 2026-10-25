// Matched public producer transactions plus explicitly labeled private standalone
// sentinel control. No timing or performance outcome is inferred from this file.
#include "core/api_internal.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
namespace {
using json=nlohmann::json;using Clock=std::chrono::steady_clock;using boost::multiprecision::cpp_int;
void check(rns8_status s){if(s!=RNS8_SUCCESS)throw std::runtime_error(rns8_status_string(s));}
double us(Clock::time_point a,Clock::time_point b){return std::chrono::duration<double,std::micro>(b-a).count();}
uint64_t number(const json& c,const char* key,uint64_t lo,uint64_t hi){if(!c.at(key).is_number_integer())throw std::invalid_argument(key);auto x=c.at(key).get<int64_t>();if(x<0 || uint64_t(x)<lo || uint64_t(x)>hi)throw std::invalid_argument(key);return uint64_t(x);}
struct Owner {
  rns8_context* ctx=nullptr;rns8_plan* plan=nullptr;rns8_workspace* work=nullptr;rns8_matrix *a=nullptr,*b=nullptr,*c=nullptr;rns8_prepared_execution* execution=nullptr;
  rns8_status close(){rns8_status s;if(execution){s=rns8_destroy_prepared_execution(execution);if(s!=RNS8_SUCCESS)return s;execution=nullptr;}
    for(auto** x:{&c,&b,&a})if(*x){s=rns8_destroy_matrix(*x);if(s!=RNS8_SUCCESS)return s;*x=nullptr;}
    if(work){s=rns8_destroy_workspace(work);if(s!=RNS8_SUCCESS)return s;work=nullptr;}if(plan){s=rns8_destroy_plan(plan);if(s!=RNS8_SUCCESS)return s;plan=nullptr;}
    if(ctx){s=rns8_destroy_context(ctx);if(s!=RNS8_SUCCESS)return s;ctx=nullptr;}return RNS8_SUCCESS;}
  ~Owner(){(void)close();}
};
}
int main(int argc,char** argv){try {
  using namespace rns8::detail;
  if(argc!=3)throw std::invalid_argument("usage: rns8-native-integrity-calibrate CASE.json unique-run-id");json c;std::ifstream(argv[1])>>c;
  if(c.size()!=13 || c.at("schema")!="native_integrity_case_v1" || !c.at("unsigned_inputs").is_boolean())throw std::invalid_argument("case ABI");
  const uint32_t m=number(c,"m",1,512),n=number(c,"n",1,512),k=number(c,"k",1,65537),prefix=number(c,"prefix",1,20),
      repeats=number(c,"repeats",3,100),warmups=number(c,"warmups",1,100),seed=number(c,"seed",0,UINT32_MAX);
  const uint64_t budget=number(c,"max_payload_bytes",8,UINT64_C(1)<<40);const bool u=c.at("unsigned_inputs");
  const std::string mode=c.at("mode"),residency=c.at("residency"),dataset=c.at("dataset");
  if(uint64_t(m)*n*k>134217728 || (mode!="compute" && mode!="sentinel-only" && mode!="compute-sentinel" && mode!="compute-cpu-reference") ||
      (residency!="resident" && residency!="cold" && residency!="repack") || (mode=="sentinel-only" && residency!="resident") ||
      (dataset!="small" && dataset!="fullwidth"))throw std::invalid_argument("route/work");
  if(!qualified_i32_gemm_address(m,n,k,k,n,n,prefix) || !qualified_i32_matrix_address(m,k,k,1,8) || !qualified_i32_matrix_address(k,n,n,1,8))throw std::invalid_argument("address");
  const uint64_t cells=uint64_t(m)*k+uint64_t(k)*n+uint64_t(m)*n;
  const uint64_t known_payload=(2*prefix+24)*cells+48*uint64_t(m)*(n+3)+8;
  if(known_payload>budget)throw std::invalid_argument("logical payload budget");
  auto value=[&](size_t i)->uint64_t{const uint64_t x=(i*37+seed)%15;if(dataset=="small")return u?x:uint64_t(int64_t(x)-7);
    return u?(x%3==0?UINT64_MAX:x%3==1?(UINT64_C(1)<<63):x):(x%3==0?uint64_t(INT64_MIN):x%3==1?uint64_t(INT64_MAX):uint64_t(-int64_t(x)));};
  std::vector<uint64_t> av(size_t(m)*k),bv(size_t(k)*n),output(size_t(m)*(n+3)*3,UINT64_C(0x5a5a5a5a5a5a5a5a));
  for(size_t i=0;i<av.size();++i)av[i]=value(i);for(size_t i=0;i<bv.size();++i)bv[i]=value(i+11);
  auto integer=[&](uint64_t x){cpp_int v=x;if(!u && (x>>63))v-=cpp_int(1)<<64;return v;};
  auto reference=[&](){std::vector<cpp_int> result(size_t(m)*n);for(uint32_t r=0;r<m;++r)for(uint32_t col=0;col<n;++col)
      for(uint32_t q=0;q<k;++q)result[size_t(r)*n+col]+=integer(av[size_t(r)*k+q])*integer(bv[size_t(q)*n+col]);return result;};
  const auto gold=reference(); // excluded from all measured production intervals
  auto audit=[&](const std::vector<cpp_int>& expected){for(uint32_t r=0;r<m;++r)for(uint32_t col=0;col<n+3;++col){const size_t cell=size_t(r)*(n+3)+col;
    if(col>=n){for(uint32_t limb=0;limb<3;++limb)if(output[cell*3+limb]!=UINT64_C(0x5a5a5a5a5a5a5a5a))throw std::runtime_error("padding");continue;}
    cpp_int got=cpp_int(output[cell*3])+(cpp_int(output[cell*3+1])<<64)+(cpp_int(output[cell*3+2])<<128);if(!u && (output[cell*3+2]>>63))got-=cpp_int(1)<<192;
    if(got!=expected[size_t(r)*n+col])throw std::runtime_error("complete integer reference");}};
  hip_direct_timing_set_enabled(false);uint64_t generation=0;std::string target,device,kernel;int runtime=0;rns8_native_integrity_report report{};
  auto pack=[&](Owner& o){++generation;if(u){check(rns8_pack_u64(o.ctx,o.a,av.data(),k,generation));check(rns8_pack_u64(o.ctx,o.b,bv.data(),n,generation));}
    else{check(rns8_pack_i64(o.ctx,o.a,reinterpret_cast<const int64_t*>(av.data()),k,generation));check(rns8_pack_i64(o.ctx,o.b,reinterpret_cast<const int64_t*>(bv.data()),n,generation));}
    if(o.execution)check(rns8_refresh_prepared_inputs(o.execution));};
  auto create=[&](){auto o=std::make_unique<Owner>();rns8_context_options co{};co.struct_size=sizeof(co);co.abi_version=RNS8_ABI_VERSION;co.requested_backend=RNS8_BACKEND_HIP_DIRECT;check(rns8_create_context(0,&co,&o->ctx));
    rns8_gemm_desc d{};d.struct_size=sizeof(d);d.abi_version=RNS8_ABI_VERSION;d.semantics=u?RNS8_EXACT_WIDE_UNSIGNED:RNS8_EXACT_WIDE_SIGNED;d.bound_kind=RNS8_BOUND_NONE;d.requested_backend=co.requested_backend;d.m=m;d.n=n;d.k=k;d.max_prefix=prefix;d.flags=RNS8_PLAN_FORCE_FIXED_PREFIX;
    check(rns8_create_plan(o->ctx,&d,&o->plan));check(rns8_create_workspace(o->ctx,o->plan,&o->work));
    auto matrix=[&](uint32_t rows,uint32_t cols,rns8_matrix** out){rns8_matrix_desc md{};md.struct_size=sizeof(md);md.abi_version=RNS8_ABI_VERSION;md.semantics=d.semantics;md.bound_kind=RNS8_BOUND_NONE;md.rows=rows;md.cols=cols;md.logical_ld=cols;md.logical_layout=RNS8_LAYOUT_ROW_MAJOR;md.max_prefix=prefix;md.flags=RNS8_MATRIX_RETAIN_NATIVE_INPUT;check(rns8_create_matrix(o->ctx,&md,out));};
    matrix(m,k,&o->a);matrix(k,n,&o->b);matrix(m,n,&o->c);pack(*o);check(rns8_prepare_gemm(o->ctx,o->plan,o->a,o->b,o->c,o->work,&o->execution));
    if(mode=="compute-sentinel"){rns8_native_integrity_options options{};options.struct_size=sizeof(options);options.abi_version=RNS8_ABI_VERSION;options.mode=1;options.max_scratch_bytes=8;check(rns8_set_prepared_native_integrity(o->execution,&options));}
    rns8_device_info di{};di.struct_size=sizeof(di);di.abi_version=RNS8_ABI_VERSION;check(rns8_get_device_info(o->ctx,&di));target=di.gcn_arch;device=di.name;runtime=di.hip_runtime_version;
    rns8_plan_backend_info pi{};pi.struct_size=sizeof(pi);pi.abi_version=RNS8_ABI_VERSION;check(rns8_get_plan_backend_info(o->plan,&pi));kernel=pi.selected_kernel;
    if(mode=="sentinel-only"){rns8_completion_token t{};check(rns8_execute_prepared(o->execution,&t));}return o;};
  const auto setup_start=Clock::now();auto owner=create();const double setup=us(setup_start,Clock::now());if(residency=="cold"){check(owner->close());owner.reset();}
  auto invoke=[&](){const auto start=Clock::now();if(residency=="cold")owner=create();else if(residency=="repack")pack(*owner);const auto prepared=Clock::now();
    if(mode=="sentinel-only"){
      const cpp_int mask=(cpp_int(1)<<64)-1;uint64_t bound[3];for(unsigned i=0;i<3;++i)bound[i]=((owner->c->exact_max_magnitude>>(64*i))&mask).convert_to<uint64_t>();
      IntegrityIdentity id{1,1,owner->a->matrix_instance_id,owner->a->content_generation,owner->b->matrix_instance_id,owner->b->content_generation,1,prefix};uint32_t failures=0;
      check(hip_direct_check_native_product_integrity(0,u?owner->a->hip_native_u64:owner->a->hip_native_i64,u?owner->b->hip_native_u64:owner->b->hip_native_i64,owner->c->hip_residues,m,n,k,k,n,u,bound,id,id,failures));
    } else {rns8_completion_token token{};check(rns8_execute_prepared(owner->execution,&token));if(mode=="compute-sentinel"){report={};report.struct_size=sizeof(report);report.abi_version=RNS8_ABI_VERSION;check(rns8_get_prepared_native_integrity_report(owner->execution,token,&report));if(report.outcome!=1 || report.failure_mask)throw std::runtime_error("sentinel outcome");}}
    const auto computed=Clock::now();if(u)check(rns8_export_exact_wide_unsigned_limbs(owner->ctx,owner->plan,owner->c,output.data(),n+3,3));else check(rns8_export_exact_wide_signed_limbs(owner->ctx,owner->plan,owner->c,output.data(),n+3,3));
    if(mode=="compute-cpu-reference")audit(reference());const auto exported=Clock::now();if(residency=="cold"){check(owner->close());owner.reset();}const auto end=Clock::now();
    return json{{"prepare_us",us(start,prepared)},{"execute_us",us(prepared,computed)},{"export_reference_us",us(computed,exported)},{"cleanup_us",us(exported,end)},{"total_us",us(start,end)}};};
  auto first=invoke();audit(gold);for(uint32_t i=0;i<warmups;++i)invoke();json samples=json::array();for(uint32_t i=0;i<repeats;++i){samples.push_back(invoke());audit(gold);}
  json sustained=json::array();for(unsigned i=0;i<3;++i){unsigned count=0;const auto start=Clock::now();do{invoke();++count;}while(count<4096 && (count<16 || us(start,Clock::now())<250000));const auto elapsed=us(start,Clock::now());audit(gold);sustained.push_back({{"calls",count},{"total_us",elapsed},{"per_call_us",elapsed/count}});}
  if(owner)check(owner->close());
  json verification=nullptr;if(mode=="compute-sentinel")verification={{"outcome",report.outcome},{"failure_mask",report.failure_mask},{"operation",report.operation_id},{"epoch",report.epoch},{"a_generation",report.a_generation},{"b_generation",report.b_generation},{"basis_version",report.computational_basis_version},{"prefix",report.computational_prefix},{"check_moduli",{257,263}},{"device_flag_bytes",4},{"pinned_flag_bytes",4}};
  std::cout<<json({{"schema","rns8_native_integrity_microcase_v1"},{"case",c},{"run_id",argv[2]},{"target",target},{"device",device},{"runtime_version",runtime},{"build_manifest",json::parse(RNS8_BUILD_MANIFEST_JSON)},
    {"selected_compute_kernel",kernel},{"host_instrumentation_state","hooks_disabled"},{"output_boundary","public_prepared_product_and_padded_three_limb_export"},{"standalone_control",mode=="sentinel-only"},
    {"standalone_control_scope","private_read_only_checker_per_call_flag_allocation_device_sync_and_export"},{"known_payload_envelope_bytes",known_payload},{"memory_scope","logical_payload_excludes_cpp_int_heap_capacity_driver_allocator_and_opaque_runtime"},
    {"verification",verification},{"guarantee","fixed_redundant_residue_consistency_not_arbitrary_correctness"},{"ideal_failure_probability",nullptr},{"check_moduli_in_computational_range",false},
    {"cpu_reference_inputs","original_host_dataset_no_input_D2H"},{"setup_us",setup},{"first_call",first},{"samples",samples},{"sustained",sustained},{"exact_reference_passed",true},{"full_cpu_reference_inside_interval",mode=="compute-cpu-reference"},{"promotion_eligible",false}}).dump(2)<<'\n';return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
