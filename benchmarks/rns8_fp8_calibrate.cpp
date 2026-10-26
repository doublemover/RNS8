#include "core/api_internal.hpp"
#include "backend_common/sha256.hpp"
#include "backend_hip_direct/hip_backend.hpp"
#include <chrono>
#include <array>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <memory>
#include <vector>
// Failure cleanup retains undrained resources until isolated process exit.
namespace {
using json=nlohmann::json;using Clock=std::chrono::steady_clock;
template<class T> std::string checksum(const std::vector<T>& values){return rns8::detail::sha256(std::string(reinterpret_cast<const char*>(values.data()),values.size()*sizeof(T)));}
void check(rns8_status s){if(s==RNS8_UNSUPPORTED_BACKEND || s==RNS8_UNSUPPORTED_ARCH)throw std::domain_error(rns8_status_string(s));if(s!=RNS8_SUCCESS)throw std::runtime_error(rns8_status_string(s));}
json phase_json(const std::array<double,5>& p){return {{"prepare_us",p[0]},{"execute_us",p[1]},{"export_us",p[2]},{"cleanup_us",p[3]},{"total_us",p[4]}};}
double us(Clock::time_point a,Clock::time_point b){return std::chrono::duration<double,std::micro>(b-a).count();}
uint32_t number(const json& c,const char* key,uint32_t low,uint32_t high){if(!c.at(key).is_number_integer())throw std::invalid_argument(key);auto v=c.at(key).get<int64_t>();if(v<low || v>high)throw std::invalid_argument(key);return uint32_t(v);}
struct Owner {
  rns8_context* ctx=nullptr;rns8_plan* plan=nullptr;rns8_workspace* work=nullptr;rns8_matrix *a=nullptr,*b=nullptr,*c=nullptr;rns8_fp8_recipe* recipe=nullptr;
  rns8_status close(){rns8_status s;if(recipe){s=rns8_destroy_fp8_recipe(recipe);if(s!=RNS8_SUCCESS)return s;recipe=nullptr;}
    for(auto** matrix:{&c,&b,&a})if(*matrix){s=rns8_destroy_matrix(*matrix);if(s!=RNS8_SUCCESS)return s;*matrix=nullptr;}
    if(work){s=rns8_destroy_workspace(work);if(s!=RNS8_SUCCESS)return s;work=nullptr;}if(plan){s=rns8_destroy_plan(plan);if(s!=RNS8_SUCCESS)return s;plan=nullptr;}
    if(ctx){s=rns8_destroy_context(ctx);if(s!=RNS8_SUCCESS)return s;ctx=nullptr;}return RNS8_SUCCESS;}
  ~Owner(){(void)close();}
};
}
int main(int argc,char** argv){try {
  if(argc!=3)throw std::invalid_argument("usage: rns8-fp8-calibrate CASE.json unique-run-id");json c;std::ifstream(argv[1])>>c;
  if(c.size()!=10 || c.at("schema")!="finite_fp8_public_case_v1")throw std::invalid_argument("case ABI");
  const uint32_t m=number(c,"m",1,512),n=number(c,"n",1,512),k=number(c,"k",1,131073),modulus=number(c,"modulus",2,256),repeats=number(c,"repeats",3,100),warmups=number(c,"warmups",1,100);
  if(std::string(argv[2]).empty() || std::string(argv[2]).size()>128 || std::string(argv[2]).find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")!=std::string::npos)throw std::invalid_argument("run id");
  const uint32_t seed=number(c,"seed",0,UINT32_MAX);const std::string mode=c.at("mode"),residency=c.at("residency");
  if((mode!="int8" && mode!="fp8" && mode!="fp8-materialized") || (residency!="cold" && residency!="repeated-b" && residency!="repeated-ab"))throw std::invalid_argument("mode/residency");
  if(uint64_t(m)*n*k>134217728)throw std::invalid_argument("complete oracle work budget");
  std::vector<uint8_t> a(size_t(m)*(k+1),0xa5),b(size_t(k)*(n+2),0xa5),out(size_t(m)*(n+3),0xa5),expected(size_t(m)*n);
  for(uint32_t r=0;r<m;++r)for(uint32_t q=0;q<k;++q)a[size_t(r)*(k+1)+q]=uint8_t((uint64_t(r)*71+uint64_t(q)*19+seed)%modulus);
  for(uint32_t q=0;q<k;++q)for(uint32_t col=0;col<n;++col)b[size_t(q)*(n+2)+col]=uint8_t((uint64_t(q)*37+uint64_t(col)*11+seed)%modulus);
  for(uint32_t r=0;r<m;++r)for(uint32_t col=0;col<n;++col){boost::multiprecision::cpp_int sum=0;for(uint32_t q=0;q<k;++q)sum+=uint64_t(a[size_t(r)*(k+1)+q])*b[size_t(q)*(n+2)+col];expected[size_t(r)*n+col]=(sum%modulus).convert_to<uint8_t>();}
  rns8::detail::hip_direct_timing_set_enabled(false);uint64_t generation=0,retained=0;std::string certificate,target,device;int runtime=0;
  auto create=[&](){auto owner=std::make_unique<Owner>();rns8_context_options options{};options.struct_size=sizeof(options);options.abi_version=RNS8_ABI_VERSION;options.requested_backend=RNS8_BACKEND_HIP_DIRECT;
    check(rns8_create_context(0,&options,&owner->ctx));rns8_gemm_desc d{};d.struct_size=sizeof(d);d.abi_version=RNS8_ABI_VERSION;d.semantics=RNS8_FINITE_RING_U8;d.bound_kind=RNS8_BOUND_NONE;d.requested_backend=RNS8_BACKEND_HIP_DIRECT;d.m=m;d.n=n;d.k=k;d.finite_modulus=modulus;
    check(rns8_create_plan(owner->ctx,&d,&owner->plan));check(rns8_create_workspace(owner->ctx,owner->plan,&owner->work));
    auto matrix=[&](uint32_t rows,uint32_t cols,rns8_matrix** result){rns8_matrix_desc md{};md.struct_size=sizeof(md);md.abi_version=RNS8_ABI_VERSION;md.semantics=d.semantics;md.bound_kind=RNS8_BOUND_NONE;md.logical_layout=RNS8_LAYOUT_ROW_MAJOR;md.rows=rows;md.cols=cols;md.logical_ld=cols;check(rns8_create_matrix(owner->ctx,&md,result));};
    matrix(m,k,&owner->a);matrix(k,n,&owner->b);matrix(m,n,&owner->c);++generation;check(rns8_pack_finite_u8(owner->ctx,owner->a,modulus,a.data(),k+1,generation));check(rns8_pack_finite_u8(owner->ctx,owner->b,modulus,b.data(),n+2,generation));
    if(mode!="int8"){rns8_fp8_desc desc{};desc.struct_size=sizeof(desc);desc.abi_version=RNS8_ABI_VERSION;desc.format=RNS8_FP8_GFX942_E4M3_FNUZ;desc.flags=RNS8_FP8_ASSUME_NAMED_INSTRUCTION_MODEL;desc.method=mode=="fp8-materialized";desc.max_preparation_bytes=4+(desc.method?uint64_t(m)*n*16:0);check(rns8_prepare_fp8_recipe(owner->ctx,owner->plan,&desc,&owner->recipe));
      rns8_fp8_info info{};info.struct_size=sizeof(info);info.abi_version=RNS8_ABI_VERSION;check(rns8_get_fp8_info(owner->recipe,&info));retained=info.retained_device_bytes;certificate=info.certificate_sha256;}
    rns8_device_info info{};info.struct_size=sizeof(info);info.abi_version=RNS8_ABI_VERSION;check(rns8_get_device_info(owner->ctx,&info));target=info.gcn_arch;device=info.name;runtime=info.hip_runtime_version;return owner;};
  const auto setup_start=Clock::now();auto owner=create();const double setup=us(setup_start,Clock::now());
  if(residency=="cold"){check(owner->close());owner.reset();}
  auto call=[&](bool phases=true){const auto begin=phases?Clock::now():Clock::time_point{};if(residency=="cold")owner=create();
    else if(residency=="repeated-b")check(rns8_pack_finite_u8(owner->ctx,owner->a,modulus,a.data(),k+1,++generation));
    const auto prepared=phases?Clock::now():Clock::time_point{};if(mode=="int8")check(rns8_gemm_finite_u8(owner->ctx,owner->plan,modulus,owner->a,owner->b,owner->c,owner->work));
    else check(rns8_gemm_fp8_research(owner->recipe,owner->a,owner->b,owner->c,owner->work));const auto computed=phases?Clock::now():Clock::time_point{};
    check(rns8_export_finite_u8(owner->ctx,owner->plan,modulus,owner->c,out.data(),n+3));const auto exported=phases?Clock::now():Clock::time_point{};
    if(residency=="cold"){check(owner->close());owner.reset();}const auto end=phases?Clock::now():Clock::time_point{};
    return std::array<double,5>{us(begin,prepared),us(prepared,computed),us(computed,exported),us(exported,end),us(begin,end)};};
  auto verify=[&](){for(uint32_t r=0;r<m;++r)for(uint32_t col=0;col<n+3;++col)
    if(out[size_t(r)*(n+3)+col]!=(col<n?expected[size_t(r)*n+col]:0xa5))throw std::runtime_error("full exact output/padding mismatch");};
  auto first=phase_json(call());verify();for(uint32_t i=0;i<warmups;++i)call(false);json samples=json::array();for(uint32_t i=0;i<repeats;++i){samples.push_back(phase_json(call()));verify();}
  json sustained=json::array();for(unsigned observation=0;observation<3;++observation){uint64_t calls=0;const auto start=Clock::now();do{call(false);++calls;}while(calls<4096 && (calls<16 || us(start,Clock::now())<250000));const double total=us(start,Clock::now());verify();sustained.push_back({{"calls",calls},{"total_us",total},{"per_call_us",total/calls},{"minimum_duration_reached",total>=250000}});}
  if(owner)check(owner->close());
  std::cout<<json({{"schema","rns8_public_fp8_microcase_v1"},{"case",c},{"run_id",argv[2]},{"target",target},{"device",device},{"runtime_version",runtime},
    {"build_manifest",json::parse(RNS8_BUILD_MANIFEST_JSON)},{"dataset","affine_finite_bytes_v1"},{"a_checksum",checksum(a)},{"b_checksum",checksum(b)},{"output_checksum",checksum(out)},{"backend_actual","hip-direct"},{"host_instrumentation_state","hooks_disabled"},
    {"output_boundary","complete_public_finite_product_and_padded_export"},{"setup_us",setup},{"first_call",first},{"samples",samples},{"sustained",sustained},
    {"retained_recipe_device_bytes",retained},{"method_actual",mode=="fp8-materialized"?1:0},{"format",mode!="int8"?"amd_e4m3_fnuz":"signed_int8"},{"conditional_instruction_model",mode!="int8"},{"target_qualified",false},{"performance_qualified",false},{"certificate_sha256",certificate},
    {"native_useful_products",uint64_t(m)*n*k},{"exact_reference_passed",true},{"oracle_inside_production_interval",false},{"promotion_eligible",false}}).dump(2)<<'\n';return 0;
}catch(const std::domain_error& error){std::cerr<<error.what()<<'\n';return 77;}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}
