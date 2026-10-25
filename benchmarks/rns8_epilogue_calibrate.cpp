#include "backend_hip_direct/hip_backend.hpp"
#include "backend_common/residue_epilogue.hpp"
#include "core/exact_range.hpp"
#include "epilogue_coefficients.hpp"
#include <array>
#include <chrono>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
using namespace rns8::detail;
using boost::multiprecision::cpp_int;
using json=nlohmann::json;
using Clock=std::chrono::steady_clock;
void check(rns8_status s){if(s!=RNS8_SUCCESS)throw std::runtime_error(rns8_status_string(s));}
double us(Clock::time_point a,Clock::time_point b){return std::chrono::duration<double,std::micro>(b-a).count();}
struct Owner {
  rns8_context* ctx=nullptr;rns8_plan* plan=nullptr;rns8_workspace* work=nullptr;
  rns8_matrix *a=nullptr,*b=nullptr,*d=nullptr,*c=nullptr;
  ~Owner(){for(auto* m:{c,d,b,a})if(m)(void)rns8_destroy_matrix(m);if(work)(void)rns8_destroy_workspace(work);if(plan)(void)rns8_destroy_plan(plan);if(ctx)(void)rns8_destroy_context(ctx);}
};
uint32_t number(const json& j,const char* key,uint32_t low,uint32_t high){
  if(!j.at(key).is_number_unsigned() && !j.at(key).is_number_integer())throw std::invalid_argument("integer case field required");
  const int64_t value=j.at(key).get<int64_t>();if(value<low || value>high)throw std::invalid_argument("case field outside bounded domain");return uint32_t(value);
}
}
int main(int argc,char** argv){try{
  if(argc!=3)throw std::invalid_argument("usage: rns8-epilogue-calibrate CASE.json unique-run-id");
  std::ifstream file(argv[1]);json config;file>>config;
  if(config.size()!=9 || config.at("schema")!="epilogue_public_case_v1")throw std::invalid_argument("case ABI mismatch");
  const uint32_t m=number(config,"m",1,512),n=number(config,"n",1,512),k=number(config,"k",1,65537),repeats=number(config,"repeats",3,1000),warmups=number(config,"warmups",1,1000);
  if(uint64_t(m)*n*k>134217728)throw std::invalid_argument("full oracle work budget exceeded");
  const std::string backend=config.at("backend"),profile=config.at("coefficient_profile"),residency=config.at("residency");
  if((backend!="cpu" && backend!="hip-direct") || (residency!="resident" && residency!="repack"))throw std::invalid_argument("unknown backend/residency");
  int64_t alpha=0,beta=0;if(!rns8::bench::epilogue_coefficients(profile,false,alpha,beta))throw std::invalid_argument("unknown coefficient profile");
  const uint32_t prefix=20,limbs=3,ld=n+2;const uint64_t sentinel=0x5a5a5a5a5a5a5a5aull;
  std::mt19937_64 rng(4747);std::vector<int64_t>a(size_t(m)*k),b(size_t(k)*n),d(size_t(m)*n);
  for(auto* input:{&a,&b,&d})for(auto& value:*input)value=int64_t(rng()%33)-16;
  std::vector<cpp_int> expected(size_t(m)*n);for(uint32_t row=0;row<m;++row)for(uint32_t col=0;col<n;++col){
    cpp_int sum=0;for(uint32_t q=0;q<k;++q)sum+=cpp_int(a[size_t(row)*k+q])*b[size_t(q)*n+col];
    expected[size_t(row)*n+col]=cpp_int(alpha)*sum+cpp_int(beta)*d[size_t(row)*n+col];
  }
  Owner owner;hip_direct_timing_set_enabled(false);const auto setup_start=Clock::now();
  rns8_context_options options{};options.struct_size=sizeof(options);options.abi_version=RNS8_ABI_VERSION;options.requested_backend=backend=="cpu"?RNS8_BACKEND_CPU_REFERENCE:RNS8_BACKEND_HIP_DIRECT;
  check(rns8_create_context(backend=="cpu"?-1:0,&options,&owner.ctx));
  rns8_gemm_desc desc{};desc.struct_size=sizeof(desc);desc.abi_version=RNS8_ABI_VERSION;desc.semantics=RNS8_EXACT_WIDE_SIGNED;desc.bound_kind=RNS8_BOUND_NONE;desc.requested_backend=options.requested_backend;desc.m=m;desc.n=n;desc.k=k;desc.max_prefix=prefix;
  check(rns8_create_plan(owner.ctx,&desc,&owner.plan));check(rns8_create_workspace(owner.ctx,owner.plan,&owner.work));
  auto matrix=[&](uint32_t rows,uint32_t cols,rns8_matrix** out){rns8_matrix_desc md{};md.struct_size=sizeof(md);md.abi_version=RNS8_ABI_VERSION;md.semantics=desc.semantics;md.bound_kind=desc.bound_kind;md.logical_layout=RNS8_LAYOUT_ROW_MAJOR;md.rows=rows;md.cols=cols;md.logical_ld=cols;md.max_prefix=prefix;check(rns8_create_matrix(owner.ctx,&md,out));};
  matrix(m,k,&owner.a);matrix(k,n,&owner.b);matrix(m,n,&owner.d);matrix(m,n,&owner.c);
  uint64_t generation=0;
  auto pack=[&](){++generation;check(rns8_pack_i64(owner.ctx,owner.a,a.data(),k,generation));check(rns8_pack_i64(owner.ctx,owner.b,b.data(),n,generation));check(rns8_pack_i64(owner.ctx,owner.d,d.data(),n,generation));};
  pack();std::vector<uint64_t> output(size_t(m)*ld*limbs,sentinel);
  const double setup_us=us(setup_start,Clock::now());
  auto verify=[&](){
    std::vector<int8_t> residues(size_t(prefix)*m*n);
    if(backend=="cpu")residues=owner.c->residues;
    else check(hip_direct_copy_device_to_host(0,residues.data(),owner.c->hip_residues,residues.size()));
    const cpp_int mask=(cpp_int(1)<<64)-1,storage=cpp_int(1)<<192;
    for(uint32_t row=0;row<m;++row)for(uint32_t col=0;col<ld;++col){
      if(col>=n){for(uint32_t limb=0;limb<limbs;++limb)if(output[(size_t(row)*ld+col)*limbs+limb]!=sentinel)throw std::runtime_error("padding changed");continue;}
      cpp_int value=expected[size_t(row)*n+col];if(value<0)value+=storage;
      for(uint32_t limb=0;limb<limbs;++limb){if(output[(size_t(row)*ld+col)*limbs+limb]!=(value&mask).convert_to<uint64_t>())throw std::runtime_error("exact public output mismatch");value>>=64;}
      for(uint32_t p=0;p<prefix;++p){const uint32_t modulus=rns8_default_modulus(p);cpp_int r=expected[size_t(row)*n+col]%modulus;if(r<0)r+=modulus;if(r>=(modulus+1)/2)r-=modulus;if(residues[(size_t(p)*m+row)*n+col]!=r.convert_to<int>())throw std::runtime_error("residue channel mismatch");}
    }
  };
  auto execute=[&](){const auto start=Clock::now();if(residency=="repack")pack();const auto packed=Clock::now();
    check(rns8_gemm_exact_axpby(owner.ctx,owner.plan,owner.a,owner.b,owner.d,owner.c,alpha,beta,owner.work));const auto computed=Clock::now();
    check(rns8_export_exact_wide_signed_limbs(owner.ctx,owner.plan,owner.c,output.data(),ld,limbs));const auto stop=Clock::now();
    return std::array<double,4>{us(start,packed),us(packed,computed),us(computed,stop),us(start,stop)};};
  auto sample_json=[](const std::array<double,4>& s){return json{{"pack_us",s[0]},{"expression_us",s[1]},{"export_us",s[2]},{"total_us",s[3]}};};
  const auto first=sample_json(execute());verify();for(uint32_t i=0;i<warmups;++i)execute();
  json samples=json::array();for(uint32_t i=0;i<repeats;++i){samples.push_back(sample_json(execute()));verify();}
  cpp_int bound;const auto proof_start=Clock::now();check(exact_axpby_range(*owner.plan,*owner.a,*owner.b,*owner.d,alpha,beta,bound));const double proof_us=us(proof_start,Clock::now());
  json sustained=json::array();for(unsigned batch=0;batch<3;++batch){const auto first_generation=generation;const auto start=Clock::now();uint32_t calls=0;double duration=0;
    do{execute();++calls;duration=us(start,Clock::now());}while(calls<4096 && (calls<16 || duration<250000));verify();
    sustained.push_back({{"calls",calls},{"duration_us",duration},{"first_input_generation",first_generation},{"last_input_generation",generation},{"exact_reference_passed",true}});
  }
  unsigned dot_channels=0,d_channels=0;for(unsigned p=0;p<prefix;++p){const auto coef=lower_residue_axpby(alpha,beta,rns8_default_modulus(p));dot_channels+=coef.alpha!=0;d_channels+=coef.beta!=0;}
  rns8_device_info device{};device.struct_size=sizeof(device);device.abi_version=RNS8_ABI_VERSION;check(rns8_get_device_info(owner.ctx,&device));
  json result={{"schema","rns8_public_epilogue_microcase_v1"},{"case",config},{"run_id",argv[2]},
    {"build_manifest",json::parse(RNS8_BUILD_MANIFEST_JSON)},{"device",device.name},{"target",device.gcn_arch},{"runtime_version",device.hip_runtime_version},
    {"dataset","signed_small_mt19937_64_seed4747_v1"},{"seed",4747},{"alpha",alpha},{"beta",beta},{"prefix",prefix},{"limbs",limbs},{"output_ld",ld},
    {"host_instrumentation_state","hooks_disabled"},{"output_boundary","complete_public_expression_and_padded_signed_limb_export"},
    {"setup_us",setup_us},{"setup_scope","context_plan_workspace_matrices_initial_three_input_pack"},{"first_call",first},{"samples",samples},{"sustained",sustained},
    {"proof_diagnostic_us",proof_us},{"proof_diagnostic_scope","separate_duplicate_host_range_check_not_subtracted_from_public_expression"},{"expression_abs_bound",bound.str()},
    {"active_dot_channels",dot_channels},{"active_addend_channels",d_channels},{"source_scalar_dot_terms_per_call",uint64_t(m)*n*k*dot_channels},{"source_counts_are_measured_instructions",false},
    {"exact_reference_passed",true},{"oracle_inside_production_interval",false},{"promotion_eligible",false},{"clock_period_numerator",Clock::period::num},{"clock_period_denominator",Clock::period::den}};
  std::cout<<result.dump(2)<<'\n';return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
