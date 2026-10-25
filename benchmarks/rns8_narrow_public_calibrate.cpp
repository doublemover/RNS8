#include "core/api_internal.hpp"
#include "backend_hip_direct/hip_backend.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
namespace {
using json=nlohmann::json;using Clock=std::chrono::steady_clock;using boost::multiprecision::cpp_int;
void check(rns8_status s){if(s!=RNS8_SUCCESS)throw std::runtime_error(rns8_status_string(s));}
double us(Clock::time_point a,Clock::time_point b){return std::chrono::duration<double,std::micro>(b-a).count();}
uint32_t number(const json& c,const char* key,uint32_t lo,uint32_t hi){if(!c.at(key).is_number_integer())throw std::invalid_argument(key);auto v=c.at(key).get<int64_t>();if(v<lo || v>hi)throw std::invalid_argument(key);return uint32_t(v);}
struct Owner {
  rns8_context* ctx=nullptr;rns8_plan* plan=nullptr;rns8_workspace* work=nullptr;rns8_matrix *a=nullptr,*b=nullptr,*c=nullptr;
  rns8_status close(){rns8_status s;for(auto** matrix:{&c,&b,&a})if(*matrix){s=rns8_destroy_matrix(*matrix);if(s!=RNS8_SUCCESS)return s;*matrix=nullptr;}
    if(work){s=rns8_destroy_workspace(work);if(s!=RNS8_SUCCESS)return s;work=nullptr;}if(plan){s=rns8_destroy_plan(plan);if(s!=RNS8_SUCCESS)return s;plan=nullptr;}
    if(ctx){s=rns8_destroy_context(ctx);if(s!=RNS8_SUCCESS)return s;ctx=nullptr;}return RNS8_SUCCESS;}
  ~Owner(){(void)close();}
};
}
int main(int argc,char** argv){try{
  if(argc!=3)throw std::invalid_argument("usage: rns8-narrow-public-calibrate CASE.json unique-run-id");json c;std::ifstream(argv[1])>>c;
  if(c.size()!=11 || c.at("schema")!="narrow_public_case_v1" || !c.at("unsigned_inputs").is_boolean())throw std::invalid_argument("case ABI");
  const uint32_t m=number(c,"m",1,512),n=number(c,"n",1,512),k=number(c,"k",1,131073),ceiling=number(c,"prefix",0,20),repeats=number(c,"repeats",3,100),warmups=number(c,"warmups",1,100),seed=number(c,"seed",0,UINT32_MAX);
  const bool u=c.at("unsigned_inputs");const std::string mode=c.at("mode"),residency=c.at("residency");
  const bool vector=mode=="vector-width8" || mode=="vector-exact-width",parallel=mode=="rns-parallel",narrow=mode.rfind("scalar-",0)==0 || mode.rfind("wmma-",0)==0;
  const bool wmma=mode=="wmma-staged" || mode=="wmma-fused",fused=mode=="scalar-fused" || mode=="wmma-fused";
  if((!vector && !parallel && !narrow && mode!="rns-public") || (narrow && mode!="scalar-staged" && mode!="scalar-fused" && !wmma) ||
     (u && narrow && !wmma) || (parallel && n>8) || (mode=="vector-exact-width" && (n<2 || n>8 || k<512)) ||
     (residency!="cold" && residency!="resident" && residency!="repack") || uint64_t(m)*n*k>134217728)throw std::invalid_argument("concrete route/domain");
  std::vector<uint64_t> av(size_t(m)*(k+1)),bv(size_t(k)*(n+2)),output(size_t(m)*(n+3)*3,UINT64_C(0x5a5a5a5a5a5a5a5a)),native(size_t(m)*(n+3));
  for(uint32_t r=0;r<m;++r)for(uint32_t q=0;q<k;++q){const int64_t v=int64_t((uint64_t(r)*71+uint64_t(q)*19+seed)%(u?8:15))-(u?0:7);av[size_t(r)*(k+1)+q]=uint64_t(v);}
  for(uint32_t q=0;q<k;++q)for(uint32_t col=0;col<n;++col){const int64_t v=int64_t((uint64_t(q)*37+uint64_t(col)*11+seed)%(u?8:15))-(u?0:7);bv[size_t(q)*(n+2)+col]=uint64_t(v);}
  std::vector<cpp_int> expected(size_t(m)*n);for(uint32_t r=0;r<m;++r)for(uint32_t col=0;col<n;++col)for(uint32_t q=0;q<k;++q)
    expected[size_t(r)*n+col]+=(u?cpp_int(av[size_t(r)*(k+1)+q]):cpp_int(int64_t(av[size_t(r)*(k+1)+q])))*(u?cpp_int(bv[size_t(q)*(n+2)+col]):cpp_int(int64_t(bv[size_t(q)*(n+2)+col])));
  const uint64_t bound=uint64_t(k)*49;uint32_t prefix=ceiling;if(!prefix){cpp_int product=1;for(prefix=1;prefix<=20;++prefix){product*=rns8_default_modulus(prefix-1);if(product>bound*(u?1:2))break;}}
  rns8::detail::hip_direct_timing_set_enabled(false);uint64_t generation=0;std::string target,device,kernel;int runtime=0;
  auto pack=[&](Owner& o){++generation;if(u){check(rns8_pack_u64(o.ctx,o.a,av.data(),k+1,generation));check(rns8_pack_u64(o.ctx,o.b,bv.data(),n+2,generation));}
    else {check(rns8_pack_i64(o.ctx,o.a,reinterpret_cast<const int64_t*>(av.data()),k+1,generation));check(rns8_pack_i64(o.ctx,o.b,reinterpret_cast<const int64_t*>(bv.data()),n+2,generation));}};
  auto create=[&](){auto o=std::make_unique<Owner>();rns8_context_options options{};options.struct_size=sizeof(options);options.abi_version=RNS8_ABI_VERSION;options.requested_backend=vector?RNS8_BACKEND_HIP_VECTOR_ALU_INT64:RNS8_BACKEND_HIP_DIRECT;check(rns8_create_context(0,&options,&o->ctx));
    rns8_gemm_desc d{};d.struct_size=sizeof(d);d.abi_version=RNS8_ABI_VERSION;d.semantics=vector?(u?RNS8_BOUNDED_U64:RNS8_BOUNDED_I64):(u?RNS8_EXACT_WIDE_UNSIGNED:RNS8_EXACT_WIDE_SIGNED);
    d.bound_kind=vector?(u?RNS8_BOUND_GLOBAL_MAX_UNSIGNED:RNS8_BOUND_GLOBAL_MAX_ABS):RNS8_BOUND_NONE;d.bound=vector?bound:0;d.requested_backend=options.requested_backend;d.m=m;d.n=n;d.k=k;d.max_prefix=prefix;
    d.flags=RNS8_PLAN_FORCE_FIXED_PREFIX|(parallel?RNS8_PLAN_DIRECT_RNS_PARALLEL:0)|(mode=="vector-exact-width"?RNS8_PLAN_VECTOR_SMALL_N_EXACT_WIDTH:0);
    check(rns8_create_plan(o->ctx,&d,&o->plan));check(rns8_create_workspace(o->ctx,o->plan,&o->work));
    auto matrix=[&](uint32_t rows,uint32_t cols,bool input,rns8_matrix** out){rns8_matrix_desc md{};md.struct_size=sizeof(md);md.abi_version=RNS8_ABI_VERSION;md.semantics=d.semantics;md.bound_kind=d.bound_kind;md.logical_layout=RNS8_LAYOUT_ROW_MAJOR;md.rows=rows;md.cols=cols;md.logical_ld=cols;md.max_prefix=prefix;md.flags=input && !vector?RNS8_MATRIX_RETAIN_NATIVE_INPUT:0;check(rns8_create_matrix(o->ctx,&md,out));};
    matrix(m,k,true,&o->a);matrix(k,n,true,&o->b);matrix(m,n,false,&o->c);pack(*o);
    rns8_device_info info{};info.struct_size=sizeof(info);info.abi_version=RNS8_ABI_VERSION;check(rns8_get_device_info(o->ctx,&info));target=info.gcn_arch;device=info.name;runtime=info.hip_runtime_version;
    rns8_plan_backend_info pi{};pi.struct_size=sizeof(pi);pi.abi_version=RNS8_ABI_VERSION;check(rns8_get_plan_backend_info(o->plan,&pi));kernel=pi.selected_kernel;return o;};
  const auto setup_start=Clock::now();auto owner=create();const double setup=us(setup_start,Clock::now());if(residency=="cold"){check(owner->close());owner.reset();}
  auto product=[&](Owner& o){if(narrow)check(rns8_gemm_narrow_exact(o.ctx,o.plan,o.a,o.b,o.c,static_cast<rns8_narrow_product_method>((wmma?1:0)+(fused?2:0)),UINT64_MAX,o.work));
    else check(rns8_gemm_rns(o.ctx,o.plan,o.a,o.b,o.c,o.work));};
  auto call=[&](){const auto start=Clock::now();if(residency=="cold")owner=create();else if(residency=="repack")pack(*owner);const auto prepared=Clock::now();
    product(*owner);const auto computed=Clock::now();
    if(vector){if(u)check(rns8_export_u64(owner->ctx,owner->plan,owner->c,native.data(),n+3));else check(rns8_export_i64(owner->ctx,owner->plan,owner->c,reinterpret_cast<int64_t*>(native.data()),n+3));
      for(uint32_t r=0;r<m;++r)for(uint32_t col=0;col<n;++col){const uint64_t value=native[size_t(r)*(n+3)+col];const size_t cell=(size_t(r)*(n+3)+col)*3;output[cell]=value;output[cell+1]=output[cell+2]=!u && (value>>63)?UINT64_MAX:0;}}
    else if(u)check(rns8_export_exact_wide_unsigned_limbs(owner->ctx,owner->plan,owner->c,output.data(),n+3,3));
    else check(rns8_export_exact_wide_signed_limbs(owner->ctx,owner->plan,owner->c,output.data(),n+3,3));const auto exported=Clock::now();
    if(residency=="cold"){check(owner->close());owner.reset();}const auto end=Clock::now();return json{{"prepare_us",us(start,prepared)},{"execute_us",us(prepared,computed)},{"export_us",us(computed,exported)},{"cleanup_us",us(exported,end)},{"total_us",us(start,end)}};};
  auto verify=[&](){for(uint32_t r=0;r<m;++r)for(uint32_t col=0;col<n+3;++col){cpp_int value=col<n?expected[size_t(r)*n+col]:0;if(value<0)value+=cpp_int(1)<<192;
    for(unsigned limb=0;limb<3;++limb){const auto expected_limb=col<n?uint64_t((value>>(64*limb))&UINT64_MAX):UINT64_C(0x5a5a5a5a5a5a5a5a);
      if(output[(size_t(r)*(n+3)+col)*3+limb]!=expected_limb)throw std::runtime_error("complete integer/padding oracle mismatch");}}};
  auto first=call();verify();for(uint32_t i=0;i<warmups;++i)call();json samples=json::array();for(uint32_t i=0;i<repeats;++i){samples.push_back(call());verify();}
  json sustained=json::array();for(unsigned i=0;i<3;++i){uint64_t calls=0;const auto start=Clock::now();do{call();++calls;}while(calls<4096 && (calls<16 || us(start,Clock::now())<250000));const double total=us(start,Clock::now());verify();sustained.push_back({{"calls",calls},{"total_us",total},{"per_call_us",total/calls},{"minimum_duration_reached",total>=250000}});}
  if(!vector){if(!owner){owner=create();product(*owner);}std::vector<int8_t> planes(size_t(prefix)*m*n);
    check(rns8::detail::hip_direct_copy_device_to_host(0,planes.data(),owner->c->hip_residues,planes.size()));
    for(uint32_t p=0;p<prefix;++p)for(size_t cell=0;cell<size_t(m)*n;++cell){const int modulus=rns8_default_modulus(p);cpp_int residue=expected[cell]%modulus;
      if(residue<0)residue+=modulus;if(residue>=(modulus+1)/2)residue-=modulus;if(planes[size_t(p)*m*n+cell]!=residue.convert_to<int>())throw std::runtime_error("all-plane exact audit mismatch");}}
  if(owner)check(owner->close());const uint64_t scratch=narrow?(wmma?32:16)+(fused?0:uint64_t(m)*n*8):0;
  std::cout<<json({{"schema","rns8_public_narrow_microcase_v1"},{"case",c},{"run_id",argv[2]},{"target",target},{"device",device},{"runtime_version",runtime},{"build_manifest",json::parse(RNS8_BUILD_MANIFEST_JSON)},
    {"dataset","affine_native_small_v1"},{"selected_prefix",prefix},{"ordinary_plan_kernel",kernel},{"narrow_producer",narrow?mode:"not_applicable"},{"known_narrow_scratch_bytes",scratch},
    {"output_boundary","complete_public_product_and_padded_three_limb_integer_adapter"},{"vector_export_adapter",vector?"public_native64_export_then_host_sign_or_zero_extension":"not_applicable"},
    {"host_instrumentation_state","hooks_disabled"},{"setup_us",setup},{"first_call",first},{"samples",samples},{"sustained",sustained},{"native_useful_products",uint64_t(m)*n*k},
    {"exact_reference_passed",true},{"all_residue_planes_audited",!vector},{"residue_audit_scope",vector?"not_applicable_native_control":"post_timing_public_product"},{"oracle_inside_production_interval",false},{"promotion_eligible",false}}).dump(2)<<'\n';return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}
