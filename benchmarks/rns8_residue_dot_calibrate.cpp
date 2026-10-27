// Same production scheduler, two separately stamped arithmetic-only builds.
#include "core/api_internal.hpp"
#include <nlohmann/json.hpp>
#include <chrono>
#include <fstream>
#include <iostream>
#include <cstring>
namespace {
using json=nlohmann::json;using Clock=std::chrono::steady_clock;using boost::multiprecision::cpp_int;
void check(rns8_status s){if(s!=RNS8_SUCCESS)throw std::runtime_error(rns8_status_string(s));}
void require(bool x,const char* why){if(!x)throw std::invalid_argument(why);}
uint64_t number(const json& c,const char* key,uint64_t lo,uint64_t hi){require(c.at(key).is_number_integer(),key);const auto x=c.at(key).get<int64_t>();require(x>=0 && uint64_t(x)>=lo && uint64_t(x)<=hi,key);return uint64_t(x);}
double us(Clock::time_point a,Clock::time_point b){return std::chrono::duration<double,std::micro>(b-a).count();}
struct Sample {double compute_us,export_us,total_us;};
json sample_json(const Sample& s){return {{"compute_us",s.compute_us},{"export_us",s.export_us},{"total_us",s.total_us}};}
int64_t bits(uint64_t x){int64_t y;std::memcpy(&y,&x,8);return y;}
uint64_t random(uint64_t& s){uint64_t z=(s+=UINT64_C(0x9e3779b97f4a7c15));z=(z^(z>>30))*UINT64_C(0xbf58476d1ce4e5b9);z=(z^(z>>27))*UINT64_C(0x94d049bb133111eb);return z^(z>>31);}
}
int main(int argc,char** argv){try{
  using namespace rns8::detail;require(argc==3 || (argc==4 && std::string(argv[3])=="--diagnostic"),"usage: rns8-residue-dot-calibrate CASE.json RUN_ID [--diagnostic]");const bool diagnostic=argc==4;json c;std::ifstream(argv[1])>>c;
  require(c.size()==12 && c.at("schema")=="residue_dot_case_v1","case ABI");
  const unsigned m=number(c,"m",2,512),n=number(c,"n",2,512),k=number(c,"k",1,65537),prefix=number(c,"prefix",1,20),modulus=number(c,"modulus",0,256),tasks=number(c,"tasks",1,8),repeats=number(c,"repeats",3,100),warmups=number(c,"warmups",1,100);
  const std::string mode=c.at("mode"),domain=c.at("domain"),dataset=c.at("dataset");const bool finite=domain=="finite",wide=domain=="wide",region=mode=="region",scheduled=mode=="scheduled";
  require((finite || wide || domain=="bounded") && (mode=="independent" || mode=="grouped" || region || scheduled),"domain/mode");
  require((!finite || (modulus>=2 && prefix==1)) && (finite || modulus==0) && (!scheduled || (!finite && !wide)) && (!region || !wide),"scheduler semantics");
  require(dataset=="zero" || dataset=="sparse" || dataset=="centered" || dataset=="fullwidth","dataset");require(dataset!="fullwidth" || (wide && prefix==20),"fullwidth admission");
  require(uint64_t(m)*n*k*tasks<=134217728 && uint64_t(m*k+k*n+m*n)*tasks*256<UINT64_C(512)*1024*1024,"work/payload budget");
  const auto manifest=json::parse(RNS8_BUILD_MANIFEST_JSON);const std::string config=manifest.at("configuration");
  const bool dot=config.find("residue_dot4=ON\n")!=std::string::npos,scalar=config.find("residue_scalar_control=ON\n")!=std::string::npos;require(dot!=scalar,"exactly one explicit arithmetic build required; legacy OFF/OFF is not a scalar control");
  std::vector<std::vector<int64_t>> av(tasks),bv(tasks);std::vector<std::vector<cpp_int>> gold(tasks);cpp_int bound=0;uint64_t seed=7171;
  for(unsigned task=0;task<tasks;++task){av[task].resize(uint64_t(m)*(k+3),99);bv[task].resize(uint64_t(k)*(n+3),99);gold[task].resize(uint64_t(m)*n);
    for(unsigned r=0;r<m;++r)for(unsigned q=0;q<k;++q){int64_t x=0;if(dataset=="sparse")x=q==(r+task)%k?(r%2?-1:1):0;else if(dataset=="centered")x=int64_t(random(seed)%256)-128;else if(dataset=="fullwidth")x=bits(random(seed));av[task][uint64_t(r)*(k+3)+q]=finite?(x%int64_t(modulus)+modulus)%modulus:x;}
    for(unsigned q=0;q<k;++q)for(unsigned col=0;col<n;++col){int64_t x=0;if(dataset=="sparse")x=(q+col+task)%2?-1:1;else if(dataset=="centered")x=int64_t(random(seed)%256)-128;else if(dataset=="fullwidth")x=bits(random(seed));bv[task][uint64_t(q)*(n+3)+col]=finite?(x%int64_t(modulus)+modulus)%modulus:x;}
    for(unsigned r=0;r<m;++r)for(unsigned col=0;col<n;++col){cpp_int sum=0;for(unsigned q=0;q<k;++q)sum+=cpp_int(av[task][uint64_t(r)*(k+3)+q])*bv[task][uint64_t(q)*(n+3)+col];if(finite)sum%=modulus;gold[task][uint64_t(r)*n+col]=sum;const cpp_int magnitude=boost::multiprecision::abs(sum);if(magnitude>bound)bound=magnitude;}}
  require(finite || 2*bound<modulus_product(prefix),"strict output range");require(wide || finite || bound<=INT64_MAX,"bounded output width");
  const auto semantics=finite?RNS8_FINITE_RING_U8:wide?RNS8_EXACT_WIDE_SIGNED:RNS8_BOUNDED_I64;const auto bound_kind=finite||wide?RNS8_BOUND_NONE:scheduled?RNS8_BOUND_PER_TILE_MAX_ABS:RNS8_BOUND_GLOBAL_MAX_ABS;
  hip_direct_timing_set_enabled(false);const auto setup_begin=Clock::now();rns8_context_options co{};co.struct_size=sizeof(co);co.abi_version=RNS8_ABI_VERSION;co.requested_backend=RNS8_BACKEND_HIP_DIRECT;rns8_context* ctx=nullptr;check(rns8_create_context(0,&co,&ctx));
  rns8_gemm_desc pd{};pd.struct_size=sizeof(pd);pd.abi_version=RNS8_ABI_VERSION;pd.semantics=semantics;pd.bound_kind=bound_kind;pd.requested_backend=co.requested_backend;pd.m=m;pd.n=n;pd.k=k;pd.max_prefix=finite?0:prefix;pd.finite_modulus=modulus;pd.flags=finite?0:RNS8_PLAN_FORCE_FIXED_PREFIX;pd.bound=finite||wide||scheduled?0:bound.convert_to<uint64_t>();pd.tile_m=64;pd.tile_n=64;
  std::vector<uint64_t> tile_bounds;if(scheduled){tile_bounds.assign(uint64_t((m+63)/64)*((n+63)/64),bound.convert_to<uint64_t>());pd.tile_bounds=tile_bounds.data();pd.tile_bounds_count=tile_bounds.size();}
  rns8_plan* plan=nullptr;check(rns8_create_plan(ctx,&pd,&plan));
  require(!scheduled || (plan->desc.bound==0 && plan->tile_bounds==tile_bounds),"scheduled bound ownership");
  const json plan_contract={{"global_bound",plan->desc.bound},{"per_tile_bound_count",plan->tile_bounds.size()},{"schedule_entries",plan->tile_schedule.size()},{"min_selected_prefix",plan->schedule_min_selected_prefix},{"max_selected_prefix",plan->schedule_max_selected_prefix},{"min_required_prefix",plan->schedule_min_required_prefix},{"max_required_prefix",plan->schedule_max_required_prefix},{"effective_dispatch",scheduled?(plan->tile_schedule.empty()?"folded_uniform":"scheduled"):mode}};std::vector<rns8_matrix*> A(tasks),B(tasks),C(tasks);std::vector<rns8_workspace*> work(tasks);std::vector<rns8_result_cache*> cache(tasks);std::vector<rns8_grouped_gemm_task> grouped(tasks);
  auto matrix=[&](unsigned rows,unsigned cols,rns8_matrix** out){rns8_matrix_desc d{};d.struct_size=sizeof(d);d.abi_version=RNS8_ABI_VERSION;d.semantics=semantics;d.bound_kind=bound_kind;d.rows=rows;d.cols=cols;d.logical_ld=cols;d.logical_layout=RNS8_LAYOUT_ROW_MAJOR;d.tile_m=64;d.tile_n=64;d.max_prefix=finite?0:prefix;check(rns8_create_matrix(ctx,&d,out));};
  for(unsigned t=0;t<tasks;++t){matrix(m,k,&A[t]);matrix(k,n,&B[t]);matrix(m,n,&C[t]);check(rns8_create_workspace(ctx,plan,&work[t]));
    if(finite){std::vector<uint8_t> a(av[t].size()),b(bv[t].size());for(size_t i=0;i<a.size();++i)a[i]=uint8_t(av[t][i]);for(size_t i=0;i<b.size();++i)b[i]=uint8_t(bv[t][i]);check(rns8_pack_finite_u8(ctx,A[t],modulus,a.data(),k+3,1));check(rns8_pack_finite_u8(ctx,B[t],modulus,b.data(),n+3,1));}
    else{check(rns8_pack_i64(ctx,A[t],av[t].data(),k+3,1));check(rns8_pack_i64(ctx,B[t],bv[t].data(),n+3,1));}
    grouped[t].struct_size=sizeof(grouped[t]);grouped[t].abi_version=RNS8_ABI_VERSION;grouped[t].a=A[t];grouped[t].b=B[t];grouped[t].c=C[t];grouped[t].workspace=work[t];
    if(region){rns8_result_cache_desc d{};d.struct_size=sizeof(d);d.abi_version=RNS8_ABI_VERSION;d.max_dirty_regions=1;check(rns8_create_result_cache(ctx,plan,&d,&cache[t]));}}
  rns8_dirty_region dirty{};dirty.struct_size=sizeof(dirty);dirty.abi_version=RNS8_ABI_VERSION;dirty.row_offset=1;dirty.col_offset=1;dirty.row_extent=m-1;dirty.col_extent=n-1;
  auto compute=[&](bool initialize=false){if(mode=="grouped")check(finite?rns8_gemm_finite_u8_grouped(ctx,plan,modulus,grouped.data(),tasks):rns8_gemm_rns_grouped(ctx,plan,grouped.data(),tasks));else for(unsigned t=0;t<tasks;++t){
      if(region)check(finite?rns8_gemm_finite_u8_incremental(ctx,plan,modulus,A[t],B[t],C[t],work[t],cache[t],initialize?nullptr:&dirty,initialize?0:1):rns8_gemm_rns_incremental(ctx,plan,A[t],B[t],C[t],work[t],cache[t],initialize?nullptr:&dirty,initialize?0:1));
      else check(finite?rns8_gemm_finite_u8(ctx,plan,modulus,A[t],B[t],C[t],work[t]):rns8_gemm_rns(ctx,plan,A[t],B[t],C[t],work[t]));}};
  if(region)compute(true);const double setup=us(setup_begin,Clock::now());
  std::vector<std::vector<uint64_t>> exported(tasks,std::vector<uint64_t>(uint64_t(m)*(n+3)*3,99));std::vector<std::vector<int64_t>> narrow(tasks,std::vector<int64_t>(uint64_t(m)*(n+3),99));std::vector<std::vector<uint8_t>> bytes(tasks,std::vector<uint8_t>(uint64_t(m)*(n+3),99));
  auto export_outputs=[&](){for(unsigned t=0;t<tasks;++t){if(finite)check(rns8_export_finite_u8(ctx,plan,modulus,C[t],bytes[t].data(),n+3));else if(wide)check(rns8_export_exact_wide_signed_limbs(ctx,plan,C[t],exported[t].data(),n+3,3));else check(rns8_export_i64(ctx,plan,C[t],narrow[t].data(),n+3));}};
  auto invoke=[&](){const auto begin=Clock::now();compute();const auto done=Clock::now();export_outputs();const auto end=Clock::now();return Sample{us(begin,done),us(done,end),us(begin,end)};};
  auto audit=[&](){for(unsigned t=0;t<tasks;++t){for(unsigned r=0;r<m;++r)for(unsigned col=0;col<n+3;++col){const uint64_t pos=uint64_t(r)*(n+3)+col;if(col>=n){if(finite)require(bytes[t][pos]==99,"padding");else if(wide)for(unsigned l=0;l<3;++l)require(exported[t][pos*3+l]==99,"padding");else require(narrow[t][pos]==99,"padding");continue;}cpp_int got=finite?cpp_int(bytes[t][pos]):wide?cpp_int(exported[t][pos*3])+(cpp_int(exported[t][pos*3+1])<<64)+(cpp_int(exported[t][pos*3+2])<<128):cpp_int(narrow[t][pos]);if(wide && exported[t][pos*3+2]>>63)got-=cpp_int(1)<<192;require(got==gold[t][uint64_t(r)*n+col],"full Boost product");}
      std::vector<int8_t> residues(uint64_t(m)*n*(finite?1:prefix));check(hip_direct_copy_device_to_host(0,residues.data(),C[t]->hip_residues,residues.size()));for(unsigned p=0;p<(finite?1:prefix);++p)for(uint64_t cell=0;cell<uint64_t(m)*n;++cell){const unsigned q=finite?modulus:kDefaultModuli[p];cpp_int expected=gold[t][cell]%q;if(expected<0)expected+=q;if(expected>(q-1)/2)expected-=q;require(residues[uint64_t(p)*m*n+cell]==expected,"every residue plane");}
      if(region){rns8_result_cache_info info{};info.struct_size=sizeof(info);info.abi_version=RNS8_ABI_VERSION;check(rns8_get_result_cache_info(cache[t],&info));require(info.last_recomputed_region_count==1 && !info.last_cache_hit && !info.last_full_fallback,"region must recompute real nonzero rectangle, not a cache hit");}}};
  auto first=invoke();audit();for(unsigned i=0;i<warmups;++i){invoke();audit();}json samples=json::array();for(unsigned i=0;i<repeats;++i){samples.push_back(sample_json(invoke()));audit();}json sustained=json::array();for(unsigned repeat=0;repeat<3;++repeat){unsigned count=0;const auto begin=Clock::now();do{compute();export_outputs();++count;}while(count<4096 && (count<16 || us(begin,Clock::now())<250000));const auto elapsed=us(begin,Clock::now());audit();sustained.push_back({{"calls",count},{"total_us",elapsed},{"per_call_us",elapsed/count}});}
  const std::string target=ctx->device_info.gcn_arch,device=ctx->device_info.name;const auto runtime=ctx->device_info.hip_runtime_version;
  json events=nullptr;if(diagnostic){hip_direct_timing_set_enabled(true);hip_direct_timing_reset();invoke();hip_direct_timing_flush_pending_events();events=json::array();for(const auto& event:hip_direct_timing_snapshot())events.push_back({{"label",event.label},{"microseconds",event.microseconds}});hip_direct_timing_set_enabled(false);audit();}
  const auto selected=plan->backend_selected_kernel+"|public-dispatch="+mode;json launches=nullptr; // actual launch/instruction/resource counts require a separate diagnostic receipt
  for(unsigned t=0;t<tasks;++t){if(cache[t])check(rns8_destroy_result_cache(cache[t]));check(rns8_destroy_workspace(work[t]));for(auto* x:{A[t],B[t],C[t]})check(rns8_destroy_matrix(x));}check(rns8_destroy_plan(plan));check(rns8_destroy_context(ctx));
  const uint64_t cells=uint64_t(region?m-1:m)*(region?n-1:n)*tasks*(finite?1:prefix),groups=(uint64_t(k)+3)/4;
  std::cout<<json({{"schema","rns8_residue_dot_microcase_v3"},{"case",c},{"run_id",argv[2]},{"arithmetic",dot?"dot4":"scalar"},{"build_manifest",manifest},{"target",target},{"device",device},{"runtime_version",runtime},{"selected_kernel",selected},{"selected_kernel_scope","plan_identity_plus_explicit_public_scheduler_not_dynamic_profiler_evidence"},
    {"output_boundary",region?"public_incremental_nonzero_rectangle_with_full_padded_export":"public_complete_products_with_full_padded_export"},{"dataset_seed",7171},{"residency","unchanged_packed_inputs"},{"hooks_enabled",diagnostic},{"diagnostic_events",events},{"diagnostic_event_scope","separate_final_hook_enabled_invocation_group_spans_not_dynamic_launch_or_instruction_counts"},{"oracle_inside_interval",false},{"all_residue_planes_checked",true},{"exact_reference_passed",true},
    {"source_logical_scalar_updates",dot?0:cells*k},{"source_logical_dot4_updates",dot?cells*groups:0},{"source_count_scope","active_output_accumulator_updates_not_issued_instructions_inactive_lanes_or_total_gather_traffic"},{"actual_launches",launches},{"isa_and_resources",nullptr},
    {"plan_contract",plan_contract},{"sustained_body","compute_export_only_no_oracle_or_json"},{"setup_us",setup},{"first_call",sample_json(first)},{"samples",samples},{"sustained",sustained},{"promotion_eligible",false}}).dump(2)<<'\n';return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
