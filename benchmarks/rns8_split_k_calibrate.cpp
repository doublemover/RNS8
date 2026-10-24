// Public split-K ownership controls. No private runtime or kernel interfaces.
#include <rns8/rns8.h>
#include <rns8/moduli.h>
#include <boost/multiprecision/cpp_int.hpp>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
namespace {
using json=nlohmann::json;
using boost::multiprecision::cpp_int;
using boost::multiprecision::int256_t;
using Clock=std::chrono::steady_clock;
static_assert(sizeof(int256_t)<=128,"logical memory admission requires bounded fixed Boost operand storage");
constexpr uint64_t budget=512ull*1024*1024,canary=UINT64_C(0xa5a5a5a5a5a5a5a5);
bool cleanup_failed=false;
void require(bool value,const char* why){if(!value)throw std::runtime_error(why);}
void check(rns8_status status){if(status!=RNS8_SUCCESS)throw std::runtime_error(rns8_status_string(status));}
struct Unsupported:std::runtime_error{using std::runtime_error::runtime_error;};
template<class T,rns8_status(*Destroy)(T*)>struct Delete{void operator()(T* p)const noexcept{if(p&&Destroy(p)!=RNS8_SUCCESS)cleanup_failed=true;}};
template<class T,rns8_status(*Destroy)(T*)>using Owner=std::unique_ptr<T,Delete<T,Destroy>>;
using Context=Owner<rns8_context,rns8_destroy_context>;
using Plan=Owner<rns8_plan,rns8_destroy_plan>;
using Matrix=Owner<rns8_matrix,rns8_destroy_matrix>;
using Workspace=Owner<rns8_workspace,rns8_destroy_workspace>;
using Split=Owner<rns8_split_k,rns8_destroy_split_k>;
uint64_t ns(Clock::time_point start){return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-start).count());}
uint64_t number(const char* s,uint64_t low,uint64_t high){uint64_t value=0;const auto end=s+std::strlen(s);auto result=std::from_chars(s,end,value);if(result.ec!=std::errc{}||result.ptr!=end||value<low||value>high)throw std::invalid_argument("numeric argument outside supported domain");return value;}
struct Case{int device;std::string mode,domain,run;size_t m,n,k;uint32_t fragments,window,reuse,repeats,warmups;};
void help(){std::cout<<"rns8-split-k-calibrate DEVICE unsplit|full|window M N K signed|unsigned FRAGMENTS WINDOW REUSE REPEATS WARMUPS RUN_ID\n"
 "Explicit HIP DEVICE0..63, no fallback. M,N1..512 K1..65537; M*N*K*REUSE<=134217728.\n"
 "Unsplit requires FRAGMENTS=1 WINDOW=0. Full requires WINDOW=FRAGMENTS. Window requires1<=WINDOW<FRAGMENTS.\n"
 "Fragments1..min(K,4096), ceil(K/fragments)<=65536 for split modes. REUSE1..256 REPEATS/WARMUPS1..1000.\n"
 "Exact-wide prefix20, full-width deterministic inputs and Boost oracle; independent export checks outside every timed interval.\n"
 "Two input generations: preparation and first-use, unchanged repeated and sustained calls, then explicit recreation after repack.\n"
 "Public synchronous wall cost; no kernel-event or hardware-overlap inference. Run ID ASCII alnum/_/- up to128.\n";}
Case parse(int argc,char** argv){require(argc==13,"expected twelve positional arguments; use --help");Case c{};c.device=int(number(argv[1],0,63));c.mode=argv[2];c.domain=argv[6];c.run=argv[12];
 c.m=size_t(number(argv[3],1,512));c.n=size_t(number(argv[4],1,512));c.k=size_t(number(argv[5],1,65537));
 c.fragments=uint32_t(number(argv[7],1,4096));c.window=uint32_t(number(argv[8],0,4096));c.reuse=uint32_t(number(argv[9],1,256));c.repeats=uint32_t(number(argv[10],1,1000));c.warmups=uint32_t(number(argv[11],1,1000));
 require(c.domain=="signed"||c.domain=="unsigned","signedness must be explicit");require(c.mode=="unsplit"||c.mode=="full"||c.mode=="window","invalid mode");
 require(c.mode=="unsplit"?c.fragments==1&&c.window==0:c.fragments<=c.k&&(c.k+c.fragments-1)/c.fragments<=65536&&(c.mode=="full"?c.window==c.fragments:c.window>0&&c.window<c.fragments),"irrelevant or inadmissible fragment/window configuration");
 require(uint64_t(c.m)*c.n*c.k*c.reuse<=134217728,"reference/batch work exceeds bounded domain");
 require(!c.run.empty()&&c.run.size()<=128&&c.run.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")==std::string::npos,"invalid run ID");return c;}
uint64_t next(uint64_t& state){uint64_t z=(state+=UINT64_C(0x9e3779b97f4a7c15));z=(z^(z>>30))*UINT64_C(0xbf58476d1ce4e5b9);z=(z^(z>>27))*UINT64_C(0x94d049bb133111eb);return z^(z>>31);}
int64_t signed_value(uint64_t value){int64_t out;std::memcpy(&out,&value,8);return out;}
std::string checksum(const std::vector<uint64_t>& values){uint64_t h=UINT64_C(14695981039346656037);for(auto v:values)for(unsigned byte=0;byte<8;++byte)h=(h^uint8_t(v>>(8*byte)))*UINT64_C(1099511628211);std::ostringstream out;out<<std::hex<<std::setfill('0')<<std::setw(16)<<h;return out.str();}
std::vector<uint64_t> oracle(const Case& c,const std::vector<uint64_t>& a,const std::vector<uint64_t>& b){
 const auto val=[&](uint64_t x){return c.domain=="signed"?int256_t(signed_value(x)):int256_t(x);};
 std::vector<int256_t> aa(a.size()),bt(b.size());for(size_t i=0;i<a.size();++i)aa[i]=val(a[i]);for(size_t t=0;t<c.k;++t)for(size_t col=0;col<c.n;++col)bt[col*c.k+t]=val(b[t*c.n+col]);
 std::vector<uint64_t> out(3*c.m*c.n);const int256_t mask=(int256_t(1)<<64)-1;
 for(size_t r=0;r<c.m;++r)for(size_t col=0;col<c.n;++col){int256_t sum=0;for(size_t t=0;t<c.k;++t)sum+=aa[r*c.k+t]*bt[col*c.k+t];
  if(col==0||col+1==c.n){cpp_int audit=0;for(size_t t=0;t<c.k;++t)audit+=cpp_int(val(a[r*c.k+t]))*cpp_int(val(b[t*c.n+col]));require(cpp_int(sum)==audit,"Boost fixed reference failed cpp_int edge audit");}
  // K<=65537 and products<2^128 prove |sum|<2^145. Three limbs suffice.
  if(sum<0)sum+=int256_t(1)<<192;for(unsigned limb=0;limb<3;++limb){out[(r*c.n+col)*3+limb]=(sum&mask).convert_to<uint64_t>();sum>>=64;}}
 return out;
}
rns8_matrix_storage_info storage(const rns8_matrix* p){rns8_matrix_storage_info info{};info.struct_size=sizeof(info);info.abi_version=RNS8_ABI_VERSION;check(rns8_get_matrix_storage_info(p,&info));return info;}
json allocation(const rns8_matrix* p){rns8_matrix_allocation_info a{};a.struct_size=sizeof(a);a.abi_version=RNS8_ABI_VERSION;check(rns8_get_matrix_allocation_info(p,&a));return {{"logical_primary_bytes",a.logical_primary_bytes},{"host_payload_capacity_bytes",a.host_payload_capacity_bytes},{"device_payload_capacity_bytes",a.device_payload_capacity_bytes},{"device_upload_staging_bytes",a.device_upload_staging_bytes},{"device_export_staging_bytes",a.device_export_staging_bytes},{"device_status_and_schedule_bytes",a.device_status_and_schedule_bytes},{"device_total_capacity_bytes",a.device_total_capacity_bytes}};}
json resources(const rns8_split_k_resources& r){return {{"fragments",r.fragments},{"fragment_window",r.fragment_window},{"window_count",r.window_count},{"partial_bytes",r.partial_bytes},{"merge_bytes",r.merge_bytes},{"running_bytes",r.running_bytes},{"scratch_bytes",r.scratch_bytes},{"merge_levels",r.merge_levels}};}
json run(const Case& c){const auto semantics=c.domain=="signed"?RNS8_EXACT_WIDE_SIGNED:RNS8_EXACT_WIDE_UNSIGNED;
 const uint64_t ab=uint64_t(c.m)*c.k+uint64_t(c.k)*c.n,mn=uint64_t(c.m)*c.n;
 rns8_split_k_resources query{};query.struct_size=sizeof(query);query.abi_version=RNS8_ABI_VERSION;
 if(c.mode!="unsplit")check(c.mode=="full"?rns8_query_split_k(c.m,c.n,c.k,20,c.fragments,budget,&query):rns8_query_split_k_windowed(c.m,c.n,c.k,20,c.fragments,c.window,budget,&query));
 const uint64_t oracle_peak=8*ab+ab*sizeof(int256_t)+24*mn+2048;
 const uint64_t steady=256*ab+256*mn+query.scratch_bytes+16384;
 require(std::max(oracle_peak,steady)<=budget,"aggregate logical payload exceeds512MiB");
 cpp_int product=1;for(unsigned p=0;p<20;++p)product*=rns8_default_modulus(p);require(2*(cpp_int(c.k)<<(c.domain=="signed"?126:128))<product,"prefix20 does not admit conservative full-expression bound");
 const auto data_start=Clock::now();std::vector<uint64_t> a(c.m*c.k),b(c.k*c.n);uint64_t state=5117;
 for(auto* values:{&a,&b})for(size_t i=0;i<values->size();++i){auto v=next(state);if(i%257==0)v=c.domain=="signed"?UINT64_C(0x8000000000000000):UINT64_MAX;else if(i%257==1)v=c.domain=="signed"?uint64_t(INT64_MAX):0;(*values)[i]=v;}
 auto gold=oracle(c,a,b);const auto first_oracle_ns=ns(data_start);
 const auto setup_start=Clock::now();rns8_context_options options{};options.struct_size=sizeof(options);options.abi_version=RNS8_ABI_VERSION;options.requested_backend=RNS8_BACKEND_HIP_DIRECT;rns8_context* raw=nullptr;
 auto status=rns8_create_context(c.device,&options,&raw);if(status==RNS8_UNSUPPORTED_BACKEND||status==RNS8_UNSUPPORTED_ARCH||status==RNS8_UNSUPPORTED_OS)throw Unsupported(rns8_status_string(status));check(status);Context ctx(raw);
 rns8_device_info device{};device.struct_size=sizeof(device);device.abi_version=RNS8_ABI_VERSION;check(rns8_get_device_info(ctx.get(),&device));require(device.backend==RNS8_BACKEND_HIP_DIRECT&&device.device_id==c.device&&device.gcn_arch[0],"requested HIP identity unavailable");
 rns8_gemm_desc pd{};pd.struct_size=sizeof(pd);pd.abi_version=RNS8_ABI_VERSION;pd.semantics=semantics;pd.bound_kind=RNS8_BOUND_NONE;pd.requested_backend=RNS8_BACKEND_HIP_DIRECT;pd.m=c.m;pd.n=c.n;pd.k=c.k;pd.max_prefix=20;pd.flags=RNS8_PLAN_FORCE_FIXED_PREFIX;
 rns8_plan* pr=nullptr;check(rns8_create_plan(ctx.get(),&pd,&pr));Plan plan(pr);rns8_plan_backend_info pi{};pi.struct_size=sizeof(pi);pi.abi_version=RNS8_ABI_VERSION;check(rns8_get_plan_backend_info(plan.get(),&pi));require(pi.backend==RNS8_BACKEND_HIP_DIRECT,"plan backend changed");
 require(pi.workspace_required_bytes<=budget-steady,"workspace exceeds aggregate logical payload budget");
 const auto envelope=std::max(oracle_peak,steady+pi.workspace_required_bytes);
 rns8_workspace* wr=nullptr;check(rns8_create_workspace(ctx.get(),plan.get(),&wr));Workspace workspace(wr);
 const auto matrix=[&](size_t rows,size_t cols){rns8_matrix_desc d{};d.struct_size=sizeof(d);d.abi_version=RNS8_ABI_VERSION;d.rows=rows;d.cols=cols;d.logical_ld=cols;d.logical_layout=RNS8_LAYOUT_ROW_MAJOR;d.semantics=semantics;d.bound_kind=RNS8_BOUND_NONE;d.max_prefix=20;rns8_matrix* p=nullptr;check(rns8_create_matrix(ctx.get(),&d,&p));return Matrix(p);};
 auto A=matrix(c.m,c.k),B=matrix(c.k,c.n),C=matrix(c.m,c.n);Split owner;
 const auto pack=[&](uint64_t version){for(const auto& pair:{std::make_pair(A.get(),&a),std::make_pair(B.get(),&b)}){
  const int64_t ld=pair.first==A.get()?int64_t(c.k):int64_t(c.n);
  if(c.domain=="signed"){std::vector<int64_t> converted(pair.second->size());for(size_t i=0;i<converted.size();++i)converted[i]=signed_value((*pair.second)[i]);check(rns8_pack_i64(ctx.get(),pair.first,converted.data(),ld,version));}
  else check(rns8_pack_u64(ctx.get(),pair.first,pair.second->data(),ld,version));}};
 const auto prepare=[&](){if(c.mode=="unsplit")return;rns8_split_k* p=nullptr;check(c.mode=="full"?rns8_create_split_k(ctx.get(),A.get(),B.get(),c.fragments,query.scratch_bytes,&p):rns8_create_split_k_windowed(ctx.get(),A.get(),B.get(),c.fragments,c.window,query.scratch_bytes,&p));owner.reset(p);};
 const auto destroy_owner=[&](){if(owner){check(rns8_destroy_split_k(owner.get()));owner.release();}};
 pack(1);const auto prepare_start=Clock::now();prepare();const auto prepare_ns=c.mode=="unsplit"?uint64_t(0):ns(prepare_start);const auto initial_setup_ns=ns(setup_start);
 const std::array<uint64_t,3> ids{{storage(A.get()).matrix_instance_id,storage(B.get()).matrix_instance_id,storage(C.get()).matrix_instance_id}};
 const auto allocations=[&](){json result=json::array();uint64_t total=query.scratch_bytes+pi.workspace_required_bytes;for(auto* p:{A.get(),B.get(),C.get()}){auto value=allocation(p);total+=value["host_payload_capacity_bytes"].get<uint64_t>()+value["device_total_capacity_bytes"].get<uint64_t>();result.push_back(value);}require(total<=envelope,"queried persistent capacities exceed conservative envelope");return result;};
 const auto at_setup=allocations();
 const auto invoke=[&](){for(uint32_t i=0;i<c.reuse;++i)check(c.mode=="unsplit"?rns8_gemm_rns(ctx.get(),plan.get(),A.get(),B.get(),C.get(),workspace.get()):rns8_execute_split_k(owner.get(),C.get()));};
 const auto audit=[&](uint64_t generation){std::vector<uint64_t> actual(3*c.m*(c.n+3),canary);
  check(c.domain=="signed"?rns8_export_exact_wide_signed_limbs(ctx.get(),plan.get(),C.get(),actual.data(),int64_t(c.n+3),3):rns8_export_exact_wide_unsigned_limbs(ctx.get(),plan.get(),C.get(),actual.data(),int64_t(c.n+3),3));
  for(size_t r=0;r<c.m;++r)for(size_t col=0;col<c.n+3;++col)for(size_t limb=0;limb<3;++limb)require(actual[(r*(c.n+3)+col)*3+limb]==(col<c.n?gold[(r*c.n+col)*3+limb]:canary),"complete Boost product or export padding mismatch");
  size_t index=0;for(auto* p:{A.get(),B.get(),C.get()}){auto info=storage(p);require(info.matrix_instance_id==ids[index++]&&info.device_residues_current,"matrix identity or currentness changed");if(p!=C.get())require(info.source_version==generation,"source version mismatch");}};
 json phases=json::array();uint64_t phase_setup=initial_setup_ns,reprepare=prepare_ns,pack_ns=0;json stale=nullptr;
 for(uint64_t generation=1;generation<=2;++generation){uint64_t oracle_ns=first_oracle_ns;
  if(generation==2){a[0]^=UINT64_C(0x0123456789abcdef);const auto t=Clock::now();auto changed_gold=oracle(c,a,b);oracle_ns=ns(t);
   auto start=Clock::now();pack(2);pack_ns=ns(start);
   if(owner){const auto rejected=rns8_execute_split_k(owner.get(),C.get());require(rejected==RNS8_INVALID_ARGUMENT,"stale owner unexpectedly admitted repacked inputs");stale="RNS8_INVALID_ARGUMENT";audit(2);}
   gold=std::move(changed_gold);
   start=Clock::now();destroy_owner();prepare();reprepare=c.mode=="unsplit"?uint64_t(0):ns(start);phase_setup=pack_ns+reprepare;
  }
  const auto cold_start=Clock::now();invoke();const auto first=ns(cold_start);audit(generation);
  for(uint32_t i=0;i<c.warmups;++i){invoke();audit(generation);}json samples=json::array();std::vector<uint64_t> times;
  for(uint32_t i=0;i<c.repeats;++i){const auto start=Clock::now();invoke();const auto elapsed=ns(start);require(elapsed>0,"host timer below resolution");audit(generation);times.push_back(elapsed);samples.push_back({{"wall_ns",elapsed},{"exact_reference_passed",true}});}
  const auto sustained_start=Clock::now();uint64_t calls=0,elapsed=0;do{invoke();++calls;elapsed=ns(sustained_start);}while(calls<4096&&(calls<16||elapsed<250000000));audit(generation);
  std::sort(times.begin(),times.end());const double median=times.size()%2?double(times[times.size()/2]):(double(times[times.size()/2-1])+double(times[times.size()/2]))/2;
  phases.push_back({{"generation",generation},{"source_versions",{generation,generation}},{"binding_action",generation==1?(c.mode=="unsplit"?"initial_pack":"initial_create"):(c.mode=="unsplit"?"repack":"repack_destroy_recreate")},
   {"stale_owner_status",stale},{"stale_owner_preserved_output",stale.is_null()?json(nullptr):json(true)},{"a_checksum",checksum(a)},{"b_checksum",checksum(b)},{"output_checksum",checksum(gold)},{"oracle_ns",oracle_ns},{"phase_setup_ns",phase_setup},{"changed_input_pack_ns",pack_ns},{"owner_prepare_or_recreate_ns",reprepare},
   {"first_batch_ns",first},{"setup_plus_first_batch_ns",phase_setup+first},{"first_batch_exact_reference_passed",true},{"samples",samples},{"median_batch_ns",median},{"gemm_calls_per_second",1e9*c.reuse/median},{"matrix_allocations_after_audit",allocations()},
   {"sustained",{{"calls",calls},{"gemm_calls",calls*c.reuse},{"elapsed_ns",elapsed},{"mean_batch_ns",double(elapsed)/calls},{"minimum_calls",16},{"maximum_calls",4096},{"target_ns",250000000},{"stop_reason",elapsed>=250000000?"duration":"call_cap"},{"independent_audit_inside_interval",false},{"final_exact_reference_passed",true}}}});
 }
 const auto release_start=Clock::now();destroy_owner();const auto release_ns=c.mode=="unsplit"?uint64_t(0):ns(release_start);
 return {{"schema","rns8_split_k_microcase_v1"},{"promotion_eligible",false},{"run_id",c.run},{"mode",c.mode},{"device",c.device},{"device_actual",device.device_id},{"backend_requested","hip-direct"},{"backend_actual","hip-direct"},{"device_name",device.name},{"target",device.gcn_arch},{"hip_runtime_version",device.hip_runtime_version},{"hip_driver_version",device.hip_driver_version},{"build_manifest",json::parse(RNS8_BUILD_MANIFEST_JSON)},
  {"m",c.m},{"n",c.n},{"k",c.k},{"domain",c.domain},{"semantics",c.domain=="signed"?"exact_wide_signed":"exact_wide_unsigned"},{"prefix",20},{"fragments",c.fragments},{"window",c.window},{"reuse",c.reuse},{"repeats",c.repeats},{"warmups",c.warmups},{"dataset_recipe","splitmix64_fullwidth_seed5117_xor_A0_generation2_v1"},{"dataset_seed",5117},{"generation2_a0_xor",UINT64_C(0x0123456789abcdef)},
  {"source_instance_ids",{ids[0],ids[1]}},{"output_instance_id",ids[2]},{"content_generations",nullptr},{"generation_contract","opaque_public_owner_generation_binding_source_version_is_separate"},{"split_resources",c.mode=="unsplit"?json(nullptr):resources(query)},
  {"reference_method","boost_int256_full_product_with_cpp_int_row_edge_audits"},{"reference_fixed_operand_object_bytes",sizeof(int256_t)},{"reference_operand_storage_bytes",ab*sizeof(int256_t)},{"reference_buffers_released_before_timing",true},{"output_contract","three_little_endian_uint64_limbs_per_cell_twos_complement_for_signed"},
  {"timer","std_chrono_steady_clock_nanoseconds"},{"clock_period_numerator",Clock::period::num},{"clock_period_denominator",Clock::period::den},{"timed_scope","reuse_public_synchronous_compute_only_no_export_or_oracle"},{"setup_scope","generation1_context_plan_workspace_matrices_pack_owner_generation2_sum_pack_and_checked_recreate_excludes_stale_probe"},{"kernel_events",nullptr},{"production_timing_hooks_enabled",false},{"observed_overlap",nullptr},{"selected_unsplit_kernel",pi.selected_kernel},
  {"memory_budget_bytes",budget},{"logical_live_payload_envelope_bytes",envelope},{"memory_exclusions","allocator_bookkeeping_spare_capacity_context_runtime_fixed_metadata_not_process_RSS_or_measured_peak"},{"plan_workspace_bytes",pi.workspace_required_bytes},{"matrix_allocations_at_setup",at_setup},{"allocation_scope","queried_matrix_payload_and_cached_staging_plus_separate_split_scratch_not_peak"},{"owner_release_ns",release_ns},{"exact_full_output_and_padding_checked",true},{"verified_outputs_per_batch",1},{"phases",phases}};
}
}
int main(int argc,char** argv){if(argc==2&&std::string(argv[1])=="--help"){help();return 0;}try{const auto result=run(parse(argc,argv));require(!cleanup_failed,"public handle cleanup failed");std::cout<<result.dump()<<'\n';return 0;}catch(const Unsupported& e){std::cerr<<"unsupported explicit HIP selection: "<<e.what()<<'\n';return 77;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
