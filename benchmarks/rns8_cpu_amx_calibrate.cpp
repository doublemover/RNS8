// Explicit scalar/AMX comparison; no benchmark or ISA execution at build time.
#include "core/api_internal.hpp"
#include <nlohmann/json.hpp>
#include <chrono>
#include <cerrno>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <thread>
#include <set>
#include <cstdlib>
#if defined(__linux__)
#include <sys/utsname.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif
namespace {
using json=nlohmann::json;using Clock=std::chrono::steady_clock;using boost::multiprecision::cpp_int;
template<class T>T abi(){T v{};v.struct_size=sizeof(v);v.abi_version=RNS8_ABI_VERSION;return v;}
void check(rns8_status s){if(s!=RNS8_SUCCESS)throw std::runtime_error(rns8_status_string(s));}
void require(bool yes,const char* message){if(!yes)throw std::runtime_error(message);}
double us(Clock::time_point a,Clock::time_point b){return std::chrono::duration<double,std::micro>(b-a).count();}
uint32_t integer(const json& c,const char* key,uint32_t lo,uint32_t hi){require(c.at(key).is_number_integer(),key);const auto n=c.at(key).get<int64_t>();require(n>=lo && n<=hi,key);return uint32_t(n);}
std::string proc_field(const char* path,const char* field){std::ifstream in(path);std::string line;while(std::getline(in,line))if(line.rfind(field,0)==0){auto pos=line.find(':');return pos==std::string::npos?line:line.substr(pos+1);}return "unavailable";}
json environment(){json out{{"hardware_threads",std::thread::hardware_concurrency()},{"parallel_runtime",rns8::detail::cpu_parallel_runtime_name()},
    {"requested_threads",rns8::detail::cpu_parallel_requested_threads()},{"maximum_threads",rns8::detail::cpu_parallel_max_threads()},
    {"affinity_policy","inherited_no_pinning"},{"page_placement_observed",false}};
#if defined(__linux__)
  utsname os{};if(uname(&os)==0)out["os"]=std::string(os.sysname)+" "+os.release+" "+os.machine;else out["os"]="unavailable";
  std::set<std::pair<std::string,std::string>> physical;std::ifstream topology("/proc/cpuinfo");std::string line,package,core;
  while(std::getline(topology,line)){if(line.rfind("physical id",0)==0)package=line.substr(line.find(':')+1);if(line.rfind("core id",0)==0)core=line.substr(line.find(':')+1);
    if(line.empty()){if(!package.empty() && !core.empty())physical.emplace(package,core);package.clear();core.clear();}}
  if(!package.empty() && !core.empty())physical.emplace(package,core);
  out["physical_core_count"]=physical.empty()?json(nullptr):json(physical.size());out["topology_scope"]="proc_cpuinfo_visible_hardware_not_affinity_filtered";
  for(const char* key:{"OMP_PROC_BIND","OMP_PLACES","OMP_DYNAMIC","OMP_THREAD_LIMIT"})out[key]=std::getenv(key)?json(std::getenv(key)):json(nullptr);
  out["process_id"]=getpid();out["cpu_model"]=proc_field("/proc/cpuinfo","model name");
  out["allowed_cpus"]=proc_field("/proc/self/status","Cpus_allowed_list");out["allowed_memory_nodes"]=proc_field("/proc/self/status","Mems_allowed_list");
  int policy=0;unsigned long mask[16]{};const long result=syscall(SYS_get_mempolicy,&policy,mask,1024,nullptr,0);
  out["numa_policy"]=result==0?json(policy):json(nullptr);out["numa_policy_errno"]=result==0?0:errno;
  if(result==0){out["numa_policy_nodes"]=json::array();for(unsigned bit=0;bit<1024;++bit)if(mask[bit/(8*sizeof(unsigned long))]&(1ul<<(bit%(8*sizeof(unsigned long)))))out["numa_policy_nodes"].push_back(bit);}
#else
  out["os"]="non-linux";out["process_id"]=nullptr;out["cpu_model"]="unavailable";out["allowed_cpus"]=nullptr;out["allowed_memory_nodes"]=nullptr;out["numa_policy"]=nullptr;
#endif
  return out;
}
struct Energy {bool available=false;uint64_t value=0,range=0;};
Energy energy(){Energy e;std::ifstream value("/sys/class/powercap/intel-rapl:0/energy_uj"),range("/sys/class/powercap/intel-rapl:0/max_energy_range_uj");e.available=bool(value>>e.value)&&bool(range>>e.range)&&e.range && e.value<e.range;return e;}
json energy_delta(Energy a,Energy b){if(!a.available || !b.available || a.range!=b.range)return nullptr;
  // The collector keeps the possible multi-wrap ambiguity explicit.
  return json{{"delta_uj_mod_counter",b.value>=a.value?b.value-a.value:a.range-a.value+b.value},{"range_uj",a.range},
      {"scope","package0_shared_system_not_process"},{"counter_wrap_count_known",false}};}
struct Owner {
  rns8_context* ctx=nullptr;rns8_plan* plan=nullptr;rns8_workspace* work=nullptr;rns8_matrix *a=nullptr,*b=nullptr,*c=nullptr;
  rns8_status close(){rns8_status s;for(auto** v:{&c,&b,&a})if(*v){s=rns8_destroy_matrix(*v);if(s!=RNS8_SUCCESS)return s;*v=nullptr;}
    if(work){s=rns8_destroy_workspace(work);if(s!=RNS8_SUCCESS)return s;work=nullptr;}if(plan){s=rns8_destroy_plan(plan);if(s!=RNS8_SUCCESS)return s;plan=nullptr;}
    if(ctx){s=rns8_destroy_context(ctx);if(s!=RNS8_SUCCESS)return s;ctx=nullptr;}return RNS8_SUCCESS;}
  ~Owner(){(void)close();}
};
}
int main(int argc,char** argv){try{
  if(argc!=3)throw std::invalid_argument("usage: rns8-cpu-amx-calibrate CASE.json unique-run-id");json c;std::ifstream(argv[1])>>c;
  require(c.size()==13 && c.at("schema")=="cpu_amx_case_v1","case ABI");
  const auto m=integer(c,"m",1,512),n=integer(c,"n",1,512),k=integer(c,"k",1,131073),prefix=integer(c,"prefix",0,20),threads=integer(c,"threads",1,256),modulus=integer(c,"modulus",0,256),warmups=integer(c,"warmups",1,100),repeats=integer(c,"repeats",3,100),seed=integer(c,"seed",0,UINT32_MAX);
  require(uint64_t(m)*n*k<=134217728,"complete oracle work bound");const std::string mode=c.at("mode"),residency=c.at("residency"),sem=c.at("semantics");
  const bool amx=mode=="cpu-amx-int8",finite=sem=="finite",wide=sem=="wide-signed" || sem=="wide-unsigned",u=sem=="wide-unsigned" || sem=="bounded-unsigned";
  require(amx || mode=="cpu-reference","mode");require(finite || wide || sem=="bounded-signed" || sem=="bounded-unsigned","semantics");
  require(finite?(modulus>=2 && prefix==0):(modulus==0 && prefix>=1),"modulus/prefix");
  require(residency=="cold" || residency=="resident" || residency=="repack-a" || residency=="repack-b" || residency=="repack-both","residency");
  rns8::detail::configure_cpu_parallel(threads,0,false);rns8::detail::hip_direct_timing_set_enabled(false);
  std::vector<int64_t> a(size_t(m)*(k+3)),b(size_t(k)*(n+5));std::vector<uint64_t> au(a.size()),bu(b.size());std::vector<uint8_t> af(a.size()),bf(b.size());
  for(uint32_t r=0;r<m;++r)for(uint32_t x=0;x<k;++x){const auto pos=size_t(r)*(k+3)+x;const int64_t value=int64_t((uint64_t(r)*71+uint64_t(x)*19+seed)%15)-(u || finite?0:7);a[pos]=value;au[pos]=value;af[pos]=value;}
  for(uint32_t x=0;x<k;++x)for(uint32_t col=0;col<n;++col){const auto pos=size_t(x)*(n+5)+col;const int64_t value=int64_t((uint64_t(x)*37+uint64_t(col)*11+seed)%15)-(u || finite?0:7);b[pos]=value;bu[pos]=value;bf[pos]=value;}
  std::vector<cpp_int> expected(size_t(m)*n);for(uint32_t r=0;r<m;++r)for(uint32_t col=0;col<n;++col)for(uint32_t x=0;x<k;++x)expected[size_t(r)*n+col]+=cpp_int(a[size_t(r)*(k+3)+x])*b[size_t(x)*(n+5)+col];
  if(finite)for(auto& v:expected)v%=modulus;
  std::vector<uint64_t> limbs(size_t(m)*(n+7)*3,91),unsigned_output(size_t(m)*(n+7),91);std::vector<int64_t> signed_output(size_t(m)*(n+7),91);std::vector<uint8_t> finite_output(size_t(m)*(n+7),91);
  uint64_t calls=0,pack_a_calls=0,pack_b_calls=0;json last_workspace=nullptr;std::string selected_kernel;
  auto pack=[&](Owner& o,bool left,bool right){auto one=[&](rns8_matrix* matrix,bool is_a){const auto ld=is_a?k+3:n+5;
      if(finite)check(rns8_pack_finite_u8(o.ctx,matrix,modulus,is_a?af.data():bf.data(),ld,0));
      else if(u)check(rns8_pack_u64(o.ctx,matrix,is_a?au.data():bu.data(),ld,0));else check(rns8_pack_i64(o.ctx,matrix,is_a?a.data():b.data(),ld,0));};
    if(left){one(o.a,true);++pack_a_calls;}if(right){one(o.b,false);++pack_b_calls;}};
  auto create=[&](){auto o=std::make_unique<Owner>();auto options=abi<rns8_context_options>();options.requested_backend=amx?RNS8_BACKEND_CPU_AMX:RNS8_BACKEND_CPU_REFERENCE;check(rns8_create_context(-1,&options,&o->ctx));
    auto d=abi<rns8_gemm_desc>();d.requested_backend=options.requested_backend;d.m=m;d.n=n;d.k=k;d.max_prefix=prefix;d.finite_modulus=modulus;
    d.semantics=finite?RNS8_FINITE_RING_U8:wide?(u?RNS8_EXACT_WIDE_UNSIGNED:RNS8_EXACT_WIDE_SIGNED):(u?RNS8_BOUNDED_U64:RNS8_BOUNDED_I64);
    d.bound_kind=finite || wide?RNS8_BOUND_NONE:u?RNS8_BOUND_GLOBAL_MAX_UNSIGNED:RNS8_BOUND_GLOBAL_MAX_ABS;d.bound=finite || wide?0:uint64_t(k)*(u?196:49);d.flags=finite?0:RNS8_PLAN_FORCE_FIXED_PREFIX;
    check(rns8_create_plan(o->ctx,&d,&o->plan));check(rns8_create_workspace(o->ctx,o->plan,&o->work));
    auto matrix=[&](uint32_t rows,uint32_t cols,rns8_matrix** out){auto md=abi<rns8_matrix_desc>();md.rows=rows;md.cols=cols;md.logical_ld=cols;md.logical_layout=RNS8_LAYOUT_ROW_MAJOR;md.semantics=d.semantics;md.bound_kind=d.bound_kind;md.max_prefix=prefix;check(rns8_create_matrix(o->ctx,&md,out));};
    matrix(m,k,&o->a);matrix(k,n,&o->b);matrix(m,n,&o->c);selected_kernel=o->plan->backend_selected_kernel;return o;};
  const auto setup_begin=Clock::now();std::unique_ptr<Owner> owner;if(residency!="cold"){owner=create();pack(*owner,true,true);}const double setup_us=us(setup_begin,Clock::now());
  auto audit=[&](){for(uint32_t r=0;r<m;++r)for(uint32_t col=0;col<n+7;++col){const auto pos=size_t(r)*(n+7)+col;
      if(col>=n){if(finite)require(finite_output[pos]==91,"finite padding");else if(wide)for(unsigned l=0;l<3;++l)require(limbs[pos*3+l]==91,"wide padding");else require((u?unsigned_output[pos]:uint64_t(signed_output[pos]))==91,"native padding");continue;}
      cpp_int value=finite?cpp_int(finite_output[pos]):wide?cpp_int(limbs[pos*3])+(cpp_int(limbs[pos*3+1])<<64)+(cpp_int(limbs[pos*3+2])<<128):u?cpp_int(unsigned_output[pos]):cpp_int(signed_output[pos]);
      if(wide && !u && (limbs[pos*3+2]>>63))value-=cpp_int(1)<<192;require(value==expected[size_t(r)*n+col],"complete independent Boost product");}};
  auto call=[&](){const auto begin=Clock::now();if(residency=="cold"){owner=create();pack(*owner,true,true);}else if(residency!="resident")pack(*owner,residency!="repack-b",residency!="repack-a");const auto packed=Clock::now();
    if(finite)check(rns8_gemm_finite_u8(owner->ctx,owner->plan,modulus,owner->a,owner->b,owner->c,owner->work));else check(rns8_gemm_rns(owner->ctx,owner->plan,owner->a,owner->b,owner->c,owner->work));const auto computed=Clock::now();
    if(finite)check(rns8_export_finite_u8(owner->ctx,owner->plan,modulus,owner->c,finite_output.data(),n+7));
    else if(wide)check(u?rns8_export_exact_wide_unsigned_limbs(owner->ctx,owner->plan,owner->c,limbs.data(),n+7,3):rns8_export_exact_wide_signed_limbs(owner->ctx,owner->plan,owner->c,limbs.data(),n+7,3));
    else check(u?rns8_export_u64(owner->ctx,owner->plan,owner->c,unsigned_output.data(),n+7):rns8_export_i64(owner->ctx,owner->plan,owner->c,signed_output.data(),n+7));const auto exported=Clock::now();
    // Query and full residue audit are excluded from the operation's phase sum.
    ++calls;return json{{"prepare_us",us(begin,packed)},{"gemm_us",us(packed,computed)},{"crt_export_us",us(computed,exported)},{"total_us",us(begin,exported)}};};
  auto observe=[&](){audit();const uint32_t physical=finite?1:prefix;for(uint32_t p=0;p<physical;++p)for(uint64_t cell=0;cell<uint64_t(m)*n;++cell)
      require(owner->c->residues[uint64_t(p)*m*n+cell]==rns8::detail::centered_residue(expected[cell],finite?modulus:rns8_default_modulus(p)),"every physical residue plane");
    if(amx){auto info=abi<rns8_cpu_amx_workspace_info>();check(rns8_get_cpu_amx_workspace_info(owner->work,&info));last_workspace={{"prepared_threads",info.prepared_threads},{"last_execution_threads",info.last_execution_threads},{"owned_payload_bytes",info.owned_payload_bytes},
      {"packed_b_bytes",info.packed_b_bytes},{"output_staging_bytes",info.output_staging_bytes},{"b_matrix_instance",info.b_matrix_instance},{"b_generation",info.b_generation},{"b_pack_count",info.b_pack_count},{"b_reuse_count",info.b_reuse_count},{"layout_version",info.layout_version},{"safe_k",info.safe_k}};}
  };
  // Cold construction and destruction are both charged. Observe after export,
  // outside compute/export intervals, then separately time cleanup.
  auto finish=[&](json sample,bool verify){if(verify)observe();const auto begin=Clock::now();if(residency=="cold"){check(owner->close());owner.reset();}const double cleanup=us(begin,Clock::now());sample["cleanup_us"]=cleanup;sample["total_us"]=sample["total_us"].get<double>()+cleanup;return sample;};
  json first=finish(call(),true);for(unsigned i=0;i<warmups;++i)finish(call(),true);json samples=json::array();for(unsigned i=0;i<repeats;++i)samples.push_back(finish(call(),true));
  json sustained=json::array();for(unsigned repeat=0;repeat<3;++repeat){const auto before=energy();uint64_t count=0;double operation_us=0;const auto begin=Clock::now();
    do{auto sample=call();sample=finish(std::move(sample),false);operation_us+=sample["total_us"].get<double>();++count;}while(count<4096 && (count<16 || us(begin,Clock::now())<250000));
    const double elapsed=us(begin,Clock::now());const auto after=energy();
    // A separate full final invocation, never inserted into the sustained interval.
    finish(call(),true);sustained.push_back({{"calls",count},{"wall_us",elapsed},{"operation_us",operation_us},{"per_call_us",elapsed/count},{"energy",energy_delta(before,after)}});}
  auto admission=abi<rns8_cpu_amx_info>();check(rns8_query_cpu_amx(0,&admission));const auto env=environment();if(owner)check(owner->close());
  std::cout<<json({{"schema","rns8_cpu_amx_microcase_v1"},{"case",c},{"run_id",argv[2]},{"build_manifest",json::parse(RNS8_BUILD_MANIFEST_JSON)},{"environment",env},{"selected_kernel",selected_kernel},
    {"admission",{{"compiled",admission.compiled},{"cpuid_tile",admission.cpuid_tile},{"cpuid_int8",admission.cpuid_int8},{"osxsave",admission.osxsave},{"palette_supported",admission.palette_supported},{"os_supported",admission.os_supported},{"permission_enabled",admission.permission_enabled},{"xcr0_enabled",admission.xcr0_enabled},{"admitted",admission.admitted},{"permission_errno",admission.permission_errno}}},
    {"last_workspace",last_workspace},{"calls",calls},{"pack_a_calls",pack_a_calls},{"pack_b_calls",pack_b_calls},{"setup_us",setup_us},{"first_call",first},{"samples",samples},{"sustained",sustained},
    {"output_boundary","public_host_pack_dense_gemm_complete_padded_export"},{"gemm_scope","public_range_dispatch_compute_reduction_publication_including_AMX_pack_when_selected"},
    {"sustained_scope","complete_calls_cleanup_and_harness_clock_json_overhead_no_oracle"},{"sample_scope","disjoint_operation_and_cleanup_phase_sum_excludes_between_phase_oracle"},{"oracle_inside_interval",false},{"all_residue_planes_checked",true},{"exact_reference_passed",true},{"promotion_eligible",false}}).dump(2)<<'\n';return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
