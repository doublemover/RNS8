// Public finite-ring verifier wall-cost experiment; no private runtime headers.
#include <rns8/rns8.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using json=nlohmann::json;
using Clock=std::chrono::steady_clock;
constexpr uint64_t budget=256ull*1024*1024;
constexpr const char* recipe="splitmix64_finite_u8_seed3135_extrema_v1";
bool cleanup_failed=false;
void require(bool good,const char* why) {if(!good) throw std::runtime_error(why);}
void check(rns8_status status) {if(status!=RNS8_SUCCESS) throw std::runtime_error(rns8_status_string(status));}
struct Unsupported:std::runtime_error {using std::runtime_error::runtime_error;};
template<class T,rns8_status(*Destroy)(T*)> struct Deleter {
  void operator()(T* p) const noexcept {if(p && Destroy(p)!=RNS8_SUCCESS) cleanup_failed=true;}
};
template<class T,rns8_status(*Destroy)(T*)> using Owner=std::unique_ptr<T,Deleter<T,Destroy>>;
using Context=Owner<rns8_context,rns8_destroy_context>;
using Plan=Owner<rns8_plan,rns8_destroy_plan>;
using Workspace=Owner<rns8_workspace,rns8_destroy_workspace>;
using Matrix=Owner<rns8_matrix,rns8_destroy_matrix>;
uint64_t number(const char* s,uint64_t low,uint64_t high) {
  uint64_t v=0;const auto end=s+std::strlen(s);const auto result=std::from_chars(s,end,v);
  if(result.ec!=std::errc{} || result.ptr!=end || v<low || v>high) throw std::invalid_argument("numeric argument outside supported domain");
  return v;
}
struct Case {std::string backend,mode,run;int device;size_t m,n,k;uint16_t q;uint32_t trials,repeats,warmups;uint64_t seed;};
void help() {
  std::cout<<"rns8-finite-verify-calibrate BACKEND DEVICE MODE M N K MODULUS TRIALS REPEATS WARMUPS CHALLENGE_SEED RUN_ID\n"
    "BACKEND cpu-reference (DEVICE -1) or hip-direct (DEVICE 0..63); no fallback.\n"
    "MODE verify_only | compute_verify | full_reference. Full reference requires TRIALS=0 and CHALLENGE_SEED=0.\n"
    "M,N 1..512; K 1..65537; M*N*K <= 134217728; MODULUS 2..256.\n"
    "Verifier TRIALS 1..256; REPEATS 1..1000; WARMUPS 1..1000; seed uint64; RUN_ID 1..128 ASCII letters/digits/_/-.\n"
    "Public synchronous API wall time, including allocation/challenge/transfer/sync/report; reference includes A/B/C exports.\n"
    "Fixed dataset seed3135, finite-ring congruence only. SplitMix64 replay is not fresh entropy or an integer equality certificate.\n";
}
Case parse(int argc,char** argv) {
  if(argc!=13) throw std::invalid_argument("expected twelve positional arguments; use --help");
  Case c{};c.backend=argv[1];c.mode=argv[3];c.run=argv[12];
  require(c.backend=="cpu-reference" || c.backend=="hip-direct","explicit backend must be cpu-reference or hip-direct");
  if(c.backend=="cpu-reference") {require(std::string(argv[2])=="-1","CPU device must be -1");c.device=-1;}
  else c.device=int(number(argv[2],0,63));
  require(c.mode=="verify_only" || c.mode=="compute_verify" || c.mode=="full_reference","invalid mode");
  c.m=size_t(number(argv[4],1,512));c.n=size_t(number(argv[5],1,512));c.k=size_t(number(argv[6],1,65537));
  require(uint64_t(c.m)*c.n*c.k<=134217728,"reference work exceeds bounded domain");
  c.q=uint16_t(number(argv[7],2,256));c.trials=uint32_t(number(argv[8],0,256));
  c.repeats=uint32_t(number(argv[9],1,1000));c.warmups=uint32_t(number(argv[10],1,1000));
  c.seed=number(argv[11],0,UINT64_MAX);
  require(c.mode=="full_reference" ? c.trials==0 && c.seed==0 : c.trials>0,"irrelevant or missing trial/seed controls");
  require(!c.run.empty() && c.run.size()<=128 && std::all_of(c.run.begin(),c.run.end(),[](char v){return (v>='a'&&v<='z')||(v>='A'&&v<='Z')||(v>='0'&&v<='9')||v=='_'||v=='-';}),"invalid run ID");
  return c;
}
uint64_t next(uint64_t& state) {uint64_t z=(state+=0x9e3779b97f4a7c15ull);z=(z^(z>>30))*0xbf58476d1ce4e5b9ull;z=(z^(z>>27))*0x94d049bb133111ebull;return z^(z>>31);}
std::string checksum(const std::vector<uint8_t>& bytes) {
  uint64_t hash=14695981039346656037ull;for(auto b:bytes) hash=(hash^b)*1099511628211ull;
  std::ostringstream s;s<<std::hex<<std::setfill('0')<<std::setw(16)<<hash;return s.str();
}
uint64_t ns(Clock::time_point begin) {return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-begin).count());}
std::vector<uint8_t> scalar(const Case& c,const std::vector<uint8_t>& a,const std::vector<uint8_t>& b) {
  std::vector<uint8_t> out(c.m*c.n);
  // K<=65537, each product<=65025: even the complete dot product fits uint64.
  for(size_t r=0;r<c.m;++r) for(size_t col=0;col<c.n;++col) {uint64_t sum=0;for(size_t t=0;t<c.k;++t) sum+=uint64_t(a[r*c.k+t])*b[t*c.n+col];out[r*c.n+col]=uint8_t(sum%c.q);}
  return out;
}
std::vector<uint8_t> blocked(const Case& c,const std::vector<uint8_t>& a,const std::vector<uint8_t>& b) {
  std::vector<uint64_t> sums(c.m*c.n,0);
  for(size_t rr=0;rr<c.m;rr+=32) for(size_t tt=0;tt<c.k;tt+=64) for(size_t cc=0;cc<c.n;cc+=32)
    for(size_t r=rr;r<std::min(rr+32,c.m);++r) for(size_t t=tt;t<std::min(tt+64,c.k);++t)
      for(size_t col=cc;col<std::min(cc+32,c.n);++col) sums[r*c.n+col]+=uint64_t(a[r*c.k+t])*b[t*c.n+col];
  std::vector<uint8_t> out(sums.size());for(size_t i=0;i<out.size();++i) out[i]=uint8_t(sums[i]%c.q);return out;
}
Plan plan(rns8_context* ctx,rns8_backend_kind backend,size_t m,size_t n,size_t k,uint16_t q) {
  rns8_gemm_desc d{};d.struct_size=sizeof(d);d.abi_version=RNS8_ABI_VERSION;d.semantics=RNS8_FINITE_RING_U8;
  d.bound_kind=RNS8_BOUND_NONE;d.requested_backend=backend;d.m=int64_t(m);d.n=int64_t(n);d.k=int64_t(k);d.finite_modulus=q;
  rns8_plan* p=nullptr;check(rns8_create_plan(ctx,&d,&p));return Plan(p);
}
Matrix matrix(rns8_context* ctx,size_t m,size_t n) {
  rns8_matrix_desc d{};d.struct_size=sizeof(d);d.abi_version=RNS8_ABI_VERSION;d.rows=int64_t(m);d.cols=int64_t(n);d.logical_ld=int64_t(n);
  d.logical_layout=RNS8_LAYOUT_ROW_MAJOR;d.semantics=RNS8_FINITE_RING_U8;d.bound_kind=RNS8_BOUND_NONE;
  rns8_matrix* p=nullptr;check(rns8_create_matrix(ctx,&d,&p));return Matrix(p);
}
rns8_matrix_storage_info storage(const rns8_matrix* p) {rns8_matrix_storage_info s{};s.struct_size=sizeof(s);s.abi_version=RNS8_ABI_VERSION;check(rns8_get_matrix_storage_info(p,&s));return s;}
uint64_t payload(const rns8_matrix_storage_info& s) {return s.host_residue_bytes+s.device_residue_bytes+s.host_byte_limb_bytes+s.device_byte_limb_bytes+s.host_native_bytes+s.device_native_bytes;}
json run(const Case& c) {
  const uint64_t cells=uint64_t(c.m)*c.k+uint64_t(c.k)*c.n+uint64_t(c.m)*c.n;
  const uint64_t scratch=c.backend=="cpu-reference"?c.n+4*c.k:2*c.n+4*c.k+8;
  // Covers host recipe+gold, resident host/device copies, reference exports and
  // uint64 blocked sums/result, padded audit output, verifier scratch. No cpp_int.
  const uint64_t conservative=5*cells+16*uint64_t(c.m)*c.n+3*c.m+scratch;
  require(conservative<=budget,"logical live payload envelope exceeds 256MiB");
  const auto oracle_begin=Clock::now();uint64_t state=3135;
  std::vector<uint8_t> a(c.m*c.k),b(c.k*c.n);
  for(auto* v:{&a,&b}) {for(auto& x:*v) x=uint8_t(next(state)%c.q);(*v)[0]=uint8_t(c.q-1);if(v->size()>1)(*v)[1]=0;}
  const auto expected=scalar(c,a,b);const auto oracle_ns=ns(oracle_begin);
  const auto begin=Clock::now();
  const auto backend=c.backend=="cpu-reference"?RNS8_BACKEND_CPU_REFERENCE:RNS8_BACKEND_HIP_DIRECT;
  rns8_context_options options{};options.struct_size=sizeof(options);options.abi_version=RNS8_ABI_VERSION;options.requested_backend=backend;
  rns8_context* raw=nullptr;const auto status=rns8_create_context(c.device,&options,&raw);
  if(status==RNS8_UNSUPPORTED_BACKEND || status==RNS8_UNSUPPORTED_ARCH || status==RNS8_UNSUPPORTED_OS) throw Unsupported(rns8_status_string(status));
  check(status);Context ctx(raw);
  rns8_device_info device{};device.struct_size=sizeof(device);device.abi_version=RNS8_ABI_VERSION;check(rns8_get_device_info(ctx.get(),&device));
  require(device.backend==backend && device.device_id==c.device,"requested backend/device not retained");
  require(backend!=RNS8_BACKEND_HIP_DIRECT || device.gcn_arch[0]!=0,"HIP target identity unavailable");
  auto p=plan(ctx.get(),backend,c.m,c.n,c.k,c.q);
  auto pa=plan(ctx.get(),backend,c.m,c.k,1,c.q),pb=plan(ctx.get(),backend,c.k,c.n,1,c.q);
  rns8_plan_backend_info pi{};pi.struct_size=sizeof(pi);pi.abi_version=RNS8_ABI_VERSION;check(rns8_get_plan_backend_info(p.get(),&pi));
  require(pi.backend==backend,"plan backend differs from request");
  require(pi.workspace_required_bytes<=budget-conservative,"workspace exceeds logical live payload envelope");
  rns8_workspace* wr=nullptr;check(rns8_create_workspace(ctx.get(),p.get(),&wr));Workspace work(wr);
  auto A=matrix(ctx.get(),c.m,c.k),B=matrix(ctx.get(),c.k,c.n),C=matrix(ctx.get(),c.m,c.n);
  check(rns8_pack_finite_u8(ctx.get(),A.get(),c.q,a.data(),int64_t(c.k),1));
  check(rns8_pack_finite_u8(ctx.get(),B.get(),c.q,b.data(),int64_t(c.n),1));
  // Every mode starts at the same correct resident C boundary. Compute+verify
  // overwrites it on each invocation; setup packing is reported separately.
  check(rns8_pack_finite_u8(ctx.get(),C.get(),c.q,expected.data(),int64_t(c.n),1));
  const auto sa=storage(A.get()),sb=storage(B.get()),sc=storage(C.get());
  const auto resident=payload(sa)+payload(sb)+payload(sc);
  require(resident<=2*cells,"unexpected resident allocation exceeds envelope");
  const auto setup_ns=ns(begin);
  rns8_finite_verify_options vo{};vo.struct_size=sizeof(vo);vo.abi_version=RNS8_ABI_VERSION;vo.repetition_count=c.trials;
  vo.random_seed=c.seed;vo.max_workspace_bytes=scratch;vo.sampler=RNS8_VERIFY_SPLITMIX64_BINARY_V1;
  rns8_finite_verify_report report{};
  auto invoke=[&]() {
    if(c.mode=="full_reference") {
      std::vector<uint8_t> ea(a.size()),eb(b.size()),ec(expected.size());
      check(rns8_export_finite_u8(ctx.get(),pa.get(),c.q,A.get(),ea.data(),int64_t(c.k)));
      check(rns8_export_finite_u8(ctx.get(),pb.get(),c.q,B.get(),eb.data(),int64_t(c.n)));
      check(rns8_export_finite_u8(ctx.get(),p.get(),c.q,C.get(),ec.data(),int64_t(c.n)));
      require(blocked(c,ea,eb)==ec,"full modular reference mismatch");
    } else {
      if(c.mode=="compute_verify") check(rns8_gemm_finite_u8(ctx.get(),p.get(),c.q,A.get(),B.get(),C.get(),work.get()));
      report={};report.struct_size=sizeof(report);report.abi_version=RNS8_ABI_VERSION;
      check(rns8_verify_finite_product(ctx.get(),A.get(),B.get(),C.get(),&vo,&report));
    }
  };
  auto audit=[&]() {
    std::vector<uint8_t> observed(c.m*(c.n+3),0xA5);
    check(rns8_export_finite_u8(ctx.get(),p.get(),c.q,C.get(),observed.data(),int64_t(c.n+3)));
    for(size_t r=0;r<c.m;++r) for(size_t col=0;col<c.n+3;++col)
      require(observed[r*(c.n+3)+col]==(col<c.n?expected[r*c.n+col]:uint8_t(0xA5)),"independent scalar reference or export padding mismatch");
    if(c.mode!="full_reference") {
      require(report.passed==1 && report.semantics==RNS8_FINITE_RING_U8 && report.modulus==c.q && report.sampler==1 && report.completed_trials==c.trials && report.random_seed==c.seed,"verifier report contract mismatch");
      require(report.workspace_bytes==scratch && report.ideal_failure_bound_exponent==c.trials && report.ideal_failure_probability==std::ldexp(1.0,-int(c.trials)),"verifier probability/resource report mismatch");
      const uint64_t ids[]={sa.matrix_instance_id,sb.matrix_instance_id,sc.matrix_instance_id};
      for(size_t i=0;i<3;++i) require(report.matrix_instance_ids[i]==ids[i] && report.generations[i]>0,"verifier report lifetime binding mismatch");
    }
    const auto after=storage(C.get());
    require(after.matrix_instance_id==sc.matrix_instance_id && (backend==RNS8_BACKEND_CPU_REFERENCE?after.host_residues_current:after.device_residues_current),"output residency/identity changed");
  };
  for(uint32_t i=0;i<c.warmups;++i) {invoke();audit();}
  json samples=json::array();std::vector<uint64_t> times;
  for(uint32_t i=0;i<c.repeats;++i) {const auto t=Clock::now();invoke();const auto elapsed=ns(t);require(elapsed>0,"call below host timer resolution");audit();times.push_back(elapsed);samples.push_back({{"wall_ns",elapsed},{"exact_reference_passed",true}});}
  const auto sustained_begin=Clock::now();uint64_t calls=0,elapsed=0;
  do {invoke();++calls;elapsed=ns(sustained_begin);} while(calls<4096 && (calls<16 || elapsed<250000000));
  audit();
  { // Read-only input preservation audit is outside every measured interval.
    std::vector<uint8_t> ea(a.size()),eb(b.size());
    check(rns8_export_finite_u8(ctx.get(),pa.get(),c.q,A.get(),ea.data(),int64_t(c.k)));
    check(rns8_export_finite_u8(ctx.get(),pb.get(),c.q,B.get(),eb.data(),int64_t(c.n)));
    require(ea==a && eb==b,"input changed during read-only verification or GEMM");
  }
  std::sort(times.begin(),times.end());
  const auto median=times.size()%2?double(times[times.size()/2]):(double(times[times.size()/2-1])+double(times[times.size()/2]))/2;
  json verification=nullptr;
  if(c.mode!="full_reference") verification={{"passed",true},{"sampler",1},{"sampler_name","splitmix64_binary_v1"},{"replay_seed",c.seed},{"trials",c.trials},
    {"matrix_instance_ids",std::vector<uint64_t>(report.matrix_instance_ids,report.matrix_instance_ids+3)},
    {"generations",std::vector<uint64_t>(report.generations,report.generations+3)},{"workspace_bytes",report.workspace_bytes},
    {"ideal_failure_bound_exponent",report.ideal_failure_bound_exponent},{"ideal_failure_probability",report.ideal_failure_probability},
    {"probability_model","independent_uniform_binary_bits_only"},{"fresh_entropy",false},{"integer_equality_certified",false}};
  return {{"schema","rns8_finite_verify_microcase_v1"},{"promotion_eligible",false},{"run_id",c.run},{"mode",c.mode},
    {"backend_requested",c.backend},{"backend_actual",c.backend},{"device_requested",c.device},{"device_actual",device.device_id},
    {"device_name",device.name},{"target",backend==RNS8_BACKEND_CPU_REFERENCE?"cpu":device.gcn_arch},{"hip_runtime_version",device.hip_runtime_version},{"hip_driver_version",device.hip_driver_version},
    {"build_manifest",json::parse(RNS8_BUILD_MANIFEST_JSON)},{"m",c.m},{"n",c.n},{"k",c.k},{"modulus",c.q},{"semantics","finite_ring_u8"},
    {"trials",c.trials},{"challenge_seed",c.seed},{"dataset_recipe",recipe},{"dataset_seed",3135},{"a_checksum",checksum(a)},{"b_checksum",checksum(b)},{"output_checksum",checksum(expected)},
    {"repeats",c.repeats},{"warmups",c.warmups},{"reference_method","scalar_uint64_full_dot_then_modulus"},{"comparator_method","blocked_uint64_32x32x64_full_dot_then_modulus"},
    {"timer","std_chrono_steady_clock_nanoseconds"},{"clock_period_numerator",Clock::period::num},{"clock_period_denominator",Clock::period::den},
    {"timed_scope",c.mode=="full_reference"?"allocate_export_A_B_C_blocked_reference_full_compare":c.mode=="compute_verify"?"public_gemm_then_public_verify_all_wall_cost":"public_verify_all_wall_cost"},
    {"setup_ns",setup_ns},{"oracle_and_dataset_ns",oracle_ns},{"setup_scope","context_plans_workspace_matrices_pack_A_B_gold_C"},
    {"memory_budget_bytes",budget},{"logical_live_payload_envelope_bytes",conservative+pi.workspace_required_bytes},{"resident_payload_bytes",resident},{"plan_workspace_bytes",pi.workspace_required_bytes},
    {"memory_exclusions","allocator bookkeeping, context/compiler runtime and fixed metadata; this is not process RSS"},
    {"selected_compute_kernel",pi.selected_kernel},{"verification",verification},{"samples",samples},{"median_wall_ns",median},{"calls_per_second",1e9/median},
    {"exact_full_output_and_padding_checked",true},{"inputs_unchanged_checked",true},
    {"sustained",{{"calls",calls},{"elapsed_ns",elapsed},{"mean_wall_ns",double(elapsed)/double(calls)},{"minimum_calls",16},{"maximum_calls",4096},{"target_ns",250000000},{"stop_reason",elapsed>=250000000?"duration":"call_cap"},{"independent_audit_inside_interval",false},{"final_exact_reference_passed",true}}}};
}
}
int main(int argc,char** argv) {
  if(argc==2 && std::string(argv[1])=="--help") {help();return 0;}
  try {const auto c=parse(argc,argv);const auto result=run(c);require(!cleanup_failed,"public handle cleanup failed");std::cout<<result.dump()<<'\n';return 0;}
  catch(const Unsupported& error) {std::cerr<<"unsupported explicit backend/device: "<<error.what()<<'\n';return 77;}
  catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
