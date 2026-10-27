#include "core/api_internal.hpp"
#include "core/hip_resources.hpp"
#include "backend_common/sha256.hpp"
#include <nlohmann/json.hpp>
#include <chrono>
#include <fstream>
#include <iostream>
namespace {
using namespace rns8::detail;using json=nlohmann::json;using Clock=std::chrono::steady_clock;
void check(rns8_status s){if(s==RNS8_UNSUPPORTED_BACKEND || s==RNS8_UNSUPPORTED_ARCH)throw std::domain_error(rns8_status_string(s));if(s!=RNS8_SUCCESS)throw std::runtime_error(rns8_status_string(s));}
uint32_t number(const json& c,const char* key,uint32_t low,uint32_t high){if(!c.at(key).is_number_integer())throw std::invalid_argument(key);auto v=c.at(key).get<int64_t>();if(v<low || v>high)throw std::invalid_argument(key);return uint32_t(v);}
double us(Clock::time_point a,Clock::time_point b){return std::chrono::duration<double,std::micro>(b-a).count();}
template<class T> std::string digest(const std::vector<T>& values){return sha256(std::string(reinterpret_cast<const char*>(values.data()),values.size()*sizeof(T)));}
struct Owner {
  hip_direct_device_buffer a,b,c,status;
  // Device buffers retain failed-drain allocations. This is an isolated capture
  // process; failure never releases memory that may still be consumed by DMA.
  bool pending=false;
  ~Owner(){if(pending && hip_direct_synchronize(0)==RNS8_SUCCESS)complete();}
  void mark(){a.mark_pending();b.mark_pending();c.mark_pending();status.mark_pending();pending=true;}
  void complete(){a.mark_completed();b.mark_completed();c.mark_completed();status.mark_completed();pending=false;}
};
}
int main(int argc,char** argv){try {
  if(argc!=3)throw std::invalid_argument("usage: rns8-tdm-calibrate CASE.json unique-run-id");
  json c;std::ifstream(argv[1])>>c;
  if(c.size()!=13 || c.at("schema")!="native_tdm_case_v1" || !c.at("signed_input").is_boolean())throw std::invalid_argument("case ABI");
  const std::string run=argv[2],mode=c.at("mode");
  if(run.empty() || run.size()>128 || run.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")!=std::string::npos)throw std::invalid_argument("run id");
  NativeTdmProblem p;p.m=number(c,"m",1,512);p.n=number(c,"n",1,512);p.k=number(c,"k",1,131073);
  p.prefix=number(c,"prefix",1,20);p.group=number(c,"group",1,4);p.signed_input=c.at("signed_input");
  p.compute=c.at("compute")=="wmma"?1:0;p.cluster_axis=c.at("cluster_axis")=="output-neighbor"?1:0;
  if((c.at("compute")!="scalar" && c.at("compute")!="wmma") || (c.at("cluster_axis")!="moduli" && c.at("cluster_axis")!="output-neighbor"))throw std::invalid_argument("compute/cluster axis");
  p.mode=mode=="ordinary"?0:mode=="tdm"?1:2;p.cluster=mode=="multicast2"?2:mode=="multicast4"?4:1;
  if(mode!="ordinary" && mode!="tdm" && mode!="multicast2" && mode!="multicast4")throw std::invalid_argument("mode");
  p.lda=p.k+2;p.ldb=p.n+2;p.ldc=p.n+2;
  const uint32_t repeats=number(c,"repeats",3,100),warmups=number(c,"warmups",1,100),seed=number(c,"seed",0,UINT32_MAX);
  const auto resources=query_native_tdm(p);check(resources.status);
  if(p.m*p.n*p.k>134217728)throw std::invalid_argument("complete oracle work budget");
  const uint64_t envelope=resources.native_a_bytes*2+resources.native_b_bytes*2+resources.output_bytes*3+32;
  if(envelope>UINT64_C(512)*1024*1024)throw std::invalid_argument("logical payload budget");
  const auto oracle_start=Clock::now();
  std::vector<uint64_t> a((resources.native_a_bytes+7)/8,UINT64_C(0xa5a5a5a5a5a5a5a5)),b((resources.native_b_bytes+7)/8,UINT64_C(0xa5a5a5a5a5a5a5a5));
  for(uint64_t row=0;row<p.m;++row)for(uint64_t q=0;q<p.k;++q)a[row*p.lda+q]=(q%3==0?UINT64_MAX:q%3==1?UINT64_C(1)<<63:1)^uint64_t(seed+row*7+q*131);
  for(uint64_t q=0;q<p.k;++q)for(uint64_t col=0;col<p.n;++col)b[q*p.ldb+col]=(q%3==0?UINT64_C(1)<<63:q%3==1?UINT64_MAX:1)^uint64_t(seed+q*17+col*31);
  std::vector<int8_t> expected(resources.output_bytes,int8_t(0x5a)),output=expected;
  const auto value=[&](uint64_t x){boost::multiprecision::cpp_int v=x;if(p.signed_input && (x>>63))v-=boost::multiprecision::cpp_int(1)<<64;return v;};
  for(uint64_t row=0;row<p.m;++row)for(uint64_t col=0;col<p.n;++col) {
    boost::multiprecision::cpp_int sum=0;for(uint64_t q=0;q<p.k;++q)sum+=value(a[row*p.lda+q])*value(b[q*p.ldb+col]);
    for(uint32_t plane=0;plane<p.prefix;++plane){const int prime=rns8_default_modulus(plane);int residue=(sum%prime).convert_to<int>();if(residue<0)residue+=prime;if(residue>prime/2)residue-=prime;expected[(uint64_t(plane)*p.m+row)*p.ldc+col]=int8_t(residue);}
  }
  const double oracle=us(oracle_start,Clock::now());
  const uint64_t rows=(p.m+15)/16,cols=(p.n+15)/16,groups=(p.prefix+p.group-1)/p.group;
  const uint64_t no_reuse_a=p.m*p.k*8*cols*groups,no_reuse_b=p.k*p.n*8*rows*groups;
  const uint64_t ideal_a=p.mode!=2?no_reuse_a:p.cluster_axis?p.m*p.k*8*((cols+p.cluster-1)/p.cluster)*groups:p.m*p.k*8*cols*((groups+p.cluster-1)/p.cluster);
  const uint64_t ideal_b=p.mode!=2 || p.cluster_axis?no_reuse_b:p.k*p.n*8*rows*((groups+p.cluster-1)/p.cluster);
  rns8_context_options options{};options.struct_size=sizeof(options);options.abi_version=RNS8_ABI_VERSION;options.requested_backend=RNS8_BACKEND_HIP_DIRECT;
  rns8_context* context=nullptr;check(rns8_create_context(0,&options,&context));rns8_device_info device{};device.struct_size=sizeof(device);device.abi_version=RNS8_ABI_VERSION;
  check(rns8_get_device_info(context,&device));check(rns8_destroy_context(context));
  const auto setup_start=Clock::now();Owner owner;check(owner.a.allocate(0,resources.native_a_bytes));check(owner.b.allocate(0,resources.native_b_bytes));
  check(owner.c.allocate(0,resources.output_bytes));check(owner.status.allocate(0,4));
  check(hip_direct_copy_host_to_device(0,owner.a.get(),a.data(),resources.native_a_bytes));check(hip_direct_copy_host_to_device(0,owner.b.get(),b.data(),resources.native_b_bytes));
  check(hip_direct_copy_host_to_device(0,owner.c.get(),output.data(),resources.output_bytes));
  uint32_t flags=0;owner.mark();const auto prepared=hip_direct_prepare_native_tdm(0,p,owner.status.get(),flags);
  check(hip_direct_synchronize(0));owner.complete();check(prepared);const double setup=us(setup_start,Clock::now());
  hip_direct_timing_set_enabled(false);
  auto execute=[&](){owner.mark();const auto result=hip_direct_native_tdm_admitted(0,owner.a.get(),owner.b.get(),owner.c.get(),owner.status.get(),p,flags);
    if(result==RNS8_SUCCESS)owner.complete();else {check(hip_direct_synchronize(0));owner.complete();check(result);}};
  auto call=[&](){const auto start=Clock::now();execute();return us(start,Clock::now());};
  auto verify=[&](){check(hip_direct_copy_device_to_host(0,output.data(),owner.c.get(),resources.output_bytes));if(output!=expected)throw std::runtime_error("complete Boost residue/padding mismatch");};
  const double first=call();verify();for(uint32_t i=0;i<warmups;++i)execute();json samples=json::array();
  for(uint32_t i=0;i<repeats;++i){samples.push_back(call());verify();}
  json sustained=json::array();for(unsigned burst=0;burst<3;++burst){uint32_t calls=0;const auto start=Clock::now();do{execute();++calls;}while(calls<4096 && (calls<16 || us(start,Clock::now())<250000));
    const double total=us(start,Clock::now());verify();sustained.push_back({{"calls",calls},{"total_us",total},{"per_call_us",total/calls},{"minimum_duration_reached",total>=250000}});}
  json report={{"schema","rns8_native_tdm_microcase_v1"},{"case",c},{"run_id",run},{"target",device.gcn_arch},{"device",device.name},{"runtime_version",device.hip_runtime_version},
    {"build_manifest",json::parse(RNS8_BUILD_MANIFEST_JSON)},{"backend_actual","hip-direct"},{"mode_actual",mode},{"cluster_actual",p.cluster},{"capability_flags",flags},
    {"source_representation","native64_signed"},{"signed_input",p.signed_input},{"compute_primitive",p.compute?"wmma_i32_16x16x64_iu8_wave32":"integer_alu_residue_dot64"},{"cluster_axis_actual",c.at("cluster_axis")},{"lds_bytes",resources.lds_bytes},{"stage_count",2},
    {"grid_blocks",resources.grid_blocks},{"status_bytes",4},{"logical_payload_envelope_bytes",envelope},{"payload_budget_bytes",UINT64_C(512)*1024*1024},
    {"modeled_no_intergroup_reuse_read_bytes",no_reuse_a+no_reuse_b},{"modeled_ideal_multicast_source_read_bytes",ideal_a+ideal_b},
    {"modeled_ideal_source_read_amplification",double(ideal_a+ideal_b)/double((p.m*p.k+p.k*p.n)*8)},
    {"actual_hbm_read_bytes",nullptr},{"tdm_stall_cycles",nullptr},{"barrier_stall_cycles",nullptr},{"achieved_occupancy",nullptr},{"compiler_private_scratch_bytes",nullptr},
    {"a_checksum",digest(a)},{"b_checksum",digest(b)},{"output_checksum",digest(output)},{"dataset","fullwidth_native_xor_affine_v1"},
    {"output_boundary","complete_resident_native64_to_centered_rns_checked_call"},{"semantic_contract","residues_only_no_native_range_claim"},
    {"timer","std_chrono_steady_clock_microseconds"},{"production_timing_hooks_enabled",false},{"oracle_inside_production_interval",false},
    {"exact_reference_passed",true},{"target_qualified",false},{"performance_qualified",false},{"promotion_eligible",false},
    {"oracle_us",oracle},{"setup_us",setup},{"first_call_us",first},{"samples",samples},{"sustained",sustained}};
  report["source_representation"]=p.signed_input?"native64_signed":"native64_unsigned";
  check(owner.status.reset());check(owner.c.reset());check(owner.b.reset());check(owner.a.reset());
  std::cout<<report.dump(2)<<'\n';return 0;
}catch(const std::domain_error& error){std::cerr<<error.what()<<'\n';return 77;}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}
