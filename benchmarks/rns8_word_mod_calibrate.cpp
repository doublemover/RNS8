#include "backend_hip_direct/hip_backend.hpp"
#include "backend_common/word_mod_plan.hpp"
#include "core/hip_resources.hpp"
#include "core/internal.hpp"
#include <charconv>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
namespace {
using namespace rns8::detail;
using Clock=std::chrono::steady_clock;
struct Unsupported:std::runtime_error {using std::runtime_error::runtime_error;};
void check(rns8_status s) {if(s==RNS8_UNSUPPORTED_BACKEND) throw Unsupported(rns8_status_string(s));if(s!=RNS8_SUCCESS) throw std::runtime_error(rns8_status_string(s));}
uint64_t number(const char* text,uint64_t low,uint64_t high) {
  uint64_t value=0;const auto end=text+std::strlen(text);const auto parsed=std::from_chars(text,end,value);
  if(parsed.ec!=std::errc{} || parsed.ptr!=end || value<low || value>high) throw std::invalid_argument("numeric argument outside admitted domain");return value;
}
uint64_t random_word(uint64_t& state) {uint64_t z=(state+=UINT64_C(0x9e3779b97f4a7c15));z=(z^(z>>30))*UINT64_C(0xbf58476d1ce4e5b9);z=(z^(z>>27))*UINT64_C(0x94d049bb133111eb);return z^(z>>31);}
std::string checksum(const std::vector<uint64_t>& values) {
  uint64_t hash=UINT64_C(14695981039346656037);
  for(auto value:values) for(unsigned byte=0;byte<8;++byte) hash=(hash^uint8_t(value>>(8*byte)))*UINT64_C(1099511628211);
  std::ostringstream out;out<<std::hex<<std::setfill('0')<<std::setw(16)<<hash;return out.str();
}
unsigned bits(uint64_t value) {unsigned n=0;while(value){++n;value>>=1;}return n;}
struct Buffers {
  hip_direct_device_buffer a,b,c;
  void allocate(int device,size_t ac,size_t bc,size_t cc) {check(a.allocate(device,ac*8));check(b.allocate(device,bc*8));check(c.allocate(device,cc*8));}
  void release() {check(c.reset());check(b.reset());check(a.reset());}
};
}
int main(int argc,char** argv) {
 try {
  const char* usage="usage: rns8-word-mod-calibrate DEVICE baseline|packed_dot4 resident|cold M N K ring|prime_field MODULUS REPEATS WARMUPS RUN_ID\nDEVICE0..63; M,N1..512 K1..65537 M*N*K<=134217728; modulus2..UINT64_MAX; repeats1..10000 warmups1..1000; no fallback. Private canonical uint64 ring/prime-field route; no asymmetric floating/WMMA schedule.";
  if(argc==2 && std::string(argv[1])=="--help") {std::cout<<usage<<'\n';return 0;}
  if(argc!=12) throw std::invalid_argument(usage);
  const int device=int(number(argv[1],0,63));const std::string method=argv[2],scope=argv[3],domain=argv[7],run_id=argv[11];
  const uint64_t m=number(argv[4],1,512),n=number(argv[5],1,512),k=number(argv[6],1,65537),modulus=number(argv[8],2,UINT64_MAX);
  const uint32_t repeats=uint32_t(number(argv[9],1,10000)),warmups=uint32_t(number(argv[10],1,1000));
  if((method!="baseline" && method!="packed_dot4") || (scope!="resident" && scope!="cold") || (domain!="ring" && domain!="prime_field") ||
     m*n*k>134217728 || run_id.empty() || run_id.size()>128 || run_id.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")!=std::string::npos)
    throw std::invalid_argument("invalid method/scope/domain/reference work/run identity");
  word_mod::Problem p;p.domain=domain=="ring"?word_mod::Domain::ring:word_mod::Domain::prime_field;
  p.modulus=modulus;p.m=m;p.n=n;p.k=k;p.lda=k+2;p.ldb=n+3;p.ldc=n+3;
  const auto support=word_mod::query(p,true);if(support.status!=RNS8_SUCCESS) throw std::invalid_argument("word modular shape/domain is not admitted");
  const uint64_t ac=m*p.lda,bc=k*p.ldb,cc=m*p.ldc+8,envelope=(3*(ac+bc)+4*cc)*8+16384;
  constexpr uint64_t budget=UINT64_C(512)*1024*1024;if(envelope>budget) throw std::invalid_argument("aggregate logical payload exceeds512MiB");
  const auto oracle_start=Clock::now();uint64_t state=6651;
  std::vector<uint64_t> a(ac,UINT64_MAX),b(bc,UINT64_MAX),gold(cc,UINT64_MAX),observed(cc,UINT64_MAX),poison(cc,UINT64_MAX),audit_a(ac),audit_b(bc);
  const auto fill=[&](std::vector<uint64_t>& data,uint64_t rows,uint64_t cols,uint64_t ld) {
    uint64_t index=0;for(uint64_t row=0;row<rows;++row) for(uint64_t col=0;col<cols;++col,++index) {
      const uint64_t raw=random_word(state),edge[]={0,1,modulus-1,modulus/2};
      data[row*ld+col]=index%257<4?edge[index%257]:raw%modulus;
    }
  };fill(a,m,k,p.lda);fill(b,k,n,p.ldb);
  using boost::multiprecision::cpp_int;using boost::multiprecision::int256_t;
  for(uint64_t row=0;row<m;++row) for(uint64_t col=0;col<n;++col) {
    // K<=65537 and native products<2^128 prove the exact sum<2^145.
    int256_t sum=0;for(uint64_t q=0;q<k;++q) sum+=int256_t(a[row*p.lda+q])*b[q*p.ldb+col];
    if(col==0 || col+1==n) {cpp_int audit=0;for(uint64_t q=0;q<k;++q) audit+=cpp_int(a[row*p.lda+q])*b[q*p.ldb+col];
      if(audit!=cpp_int(sum)) throw std::runtime_error("fixed exact reference failed unbounded edge audit");}
    gold[row*p.ldc+col]=(sum%modulus).convert_to<uint64_t>();
  }
  const double oracle_ns=std::chrono::duration<double,std::nano>(Clock::now()-oracle_start).count();
  rns8_device_info info{};check(hip_direct_probe(device,info));hip_direct_timing_set_enabled(false);
  const auto initialize=[&](Buffers& buffers) {buffers.allocate(device,ac,bc,cc);
    check(hip_direct_copy_host_to_device(device,buffers.a.get(),a.data(),ac*8));check(hip_direct_copy_host_to_device(device,buffers.b.get(),b.data(),bc*8));
    check(hip_direct_copy_host_to_device(device,buffers.c.get(),poison.data(),cc*8));};
  const auto compute=[&](Buffers& buffers) {bool started=false;check(hip_direct_word_mod_device(device,buffers.a.get(),buffers.b.get(),buffers.c.get(),p,4,started,method=="packed_dot4"));
    if(!started) throw std::runtime_error("successful method did not report issued output");};
  const auto audit_inputs=[&](Buffers& buffers) {check(hip_direct_copy_device_to_host(device,audit_a.data(),buffers.a.get(),ac*8));
    check(hip_direct_copy_device_to_host(device,audit_b.data(),buffers.b.get(),bc*8));if(audit_a!=a || audit_b!=b) throw std::runtime_error("word-modular inputs/padding changed");};
  Buffers resident;const auto setup_start=Clock::now();if(scope=="resident") initialize(resident);
  const double setup_ns=std::chrono::duration<double,std::nano>(Clock::now()-setup_start).count();
  const auto execute=[&]() {if(scope=="resident") compute(resident);else {Buffers cold;initialize(cold);compute(cold);
    check(hip_direct_copy_device_to_host(device,observed.data(),cold.c.get(),cc*8));cold.release();}};
  const auto prepare=[&]() {if(scope=="resident") check(hip_direct_copy_host_to_device(device,resident.c.get(),poison.data(),cc*8));};
  const auto verify=[&]() {if(scope=="resident") {check(hip_direct_copy_device_to_host(device,observed.data(),resident.c.get(),cc*8));audit_inputs(resident);}
    if(observed!=gold) throw std::runtime_error("full modular product/output padding/canary differs from Boost");};
  if(scope=="cold") {Buffers audit;initialize(audit);compute(audit);audit_inputs(audit);audit.release();}
  for(uint32_t i=0;i<warmups;++i) {prepare();execute();verify();}
  nlohmann::json samples=nlohmann::json::array();
  for(uint32_t i=0;i<repeats;++i) {
    prepare();hip_direct_timing_reset();hip_direct_timing_set_enabled(true);auto start=Clock::now();execute();
    const double instrumented=std::chrono::duration<double,std::nano>(Clock::now()-start).count();hip_direct_timing_flush_pending_events();nlohmann::json events=nlohmann::json::array();
    for(const auto& event:hip_direct_timing_snapshot()) events.push_back({{"label",event.label},{"elapsed_ns",event.microseconds>0?nlohmann::json(event.microseconds*1000):nlohmann::json(nullptr)}});
    hip_direct_timing_set_enabled(false);verify();prepare();start=Clock::now();execute();const double production=std::chrono::duration<double,std::nano>(Clock::now()-start).count();verify();
    samples.push_back({{"host_call_ns",instrumented},{"uninstrumented_host_call_ns",production},{"events",events},{"exact_reference_passed",true},{"uninstrumented_exact_reference_passed",true}});
  }
  prepare();uint32_t calls=0;double elapsed=0;const auto start=Clock::now();do {execute();++calls;elapsed=std::chrono::duration<double,std::nano>(Clock::now()-start).count();}
  while(calls<4096 && (calls<16 || elapsed<250000000));verify();
  nlohmann::json report={{"schema","rns8_word_mod_microcase_v1"},{"run_id",run_id},{"method",method},{"scope",scope},{"domain",domain},{"modulus",modulus},
    {"m",m},{"n",n},{"k",k},{"lda",p.lda},{"ldb",p.ldb},{"ldc",p.ldc},{"operand_bits",bits(modulus-1)},{"modulus_bits",bits(modulus)},
    {"device_requested",device},{"device_actual",device},{"backend_requested","hip-direct"},{"backend_actual","hip-direct"},
    {"device",device},{"device_name",info.name},{"target",info.gcn_arch},{"runtime_version",info.hip_runtime_version},{"build_manifest",nlohmann::json::parse(RNS8_BUILD_MANIFEST_JSON)},
    {"seed",6651},{"dataset_recipe","splitmix64_word_mod_seed6651_canonical_extrema_padding_v1"},{"repeats",repeats},{"warmups",warmups},
    {"input_contract","canonical_u64_padded_rows"},{"output_contract","canonical_u64_modulus_with_padding"},{"output_boundary",scope=="cold"?"host":"device"},
    {"reference_method","boost_int256_all_products_modulus_with_cpp_int_row_edge_audits"},{"a_checksum",checksum(a)},{"b_checksum",checksum(b)},{"output_checksum",checksum(gold)},
    {"a_allocation_bytes",ac*8},{"b_allocation_bytes",bc*8},{"output_allocation_bytes",cc*8},{"explicit_scratch_peak_bytes",4},{"compiler_private_scratch_bytes",nullptr},{"canary_bytes",64},
    {"logical_payload_envelope_bytes",envelope},{"memory_budget_bytes",budget},{"oracle_and_dataset_ns",oracle_ns},
    {"resident_setup_ns",scope=="resident"?nlohmann::json(setup_ns):nlohmann::json(nullptr)},
    {"slab_k",method=="packed_dot4"?nlohmann::json(4096):nlohmann::json(nullptr)},
    {"digit_diagonals",method=="packed_dot4"?15:0},{"max_diagonal_accumulator",method=="packed_dot4"?nlohmann::json(UINT64_C(2130739200)):nlohmann::json(nullptr)},
    {"arithmetic_kernel",method=="packed_dot4"?"word_mod_byte_dot4":"word_mod_double_add"},
    {"timing_scope",scope=="cold"?"allocate_upload_checked_compute_readback_checked_free":"resident_checked_compute_including_canonical_scan_and_domain_admission"},
    {"input_preservation_scope",scope=="cold"?"one_separate_resident_audit_before_warmups":"outside_each_verified_call_and_final_burst"},
    {"timer","std_chrono_steady_clock_nanoseconds"},{"production_timing_hooks_enabled",false},{"public_ownership_measured",false},
    {"asymmetric_floating_schedule_implemented",false},{"actual_hbm_bytes",nullptr},{"timing_resolution_ns",nullptr},{"isa_verified",false},{"promotion_eligible",false},
    {"samples",samples},{"sustained",{{"calls",calls},{"elapsed_ns",elapsed},{"mean_call_ns",elapsed/calls},{"minimum_calls",16},{"maximum_calls",4096},{"target_ns",250000000},
      {"independent_audit_inside_interval",false},{"final_exact_reference_passed",true}}}};
  resident.release();std::cout<<report.dump(2)<<'\n';return 0;
 } catch(const Unsupported& error) {hip_direct_timing_set_enabled(false);std::cerr<<error.what()<<'\n';return 77;}
 catch(const std::exception& error) {hip_direct_timing_set_enabled(false);std::cerr<<error.what()<<'\n';return 1;}
}
