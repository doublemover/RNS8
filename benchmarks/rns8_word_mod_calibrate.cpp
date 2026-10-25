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
  struct Allocation {void* pointer=nullptr;void* get()const noexcept{return pointer;}} a,b,c;
  int device=-1;
  Buffers()=default;Buffers(const Buffers&)=delete;Buffers& operator=(const Buffers&)=delete;
  ~Buffers() noexcept {
    if(!a.pointer&&!b.pointer&&!c.pointer)return;
    // Failed borrowed-buffer operations do not prove device completion. Keep
    // uncertain allocations until process exit instead of freeing live storage.
    if(hip_direct_synchronize(device)!=RNS8_SUCCESS)return;
    for(auto** pointer:{&c.pointer,&b.pointer,&a.pointer})
      if(*pointer&&hip_direct_free(device,*pointer)==RNS8_SUCCESS)*pointer=nullptr;
  }
  void allocate(int selected,size_t ac,size_t bc,size_t cc) {
    if(a.pointer||b.pointer||c.pointer)throw std::runtime_error("buffers already allocated");device=selected;
    check(hip_direct_allocate(device,ac*8,&a.pointer));check(hip_direct_allocate(device,bc*8,&b.pointer));check(hip_direct_allocate(device,cc*8,&c.pointer));
  }
  void release() {for(auto** pointer:{&c.pointer,&b.pointer,&a.pointer})if(*pointer){check(hip_direct_free(device,*pointer));*pointer=nullptr;}}
};
}
int main(int argc,char** argv) {
 try {
  const char* usage="usage: rns8-word-mod-calibrate DEVICE baseline|packed_dot4|canonical_dot4|byte_wmma resident|cold M N K ring|prime_field MODULUS REPEATS WARMUPS RUN_ID\n       rns8-word-mod-calibrate DEVICE asymmetric_binary64|asymmetric_fp64_mfma|asymmetric_fp64_mfma_rows|asymmetric_fp64_mfma_columns resident|repeated_b|cold M N K ring|prime_field MODULUS REPEATS WARMUPS LEFT_DIGITS RIGHT_DIGITS SLAB_K RUN_ID\nDEVICE0..63; M,N1..512 K1..65537 M*N*K<=134217728; modulus2..UINT64_MAX; repeats1..10000 warmups1..1000; digits1..4 slab1..65536. Floating requests must satisfy exact integer lambda*alpha*beta+p-1<=2^53. byte_wmma uses explicit unsigned radix256 diagonals and 16x16 output tiles; scalar binary64 remains separate. No fallback.";
  if(argc==2 && std::string(argv[1])=="--help") {std::cout<<usage<<'\n';return 0;}
  if(argc!=12 && argc!=15) throw std::invalid_argument(usage);
  const int device=int(number(argv[1],0,63));const std::string method=argv[2],scope=argv[3],domain=argv[7];
  const bool canonical_dot4=method=="canonical_dot4",dot4=canonical_dot4 || method=="packed_dot4";
  const bool mfma=method=="asymmetric_fp64_mfma" || method=="asymmetric_fp64_mfma_rows" || method=="asymmetric_fp64_mfma_columns",floating=method=="asymmetric_binary64" || mfma,wmma=method=="byte_wmma";
  if(argc!=(floating?15:12)) throw std::invalid_argument("floating method requires explicit left/right digits and slab; other methods reject them");
  const std::string run_id=argv[floating?14:11];
  const uint64_t m=number(argv[4],1,512),n=number(argv[5],1,512),k=number(argv[6],1,65537),modulus=number(argv[8],2,UINT64_MAX);
  const uint32_t repeats=uint32_t(number(argv[9],1,10000)),warmups=uint32_t(number(argv[10],1,1000));
  if((method!="baseline" && method!="packed_dot4" && !canonical_dot4 && !wmma && !floating) || (scope!="resident" && scope!="repeated_b" && scope!="cold") || (domain!="ring" && domain!="prime_field") ||
     m*n*k>134217728 || run_id.empty() || run_id.size()>128 || run_id.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")!=std::string::npos)
    throw std::invalid_argument("invalid method/scope/domain/reference work/run identity");
  word_mod::Problem p;p.domain=domain=="ring"?word_mod::Domain::ring:word_mod::Domain::prime_field;
  p.modulus=modulus;p.m=m;p.n=n;p.k=k;p.lda=k+2;p.ldb=n+3;p.ldc=n+3;
  const auto support=word_mod::query(p,true);if(support.status!=RNS8_SUCCESS) throw std::invalid_argument("word modular shape/domain is not admitted");
  word_mod::FloatingRequest request{};word_mod::FloatingSchedule schedule{};
  if(floating) {
    request={uint32_t(number(argv[11],1,4)),uint32_t(number(argv[12],1,4)),uint32_t(number(argv[13],1,65536))};
    request.concatenation=method=="asymmetric_fp64_mfma_rows"?1:method=="asymmetric_fp64_mfma_columns"?2:0;
    if(word_mod::floating_schedule(modulus,request,schedule)!=RNS8_SUCCESS)
      throw std::invalid_argument("binary64 schedule rejected: exact lambda*alpha*beta+p-1 exceeds2^53");
  }
  const uint64_t scratch_bytes=4+(request.concatenation?m*n*request.left_digits*request.right_digits*8:0);
  const uint64_t ac=m*p.lda,bc=k*p.ldb,cc=m*p.ldc+8,envelope=(3*(ac+bc)+4*cc)*8+16384+scratch_bytes-4;
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
  const auto compute=[&](Buffers& buffers) {bool started=false;check(hip_direct_word_mod_device(device,buffers.a.get(),buffers.b.get(),buffers.c.get(),p,scratch_bytes,started,dot4,floating?&request:nullptr,wmma,canonical_dot4,mfma));
    if(!started) throw std::runtime_error("successful method did not report issued output");};
  const auto audit_inputs=[&](Buffers& buffers) {check(hip_direct_copy_device_to_host(device,audit_a.data(),buffers.a.get(),ac*8));
    check(hip_direct_copy_device_to_host(device,audit_b.data(),buffers.b.get(),bc*8));if(audit_a!=a || audit_b!=b) throw std::runtime_error("word-modular inputs/padding changed");};
  Buffers resident;const auto setup_start=Clock::now();if(scope!="cold") initialize(resident);
  const double setup_ns=std::chrono::duration<double,std::nano>(Clock::now()-setup_start).count();
  const auto execute=[&]() {if(scope=="resident") compute(resident);else if(scope=="repeated_b"){check(hip_direct_copy_host_to_device(device,resident.a.get(),a.data(),ac*8));compute(resident);check(hip_direct_copy_device_to_host(device,observed.data(),resident.c.get(),cc*8));}else {Buffers cold;initialize(cold);compute(cold);
    check(hip_direct_copy_device_to_host(device,observed.data(),cold.c.get(),cc*8));cold.release();}};
  const auto prepare=[&]() {if(scope!="cold") check(hip_direct_copy_host_to_device(device,resident.c.get(),poison.data(),cc*8));};
  const auto verify=[&]() {if(scope!="cold") {check(hip_direct_copy_device_to_host(device,observed.data(),resident.c.get(),cc*8));audit_inputs(resident);}
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
    {"input_contract","canonical_u64_padded_rows"},{"output_contract","canonical_u64_modulus_with_padding"},{"output_boundary",scope=="resident"?"device":"host"},
    {"reference_method","boost_int256_all_products_modulus_with_cpp_int_row_edge_audits"},{"a_checksum",checksum(a)},{"b_checksum",checksum(b)},{"output_checksum",checksum(gold)},
    {"a_allocation_bytes",ac*8},{"b_allocation_bytes",bc*8},{"output_allocation_bytes",cc*8},{"explicit_scratch_peak_bytes",scratch_bytes},{"compiler_private_scratch_bytes",nullptr},{"canary_bytes",64},
    {"logical_payload_envelope_bytes",envelope},{"memory_budget_bytes",budget},{"oracle_and_dataset_ns",oracle_ns},
    {"resident_setup_ns",scope!="cold"?nlohmann::json(setup_ns):nlohmann::json(nullptr)},
    {"slab_k",floating?nlohmann::json(request.slab_k):(dot4 || wmma)?nlohmann::json(4096):nlohmann::json(nullptr)},
    {"digit_diagonals",(wmma || canonical_dot4)?2*word_mod::canonical_byte_width(modulus)-1:dot4?15:0},
    {"max_diagonal_accumulator",(wmma || canonical_dot4)?nlohmann::json(UINT64_C(4096)*word_mod::canonical_byte_width(modulus)*65025):dot4?nlohmann::json(UINT64_C(2130739200)):nlohmann::json(nullptr)},
    {"arithmetic_kernel",mfma?(request.concatenation?"word_mod_asymmetric_fp64_concat":"word_mod_asymmetric_fp64_mfma"):canonical_dot4?"word_mod_byte_dot4_canonical_width":wmma?"word_mod_byte_wmma":floating?"word_mod_asymmetric_binary64":method=="packed_dot4"?"word_mod_byte_dot4":"word_mod_double_add"},
    {"timing_scope",scope=="repeated_b"?"retained_b_upload_a_checked_compute_readback":scope=="cold"?"allocate_upload_checked_compute_readback_checked_free":"resident_checked_compute_including_canonical_scan_and_domain_admission"},
    {"input_preservation_scope",scope=="cold"?"one_separate_resident_audit_before_warmups":"outside_each_verified_call_and_final_burst"},
    {"timer","std_chrono_steady_clock_nanoseconds"},{"production_timing_hooks_enabled",false},{"public_ownership_measured",false},
    {"asymmetric_floating_schedule_implemented",floating},{"actual_hbm_bytes",nullptr},{"timing_resolution_ns",nullptr},{"isa_verified",false},{"promotion_eligible",false},
    {"samples",samples},{"sustained",{{"calls",calls},{"elapsed_ns",elapsed},{"mean_call_ns",elapsed/calls},{"minimum_calls",16},{"maximum_calls",4096},{"target_ns",250000000},
      {"independent_audit_inside_interval",false},{"final_exact_reference_passed",true}}}};
  if(floating) {
    const cpp_int lhs=cpp_int(request.slab_k)*schedule.alpha*schedule.beta+modulus-1;
    const cpp_int coefficient=cpp_int(request.slab_k)*(schedule.alpha-1)*(schedule.beta-1);
    nlohmann::json weights=nlohmann::json::array();for(auto weight:schedule.weights) weights.push_back(weight);
    report["left_digits"]=request.left_digits;report["right_digits"]=request.right_digits;
    report["floating_schedule"]={{"version",1},{"arithmetic",mfma?"cdna_fp64_mfma_integer_coefficients":"scalar_binary64_fma_integer_coefficients"},
      {"left_digits",request.left_digits},{"right_digits",request.right_digits},{"slab_k",request.slab_k},
      {"alpha",schedule.alpha},{"beta",schedule.beta},{"mantissa_bits",53},
      {"admission_lhs_decimal",lhs.str()},{"admission_limit_decimal","9007199254740992"},
      {"coefficient_max_decimal",coefficient.str()},{"weights_stride",4},{"weights",weights},
      {"weighting","overflow_safe_integer_modular_recombination"},{"matrix_engine",mfma}};
  }
  if(mfma) report["fp64_mfma_schedule"]={{"version",1},{"opcode","mfma_f64_16x16x4f64"},{"wave_size",64},{"tile_m",16},{"tile_n",16},{"instruction_k",4},{"modifiers",{0,0,0}},{"layout","cdna_fp64_interleaved_rows_v1"},{"a_map","row=lane%16;k=lane/16"},{"b_map","k=lane/16;col=lane%16"},{"c_map","row=4*reg+lane/16;col=lane%16"},{"active_terms_bound",request.slab_k},{"padded_instruction_terms",((request.slab_k+3)/4)*4},{"tail_padding","exact_zero"},{"source_pairs_per_instruction_tile",request.concatenation?1:request.left_digits*request.right_digits},{"concatenation",request.concatenation==1?"word_major":request.concatenation==2?"cell_major":"none"},{"coefficient_payload_bytes",scratch_bytes-4},{"physical_isa_verified",false}};
  if(canonical_dot4) {
    const auto bytes=word_mod::canonical_byte_width(modulus);
    report["canonical_dot4_schedule"]={{"version",1},{"bytes_per_operand",bytes},{"diagonals",2*bytes-1},
      {"source_dot4_updates_per_four_k",bytes*bytes},{"slab_k",4096},
      {"maximum_accumulator",UINT64_C(4096)*bytes*65025},{"width_proof","canonical_inputs_less_than_modulus"},
      {"zero_pad_k_tail",true},{"full_width_control","packed_dot4"},{"issued_instruction_count_measured",false}};
  }
  if(wmma) {
    const unsigned bytes=(bits(modulus-1)+7)/8;
    report["operand_bytes"]=bytes;
    report["byte_wmma_schedule"]={{"version",1},{"arithmetic","unsigned_radix256_wmma_diagonal_integer_recombination"},
      {"bytes_per_operand",bytes},{"diagonals",2*bytes-1},{"slab_k",4096},
      {"maximum_accumulator",UINT64_C(4096)*bytes*65025},{"tile_m",16},{"tile_n",16},{"tile_k",16},{"threads_per_block",32},
      {"output_tile_count",((m+15)/16)*((n+15)/16)},{"zero_pad_spatial_and_k_tails",true},
      {"left_signed",false},{"right_signed",false},{"saturating_accumulator",false},
      {"native_operands_reread_per_diagonal",true},{"asymmetric_floating_decomposition",false}};
  }
  resident.release();std::cout<<report.dump(2)<<'\n';return 0;
 } catch(const Unsupported& error) {hip_direct_timing_set_enabled(false);std::cerr<<error.what()<<'\n';return 77;}
 catch(const std::exception& error) {hip_direct_timing_set_enabled(false);std::cerr<<error.what()<<'\n';return 1;}
}
