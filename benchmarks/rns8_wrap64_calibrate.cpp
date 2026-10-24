// Private canonical wrap64 controls. No public dispatch qualification is implied.
#include "backend_hip_direct/hip_backend.hpp"
#include "backend_rocwmma/rocwmma_backend.hpp"
#include "backend_wrap64/wrap64_hip.hpp"
#include <boost/multiprecision/cpp_int.hpp>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
namespace {
using namespace rns8::detail;
using boost::multiprecision::cpp_int;
using boost::multiprecision::int256_t;
using json=nlohmann::json;
using Clock=std::chrono::steady_clock;
constexpr uint64_t budget=256ull*1024*1024,canary=UINT64_C(0xa5a5a5a5a5a5a5a5);
constexpr const char* recipe="splitmix64_u64_extrema_seed3135_v1";
constexpr std::array<uint64_t,8> edges{{0,1,UINT64_MAX,UINT64_C(0x8080808080808080),UINT64_C(0x7f807f807f807f80),UINT64_C(0xfefdfcfbfaf9f8f7),UINT64_C(0x0101010101010101),UINT64_C(0x8000000000000000)}};
bool cleanup_failed=false;
void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
void check(rns8_status s){if(s!=RNS8_SUCCESS)throw std::runtime_error(rns8_status_string(s));}
struct Unsupported:std::runtime_error{using std::runtime_error::runtime_error;};
uint64_t number(const char* s,uint64_t lo,uint64_t hi){uint64_t value=0;const auto end=s+std::strlen(s);const auto p=std::from_chars(s,end,value);if(p.ec!=std::errc{}||p.ptr!=end||value<lo||value>hi)throw std::invalid_argument("numeric argument outside supported domain");return value;}
uint64_t ns(Clock::time_point t){return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-t).count());}
uint64_t next(uint64_t& state){uint64_t z=(state+=UINT64_C(0x9e3779b97f4a7c15));z=(z^(z>>30))*UINT64_C(0xbf58476d1ce4e5b9);z=(z^(z>>27))*UINT64_C(0x94d049bb133111eb);return z^(z>>31);}
std::string checksum(const std::vector<uint64_t>& values){uint64_t h=UINT64_C(14695981039346656037);for(auto value:values)for(unsigned b=0;b<8;++b)h=(h^((value>>(8*b))&255))*UINT64_C(1099511628211);std::ostringstream s;s<<std::hex<<std::setfill('0')<<std::setw(16)<<h;return s.str();}
struct Case{int device;std::string candidate,scope,pattern,run;size_t m,n,k;uint32_t reuse,repeats,warmups;};
void help(){std::cout<<"rns8-wrap64-calibrate DEVICE CANDIDATE SCOPE M N K PATTERN REUSE REPEATS WARMUPS RUN_ID\n"
 "DEVICE 0..63; CANDIDATE signed_radix|native_u8|direct_v4|rocwmma_corrected; no fallback.\n"
 "SCOPE resident|roundtrip|cold. Cold allocates/packs/frees input and transport buffers each batch; output allocation and context retained.\n"
 "M,N 1..512; K 1..131071; M*N*K<=134217728; rocwmma_corrected requires K<=32768.\n"
 "PATTERN seeded|carry|zero; fixed dataset seed3135. REUSE 1..256; REPEATS/WARMUPS 1..1000; RUN_ID ASCII alnum/_/- 1..128.\n"
 "Rejects active RNS8_WRAP64_HIP_COLPAIR_EXPERIMENT; direct_v4 is an explicit control.\n"
 "Paired diagnostic HIP-event/phase timing and hook-disabled production calls; full Boost/byte-canary/export-padding audit outside timing.\n";}
Case parse(int argc,char** argv){require(argc==12,"expected eleven positional arguments; use --help");Case c{};c.device=int(number(argv[1],0,63));c.candidate=argv[2];c.scope=argv[3];c.pattern=argv[7];c.run=argv[11];
 require(c.candidate=="signed_radix"||c.candidate=="native_u8"||c.candidate=="direct_v4"||c.candidate=="rocwmma_corrected","invalid candidate");require(c.scope=="resident"||c.scope=="roundtrip"||c.scope=="cold","invalid scope");
 c.m=size_t(number(argv[4],1,512));c.n=size_t(number(argv[5],1,512));c.k=size_t(number(argv[6],1,131071));require(uint64_t(c.m)*c.n*c.k<=134217728,"reference work exceeds bound");require(c.candidate!="rocwmma_corrected"||c.k<=32768,"existing rocWMMA control has explicit K<=32768 cap");
 require(c.pattern=="seeded"||c.pattern=="carry"||c.pattern=="zero","invalid pattern");c.reuse=uint32_t(number(argv[8],1,256));c.repeats=uint32_t(number(argv[9],1,1000));c.warmups=uint32_t(number(argv[10],1,1000));
 require(!c.run.empty()&&c.run.size()<=128&&std::all_of(c.run.begin(),c.run.end(),[](char x){return(x>='a'&&x<='z')||(x>='A'&&x<='Z')||(x>='0'&&x<='9')||x=='_'||x=='-';}),"invalid run ID");
 require(!wrap64_hip_colpair_experiment_enabled(),"active colpair experiment conflicts with matched direct_v4 control");return c;}
struct Buffer{int device;void* ptr=nullptr;size_t bytes=0;explicit Buffer(int d):device(d){}Buffer(const Buffer&)=delete;Buffer& operator=(const Buffer&)=delete;
 ~Buffer(){if(ptr&&hip_direct_free(device,ptr)!=RNS8_SUCCESS)cleanup_failed=true;}
 void allocate(size_t count){require(!ptr,"buffer already owned");check(hip_direct_allocate(device,count,&ptr));bytes=count;}
 void close(){if(ptr){check(hip_direct_free(device,ptr));ptr=nullptr;bytes=0;}}
};
struct ContextDelete{void operator()(rns8_context* p)const noexcept{if(p&&rns8_destroy_context(p)!=RNS8_SUCCESS)cleanup_failed=true;}};
struct Phases{uint64_t allocation=0,pack=0,gemm=0,export_result=0,release=0;};
json run(const Case& c){const uint64_t ab=uint64_t(c.m)*c.k+uint64_t(c.k)*c.n,mn=uint64_t(c.m)*c.n;
 const uint64_t reference_bytes=ab*sizeof(int256_t),oracle_peak=8*ab+8*mn+reference_bytes+2048;
 const uint64_t execution_peak=24*ab+64*mn+24*c.m+4096;
 require(std::max(oracle_peak,execution_peak)<=budget,"logical live-payload envelope exceeds 256MiB");
 const auto oracle_begin=Clock::now();uint64_t state=3135;std::vector<uint64_t> a(c.m*c.k),b(c.k*c.n),expected(mn);
 for(auto* values:{&a,&b})for(size_t i=0;i<values->size();++i){auto word=next(state);if(c.pattern=="zero")word=0;else if(c.pattern=="carry")word=edges[i%8];else if(i%257<8)word=edges[i%257];(*values)[i]=word;}
 {std::vector<int256_t> aa(a.begin(),a.end()),bt(b.size());for(size_t t=0;t<c.k;++t)for(size_t col=0;col<c.n;++col)bt[col*c.k+t]=b[t*c.n+col];const int256_t mask=(int256_t(1)<<64)-1;
  for(size_t r=0;r<c.m;++r)for(size_t col=0;col<c.n;++col){int256_t sum=0;uint64_t native=0;for(size_t t=0;t<c.k;++t){sum+=aa[r*c.k+t]*bt[col*c.k+t];native+=a[r*c.k+t]*b[t*c.n+col];}
   // Full unsigned product < K*2^128 < 2^145: exact in Boost int256.
   expected[r*c.n+col]=(sum&mask).convert_to<uint64_t>();require(native==expected[r*c.n+col],"native ring arithmetic disagrees with full Boost product");
   if(col==0||col+1==c.n){cpp_int audit=0;for(size_t t=0;t<c.k;++t)audit+=cpp_int(a[r*c.k+t])*b[t*c.n+col];require(cpp_int(sum)==audit,"cpp_int row-edge audit disagrees with fixed Boost");}
  }
 }
 const auto oracle_ns=ns(oracle_begin);const auto setup_begin=Clock::now();hip_direct_timing_set_enabled(false);
 rns8_context_options options{};options.struct_size=sizeof(options);options.abi_version=RNS8_ABI_VERSION;options.requested_backend=RNS8_BACKEND_HIP_DIRECT;rns8_context* raw=nullptr;const auto cs=rns8_create_context(c.device,&options,&raw);
 if(cs==RNS8_UNSUPPORTED_BACKEND||cs==RNS8_UNSUPPORTED_ARCH||cs==RNS8_UNSUPPORTED_OS)throw Unsupported(rns8_status_string(cs));check(cs);std::unique_ptr<rns8_context,ContextDelete> ctx(raw);
 rns8_device_info device{};device.struct_size=sizeof(device);device.abi_version=RNS8_ABI_VERSION;check(rns8_get_device_info(ctx.get(),&device));require(device.backend==RNS8_BACKEND_HIP_DIRECT&&device.device_id==c.device&&device.gcn_arch[0],"explicit device identity unavailable");
 Buffer wa(c.device),wb(c.device),wu(c.device),we(c.device),output(c.device),audit_export(c.device);output.allocate(size_t(mn*8+64));audit_export.allocate(size_t(mn*8));
 std::vector<uint8_t> poison(size_t(mn*8+64),0xA5),raw_output(poison.size());std::vector<uint64_t> host_output(c.m*(c.n+3),canary);
 if(c.scope!="cold"){wa.allocate(a.size()*8);wb.allocate(b.size()*8);}
 const auto pack=[&](Buffer& da,Buffer& db,Buffer& upload){check(wrap64_hip_pack_u64_device(c.device,a.data(),&upload.ptr,&upload.bytes,da.ptr,int64_t(c.m),int64_t(c.k),int64_t(c.k)));check(wrap64_hip_pack_u64_device(c.device,b.data(),&upload.ptr,&upload.bytes,db.ptr,int64_t(c.k),int64_t(c.n),int64_t(c.n)));};
 json resident_prepare=nullptr;if(c.scope=="resident"){const auto t=Clock::now();pack(wa,wb,wu);resident_prepare=ns(t);}const auto setup_ns=ns(setup_begin);
 const auto invoke=[&](bool phases){Phases result;Buffer ca(c.device),cb(c.device),cu(c.device),ce(c.device);auto& da=c.scope=="cold"?ca:wa;auto& db=c.scope=="cold"?cb:wb;auto& upload=c.scope=="cold"?cu:wu;auto& exported=c.scope=="cold"?ce:we;
  auto t=phases?Clock::now():Clock::time_point{};
  if(c.scope=="cold"){da.allocate(a.size()*8);db.allocate(b.size()*8);}if(phases){result.allocation=c.scope=="cold"?ns(t):0;t=Clock::now();}
  if(c.scope!="resident")pack(da,db,upload);if(phases){result.pack=c.scope!="resident"?ns(t):0;t=Clock::now();}
  for(uint32_t i=0;i<c.reuse;++i){rns8_status status;
   if(c.candidate=="direct_v4")status=wrap64_hip_gemm_byte_limbs_device_resident(c.device,da.ptr,db.ptr,output.ptr,int64_t(c.m),int64_t(c.n),int64_t(c.k));
   else if(c.candidate=="rocwmma_corrected")status=rocwmma_wrap64_gemm_byte_limbs_candidate_device(c.device,da.ptr,db.ptr,output.ptr,int64_t(c.m),int64_t(c.n),int64_t(c.k));
   else{bool started=false;status=hip_direct_wrap_radix_wmma_device(c.device,da.ptr,db.ptr,output.ptr,int64_t(c.m),int64_t(c.n),int64_t(c.k),c.candidate=="signed_radix",4,started);if(status==RNS8_SUCCESS)require(started,"radix candidate returned success without committing output");}
   if(status==RNS8_UNSUPPORTED_BACKEND||status==RNS8_UNSUPPORTED_ARCH)throw Unsupported(rns8_status_string(status));check(status);
  }
  if(phases){result.gemm=ns(t);t=Clock::now();}
  if(c.scope!="resident")check(wrap64_hip_export_u64_device(c.device,output.ptr,&exported.ptr,&exported.bytes,int64_t(c.m),int64_t(c.n),host_output.data(),int64_t(c.n+3)));
  if(phases){result.export_result=c.scope!="resident"?ns(t):0;t=Clock::now();}
  if(c.scope=="cold"){ce.close();cu.close();cb.close();ca.close();}if(phases)result.release=c.scope=="cold"?ns(t):0;return result;
 };
 const auto prepare=[&](){check(hip_direct_copy_host_to_device(c.device,output.ptr,poison.data(),poison.size()));std::fill(host_output.begin(),host_output.end(),canary);};
 const auto audit=[&](){if(c.scope=="resident")check(wrap64_hip_export_u64_device(c.device,output.ptr,&audit_export.ptr,&audit_export.bytes,int64_t(c.m),int64_t(c.n),host_output.data(),int64_t(c.n+3)));
  for(size_t r=0;r<c.m;++r)for(size_t col=0;col<c.n+3;++col)require(host_output[r*(c.n+3)+col]==(col<c.n?expected[r*c.n+col]:canary),"full Boost product or export padding mismatch");
  check(hip_direct_copy_device_to_host(c.device,raw_output.data(),output.ptr,raw_output.size()));for(size_t cell=0;cell<mn;++cell)for(unsigned byte=0;byte<8;++byte)require(raw_output[cell*8+byte]==uint8_t(expected[cell]>>(8*byte)),"canonical output byte mismatch");
  require(std::all_of(raw_output.begin()+size_t(mn*8),raw_output.end(),[](uint8_t x){return x==0xA5;}),"device output canary changed");
 };
 for(uint32_t i=0;i<c.warmups;++i){prepare();invoke(false);audit();}
 json samples=json::array();std::vector<uint64_t> production;
 for(uint32_t i=0;i<c.repeats;++i){prepare();hip_direct_timing_reset();hip_direct_timing_set_enabled(true);const auto t=Clock::now();const auto p=invoke(true);const auto instrumented=ns(t);hip_direct_timing_flush_pending_events();json events=json::array();for(const auto& event:hip_direct_timing_snapshot())events.push_back({{"label",event.label},{"elapsed_ns",event.microseconds>0?json(event.microseconds*1000):json(nullptr)}});hip_direct_timing_set_enabled(false);audit();
  prepare();const auto start=Clock::now();invoke(false);const auto plain=ns(start);require(plain>0,"production call below timer resolution");audit();production.push_back(plain);
  samples.push_back({{"host_call_ns",instrumented?json(instrumented):json(nullptr)},{"uninstrumented_host_call_ns",plain},{"events",events},{"exact_reference_passed",true},{"uninstrumented_exact_reference_passed",true},
   {"instrumented_phases_ns",{{"input_buffer_allocation",p.allocation},{"input_pack_and_upload",p.pack},{"gemm_calls",p.gemm},{"output_export",p.export_result},{"input_and_transport_release",p.release}}}});
 }
 hip_direct_timing_set_enabled(false);prepare();const auto sustained_begin=Clock::now();uint64_t calls=0,elapsed=0;do{invoke(false);++calls;elapsed=ns(sustained_begin);}while(calls<4096&&(calls<16||elapsed<250000000));audit();std::sort(production.begin(),production.end());const double median=production.size()%2?double(production[production.size()/2]):(double(production[production.size()/2-1])+double(production[production.size()/2]))/2;
 const bool radix=c.candidate=="signed_radix"||c.candidate=="native_u8";
 const std::string selection=c.candidate=="direct_v4"?wrap64_hip_selected_kernel_for_k(int64_t(c.k)):c.candidate=="rocwmma_corrected"?"rocwmma_wrap64_byte_gemm36_candidate_kernel":c.candidate=="signed_radix"?"wrap64_signed_radix_wmma":"wrap64_native_u8_wmma";
 return {{"schema","rns8_wrap64_microcase_v1"},{"promotion_eligible",false},{"public_ownership_measured",false},{"run_id",c.run},{"candidate",c.candidate},{"scope",c.scope},{"device_requested",c.device},{"device_actual",device.device_id},{"device_name",device.name},{"target",device.gcn_arch},{"hip_runtime_version",device.hip_runtime_version},{"hip_driver_version",device.hip_driver_version},{"context_backend","hip-direct"},{"execution_backend",c.candidate=="rocwmma_corrected"?"rocwmma":"hip-direct"},{"build_manifest",json::parse(RNS8_BUILD_MANIFEST_JSON)},
  {"m",c.m},{"n",c.n},{"k",c.k},{"pattern",c.pattern},{"reuse_count",c.reuse},{"repeats",c.repeats},{"warmups",c.warmups},{"semantics","wrap_u64_mod_2_64"},{"input_domain","canonical_compact_byte_limbs_little_endian"},{"output_contract","canonical_compact_low64_byte_limbs"},{"dataset_recipe",recipe},{"dataset_seed",3135},{"a_checksum",checksum(a)},{"b_checksum",checksum(b)},{"output_checksum",checksum(expected)},
  {"reference_method","boost_int256_full_unsigned_product_native_u64_full_audit_cpp_int_row_edges"},{"reference_operand_storage_bytes",reference_bytes},{"reference_fixed_operand_object_bytes",sizeof(int256_t)},{"reference_buffers_released_before_setup",true},{"reference_and_dataset_ns",oracle_ns},{"setup_ns",setup_ns},{"resident_input_prepare_ns",resident_prepare},{"setup_scope","device_probe_common_output_and_audit_buffers_optional_resident_input_alloc_pack"},{"cold_boundary","context_and_output_retained_input_and_transport_alloc_pack_export_free_per_batch"},
  {"kernel_selection",selection},{"colpair_experiment_enabled",false},{"mma_terms_per_fold",radix?json(16):json(nullptr)},{"mathematical_pair_i32_cap",radix?json(c.candidate=="signed_radix"?131071:33025):json(nullptr)},{"static_lds_bytes",radix?json(4096):json(nullptr)},{"explicit_call_scratch_bytes",radix?4:0},{"rocwmma_max_k",32768},
  {"timer","std_chrono_steady_clock_nanoseconds"},{"clock_period_numerator",Clock::period::num},{"clock_period_denominator",Clock::period::den},{"production_timing_hooks_enabled",false},{"phase_clocks_inside_production_call",false},{"memory_budget_bytes",budget},{"logical_live_payload_envelope_bytes",std::max(oracle_peak,execution_peak)},{"memory_exclusions","allocator bookkeeping, driver/HIP events, context and fixed metadata; logical payload only, not RSS"},
  {"device_output_canary_bytes",64},{"device_output_canary_checked",true},{"host_export_padding_columns",3},{"host_export_padding_checked",true},{"verified_outputs_per_batch",1},{"samples",samples},{"median_production_batch_ns",median},{"gemm_calls_per_second",1e9*double(c.reuse)/median},
  {"sustained",{{"calls",calls},{"gemm_calls",calls*c.reuse},{"elapsed_ns",elapsed},{"mean_batch_ns",double(elapsed)/double(calls)},{"mean_gemm_ns",double(elapsed)/double(calls*c.reuse)},{"minimum_calls",16},{"maximum_calls",4096},{"target_ns",250000000},{"stop_reason",elapsed>=250000000?"duration":"call_cap"},{"independent_audit_inside_interval",false},{"final_exact_reference_passed",true}}}};
}
}
int main(int argc,char** argv){if(argc==2&&std::string(argv[1])=="--help"){help();return 0;}try{const auto result=run(parse(argc,argv));require(!cleanup_failed,"owned buffer/context cleanup failed");std::cout<<result.dump()<<'\n';return 0;}catch(const Unsupported& error){std::cerr<<"unsupported explicit wrap candidate/device: "<<error.what()<<'\n';return 77;}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}
