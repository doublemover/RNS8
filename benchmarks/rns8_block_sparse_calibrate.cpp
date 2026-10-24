// Public generation-bound block-sparse ownership and whole-call cost controls.
#include <rns8/rns8.h>
#include <rns8/moduli.h>
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
using json=nlohmann::json;
using boost::multiprecision::cpp_int;
using boost::multiprecision::int256_t;
using Clock=std::chrono::steady_clock;
constexpr uint64_t budget=256ull*1024*1024;
constexpr uint64_t canary=UINT64_C(0xa5a5a5a5a5a5a5a5);
constexpr const char* recipe="splitmix64_signed64_block_masks_seed3135_v1";
bool cleanup_failed=false;
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
void check(rns8_status s){if(s!=RNS8_SUCCESS)throw std::runtime_error(rns8_status_string(s));}
struct Unsupported:std::runtime_error {using std::runtime_error::runtime_error;};
template<class T,rns8_status(*Destroy)(T*)> struct Delete {
  void operator()(T* p)const noexcept{if(p && Destroy(p)!=RNS8_SUCCESS)cleanup_failed=true;}
};
template<class T,rns8_status(*Destroy)(T*)>using Owner=std::unique_ptr<T,Delete<T,Destroy>>;
using Context=Owner<rns8_context,rns8_destroy_context>;
using Plan=Owner<rns8_plan,rns8_destroy_plan>;
using Matrix=Owner<rns8_matrix,rns8_destroy_matrix>;
using Workspace=Owner<rns8_workspace,rns8_destroy_workspace>;
using Sparse=Owner<rns8_block_sparse_operand,rns8_destroy_block_sparse_operand>;
uint64_t number(const char* s,uint64_t low,uint64_t high){uint64_t v=0;const auto end=s+std::strlen(s);const auto p=std::from_chars(s,end,v);if(p.ec!=std::errc{}||p.ptr!=end||v<low||v>high)throw std::invalid_argument("numeric argument outside supported domain");return v;}
uint64_t ns(Clock::time_point t){return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-t).count());}
uint64_t next(uint64_t& state){uint64_t z=(state+=UINT64_C(0x9e3779b97f4a7c15));z=(z^(z>>30))*UINT64_C(0xbf58476d1ce4e5b9);z=(z^(z>>27))*UINT64_C(0x94d049bb133111eb);return z^(z>>31);}
template<class T>std::string checksum(const std::vector<T>& values){uint64_t h=UINT64_C(14695981039346656037);for(auto value:values)for(unsigned b=0;b<8;++b){h=(h^((uint64_t(value)>>(8*b))&255))*UINT64_C(1099511628211);}std::ostringstream s;s<<std::hex<<std::setfill('0')<<std::setw(16)<<h;return s.str();}
struct Case{int device;std::string mode,pattern,run;size_t m,n,k;uint32_t reuse,threshold,repeats,warmups;};
bool dense(const Case& c){return c.mode=="dense";}
bool cold(const Case& c){return c.mode.find("_cold")!=std::string::npos;}
bool hybrid(const Case& c){return c.mode.find("hybrid")!=std::string::npos;}
void help(){std::cout<<"rns8-block-sparse-calibrate DEVICE MODE M N K PATTERN REUSE_COUNT HYBRID_THRESHOLD REPEATS WARMUPS RUN_ID\n"
  "Explicit direct-HIP DEVICE 0..63; no CPU fallback. MODE dense | sparse_scalar | sparse_wmma | sparse_hybrid | sparse_scalar_cold | sparse_wmma_cold | sparse_hybrid_cold.\n"
  "M,N 1..512; K 1..65537; M*N*K<=134217728. PATTERN dense|mixed|disjoint.\n"
  "REUSE_COUNT 1..256 GEMMs per wall interval; threshold uint32 only for hybrid, otherwise zero. REPEATS/WARMUPS 1..1000.\n"
  "Run ID 1..128 ASCII alnum/_/-. Signed exact-wide prefix20, full Boost reference outside timing, 256MiB logical live-payload cap.\n"
  "Warm sparse setup prepares once; cold modes include two preparations, reuse GEMMs and checked destruction inside every call.\n";}
Case parse(int argc,char** argv){require(argc==12,"expected eleven positional arguments; use --help");Case c{};c.device=int(number(argv[1],0,63));c.mode=argv[2];c.pattern=argv[6];c.run=argv[11];
  const std::array<std::string,7> modes{{"dense","sparse_scalar","sparse_wmma","sparse_hybrid","sparse_scalar_cold","sparse_wmma_cold","sparse_hybrid_cold"}};
  require(std::find(modes.begin(),modes.end(),c.mode)!=modes.end(),"invalid explicit mode");
  c.m=size_t(number(argv[3],1,512));c.n=size_t(number(argv[4],1,512));c.k=size_t(number(argv[5],1,65537));require(uint64_t(c.m)*c.n*c.k<=134217728,"reference work exceeds bound");
  require(c.pattern=="dense"||c.pattern=="mixed"||c.pattern=="disjoint","invalid exact-zero recipe pattern");
  c.reuse=uint32_t(number(argv[7],1,256));c.threshold=uint32_t(number(argv[8],0,UINT32_MAX));require(hybrid(c)||c.threshold==0,"threshold is relevant only to hybrid modes");
  c.repeats=uint32_t(number(argv[9],1,1000));c.warmups=uint32_t(number(argv[10],1,1000));
  require(!c.run.empty()&&c.run.size()<=128&&std::all_of(c.run.begin(),c.run.end(),[](char x){return (x>='a'&&x<='z')||(x>='A'&&x<='Z')||(x>='0'&&x<='9')||x=='_'||x=='-';}),"invalid run ID");return c;}
bool active(const Case& c,bool lhs,size_t tile,size_t kb){if(c.pattern=="dense")return true;if(c.pattern=="disjoint")return kb%2==(lhs?0u:1u);if(tile%3==0)return true;if(tile%3==2)return kb==0;return kb%2==(lhs?0u:1u);}
struct Structure{uint64_t a_blocks=0,b_blocks=0,payload=0,metadata=0,intersections=0,empty_tiles=0,scalar_tiles=0,wmma_tiles=0;};
Structure structure(const Case& c){Structure s;const size_t rt=(c.m+15)/16,ct=(c.n+15)/16,kt=(c.k+63)/64;
  for(size_t r=0;r<rt;++r)for(size_t k=0;k<kt;++k)if(active(c,true,r,k))++s.a_blocks;
  for(size_t col=0;col<ct;++col)for(size_t k=0;k<kt;++k)if(active(c,false,col,k))++s.b_blocks;
  s.payload=(s.a_blocks+s.b_blocks)*20*1024;s.metadata=4*(rt+ct+2+s.a_blocks+s.b_blocks);
  for(size_t r=0;r<rt;++r)for(size_t col=0;col<ct;++col){uint64_t count=0;for(size_t k=0;k<kt;++k)if(active(c,true,r,k)&&active(c,false,col,k))++count;s.intersections+=count;if(!count)++s.empty_tiles;if(count<=c.threshold)++s.scalar_tiles;else ++s.wmma_tiles;}return s;}
Plan plan(rns8_context* ctx,const Case& c){rns8_gemm_desc d{};d.struct_size=sizeof(d);d.abi_version=RNS8_ABI_VERSION;d.semantics=RNS8_EXACT_WIDE_SIGNED;d.bound_kind=RNS8_BOUND_NONE;d.requested_backend=RNS8_BACKEND_HIP_DIRECT;d.m=int64_t(c.m);d.n=int64_t(c.n);d.k=int64_t(c.k);d.max_prefix=20;d.flags=RNS8_PLAN_FORCE_FIXED_PREFIX;rns8_plan* p=nullptr;check(rns8_create_plan(ctx,&d,&p));return Plan(p);}
Matrix matrix(rns8_context* ctx,size_t rows,size_t cols,bool native){rns8_matrix_desc d{};d.struct_size=sizeof(d);d.abi_version=RNS8_ABI_VERSION;d.rows=int64_t(rows);d.cols=int64_t(cols);d.logical_ld=int64_t(cols);d.logical_layout=RNS8_LAYOUT_ROW_MAJOR;d.semantics=RNS8_EXACT_WIDE_SIGNED;d.bound_kind=RNS8_BOUND_NONE;d.max_prefix=20;d.flags=native?RNS8_MATRIX_RETAIN_NATIVE_INPUT:0;rns8_matrix* p=nullptr;check(rns8_create_matrix(ctx,&d,&p));return Matrix(p);}
rns8_matrix_storage_info storage(const rns8_matrix* matrix){rns8_matrix_storage_info s{};s.struct_size=sizeof(s);s.abi_version=RNS8_ABI_VERSION;check(rns8_get_matrix_storage_info(matrix,&s));return s;}
uint64_t payload(const rns8_matrix_storage_info& s){return s.host_residue_bytes+s.device_residue_bytes+s.host_byte_limb_bytes+s.device_byte_limb_bytes+s.host_native_bytes+s.device_native_bytes;}
json allocations(const std::array<const rns8_matrix*,3>& matrices,uint64_t limit){json values=json::array();uint64_t total=0;
  for(const auto* matrix:matrices){rns8_matrix_allocation_info a{};a.struct_size=sizeof(a);a.abi_version=RNS8_ABI_VERSION;check(rns8_get_matrix_allocation_info(matrix,&a));
    require(a.host_payload_capacity_bytes<=limit-total,"queried host matrix capacity exceeds payload envelope");total+=a.host_payload_capacity_bytes;
    require(a.device_total_capacity_bytes<=limit-total,"queried device matrix capacity exceeds payload envelope");total+=a.device_total_capacity_bytes;
    values.push_back({{"logical_primary_bytes",a.logical_primary_bytes},{"host_payload_capacity_bytes",a.host_payload_capacity_bytes},{"device_payload_capacity_bytes",a.device_payload_capacity_bytes},
      {"host_current_bytes",a.host_current_bytes},{"device_current_bytes",a.device_current_bytes},{"device_upload_staging_bytes",a.device_upload_staging_bytes},
      {"device_export_staging_bytes",a.device_export_staging_bytes},{"device_status_and_schedule_bytes",a.device_status_and_schedule_bytes},{"device_total_capacity_bytes",a.device_total_capacity_bytes}});
  }return values;
}
json run(const Case& c){const auto counts=structure(c);const uint64_t ab=uint64_t(c.m)*c.k+uint64_t(c.k)*c.n,mn=uint64_t(c.m)*c.n;
  const uint64_t reference_bytes=ab*sizeof(int256_t),oracle_peak=8*ab+24*mn+reference_bytes+2048;
  const uint64_t modeled=80*ab+112*mn+2*(counts.payload+counts.metadata)+8*std::max(uint64_t(c.m)*c.k,uint64_t(c.k)*c.n)+counts.metadata+4;
  require(std::max(oracle_peak,modeled)<=budget,"logical live payload envelope exceeds 256MiB");
  cpp_int product=1;for(uint32_t p=0;p<20;++p)product*=rns8_default_modulus(p);require(2*(cpp_int(c.k)<<126)<product,"default prefix20 cannot admit full-expression signed64 bound");
  const auto oracle_begin=Clock::now();std::vector<int64_t> av(c.m*c.k),bv(c.k*c.n);uint64_t state=3135;
  const auto generate=[&](std::vector<int64_t>& values,bool lhs){for(size_t i=0;i<values.size();++i){const auto bits=next(state);int64_t value=0;std::memcpy(&value,&bits,8);if(i%257==0)value=INT64_MIN;else if(i%257==1)value=INT64_MAX;
    const size_t tile=lhs?(i/c.k)/16:(i%c.n)/16,kb=lhs?(i%c.k)/64:(i/c.n)/64;values[i]=active(c,lhs,tile,kb)?(value?value:int64_t(1)):0;}};
  generate(av,true);generate(bv,false);std::vector<uint64_t> expected(3*mn);
  { // Fixed-width Boost operands are released before GPU state is created.
    std::vector<int256_t> aa(av.begin(),av.end()),bt(bv.size());for(size_t t=0;t<c.k;++t)for(size_t col=0;col<c.n;++col)bt[col*c.k+t]=bv[t*c.n+col];
    const int256_t mask=(int256_t(1)<<64)-1;
    for(size_t r=0;r<c.m;++r)for(size_t col=0;col<c.n;++col){int256_t sum=0;for(size_t t=0;t<c.k;++t)sum+=aa[r*c.k+t]*bt[col*c.k+t];
      // |sum| <= K*2^126 < 2^143, so this is an exact Boost product,
      // never a native128 substitute. cpp_int separately audits row edges.
      if(col==0||col+1==c.n){cpp_int audit=0;for(size_t t=0;t<c.k;++t)audit+=cpp_int(av[r*c.k+t])*bv[t*c.n+col];require(cpp_int(sum)==audit,"fixed Boost reference disagrees with cpp_int edge audit");}
      if(sum<0)sum+=int256_t(1)<<192;for(size_t limb=0;limb<3;++limb){expected[(r*c.n+col)*3+limb]=(sum&mask).convert_to<uint64_t>();sum>>=64;}
    }
  }
  const auto oracle_ns=ns(oracle_begin);
  const auto begin=Clock::now();
  rns8_context_options opt{};opt.struct_size=sizeof(opt);opt.abi_version=RNS8_ABI_VERSION;opt.requested_backend=RNS8_BACKEND_HIP_DIRECT;rns8_context* raw=nullptr;const auto cs=rns8_create_context(c.device,&opt,&raw);
  if(cs==RNS8_UNSUPPORTED_ARCH||cs==RNS8_UNSUPPORTED_BACKEND||cs==RNS8_UNSUPPORTED_OS)throw Unsupported(rns8_status_string(cs));check(cs);Context ctx(raw);
  rns8_device_info device{};device.struct_size=sizeof(device);device.abi_version=RNS8_ABI_VERSION;check(rns8_get_device_info(ctx.get(),&device));require(device.backend==RNS8_BACKEND_HIP_DIRECT&&device.device_id==c.device&&device.gcn_arch[0],"selected HIP identity unavailable");
  auto selected=plan(ctx.get(),c);rns8_plan_backend_info pi{};pi.struct_size=sizeof(pi);pi.abi_version=RNS8_ABI_VERSION;check(rns8_get_plan_backend_info(selected.get(),&pi));require(pi.backend==RNS8_BACKEND_HIP_DIRECT,"plan backend changed");
  require(pi.workspace_required_bytes<=budget-modeled,"workspace exceeds aggregate payload cap");
  rns8_workspace* wr=nullptr;check(rns8_create_workspace(ctx.get(),selected.get(),&wr));Workspace work(wr);
  auto A=matrix(ctx.get(),c.m,c.k,true),B=matrix(ctx.get(),c.k,c.n,true),C=matrix(ctx.get(),c.m,c.n,false);
  check(rns8_pack_i64(ctx.get(),A.get(),av.data(),int64_t(c.k),1));check(rns8_pack_i64(ctx.get(),B.get(),bv.data(),int64_t(c.n),1));
  const auto sa=storage(A.get()),sb=storage(B.get()),sc=storage(C.get());const auto resident=payload(sa)+payload(sb)+payload(sc);
  require(resident<=64*ab+64*mn,"resident storage exceeds conservative bound");
  const std::array<const rns8_matrix*,3> owned{{A.get(),B.get(),C.get()}};
  const auto allocation_setup=allocations(owned,modeled);
  const auto native_setup_ns=ns(begin);
  auto prepare=[&](const rns8_matrix* source,rns8_block_sparse_role role){rns8_block_sparse_operand* p=nullptr;check(rns8_prepare_block_sparse_operand(ctx.get(),source,role,budget,&p));return Sparse(p);};
  Sparse warm_a,warm_b;json once=nullptr;
  if(!dense(c)&&!cold(c)){const auto t=Clock::now();warm_a=prepare(A.get(),RNS8_BLOCK_SPARSE_LHS);warm_b=prepare(B.get(),RNS8_BLOCK_SPARSE_RHS);once=ns(t);}
  const auto setup_ns=ns(begin);
  const auto method=hybrid(c)?RNS8_BLOCK_SPARSE_HYBRID:c.mode.find("wmma")!=std::string::npos?RNS8_BLOCK_SPARSE_WMMA:RNS8_BLOCK_SPARSE_SCALAR;
  struct Batch {uint64_t preparation=0,destruction=0;};
  const auto invoke=[&](){Batch result;Sparse ca,cb;
    if(cold(c)){const auto t=Clock::now();ca=prepare(A.get(),RNS8_BLOCK_SPARSE_LHS);cb=prepare(B.get(),RNS8_BLOCK_SPARSE_RHS);result.preparation=ns(t);}
    for(uint32_t i=0;i<c.reuse;++i){rns8_status status;
      if(dense(c))status=rns8_gemm_rns(ctx.get(),selected.get(),A.get(),B.get(),C.get(),work.get());
      else status=rns8_gemm_block_sparse_exact_with_method(ctx.get(),selected.get(),cold(c)?ca.get():warm_a.get(),cold(c)?cb.get():warm_b.get(),C.get(),method,c.threshold,budget,work.get());
      if(status==RNS8_UNSUPPORTED_BACKEND)throw Unsupported(rns8_status_string(status));check(status);
    }
    if(cold(c)){const auto t=Clock::now();check(rns8_destroy_block_sparse_operand(cb.release()));check(rns8_destroy_block_sparse_operand(ca.release()));result.destruction=ns(t);}return result;
  };
  const auto audit=[&](){std::vector<uint64_t> actual(3*c.m*(c.n+3),canary);check(rns8_export_exact_wide_signed_limbs(ctx.get(),selected.get(),C.get(),actual.data(),int64_t(c.n+3),3));
    for(size_t r=0;r<c.m;++r)for(size_t col=0;col<c.n+3;++col)for(size_t limb=0;limb<3;++limb)require(actual[(r*(c.n+3)+col)*3+limb]==(col<c.n?expected[(r*c.n+col)*3+limb]:canary),"complete Boost output or padding audit failed");
    const auto a=storage(A.get()),b=storage(B.get()),out=storage(C.get());require(a.matrix_instance_id==sa.matrix_instance_id&&b.matrix_instance_id==sb.matrix_instance_id&&out.matrix_instance_id==sc.matrix_instance_id&&a.source_version==1&&b.source_version==1&&a.device_residues_current&&b.device_residues_current&&a.device_native_current&&b.device_native_current&&out.device_residues_current,"source identity/import version or residency changed");
  };
  for(uint32_t i=0;i<c.warmups;++i){invoke();audit();}json samples=json::array();std::vector<uint64_t> times;
  for(uint32_t i=0;i<c.repeats;++i){const auto t=Clock::now();const auto batch=invoke();const auto elapsed=ns(t);require(elapsed>0,"call below host timer resolution");audit();times.push_back(elapsed);samples.push_back({{"wall_ns",elapsed},{"preparation_ns",batch.preparation},{"destruction_ns",batch.destruction},{"exact_reference_passed",true}});}
  const auto sustained_begin=Clock::now();uint64_t calls=0,elapsed=0;do{invoke();++calls;elapsed=ns(sustained_begin);}while(calls<4096&&(calls<16||elapsed<250000000));audit();std::sort(times.begin(),times.end());const double median=times.size()%2?double(times[times.size()/2]):(double(times[times.size()/2-1])+double(times[times.size()/2]))/2;
  const auto allocation_final=allocations(owned,modeled);
  return {{"schema","rns8_block_sparse_microcase_v1"},{"promotion_eligible",false},{"run_id",c.run},{"backend_requested","hip-direct"},{"backend_actual","hip-direct"},{"device_requested",c.device},{"device_actual",device.device_id},{"device_name",device.name},{"target",device.gcn_arch},{"hip_runtime_version",device.hip_runtime_version},{"hip_driver_version",device.hip_driver_version},{"build_manifest",json::parse(RNS8_BUILD_MANIFEST_JSON)},
    {"mode",c.mode},{"m",c.m},{"n",c.n},{"k",c.k},{"prefix",20},{"semantics","exact_wide_signed"},{"pattern",c.pattern},{"reuse_count",c.reuse},{"hybrid_threshold",c.threshold},{"repeats",c.repeats},{"warmups",c.warmups},{"dataset_recipe",recipe},{"dataset_seed",3135},{"a_checksum",checksum(av)},{"b_checksum",checksum(bv)},{"output_checksum",checksum(expected)},{"source_instance_ids",std::vector<uint64_t>{sa.matrix_instance_id,sb.matrix_instance_id}},{"output_instance_id",sc.matrix_instance_id},{"source_versions",std::vector<uint64_t>{sa.source_version,sb.source_version}},{"content_generations",nullptr},{"content_generation_visibility","opaque_handle_checked_by_public_owner_api"},
    {"reference_method","boost_int256_full_product_with_cpp_int_row_edge_audits"},{"reference_operand_storage_bytes",reference_bytes},{"reference_fixed_operand_object_bytes",sizeof(int256_t)},{"reference_buffers_released_before_setup",true},{"reference_and_dataset_ns",oracle_ns},{"native_setup_ns",native_setup_ns},{"setup_ns",setup_ns},{"warm_prepare_ns",once},{"setup_scope","context_plan_workspace_matrices_native_and_RNS_pack_then_optional_warm_prepare"},
    {"timed_scope",dense(c)?"reuse_public_dense_GEMMs":cold(c)?"prepare_both_reuse_public_sparse_GEMMs_checked_destroy_both":"reuse_public_prepared_sparse_GEMMs"},{"timer","std_chrono_steady_clock_nanoseconds"},{"clock_period_numerator",Clock::period::num},{"clock_period_denominator",Clock::period::den},{"selected_dense_kernel",pi.selected_kernel},
    {"modeled_a_blocks",counts.a_blocks},{"modeled_b_blocks",counts.b_blocks},{"modeled_packed_payload_bytes",counts.payload},{"modeled_directory_bytes",counts.metadata},{"actual_cache_allocation_bytes",nullptr},{"structure_origin","deterministic_exact_zero_native_recipe_not_device_query"},{"modeled_total_block_intersections",counts.intersections},{"modeled_empty_output_tiles",counts.empty_tiles},{"modeled_hybrid_scalar_tiles",hybrid(c)?json(counts.scalar_tiles):json(nullptr)},{"modeled_hybrid_wmma_tiles",hybrid(c)?json(counts.wmma_tiles):json(nullptr)},
    {"memory_budget_bytes",budget},{"logical_live_payload_envelope_bytes",std::max(oracle_peak,modeled+pi.workspace_required_bytes)},{"resident_payload_bytes",resident},{"resident_payload_scope","queried_after_input_packing_before_first_call"},
    {"matrix_allocations_at_setup",allocation_setup},{"matrix_allocations_after_final_audit",allocation_final},{"allocation_query_scope","A_B_C_requested_capacity_and_cached_staging_at_two_boundaries_not_peak_or_sparse_cache"},
    {"plan_workspace_bytes",pi.workspace_required_bytes},{"memory_exclusions","allocator/vector spare capacity, context runtime and fixed metadata; logical payload only, not process RSS"},
    {"exact_full_output_and_padding_checked",true},{"verified_outputs_per_batch",1},{"samples",samples},{"median_wall_ns",median},{"gemm_calls_per_second",1e9*double(c.reuse)/median},{"sustained",{{"calls",calls},{"gemm_calls",calls*c.reuse},{"elapsed_ns",elapsed},{"mean_batch_ns",double(elapsed)/double(calls)},{"mean_gemm_ns",double(elapsed)/double(calls*c.reuse)},{"minimum_calls",16},{"maximum_calls",4096},{"target_ns",250000000},{"stop_reason",elapsed>=250000000?"duration":"call_cap"},{"independent_audit_inside_interval",false},{"final_exact_reference_passed",true}}}};
}
}
int main(int argc,char** argv){if(argc==2&&std::string(argv[1])=="--help"){help();return 0;}try{const auto result=run(parse(argc,argv));require(!cleanup_failed,"public owner cleanup failed");std::cout<<result.dump()<<'\n';return 0;}catch(const Unsupported& error){std::cerr<<"unsupported explicit HIP method/device: "<<error.what()<<'\n';return 77;}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}
