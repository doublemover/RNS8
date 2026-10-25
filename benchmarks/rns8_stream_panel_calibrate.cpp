// Public serial panel owner versus ordinary in-core host-to-host GEMM.
#include <rns8/rns8.h>
#include <rns8/moduli.h>
#include <boost/multiprecision/cpp_int.hpp>
#include <nlohmann/json.hpp>
#include <algorithm>
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
#include <tuple>
#include <vector>
namespace {
using json=nlohmann::json;
using boost::multiprecision::cpp_int;
using boost::multiprecision::int256_t;
using Clock=std::chrono::steady_clock;
constexpr uint64_t budget=512ull*1024*1024,canary=UINT64_C(0xa5a5a5a5a5a5a5a5);
static_assert(sizeof(int256_t)<=128,"reference object exceeds admitted allowance");
bool cleanup_failed=false;
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
void check(rns8_status s){if(s!=RNS8_SUCCESS)throw std::runtime_error(rns8_status_string(s));}
struct Unsupported:std::runtime_error{using std::runtime_error::runtime_error;};
template<class T,rns8_status(*Destroy)(T*)>struct Delete{void operator()(T* p)const noexcept{if(p&&Destroy(p)!=RNS8_SUCCESS)cleanup_failed=true;}};
template<class T,rns8_status(*Destroy)(T*)>using Owner=std::unique_ptr<T,Delete<T,Destroy>>;
using Context=Owner<rns8_context,rns8_destroy_context>;
using Plan=Owner<rns8_plan,rns8_destroy_plan>;
using Matrix=Owner<rns8_matrix,rns8_destroy_matrix>;
using Workspace=Owner<rns8_workspace,rns8_destroy_workspace>;
using Stream=Owner<rns8_stream_gemm_plan,rns8_destroy_stream_gemm>;
uint64_t ns(Clock::time_point start){return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-start).count());}
uint64_t number(const char* s,uint64_t low,uint64_t high){uint64_t value=0;const auto end=s+std::strlen(s);const auto r=std::from_chars(s,end,value);require(r.ec==std::errc{}&&r.ptr==end&&value>=low&&value<=high,"numeric argument outside supported domain");return value;}
struct Case{int device;std::string mode,domain,run;size_t m,n,k;uint32_t bits,prefix,pm,pn,pk,reuse,repeats,warmups;uint64_t db,hb,cb;};
void help(){std::cout<<"rns8-stream-panel-calibrate DEVICE incore|stream M N K signed|unsigned BITS PREFIX PANEL_M PANEL_N PANEL_K DEVICE_BYTES HOST_BYTES REUSE REPEATS WARMUPS RUN_ID RETAINED_B_BYTES\n"
 "Explicit direct-HIP DEVICE0..63; M,N1..512 K1..65537 BITS1..31 PREFIX1..20.\n"
 "Incore requires all panel and budget arguments zero. Stream requires positive panel dimensions and explicit byte budgets; retained B is an optional budget allowance.\n"
 "K*(2^BITS-1)^2 must fit selected native output and CRT prefix; total batch work<=134217728.\n"
 "REUSE1..256 REPEATS/WARMUPS1..1000. Both modes time native host input through checked native host output.\n"
 "Two generations reuse the same owner; separate destroy/create/first-call evidence reports recreation.\n"
 "Full Boost output and padding audits are outside timers; sustained calls target250ms with16..4096 batches.\n"
 "Only queried payload/staging capacities and observed serial coverage; no peak, bus-traffic or overlap claim.\n";}
Case parse(int argc,char** argv){require(argc==19,"expected eighteen positional arguments; use --help");Case c{};
 c.device=int(number(argv[1],0,63));c.mode=argv[2];c.m=size_t(number(argv[3],1,512));c.n=size_t(number(argv[4],1,512));c.k=size_t(number(argv[5],1,65537));c.domain=argv[6];
 c.bits=uint32_t(number(argv[7],1,31));c.prefix=uint32_t(number(argv[8],1,20));c.pm=uint32_t(number(argv[9],0,65537));c.pn=uint32_t(number(argv[10],0,65537));c.pk=uint32_t(number(argv[11],0,65537));
 c.db=number(argv[12],0,budget);c.hb=number(argv[13],0,budget);c.reuse=uint32_t(number(argv[14],1,256));c.repeats=uint32_t(number(argv[15],1,1000));c.warmups=uint32_t(number(argv[16],1,1000));c.run=argv[17];c.cb=number(argv[18],0,budget);
 require(c.mode=="incore"||c.mode=="stream","mode must be incore or stream");require(c.domain=="signed"||c.domain=="unsigned","signedness must be explicit");
 require(c.mode=="incore"?!(c.pm||c.pn||c.pk||c.db||c.hb||c.cb):c.pm&&c.pn&&c.pk&&c.db&&c.hb,"irrelevant or missing panel/budget configuration");
 require(uint64_t(c.m)*c.n*c.k*c.reuse<=134217728,"reference/batch work exceeds bounded domain");
 require(!c.run.empty()&&c.run.size()<=128&&c.run.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")==std::string::npos,"invalid run ID");return c;}
uint64_t next(uint64_t& state){uint64_t z=(state+=UINT64_C(0x9e3779b97f4a7c15));z=(z^(z>>30))*UINT64_C(0xbf58476d1ce4e5b9);z=(z^(z>>27))*UINT64_C(0x94d049bb133111eb);return z^(z>>31);}
int64_t signed_value(uint64_t x){int64_t v;std::memcpy(&v,&x,8);return v;}
std::string checksum(const std::vector<uint64_t>& v,size_t rows,size_t cols,size_t ld){uint64_t h=UINT64_C(14695981039346656037);for(size_t r=0;r<rows;++r)for(size_t c=0;c<cols;++c)for(unsigned b=0;b<8;++b)h=(h^uint8_t(v[r*ld+c]>>(8*b)))*UINT64_C(1099511628211);std::ostringstream s;s<<std::hex<<std::setfill('0')<<std::setw(16)<<h;return s.str();}
json capacities(const rns8_stream_gemm_resources& r){return {{"device_bytes",r.device_bytes},{"host_bytes",r.host_bytes},{"minimum_device_bytes",r.minimum_device_bytes},{"minimum_host_bytes",r.minimum_host_bytes},{"source_a_span_bytes",r.source_a_span_bytes},{"source_b_span_bytes",r.source_b_span_bytes},{"destination_span_bytes",r.destination_span_bytes},{"output_tiles",r.output_tiles},{"k_panels_per_tile",r.k_panels_per_tile},{"retained_b_entries",r.retained_b_entries},{"retained_b_device_bytes",r.retained_b_device_bytes},{"cold_b_panel_loads",r.cold_b_panel_loads},{"warm_b_panel_loads",r.warm_b_panel_loads}};}
json allocation(const rns8_matrix* p){rns8_matrix_allocation_info a{};a.struct_size=sizeof(a);a.abi_version=RNS8_ABI_VERSION;check(rns8_get_matrix_allocation_info(p,&a));return {{"logical_primary_bytes",a.logical_primary_bytes},{"host_payload_capacity_bytes",a.host_payload_capacity_bytes},{"device_payload_capacity_bytes",a.device_payload_capacity_bytes},{"device_upload_staging_bytes",a.device_upload_staging_bytes},{"device_export_staging_bytes",a.device_export_staging_bytes},{"device_status_and_schedule_bytes",a.device_status_and_schedule_bytes},{"device_total_capacity_bytes",a.device_total_capacity_bytes}};}
json run(const Case& c){const bool panel=c.mode=="stream",sign=c.domain=="signed";const auto semantics=sign?RNS8_BOUNDED_I64:RNS8_BOUNDED_U64;
 const uint64_t maximum=(UINT64_C(1)<<c.bits)-1;const cpp_int bound_big=cpp_int(c.k)*maximum*maximum;
 require(bound_big<=(sign?cpp_int(INT64_MAX):cpp_int(UINT64_MAX)),"full-expression bound exceeds selected native output");
 cpp_int product=1;for(uint32_t p=0;p<c.prefix;++p)product*=rns8_default_modulus(p);
 const cpp_int required=sign?cpp_int(2*bound_big):bound_big;require(required<product,"CRT prefix does not admit full-expression bound");const auto bound=bound_big.convert_to<uint64_t>();
 const size_t lda=c.k+3,ldb=c.n+3,ldc=c.n+3;const uint64_t ab=uint64_t(c.m)*c.k+uint64_t(c.k)*c.n,mn=uint64_t(c.m)*c.n;
 const uint64_t caller_bytes=8*(uint64_t(c.m)*lda+uint64_t(c.k)*ldb+uint64_t(c.m)*ldc),host_live=caller_bytes+8*mn;
 rns8_stream_gemm_desc sd{};sd.struct_size=sizeof(sd);sd.abi_version=RNS8_ABI_VERSION;sd.semantics=semantics;sd.m=c.m;sd.n=c.n;sd.k=c.k;sd.lda=lda;sd.ldb=ldb;sd.ldc=ldc;sd.panel_m=c.pm;sd.panel_n=c.pn;sd.panel_k=c.pk;sd.prefix=c.prefix;sd.max_abs_a=sd.max_abs_b=maximum;sd.device_budget_bytes=c.db;sd.host_budget_bytes=c.hb;sd.retained_b_budget_bytes=c.cb;
 rns8_stream_gemm_resources sr{};sr.struct_size=sizeof(sr);sr.abi_version=RNS8_ABI_VERSION;uint64_t query_ns=0;
 if(panel){const auto t=Clock::now();check(rns8_query_stream_gemm(&sd,&sr));query_ns=ns(t);
  auto baseline=sd;baseline.retained_b_budget_bytes=0;auto base=sr;check(rns8_query_stream_gemm(&baseline,&base));auto negative=baseline;auto r=base;negative.device_budget_bytes=base.device_bytes-1;require(rns8_query_stream_gemm(&negative,&r)==RNS8_WORKSPACE_TOO_SMALL&&capacities(r)==capacities(base),"device budget preflight did not preserve capacity receipt");
  negative=baseline;r=base;negative.host_budget_bytes=base.host_bytes-1;require(rns8_query_stream_gemm(&negative,&r)==RNS8_WORKSPACE_TOO_SMALL&&capacities(r)==capacities(base),"host budget preflight did not preserve capacity receipt");}
 const uint64_t base_envelope=512*(ab+mn)+host_live+16384+sr.device_bytes+sr.host_bytes;require(base_envelope<=budget,"aggregate logical payload exceeds512MiB");
 std::vector<uint64_t> a(c.m*lda,canary),b(c.k*ldb,canary),out(c.m*ldc,canary),gold(mn);
 uint64_t state=6811;for(auto tuple:{std::make_tuple(&a,c.m,c.k,lda),std::make_tuple(&b,c.k,c.n,ldb)}){auto& values=*std::get<0>(tuple);for(size_t r=0;r<std::get<1>(tuple);++r)for(size_t col=0;col<std::get<2>(tuple);++col){const auto z=next(state);const int64_t v=sign?int64_t(z%(2*maximum+1))-int64_t(maximum):int64_t(z%(maximum+1));values[r*std::get<3>(tuple)+col]=uint64_t(v);}}
 a[0]=b[0]=maximum;
 const auto oracle=[&](){std::vector<int256_t> aa(c.m*c.k),bt(c.k*c.n);const auto value=[&](uint64_t v){return sign?int256_t(signed_value(v)):int256_t(v);};
  for(size_t r=0;r<c.m;++r)for(size_t q=0;q<c.k;++q)aa[r*c.k+q]=value(a[r*lda+q]);for(size_t q=0;q<c.k;++q)for(size_t col=0;col<c.n;++col)bt[col*c.k+q]=value(b[q*ldb+col]);
  for(size_t r=0;r<c.m;++r)for(size_t col=0;col<c.n;++col){int256_t sum=0;for(size_t q=0;q<c.k;++q)sum+=aa[r*c.k+q]*bt[col*c.k+q];require(boost::multiprecision::abs(sum)<=bound,"oracle exceeds declared complete-expression bound");
   if(col==0||col+1==c.n){cpp_int audit=0;for(size_t q=0;q<c.k;++q)audit+=cpp_int(value(a[r*lda+q]))*cpp_int(value(b[q*ldb+col]));require(cpp_int(sum)==audit,"cpp_int row-edge audit mismatch");}
   gold[r*c.n+col]=sum<0?uint64_t(sum.convert_to<int64_t>()):sum.convert_to<uint64_t>();}};
 const auto setup_start=Clock::now();rns8_context_options options{};options.struct_size=sizeof(options);options.abi_version=RNS8_ABI_VERSION;options.requested_backend=RNS8_BACKEND_HIP_DIRECT;rns8_context* cp=nullptr;
 const auto status=rns8_create_context(c.device,&options,&cp);if(status==RNS8_UNSUPPORTED_BACKEND||status==RNS8_UNSUPPORTED_ARCH||status==RNS8_UNSUPPORTED_OS)throw Unsupported(rns8_status_string(status));check(status);Context ctx(cp);
 rns8_device_info device{};device.struct_size=sizeof(device);device.abi_version=RNS8_ABI_VERSION;check(rns8_get_device_info(ctx.get(),&device));require(device.backend==RNS8_BACKEND_HIP_DIRECT&&device.device_id==c.device&&device.gcn_arch[0],"explicit HIP identity unavailable");
 Plan plan;Workspace workspace;Matrix A,B,C;Stream owner;uint64_t workspace_bytes=0,prepare_ns=0;std::string kernel="serial_panel_public_rns_axpby";
 const auto create_owner=[&](){rns8_stream_gemm_plan* raw=nullptr;check(rns8_create_stream_gemm(ctx.get(),&sd,&raw));owner.reset(raw);};
 const auto destroy_owner=[&](){if(owner){check(rns8_destroy_stream_gemm(owner.get()));owner.release();}};
 if(panel){const auto t=Clock::now();create_owner();prepare_ns=ns(t);}else{rns8_gemm_desc d{};d.struct_size=sizeof(d);d.abi_version=RNS8_ABI_VERSION;d.semantics=semantics;d.bound_kind=sign?RNS8_BOUND_GLOBAL_MAX_ABS:RNS8_BOUND_GLOBAL_MAX_UNSIGNED;d.requested_backend=RNS8_BACKEND_HIP_DIRECT;d.m=c.m;d.n=c.n;d.k=c.k;d.bound=bound;d.max_prefix=c.prefix;d.flags=RNS8_PLAN_FORCE_FIXED_PREFIX;
  rns8_plan* raw=nullptr;check(rns8_create_plan(ctx.get(),&d,&raw));plan.reset(raw);rns8_plan_backend_info info{};info.struct_size=sizeof(info);info.abi_version=RNS8_ABI_VERSION;check(rns8_get_plan_backend_info(plan.get(),&info));require(info.backend==RNS8_BACKEND_HIP_DIRECT,"in-core backend changed");workspace_bytes=info.workspace_required_bytes;kernel=info.selected_kernel;
  require(workspace_bytes<=budget-base_envelope,"workspace exceeds aggregate payload admission");rns8_workspace* wr=nullptr;check(rns8_create_workspace(ctx.get(),plan.get(),&wr));workspace.reset(wr);
  const auto matrix=[&](size_t rows,size_t cols){rns8_matrix_desc md{};md.struct_size=sizeof(md);md.abi_version=RNS8_ABI_VERSION;md.rows=rows;md.cols=cols;md.logical_ld=cols;md.logical_layout=RNS8_LAYOUT_ROW_MAJOR;md.semantics=semantics;md.bound_kind=d.bound_kind;md.max_prefix=c.prefix;rns8_matrix* p=nullptr;check(rns8_create_matrix(ctx.get(),&md,&p));return Matrix(p);};A=matrix(c.m,c.k);B=matrix(c.k,c.n);C=matrix(c.m,c.n);}
 const uint64_t setup_ns=ns(setup_start),envelope=base_envelope+workspace_bytes;uint64_t calls=0,owner_epoch=0,generation=1;
 rns8_stream_gemm_result last{};last.struct_size=sizeof(last);last.abi_version=RNS8_ABI_VERSION;
 const auto invoke=[&](){for(uint32_t i=0;i<c.reuse;++i){if(panel){check(rns8_execute_stream_gemm(owner.get(),a.data(),b.data(),out.data(),101,generation,202,1,&last));++owner_epoch;}else{
   // Version0 forces conversion of the supplied host data on every call.
   if(sign){check(rns8_pack_i64(ctx.get(),A.get(),reinterpret_cast<const int64_t*>(a.data()),lda,0));check(rns8_pack_i64(ctx.get(),B.get(),reinterpret_cast<const int64_t*>(b.data()),ldb,0));}else{check(rns8_pack_u64(ctx.get(),A.get(),a.data(),lda,0));check(rns8_pack_u64(ctx.get(),B.get(),b.data(),ldb,0));}
   check(rns8_gemm_rns(ctx.get(),plan.get(),A.get(),B.get(),C.get(),workspace.get()));check(sign?rns8_export_i64(ctx.get(),plan.get(),C.get(),reinterpret_cast<int64_t*>(out.data()),ldc):rns8_export_u64(ctx.get(),plan.get(),C.get(),out.data(),ldc));}++calls;}};
 const auto coverage=[&]()->json{if(!panel)return nullptr;require(last.status==RNS8_SUCCESS&&last.epoch==owner_epoch&&last.completed_tiles==sr.output_tiles&&last.completed_panels==sr.output_tiles*sr.k_panels_per_tile&&last.output_started==1&&last.output_valid==1&&!last.quarantined,"public panel coverage/epoch incomplete");require(last.b_panel_loads==(last.epoch==1?sr.cold_b_panel_loads:sr.warm_b_panel_loads)&&last.b_panel_cache_hits+last.b_panel_loads==last.completed_panels,"cache load/hit receipt mismatch");return {{"b_panel_cache_hits",last.b_panel_cache_hits},{"b_panel_loads",last.b_panel_loads},{"epoch",last.epoch},{"completed_tiles",last.completed_tiles},{"completed_panels",last.completed_panels},{"output_started",true},{"output_valid",true},{"quarantined",false}};};
 const auto audit=[&](){for(size_t r=0;r<c.m;++r)for(size_t col=0;col<ldc;++col)require(out[r*ldc+col]==(col<c.n?gold[r*c.n+col]:canary),"complete Boost product or output padding mismatch");for(auto tuple:{std::make_tuple(&a,c.m,c.k,lda),std::make_tuple(&b,c.k,c.n,ldb)})for(size_t r=0;r<std::get<1>(tuple);++r)for(size_t col=std::get<2>(tuple);col<std::get<3>(tuple);++col)require((*std::get<0>(tuple))[r*std::get<3>(tuple)+col]==canary,"input padding modified");(void)coverage();};
 const auto allocations=[&](){json list=json::array();uint64_t total=host_live+workspace_bytes+sr.device_bytes+sr.host_bytes;if(!panel)for(auto* p:{A.get(),B.get(),C.get()}){auto value=allocation(p);total+=value["host_payload_capacity_bytes"].get<uint64_t>()+value["device_total_capacity_bytes"].get<uint64_t>();list.push_back(value);}require(total<=envelope,"queried retained payload exceeds logical envelope");return list;};
 json phases=json::array();for(generation=1;generation<=2;++generation){if(generation==2)a[0]=0;const auto ot=Clock::now();oracle();const auto oracle_ns=ns(ot);
  auto t=Clock::now();invoke();const auto first_ns=ns(t);audit();const auto first_coverage=coverage();
  for(uint32_t i=0;i<c.warmups;++i){invoke();audit();}json samples=json::array();std::vector<uint64_t> times;
  for(uint32_t i=0;i<c.repeats;++i){t=Clock::now();invoke();const auto elapsed=ns(t);require(elapsed>0,"host timer below resolution");audit();times.push_back(elapsed);samples.push_back({{"wall_ns",elapsed},{"exact_reference_passed",true},{"coverage",coverage()}});}
  t=Clock::now();uint64_t batches=0,elapsed=0;do{invoke();++batches;elapsed=ns(t);}while(batches<4096&&(batches<16||elapsed<250000000));audit();std::sort(times.begin(),times.end());const double median=times.size()%2?double(times[times.size()/2]):(double(times[times.size()/2-1])+double(times[times.size()/2]))/2;
  phases.push_back({{"generation",generation},{"source_binding",{101,generation,202,1}},{"binding_action",generation==1?"initial_inputs":"changed_A0_same_plan"},{"a_checksum",checksum(a,c.m,c.k,lda)},{"b_checksum",checksum(b,c.k,c.n,ldb)},{"output_checksum",checksum(gold,c.m,c.n,c.n)},{"oracle_ns",oracle_ns},{"first_batch_ns",first_ns},{"first_coverage",first_coverage},{"first_exact_reference_passed",true},{"samples",samples},{"median_batch_ns",median},{"gemm_calls_per_second",1e9*c.reuse/median},{"matrix_allocations_after_audit",allocations()},
   {"sustained",{{"batches",batches},{"gemm_calls",batches*c.reuse},{"elapsed_ns",elapsed},{"mean_batch_ns",double(elapsed)/batches},{"minimum_batches",16},{"maximum_batches",4096},{"target_ns",250000000},{"stop_reason",elapsed>=250000000?"duration":"call_cap"},{"independent_audit_inside_interval",false},{"final_exact_reference_passed",true},{"coverage",coverage()}}}});}
 generation=2;json rejection=nullptr,recreation=nullptr;
 if(panel){const auto before=out;const auto previous=last;a[0]=maximum+1;const auto rejected=rns8_execute_stream_gemm(owner.get(),a.data(),b.data(),out.data(),101,3,202,1,&last);a[0]=0;
  require(rejected==RNS8_RANGE_ERROR&&out==before&&last.epoch==previous.epoch,"out-of-bound preflight modified output or epoch");rejection={{"status","RNS8_RANGE_ERROR"},{"output_preserved",true},{"epoch_preserved",true}};
  const auto t=Clock::now();destroy_owner();create_owner();const auto recreate_ns=ns(t);owner_epoch=0;const auto first=Clock::now();invoke();const auto first_ns=ns(first);audit();recreation={{"destroy_create_ns",recreate_ns},{"first_batch_ns",first_ns},{"setup_plus_first_batch_ns",recreate_ns+first_ns},{"coverage",coverage()},{"exact_reference_passed",true}};}
 const auto at_end=allocations();const auto release_start=Clock::now();destroy_owner();const auto release_ns=panel?ns(release_start):0;
 return {{"schema","rns8_stream_panel_microcase_v2"},{"promotion_eligible",false},{"run_id",c.run},{"device",c.device},{"device_actual",device.device_id},{"backend_requested","hip-direct"},{"backend_actual","hip-direct"},{"device_name",device.name},{"target",device.gcn_arch},{"hip_runtime_version",device.hip_runtime_version},{"hip_driver_version",device.hip_driver_version},{"build_manifest",json::parse(RNS8_BUILD_MANIFEST_JSON)},
  {"mode",c.mode},{"m",c.m},{"n",c.n},{"k",c.k},{"domain",c.domain},{"bits",c.bits},{"prefix",c.prefix},{"panel_m",c.pm},{"panel_n",c.pn},{"panel_k",c.pk},{"device_budget_bytes",c.db},{"host_budget_bytes",c.hb},{"retained_b_budget_bytes",c.cb},{"reuse",c.reuse},{"repeats",c.repeats},{"warmups",c.warmups},{"semantics",sign?"bounded_i64":"bounded_u64"},{"max_abs_a",maximum},{"max_abs_b",maximum},{"full_expression_bound",bound},{"lda",lda},{"ldb",ldb},{"ldc",ldc},
  {"dataset_recipe","splitmix64_seed6811_bounded_native_A0_B0_max_then_A0_zero_v1"},{"dataset_seed",6811},{"source_instance_labels",{101,202}},{"source_label_scope","dataset_labels_passed_to_stream_owner_only_not_matrix_instances_or_cache_hits"},{"incore_pack_source_version",panel?json(nullptr):json(0)},{"input_scan",panel?"public_owner_full_input_scan_each_call":"recipe_bound_independently_verified_before_timing"},
  {"reference_method","boost_int256_full_product_with_cpp_int_row_edge_audits"},{"reference_fixed_operand_object_bytes",sizeof(int256_t)},{"reference_operand_storage_bytes",ab*sizeof(int256_t)},{"reference_buffers_released_before_timing",true},{"output_contract","native64_word_per_cell_twos_complement_for_signed"},{"exact_full_output_and_padding_checked",true},
  {"timer","std_chrono_steady_clock_nanoseconds"},{"clock_period_numerator",Clock::period::num},{"clock_period_denominator",Clock::period::den},{"timed_scope","reuse_native_host_inputs_to_checked_native_host_output_no_independent_oracle"},{"setup_scope","query_then_context_and_owner_or_plan_workspace_matrices_no_input_conversion"},{"kernel_events",nullptr},{"production_timing_hooks_enabled",false},{"observed_overlap",nullptr},{"selected_control",kernel},
  {"query_ns",query_ns},{"initial_setup_ns",setup_ns},{"owner_prepare_ns",prepare_ns},{"cold_setup_plus_first_batch_ns",query_ns+setup_ns+phases[0]["first_batch_ns"].get<uint64_t>()},{"panel_resources",panel?capacities(sr):json(nullptr)},{"budget_preflight_receipts",panel?json({{"cache_disabled_for_preflight",true},{"baseline_device_minus_one","RNS8_WORKSPACE_TOO_SMALL"},{"baseline_host_minus_one","RNS8_WORKSPACE_TOO_SMALL"},{"queried_resources_preserved",true}}):json(nullptr)},
  {"caller_native_allocation_bytes",caller_bytes},{"reference_output_bytes",8*mn},{"memory_budget_bytes",budget},{"logical_live_payload_envelope_bytes",envelope},{"plan_workspace_bytes",workspace_bytes},{"matrix_allocations_after_final_audit",at_end},{"full_incore_residue_payload_bytes",(ab+mn)*c.prefix},{"capacity_scope","public_query_required_payload_and_staging_not_measured_peak_or_bus_traffic"},{"memory_exclusions","allocator_bookkeeping_spare_capacity_context_runtime_fixed_metadata_not_process_RSS_or_measured_peak"},
  {"production_native_calls",calls},{"out_of_bound_probe",rejection},{"recreation",recreation},{"owner_release_ns",release_ns},{"phases",phases}};
}
}
int main(int argc,char** argv){if(argc==2&&std::string(argv[1])=="--help"){help();return 0;}try{const auto result=run(parse(argc,argv));require(!cleanup_failed,"public handle cleanup failed");std::cout<<result.dump()<<'\n';return 0;}catch(const Unsupported& e){std::cerr<<"unsupported explicit HIP selection: "<<e.what()<<'\n';return 77;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
