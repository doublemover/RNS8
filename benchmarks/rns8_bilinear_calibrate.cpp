#include "call_observation.hpp"
#include "core/api_internal.hpp"
#include "backend_common/rank23_slp_resources.hpp"
#include "backend_common/bilinear_selected_identity.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
namespace {
using json=nlohmann::json;using Clock=std::chrono::steady_clock;using boost::multiprecision::cpp_int;
void check_status(rns8_status s){if(s!=RNS8_SUCCESS)throw std::runtime_error(rns8_status_string(s));}
double us(Clock::time_point a,Clock::time_point b){return std::chrono::duration<double,std::micro>(b-a).count();}
uint64_t number(const json& c,const char* key,uint64_t lo,uint64_t hi){if(!c.at(key).is_number_unsigned() && !c.at(key).is_number_integer())throw std::invalid_argument(key);const auto n=c.at(key).get<int64_t>();if(n<0 || uint64_t(n)<lo || uint64_t(n)>hi)throw std::invalid_argument(key);return uint64_t(n);}
struct Owner {
  rns8_context* ctx=nullptr;rns8_plan* plan=nullptr;rns8_workspace* work=nullptr;rns8_matrix *a=nullptr,*b=nullptr,*c=nullptr;
  rns8_status close(){rns8_status s;for(auto** x:{&c,&b,&a})if(*x){s=rns8_destroy_matrix(*x);if(s!=RNS8_SUCCESS)return s;*x=nullptr;}
    if(work){s=rns8_destroy_workspace(work);if(s!=RNS8_SUCCESS)return s;work=nullptr;}if(plan){s=rns8_destroy_plan(plan);if(s!=RNS8_SUCCESS)return s;plan=nullptr;}
    if(ctx){s=rns8_destroy_context(ctx);if(s!=RNS8_SUCCESS)return s;ctx=nullptr;}return RNS8_SUCCESS;}
  ~Owner(){(void)close();}
};
}
int main(int argc,char** argv){try{
  using namespace rns8::detail;
  if(argc!=3)throw std::invalid_argument("usage: rns8-bilinear-calibrate CASE.json unique-run-id");json c;std::ifstream(argv[1])>>c;
  if(c.size()!=16 || c.at("schema")!="bilinear_case_v1" || !c.at("unsigned_inputs").is_boolean() || !c.at("instrumented").is_boolean())throw std::invalid_argument("case ABI");
  const uint32_t m=number(c,"m",1,32768),n=number(c,"n",1,32768),k=number(c,"k",1,262149),prefix=number(c,"prefix",1,20),
      repeats=number(c,"repeats",3,100),warmups=number(c,"warmups",1,100),seed=number(c,"seed",0,UINT32_MAX);
  const uint64_t budget=number(c,"max_scratch_bytes",1,UINT64_C(1)<<40),host_budget=number(c,"max_host_bytes",1,UINT64_C(1)<<40);
  const uint32_t matrix_variant=number(c,"matrix_variant",0,2);
  const bool u=c.at("unsigned_inputs"),instrumented=c.at("instrumented");const std::string mode=c.at("mode"),backend=c.at("subproduct_backend"),residency=c.at("residency");
  const std::vector<std::string> modes{"classical","strassen","coefficient-rank23","september-slp","october55-retained","october55-recompute","rank48-dyadic","rank48-slp-retained","rank48-slp-recompute","october55-pruned","rank48-slp-pruned"};
  const auto found=std::find(modes.begin(),modes.end(),mode);if(found==modes.end() || (backend!="direct" && backend!="matrix") || (residency!="cold" && residency!="resident" && residency!="repack"))throw std::invalid_argument("route");
  const int algorithm=int(found-modes.begin())-1;const bool matrix=backend=="matrix";if(!matrix && matrix_variant)throw std::invalid_argument("matrix variant on direct route");
  const auto method=static_cast<rns8_bilinear_method>(algorithm>=6?12+2*(algorithm-6)+int(matrix):algorithm==5?10+int(matrix):algorithm+(matrix?5:0));
  const auto selected=bilinear_selection(uint32_t(method));const bool rank48=algorithm>=0 && selected.blocks==4;
  cpp_int capacity=1;for(uint32_t p=0;p<prefix;++p)capacity*=rns8_default_modulus(p);
  if(capacity<=uint64_t(k)*49*(u?1:2))throw std::invalid_argument("strict CRT capacity");
  BilinearResources resources{};
  if(algorithm>=0)resources=query_bilinear_method_resources(uint32_t(method),m,n,k,k,n,n,prefix,budget);
  else {resources.status=qualified_i32_gemm_address(m,n,k,k,n,n,prefix)?RNS8_SUCCESS:RNS8_UNSUPPORTED_BACKEND;}
  check_status(resources.status);
  if(matrix && (algorithm<0?k:resources.block_k)>65536)throw std::invalid_argument("matrix INT32 K cap");
  const uint64_t residue_payload=uint64_t(prefix)*(uint64_t(m)*k+uint64_t(k)*n+uint64_t(m)*n);
  const uint64_t host_payload=8*(uint64_t(m)*(k+1)+uint64_t(k)*(n+2)+uint64_t(m)*n)+24*uint64_t(m)*(n+3)+residue_payload+uint64_t(prefix)*m*n;
  if(host_payload>host_budget)throw std::invalid_argument("declared complete host payload budget");
  auto value=[&](bool lhs,uint32_t outer,uint32_t q)->int64_t{return int64_t((uint64_t(outer)*(lhs?71:11)+uint64_t(q%15)*(lhs?19:37)+seed)%(u?8:15))-(u?0:7);};
  std::vector<uint64_t> av(size_t(m)*(k+1)),bv(size_t(k)*(n+2)),output(size_t(m)*(n+3)*3,UINT64_C(0x5a5a5a5a5a5a5a5a));
  for(uint32_t r=0;r<m;++r)for(uint32_t q=0;q<k;++q)av[size_t(r)*(k+1)+q]=uint64_t(value(true,r,q));
  for(uint32_t q=0;q<k;++q)for(uint32_t col=0;col<n;++col)bv[size_t(q)*(n+2)+col]=uint64_t(value(false,col,q));
  // Exact periodic oracle, independent of block decomposition: every q mod 15
  // occurs floor(K/15) or ceil(K/15) times. O(15*M*N), including large cohorts.
  std::vector<int64_t> reference(size_t(m)*n);
  for(uint32_t r=0;r<m;++r)for(uint32_t col=0;col<n;++col){cpp_int sum=0;for(uint32_t q=0;q<15;++q)sum+=cpp_int(value(true,r,q))*value(false,col,q)*(k/15+(q<k%15));reference[size_t(r)*n+col]=sum.convert_to<int64_t>();}
  hip_direct_timing_set_enabled(instrumented);uint64_t generation=0,ordinary_workspace=0;std::string target,device,plan_kernel;int runtime=0;
  auto pack=[&](Owner& o){++generation;if(u){check_status(rns8_pack_u64(o.ctx,o.a,av.data(),k+1,generation));check_status(rns8_pack_u64(o.ctx,o.b,bv.data(),n+2,generation));}
    else{check_status(rns8_pack_i64(o.ctx,o.a,reinterpret_cast<const int64_t*>(av.data()),k+1,generation));check_status(rns8_pack_i64(o.ctx,o.b,reinterpret_cast<const int64_t*>(bv.data()),n+2,generation));}};
  auto create=[&](){auto o=std::make_unique<Owner>();rns8_context_options options{};options.struct_size=sizeof(options);options.abi_version=RNS8_ABI_VERSION;options.requested_backend=matrix && algorithm<0?RNS8_BACKEND_AMDGPU_BUILTINS:RNS8_BACKEND_HIP_DIRECT;check_status(rns8_create_context(0,&options,&o->ctx));
    rns8_gemm_desc d{};d.struct_size=sizeof(d);d.abi_version=RNS8_ABI_VERSION;d.semantics=u?RNS8_EXACT_WIDE_UNSIGNED:RNS8_EXACT_WIDE_SIGNED;d.bound_kind=RNS8_BOUND_NONE;d.requested_backend=options.requested_backend;d.m=m;d.n=n;d.k=k;d.max_prefix=prefix;d.flags=RNS8_PLAN_FORCE_FIXED_PREFIX|RNS8_PLAN_EXACT_VALUE_BOUND;
    check_status(rns8_create_plan(o->ctx,&d,&o->plan));check_status(rns8_create_workspace(o->ctx,o->plan,&o->work));ordinary_workspace=o->work->hip_workspace_arena_bytes;
    auto make=[&](uint32_t rows,uint32_t cols,rns8_matrix** out){rns8_matrix_desc md{};md.struct_size=sizeof(md);md.abi_version=RNS8_ABI_VERSION;md.semantics=d.semantics;md.bound_kind=RNS8_BOUND_NONE;md.logical_layout=RNS8_LAYOUT_ROW_MAJOR;md.rows=rows;md.cols=cols;md.logical_ld=cols;md.max_prefix=prefix;check_status(rns8_create_matrix(o->ctx,&md,out));};
    make(m,k,&o->a);make(k,n,&o->b);make(m,n,&o->c);pack(*o);
    if(algorithm>=0)check_status(rns8_prepare_bilinear_workspace(o->ctx,o->plan,method,budget,o->work));
    rns8_device_info info{};info.struct_size=sizeof(info);info.abi_version=RNS8_ABI_VERSION;check_status(rns8_get_device_info(o->ctx,&info));target=info.gcn_arch;device=info.name;runtime=info.hip_runtime_version;
    rns8_plan_backend_info pi{};pi.struct_size=sizeof(pi);pi.abi_version=RNS8_ABI_VERSION;check_status(rns8_get_plan_backend_info(o->plan,&pi));plan_kernel=pi.selected_kernel;
    // Same explicit dense matrix family for full and subproduct shape controls.
    if(matrix)amdgpu_builtins_set_dense_kernel_variant_override(static_cast<amdgpu_builtins_dense_kernel_variant>(matrix_variant));
    return o;};
  const auto start_setup=Clock::now();auto owner=create();const double setup=us(start_setup,Clock::now());if(residency=="cold"){check_status(owner->close());owner.reset();}
  auto product=[&](Owner& o){if(algorithm<0)check_status(rns8_gemm_rns(o.ctx,o.plan,o.a,o.b,o.c,o.work));else{double ratio=-1;check_status(rns8_gemm_bilinear_exact(o.ctx,o.plan,o.a,o.b,o.c,method,budget,o.work,&ratio));}};
  auto call=[&](bool timed=true){const rns8::bench::CallClock<> clock(timed);if(instrumented)hip_direct_timing_reset();const auto start=clock.now();if(residency=="cold")owner=create();else if(residency=="repack")pack(*owner);const auto prepared=clock.now();product(*owner);const auto computed=clock.now();
    if(u)check_status(rns8_export_exact_wide_unsigned_limbs(owner->ctx,owner->plan,owner->c,output.data(),n+3,3));else check_status(rns8_export_exact_wide_signed_limbs(owner->ctx,owner->plan,owner->c,output.data(),n+3,3));const auto exported=clock.now();
    if(residency=="cold"){check_status(owner->close());owner.reset();}const auto end=clock.now();return rns8::bench::PhaseObservation<4>{{us(start,prepared),us(prepared,computed),us(computed,exported),us(exported,end)},us(start,end)};};
  const auto sample_json=[](const rns8::bench::PhaseObservation<4>& value){auto sample=rns8::bench::phase_observation_json(value,{"prepare_us","execute_us","export_us","cleanup_us"},"total_us");return sample;};
  auto verify=[&](){for(uint32_t r=0;r<m;++r)for(uint32_t col=0;col<n+3;++col)for(uint32_t limb=0;limb<3;++limb){const auto expected=col>=n?UINT64_C(0x5a5a5a5a5a5a5a5a):limb==0?uint64_t(reference[size_t(r)*n+col]):reference[size_t(r)*n+col]<0?UINT64_MAX:0;
      if(output[(size_t(r)*(n+3)+col)*3+limb]!=expected)throw std::runtime_error("complete integer/padding oracle mismatch");}};
  auto first=sample_json(call());verify();for(uint32_t i=0;i<warmups;++i)call(false);json samples=json::array();for(uint32_t i=0;i<repeats;++i){samples.push_back(sample_json(call()));verify();}
  json sustained=json::array();for(uint32_t i=0;i<3;++i){const auto interval=rns8::bench::measure_completed_calls([&](){call(false);});const auto count=interval.calls;const double elapsed=interval.duration_us;verify();sustained.push_back({{"calls",count},{"total_us",elapsed},{"per_call_us",elapsed/count},{"minimum_duration_reached",elapsed>=250000}});}
  json events=json::array();if(instrumented){hip_direct_timing_flush_pending_events();for(const auto& e:hip_direct_timing_snapshot())events.push_back({{"label",e.label},{"microseconds",e.microseconds}});}hip_direct_timing_set_enabled(false);
  if(!owner){owner=create();hip_direct_timing_set_enabled(false);product(*owner);}std::vector<int8_t> residues(size_t(prefix)*m*n);check_status(hip_direct_copy_device_to_host(0,residues.data(),owner->c->hip_residues,residues.size()));
  for(uint32_t p=0;p<prefix;++p)for(size_t cell=0;cell<size_t(m)*n;++cell){const int modulus=rns8_default_modulus(p);int64_t expected=reference[cell]%modulus;if(expected<0)expected+=modulus;if(expected>=(modulus+1)/2)expected-=modulus;if(residues[size_t(p)*m*n+cell]!=expected)throw std::runtime_error("all-plane residue audit mismatch");}
  check_status(owner->close());const uint64_t export_stage=uint64_t(m)*n*24,known_peak=residue_payload+ordinary_workspace+resources.total_bytes+export_stage;
  std::vector<uint32_t> algebraic_planes,classical_planes,algebraic_moduli;
  if(algorithm>=0) {
    for(uint32_t compact=0;compact<resources.planes.eligible_count;++compact){algebraic_planes.push_back(resources.planes.eligible[compact]);algebraic_moduli.push_back(resources.planes.moduli[compact]);}
    for(uint32_t index=0;index<resources.planes.fallback_count;++index)classical_planes.push_back(resources.planes.fallback[index]);
  } else for(uint32_t plane=0;plane<prefix;++plane)classical_planes.push_back(plane);
  const uint32_t algebraic_products=algorithm<0?0:rank48?48:algorithm==0?7:23;
  json pruned_schedule=nullptr;
  if(selected.pruned) {
    const auto work=selected_bilinear_work(rank48);
    pruned_schedule={{"lowering","immutable_closure_before_slots_v1"},{"sha256",selected_bilinear_schedule_sha256(rank48)},
      {"binary",{work.binary[0],work.binary[1]}},{"halving",{work.halving[0],work.halving[1]}},{"input_loads",{work.input_loads[0],work.input_loads[1]}}};
  }
  std::cout<<json({{"hot_loop_serialization","typed_observation_no_phase_clocks_in_sustained_v2"},{"schema","rns8_bilinear_microcase_v3"},{"case",c},{"run_id",argv[2]},{"target",target},{"device",device},{"runtime_version",runtime},{"build_manifest",json::parse(RNS8_BUILD_MANIFEST_JSON)},
    {"completion_contract",algorithm<0?"ordinary_public_gemm_v1":"ordered_nondefault_single_terminal_v1"},{"dataset","period15_native_small_v1"},{"selected_prefix",prefix},{"algorithm",mode},{"subproduct_backend",backend},{"matrix_family_override",matrix?(matrix_variant==0?"auto":matrix_variant==1?"cdna3_16x16x32_or_target_wmma":"cdna3_32x32x16_or_target_wmma"):"not_applicable"},{"subproduct_launch_schedule",matrix && algorithm==6?"batched48":matrix && (algorithm==2 || algorithm==3)?"batched23":"serial"},{"ordinary_plan_kernel",plan_kernel},
    {"required_bilinear_scratch_bytes",resources.total_bytes},{"resident_residue_payload_bytes",residue_payload},{"ordinary_workspace_bytes",ordinary_workspace},{"known_peak_device_payload_bytes",known_peak},{"host_payload_envelope_bytes",host_payload},
    {"memory_scope","logical_live_payload_excludes_driver_allocator_and_opaque_runtime_overhead"},{"transform_binary_nodes",rank48 && selected.slp?200:algorithm==2?56:algorithm==3 || algorithm==4 || algorithm==8?55:0},{"transform_halving_nodes",rank48 && selected.slp?4:0},{"input_transform_evaluations_per_cell",algorithm==5 || (rank48 && selected.recompute)?48:algorithm==4 || algorithm==8?23:1},
    {"dyadic_channel_policy",rank48?"odd_rank48_even_classical_compact_v2":"not_applicable"},
    {"pruned_transform_schedule",pruned_schedule},{"channel_contract_version","indexed_modulus_planes_v1"},{"algebraic_plane_ids",algebraic_planes},{"classical_plane_ids",classical_planes},
    {"algebraic_moduli",algebraic_moduli},{"algebraic_product_plane_gemms",uint64_t(algebraic_products)*algebraic_planes.size()},
    {"output_boundary","complete_public_product_and_padded_three_limb_export"},{"host_instrumentation_state",instrumented?"diagnostic_hooks_enabled":"hooks_disabled"},{"diagnostic_events",events},{"setup_us",setup},{"first_call",first},{"samples",samples},{"sustained",sustained},
    {"exact_reference_passed",true},{"all_residue_planes_audited",true},{"residue_audit_scope","post_timing_public_product"},{"oracle_inside_production_interval",false},{"promotion_eligible",false},{"native_useful_products",uint64_t(m)*n*k}}).dump(2)<<'\n';return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
