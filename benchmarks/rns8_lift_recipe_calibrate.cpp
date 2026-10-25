// Exact output boundaries are identical except explicitly labeled eager-observe.
#include "core/api_internal.hpp"
#include "core/runtime_product_recipe.hpp"
#include "product_stage_prices.hpp"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
namespace {
using json=nlohmann::json;using Clock=std::chrono::steady_clock;using boost::multiprecision::cpp_int;
void check(rns8_status s){if(s!=RNS8_SUCCESS)throw std::runtime_error(rns8_status_string(s));}
double us(Clock::time_point a,Clock::time_point b){return std::chrono::duration<double,std::micro>(b-a).count();}
uint64_t number(const json& c,const char* key,uint64_t lo,uint64_t hi){if(!c.at(key).is_number_integer())throw std::invalid_argument(key);const auto n=c.at(key).get<int64_t>();if(n<0 || uint64_t(n)<lo || uint64_t(n)>hi)throw std::invalid_argument(key);return uint64_t(n);}
}
int main(int argc,char** argv){try {
  using namespace rns8::detail;using namespace rns8::detail::api;
  if(argc!=3 && argc!=4)throw std::invalid_argument("usage: rns8-lift-recipe-calibrate CASE.json unique-run-id [prices.json]");json c;std::ifstream(argv[1])>>c;
  if(c.size()!=12 || c.at("schema")!="lift_recipe_case_v1" || !c.at("unsigned_inputs").is_boolean())throw std::invalid_argument("case ABI");
  const unsigned side=number(c,"side",1,256),depth=number(c,"depth",1,12),prefix=number(c,"prefix",1,20),source=number(c,"source_prefix",1,prefix),
      repeats=number(c,"repeats",3,100),warmups=number(c,"warmups",1,100),model_launch=number(c,"model_launch_ns",0,1000000000);
  const uint64_t budget=number(c,"max_payload_bytes",64,UINT64_C(1)<<40);const bool u=c.at("unsigned_inputs");const std::string mode=c.at("mode"),shape=c.at("topology");
  const bool lift=mode.rfind("lift-",0)==0,recipe=mode.rfind("recipe-",0)==0,ragged=mode=="recipe-ragged-full" || mode=="recipe-ragged-minimum",eager=mode=="eager-observe",cpu_lift=mode=="lift-cpu-roundtrip";
  const bool native_roundtrip=mode=="native-roundtrip",scalar_first=mode=="manual-scalar-first";
  const bool measured=mode=="recipe-measured";
  if((argc==4)!=measured)throw std::invalid_argument("measured recipe requires exactly one frozen price file");
  const bool materialized=mode=="recipe-packed-full" || mode=="recipe-packed-minimum" || mode=="recipe-layout-model" || measured;
  if((shape!="chain" && shape!="diamond") || (mode!="recipe-ragged-full" && mode!="recipe-ragged-minimum" && mode!="recipe-full" && mode!="recipe-minimum" && mode!="recipe-cost-model" && mode!="manual-full" && mode!="manual-auto" && !native_roundtrip && !scalar_first && !eager && !materialized &&
      mode!="lift-wide" && mode!="lift-mixed" && mode!="lift-graph" && !cpu_lift && mode!="wider-producer") || (lift && shape!="chain"))throw std::invalid_argument("route/topology");
  const uint64_t cells=uint64_t(side)*side;const auto semantics=u?RNS8_EXACT_WIDE_UNSIGNED:RNS8_EXACT_WIDE_SIGNED;
  std::vector<ProductNode> nodes;if(shape=="chain"){for(unsigned i=0;i<depth;++i)nodes.push_back({i?i+1:0,1});}
  else{nodes={{0,1},{2,1},{0,2},{3,4}};if(depth!=4)throw std::invalid_argument("diamond depth is four");}
  std::vector<cpp_int> values{cpp_int(1),cpp_int(1)};for(auto node:nodes)values.push_back(cpp_int(side)*values[node.a]*values[node.b]);
  if(native_roundtrip) {
    const cpp_int native_min=u?cpp_int(0):cpp_int(-(cpp_int(1)<<63)),native_max=(cpp_int(1)<<(u?64:63))-1;
    for(size_t id=2;id+1<values.size();++id)
      if(values[id]<native_min || values[id]>native_max)throw std::invalid_argument("native roundtrip intermediate exceeds native64");
  }
  const uint64_t native_host_bytes=native_roundtrip?(nodes.size()+1)*cells*sizeof(uint64_t):0;
  uint64_t native_h2d=0,native_d2h=native_roundtrip?(nodes.size()-1)*cells*sizeof(uint64_t):0;
  if(native_roundtrip)for(auto node:nodes)native_h2d+=cells*sizeof(uint64_t)*(node.a==node.b?1:2);
  const cpp_int final_bound=(lift || mode=="wider-producer")?cpp_int(side):values.back();if(!exact_range_fits(semantics,final_bound,modulus_product(prefix)) ||
      (lift && !exact_range_fits(semantics,cpp_int(side),modulus_product(source))))throw std::invalid_argument("strict range");
  const uint64_t known_payload=(nodes.size()+3)*cells*prefix+cells*48*(nodes.size()+1)+cells*16+native_host_bytes;
  if(known_payload>=budget || cells*prefix>INT32_MAX || uint64_t(side)*side*side*nodes.size()>134217728)throw std::invalid_argument("payload/work");
  hip_direct_timing_set_enabled(false);
  rns8_context_options co{};co.struct_size=sizeof(co);co.abi_version=RNS8_ABI_VERSION;co.requested_backend=RNS8_BACKEND_HIP_DIRECT;
  rns8_context* ctx=nullptr;rns8_plan *fixed=nullptr,*adaptive=nullptr,*producer=nullptr;rns8_workspace *fixed_work=nullptr,*auto_work=nullptr,*producer_work=nullptr;
  std::vector<rns8_matrix*> matrices(nodes.size()+2);std::vector<uint64_t> ones(cells,1),output(size_t(side)*(side+3)*3,99);std::unique_ptr<RuntimeProductRecipe> owner;
  std::vector<std::vector<uint64_t>> native_values;
  uint64_t native_source_version=1;
  rns8_ragged_materialization* ragged_source=nullptr;rns8_matrix* ragged_converted=nullptr;rns8_product_recipe* ragged_recipe=nullptr;
  uint64_t ragged_retained=0,ragged_input_bytes=0,ragged_peak=0;
  rns8_prepared_execution* prepared_lift=nullptr;rns8_context* cpu=nullptr;rns8_matrix* host_matrix=nullptr;
  auto create_plan=[&](rns8_context* context,unsigned p,bool force,rns8_plan** out,rns8_workspace** work){rns8_gemm_desc d{};d.struct_size=sizeof(d);d.abi_version=RNS8_ABI_VERSION;d.semantics=semantics;d.bound_kind=RNS8_BOUND_NONE;d.requested_backend=context->backend;d.m=side;d.n=side;d.k=side;d.max_prefix=p;d.flags=force?(RNS8_PLAN_FORCE_FIXED_PREFIX|RNS8_PLAN_EXACT_VALUE_BOUND|(scalar_first?RNS8_PLAN_EXACT_SCALAR_FIRST:0)):0;check(rns8_create_plan(context,&d,out));if(work)check(rns8_create_workspace(context,*out,work));};
  auto matrix=[&](rns8_context* context,rns8_matrix** out){rns8_matrix_desc d{};d.struct_size=sizeof(d);d.abi_version=RNS8_ABI_VERSION;d.semantics=semantics;d.bound_kind=RNS8_BOUND_NONE;d.rows=side;d.cols=side;d.logical_ld=side;d.logical_layout=RNS8_LAYOUT_ROW_MAJOR;d.max_prefix=prefix;check(rns8_create_matrix(context,&d,out));};
  const auto setup_begin=Clock::now();check(rns8_create_context(0,&co,&ctx));create_plan(ctx,prefix,true,&fixed,&fixed_work);create_plan(ctx,prefix,false,&adaptive,&auto_work);create_plan(ctx,source,true,&producer,&producer_work);
  for(auto& m:matrices)matrix(ctx,&m);if(!native_roundtrip)for(unsigned i=0;i<2;++i)check(u?rns8_pack_u64(ctx,matrices[i],ones.data(),side,1):rns8_pack_i64(ctx,matrices[i],reinterpret_cast<const int64_t*>(ones.data()),side,1));
  if(native_roundtrip) {
    native_values.resize(nodes.size()+1);
    for(auto& value:native_values)value.resize(cells);
    std::fill(native_values[0].begin(),native_values[0].end(),1);
    std::fill(native_values[1].begin(),native_values[1].end(),1);
  }
  std::vector<const rns8_matrix*> inputs{matrices[0],matrices[1]};json selected=json::array();uint64_t arena_bytes=0,lift_bytes=0;json cost_model=nullptr,materialization_plan=nullptr,stage_price_identity=nullptr;
  if(ragged){matrix(ctx,&ragged_converted);check(rns8_prepare_ragged_materialization(ctx,matrices[0],3,5,budget-known_payload,&ragged_source));inputs[0]=ragged_converted;
    std::vector<rns8_product_node> public_nodes;for(auto node:nodes)public_nodes.push_back({node.a,node.b});rns8_product_recipe_desc descriptor{};descriptor.struct_size=sizeof(descriptor);descriptor.abi_version=RNS8_ABI_VERSION;
    descriptor.inputs=inputs.data();descriptor.input_count=inputs.size();descriptor.nodes=public_nodes.data();descriptor.node_count=public_nodes.size();descriptor.prefix_policy=mode=="recipe-ragged-full"?RNS8_PRODUCT_FULL_PREFIX:RNS8_PRODUCT_MINIMUM_COMPUTE_PREFIX;descriptor.max_arena_and_lift_bytes=budget-known_payload;
    rns8_ragged_recipe_binding binding{0,ragged_source,ragged_converted};check(rns8_create_product_recipe_ragged(ctx,&descriptor,&binding,1,nullptr,0,budget-known_payload,&ragged_recipe));
    rns8_product_recipe_resources resources{};resources.struct_size=sizeof(resources);resources.abi_version=RNS8_ABI_VERSION;check(rns8_get_product_recipe_resources(ragged_recipe,&resources));arena_bytes=resources.arena_bytes;lift_bytes=resources.lift_scratch_bytes;
    ragged_retained=resources.ragged_retained_bytes;ragged_input_bytes=resources.materialized_input_bytes;ragged_peak=resources.materialization_peak_bytes;
    std::vector<uint32_t> prefixes(nodes.size());check(rns8_get_product_recipe_prefixes(ragged_recipe,prefixes.data(),prefixes.size()));selected=prefixes;
  }
  else if(recipe){ProductCostProfile costs;ProductMaterializationProfile layouts;const ProductCostProfile* chosen=nullptr;const ProductMaterializationProfile* layout_prices=nullptr;
    if(mode=="recipe-cost-model" || materialized){costs.backend=ctx->backend;costs.semantics=semantics;costs.target=ctx->device_info.gcn_arch;costs.provenance="explicit-unqualified-analytical-work-model-v2";costs.graph_key=product_graph_key({{side,side},{side,side}},nodes,prefix);costs.nodes.resize(nodes.size());
      for(auto& row:costs.nodes)for(unsigned p=1;p<=prefix;++p){row.gemm[p]=model_launch+cells*side*p;for(unsigned q=p;q<=prefix;++q)row.lift[p*21+q]=q==p?0:3*uint64_t(model_launch)+cells*(uint64_t(p)*p+(q-p)*p);}
      chosen=&costs;cost_model={{"kind",costs.provenance},{"launch_ns",model_launch},{"gemm_units","one_ns_per_scalar_modular_multiply"},{"qualified",false}};}
    if(materialized) {
      layouts.uniform=costs;layouts.uniform.provenance="explicit-unqualified-layout-work-model-v2";layouts.packed.resize(nodes.size());
      for(size_t i=0;i<nodes.size();++i)for(unsigned p=1;p<=20;++p) {
        for(unsigned capacity=p;capacity<=20;++capacity) {
          layouts.packed[i].gemm[ProductPackedCosts::gemm_index(capacity,p)]=model_launch+cells*((side+3)/4)*p;
          for(unsigned old=0;old<p;++old)
            layouts.packed[i].pack[ProductPackedCosts::pack_index(capacity,old,p)]=model_launch+2*cells*(p-old);
        }
        if(mode!="recipe-layout-model" && !measured)layouts.uniform.nodes[i].gemm[p]=UINT64_MAX;
      }
      chosen=nullptr;layout_prices=&layouts;
      cost_model={{"kind",layouts.uniform.provenance},{"launch_ns",model_launch},{"gemm_units","one_ns_per_scalar_modular_multiply"},
          {"packed_gemm_units","one_ns_per_four_term_group"},{"pack_units","one_ns_per_logical_byte_read_or_write"},
          {"cost_schema_version",layouts.cost_schema_version},{"packed_cost_capacity","physical_shared_b_prefix"},{"qualified",false}};
      if(measured) {
        std::vector<rns8_product_node> public_nodes;for(auto node:nodes)public_nodes.push_back({node.a,node.b});
        rns8_product_recipe_desc descriptor{};descriptor.struct_size=sizeof(descriptor);descriptor.abi_version=RNS8_ABI_VERSION;
        descriptor.inputs=inputs.data();descriptor.input_count=inputs.size();descriptor.nodes=public_nodes.data();descriptor.node_count=public_nodes.size();
        descriptor.prefix_policy=RNS8_PRODUCT_MINIMUM_COMPUTE_PREFIX;descriptor.max_arena_and_lift_bytes=budget-known_payload;
        rns8_product_materialization_identity identity{};identity.struct_size=sizeof(identity);identity.abi_version=RNS8_ABI_VERSION;
        check(rns8_query_product_materialization_identity(ctx,&descriptor,nullptr,0,&identity));
        const auto document=rns8::bench::read_product_stage_prices(argv[3]);rns8::bench::apply_product_stage_prices(document,identity,prefix,ctx->device_info.name,ctx->device_info.hip_runtime_version,layouts);
        stage_price_identity=sha256(document.dump());
        cost_model={{"kind","measured-stage-prices-unqualified"},{"cost_schema_version",2},{"price_sha256",stage_price_identity},{"qualified",false}};
      }
    }
    check(RuntimeProductRecipe::create(ctx,inputs,nodes,budget-known_payload,owner,
        mode!="recipe-full" && mode!="recipe-packed-full",chosen,{},layout_prices));arena_bytes=owner->allocation().arena_bytes;lift_bytes=owner->lift_scratch_bytes();selected=owner->compute_prefixes();}
  if(lift){check(rns8_gemm_rns(ctx,producer,matrices[0],matrices[1],matrices[2],producer_work));
    if(cpu_lift){co.requested_backend=RNS8_BACKEND_CPU_REFERENCE;check(rns8_create_context(-1,&co,&cpu));matrix(cpu,&host_matrix);}
    else{rns8_exact_lift_options o{};o.struct_size=sizeof(o);o.abi_version=RNS8_ABI_VERSION;o.target_prefix=prefix;o.algorithm=mode=="lift-wide"?0:1;o.max_total_scratch_bytes=budget-known_payload;check(rns8_prepare_exact_lift_hip(ctx,matrices[2],&o,&prepared_lift));if(mode=="lift-graph")check(rns8_capture_prepared_hip_graph(prepared_lift));}}
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  hip_unique_stream stream;if(mode=="lift-graph" && stream.create_non_blocking()!=hipSuccess)throw std::runtime_error("stream setup");
#endif
  if(native_roundtrip)selected=std::vector<uint32_t>(nodes.size(),prefix);
  const double setup=us(setup_begin,Clock::now());uint64_t replay=0;
  const auto export_output=[&](rns8_matrix* result){check(u?rns8_export_exact_wide_unsigned_limbs(ctx,fixed,result,output.data(),side+3,3):rns8_export_exact_wide_signed_limbs(ctx,fixed,result,output.data(),side+3,3));};
  auto invoke=[&](){const auto start=Clock::now();++replay;rns8_matrix* result=matrices.back();
    if(lift){result=matrices[2];check(rns8_gemm_rns(ctx,producer,matrices[0],matrices[1],result,producer_work));}
    const auto produced=Clock::now();
    if(ragged)check(rns8_execute_product_recipe(ragged_recipe,result));
    else if(recipe)check(owner->execute(result));
    else if(lift){if(cpu_lift){check(hip_direct_copy_device_to_host(ctx->device_id,host_matrix->residues.data(),result->hip_residues,size_t(cells*source)));
        // The control explicitly transfers existing residue planes, not native
        // integers. Copy their already-established range/identity proof locally.
        host_matrix->host_residues_current=true;host_matrix->device_residues_current=false;host_matrix->exact_max_magnitude=result->exact_max_magnitude;host_matrix->exact_range_prefix=source;host_matrix->materialized_rns_prefix=source;
        check(rns8_lift_exact_wide_cpu(cpu,host_matrix,prefix,budget-known_payload));if(prefix>source)check(hip_direct_copy_host_to_device(ctx->device_id,static_cast<int8_t*>(result->hip_residues)+cells*source,host_matrix->residues.data()+cells*source,size_t(cells*(prefix-source))));
        result->exact_range_prefix=prefix;result->materialized_rns_prefix=prefix;result->host_residues_current=false;
      }else{check(rns8_refresh_prepared_inputs(prepared_lift));rns8_completion_token token{};
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
        if(mode=="lift-graph"){check(rns8_submit_prepared_hip_graph(prepared_lift,stream.get(),&token));rns8_completion_info info{};info.struct_size=sizeof(info);info.abi_version=RNS8_ABI_VERSION;check(rns8_wait_prepared_hip(prepared_lift,token,&info));check(info.operation_status);}
        else
#endif
        check(rns8_execute_prepared(prepared_lift,&token));}}
    else if(mode=="wider-producer"){result=matrices[2];check(rns8_gemm_rns(ctx,fixed,matrices[0],matrices[1],result,fixed_work));}
    else if(native_roundtrip) {
      for(size_t i=0;i<nodes.size();++i) {
        const auto node=nodes[i];
        const auto import=[&](uint32_t id) {
          ++native_source_version;
          check(u?rns8_pack_u64(ctx,matrices[id],native_values[id].data(),side,native_source_version):
              rns8_pack_i64(ctx,matrices[id],reinterpret_cast<const int64_t*>(native_values[id].data()),side,native_source_version));
        };
        import(node.a);if(node.b!=node.a)import(node.b);
        auto* target=matrices[i+2];check(rns8_gemm_rns(ctx,fixed,matrices[node.a],matrices[node.b],target,fixed_work));
        if(i+1<nodes.size())check(u?rns8_export_exact_wide_unsigned_limbs(ctx,fixed,target,native_values[i+2].data(),side,1):
            rns8_export_exact_wide_signed_limbs(ctx,fixed,target,native_values[i+2].data(),side,1));
      }
    }
    else{selected=json::array();for(size_t i=0;i<nodes.size();++i){const auto node=nodes[i];auto* target=matrices[i+2];
        if(mode=="manual-auto"){uint32_t p=0;check(rns8_gemm_exact_wide_hip_auto(ctx,adaptive,matrices[node.a],matrices[node.b],target,auto_work,1,budget-known_payload,&p));selected.push_back(p);}
        else{check(rns8_gemm_rns(ctx,fixed,matrices[node.a],matrices[node.b],target,fixed_work));selected.push_back(prefix);}
        if(eager && i+1<nodes.size())export_output(target);}
      if(mode=="manual-auto")check(rns8_lift_exact_wide_hip(ctx,result,prefix,budget-known_payload));}
    const auto computed=Clock::now();export_output(result);const auto end=Clock::now();
    return json{{"producer_us",us(start,produced)},{"execute_lift_us",us(produced,computed)},{"final_export_us",us(computed,end)},{"total_us",us(start,end)}};};
  auto audit=[&](){
    if(native_roundtrip)for(size_t id=2;id<native_values.size();++id)for(auto word:native_values[id]) {
      cpp_int got=word;if(!u && word>>63)got-=cpp_int(1)<<64;
      if(got!=values[id])throw std::runtime_error("native intermediate integer oracle");
    }
    for(unsigned row=0;row<side;++row)for(unsigned col=0;col<side+3;++col){const auto index=(uint64_t(row)*(side+3)+col)*3;if(col>=side){for(unsigned l=0;l<3;++l)if(output[index+l]!=99)throw std::runtime_error("padding");continue;}
      cpp_int got=cpp_int(output[index])+(cpp_int(output[index+1])<<64)+(cpp_int(output[index+2])<<128);if(!u && output[index+2]>>63)got-=cpp_int(1)<<192;
      const auto& expected=(lift || mode=="wider-producer")?cpp_int(side):values.back();if(got!=expected)throw std::runtime_error("independent integer oracle");}};
  auto first=invoke();audit();for(unsigned i=0;i<warmups;++i)invoke();const auto before=hip_direct_allocation_counters_snapshot();json samples=json::array();for(unsigned i=0;i<repeats;++i){samples.push_back(invoke());audit();}const auto after=hip_direct_allocation_counters_snapshot();
  json sustained=json::array();for(unsigned i=0;i<3;++i){unsigned count=0;const auto begin=Clock::now();do{invoke();++count;}while(count<4096 && (count<16 || us(begin,Clock::now())<250000));const double elapsed=us(begin,Clock::now());audit();sustained.push_back({{"calls",count},{"total_us",elapsed},{"per_call_us",elapsed/count}});}
  const uint64_t retained_auto=auto_work->exact_auto_scratch_owner.bytes();const std::string target=ctx->device_info.gcn_arch,device=ctx->device_info.name;const int runtime=ctx->device_info.hip_runtime_version;
  if(materialized) {
    const auto& plan=owner->materialization_selection();json packed=json::array();
    for(size_t id=0;id<plan.packed.size();++id)if(plan.packed[id].used) {
      const auto& view=plan.packed[id];packed.push_back({{"value_id",id},{"capacity_prefix",view.layout.prefix},
          {"first_reader",view.first},{"last_reader",view.last},{"bytes",view.layout.bytes},{"arena_offset",view.offset}});
    }
    materialization_plan={{"packed_b",plan.packed_b},{"packed_values",packed},{"evaluated",plan.evaluated},
        {"exhaustive",plan.exhaustive},{"chain_dynamic_programming",plan.chain_dynamic_programming},{"cost_optimal",plan.cost_optimal},
        {"estimated_ns",plan.estimated_ns},{"completed_packing_calls",owner->completed_packing_calls()}};
  }
  if(owner)check(owner->close());owner.reset();if(ragged_recipe)check(rns8_destroy_product_recipe(ragged_recipe));if(ragged_source)check(rns8_destroy_ragged_materialization(ragged_source));if(ragged_converted)check(rns8_destroy_matrix(ragged_converted));if(prepared_lift)check(rns8_destroy_prepared_execution(prepared_lift));if(host_matrix)check(rns8_destroy_matrix(host_matrix));if(cpu)check(rns8_destroy_context(cpu));for(auto* m:matrices)check(rns8_destroy_matrix(m));
  for(auto* w:{fixed_work,auto_work,producer_work})check(rns8_destroy_workspace(w));for(auto* p:{fixed,adaptive,producer})check(rns8_destroy_plan(p));check(rns8_destroy_context(ctx));
  std::cout<<json({{"schema","rns8_lift_recipe_microcase_v3"},{"case",c},{"run_id",argv[2]},{"build_manifest",json::parse(RNS8_BUILD_MANIFEST_JSON)},{"target",target},{"device",device},{"runtime_version",runtime},
    {"timing_source","host_monotonic_completed_public_call"},{"nominal_clock_tick_ns",1e9*double(Clock::period::num)/double(Clock::period::den)},
    {"output_boundary",lift || mode=="wider-producer"?"one_product_full_prefix_final_three_limb_export":"complete_dag_final_three_limb_export"},{"extra_intermediate_observations",eager},{"private_residue_transfer_control",cpu_lift},{"native_roundtrip_control",native_roundtrip},
    {"fixed_plan_exact_proof_policy",scalar_first?"scalar_first":"eager_axis"},{"logical_payload_estimate_bytes",known_payload},
    {"ragged_retained_bytes",ragged_retained},{"ragged_materialized_input_bytes",ragged_input_bytes},{"ragged_binding_peak_bytes",ragged_peak},{"recipe_arena_bytes",arena_bytes},{"recipe_retained_lift_bytes",lift_bytes},{"auto_retained_device_bytes",retained_auto},{"selected_compute_prefixes",selected},{"cost_model",cost_model},{"materialization_plan",materialization_plan},{"stage_price_identity",stage_price_identity},
    {"host_payload_bytes_before_final_export",cpu_lift?cells*prefix:native_h2d+native_d2h},{"final_export_payload_bytes",cells*24},
    {"native_roundtrip_h2d_payload_bytes_per_replay",native_h2d},{"native_roundtrip_d2h_payload_bytes_per_replay",native_d2h},{"native_roundtrip_retained_host_bytes",native_host_bytes},
    {"replay_hip_counter_samples",repeats},{"replay_hip_h2d_bytes",after.h2d_bytes-before.h2d_bytes},{"replay_hip_d2h_bytes",after.d2h_bytes-before.d2h_bytes},
    {"replay_hip_h2d_calls",after.h2d_calls-before.h2d_calls},{"replay_hip_d2h_calls",after.d2h_calls-before.d2h_calls},
    {"replay_hip_allocations",after.allocate_calls-before.allocate_calls},{"replay_hip_frees",after.free_calls-before.free_calls},
    {"setup_us",setup},{"first_call",first},{"samples",samples},{"sustained",sustained},{"exact_reference_passed",true},{"promotion_eligible",false}}).dump(2)<<'\n';return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
