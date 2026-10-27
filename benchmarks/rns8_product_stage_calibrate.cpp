// Synchronous stage prices for the resident product executor. Setup, transfers
// that construct fixtures, and independent integer audits are outside samples.
#include "core/api_internal.hpp"
#include "core/hip_resources.hpp"
#include "backend_common/product_materialization_plan.hpp"
#include "rns8/experimental_product.h"
#include "test_product_stage_prices.hpp"
#include "product_dataset.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <set>
namespace {
using namespace rns8::detail;
using namespace rns8::detail::api;
using json=nlohmann::json;
using Clock=std::chrono::steady_clock;
void check(rns8_status status){if(status!=RNS8_SUCCESS)throw std::runtime_error(rns8_status_string(status));}
uint32_t number(const json& c,const char* key,uint32_t lo,uint32_t hi) {
  const auto& v=c.at(key);if(!v.is_number_unsigned() && (!v.is_number_integer() || v.get<int64_t>()<0))throw std::invalid_argument(key);
  const auto n=v.get<uint64_t>();if(n<lo || n>hi)throw std::invalid_argument(key);return static_cast<uint32_t>(n);
}
std::vector<uint32_t> domain(const json& c,const char* key,uint32_t prefix) {
  if(!c.at(key).is_array() || c.at(key).empty() || c.at(key).size()>20)throw std::invalid_argument(key);
  std::vector<uint32_t> values;
  for(const auto& v:c.at(key)){json row={{"p",v}};values.push_back(number(row,"p",1,prefix));}
  if(!std::is_sorted(values.begin(),values.end()) || std::adjacent_find(values.begin(),values.end())!=values.end())throw std::invalid_argument(key);
  return values;
}
struct Context {
  rns8_context* value=nullptr;
  std::vector<rns8_matrix*> inputs;
  ~Context(){for(auto* m:inputs)(void)rns8_destroy_matrix(m);if(value)(void)rns8_destroy_context(value);}
};
}
int main(int argc,char** argv){try {
  if(argc==2 && std::string(argv[1])=="--self-test"){rns8::bench::test_product_stage_price_ingestion();std::cout<<"stage price ingestion passed\n";return 0;}
  if(argc!=3 || !argv[2][0])throw std::invalid_argument("usage: rns8-product-stage-calibrate CASE.json unique-run-id");
  json c;std::ifstream(argv[1])>>c;
  if(c.size()!=11+unsigned(c.contains("input_values")) || c.at("schema")!="product_stage_case_v1" || !c.at("unsigned_inputs").is_boolean())throw std::invalid_argument("case ABI");
  const auto side=number(c,"side",1,256),depth=number(c,"depth",1,12),prefix=number(c,"prefix",1,20);
  const auto warmups=number(c,"warmups",1,100),repeats=number(c,"repeats",3,100);
  const auto ps=domain(c,"prefixes",prefix),caps=domain(c,"capacities",prefix);
  const auto budget=number(c,"max_payload_bytes",64,UINT32_MAX);
  const bool uns=c.at("unsigned_inputs");const std::string topology=c.at("topology");
  const auto semantics=uns?RNS8_EXACT_WIDE_UNSIGNED:RNS8_EXACT_WIDE_SIGNED;
  std::vector<ProductNode> nodes;
  if(topology=="chain")for(uint32_t i=0;i<depth;++i)nodes.push_back({i?i+1:0,1});
  else if(topology=="diamond" && depth==4)nodes={{0,1},{2,1},{0,2},{3,4}};
  else throw std::invalid_argument("topology");
  const uint64_t cells=uint64_t(side)*side;
  const auto dataset=rns8::bench::product_dataset_inputs(c);
  std::vector<cpp_int> values(dataset.begin(),dataset.end());
  for(auto node:nodes){cpp_int value=cpp_int(side)*values[node.a]*values[node.b];
    if(!exact_range_fits(semantics,boost::multiprecision::abs(value),modulus_product(prefix)))throw std::invalid_argument("strict range");values.push_back(std::move(value));}
  ResidueLayout uniform;check(make_residue_layout(ResidueLayoutKind::modulus_major,side,side,side,prefix,0,0,uniform)?RNS8_SUCCESS:RNS8_RANGE_ERROR);
  ResidueLayout largest;check(make_residue_layout(ResidueLayoutKind::tile_major,side,side,side,caps.back(),16,16,largest)?RNS8_SUCCESS:RNS8_RANGE_ERROR);
  const uint64_t scratch_bytes=sizeof(int)+cells*(prefix-1);
  const uint64_t payload=3*uniform.bytes+largest.bytes+scratch_bytes+2*cells*prefix;
  if(payload>=budget || cells*side*nodes.size()>134217728)throw std::invalid_argument("payload/work");
  const auto setup_begin=Clock::now();Context context;
  rns8_context_options options{};options.struct_size=sizeof(options);options.abi_version=RNS8_ABI_VERSION;options.requested_backend=RNS8_BACKEND_HIP_DIRECT;
  hip_direct_timing_set_enabled(false);check(rns8_create_context(0,&options,&context.value));auto* ctx=context.value;
  std::vector<uint64_t> ones(cells,1);
  for(unsigned i=0;i<2;++i){rns8_matrix_desc d{};d.struct_size=sizeof(d);d.abi_version=RNS8_ABI_VERSION;d.semantics=semantics;
    std::fill(ones.begin(),ones.end(),rns8::bench::product_dataset_word(dataset[i]));
    d.rows=side;d.cols=side;d.logical_ld=side;d.logical_layout=RNS8_LAYOUT_ROW_MAJOR;d.max_prefix=prefix;
    rns8_matrix* m=nullptr;check(rns8_create_matrix(ctx,&d,&m));context.inputs.push_back(m);
    check(uns?rns8_pack_u64(ctx,m,ones.data(),side,1):rns8_pack_i64(ctx,m,reinterpret_cast<const int64_t*>(ones.data()),side,1));}
  std::vector<rns8_product_node> public_nodes;for(auto node:nodes)public_nodes.push_back({node.a,node.b});
  std::vector<const rns8_matrix*> input_views(context.inputs.begin(),context.inputs.end());
  rns8_product_recipe_desc desc{};desc.struct_size=sizeof(desc);desc.abi_version=RNS8_ABI_VERSION;desc.inputs=input_views.data();desc.input_count=2;
  desc.nodes=public_nodes.data();desc.node_count=static_cast<uint32_t>(public_nodes.size());desc.prefix_policy=RNS8_PRODUCT_MINIMUM_COMPUTE_PREFIX;desc.max_arena_and_lift_bytes=budget;
  rns8_product_materialization_identity identity{};identity.struct_size=sizeof(identity);identity.abi_version=RNS8_ABI_VERSION;
  check(rns8_query_product_materialization_identity(ctx,&desc,nullptr,0,&identity));check(identity.admission_status);
  hip_direct_device_buffer a,b,result,packed,scratch;
  for(auto* buffer:{&a,&b,&result})check(buffer->allocate(ctx->device_id,uniform.bytes));
  check(packed.allocate(ctx->device_id,largest.bytes));check(scratch.allocate(ctx->device_id,scratch_bytes));
  const double setup_us=std::chrono::duration<double,std::micro>(Clock::now()-setup_begin).count();
  std::vector<int8_t> ah(uniform.bytes),bh(uniform.bytes),initial(uniform.bytes),expected(uniform.bytes),got(uniform.bytes);
  auto fill=[&](std::vector<int8_t>& host,const cpp_int& value){for(uint32_t p=0;p<prefix;++p)
    std::fill_n(host.data()+p*cells,cells,centered_residue(value,kDefaultModuli[p]));};
  auto upload=[&](void* dst,const std::vector<int8_t>& host){check(hip_direct_copy_host_to_device(ctx->device_id,dst,host.data(),host.size()));};
  json stages=json::array();
  for(uint32_t i=0;i<nodes.size();++i) {
    fill(ah,values[nodes[i].a]);fill(bh,values[nodes[i].b]);upload(a.get(),ah);upload(b.get(),bh);
    const auto record=[&](const char* kind,uint32_t capacity,uint32_t old,uint32_t p,const auto& invoke,void* output,const std::vector<int8_t>& reference) {
      for(uint32_t n=0;n<warmups;++n)invoke();
      json samples=json::array();const auto before=hip_direct_allocation_counters_snapshot();
      for(uint32_t n=0;n<repeats;++n){const auto begin=Clock::now();invoke();const auto end=Clock::now();
        const double ns=std::chrono::duration<double,std::nano>(end-begin).count();if(!(ns>0))throw std::runtime_error("censored stage duration");samples.push_back(ns);}
      const auto after=hip_direct_allocation_counters_snapshot();
      if(after.allocate_calls!=before.allocate_calls || after.free_calls!=before.free_calls)throw std::runtime_error("stage allocated on replay");
      got.resize(reference.size());check(hip_direct_copy_device_to_host(ctx->device_id,got.data(),output,got.size()));
      if(got!=reference)throw std::runtime_error(std::string(kind)+" independent integer/physical-padding oracle");
      stages.push_back({{"node",i},{"kind",kind},{"capacity",capacity},{"old_prefix",old},{"prefix",p},{"elapsed_ns",samples},
        {"sample_h2d_bytes",after.h2d_bytes-before.h2d_bytes},{"sample_d2h_bytes",after.d2h_bytes-before.d2h_bytes},
        {"sample_d2d_bytes",after.d2d_bytes-before.d2d_bytes},{"exact_reference_passed",true}});
    };
    for(auto p:ps)if(p>=identity.minimum_prefixes[i]) {
      std::fill(initial.begin(),initial.end(),63);expected=initial;
      for(uint32_t plane=0;plane<p;++plane)std::fill_n(expected.data()+plane*cells,cells,centered_residue(values[i+2],kDefaultModuli[plane]));
      upload(result.get(),initial);
      record("uniform_gemm",0,0,p,[&]{check(hip_direct_gemm_rns_device(ctx->device_id,static_cast<const int8_t*>(a.get()),static_cast<const int8_t*>(b.get()),static_cast<int8_t*>(result.get()),side,side,side,side,side,side,p));},result.get(),expected);
      for(auto capacity:caps)if(capacity>=p) {
        ResidueLayout view;check(make_residue_layout(ResidueLayoutKind::tile_major,side,side,side,capacity,16,16,view)?RNS8_SUCCESS:RNS8_RANGE_ERROR);
        std::vector<int8_t> ph(view.bytes,63);if(!convert_residue_planes_host(bh.data(),uniform,ph.data(),view,0,capacity))throw std::runtime_error("host packing");
        upload(packed.get(),ph);upload(result.get(),initial);
        record("packed_gemm",capacity,0,p,[&]{check(hip_direct_gemm_residue_layout(ctx->device_id,a.get(),uniform,packed.get(),view,result.get(),uniform,p));},result.get(),expected);
      }
      for(auto q:ps)if(q>p) {
        initial.assign(uniform.bytes,63);fill(expected,values[i+2]);std::copy_n(expected.data(),p*cells,initial.data());upload(result.get(),initial);
        std::fill(expected.begin()+q*cells,expected.end(),63);
        uint64_t words[3]{};cpp_int bound=boost::multiprecision::abs(values[i+2]),mask=(cpp_int(1)<<64)-1;
        for(auto& word:words){word=(bound&mask).convert_to<uint64_t>();bound>>=64;}if(bound!=0)throw std::runtime_error("bound width");
        record("lift",0,p,q,[&]{bool touched=false;check(hip_direct_lift_exact_suffix_workspace(ctx->device_id,result.get(),cells,p,q,!uns,words,scratch.get(),scratch_bytes,touched));},result.get(),expected);
      }
    }
    for(auto capacity:caps)for(auto p:ps)if(p<=capacity) {
      ResidueLayout view;check(make_residue_layout(ResidueLayoutKind::tile_major,side,side,side,capacity,16,16,view)?RNS8_SUCCESS:RNS8_RANGE_ERROR);
      auto old_domain=ps;old_domain.insert(old_domain.begin(),0);
      for(auto old:old_domain)if(old<p) {
        std::vector<int8_t> ph(view.bytes,63);
        if(old && !convert_residue_planes_host(bh.data(),uniform,ph.data(),view,0,old))throw std::runtime_error("old packing");
        upload(packed.get(),ph);
        if(!convert_residue_planes_host(bh.data(),uniform,ph.data(),view,old,p))throw std::runtime_error("expected packing");
        record("pack",capacity,old,p,[&]{check(hip_direct_convert_residue_planes(ctx->device_id,b.get(),uniform,packed.get(),view,old,p));},packed.get(),ph);
      }
    }
    if(i+1==nodes.size()) {
      fill(expected,values[i+2]);upload(result.get(),expected);
      record("publish",0,0,prefix,[&]{check(hip_direct_copy_device_to_device(ctx->device_id,a.get(),result.get(),uniform.bytes));},a.get(),expected);
    }
  }
  json minima=json::array();for(uint32_t i=0;i<nodes.size();++i)minima.push_back(identity.minimum_prefixes[i]);
  std::cout<<json({{"schema","rns8_product_stage_capture_v1"},{"case",c},{"run_id",argv[2]},{"build_manifest",json::parse(RNS8_BUILD_MANIFEST_JSON)},
    {"identity",{{"graph_sha256",identity.graph_sha256},{"build_sha256",identity.build_sha256},{"target",identity.target},{"node_count",identity.node_count},{"minimum_prefixes",minima}}},
    {"device",ctx->device_info.name},{"runtime_version",ctx->device_info.hip_runtime_version},{"cost_schema_version",2},
    {"dataset",rns8::bench::product_dataset_identity(c)},{"cache_state","hot_repeated_stage"},{"timing_source","host_monotonic_completed_backend_call"},
    {"nominal_clock_tick_ns",1e9*double(Clock::period::num)/double(Clock::period::den)},
    {"host_instrumentation_state","hooks_disabled"},{"setup_us",setup_us},
    {"calibration_total_us",std::chrono::duration<double,std::micro>(Clock::now()-setup_begin).count()},{"logical_device_payload_bytes",payload},
    {"stages",stages},{"exact_reference_passed",true},{"promotion_eligible",false}}).dump(2)<<'\n';return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
