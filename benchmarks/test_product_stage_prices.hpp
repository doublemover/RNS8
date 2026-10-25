#pragma once
#include "product_stage_prices.hpp"
namespace rns8::bench {
inline void test_product_stage_price_ingestion() {
  rns8_product_materialization_identity id{};id.admission_status=RNS8_SUCCESS;id.node_count=1;id.minimum_prefixes[0]=1;
  std::fill_n(id.graph_sha256,64,'a');std::fill_n(id.build_sha256,64,'b');std::copy_n("synthetic",10,id.target);
  StageJson runs=StageJson::array(),executions=StageJson::array(),artifacts=StageJson::array();
  for(unsigned i=0;i<5;++i){runs.push_back("run"+std::to_string(i));executions.push_back("process"+std::to_string(i));
    artifacts.push_back({{"executable_sha256",std::string(64,'c')},{"capture_sha256",std::string(64,'d')}});}
  auto cell=[](const char* kind,uint32_t cap,uint32_t old,uint32_t p,uint64_t ns){return StageJson{
    {"node",0},{"kind",kind},{"capacity",cap},{"old_prefix",old},{"prefix",p},{"cost_ns",ns},{"interval_ns",{double(ns-1),double(ns+1)}}};};
  StageJson document={{"schema","rns8_product_stage_prices_v1"},{"cost_schema_version",2},{"qualified",false},{"promotion_eligible",false},
    {"identity",{{"graph_sha256",id.graph_sha256},{"build_sha256",id.build_sha256},{"target",id.target},{"node_count",1},{"minimum_prefixes",{1}}}},
    {"training_run_ids",runs},{"training_execution_ids",executions},{"source_artifacts",artifacts},
    {"dataset","all_one_inputs_exact_integer_dag_v1"},{"cache_state","hot_repeated_stage"},{"timing_source","host_monotonic_completed_backend_call"},{"device","synthetic-device"},{"runtime_version",7},
    {"cells",{cell("uniform_gemm",0,0,1,100),cell("packed_gemm",2,0,1,80),cell("pack",2,0,1,20),cell("lift",0,1,2,40),cell("publish",0,0,3,10)}}};
  detail::ProductMaterializationProfile profile;apply_product_stage_prices(document,id,3,"synthetic-device",7,profile);
  const auto assert_valid=[&]{if(profile.uniform.nodes[0].gemm[1]!=100 || profile.uniform.nodes[0].gemm[2]!=UINT64_MAX ||
      profile.packed[0].gemm[detail::ProductPackedCosts::gemm_index(2,1)]!=80 ||
      profile.packed[0].gemm[detail::ProductPackedCosts::gemm_index(1,1)]!=UINT64_MAX ||
      profile.packed[0].pack[detail::ProductPackedCosts::pack_index(2,0,1)]!=20 ||
      profile.uniform.nodes[0].lift[1*21+2]!=40)throw std::runtime_error("stage price index/unknown regression");};
  assert_valid();const auto provenance=profile.uniform.provenance;
  const auto reject=[&](const StageJson& bad){bool rejected=false;try{apply_product_stage_prices(bad,id,3,"synthetic-device",7,profile);}catch(const std::exception&){rejected=true;}
    if(!rejected || profile.uniform.provenance!=provenance)throw std::runtime_error("stage price rejection/transaction regression");assert_valid();};
  auto bad=document;bad["identity"]["graph_sha256"]=std::string(64,'e');reject(bad);
  bad=document;bad["identity"]["build_sha256"]=std::string(64,'e');reject(bad);
  bad=document;bad["identity"]["target"]="other";reject(bad);
  bad=document;bad["identity"]["minimum_prefixes"]={2};reject(bad);
  bad=document;bad["cost_schema_version"]=1;reject(bad);
  bad=document;bad["cost_schema_version"]=2.0;reject(bad);
  bad=document;bad["device"]="other-device";reject(bad);
  bad=document;bad["runtime_version"]=8;reject(bad);
  bad=document;bad["training_execution_ids"][4]="process0";reject(bad);
  bad=document;bad["training_run_ids"].erase(4);reject(bad);
  bad=document;bad["cells"].push_back(bad["cells"][0]);reject(bad);
  bad=document;bad["cells"][0]["cost_ns"]=UINT64_MAX;reject(bad);
  bad=document;bad["cells"][0]["cost_ns"]=-1;reject(bad);
  bad=document;bad["cells"][0]["node"]=1;reject(bad);
  bad=document;bad["cells"][1]["capacity"]=0;reject(bad);
  bad=document;bad["cells"][3]["old_prefix"]=2;reject(bad);
  bad=document;bad["cells"][4]["prefix"]=2;reject(bad);
  bad=document;bad["cells"][4]["interval_ns"]={20,21};reject(bad);
  bad=document;bad["qualified"]=true;reject(bad);
}
} // namespace rns8::bench
