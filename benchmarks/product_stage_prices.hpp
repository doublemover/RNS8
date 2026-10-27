#pragma once
#include "backend_common/product_materialization_plan.hpp"
#include "backend_common/sha256.hpp"
#include "rns8/experimental_product.h"
#include <cmath>
#include <cstring>
#include <fstream>
#include <nlohmann/json.hpp>
#include <set>
namespace rns8::bench {
using StageJson=nlohmann::json;
inline uint64_t stage_uint(const StageJson& value,uint64_t lo,uint64_t hi) {
  if(!value.is_number_unsigned() && (!value.is_number_integer() || value.get<int64_t>()<0))throw std::invalid_argument("stage integer");
  const auto n=value.get<uint64_t>();if(n<lo || n>hi)throw std::invalid_argument("stage integer range");return n;
}
inline bool stage_digest(const StageJson& value) {
  if(!value.is_string())return false;const auto s=value.get<std::string>();
  return s.size()==64 && s.find_first_not_of("0123456789abcdef")==std::string::npos;
}
inline std::string product_stage_price_identity(const StageJson& document) {
  static_assert(sizeof(double)==sizeof(uint64_t) && std::numeric_limits<double>::is_iec559);
  // Serializers need not spell identical binary64 values identically. Normalize
  // intervals to exact bit words before hashing their sorted JSON identity.
  auto normalized=document;
  for(auto& cell:normalized.at("cells"))for(auto& endpoint:cell.at("interval_ns")) {
    const double value=endpoint.get<double>();uint64_t bits=0;std::memcpy(&bits,&value,sizeof(bits));
    endpoint="binary64:"+std::to_string(bits);
  }
  return detail::sha256(normalized.dump());
}
// Measured estimates remain caller data. The public identity/range checker owns
// admission; this parser cannot waive it or install an AUTO route.
inline void apply_product_stage_prices(const StageJson& document,
    const rns8_product_materialization_identity& identity,uint32_t prefix,const std::string& device,int runtime_version,detail::ProductMaterializationProfile& profile,
    const std::string& dataset_identity="all_one_inputs_exact_integer_dag_v1") {
  if(document.size()!=15 || document.at("schema")!="rns8_product_stage_prices_v1" ||
      stage_uint(document.at("cost_schema_version"),2,2)!=2 || document.at("promotion_eligible")!=false || document.at("qualified")!=false ||
      document.at("timing_source")!="host_monotonic_completed_backend_call" || document.at("cache_state")!="hot_repeated_stage" ||
      document.at("dataset")!=dataset_identity || document.at("device")!=device ||
      stage_uint(document.at("runtime_version"),1,INT32_MAX)!=uint64_t(runtime_version) || identity.admission_status!=RNS8_SUCCESS || !prefix || prefix>20)
    throw std::invalid_argument("stage price schema/scope");
  const auto training_charge=stage_uint(document.at("training_charge_ns"),1,UINT64_MAX-1);
  const auto& key=document.at("identity");
  if(key.size()!=5 || !stage_digest(key.at("graph_sha256")) || !stage_digest(key.at("build_sha256")) ||
      key.at("graph_sha256")!=identity.graph_sha256 || key.at("build_sha256")!=identity.build_sha256 ||
      key.at("target")!=identity.target || stage_uint(key.at("node_count"),1,64)!=identity.node_count ||
      !key.at("minimum_prefixes").is_array() || key.at("minimum_prefixes").size()!=identity.node_count)
    throw std::invalid_argument("stage price identity mismatch");
  for(uint32_t i=0;i<identity.node_count;++i)if(stage_uint(key.at("minimum_prefixes")[i],1,20)!=identity.minimum_prefixes[i])
    throw std::invalid_argument("stage price independent range mismatch");
  for(const auto* name:{"training_run_ids","training_execution_ids"}) {
    const auto& ids=document.at(name);if(!ids.is_array() || ids.size()<5 || ids.size()>100)throw std::invalid_argument("stage price independent runs");
    std::set<std::string> unique;
    for(const auto& id:ids)if(!id.is_string() || id.get<std::string>().empty() || !unique.insert(id.get<std::string>()).second)
      throw std::invalid_argument("stage price duplicate process");
  }
  if(document.at("training_run_ids").size()!=document.at("training_execution_ids").size())throw std::invalid_argument("stage price process mapping");
  const auto& artifacts=document.at("source_artifacts");
  if(!artifacts.is_array() || artifacts.size()!=document.at("training_run_ids").size())throw std::invalid_argument("stage price artifacts");
  for(const auto& artifact:artifacts)if(artifact.size()!=2 || !stage_digest(artifact.at("executable_sha256")) || !stage_digest(artifact.at("capture_sha256")))
    throw std::invalid_argument("stage price artifact digest");
  const auto& cells=document.at("cells");if(!cells.is_array() || cells.empty() || cells.size()>640000)throw std::invalid_argument("stage price cells");
  // Parse transactionally: a bad late record cannot leave a half-updated table.
  auto next=profile;next.uniform.nodes.assign(identity.node_count,{});next.packed.assign(identity.node_count,{});
  std::set<std::tuple<uint32_t,std::string,uint32_t,uint32_t,uint32_t>> seen;
  for(const auto& cell:cells) {
    if(cell.size()!=7)throw std::invalid_argument("stage cell ABI");
    const auto node=static_cast<uint32_t>(stage_uint(cell.at("node"),0,identity.node_count-1));
    const auto p=static_cast<uint32_t>(stage_uint(cell.at("prefix"),1,prefix));
    const auto old=static_cast<uint32_t>(stage_uint(cell.at("old_prefix"),0,prefix));
    const auto capacity=static_cast<uint32_t>(stage_uint(cell.at("capacity"),0,prefix));
    const auto cost=stage_uint(cell.at("cost_ns"),1,UINT64_MAX-1);const auto kind=cell.at("kind").get<std::string>();
    if(!seen.emplace(node,kind,capacity,old,p).second)throw std::invalid_argument("duplicate stage price");
    const auto& interval=cell.at("interval_ns");if(!interval.is_array() || interval.size()!=2)throw std::invalid_argument("stage interval");
    const double lower=interval[0].get<double>(),upper=interval[1].get<double>();
    if(!std::isfinite(lower) || !std::isfinite(upper) || lower<=0 || upper<lower || double(cost)<lower || double(cost)>std::ceil(upper))
      throw std::invalid_argument("stage interval range");
    if(kind=="uniform_gemm" && !capacity && !old && p>=identity.minimum_prefixes[node])next.uniform.nodes[node].gemm[p]=cost;
    else if(kind=="lift" && !capacity && old>=identity.minimum_prefixes[node] && old<p)next.uniform.nodes[node].lift[old*21+p]=cost;
    else if(kind=="packed_gemm" && !old && p>=identity.minimum_prefixes[node] && capacity>=p)
      next.packed[node].gemm[detail::ProductPackedCosts::gemm_index(capacity,p)]=cost;
    else if(kind=="pack" && old<p && capacity>=p)next.packed[node].pack[detail::ProductPackedCosts::pack_index(capacity,old,p)]=cost;
    // Final publication is a common cost, independent of schedule choice. It
    // is retained in the file for complete held-out execution predictions.
    else if(kind=="publish" && !capacity && !old && p==prefix && node+1==identity.node_count)
      next.replay.publication_ns=std::max(next.replay.publication_ns,cost);
    else throw std::invalid_argument("stage price physical axes/range");
  }
  next.replay.calibration_ns=std::max(next.replay.calibration_ns,training_charge);
  next.cost_schema_version=3;next.uniform.provenance="measured-stage-prices-unqualified:"+product_stage_price_identity(document);profile=std::move(next);
}
inline StageJson read_product_stage_prices(const std::string& path) {
  std::ifstream input(path,std::ios::binary|std::ios::ate);
  if(!input || input.tellg()<0 || input.tellg()>16*1024*1024)throw std::invalid_argument("stage price file size");
  input.seekg(0);StageJson result;input>>result;input>>std::ws;if(!input.eof())throw std::invalid_argument("stage price trailing data");return result;
}
} // namespace rns8::bench
