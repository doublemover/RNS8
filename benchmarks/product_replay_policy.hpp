#pragma once
#include "backend_common/replay_cost.hpp"
#include <nlohmann/json.hpp>
#include <set>
#include <string>
namespace rns8::bench {
inline detail::ReplayCostPolicy product_replay_policy(const nlohmann::json& c) {
  detail::ReplayCostPolicy p;
  if(!c.contains("replay_policy"))return p;
  const auto& j=c.at("replay_policy");
  const std::set<std::string> fields{"expected_replays","retain_external_b","require_amortization",
      "preparation_ns","calibration_ns","orchestration_ns","publication_ns","baseline_replay_ns","baseline_preparation_ns"};
  if(!j.is_object())throw std::invalid_argument("replay policy object");
  for(const auto& item:j.items())if(!fields.count(item.key()))throw std::invalid_argument("replay policy field");
  const auto integer=[&](const char* key,uint64_t& dst) {
    if(!j.contains(key))return;
    const auto& v=j.at(key);
    if(!v.is_number_integer() || (!v.is_number_unsigned() && v.get<int64_t>()<0))throw std::invalid_argument("replay policy integer");
    dst=v.get<uint64_t>();
  };
  integer("expected_replays",p.replays);integer("preparation_ns",p.preparation_ns);integer("calibration_ns",p.calibration_ns);
  integer("orchestration_ns",p.orchestration_ns);integer("publication_ns",p.publication_ns);
  integer("baseline_replay_ns",p.baseline_replay_ns);integer("baseline_preparation_ns",p.baseline_preparation_ns);
  for(auto entry:{std::pair<const char*,bool*>{"retain_external_b",&p.retain_external_b},{"require_amortization",&p.require_amortization}})
    if(j.contains(entry.first)) {
      if(!j.at(entry.first).is_boolean())throw std::invalid_argument("replay policy boolean");
      *entry.second=j.at(entry.first).get<bool>();
    }
  if(!p.replays || p.replays==UINT64_MAX)throw std::invalid_argument("replay lifetime");
  return p;
}
inline nlohmann::json product_replay_policy_json(const detail::ReplayCostPolicy& p) {
  return {{"expected_replays",p.replays},{"retain_external_b",p.retain_external_b},{"require_amortization",p.require_amortization},
      {"preparation_ns",p.preparation_ns},{"calibration_ns",p.calibration_ns},{"orchestration_ns",p.orchestration_ns},
      {"publication_ns",p.publication_ns},{"baseline_replay_ns",p.baseline_replay_ns},{"baseline_preparation_ns",p.baseline_preparation_ns}};
}
} // namespace rns8::bench
