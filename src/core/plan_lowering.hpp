#ifndef RNS8_CORE_PLAN_LOWERING_HPP
#define RNS8_CORE_PLAN_LOWERING_HPP

#include <string>

#include "rns8/rns8.h"

namespace rns8::detail {

struct PlanLoweringDescription {
  std::string operation;
  std::string semantic_contract;
  std::string backend_family;
  std::string input_domain;
  std::string output_domain;
  std::string desired_output;
  std::string schedule_strategy;
  std::string packing_strategy;
  std::string reuse_strategy;
  std::string conversion_strategy;
  std::string lowering_path;
  bool final_export_available = false;
  bool rns_continuation_available = false;
  bool native_continuation_available = false;
  bool native_to_rns_available = false;
  bool reusable_b_prepack_available = false;
};

PlanLoweringDescription describe_plan_lowering(
    const rns8_plan_backend_info& backend,
    const rns8_plan_packing_info& packing,
    const rns8_plan_schedule_info& schedule);

}  // namespace rns8::detail

#endif  // RNS8_CORE_PLAN_LOWERING_HPP
