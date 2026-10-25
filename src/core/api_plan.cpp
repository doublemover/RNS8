#include "core/api_internal.hpp"
#include "backend_common/garner_constants.hpp"
#include <cmath>
#include "backend_common/bilinear_resources.hpp"
#include "backend_common/rank23_slp_resources.hpp"
#include "backend_common/gram_plan.hpp"

namespace rns8::detail::api {
#include "api_plan_schedule.inc"
#include "api_plan_contract_names.inc"
#include "api_plan_workspace_packing.inc"
#include "api_operation_provider.inc"
#include "api_plan_arithmetic_certificate.inc"
#include "api_plan_auto_backend.inc"
#include "api_plan_workspace_fingerprint.inc"
#include "api_plan_public_api.inc"

#include "api_pipeline_planner.inc"
