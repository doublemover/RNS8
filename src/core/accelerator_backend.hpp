#ifndef RNS8_CORE_ACCELERATOR_BACKEND_HPP
#define RNS8_CORE_ACCELERATOR_BACKEND_HPP

#include <cstdint>

#include "rns8/rns8.h"

namespace rns8::detail {

struct accelerator_backend_descriptor {
  rns8_backend_kind backend = RNS8_BACKEND_AUTO;
  const char* backend_name = "";
  const char* library_name = "";
  const char* enable_flag = "";
  const char* disabled_selected_kernel = "not_implemented";
  const char* disabled_epilogue_mode = "not_implemented";
  const char* disabled_workspace_mode = "not_implemented";
  const char* disabled_isa_evidence = "not_validated";
  const char* disabled_status = "not_implemented_evidence_only";
  const char* disabled_detail = "";
  uint32_t supports_bounded_rns = 0;
  uint32_t supports_exact_wide_rns = 0;
  uint32_t supports_finite_u8 = 0;
  uint32_t supports_wrap64 = 0;
  uint32_t is_matrix_engine_candidate = 0;
};

bool accelerator_backend_kind(rns8_backend_kind backend);
bool accelerator_backend_compiled(rns8_backend_kind backend);
bool accelerator_backend_supports_semantics(rns8_backend_kind backend, rns8_semantics semantics);
const accelerator_backend_descriptor* accelerator_backend_descriptor_for(rns8_backend_kind backend);
void fill_disabled_accelerator_capability(rns8_backend_kind backend, rns8_backend_capability_info& info);

}  // namespace rns8::detail

#endif
