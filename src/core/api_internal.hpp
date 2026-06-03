#ifndef RNS8_CORE_API_INTERNAL_HPP
#define RNS8_CORE_API_INTERNAL_HPP

#include "core/internal.hpp"

#include <boost/multiprecision/cpp_int.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <vector>

#include "backend_ck/ck_backend.hpp"
#include "backend_hip_direct/hip_backend.hpp"
#include "backend_hipblaslt/hipblaslt_backend.hpp"
#include "backend_wmma/wmma_backend.hpp"
#include "backend_wrap64/wrap64_hip.hpp"
#include "core/accelerator_backend.hpp"
#include "core/autotune_cache.hpp"

namespace rns8::detail::api {

template <typename Fn>
rns8_status guard_api(Fn&& fn) {
  try {
    return fn();
  } catch (const std::bad_alloc&) {
    return RNS8_INTERNAL_ERROR;
  } catch (...) {
    return RNS8_INTERNAL_ERROR;
  }
}

rns8_backend_kind effective_backend(rns8_backend_kind requested, rns8_backend_kind default_backend);
bool backend_supports_semantics(rns8_backend_kind backend, rns8_semantics semantics);
bool known_backend_kind(rns8_backend_kind backend);
const char* backend_name(rns8_backend_kind backend);
bool accelerator_backend(rns8_backend_kind backend);
uint32_t direct_hip_compiled();
uint32_t hipblaslt_backend_compiled();
bool hip_resident_rns_backend(rns8_backend_kind backend);
bool context_accepts_backend(const rns8_context& ctx, rns8_backend_kind backend);
bool matrix_backend_compatible_with_plan(
    const rns8_context& ctx,
    const rns8_matrix& matrix,
    rns8_backend_kind plan_backend);
void set_text(char* dst, std::size_t dst_size, const char* text);
void set_text(char* dst, std::size_t dst_size, const std::string& text);
void fill_backend_capability_info(rns8_backend_kind backend, rns8_backend_capability_info& info);

bool uses_rns_storage(rns8_semantics semantics);
bool uses_finite_storage(rns8_semantics semantics);
rns8_matrix_desc make_matrix_desc(
    int64_t rows,
    int64_t cols,
    rns8_semantics semantics,
    rns8_bound_kind bound_kind,
    uint32_t prefix,
    uint32_t tile_m = 128,
    uint32_t tile_n = 128);
bool valid_matrix_access(int64_t rows, int64_t cols, int64_t ld);
bool valid_api_tile_size(uint32_t value);
bool finite_backend_supports(rns8_backend_kind backend);
rns8_status validate_finite_u8_oneshot_contract(
    const rns8_context& ctx,
    const rns8_gemm_desc& desc,
    rns8_semantics expected_semantics,
    uint16_t modulus,
    int64_t lda,
    int64_t ldb,
    int64_t ldc);
bool valid_limb_export_access(int64_t rows, int64_t cols, int64_t ld, uint32_t limb_count);
rns8_status validate_typed_oneshot_contract(
    const rns8_context& ctx,
    const rns8_gemm_desc& desc,
    rns8_semantics expected_semantics,
    int64_t lda,
    int64_t ldb,
    int64_t ldc);

struct resident_oneshot_state {
  rns8_plan* plan = nullptr;
  rns8_matrix* A = nullptr;
  rns8_matrix* B = nullptr;
  rns8_matrix* C = nullptr;
  rns8_workspace* workspace = nullptr;

  resident_oneshot_state() = default;
  resident_oneshot_state(const resident_oneshot_state&) = delete;
  resident_oneshot_state& operator=(const resident_oneshot_state&) = delete;
  ~resident_oneshot_state();
};

rns8_status create_resident_oneshot_state(
    rns8_context* ctx,
    const rns8_gemm_desc& desc,
    resident_oneshot_state& state);
rns8_status finite_u8_oneshot_resident(
    rns8_context* ctx,
    const rns8_gemm_desc& desc,
    uint16_t modulus,
    const uint8_t* A,
    int64_t lda,
    const uint8_t* B,
    int64_t ldb,
    uint8_t* C,
    int64_t ldc);

uint64_t ceil_div_i64_u32(int64_t value, uint32_t divisor);
boost::multiprecision::cpp_int schedule_required_range(const rns8_gemm_desc& desc);
bool is_per_tile_bound_kind(rns8_bound_kind bound_kind);
boost::multiprecision::cpp_int bounded_range_from_bound(rns8_semantics semantics, uint64_t bound);
rns8_plan_tile_schedule_entry make_tile_schedule_entry(
    const rns8_plan& plan,
    uint64_t tile_row,
    uint64_t tile_col,
    int64_t row_offset,
    int64_t col_offset,
    int64_t row_extent,
    int64_t col_extent,
    uint32_t required_prefix,
    uint32_t selected_prefix,
    uint32_t group_index,
    uint32_t range_bit_length);
rns8_status configure_plan_schedule(rns8_plan& plan);
std::string selected_kernel_for_plan(const rns8_plan& plan);
std::string epilogue_mode_for_plan(const rns8_plan& plan);
std::string workspace_mode_for_plan(const rns8_plan& plan);
std::string isa_evidence_for_plan(const rns8_plan& plan);
uint64_t workspace_required_bytes_for_plan(const rns8_plan& plan);
bool accelerator_workspace_shape_for_plan(const rns8_plan& plan, int64_t& max_m, int64_t& max_n);
bool hipblaslt_pack_workspace_breakdown(
    const rns8_plan& plan,
    uint64_t& a_bytes,
    uint64_t& b_bytes,
    uint64_t& total_bytes);
bool ck_pack_workspace_breakdown(
    const rns8_plan& plan,
    uint64_t& a_bytes,
    uint64_t& b_bytes,
    uint64_t& total_bytes);
bool wmma_pack_workspace_breakdown(
    const rns8_plan& plan,
    uint64_t& a_bytes,
    uint64_t& b_bytes,
    uint64_t& total_bytes);
const char* persistent_layout_version_for_plan(const rns8_plan& plan);
bool hipblaslt_scratch_bytes_for_plan(const rns8_plan& plan, std::size_t& bytes);
bool hipblaslt_workspace_bytes_for_plan(const rns8_plan& plan, std::size_t& bytes);
std::string build_autotune_key(const rns8_plan& plan);
bool backend_library_version_matches_plan(
    const rns8_plan& plan,
    const rns8_backend_capability_info& capability);
rns8::detail::AutotuneRuntimeIdentity autotune_runtime_identity_for_plan(
    const rns8_plan& plan,
    const rns8_backend_capability_info& capability);
void configure_plan_backend_metadata(rns8_plan& plan, const rns8_context* ctx = nullptr);
bool prepare_auto_candidate_backend(rns8_context& ctx, rns8_backend_kind backend);
void select_auto_backend_from_reviewed_cache(rns8_context& ctx, rns8_plan& plan);
std::vector<int8_t> gather_cell_residues(
    const rns8_matrix& matrix,
    int64_t row,
    int64_t col,
    uint32_t prefix);
uint64_t tile_index_for_cell(const rns8_plan& plan, int64_t row, int64_t col);
const rns8_plan_tile_schedule_entry* tile_schedule_entry_for_cell(
    const rns8_plan& plan,
    int64_t row,
    int64_t col);
uint32_t selected_prefix_for_cell(const rns8_plan& plan, int64_t row, int64_t col);
uint64_t bound_for_cell(const rns8_plan& plan, int64_t row, int64_t col);
uint64_t workspace_fingerprint_mix(uint64_t hash, uint64_t value);
uint64_t workspace_fingerprint_mix_string(uint64_t hash, const std::string& value);
uint64_t signed_to_fingerprint(int64_t value);
uint64_t gemm_output_source_version(const rns8_matrix& A, const rns8_matrix& B);
uint64_t plan_workspace_fingerprint(const rns8_plan& plan);

bool matrix_descriptor_matches(
    const rns8_matrix& matrix,
    rns8_semantics semantics,
    rns8_bound_kind bound_kind,
    int64_t rows,
    int64_t cols,
    uint32_t prefix,
    uint32_t tile_m,
    uint32_t tile_n);
bool configured_tile_size_valid(uint32_t value);
bool wrap_byte_limb_bytes(int64_t rows, int64_t cols, std::size_t& bytes);
bool matrix_cell_count(int64_t rows, int64_t cols, std::size_t& cells);
bool rns_residue_count(int64_t rows, int64_t cols, uint32_t prefix, std::size_t& residues);
bool rns_matrix_storage_matches(
    const rns8_matrix& matrix,
    rns8_backend_kind backend,
    int64_t rows,
    int64_t cols,
    uint32_t prefix);
bool finite_matrix_storage_matches(
    const rns8_matrix& matrix,
    rns8_backend_kind backend,
    int64_t rows,
    int64_t cols);
bool rns_residue_state_current_for_backend(const rns8_matrix& matrix, rns8_backend_kind backend);
bool plan_schedule_contract_matches(const rns8_plan& plan);
bool wrap_matrix_storage_matches(
    const rns8_matrix& matrix,
    rns8_backend_kind backend,
    int64_t rows,
    int64_t cols);
bool wrap_byte_limb_state_current_for_backend(const rns8_matrix& matrix, rns8_backend_kind backend);
rns8_status validate_plan_context_workspace(
    const rns8_context& ctx,
    const rns8_plan& plan,
    const rns8_workspace& workspace);
rns8_status validate_rns_gemm_operands(
    const rns8_context& ctx,
    const rns8_plan& plan,
    const rns8_matrix& A,
    const rns8_matrix& B,
    const rns8_matrix& C);
rns8_status validate_finite_gemm_operands(
    const rns8_context& ctx,
    const rns8_plan& plan,
    uint16_t modulus,
    const rns8_matrix& A,
    const rns8_matrix& B,
    const rns8_matrix& C);
rns8_status validate_wrap_gemm_operands(
    const rns8_context& ctx,
    const rns8_plan& plan,
    const rns8_matrix& A,
    const rns8_matrix& B,
    const rns8_matrix& C);
rns8_status validate_export_matrix(
    const rns8_context& ctx,
    const rns8_plan& plan,
    const rns8_matrix& matrix,
    rns8_semantics expected_semantics,
    rns8_bound_kind expected_bound_kind,
    uint32_t expected_prefix);
rns8_status validate_finite_export_matrix(
    const rns8_context& ctx,
    const rns8_plan& plan,
    uint16_t modulus,
    const rns8_matrix& C);
rns8_status free_hip_matrix_storage(rns8_matrix& matrix);
rns8_status allocate_hip_matrix_storage(rns8_context& ctx, rns8_matrix& matrix);
rns8_status ensure_hip_export_tile_metadata(
    rns8_context& ctx,
    const rns8_plan& plan,
    rns8_matrix& matrix);

}  // namespace rns8::detail::api

#endif
