#ifndef RNS8_CORE_API_INTERNAL_HPP
#define RNS8_CORE_API_INTERNAL_HPP

#include "core/internal.hpp"
#include "core/exact_range.hpp"
#include "backend_common/bilinear_selection.hpp"

#include <boost/multiprecision/cpp_int.hpp>

#include <algorithm>
#include <cstddef>
#include <chrono>
#include <thread>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <type_traits>
#include <vector>

#include "backend_ck/ck_backend.hpp"
#include "backend_amdgpu_builtins/amdgpu_builtins_backend.hpp"
#include "backend_common/finite_u8_reducer.hpp"
#include "backend_hip_direct/hip_backend.hpp"
#include "backend_hipblaslt/hipblaslt_backend.hpp"
#include "backend_vector_alu/vector_alu_backend.hpp"
#include "backend_rocwmma/rocwmma_backend.hpp"
#include "backend_wrap64/wrap64_hip.hpp"
#include "core/accelerator_backend.hpp"
#include "core/autotune_cache.hpp"
#include "core/backend_common.hpp"

namespace rns8::detail::api {
// Preparation-only envelope: runtime publication still requires the actual
// producer token and its checked pending range. No fictitious matrix currentness.
rns8_status prepare_bound_host_export_hip(rns8_context*,rns8_matrix*,rns8_pinned_host_buffer*,
    uint32_t,const boost::multiprecision::cpp_int&,uint64_t,rns8_prepared_execution**);
// Discovery and the concrete resident-graph owner must admit the same backend
// families. Dense/synchronous GEMM support alone does not provide a DAG owner.
inline bool resident_graph_backend_supported(rns8_backend_kind backend) {
  return backend==RNS8_BACKEND_CPU_REFERENCE || backend==RNS8_BACKEND_HIP_DIRECT ||
      backend==RNS8_BACKEND_WRAP64_BYTE_LIMB;
}
inline bool checked_crt_backend_supported(rns8_backend_kind backend) {
  return backend==RNS8_BACKEND_CPU_REFERENCE || backend==RNS8_BACKEND_HIP_DIRECT;
}
// Quarter export already consumes ordinary uniform RNS storage, independently
// of the GEMM producer. This source predicate must not disappear with compile
// flags, nor claim native-vector or wrap-byte storage is a residue owner.
inline bool quarter_crt_backend_supported(rns8_backend_kind backend) {
  return checked_crt_backend_supported(backend) || backend==RNS8_BACKEND_CPU_AMX ||
      backend==RNS8_BACKEND_HIPBLASLT || backend==RNS8_BACKEND_CK ||
      backend==RNS8_BACKEND_ROCWMMA || backend==RNS8_BACKEND_AMDGPU_BUILTINS;
}
rns8_operation_support operation_support(const rns8_context&,const rns8_operation_query&,rns8_backend_kind);
rns8_status begin_synchronous_lease(rns8_context&,const rns8_matrix&,const rns8_matrix&,rns8_matrix&,rns8_workspace&,size_t&,bool input_writers=false);
bool finish_synchronous_lease(rns8_context&,size_t,rns8_status,bool may_have_written) noexcept;
rns8_status begin_synchronous_matrix_lease(rns8_context&,rns8_matrix&,rns8_workspace&,size_t&,bool write);
rns8_status plan_operation_preflight(const rns8_context&,const rns8_plan&,uint32_t operation,uint32_t modes,uint32_t proof_flags=0);

template<class Matrix> struct ComplexMatrixPair {
  Matrix* real = nullptr;
  Matrix* imag = nullptr;
};
// Internal synchronous paired-value owner. Uses matrix-owned exact-wide signed
// proofs and existing reader/writer generations; backs the experimental public API.
rns8_status complex_exact_signed_gemm(
    rns8_context* ctx, const rns8_plan* plan,
    ComplexMatrixPair<const rns8_matrix> a, ComplexMatrixPair<const rns8_matrix> b,
    ComplexMatrixPair<rns8_matrix> c, rns8::detail::ComplexProductMethod method,
    uint64_t max_scratch_bytes, rns8_workspace* workspace);

struct ResourceBusy {};

inline rns8_async_support_info async_support_info(rns8_status status, int device) noexcept {
  rns8_async_support_info info{};
  info.struct_size = sizeof(info);
  info.abi_version = RNS8_ABI_VERSION;
  info.support_status = status;
  info.device_id = device;
  if (status == RNS8_SUCCESS) {
    info.max_in_flight_epochs = 1;
    info.requires_borrowed_nondefault_stream = 1;
    info.submission_may_allocate_host_proof = 1;
    info.supports_owned_graph = 1;
    info.supports_cooperative_discard = 1;
    info.supports_timed_observation = 1;
  }
  return info;
}

inline void record_async_failure(rns8_async_failure_info& failure,
                                 rns8_status status, rns8_async_failure_stage stage,
                                 int backend_error, int device, uint64_t epoch,
                                 uint32_t node = UINT32_MAX) noexcept {
  if (failure.stage != RNS8_ASYNC_FAILURE_NONE) return;
  failure.struct_size = sizeof(failure);
  failure.abi_version = RNS8_ABI_VERSION;
  failure.stage = stage;
  failure.operation_status = status;
  failure.backend_error = backend_error;
  failure.device_id = device;
  failure.epoch = epoch;
  failure.failed_node = node;
  failure.resources_retained = 1;
}

inline bool record_async_drain(rns8_context& ctx, rns8_async_failure_info& failure,
                               int cleanup_error) noexcept {
  if (cleanup_error != 0) {
    if (failure.cleanup_error == 0) failure.cleanup_error = cleanup_error;
    ctx.hip_failure_poisoned = true;
    failure.resources_retained = 1;
    return false;
  }
  failure.resources_retained = 0;
  return true;
}
// A timeout is an observation result, never an ownership transition. Each
// poll is nonblocking; kernel errors are left for explicit blocking cleanup.
template<class Poll>
rns8_status poll_async_until(uint64_t timeout_us, Poll&& poll) {
  const auto start = std::chrono::steady_clock::now();
  for (;;) {
    const auto status = poll();
    if (status != RNS8_BUSY) return status;
    const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - start).count();
    if (elapsed >= 0 && static_cast<uint64_t>(elapsed) >= timeout_us) return RNS8_TIMEOUT;
    // Sleeping limits host overhead; OS scheduling can overshoot the deadline.
    const auto remaining = timeout_us - static_cast<uint64_t>(std::max<int64_t>(elapsed, 0));
    std::this_thread::sleep_for(std::chrono::microseconds(std::min<uint64_t>(remaining, 50)));
  }
}

inline void require_matrix_writable(const rns8_matrix& matrix) {
  if (matrix.in_flight_readers || matrix.in_flight_writer) throw ResourceBusy{};
}

template <typename Fn>
rns8_status guard_api(Fn&& fn) {
  try {
    return fn();
  } catch (const ResourceBusy&) {
    return RNS8_BUSY;
  } catch (const std::bad_alloc&) {
    return RNS8_INTERNAL_ERROR;
  } catch (...) {
    return RNS8_INTERNAL_ERROR;
  }
}

template <typename Fn>
rns8_status run_timed_api_status(const char* label, Fn&& fn) {
  rns8_status status = RNS8_SUCCESS;
  const int code = rns8::detail::run_timed_device_code(label, [&]() {
    status = fn();
    return status == RNS8_SUCCESS ? 0 : 3;
  });
  if (code != 0 && status == RNS8_SUCCESS) {
    return RNS8_BACKEND_FAILURE;
  }
  return status;
}

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

rns8_backend_kind effective_backend(rns8_backend_kind requested, rns8_backend_kind default_backend);

inline bool host_rns_backend(rns8_backend_kind backend) {
  return backend==RNS8_BACKEND_CPU_REFERENCE || backend==RNS8_BACKEND_CPU_AMX;
}
bool backend_source_supports_semantics(rns8_backend_kind backend, rns8_semantics semantics);
bool backend_supports_semantics(rns8_backend_kind backend, rns8_semantics semantics);

bool known_backend_kind(rns8_backend_kind backend);

const char* backend_name(rns8_backend_kind backend);

bool accelerator_backend(rns8_backend_kind backend);

uint32_t direct_hip_compiled();

uint32_t hipblaslt_backend_compiled();

bool hip_resident_rns_backend(rns8_backend_kind backend);

bool native_vector_backend(rns8_backend_kind backend);

bool hip_device_backend(rns8_backend_kind backend);
bool hip_device_backend_kind(rns8_backend_kind backend);

bool context_accepts_backend(const rns8_context& ctx, rns8_backend_kind backend);

bool matrix_backend_compatible_with_plan(
    const rns8_context& ctx,
    const rns8_matrix& matrix,
    rns8_backend_kind plan_backend, bool allow_pending_writer = false);

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

bool valid_matrix_access(int64_t rows, int64_t cols, int64_t ld, uint64_t element_bytes = sizeof(uint64_t));

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

bool valid_limb_export_access(int64_t rows, int64_t cols, int64_t ld, uint32_t limb_count);

rns8_status validate_typed_oneshot_contract(
    const rns8_context& ctx,
    const rns8_gemm_desc& desc,
    rns8_semantics expected_semantics,
    int64_t lda,
    int64_t ldb,
    int64_t ldc);

uint64_t ceil_div_i64_u32(int64_t value, uint32_t divisor);

boost::multiprecision::cpp_int schedule_required_range(const rns8_gemm_desc& desc);

bool is_per_tile_bound_kind(rns8_bound_kind bound_kind);

bool is_input_range_bound_kind(rns8_bound_kind bound_kind);

boost::multiprecision::cpp_int bounded_range_from_bound(rns8_semantics semantics, uint64_t bound);

rns8_bound_kind storage_bound_kind_for_plan(const rns8_plan& plan);

uint32_t rns_storage_prefix_for_plan(const rns8_plan& plan);

rns8_status configure_plan_schedule(rns8_plan& plan);

const char* semantics_name_for_key(rns8_semantics semantics);

std::string selected_kernel_for_plan(const rns8_plan& plan);

std::string epilogue_mode_for_plan(const rns8_plan& plan);

std::string workspace_mode_for_plan(const rns8_plan& plan);

std::string isa_evidence_for_plan(const rns8_plan& plan);

uint64_t accumulator_k_block_size_for_plan(const rns8_plan& plan);

uint64_t accumulator_k_block_cap_for_plan(const rns8_plan& plan);

std::string accumulator_type_for_plan(const rns8_plan& plan);

std::string accumulator_signedness_for_plan(const rns8_plan& plan);

std::string accumulator_modulus_policy_for_plan(const rns8_plan& plan);

std::string accumulator_safety_status_for_plan(const rns8_plan& plan);

uint64_t workspace_required_bytes_for_plan(const rns8_plan& plan);

bool hip_direct_gemm_requires_device_tile_schedule(const rns8_plan& plan);

bool plan_all_zero_output_tiles(const rns8_plan& plan);

enum class export_output_layout {
  scalar_i64,
  scalar_u64,
  finite_u8,
  fixed_u64_limbs,
};

enum class export_status_policy {
  none,
  range_checked_status_buffer,
};

enum class export_d2h_policy {
  host_ld_padded,
  compact_contiguous,
  device_residue_current,
};

struct export_reconstruction_plan {
  export_output_layout output_layout = export_output_layout::scalar_i64;
  uint32_t limb_count = 0;
  export_status_policy status_policy = export_status_policy::none;
  export_d2h_policy d2h_policy = export_d2h_policy::host_ld_padded;
  const char* selected_export_kernel = "cpu_reference_export";
  std::string selector_key;
  std::string selector_policy = "default_export_selector";
  std::string semantic_contract;
  std::string backend;
  std::string target_id;
  std::string prefix_contract;
  std::string signedness;
  std::string final_output_mode = "final_host_output";
  std::string cache_visibility = "exact_shape_selector_metadata_only";
  std::string stale_entry_reason;
  std::string status_elision_reason;
  bool requires_hip_tile_metadata = false;
  bool all_zero_tiled_output = false;
};

const char* export_output_layout_name(export_output_layout layout);

const char* export_status_policy_name(export_status_policy policy);

const char* export_d2h_policy_name(export_d2h_policy policy);

export_reconstruction_plan make_export_plan(
    const rns8_plan& plan,
    rns8_semantics semantics,
    uint32_t limb_count = 0);

bool accelerator_workspace_shape_for_plan(const rns8_plan& plan, int64_t& max_m, int64_t& max_n);

bool hipblaslt_pack_workspace_breakdown(
    const rns8_plan& plan,
    uint64_t& a_pack_bytes,
    uint64_t& b_pack_bytes,
    uint64_t& accumulator_bytes,
    uint64_t& library_workspace_bytes,
    uint64_t& total_bytes);

bool ck_pack_workspace_breakdown(
    const rns8_plan& plan,
    uint64_t& a_pack_bytes,
    uint64_t& b_pack_bytes,
    uint64_t& accumulator_bytes,
    uint64_t& total_bytes);

bool rocwmma_pack_workspace_breakdown(
    const rns8_plan& plan,
    uint64_t& a_pack_bytes,
    uint64_t& b_pack_bytes,
    uint64_t& total_bytes);

const char* persistent_layout_version_for_semantics(rns8_semantics semantics);

const char* native_layout_version_for_semantics(rns8_semantics semantics);

const char* persistent_layout_version_for_plan(const rns8_plan& plan);

const char* storage_scope_for_matrix(const rns8_matrix& matrix);

const char* storage_detail_for_matrix(const rns8_matrix& matrix);

bool hipblaslt_scratch_bytes_for_plan(const rns8_plan& plan, std::size_t& bytes);

bool hipblaslt_workspace_bytes_for_plan(const rns8_plan& plan, std::size_t& bytes);

std::string build_autotune_key(const rns8_plan& plan);

bool backend_library_version_matches_plan(const rns8_plan& plan, const rns8_backend_capability_info& capability);

rns8::detail::AutotuneRuntimeIdentity autotune_runtime_identity_for_plan(
    const rns8_context& ctx,
    const rns8_plan& plan);

void configure_plan_backend_metadata(rns8_plan& plan, const rns8_context* ctx = nullptr);

bool prepare_auto_candidate_backend(rns8_context& ctx, rns8_backend_kind backend);

void select_auto_backend_from_reviewed_cache(rns8_context& ctx, rns8_plan& plan);

std::vector<int8_t> gather_cell_residues(const rns8_matrix& matrix, int64_t row, int64_t col, uint32_t prefix);

uint64_t tile_index_for_cell(const rns8_plan& plan, int64_t row, int64_t col);

const rns8_plan_tile_schedule_entry* tile_schedule_entry_for_cell(const rns8_plan& plan, int64_t row, int64_t col);

uint32_t selected_prefix_for_cell(const rns8_plan& plan, int64_t row, int64_t col);

uint64_t bound_for_cell(const rns8_plan& plan, int64_t row, int64_t col);

uint64_t workspace_fingerprint_mix(uint64_t hash, uint64_t value);

uint64_t workspace_fingerprint_mix_string(uint64_t hash, const std::string& value);

uint64_t signed_to_fingerprint(int64_t value);

uint64_t plan_workspace_fingerprint(const rns8_plan& plan);
bool seal_plan_contract(rns8_plan& plan);
bool revalidate_plan_schedule_contract(const rns8_plan& plan);
struct plan_validation_counters {
  uint64_t exhaustive_validations = 0;
  uint64_t fingerprint_traversals = 0;
  uint64_t equivalent_comparisons = 0;
  uint64_t backend_schedule_seals = 0;
  uint64_t backend_schedule_sealed_tile_visits = 0;
  uint64_t backend_schedule_fast_submissions = 0;
  uint64_t backend_schedule_raw_submissions = 0;
};
inline thread_local plan_validation_counters plan_validation_work{};

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

bool native_matrix_bytes(int64_t rows, int64_t cols, std::size_t element_size, std::size_t& bytes);

bool bounded_native_storage_matches(
    const rns8_matrix& matrix,
    rns8_semantics semantics,
    int64_t rows,
    int64_t cols);

bool bounded_native_state_current(const rns8_matrix& matrix);

bool matrix_has_native_storage(const rns8_matrix& matrix);

void clear_native_current(rns8_matrix& matrix);

void clear_device_native_current(rns8_matrix& matrix);

void clear_residue_current(rns8_matrix& matrix);

void clear_device_residues_current(rns8_matrix& matrix);

void clear_byte_limb_current(rns8_matrix& matrix);

void clear_device_byte_limb_current(rns8_matrix& matrix);

rns8_status import_exact_wide_limbs(rns8_context& ctx, rns8_matrix& matrix, const uint64_t* src,
                                    int64_t ld, uint32_t limb_count, bool signed_values, uint64_t source_version);

void publish_matrix_value(rns8_matrix& matrix, bool imported = false, uint64_t source_version = 0);

// Invalidates on unwinding or any failed post-write status, including native uploads.
struct MatrixWriteGuard {
  rns8_matrix& matrix;
  bool committed = false;
  explicit MatrixWriteGuard(rns8_matrix& value) : matrix(value) { invalidate(); }
  void invalidate();
  ~MatrixWriteGuard() { if (!committed) invalidate(); }
};

void invalidate_output_currentness(rns8_matrix& matrix);

// Internal resident-cache state transitions; success is published by the
// incremental dispatcher only after the snapshot and binding are complete.
rns8_status free_result_cache_snapshot(rns8_result_cache& cache);
rns8_status snapshot_result_cache_from_output(rns8_context& ctx, rns8_result_cache& cache, const rns8_matrix& C);
rns8_status restore_result_cache_to_output(rns8_context& ctx, rns8_result_cache& cache, rns8_matrix& C);

bool bounded_residue_semantics(rns8_semantics semantics);
bool materialized_rns_prefix_covers(const rns8_matrix& matrix, uint32_t prefix);
bool materialized_rns_export_covers(const rns8_matrix& matrix, const rns8_plan& plan);
void record_materialized_rns_output(rns8_matrix& matrix, const rns8_plan* plan);
void mark_host_residues_current(rns8_matrix& matrix, uint32_t written_prefix = UINT32_MAX);

void mark_device_residues_current(rns8_matrix& matrix, uint32_t written_prefix = UINT32_MAX);

void mark_host_byte_limbs_current(rns8_matrix& matrix);

void mark_device_byte_limbs_current(rns8_matrix& matrix);

void mark_device_native_current(rns8_matrix& matrix);

void mark_output_host_residues_current(rns8_matrix& matrix, const rns8_plan* plan = nullptr);

void mark_output_device_residues_current(rns8_matrix& matrix, const rns8_plan* plan = nullptr);

void mark_output_host_byte_limbs_current(rns8_matrix& matrix);

void mark_output_device_byte_limbs_current(rns8_matrix& matrix);

void mark_output_device_native_current(rns8_matrix& matrix);

bool rns_residue_count(int64_t rows, int64_t cols, uint32_t prefix, std::size_t& residues);

bool rns_matrix_storage_matches(const rns8_matrix& matrix, rns8_backend_kind backend, int64_t rows, int64_t cols, uint32_t prefix);

bool finite_matrix_storage_matches(const rns8_matrix& matrix, rns8_backend_kind backend, int64_t rows, int64_t cols);

bool rns_residue_state_current_for_backend(const rns8_matrix& matrix, rns8_backend_kind backend);

bool plan_schedule_contract_matches(const rns8_plan& plan);

bool wrap_matrix_storage_matches(const rns8_matrix& matrix, rns8_backend_kind backend, int64_t rows, int64_t cols);

bool wrap_byte_limb_state_current_for_backend(const rns8_matrix& matrix, rns8_backend_kind backend);

bool valid_prepack_operand_role(rns8_operand_role operand_role);

const char* operand_role_name(rns8_operand_role operand_role);

bool prepack_operand_shape_for_plan(
    const rns8_plan& plan,
    rns8_operand_role operand_role,
    int64_t& rows,
    int64_t& cols);

bool matrix_backend_can_feed_plan(const rns8_matrix& matrix, const rns8_plan& plan);

const char* prepack_operand_layout_version_for_plan(const rns8_plan& plan, rns8_operand_role operand_role);

const char* rocwmma_b_prepack_kernel_variant();

uint64_t prepack_prefix_schedule_fingerprint(const rns8_plan& plan);

uint64_t rocwmma_b_prepack_k_block_cap();

uint64_t rocwmma_b_prepack_k_block_size(const rns8_plan& plan);

std::string prepack_target_id_for_device(int hip_device_id);

std::string prepack_target_id_for_context(const rns8_context& ctx);

bool rocwmma_b_prepack_cache_supported(const rns8_plan& plan);

bool rocwmma_b_prepack_cache_production_supported(const rns8_plan& plan);

bool rocwmma_b_prepack_cache_production_supported(const rns8_prepack_cache& cache);

bool prepack_operand_matrix_compatible(
    const rns8_plan& plan,
    const rns8_matrix& matrix,
    rns8_operand_role operand_role);

uint64_t prepack_cache_key_hash(
    const rns8_plan& plan,
    const rns8_matrix& matrix,
    rns8_operand_role operand_role,
    const std::string& matrix_layout_version,
    const std::string& operand_layout_version);

std::string build_prepack_cache_key(
    const rns8_plan& plan,
    const rns8_matrix& matrix,
    rns8_operand_role operand_role,
    uint64_t plan_fingerprint,
    uint64_t cache_key_hash,
    const std::string& matrix_layout_version,
    const std::string& operand_layout_version);

bool rocwmma_b_prepack_bytes_for_plan(const rns8_plan& plan, std::size_t& b_pack_bytes, std::size_t& total_cache_bytes);

bool cache_key_contains_field(const std::string& key, const std::string& field, const std::string& value);

bool prepack_cache_key_self_consistent(const rns8_prepack_cache& cache);

bool prepack_cache_matches_plan(const rns8_prepack_cache& cache, const rns8_plan& plan);

rns8_status validate_rns_gemm_prepacked_b_operands(
    const rns8_context& ctx,
    const rns8_plan& plan,
    const rns8_matrix& A,
    const rns8_prepack_cache& B,
    const rns8_matrix& C);

rns8_status validate_plan_context_workspace(
    const rns8_context& ctx,
    const rns8_plan& plan,
    const rns8_workspace& workspace);

rns8_status validate_rns_gemm_operands(
    const rns8_context& ctx,
    const rns8_plan& plan,
    const rns8_matrix& A,
    const rns8_matrix& B,
    const rns8_matrix& C,
    bool require_current_inputs = true, bool allow_pending_input_writers = false);

rns8_status validate_exact_axpby_operands(
    const rns8_context& ctx, const rns8_plan& plan,
    const rns8_matrix& A, const rns8_matrix& B, const rns8_matrix& D,
    const rns8_matrix& C, int64_t alpha, int64_t beta,
    bool require_current_inputs = true, bool allow_pending_input_writers = false);

rns8_status validate_finite_gemm_operands(
    const rns8_context& ctx,
    const rns8_plan& plan,
    uint16_t modulus,
    const rns8_matrix& A,
    const rns8_matrix& B,
    const rns8_matrix& C, bool require_current_inputs = true, bool allow_pending_input_writers = false);

rns8_status validate_wrap_gemm_operands(
    const rns8_context& ctx,
    const rns8_plan& plan,
    const rns8_matrix& A,
    const rns8_matrix& B,
    const rns8_matrix& C,
    bool require_current_inputs = true, bool allow_pending_input_writers = false);

rns8_status validate_export_matrix(
    const rns8_context& ctx,
    const rns8_plan& plan,
    const rns8_matrix& C,
    rns8_semantics semantics,
    rns8_bound_kind bound_kind,
    uint32_t prefix,
    bool require_current = true);

rns8_status validate_finite_export_matrix(
    const rns8_context& ctx,
    const rns8_plan& plan,
    uint16_t modulus,
    const rns8_matrix& C);

rns8_status free_hip_matrix_storage(rns8_matrix& matrix);
rns8_status drain_failed_ragged_cleanup(rns8_context& ctx);
rns8_status drain_failed_native_cleanup(rns8_context& ctx);

rns8_status allocate_hip_matrix_storage(rns8_context& ctx, rns8_matrix& matrix);

rns8_status ensure_hip_native_storage(rns8_context& ctx, rns8_matrix& matrix);

rns8_status upload_native_i64(rns8_context& ctx, rns8_matrix& matrix, const int64_t* src, int64_t ld);

rns8_status upload_native_u64(rns8_context& ctx, rns8_matrix& matrix, const uint64_t* src, int64_t ld);

bool should_populate_native_on_pack(const rns8_context& ctx, const rns8_matrix& matrix);

rns8_status ensure_bounded_native_residues_current_for_rns_plan(
    rns8_context& ctx,
    const rns8_plan& plan,
    rns8_matrix& matrix);

bool signed_value_within_bound(int64_t value, uint64_t bound);

rns8_status export_native_i64(
    rns8_context& ctx,
    const rns8_plan& plan,
    const rns8_matrix& C,
    int64_t* dst,
    int64_t ld);

rns8_status export_native_u64(
    rns8_context& ctx,
    const rns8_plan& plan,
    const rns8_matrix& C,
    uint64_t* dst,
    int64_t ld);

rns8_status ensure_hip_export_tile_metadata(rns8_context& ctx, const rns8_plan& plan, rns8_matrix& matrix);

}  // namespace rns8::detail::api

#endif
