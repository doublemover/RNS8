#ifndef RNS8_CORE_INTERNAL_HPP
#define RNS8_CORE_INTERNAL_HPP

#include <boost/multiprecision/cpp_int.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "rns8/rns8.h"

struct rns8_context {
  rns8_backend_kind backend = RNS8_BACKEND_CPU_REFERENCE;
  bool auto_backend_selection = false;
  int device_id = -1;
  rns8_device_info device_info{};
  void* hipblaslt_handle = nullptr;
  std::string hipblaslt_library_version;
};

struct rns8_plan {
  rns8_gemm_desc desc{};
  uint32_t prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
  boost::multiprecision::cpp_int modulus_product = 0;
  rns8_backend_kind backend = RNS8_BACKEND_CPU_REFERENCE;
  uint64_t schedule_tile_rows = 0;
  uint64_t schedule_tile_cols = 0;
  uint64_t schedule_tile_count = 0;
  uint32_t schedule_min_required_prefix = 0;
  uint32_t schedule_max_required_prefix = 0;
  uint32_t schedule_min_selected_prefix = 0;
  uint32_t schedule_max_selected_prefix = 0;
  uint32_t schedule_prefix_group_count = 0;
  uint32_t schedule_range_bit_length = 0;
  uint32_t schedule_adaptive_prefix_active = 0;
  uint32_t schedule_adaptive_skip_active = 0;
  uint32_t schedule_flags = 0;
  uint64_t backend_workspace_required_bytes = 0;
  std::string backend_selected_kernel;
  std::string backend_library;
  std::string backend_library_version;
  std::string backend_capability_status;
  std::string backend_epilogue_mode;
  std::string backend_workspace_mode;
  std::string backend_isa_evidence;
  std::string backend_autotune_key;
  uint32_t backend_performance_validated = 0;
  std::vector<uint64_t> tile_bounds;
  std::vector<rns8_plan_tile_schedule_entry> tile_schedule;
};

struct rns8_matrix {
  rns8_matrix_desc desc{};
  rns8_backend_kind backend = RNS8_BACKEND_CPU_REFERENCE;
  uint32_t prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
  uint16_t finite_modulus = 0;
  uint64_t source_version = 0;
  std::vector<int8_t> residues;
  std::vector<uint8_t> byte_limbs;
  std::vector<int64_t> native_i64;
  std::vector<uint64_t> native_u64;
  bool host_residues_current = true;
  bool device_residues_current = false;
  bool host_byte_limbs_current = false;
  bool device_byte_limbs_current = false;
  bool host_native_current = false;
  bool device_native_current = false;
  int hip_device_id = -1;
  void* hip_residues = nullptr;
  std::size_t hip_residue_bytes = 0;
  void* hip_byte_limbs = nullptr;
  std::size_t hip_byte_limb_bytes = 0;
  void* hip_native_i64 = nullptr;
  std::size_t hip_native_i64_bytes = 0;
  void* hip_native_u64 = nullptr;
  std::size_t hip_native_u64_bytes = 0;
  void* hip_upload_buffer = nullptr;
  std::size_t hip_upload_bytes = 0;
  void* hip_export_buffer = nullptr;
  std::size_t hip_export_bytes = 0;
  void* hip_status_buffer = nullptr;
  std::size_t hip_status_bytes = 0;
  void* hip_export_tile_schedule = nullptr;
  std::size_t hip_export_tile_schedule_bytes = 0;
  uint64_t hip_export_tile_schedule_count = 0;
  void* hip_export_tile_bounds = nullptr;
  std::size_t hip_export_tile_bounds_bytes = 0;
  uint64_t hip_export_tile_bounds_count = 0;
  uint64_t hip_export_schedule_fingerprint = 0;
  uint64_t hip_export_tile_max_elements = 0;
};

struct rns8_workspace {
  rns8_backend_kind backend = RNS8_BACKEND_CPU_REFERENCE;
  rns8_semantics semantics = RNS8_BOUNDED_I64;
  rns8_bound_kind bound_kind = RNS8_BOUND_NONE;
  int64_t m = 0;
  int64_t n = 0;
  int64_t k = 0;
  uint64_t bound = 0;
  uint32_t finite_modulus = 0;
  uint32_t tile_m = 0;
  uint32_t tile_n = 0;
  uint32_t prefix = 0;
  uint64_t schedule_tile_rows = 0;
  uint64_t schedule_tile_cols = 0;
  uint64_t schedule_tile_count = 0;
  uint32_t schedule_min_required_prefix = 0;
  uint32_t schedule_max_required_prefix = 0;
  uint32_t schedule_min_selected_prefix = 0;
  uint32_t schedule_max_selected_prefix = 0;
  uint32_t schedule_prefix_group_count = 0;
  uint32_t schedule_range_bit_length = 0;
  uint32_t schedule_adaptive_prefix_active = 0;
  uint32_t schedule_adaptive_skip_active = 0;
  uint32_t schedule_flags = 0;
  uint64_t schedule_fingerprint = 0;
  uint64_t backend_workspace_required_bytes = 0;
  std::string backend_selected_kernel;
  std::string backend_library;
  std::string backend_library_version;
  std::string backend_capability_status;
  std::string backend_epilogue_mode;
  std::string backend_workspace_mode;
  std::string backend_isa_evidence;
  std::string backend_autotune_key;
  uint32_t backend_performance_validated = 0;
  int hip_device_id = -1;
  void* hip_tile_schedule = nullptr;
  std::size_t hip_tile_schedule_bytes = 0;
  uint64_t hip_tile_schedule_count = 0;
  void* hipblaslt_int32_scratch = nullptr;
  std::size_t hipblaslt_int32_scratch_bytes = 0;
  void* hipblaslt_workspace = nullptr;
  std::size_t hipblaslt_workspace_bytes = 0;
  void* accelerator_workspace = nullptr;
  std::size_t accelerator_workspace_bytes = 0;
  void* accelerator_auxiliary = nullptr;
  std::size_t accelerator_auxiliary_bytes = 0;
};

struct rns8_prepack_cache {
  rns8_backend_kind backend = RNS8_BACKEND_CPU_REFERENCE;
  rns8_semantics semantics = RNS8_BOUNDED_I64;
  rns8_operand_role operand_role = RNS8_OPERAND_B;
  int64_t rows = 0;
  int64_t cols = 0;
  int64_t k = 0;
  uint32_t prefix = 0;
  uint32_t finite_modulus = 0;
  uint64_t source_version = 0;
  uint64_t plan_fingerprint = 0;
  uint64_t cache_key_hash = 0;
  std::string cache_key;
  std::string matrix_layout_version;
  std::string operand_layout_version;
  int hip_device_id = -1;
  void* device_data = nullptr;
  std::size_t device_bytes = 0;
  std::size_t operand_pack_bytes = 0;
};

namespace rns8::detail {

using boost::multiprecision::cpp_int;

constexpr uint16_t kDefaultModuli[RNS8_DEFAULT_MODULUS_COUNT] = {
    256, 255, 253, 251, 247, 239, 233, 229, 227, 223, 217, 211, 199, 197,
    193, 191, 181, 179, 173, 167, 163, 157, 151, 149, 139, 137, 131, 127};

bool valid_abi(uint64_t struct_size, uint32_t abi_version, std::size_t expected_size);
void fill_cpu_device_info(rns8_device_info& info);
void fill_wrap64_device_info(rns8_device_info& info);
void copy_c_string(char* dst, std::size_t dst_size, const std::string& src);

bool default_moduli_pairwise_coprime();
bool valid_finite_ring_modulus(uint32_t modulus);
bool valid_finite_field_modulus(uint32_t modulus);
bool valid_finite_modulus_for_semantics(rns8_semantics semantics, uint32_t modulus);
cpp_int modulus_product(uint32_t prefix);
uint32_t bit_length(const cpp_int& value);
uint32_t required_prefix_for_range(const cpp_int& range);
uint32_t default_prefix_for_semantics(rns8_semantics semantics);
rns8_status validate_gemm_desc(const rns8_gemm_desc& desc, uint32_t prefix);
rns8_status validate_matrix_desc(const rns8_matrix_desc& desc, uint32_t prefix);
rns8_status validate_bound_contract(
    rns8_semantics semantics,
    rns8_bound_kind bound_kind,
    uint64_t bound,
    uint32_t prefix);

uint32_t canonical_residue(const cpp_int& value, uint16_t modulus);
uint32_t canonical_from_centered(int8_t residue, uint16_t modulus);
int8_t centered_residue(const cpp_int& value, uint16_t modulus);
int8_t reduce_to_centered(int64_t value, uint16_t modulus);

std::size_t residue_index(const rns8_matrix& matrix, uint32_t modulus_index, int64_t row, int64_t col);
void pack_i64_matrix(rns8_matrix& matrix, const int64_t* src, int64_t ld);
void pack_u64_matrix(rns8_matrix& matrix, const uint64_t* src, int64_t ld);
void pack_finite_u8_matrix(rns8_matrix& matrix, const uint8_t* src, int64_t ld, uint16_t modulus);
void export_finite_u8_matrix(const rns8_matrix& matrix, uint8_t* dst, int64_t ld, uint16_t modulus);
void pack_wrap_u64_matrix(rns8_matrix& matrix, const uint64_t* src, int64_t ld);
uint64_t wrap_u64_matrix_cell(const rns8_matrix& matrix, int64_t row, int64_t col);
void set_wrap_u64_matrix_cell(rns8_matrix& matrix, int64_t row, int64_t col, uint64_t value);

void ring_gemm_modulus(
    const int8_t* A,
    const int8_t* B,
    int8_t* C,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    uint16_t modulus);

rns8_status cpu_gemm_rns(const rns8_plan& plan, const rns8_matrix& A, const rns8_matrix& B, rns8_matrix& C);
rns8_status cpu_gemm_finite_u8(
    const rns8_plan& plan,
    uint16_t modulus,
    const rns8_matrix& A,
    const rns8_matrix& B,
    rns8_matrix& C);
rns8_status cpu_gemm_wrap_u64(const rns8_plan& plan, const rns8_matrix& A, const rns8_matrix& B, rns8_matrix& C);

rns8_status reconstruct_unsigned(
    const std::vector<int8_t>& residues,
    uint32_t prefix,
    uint64_t bound,
    uint64_t& out);

rns8_status reconstruct_signed(
    const std::vector<int8_t>& residues,
    uint32_t prefix,
    uint64_t bound,
    int64_t& out);
cpp_int reconstruct_canonical(const std::vector<int8_t>& residues, uint32_t prefix);
rns8_status export_exact_wide_unsigned_limbs(
    const std::vector<int8_t>& residues,
    uint32_t prefix,
    uint64_t* out,
    uint32_t limb_count);
rns8_status export_exact_wide_signed_limbs(
    const std::vector<int8_t>& residues,
    uint32_t prefix,
    uint64_t* out,
    uint32_t limb_count);

cpp_int exact_i64_gemm_cell(const int64_t* A, int64_t lda, const int64_t* B, int64_t ldb, int64_t row, int64_t col, int64_t k);
cpp_int exact_u64_gemm_cell(const uint64_t* A, int64_t lda, const uint64_t* B, int64_t ldb, int64_t row, int64_t col, int64_t k);
int32_t wrap64_signed_i8_lane_value(uint8_t value);
int32_t wrap64_signed_i8_product_correction(uint8_t a, uint8_t b);
uint32_t wrap64_unsigned_byte_product_from_signed_i8(uint8_t a, uint8_t b);
uint64_t wrap64_byte_limb_product(uint64_t a, uint64_t b);
uint64_t wrap64_low_diagonal_byte_pair_gemm_cell(
    const uint64_t* A,
    int64_t lda,
    const uint64_t* B,
    int64_t ldb,
    int64_t row,
    int64_t col,
    int64_t k);
uint64_t wrap64_byte_limb_gemm_cell(
    const uint64_t* A,
    int64_t lda,
    const uint64_t* B,
    int64_t ldb,
    int64_t row,
    int64_t col,
    int64_t k);

}  // namespace rns8::detail

#endif
