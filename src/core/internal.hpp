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
  int device_id = -1;
  rns8_device_info device_info{};
};

struct rns8_plan {
  rns8_gemm_desc desc{};
  uint32_t prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
  boost::multiprecision::cpp_int modulus_product = 0;
  rns8_backend_kind backend = RNS8_BACKEND_CPU_REFERENCE;
};

struct rns8_matrix {
  rns8_matrix_desc desc{};
  uint32_t prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
  uint64_t source_version = 0;
  std::vector<int8_t> residues;
};

struct rns8_workspace {
  int64_t m = 0;
  int64_t n = 0;
  int64_t k = 0;
  uint32_t prefix = 0;
};

namespace rns8::detail {

using boost::multiprecision::cpp_int;

constexpr uint16_t kDefaultModuli[RNS8_DEFAULT_MODULUS_COUNT] = {
    256, 255, 253, 251, 247, 239, 233, 229, 227, 223, 217, 211, 199, 197,
    193, 191, 181, 179, 173, 167, 163, 157, 151, 149, 139, 137, 131, 127};

bool valid_abi(uint64_t struct_size, uint32_t abi_version, std::size_t expected_size);
void fill_cpu_device_info(rns8_device_info& info);
void copy_c_string(char* dst, std::size_t dst_size, const std::string& src);

bool default_moduli_pairwise_coprime();
cpp_int modulus_product(uint32_t prefix);
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

cpp_int exact_i64_gemm_cell(const int64_t* A, int64_t lda, const int64_t* B, int64_t ldb, int64_t row, int64_t col, int64_t k);
cpp_int exact_u64_gemm_cell(const uint64_t* A, int64_t lda, const uint64_t* B, int64_t ldb, int64_t row, int64_t col, int64_t k);

}  // namespace rns8::detail

#endif

