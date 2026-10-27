#pragma once
#include "core/internal.hpp"
#include "backend_common/basis_packet.hpp"
#include <boost/multiprecision/cpp_int.hpp>
#include <array>
#include <memory>
namespace rns8::detail {
struct ExperimentalBasis {
  BasisPacket packet;
  std::array<boost::multiprecision::cpp_int,21> products;
};
std::shared_ptr<const ExperimentalBasis> default_experimental_basis();
inline std::shared_ptr<const ExperimentalBasis> stored_basis(const rns8_matrix& matrix) {
  return matrix.experimental_basis?matrix.experimental_basis:default_experimental_basis();
}
inline bool basis_storage_matches(const rns8_matrix& matrix,rns8_backend_kind backend) {
  const auto data=stored_basis(matrix);AddressDomain span;
  if(matrix.backend!=backend || !matrix.prefix || matrix.prefix>data->packet.count ||
      matrix.desc.max_prefix!=matrix.prefix || matrix.desc.logical_ld<matrix.desc.cols ||
      (matrix.desc.flags&~uint32_t(RNS8_MATRIX_RETAIN_NATIVE_INPUT)) ||
      (matrix.desc.flags && (backend!=RNS8_BACKEND_HIP_DIRECT || matrix.experimental_basis)) ||
      matrix.materialized_rns_schedule || matrix.desc.bound_kind!=RNS8_BOUND_NONE ||
      matrix.desc.logical_layout!=RNS8_LAYOUT_ROW_MAJOR ||
      !checked_host_plane_address(matrix.desc.rows,matrix.desc.cols,matrix.desc.cols,matrix.prefix,1,span) ||
      !matrix.byte_limbs.empty() || matrix.hip_byte_limbs || matrix.hip_byte_limb_bytes ||
      matrix.host_byte_limbs_current || matrix.device_byte_limbs_current)return false;
  return backend==RNS8_BACKEND_HIP_DIRECT?
      matrix.hip_residues && matrix.hip_residue_bytes==span.byte_extent &&
        (matrix.residues.size()==span.byte_extent || (matrix.residues.empty() && !matrix.host_residues_current)):
      backend==RNS8_BACKEND_CPU_REFERENCE && matrix.residues.size()==span.byte_extent &&
        !matrix.hip_residues && !matrix.hip_residue_bytes && !matrix.device_residues_current;
}
}
