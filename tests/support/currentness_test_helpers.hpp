#pragma once

#include "core/internal.hpp"

namespace rns8::test {

inline void set_host_residues_current(rns8_matrix& matrix, bool value) {
  matrix.host_residues_current = value;
}

inline void set_device_residues_current(rns8_matrix& matrix, bool value) {
  matrix.device_residues_current = value;
}

inline void set_host_byte_limbs_current(rns8_matrix& matrix, bool value) {
  matrix.host_byte_limbs_current = value;
}

inline void set_device_byte_limbs_current(rns8_matrix& matrix, bool value) {
  matrix.device_byte_limbs_current = value;
}

inline void set_host_native_current(rns8_matrix& matrix, bool value) {
  matrix.host_native_current = value;
}

inline void set_device_native_current(rns8_matrix& matrix, bool value) {
  matrix.device_native_current = value;
}

}  // namespace rns8::test
