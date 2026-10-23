#pragma once
#include <cstdint>
namespace rns8::detail {
struct IntegrityIdentity {
  uint64_t operation = 0, epoch = 0, a_instance = 0, a_generation = 0,
           b_instance = 0, b_generation = 0;
  uint32_t basis_version = 1, prefix = 0;
};
inline bool integrity_identity_matches(const IntegrityIdentity &a,
                                       const IntegrityIdentity &b) {
  return a.operation && a.epoch && a.a_instance && a.b_instance &&
         a.epoch != UINT64_MAX && a.a_generation != UINT64_MAX &&
         a.b_generation != UINT64_MAX && a.basis_version == 1 && a.prefix &&
         a.prefix <= 20 && a.operation == b.operation && a.epoch == b.epoch &&
         a.a_instance == b.a_instance && a.a_generation == b.a_generation &&
         a.b_instance == b.b_instance && a.b_generation == b.b_generation &&
         a.basis_version == b.basis_version && a.prefix == b.prefix;
}
} // namespace rns8::detail
