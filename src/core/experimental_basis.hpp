#pragma once
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
}
