#pragma once
#include <hipblaslt/hipblaslt-ext.hpp>

// Compile with the SDK compiler: unrelated packed declarations in the official
// extension header are not accepted by MSVC. Keep the actual SDK declaration.
extern "C" int rns8_hipblaslt_algorithm_index(const hipblasLtMatmulAlgo_t* algo) {
  auto query=*algo;
  return hipblaslt_ext::getIndexFromAlgo(query);
}
