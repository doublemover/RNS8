#pragma once
#include "backend_common/encoded_b_layout.hpp"
namespace rns8::detail {
rns8_status hip_prepare_encoded_b(int device,const void* source,void* storage,
    int64_t k,int64_t n,int64_t ldb,uint32_t prefix,bool unsigned_inputs,uint64_t bytes);
// Requires preparation on the same compiled device. No fresh capability scratch.
rns8_status hip_gemm_encoded_b(int device,const void* a,const void* encoded,void* c,
    int64_t m,int64_t n,int64_t k,int64_t lda,int64_t ldc,uint32_t prefix,uint32_t group,
    bool unsigned_inputs,bool& commit_started);
}
