#pragma once
#include <cstdint>
#include <string>
#include "backend_common/packed_gemm_layout.hpp"

namespace rns8::detail {
enum class CkMatrixFamily : uint32_t { unknown, wmma, xdl };
enum class CkOperandAlphabet : uint32_t { unknown, centered_signed, canonical_unsigned };

struct CkImplementation {
  CkMatrixFamily family=CkMatrixFamily::unknown;
  CkOperandAlphabet alphabet=CkOperandAlphabet::unknown;
  uint32_t operand_bits=8, accumulator_bits=32, shape_alignment=0, k_alignment=1, k_tile=64, slab_k=32768;
  bool signed_operands=true, signed_accumulator=true, zero_initialized_dot=true, centered_post_merge=true;
  constexpr bool valid() const {
    return operand_bits==8 && accumulator_bits==32 && signed_accumulator && zero_initialized_dot &&
      centered_post_merge && k_alignment==1 && k_tile==64 && slab_k==32768 &&
      ((family==CkMatrixFamily::wmma && shape_alignment==64 &&
        ((signed_operands && alphabet==CkOperandAlphabet::centered_signed) ||
         (!signed_operands && alphabet==CkOperandAlphabet::canonical_unsigned))) ||
       (family==CkMatrixFamily::xdl && shape_alignment==128 && signed_operands &&
        alphabet==CkOperandAlphabet::centered_signed));
  }
  constexpr const char* kernel_family() const {
    return !valid()?"ck_unknown":family==CkMatrixFamily::xdl?"ck_xdl_cshuffle_s8_i32_v4":
      signed_operands?"ck_wmma_cshuffle_s8_i32_v4":"ck_wmma_cshuffle_u8_i32_v4";
  }
  std::string lowering_identity() const {
    return std::string(kernel_family())+";mn_padding="+std::to_string(shape_alignment)+
      ";packed_k_min=64;packed_k_alignment=1;internal_k_tile=64;slab=32768;dot_reset=zero;epilogue=centered;carry=post_epilogue_centered_i32";
  }
  std::string selected_kernel(uint32_t modulus=0,bool tiled=false) const {
    if(!modulus)return std::string(kernel_family())+(tiled?"_tiled_default_moduli_centered_epilogue":"_default_moduli_centered_epilogue");
    return std::string(kernel_family())+"_finite_u8_"+
      (modulus==251 || modulus==255 || modulus==256?"mod"+std::to_string(modulus):"static_modulus")+"_centered_epilogue";
  }
  constexpr uint32_t max_input(uint32_t modulus=256) const {
    return !valid() || modulus<2 || modulus>256?0:signed_operands?modulus/2:modulus-1;
  }
};
inline bool ck_packed_gemm_layout(uint64_t m,uint64_t n,uint64_t k,const CkImplementation& d,PackedGemmLayout& out) {
  return d.valid() && packed_gemm_layout(m,n,k,d.shape_alignment,d.k_tile,d.k_alignment,d.slab_k,true,out);
}
constexpr CkImplementation make_ck_implementation(int xdl, int signed_inputs, uint32_t alignment) {
  CkImplementation d;
  if((xdl!=0 && xdl!=1) || (signed_inputs!=0 && signed_inputs!=1))return d;
  d.family=xdl?CkMatrixFamily::xdl:CkMatrixFamily::wmma;
  d.signed_operands=signed_inputs!=0;
  d.alphabet=d.signed_operands?CkOperandAlphabet::centered_signed:CkOperandAlphabet::canonical_unsigned;
  d.shape_alignment=alignment;
  return d;
}
constexpr CkImplementation compiled_ck_implementation() {
#if defined(RNS8_CK_USE_XDL) && defined(RNS8_CK_WMMA_SIGNED_INPUTS) && defined(RNS8_CK_SHAPE_ALIGNMENT)
  return make_ck_implementation(RNS8_CK_USE_XDL,RNS8_CK_WMMA_SIGNED_INPUTS,RNS8_CK_SHAPE_ALIGNMENT);
#else
  return {};
#endif
}
inline bool ck_kernel_matches(const std::string& kernel,uint32_t modulus=0,bool allow_tiled=false) {
  for(const auto d:{make_ck_implementation(0,1,64),make_ck_implementation(0,0,64),make_ck_implementation(1,1,128)})
    if(kernel==d.selected_kernel(modulus) || (!modulus && allow_tiled && kernel==d.selected_kernel(0,true)))return true;
  return false;
}
inline const char* ck_dependency_identity() {
#ifdef RNS8_CK_DEPENDENCY_ID
  return RNS8_CK_DEPENDENCY_ID;
#else
  return "CK dependency identity unavailable";
#endif
}
}
