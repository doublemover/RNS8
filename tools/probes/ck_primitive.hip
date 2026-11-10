#include <cstdint>
#include <ck/ck.hpp>
#include <ck/tensor_operation/gpu/device/gemm_specialization.hpp>
#include <ck/tensor_operation/gpu/device/tensor_layout.hpp>
#ifndef RNS8_CK_PRIMITIVE_USE_XDL
#define RNS8_CK_PRIMITIVE_USE_XDL 0
#endif
#if RNS8_CK_PRIMITIVE_USE_XDL
#include <ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle.hpp>
#else
#include <ck/tensor_operation/gpu/device/impl/device_gemm_wmma.hpp>
#endif
#include <ck/tensor_operation/gpu/element/element_wise_operation.hpp>
#include <finite_u8_reducer.hpp>

template <ck::index_t... Is> using S = ck::Sequence<Is...>;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;

struct CenteredModulo {
  int32_t modulus;
  uint32_t reciprocal;

  __host__ __device__ CenteredModulo(int32_t m = 253, uint32_t r = 0) : modulus(m), reciprocal(r) {}

  template <typename Y, typename X> __host__ __device__ void operator()(Y& y, const X& x) const {
    y = static_cast<Y>(rns8::detail::finite_u8::reduce_to_centered_ck_i32(
        static_cast<int32_t>(x), static_cast<uint32_t>(modulus), reciprocal));
  }
};

#if RNS8_CK_PRIMITIVE_USE_XDL
using DeviceGemmInstance = ck::tensor_operation::device::DeviceGemm_Xdl_CShuffle<
    Row, Col, Row, int8_t, int8_t, int8_t, int32_t, int32_t, PassThrough, PassThrough,
    CenteredModulo, ck::tensor_operation::device::GemmSpecialization::MNKPadding, 1, 256, 128, 128,
    64, 16, 16, 32, 32, 4, 2, S<4, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 16, 16, true, S<4, 32, 1>,
    S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, true, 1, 1, S<1, 32, 1, 4>, 16>;
#else
using DeviceGemmInstance = ck::tensor_operation::device::DeviceGemmWmma_CShuffle<
    Row, Col, Row, int8_t, int8_t, int8_t, int32_t, int32_t, PassThrough, PassThrough,
    CenteredModulo, ck::tensor_operation::device::GemmSpecialization::MNKPadding, 1, 128, 64, 128,
    64, 2, 16, 16, 2, 4, S<4, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 2, 2, true, S<4, 32, 1>,
    S<1, 0, 2>, S<1, 0, 2>, 2, 2, 2, true, 1, 1, S<1, 32, 1, 4>, 8>;
#endif

extern "C" float rns8_ck_i8_primitive_probe(const int8_t* a, const int8_t* b, int8_t* c) {
#if RNS8_CK_PRIMITIVE_USE_XDL
  auto arg = DeviceGemmInstance::MakeArgument(
      a, b, c, 64, 128, 64, 64, 64, 128, PassThrough{}, PassThrough{},
      CenteredModulo{253, rns8::detail::finite_u8::modulus_reciprocal_u32(253)});
#else
  auto arg = DeviceGemmInstance::MakeArgument(
      a, b, c, 64, 128, 64, 64, 128, 128, PassThrough{}, PassThrough{},
      CenteredModulo{253, rns8::detail::finite_u8::modulus_reciprocal_u32(253)});
#endif
  if (!DeviceGemmInstance::IsSupportedArgument(arg) ||
      !DeviceGemmInstance::IsValidCompilationParameter()) {
    return -1.0f;
  }
  auto invoker = DeviceGemmInstance::MakeInvoker();
  return invoker.Run(arg);
}
