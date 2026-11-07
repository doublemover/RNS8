
template <bool Rank48, bool Lhs, unsigned Product, unsigned Index = 0>
__device__ __forceinline__ void rns8_selected_bilinear_steps(int8_t* slots, int q) {
  constexpr auto plan = rns8::detail::selected_bilinear_plan<Rank48, Lhs, Product>;
  if constexpr (Index < plan.count) {
    constexpr auto step = plan.steps[Index];
    int value = int(slots[step.left]) * step.left_sign + int(slots[step.right]) * step.right_sign;
    if constexpr (step.halve)
      value *= ((q + 1) / 2);
    slots[step.target] = int8_t(rns8::detail::selected_linear_center(value, q));
    rns8_selected_bilinear_steps<Rank48, Lhs, Product, Index + 1>(slots, q);
  }
}
template <bool Rank48, bool Lhs, unsigned Product>
__global__ void rns8_selected_bilinear_transform_kernel(const int8_t* source, int8_t* output,
                                                        int rows, int cols, int ld, int hr, int hc,
                                                        rns8::detail::IndexedModulusPlanes planes) {
  constexpr auto plan = rns8::detail::selected_bilinear_plan<Rank48, Lhs, Product>;
  static_assert(plan.valid && plan.slots > 0, "selected closure must lower before launch");
  const int64_t cell = int64_t(blockIdx.x) * blockDim.x + threadIdx.x, cells = int64_t(hr) * hc;
  if (cell >= cells)
    return;
  constexpr int blocks = Rank48 ? 4 : 3;
  const uint32_t compact = blockIdx.y, plane = planes.eligible[compact];
  const int q = planes.moduli[compact];
  int8_t slots[plan.slots]{};
#pragma unroll
  for (unsigned index = 0; index < plan.input_count; ++index) {
    const int block = plan.input_ids[index];
    const int64_t row = cell / hc + int64_t(block / blocks) * hr,
                  col = cell % hc + int64_t(block % blocks) * hc;
    if (row < rows && col < cols)
      slots[index] = int8_t(rns8::detail::selected_linear_center(
          source[(int64_t(plane) * rows + row) * ld + col], q));
  }
  rns8_selected_bilinear_steps<Rank48, Lhs, Product>(slots, q);
  output[int64_t(compact) * cells + cell] = slots[plan.result];
}
template <bool Rank48, bool Lhs, unsigned Product>
static int rns8_launch_selected_bilinear(const int8_t* source, int8_t* output, int rows, int cols,
                                         int ld, int hr, int hc,
                                         rns8::detail::IndexedModulusPlanes planes, void* stream) {
  if constexpr (Product < (Rank48 ? 48 : 23)) {
    hipLaunchKernelGGL((rns8_selected_bilinear_transform_kernel<Rank48, Lhs, Product>),
                       dim3((uint64_t(hr) * hc + 255) / 256, planes.eligible_count), dim3(256), 0,
                       static_cast<hipStream_t>(stream), source, output, rows, cols, ld, hr, hc,
                       planes);
    return int(hipGetLastError());
  } else
    return int(hipErrorInvalidValue);
}
template <bool Rank48, bool Lhs>
static int rns8_dispatch_selected_bilinear(const int8_t* source, int8_t* output, int rows, int cols,
                                           int ld, int hr, int hc, int product,
                                           rns8::detail::IndexedModulusPlanes planes,
                                           void* stream) {
// Product identity is a compile-time kernel parameter: every arithmetic source
// and target is static, rather than indexing a descriptor array per thread.
#define RNS8_SELECTED_CASE(N) case N:return rns8_launch_selected_bilinear<Rank48,Lhs,N>(source,output,rows,cols,ld,hr,hc,planes,stream)
  switch (product) {
    RNS8_SELECTED_CASE(0);
    RNS8_SELECTED_CASE(1);
    RNS8_SELECTED_CASE(2);
    RNS8_SELECTED_CASE(3);
    RNS8_SELECTED_CASE(4);
    RNS8_SELECTED_CASE(5);
    RNS8_SELECTED_CASE(6);
    RNS8_SELECTED_CASE(7);
    RNS8_SELECTED_CASE(8);
    RNS8_SELECTED_CASE(9);
    RNS8_SELECTED_CASE(10);
    RNS8_SELECTED_CASE(11);
    RNS8_SELECTED_CASE(12);
    RNS8_SELECTED_CASE(13);
    RNS8_SELECTED_CASE(14);
    RNS8_SELECTED_CASE(15);
    RNS8_SELECTED_CASE(16);
    RNS8_SELECTED_CASE(17);
    RNS8_SELECTED_CASE(18);
    RNS8_SELECTED_CASE(19);
    RNS8_SELECTED_CASE(20);
    RNS8_SELECTED_CASE(21);
    RNS8_SELECTED_CASE(22);
    RNS8_SELECTED_CASE(23);
    RNS8_SELECTED_CASE(24);
    RNS8_SELECTED_CASE(25);
    RNS8_SELECTED_CASE(26);
    RNS8_SELECTED_CASE(27);
    RNS8_SELECTED_CASE(28);
    RNS8_SELECTED_CASE(29);
    RNS8_SELECTED_CASE(30);
    RNS8_SELECTED_CASE(31);
    RNS8_SELECTED_CASE(32);
    RNS8_SELECTED_CASE(33);
    RNS8_SELECTED_CASE(34);
    RNS8_SELECTED_CASE(35);
    RNS8_SELECTED_CASE(36);
    RNS8_SELECTED_CASE(37);
    RNS8_SELECTED_CASE(38);
    RNS8_SELECTED_CASE(39);
    RNS8_SELECTED_CASE(40);
    RNS8_SELECTED_CASE(41);
    RNS8_SELECTED_CASE(42);
    RNS8_SELECTED_CASE(43);
    RNS8_SELECTED_CASE(44);
    RNS8_SELECTED_CASE(45);
    RNS8_SELECTED_CASE(46);
    RNS8_SELECTED_CASE(47);
  default:
    return int(hipErrorInvalidValue);
  }
#undef RNS8_SELECTED_CASE
}
extern "C" int rns8_hip_bilinear_selected_transform(const int8_t* source, int8_t* output, int rows,
                                                    int cols, int ld, int hr, int hc, bool rank48,
                                                    bool lhs, int product,
                                                    rns8::detail::IndexedModulusPlanes planes,
                                                    void* stream) {
  if (!source || !output || rows <= 0 || cols <= 0 || ld < cols || hr <= 0 || hc <= 0 ||
      !rns8::detail::valid_indexed_modulus_planes(planes, rank48) || !planes.eligible_count)
    return int(hipErrorInvalidValue);
  if (rank48)
    return lhs ? rns8_dispatch_selected_bilinear<true, true>(source, output, rows, cols, ld, hr, hc,
                                                             product, planes, stream)
               : rns8_dispatch_selected_bilinear<true, false>(source, output, rows, cols, ld, hr,
                                                              hc, product, planes, stream);
  return lhs ? rns8_dispatch_selected_bilinear<false, true>(source, output, rows, cols, ld, hr, hc,
                                                            product, planes, stream)
             : rns8_dispatch_selected_bilinear<false, false>(source, output, rows, cols, ld, hr, hc,
                                                             product, planes, stream);
}
