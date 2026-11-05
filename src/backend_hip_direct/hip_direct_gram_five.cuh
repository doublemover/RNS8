namespace {
__global__ void gram_five_transform_kernel(const int8_t* source, int8_t* left, int8_t* right,
                                           rns8::detail::GramFivePacket packet, int term) {
  const int64_t cell = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (cell >= packet.h * packet.hk)
    return;
  const uint32_t compact = uint32_t(blockIdx.y), plane = packet.planes.eligible[compact];
  const int64_t row = cell / packet.hk, col = cell % packet.hk;
  left[int64_t(compact) * packet.h * packet.hk + cell] =
      rns8::detail::gram_five_transform_value(source, packet, plane, term, true, row, col);
  right[(int64_t(compact) * packet.hk + col) * packet.h + row] =
      rns8::detail::gram_five_transform_value(source, packet, plane, term, false, row, col);
}
__global__ void gram_five_combine_kernel(const int8_t* products, int8_t* output,
                                         rns8::detail::GramFivePacket packet) {
  const int64_t cell = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto plane = packet.planes.eligible[uint32_t(blockIdx.y)];
  if (cell >= packet.n * packet.n)
    return;
  output[int64_t(plane) * packet.n * packet.n + cell] = rns8::detail::gram_five_combine_value(
      products, packet, plane, cell / packet.n, cell % packet.n);
}

} // namespace
extern "C" int rns8_hip_gram_five_transform(const int8_t* source, int8_t* left, int8_t* right,
                                            rns8::detail::GramFivePacket packet, int term,
                                            void* stream) {
  if (!source || !left || !right || !stream || !packet.planes.eligible_count || term < 0 ||
      term >= 5)
    return int(hipErrorInvalidValue);
  hipLaunchKernelGGL(
      gram_five_transform_kernel,
      dim3((uint64_t(packet.h) * packet.hk + 255) / 256, packet.planes.eligible_count), dim3(256),
      0, static_cast<hipStream_t>(stream), source, left, right, packet, term);
  return int(hipGetLastError());
}
extern "C" int rns8_hip_gram_five_finish(const int8_t* products, int8_t* output,
                                         rns8::detail::GramFivePacket packet, void* stream) {
  if (!products || !output || !stream || !packet.planes.eligible_count)
    return int(hipErrorInvalidValue);
  const dim3 grid((uint64_t(packet.n) * packet.n + 255) / 256, packet.planes.eligible_count);
  hipLaunchKernelGGL(gram_five_combine_kernel, grid, dim3(256), 0, static_cast<hipStream_t>(stream),
                     products, output, packet);
  return int(hipGetLastError());
}
