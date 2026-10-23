__global__ void rns8_convert_residue_layout_kernel(
    const int8_t* src, rns8::detail::ResidueLayout source, int8_t* dst, rns8::detail::ResidueLayout destination) {
  const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t cells = source.rows * source.cols;
  if (index >= cells * source.prefix) return;
  const auto plane = static_cast<uint32_t>(index / cells);
  const uint64_t cell = index % cells, row = cell / source.cols, col = cell % source.cols;
  dst[destination.offset(plane, row, col)] = src[source.offset(plane, row, col)];
}
