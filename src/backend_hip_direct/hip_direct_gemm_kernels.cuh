#include "hip_direct_gemm_tiled_kernels.cuh"
#include "hip_direct_gemm_native_kernels.cuh"
#include "hip_direct_gemm_grouped_scheduled_kernels.cuh"


// Persistent small-shape GEMM kernel
// For compact M*N <= 64: single launch processes all planes in one kernel.
// Eliminates per-modulus launch overhead for small shapes.

__global__ void rns8_persistent_small_gemm_rns_kernel(
    const int8_t* __restrict__ a_residues,
    const int8_t* __restrict__ b_residues,
    int8_t* __restrict__ c_residues,
    int m,
    int n,
    int k,
    int prefix) {
  const int total_cells = m * n;
  const int cell = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell >= total_cells) return;

  const int row = cell / n;
  const int col = cell - row * n;

  // One thread computes one output cell across all prefix planes
  for (int plane = 0; plane < prefix; ++plane) {
    const int8_t* a_plane = a_residues + static_cast<int64_t>(plane) * m * k;
    const int8_t* b_plane = b_residues + static_cast<int64_t>(plane) * k * n;
    int8_t* c_plane = c_residues + static_cast<int64_t>(plane) * m * n;

    c_plane[cell] = rns8::detail::centered_residue_dot(
        a_plane + static_cast<int64_t>(row) * k,
        b_plane + col, k, n, static_cast<uint32_t>(rns8_default_moduli_device[plane]));
  }
}
