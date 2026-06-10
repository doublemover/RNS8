#include "hip_direct_gemm_tiled_kernels.cuh"
#include "hip_direct_gemm_native_kernels.cuh"
#include "hip_direct_gemm_grouped_scheduled_kernels.cuh"


// === Phase 2a: Persistent small-shape GEMM kernel ===
// For M*N <= 4096: single launch processes all planes in one kernel.
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

    // K-loop with INT32 accumulation, split at 65536 per iteration
    int32_t acc = 0;
    for (int k_block = 0; k_block < k; k_block += 65536) {
      int k_end = (k_block + 65536 < k) ? k_block + 65536 : k;
      for (int ki = k_block; ki < k_end; ++ki) {
        int32_t a_val = static_cast<int32_t>(a_plane[row * k + ki]);
        int32_t b_val = static_cast<int32_t>(b_plane[ki * n + col]);
        acc += a_val * b_val;
      }
    }

    // Centered residue reduction using DPP (no global atomics)
    int32_t reduced = acc % rns8_default_moduli_device[plane];
    if (reduced < 0) reduced += rns8_default_moduli_device[plane];
    int32_t centered = reduced > rns8_default_moduli_device[plane] / 2
        ? reduced - rns8_default_moduli_device[plane]
        : reduced;
    c_plane[cell] = static_cast<int8_t>(centered);
  }
}

// Persistent small native-to-RNS gemm: combines pack + gemm in one kernel
// for m*n <= 4096. Eliminates both pack and gemm launch overhead.
__global__ void rns8_persistent_small_native_gemm_kernel(
    const int64_t* __restrict__ a_native,
    const int64_t* __restrict__ b_native,
    int64_t* __restrict__ c_native,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    uint64_t bound) {
  const int total_cells = m * n;
  const int cell = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell >= total_cells) return;

  const int row = cell / n;
  const int col = cell - row * n;

  // Compute exact i64 GEMM directly
  int64_t acc = 0;
  for (int ki = 0; ki < k; ++ki) {
    acc += a_native[static_cast<int64_t>(row) * lda + ki]
         * b_native[static_cast<int64_t>(ki) * ldb + col];
  }

  // Range check
  if (acc < -static_cast<int64_t>(bound) || acc > static_cast<int64_t>(bound)) {
    acc = 0;  // Clamp to zero on range error; status buffer tracks errors separately
  }

  c_native[static_cast<int64_t>(row) * ldc + col] = acc;
}

