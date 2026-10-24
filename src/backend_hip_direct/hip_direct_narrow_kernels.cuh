__global__ void rns8_native_i64_abs_max_kernel(
    const uint64_t* source, int rows, int cols, int ld, unsigned long long* maximum, bool signed_values) {
  __shared__ unsigned long long partial[256];
  const int lane = threadIdx.x;
  const int64_t cell = static_cast<int64_t>(blockIdx.x) * blockDim.x + lane;
  uint64_t magnitude = 0;
  if (cell < static_cast<int64_t>(rows) * cols) {
    const uint64_t value = source[(cell / cols) * ld + cell % cols];
    magnitude = signed_values && (value >> 63) ? uint64_t{0} - value : value;
  }
  partial[lane] = magnitude;
  __syncthreads();
  for (int stride = 128; stride; stride /= 2) {
    if (lane < stride && partial[lane + stride] > partial[lane]) partial[lane] = partial[lane + stride];
    __syncthreads();
  }
  if (lane == 0) atomicMax(maximum, partial[0]);
}

__global__ void rns8_native_i64_narrow_gemm_kernel(
    const int64_t* a, const int64_t* b, int64_t* c, int m, int n, int k, int lda, int ldb, int ldc) {
  const int64_t cell = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (cell >= static_cast<int64_t>(m) * n) return;
  const int64_t row = cell / n, col = cell % n;
  int64_t value = 0;
  // Host admission proves K*max_abs(A)*max_abs(B) <= INT64_MAX from the actual
  // device inputs. Thus products and every intermediate sum are representable.
  for (int q = 0; q < k; ++q) value += a[row * lda + q] * b[static_cast<int64_t>(q) * ldb + col];
  c[row * ldc + col] = value;
}
