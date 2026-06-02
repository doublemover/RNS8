#include <hip/hip_runtime.h>

#include <cstdint>
#include <limits>
#include <type_traits>

namespace {

struct UInt192 {
  uint64_t lo;
  uint64_t mid;
  uint64_t hi;
};

__device__ UInt192 zero192() {
  return UInt192{0, 0, 0};
}

__device__ bool is_zero(UInt192 value) {
  return value.lo == 0 && value.mid == 0 && value.hi == 0;
}

__device__ UInt192 product_u64_u64(uint64_t a, uint64_t b) {
  const unsigned __int128 product = static_cast<unsigned __int128>(a) * static_cast<unsigned __int128>(b);
  return UInt192{
      static_cast<uint64_t>(product),
      static_cast<uint64_t>(product >> 64),
      0};
}

__device__ void add_assign(UInt192& acc, UInt192 value) {
  const uint64_t old_lo = acc.lo;
  acc.lo += value.lo;
  const uint64_t carry0 = acc.lo < old_lo ? 1ull : 0ull;
  const uint64_t old_mid = acc.mid;
  acc.mid += value.mid;
  const uint64_t carry1 = acc.mid < old_mid ? 1ull : 0ull;
  const uint64_t old_mid_after_value = acc.mid;
  acc.mid += carry0;
  const uint64_t carry2 = acc.mid < old_mid_after_value ? 1ull : 0ull;
  acc.hi += value.hi + carry1 + carry2;
}

__device__ int compare(UInt192 lhs, UInt192 rhs) {
  if (lhs.hi != rhs.hi) {
    return lhs.hi < rhs.hi ? -1 : 1;
  }
  if (lhs.mid != rhs.mid) {
    return lhs.mid < rhs.mid ? -1 : 1;
  }
  if (lhs.lo != rhs.lo) {
    return lhs.lo < rhs.lo ? -1 : 1;
  }
  return 0;
}

__device__ UInt192 subtract(UInt192 lhs, UInt192 rhs) {
  UInt192 out{};
  out.lo = lhs.lo - rhs.lo;
  const uint64_t borrow0 = lhs.lo < rhs.lo ? 1ull : 0ull;
  const uint64_t rhs_mid = rhs.mid + borrow0;
  const uint64_t borrow_mid_overflow = rhs_mid < rhs.mid ? 1ull : 0ull;
  out.mid = lhs.mid - rhs_mid;
  const uint64_t borrow1 = lhs.mid < rhs_mid ? 1ull : 0ull;
  out.hi = lhs.hi - rhs.hi - borrow_mid_overflow - borrow1;
  return out;
}

__device__ uint64_t abs_i64_to_u64(int64_t value) {
  const uint64_t bits = static_cast<uint64_t>(value);
  return value < 0 ? (~bits + 1ull) : bits;
}

__device__ void add_signed_product(bool& negative, UInt192& magnitude, bool product_negative, UInt192 product) {
  if (is_zero(product)) {
    return;
  }
  if (is_zero(magnitude)) {
    magnitude = product;
    negative = product_negative;
    return;
  }
  if (negative == product_negative) {
    add_assign(magnitude, product);
    return;
  }
  const int cmp = compare(magnitude, product);
  if (cmp == 0) {
    magnitude = zero192();
    negative = false;
  } else if (cmp > 0) {
    magnitude = subtract(magnitude, product);
  } else {
    magnitude = subtract(product, magnitude);
    negative = product_negative;
  }
}

__global__ void gemm_i64_kernel(
    const int64_t* a,
    const int64_t* b,
    int64_t* c,
    uint32_t* status,
    int64_t m,
    int64_t n,
    int64_t k) {
  const int64_t row = static_cast<int64_t>(blockIdx.y * blockDim.y + threadIdx.y);
  const int64_t col = static_cast<int64_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= m || col >= n) {
    return;
  }

  bool negative = false;
  UInt192 magnitude = zero192();
  for (int64_t kk = 0; kk < k; ++kk) {
    const int64_t av = a[row * k + kk];
    const int64_t bv = b[kk * n + col];
    const bool product_negative = (av < 0) != (bv < 0);
    const UInt192 product = product_u64_u64(abs_i64_to_u64(av), abs_i64_to_u64(bv));
    add_signed_product(negative, magnitude, product_negative, product);
  }

  int64_t value = 0;
  if (!negative) {
    if (magnitude.hi != 0 || magnitude.mid != 0 ||
        magnitude.lo > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
      atomicOr(status, 1u);
    } else {
      value = static_cast<int64_t>(magnitude.lo);
    }
  } else {
    const uint64_t int64_min_abs = uint64_t{1} << 63u;
    if (magnitude.hi != 0 || magnitude.mid != 0 || magnitude.lo > int64_min_abs) {
      atomicOr(status, 1u);
    } else if (magnitude.lo == int64_min_abs) {
      value = std::numeric_limits<int64_t>::min();
    } else {
      value = -static_cast<int64_t>(magnitude.lo);
    }
  }
  c[row * n + col] = value;
}

__global__ void gemm_u64_kernel(
    const uint64_t* a,
    const uint64_t* b,
    uint64_t* c,
    uint32_t* status,
    int64_t m,
    int64_t n,
    int64_t k) {
  const int64_t row = static_cast<int64_t>(blockIdx.y * blockDim.y + threadIdx.y);
  const int64_t col = static_cast<int64_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= m || col >= n) {
    return;
  }

  UInt192 acc = zero192();
  for (int64_t kk = 0; kk < k; ++kk) {
    add_assign(acc, product_u64_u64(a[row * k + kk], b[kk * n + col]));
  }
  if (acc.hi != 0 || acc.mid != 0) {
    atomicOr(status, 1u);
  }
  c[row * n + col] = acc.lo;
}

template <typename T>
int launch_vector_gemm(
    int device_id,
    const T* a,
    const T* b,
    T* c,
    uint32_t* status,
    int64_t m,
    int64_t n,
    int64_t k) {
  hipError_t error = hipSetDevice(device_id);
  if (error != hipSuccess) {
    return static_cast<int>(error);
  }
  const dim3 block(16, 16);
  const dim3 grid(
      static_cast<unsigned int>((static_cast<uint64_t>(n) + block.x - 1u) / block.x),
      static_cast<unsigned int>((static_cast<uint64_t>(m) + block.y - 1u) / block.y));
  if constexpr (std::is_same<T, int64_t>::value) {
    hipLaunchKernelGGL(gemm_i64_kernel, grid, block, 0, 0, a, b, c, status, m, n, k);
  } else {
    hipLaunchKernelGGL(gemm_u64_kernel, grid, block, 0, 0, a, b, c, status, m, n, k);
  }
  error = hipGetLastError();
  if (error != hipSuccess) {
    return static_cast<int>(error);
  }
  return static_cast<int>(hipDeviceSynchronize());
}

}  // namespace

extern "C" int rns8_bench_vector_i64_gemm_device(
    int device_id,
    const int64_t* a,
    const int64_t* b,
    int64_t* c,
    uint32_t* status,
    int64_t m,
    int64_t n,
    int64_t k) {
  return launch_vector_gemm(device_id, a, b, c, status, m, n, k);
}

extern "C" int rns8_bench_vector_u64_gemm_device(
    int device_id,
    const uint64_t* a,
    const uint64_t* b,
    uint64_t* c,
    uint32_t* status,
    int64_t m,
    int64_t n,
    int64_t k) {
  return launch_vector_gemm(device_id, a, b, c, status, m, n, k);
}
