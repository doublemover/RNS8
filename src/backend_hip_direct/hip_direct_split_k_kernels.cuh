#include "backend_common/split_k_partial.hpp"
// Each (fragment, plane, cell) has one producer. Fragment length <= 65536
// bounds every signed-byte dot by 65536*128*128 = 2^30, without a carry.
__global__ void rns8_split_k_produce_kernel(
    const int8_t* a, const int8_t* b, uint8_t* scratch,
    int m, int n, int k, int lda, int ldb, int prefix, int splits) {
  const int64_t cell = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t cells = static_cast<int64_t>(m) * n;
  if (cell >= cells) return;
  const int plane = blockIdx.y;
  const int fragment = blockIdx.z;
  const int row = static_cast<int>(cell / n), col = static_cast<int>(cell % n);
  const int begin = static_cast<int>(static_cast<int64_t>(k) * fragment / splits);
  const int end = static_cast<int>(static_cast<int64_t>(k) * (fragment + 1) / splits);
  int32_t value = 0;
  for (int q = begin; q < end; ++q)
    value += static_cast<int32_t>(a[static_cast<int64_t>(plane) * m * lda + static_cast<int64_t>(row) * lda + q]) *
             static_cast<int32_t>(b[static_cast<int64_t>(plane) * k * ldb + static_cast<int64_t>(q) * ldb + col]);
  const int modulus = rns8_default_moduli_device[plane];
  int residue = value % modulus;
  if (residue < 0) residue += modulus;
  scratch[(static_cast<int64_t>(fragment) * prefix + plane) * cells + cell] = static_cast<uint8_t>(residue);
}

// Native WMMA producers store centered bytes; explicitly canonicalize before
// the unsigned merge tree. Reinterpreting negative centered bytes is incorrect
// for every odd modulus, even though it happens to work for modulus 256.
__global__ void rns8_split_k_canonicalize_kernel(uint8_t* values, int64_t cells, int prefix, int fragments) {
  const int64_t cell = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (cell >= cells) return;
  const int plane = blockIdx.y, fragment = blockIdx.z;
  if (fragment >= fragments) return;
  const int64_t offset = (static_cast<int64_t>(fragment) * prefix + plane) * cells + cell;
  const int value = reinterpret_cast<int8_t*>(values)[offset];
  values[offset] = static_cast<uint8_t>(value < 0 ? value + rns8_default_moduli_device[plane] : value);
}

// Fixed fan-in 128: canonical inputs are <=255, sum <=32640. Every level
// reduces before global storage, so the proof is independent of tree depth.
template<bool CenteredInput=false>
__global__ void rns8_split_k_merge_kernel(
    const uint8_t* input, uint8_t* output, int64_t cells, int prefix, int fragments) {
  const int64_t cell = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (cell >= cells) return;
  const int plane = blockIdx.y, group = blockIdx.z;
  const int begin = group * 128, end = min(begin + 128, fragments);
  uint32_t value = 0;
  for (int f = begin; f < end; ++f)
    value += rns8::detail::split_k_partial_canonical(
        input[(static_cast<int64_t>(f) * prefix + plane) * cells + cell],
        rns8_default_moduli_device[plane],CenteredInput);
  output[(static_cast<int64_t>(group) * prefix + plane) * cells + cell] =
      static_cast<uint8_t>(value % rns8_default_moduli_device[plane]);
}

template<bool CenteredInput=false>
__global__ void rns8_split_k_store_kernel(const uint8_t* input, int8_t* output, int m, int n, int ldc) {
  const int64_t cell = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t cells = static_cast<int64_t>(m) * n;
  if (cell >= cells) return;
  const int plane = blockIdx.y, modulus = rns8_default_moduli_device[plane];
  const int value = input[static_cast<int64_t>(plane) * cells + cell];
  output[static_cast<int64_t>(plane) * m * ldc + (cell / n) * ldc + cell % n] =
      rns8::detail::split_k_partial_centered(static_cast<uint8_t>(value),modulus,CenteredInput);
}

// Window-local slots have global fragment identities. Fragment boundaries are
// determined by the total split count, never by the current window's tail size.
__global__ void rns8_split_k_window_produce_kernel(
    const int8_t* a,const int8_t* b,uint8_t* scratch,
    int m,int n,int k,int lda,int ldb,int prefix,int splits,int first_fragment) {
  const int64_t cell=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  const int64_t cells=int64_t(m)*n;if(cell>=cells) return;
  const int plane=blockIdx.y,local_fragment=blockIdx.z,fragment=first_fragment+local_fragment;
  const int row=int(cell/n),col=int(cell%n);
  const int begin=int(int64_t(k)*fragment/splits),end=int(int64_t(k)*(fragment+1)/splits);
  int32_t value=0;
  for(int q=begin;q<end;++q)
    value+=int32_t(a[(int64_t(plane)*m+row)*lda+q])*b[(int64_t(plane)*k+q)*ldb+col];
  const int modulus=rns8_default_moduli_device[plane];int residue=value%modulus;
  if(residue<0) residue+=modulus;
  scratch[(int64_t(local_fragment)*prefix+plane)*cells+cell]=uint8_t(residue);
}
// Both operands are canonical bytes. Including the carried residue bounds the
// sum by 510; reduce before the next window can observe it. No atomics/barriers.
__global__ void rns8_split_k_window_accumulate_kernel(
    const uint8_t* partial,uint8_t* running,int64_t cells,bool first_window) {
  const int64_t cell=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(cell>=cells) return;const int plane=blockIdx.y;
  const int64_t offset=int64_t(plane)*cells+cell;
  const uint32_t value=uint32_t(partial[offset])+(first_window?0u:uint32_t(running[offset]));
  running[offset]=uint8_t(value%rns8_default_moduli_device[plane]);
}
