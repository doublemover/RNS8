// Included after the maintained exact CRT/device helpers.
__global__ void rns8_distributed_pack_kernel(const uint64_t* native_a,
    const uint64_t* native_b, int8_t* a, int8_t* b, uint64_t row, uint64_t col,
    uint64_t rows, uint64_t cols, uint64_t k, uint64_t lda, uint64_t ldb,
    bool signed_value, rns8::detail::finite_u8::Native64Reducer reducer,uint64_t k_begin) {
  const uint64_t ac = rows * k, bc = k * cols;
  for (uint64_t cell = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       cell < ac + bc; cell += uint64_t(gridDim.x) * blockDim.x) {
    const uint64_t bits = cell < ac ? native_a[(row + cell / k) * lda + k_begin+cell % k]
        : native_b[(k_begin+(cell - ac) / cols) * ldb + col + (cell - ac) % cols];
    int residue = int(rns8::detail::finite_u8::canonical_native_u64(bits, reducer));
    if (signed_value && (bits >> 63)) {
      residue -= int(reducer.signed_bias); if (residue < 0) residue += int(reducer.modulus);
    }
    if (residue >= int((reducer.modulus + 1) / 2)) residue -= int(reducer.modulus);
    if (cell < ac) a[cell] = static_cast<int8_t>(residue);
    else b[cell - ac] = static_cast<int8_t>(residue);
  }
}

__global__ void rns8_distributed_export_kernel(const int8_t* residues, uint64_t* output,
    int elements, int cols, uint32_t tile_rows, uint32_t tile_cols,
    const rns8::detail::distributed::DeviceTile* tiles, bool signed_value,
    bool bounded, int* status) {
  const uint64_t index = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= uint64_t(elements)) return;
  const int cell = int(index);
  const uint64_t row = uint64_t(cell) / cols, col = uint64_t(cell) % cols;
  const uint64_t tile_grid_cols = (uint64_t(cols) + tile_cols - 1) / tile_cols;
  const auto& tile = tiles[(row / tile_rows) * tile_grid_cols + col / tile_cols];
  rns8_u192_device root{}, product{};
  rns8_reconstruct_canonical_wide_device(residues, cell, elements, int(tile.prefix), &root, &product);
  const bool negative = signed_value && rns8_u192_centered_is_negative_device(root, product);
  const auto magnitude = negative ? rns8_u192_sub_device(product, root) : root;
  const rns8_u192_device bound{tile.bound[0], tile.bound[1], tile.bound[2]};
  if (rns8_u192_compare_device(magnitude, bound) > 0 ||
      (bounded && (magnitude.limb1 || magnitude.limb2 ||
       (signed_value && magnitude.limb0 > (negative ? (uint64_t{1} << 63) : uint64_t(INT64_MAX)))))) {
    atomicExch(status, 1); return;
  }
  if (negative) rns8_store_u192_negative_twos_complement_limbs_device(output + uint64_t(cell) * 3, magnitude, 3);
  else rns8_store_u192_unsigned_limbs_device(output + uint64_t(cell) * 3, root, 3);
}

// K fragments remain separate until complete transfer membership is established
// by the owner. Sum centered byte partials in INT32 (at most64*128), reduce in
// their actual channel, and reconstruct only the fully combined residues.
__global__ void rns8_distributed_reduce_partials_kernel(const int8_t* partials,int8_t* residues,
    const rns8::detail::distributed::DeviceTile* tiles,uint64_t rows,uint64_t cols) {
  const auto& tile=tiles[blockIdx.x];const auto plane=uint32_t(blockIdx.y);
  if(plane>=tile.prefix)return;const uint64_t cells=tile.rows*tile.cols;
  const int modulus=rns8_default_moduli_device[plane];
  for(uint64_t cell=threadIdx.x;cell<cells;cell+=blockDim.x){
    int sum=0;for(uint32_t part=0;part<tile.fragments;++part)
      sum+=partials[tile.partial_offset+(uint64_t(plane)*tile.fragments+part)*cells+cell];
    sum%=modulus;if(sum<0)sum+=modulus;if(sum>=(modulus+1)/2)sum-=modulus;
    residues[(uint64_t(plane)*rows+tile.row+cell/tile.cols)*cols+tile.col+cell%tile.cols]=int8_t(sum);
  }
}

__device__ uint32_t rns8_distributed_check_native(uint64_t bits,bool signed_value,uint32_t q) {
  uint32_t value=uint32_t(bits%q);
  if(signed_value && (bits>>63))value=(value+q-uint32_t(((UINT64_MAX%q)+1)%q))%q;
  return value;
}
__global__ void rns8_distributed_integrity_kernel(const uint64_t* a,const uint64_t* b,
    const uint64_t* candidate,uint64_t m,uint64_t n,uint64_t k,uint64_t lda,uint64_t ldb,
    bool signed_value,int* status) {
  const uint64_t cell=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(cell>=m*n || (atomicAdd(status,0)&1))return;
  rns8_u192_device value{candidate[cell*3],candidate[cell*3+1],candidate[cell*3+2]};
  const bool negative=signed_value && (value.limb2>>63);
  if(negative)value=rns8_u192_sub_device(rns8_u192_device{},value);
  const uint32_t primes[2]={257,263};uint32_t checks[2]{};
  for(uint64_t inner=0;inner<k;++inner)for(unsigned p=0;p<2;++p) {
    const auto q=primes[p];
    checks[p]=(checks[p]+rns8_distributed_check_native(a[(cell/n)*lda+inner],signed_value,q)*
        rns8_distributed_check_native(b[inner*ldb+cell%n],signed_value,q))%q;
  }
  for(unsigned p=0;p<2;++p) {
    uint32_t observed=rns8_u192_mod_u32_device(value,primes[p]);
    if(negative && observed)observed=primes[p]-observed;
    if(observed!=checks[p])atomicOr(reinterpret_cast<unsigned*>(status),4u<<p);
  }
}

__global__ void rns8_distributed_materialize_kernel(const uint64_t* limbs, int8_t* residues,
    int elements, uint32_t prefix, bool signed_value) {
  const uint64_t index = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= uint64_t(elements) * prefix) return;
  const uint64_t cell = index % uint64_t(elements);
  const uint32_t plane = uint32_t(index / uint64_t(elements));
  rns8_u192_device value{limbs[cell * 3], limbs[cell * 3 + 1], limbs[cell * 3 + 2]};
  const bool negative = signed_value && (value.limb2 >> 63);
  if (negative) value = rns8_u192_sub_device(rns8_u192_device{}, value);
  const int modulus = rns8_default_moduli_device[plane];
  int residue = int(rns8_u192_mod_u32_device(value, uint32_t(modulus)));
  if (negative && residue) residue = modulus - residue;
  if (residue >= (modulus + 1) / 2) residue -= modulus;
  residues[index] = static_cast<int8_t>(residue);
}
