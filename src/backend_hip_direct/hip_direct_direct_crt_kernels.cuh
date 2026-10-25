__device__ __constant__ rns8::detail::direct_crt::Constants rns8_direct_crt_table = rns8::detail::direct_crt::make_constants();

__global__ void rns8_export_canonical_direct_crt_kernel(const int8_t* residues, uint64_t* limbs, int cells, int prefix) {
  const int cell = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (cell >= cells) return;
  const auto* product_limbs = rns8_direct_crt_table.products[prefix];
  const rns8_u192_device product{product_limbs[0], product_limbs[1], product_limbs[2]};
  rns8_u192_device sum{};
  for (int p = 0; p < prefix; ++p) {
    const uint32_t modulus = rns8_default_moduli_device[p];
    const int8_t stored = residues[static_cast<int64_t>(p) * cells + cell];
    const uint32_t residue = rns8_canonical_from_centered_device(stored, modulus);
    const uint32_t coefficient = residue * rns8_direct_crt_table.inverses[prefix][p] % modulus;
    const auto* weight = rns8_direct_crt_table.weights[prefix][p];
    sum = rns8_u192_add_device(sum, rns8_u192_mul_u32_device({weight[0], weight[1], weight[2]}, coefficient));
  }
  // Each term is below P, so at most prefix-1 subtractions are needed.
  while (rns8_u192_compare_device(sum, product) >= 0) sum = rns8_u192_sub_device(sum, product);
  limbs[static_cast<int64_t>(cell) * 3] = sum.limb0;
  limbs[static_cast<int64_t>(cell) * 3 + 1] = sum.limb1;
  limbs[static_cast<int64_t>(cell) * 3 + 2] = sum.limb2;
}

__device__ __constant__ rns8::detail::quarter_crt::Constants rns8_quarter_crt_table=rns8::detail::quarter_crt::make_constants();
__global__ void rns8_export_quarter_crt_kernel(const int8_t* residues,uint64_t* output,int cells,int prefix,int digits,int limbs,int mode) {
  const int cell=int(blockIdx.x*blockDim.x+threadIdx.x);if(cell>=cells)return;
  uint32_t canonical[20];float z=0.0f;
  for(int p=0;p<prefix;++p){canonical[p]=rns8_canonical_from_centered_device(residues[int64_t(p)*cells+cell],rns8_default_moduli_device[p]);
    z=__fmaf_rn(float(canonical[p]),__uint_as_float(rns8_quarter_crt_table.fractions[prefix][p]),z);}
  const uint32_t q=__float2uint_rn(z);uint16_t values[12]{};
  rns8::detail::quarter_crt::delayed_digits(rns8_quarter_crt_table,canonical,prefix,q,digits,values);
  bool negative=(values[digits-1]&0x8000)!=0;
  if(mode==2){rns8::detail::quarter_crt::canonicalize(rns8_quarter_crt_table,prefix,values);negative=false;}
  for(int limb=0;limb<limbs;++limb){uint64_t value=0;for(int part=0;part<4;++part){const int d=4*limb+part;
      value|=uint64_t(d<digits?values[d]:(negative?65535:0))<<(16*part);}output[int64_t(cell)*limbs+limb]=value;}
}
