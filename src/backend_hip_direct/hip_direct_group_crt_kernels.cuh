__global__ void rns8_export_group_crt_local_kernel(const int8_t* residues,
    uint64_t* output,int cells,rns8::detail::group_crt::Plan plan) {
  const int cell=static_cast<int>(blockIdx.x*blockDim.x+threadIdx.x);
  if(cell>=cells)return;
  uint64_t values[4]{};
  for(uint32_t group=0;group<plan.group_count;++group)
    values[group]=rns8::detail::group_crt::reconstruct_group(residues,cells,cell,plan.groups[group]);
  const auto root=rns8::detail::group_crt::merge_groups(values,plan);
  for(int limb=0;limb<3;++limb)output[int64_t(cell)*3+limb]=root.limb[limb];
}
__global__ void rns8_produce_group_crt_kernel(const int8_t* residues,uint64_t* partials,
    int cells,rns8::detail::group_crt::Plan plan,uint32_t order) {
  const int cell=static_cast<int>(blockIdx.x*blockDim.x+threadIdx.x);
  if(cell>=cells)return;
  const auto group=rns8::detail::group_crt::producer_group(blockIdx.y,order,plan.group_count);
  partials[uint64_t(group)*cells+cell]=rns8::detail::group_crt::reconstruct_group(residues,cells,cell,plan.groups[group]);
}
__global__ void rns8_merge_group_crt_kernel(const uint64_t* partials,uint64_t* output,
    int cells,rns8::detail::group_crt::Plan plan) {
  const int cell=static_cast<int>(blockIdx.x*blockDim.x+threadIdx.x);
  if(cell>=cells)return;
  uint64_t values[4]{};
  for(uint32_t group=0;group<plan.group_count;++group)values[group]=partials[uint64_t(group)*cells+cell];
  const auto root=rns8::detail::group_crt::merge_groups(values,plan);
  for(int limb=0;limb<3;++limb)output[int64_t(cell)*3+limb]=root.limb[limb];
}
