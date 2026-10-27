namespace {
__global__ void gram_five_transform_kernel(const int8_t* source,int8_t* left,int8_t* right,
    rns8::detail::GramFivePacket packet,int term) {
  const int64_t cell=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(cell>=packet.h*packet.hk)return;const auto plane=uint32_t(blockIdx.y);
  const int64_t row=cell/packet.hk,col=cell%packet.hk;
  left[int64_t(plane)*packet.h*packet.hk+cell]=rns8::detail::gram_five_transform_value(source,packet,plane,term,true,row,col);
  right[(int64_t(plane)*packet.hk+col)*packet.h+row]=rns8::detail::gram_five_transform_value(source,packet,plane,term,false,row,col);
}
__global__ void gram_five_combine_kernel(const int8_t* products,int8_t* output,rns8::detail::GramFivePacket packet) {
  const int64_t cell=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  const auto plane=uint32_t(blockIdx.y);if(cell>=packet.n*packet.n || !packet.certified[plane])return;
  output[int64_t(plane)*packet.n*packet.n+cell]=rns8::detail::gram_five_combine_value(products,packet,plane,cell/packet.n,cell%packet.n);
}
__global__ void gram_five_classical_kernel(const int8_t* source,int8_t* output,rns8::detail::GramFivePacket packet) {
  const int64_t cell=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;const auto plane=uint32_t(blockIdx.y);
  if(cell>=packet.n*packet.n || packet.certified[plane])return;
  const int64_t row=cell/packet.n,col=cell%packet.n;if(row>col)return;
  const int modulus=packet.moduli[plane];int residue=0;
  for(int64_t first=0;first<packet.k;) {
    const int64_t end=first+(packet.k-first>65536?65536:packet.k-first);int sum=residue;
    for(int64_t k=first;k<end;++k)sum+=rns8::detail::gram_five_load(source,packet,plane,row,k)*rns8::detail::gram_five_load(source,packet,plane,col,k);
    residue=rns8::detail::gram_five_center(sum,modulus);first=end;
  }
  output[(int64_t(plane)*packet.n+row)*packet.n+col]=int8_t(residue);
  if(row!=col)output[(int64_t(plane)*packet.n+col)*packet.n+row]=int8_t(residue);
}
}
extern "C" int rns8_hip_gram_five_transform(const int8_t* source,int8_t* left,int8_t* right,rns8::detail::GramFivePacket packet,int term) {
  if(!source || !left || !right || term<0 || term>=5)return int(hipErrorInvalidValue);
  hipLaunchKernelGGL(gram_five_transform_kernel,dim3((uint64_t(packet.h)*packet.hk+255)/256,packet.prefix),dim3(256),0,nullptr,source,left,right,packet,term);
  return int(hipGetLastError());
}
extern "C" int rns8_hip_gram_five_finish(const int8_t* source,const int8_t* products,int8_t* output,rns8::detail::GramFivePacket packet) {
  if(!source || !products || !output)return int(hipErrorInvalidValue);
  const dim3 grid((uint64_t(packet.n)*packet.n+255)/256,packet.prefix);
  hipLaunchKernelGGL(gram_five_combine_kernel,grid,dim3(256),0,nullptr,products,output,packet);
  auto code=hipGetLastError();if(code!=hipSuccess)return int(code);
  hipLaunchKernelGGL(gram_five_classical_kernel,grid,dim3(256),0,nullptr,source,output,packet);
  return int(hipGetLastError());
}
