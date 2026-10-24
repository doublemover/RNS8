template<class T, int Group, int Depth, int Buffers>
int rns8_launch_native_tiled(const void* a,const void* b,int8_t* c,int m,int n,int k,
                            int lda,int ldb,int ldc,int prefix) {
  const dim3 grid((uint64_t(n)+31)/32,(uint64_t(m)+31)/32,(prefix+Group-1)/Group);
  hipLaunchKernelGGL((rns8_native_tiled_wmma_kernel<T,Group,Depth,Buffers>),grid,dim3(128),0,nullptr,
      static_cast<const T*>(a),static_cast<const T*>(b),c,m,n,k,lda,ldb,ldc,prefix);
  return static_cast<int>(hipGetLastError());
}
template<class T, int Group>
int rns8_select_native_tiled_storage(const void* a,const void* b,int8_t* c,int m,int n,int k,
                                   int lda,int ldb,int ldc,int prefix,int depth,int buffers) {
  if (buffers==0 && depth==16) return rns8_launch_native_tiled<T,Group,16,0>(a,b,c,m,n,k,lda,ldb,ldc,prefix);
  if (buffers==1 && depth==16) return rns8_launch_native_tiled<T,Group,16,1>(a,b,c,m,n,k,lda,ldb,ldc,prefix);
  if (buffers==1 && depth==32) return rns8_launch_native_tiled<T,Group,32,1>(a,b,c,m,n,k,lda,ldb,ldc,prefix);
  if (buffers==2 && depth==16) return rns8_launch_native_tiled<T,Group,16,2>(a,b,c,m,n,k,lda,ldb,ldc,prefix);
  if (buffers==2 && depth==32) return rns8_launch_native_tiled<T,Group,32,2>(a,b,c,m,n,k,lda,ldb,ldc,prefix);
  return static_cast<int>(hipErrorInvalidValue);
}
template<class T>
int rns8_select_native_tiled_group(const void* a,const void* b,int8_t* c,int m,int n,int k,
                                 int lda,int ldb,int ldc,int prefix,int group,int depth,int buffers) {
  switch (group) {
    case 1: return rns8_select_native_tiled_storage<T,1>(a,b,c,m,n,k,lda,ldb,ldc,prefix,depth,buffers);
    case 2: return rns8_select_native_tiled_storage<T,2>(a,b,c,m,n,k,lda,ldb,ldc,prefix,depth,buffers);
    case 4: return rns8_select_native_tiled_storage<T,4>(a,b,c,m,n,k,lda,ldb,ldc,prefix,depth,buffers);
    default: return static_cast<int>(hipErrorInvalidValue);
  }
}
extern "C" int rns8_hip_native_tiled(const void* a,const void* b,int8_t* c,int m,int n,int k,
                                   int lda,int ldb,int ldc,int prefix,int group,int depth,int buffers,bool unsign) {
  return unsign ? rns8_select_native_tiled_group<uint64_t>(a,b,c,m,n,k,lda,ldb,ldc,prefix,group,depth,buffers)
                : rns8_select_native_tiled_group<int64_t>(a,b,c,m,n,k,lda,ldb,ldc,prefix,group,depth,buffers);
}
