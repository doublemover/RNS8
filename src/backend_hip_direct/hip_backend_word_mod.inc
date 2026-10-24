#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
extern "C" int rns8_hip_word_mod_validate(const void*,const void*,int,int,int,int,int,uint64_t,int*);
extern "C" int rns8_hip_word_mod_gemm(const void*,const void*,void*,int,int,int,int,int,int,uint64_t);
#endif
rns8_status hip_direct_word_mod_device(int device_id,const void* a,const void* b,void* c,
    const word_mod::Problem& p,uint64_t budget,bool& commit_started) {
  commit_started=false;const auto s=word_mod::query(p,true);if(s.status!=RNS8_SUCCESS) return s.status;
  if(!a || !b || !c || reinterpret_cast<uintptr_t>(a)%8 || reinterpret_cast<uintptr_t>(b)%8 ||
     reinterpret_cast<uintptr_t>(c)%8 || !nonoverlapping_byte_ranges(a,s.a_bytes,c,s.c_bytes) ||
     !nonoverlapping_byte_ranges(b,s.b_bytes,c,s.c_bytes)) return RNS8_INVALID_ARGUMENT;
  if(budget<4) return RNS8_WORKSPACE_TOO_SMALL;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  auto status=set_hip_device(device_id);if(status!=RNS8_SUCCESS) return status;
  hip_direct_device_buffer flag;status=flag.allocate(device_id,4);if(status!=RNS8_SUCCESS) return status;
  int invalid=0;status=hip_direct_copy_host_to_device(device_id,flag.get(),&invalid,4);if(status!=RNS8_SUCCESS) return status;
  const auto launch=static_cast<hipError_t>(rns8_hip_word_mod_validate(a,b,int(p.m),int(p.n),int(p.k),int(p.lda),int(p.ldb),p.modulus,static_cast<int*>(flag.get())));
  const auto drain=hipDeviceSynchronize();if(launch!=hipSuccess || drain!=hipSuccess) return RNS8_BACKEND_FAILURE;
  status=hip_direct_copy_device_to_host(device_id,&invalid,flag.get(),4);if(status!=RNS8_SUCCESS) return status;
  if(invalid) return RNS8_INVALID_ARGUMENT;
  commit_started=true;
  const auto error=timed_hip_operation("word_mod_double_add",[&]() {
    const auto issued=static_cast<hipError_t>(rns8_hip_word_mod_gemm(a,b,c,int(p.m),int(p.n),int(p.k),int(p.lda),int(p.ldb),int(p.ldc),p.modulus));
    const auto completed=hipDeviceSynchronize();return issued==hipSuccess?completed:issued;
  });
  return error==hipSuccess?RNS8_SUCCESS:RNS8_BACKEND_FAILURE;
#else
  (void)device_id;return RNS8_UNSUPPORTED_BACKEND;
#endif
}
