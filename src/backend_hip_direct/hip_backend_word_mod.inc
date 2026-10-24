#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
extern "C" int rns8_hip_word_mod_floating(const void*,const void*,void*,int,int,int,int,int,int,uint64_t,word_mod::FloatingSchedule);
extern "C" int rns8_hip_word_mod_wmma_capability(int*);
extern "C" int rns8_hip_word_mod_wmma(const void*,const void*,void*,int,int,int,int,int,int,uint64_t,word_mod::DigitWeights);
extern "C" int rns8_hip_word_mod_dot4_capability(int*);
extern "C" int rns8_hip_word_mod_dot4_width(const void*,const void*,void*,int,int,int,int,int,int,uint64_t,word_mod::DigitWeights);
extern "C" int rns8_hip_word_mod_dot4(const void*,const void*,void*,int,int,int,int,int,int,uint64_t,word_mod::DigitWeights);
extern "C" int rns8_hip_word_mod_validate(const void*,const void*,int,int,int,int,int,uint64_t,int*);
extern "C" int rns8_hip_word_mod_gemm(const void*,const void*,void*,int,int,int,int,int,int,uint64_t);
#endif
rns8_status hip_direct_word_mod_device(int device_id,const void* a,const void* b,void* c,
    const word_mod::Problem& p,uint64_t budget,bool& commit_started,bool packed_dot4,const word_mod::FloatingRequest* floating,bool byte_wmma,bool canonical_dot4) {
  commit_started=false;const auto s=word_mod::query(p,true);if(s.status!=RNS8_SUCCESS) return s.status;
  if(!a || !b || !c || reinterpret_cast<uintptr_t>(a)%8 || reinterpret_cast<uintptr_t>(b)%8 ||
     reinterpret_cast<uintptr_t>(c)%8 || !nonoverlapping_byte_ranges(a,s.a_bytes,c,s.c_bytes) ||
     !nonoverlapping_byte_ranges(b,s.b_bytes,c,s.c_bytes)) return RNS8_INVALID_ARGUMENT;
  if(budget<4) return RNS8_WORKSPACE_TOO_SMALL;
  if(canonical_dot4 && (!packed_dot4 || floating || byte_wmma))return RNS8_INVALID_ARGUMENT;
  if(byte_wmma && (packed_dot4 || floating)) return RNS8_INVALID_ARGUMENT;
  word_mod::FloatingSchedule schedule;
  if(floating) {
    if(packed_dot4) return RNS8_INVALID_ARGUMENT;
    const auto admitted=word_mod::floating_schedule(p.modulus,*floating,schedule);
    if(admitted!=RNS8_SUCCESS) return admitted;
  }
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  auto status=set_hip_device(device_id);if(status!=RNS8_SUCCESS) return status;
  hip_probe_buffer flag;status=flag.allocate(device_id,4);if(status!=RNS8_SUCCESS) return status;
  if(packed_dot4 || byte_wmma) {
    flag.mark_pending();
    const auto launched=static_cast<hipError_t>((byte_wmma?rns8_hip_word_mod_wmma_capability:rns8_hip_word_mod_dot4_capability)(static_cast<int*>(flag.get())));
    const auto completed=hipDeviceSynchronize();if(completed==hipSuccess) flag.mark_completed();
    if(launched!=hipSuccess || completed!=hipSuccess) return RNS8_BACKEND_FAILURE;
    int supported=0;flag.mark_pending();
    status=hip_direct_copy_device_to_host(device_id,&supported,flag.get(),4);
    if(status==RNS8_SUCCESS) flag.mark_completed();
    if(status!=RNS8_SUCCESS) return status;if(!supported) return RNS8_UNSUPPORTED_BACKEND;
  }
  word_mod::DigitWeights weights{};if(packed_dot4 || byte_wmma) weights=word_mod::digit_weights(p.modulus);
  int invalid=0;flag.mark_pending();
  status=hip_direct_copy_host_to_device(device_id,flag.get(),&invalid,4);if(status!=RNS8_SUCCESS) return status;
  flag.mark_completed();
  flag.mark_pending();
  const auto launch=static_cast<hipError_t>(rns8_hip_word_mod_validate(a,b,int(p.m),int(p.n),int(p.k),int(p.lda),int(p.ldb),p.modulus,static_cast<int*>(flag.get())));
  const auto drain=hipDeviceSynchronize();if(drain==hipSuccess) flag.mark_completed();
  if(launch!=hipSuccess || drain!=hipSuccess) return RNS8_BACKEND_FAILURE;
  flag.mark_pending();
  status=hip_direct_copy_device_to_host(device_id,&invalid,flag.get(),4);if(status!=RNS8_SUCCESS) return status;
  flag.mark_completed();
  if(invalid) return RNS8_INVALID_ARGUMENT;
  commit_started=true;
  const auto error=timed_hip_operation(canonical_dot4?"word_mod_byte_dot4_canonical_width":byte_wmma?"word_mod_byte_wmma":floating?"word_mod_asymmetric_binary64":packed_dot4?"word_mod_byte_dot4":"word_mod_double_add",[&]() {
    const auto issued=static_cast<hipError_t>(floating?
        rns8_hip_word_mod_floating(a,b,c,int(p.m),int(p.n),int(p.k),int(p.lda),int(p.ldb),int(p.ldc),p.modulus,schedule):byte_wmma?
        rns8_hip_word_mod_wmma(a,b,c,int(p.m),int(p.n),int(p.k),int(p.lda),int(p.ldb),int(p.ldc),p.modulus,weights):packed_dot4?
        (canonical_dot4?rns8_hip_word_mod_dot4_width:rns8_hip_word_mod_dot4)(a,b,c,int(p.m),int(p.n),int(p.k),int(p.lda),int(p.ldb),int(p.ldc),p.modulus,weights):
        rns8_hip_word_mod_gemm(a,b,c,int(p.m),int(p.n),int(p.k),int(p.lda),int(p.ldb),int(p.ldc),p.modulus));
    const auto completed=hipDeviceSynchronize();return issued==hipSuccess?completed:issued;
  });
  return error==hipSuccess?RNS8_SUCCESS:RNS8_BACKEND_FAILURE;
#else
  (void)device_id;(void)packed_dot4;(void)byte_wmma;return RNS8_UNSUPPORTED_BACKEND;
#endif
}
