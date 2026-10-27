// This translation unit alone is compiled with +i8mm. It has no static
// initializers and is reachable only after OS HWCAP admission in cpu_i8mm.cpp.
#include <stdint.h>
#include <arm_neon.h>
static_assert(uint64_t(65536)*128*128+128<=INT32_MAX,"I8MM slabs include their prior centered residue");
extern "C" void rns8_cpu_i8mm_tile(const int8_t* a,const int8_t* packed_b,int8_t* output,
    int64_t m,int64_t n,int64_t k,int64_t row,int64_t col,uint32_t modulus) noexcept {
  int32x4_t acc=vdupq_n_s32(0);
  for(int64_t first=0;first<k;) {
    const int64_t depth=k-first>65536?65536:k-first,end=first+depth;
    for(int64_t kb=first;kb<end;kb+=8) {
      int8_t left[16]{};
      for(int r=0;r<2;++r)for(int q=0;q<8;++q)
        if(row+r<m && kb+q<end)left[r*8+q]=a[(row+r)*k+kb+q];
      acc=vmmlaq_s32(acc,vld1q_s8(left),vld1q_s8(packed_b+(kb/8)*16));
    }
    int32_t centered[4];vst1q_s32(centered,acc);
    for(auto& v:centered){int32_t r=v%int32_t(modulus);if(r<0)r+=int32_t(modulus);if(r>=int32_t((modulus+1)/2))r-=int32_t(modulus);v=r;}
    acc=vld1q_s32(centered);first=end;
  }
  int32_t centered[4];vst1q_s32(centered,acc);
  for(int r=0;r<2;++r)for(int c=0;c<2;++c)
    if(row+r<m && col+c<n)output[(row+r)*n+col+c]=static_cast<int8_t>(centered[r*2+c]);
}
