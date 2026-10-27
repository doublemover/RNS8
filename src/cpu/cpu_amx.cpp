#include "core/api_internal.hpp"
#include "cpu/cpu_amx.hpp"
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <limits>
#if defined(RNS8_CPU_PARALLEL_OPENMP) && RNS8_CPU_PARALLEL_OPENMP
#include <omp.h>
#endif
#if defined(RNS8_ENABLE_CPU_AMX) && RNS8_ENABLE_CPU_AMX && defined(__linux__) && defined(__x86_64__) && \
    ((defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 11) || (defined(__clang__) && __clang_major__ >= 14))
#define RNS8_CPU_AMX_COMPILED 1
#include <cpuid.h>
#include <immintrin.h>
#include <sys/syscall.h>
#include <unistd.h>
#else
#define RNS8_CPU_AMX_COMPILED 0
#endif

namespace rns8::detail {
namespace {
constexpr uint32_t safe_k=65536;
constexpr uint32_t layout_version=1;
static_assert(uint64_t(safe_k)*128*128+128<=INT32_MAX,"previous centered residue must fit too");
struct alignas(64) Worker {
  int8_t a[16*64]{};
  int32_t accum[16*16]{};
};
struct alignas(64) TileConfig {
  uint8_t palette=1,start=0,reserved[14]{};
  uint16_t colsb[8]{};uint8_t reserved2[16]{};
  uint8_t rows[8]{};uint8_t reserved3[8]{};
};
static_assert(sizeof(TileConfig)==64);
bool mul(uint64_t a,uint64_t b,uint64_t& out) noexcept {
  if(b && a>SIZE_MAX/b)return false;out=a*b;return true;
}
uint32_t planes(const rns8_plan& p) {return api::uses_finite_storage(p.desc.semantics)?1:api::rns_storage_prefix_for_plan(p);}
}
struct CpuAmxWorkspace {
  uint64_t m=0,n=0,k=0,nt=0,kt=0,mt=0;uint32_t prefix=0;
  std::vector<Worker> workers;
  std::vector<int8_t> packed_b,output;
  uint64_t b_instance=0,b_generation=0,b_packs=0,b_reuses=0;
  uint32_t last_threads=0;
  uint16_t b_modulus=0;
  rns8_layout b_layout=RNS8_LAYOUT_ROW_MAJOR;
  int64_t b_logical_ld=0;
  bool b_ready=false;
};
bool cpu_amx_compiled() noexcept {return RNS8_CPU_AMX_COMPILED;}
uint32_t cpu_amx_workers() noexcept {
  const auto value=cpu_parallel_effective_threads();return std::max(1u,value);
}
bool cpu_amx_admitted(const rns8_cpu_amx_info& i) noexcept {
  return i.compiled && i.cpuid_tile && i.cpuid_int8 && i.osxsave && i.palette_supported &&
      i.os_supported && i.permission_enabled && i.xcr0_enabled;
}
rns8_status cpu_amx_probe(bool request,rns8_cpu_amx_info& i) noexcept {
  const auto size=i.struct_size;const auto abi=i.abi_version;i={};i.struct_size=size;i.abi_version=abi;
  i.compiled=cpu_amx_compiled();i.configured_threads=cpu_amx_workers();
#if RNS8_CPU_AMX_COMPILED
  unsigned a=0,b=0,c=0,d=0;
  if(__get_cpuid_max(0,nullptr)<0x1e)return RNS8_UNSUPPORTED_BACKEND;
  __cpuid_count(1,0,a,b,c,d);i.osxsave=bool(c&(1u<<27));
  __cpuid_count(7,0,a,b,c,d);i.cpuid_tile=bool(d&(1u<<24));i.cpuid_int8=bool(d&(1u<<25));
  __cpuid_count(0x1d,1,a,b,c,d);
  i.palette_supported=(a&65535)>=3072 && (a>>16)>=1024 && (b&65535)>=64 && (b>>16)>=3 && (c&65535)>=16;
  __cpuid_count(0x1e,0,a,b,c,d);i.palette_supported&=(b&255)>=16 && ((b>>8)&65535)>=64;
  if(__get_cpuid_max(0x80000000,nullptr)>=0x80000004){unsigned brand[12]{};
    for(unsigned j=0;j<3;++j)__cpuid(0x80000002+j,brand[4*j],brand[4*j+1],brand[4*j+2],brand[4*j+3]);
    std::memcpy(i.cpu_model,brand,48);}
  if(!i.cpuid_tile || !i.cpuid_int8 || !i.osxsave || !i.palette_supported)return RNS8_UNSUPPORTED_BACKEND;
  constexpr unsigned long get_supported=0x1021,get_permission=0x1022,request_permission=0x1023;
  constexpr uint64_t mask=(UINT64_C(1)<<17)|(UINT64_C(1)<<18);
  uint64_t supported=0,permission=0;
  i.os_supported=syscall(SYS_arch_prctl,get_supported,&supported)==0 && (supported&mask)==mask;
  if(!i.os_supported)return RNS8_UNSUPPORTED_BACKEND;
  if(syscall(SYS_arch_prctl,get_permission,&permission)!=0){i.permission_errno=errno;return RNS8_UNSUPPORTED_BACKEND;}
  if((permission&mask)!=mask && request){
    if(syscall(SYS_arch_prctl,request_permission,18)!=0){i.permission_errno=errno;return RNS8_UNSUPPORTED_BACKEND;}
    if(syscall(SYS_arch_prctl,get_permission,&permission)!=0){i.permission_errno=errno;return RNS8_UNSUPPORTED_BACKEND;}
  }
  i.permission_enabled=(permission&mask)==mask;
  unsigned lo=0,hi=0;__asm__ volatile("xgetbv":"=a"(lo),"=d"(hi):"c"(0));
  i.xcr0_enabled=(((uint64_t(hi)<<32)|lo)&mask)==mask;
#else
  (void)request;
#endif
  i.admitted=cpu_amx_admitted(i);return i.admitted?RNS8_SUCCESS:RNS8_UNSUPPORTED_BACKEND;
}
bool cpu_amx_workspace_bytes(int64_t m,int64_t n,int64_t k,uint32_t prefix,uint32_t workers,uint64_t& bytes) noexcept {
  bytes=0;if(m<=0 || n<=0 || k<=0 || !prefix || prefix>20 || !workers || workers>INT_MAX)return false;
  uint64_t b=0,c=0,w=0,t=0;
  if(!mul((uint64_t(n)+15)/16,(uint64_t(k)+63)/64,t) || !mul(t,1024*uint64_t(prefix),b) ||
      !mul(m,n,t) || !mul(t,prefix,c) || !mul(workers,sizeof(Worker),w))return false;
  if(b>SIZE_MAX-c || b+c>SIZE_MAX-w || b+c+w>SIZE_MAX-sizeof(CpuAmxWorkspace))return false;
  bytes=b+c+w+sizeof(CpuAmxWorkspace);return true;
}
rns8_status cpu_amx_prepare(const rns8_plan& p,std::shared_ptr<CpuAmxWorkspace>& out) {
  if(!cpu_amx_compiled())return RNS8_UNSUPPORTED_BACKEND;
  uint64_t bytes=0;if(!cpu_amx_workspace_bytes(p.desc.m,p.desc.n,p.desc.k,planes(p),p.cpu_amx_prepared_threads,bytes))return RNS8_RANGE_ERROR;
  auto state=std::make_shared<CpuAmxWorkspace>();state->m=p.desc.m;state->n=p.desc.n;state->k=p.desc.k;
  state->mt=(state->m+15)/16;state->nt=(state->n+15)/16;state->kt=(state->k+63)/64;state->prefix=planes(p);
  state->workers.resize(p.cpu_amx_prepared_threads);
  state->packed_b.resize(state->prefix*state->nt*state->kt*1024);
  state->output.resize(state->prefix*state->m*state->n);
  out=std::move(state);return RNS8_SUCCESS;
}
void cpu_amx_pack_b_tile(const int8_t* b,int64_t k,int64_t n,int64_t ld,int64_t kb,int64_t nb,int8_t* tile) noexcept {
  for(int x=0;x<64;++x)for(int y=0;y<16;++y)
    tile[(x/4)*64+4*y+x%4]=(kb+x<k && nb+y<n)?b[(kb+x)*ld+nb+y]:0;
}
#if RNS8_CPU_AMX_COMPILED
// Only called after every worker passed OS/xstate admission. Target isolation
// prevents AMX instructions entering context creation, dispatch or the oracle.
__attribute__((target("amx-tile,amx-int8"),noinline))
static void tile_job(const rns8_plan& p,const rns8_matrix& a,CpuAmxWorkspace& s,Worker& w,uint64_t job) noexcept {
  const uint64_t nt=job%s.nt,mt=(job/s.nt)%s.mt,plane=job/(s.nt*s.mt),row=mt*16,col=nt*16;
  if(!p.tile_schedule.empty()){
    const auto& entry=p.tile_schedule[(row/p.desc.tile_m)*p.schedule_tile_cols+col/p.desc.tile_n];
    if(plane>=entry.selected_prefix || (entry.flags&RNS8_TILE_SCHEDULE_ZERO_OUTPUT))return;
  }
  const uint16_t modulus=api::uses_finite_storage(p.desc.semantics)?p.desc.finite_modulus:kDefaultModuli[plane];
  TileConfig cfg{};for(unsigned i=0;i<3;++i){cfg.colsb[i]=64;cfg.rows[i]=16;}
  _tile_loadconfig(&cfg);
  std::fill(std::begin(w.accum),std::end(w.accum),0);
  const auto* source=a.residues.data()+plane*s.m*s.k;
  for(uint64_t first=0;first<s.k;first+=safe_k){
    _tile_loadd(0,w.accum,64);
    const uint64_t end=std::min(s.k,first+safe_k);
    for(uint64_t kb=first;kb<end;kb+=64){
      for(uint64_t r=0;r<16;++r)for(uint64_t x=0;x<64;++x)
        w.a[r*64+x]=(row+r<s.m && kb+x<end)?source[(row+r)*s.k+kb+x]:0;
      _tile_loadd(1,w.a,64);
      const auto* packed=s.packed_b.data()+((plane*s.nt+nt)*s.kt+kb/64)*1024;
      _tile_loadd(2,packed,64);_tile_dpbssd(0,1,2);
    }
    _tile_stored(0,w.accum,64);
    // Reduce all tile cells locally; no whole-matrix INT32 scratch/global pass.
    for(auto& v:w.accum){int32_t r=v%modulus;if(r<0)r+=modulus;if(r>=int32_t((modulus+1)/2))r-=modulus;v=r;}
  }
  _tile_release();
  auto* dst=s.output.data()+plane*s.m*s.n;
  for(uint64_t r=0;r<16 && row+r<s.m;++r)for(uint64_t c=0;c<16 && col+c<s.n;++c)
    dst[(row+r)*s.n+col+c]=static_cast<int8_t>(w.accum[r*16+c]);
}
#endif
rns8_status cpu_amx_gemm(const rns8_plan& p,const rns8_matrix& a,const rns8_matrix& b,rns8_matrix& c,CpuAmxWorkspace& s) {
  if(!cpu_amx_compiled())return RNS8_UNSUPPORTED_BACKEND;
  if(s.m!=uint64_t(p.desc.m) || s.n!=uint64_t(p.desc.n) || s.k!=uint64_t(p.desc.k) || s.prefix!=planes(p) ||
      s.workers.empty() || !a.host_residues_current || !b.host_residues_current ||
      a.residues.size()<s.m*s.k*s.prefix || b.residues.size()<s.k*s.n*s.prefix || c.residues.size()<s.output.size())return RNS8_INVALID_ARGUMENT;
  const bool reuse=s.b_ready && !b.generation_exhausted && s.b_instance==b.matrix_instance_id && s.b_generation==b.content_generation &&
      s.b_modulus==b.finite_modulus && s.b_layout==b.desc.logical_layout && s.b_logical_ld==b.desc.logical_ld;
  if(!reuse){s.b_ready=false;
    for(uint64_t plane=0;plane<s.prefix;++plane)for(uint64_t nt=0;nt<s.nt;++nt)for(uint64_t kt=0;kt<s.kt;++kt)
      cpu_amx_pack_b_tile(b.residues.data()+plane*s.k*s.n,s.k,s.n,s.n,kt*64,nt*16,
          s.packed_b.data()+((plane*s.nt+nt)*s.kt+kt)*1024);
    s.b_instance=b.matrix_instance_id;s.b_generation=b.content_generation;s.b_modulus=b.finite_modulus;
    s.b_layout=b.desc.logical_layout;s.b_logical_ld=b.desc.logical_ld;s.b_ready=true;
    if(s.b_packs!=UINT64_MAX)++s.b_packs;
  }else if(s.b_reuses!=UINT64_MAX)++s.b_reuses;
  std::fill(s.output.begin(),s.output.end(),0);
  std::atomic<bool> admitted{true};const uint64_t jobs=s.prefix*s.mt*s.nt;
  uint32_t workers=1;
#if defined(RNS8_CPU_PARALLEL_OPENMP) && RNS8_CPU_PARALLEL_OPENMP
  if(!omp_in_parallel() && cpu_parallel_should_use(cpu_parallel_saturating_mul3(s.m,s.n,s.k)))
    workers=std::min<uint64_t>(std::min<uint64_t>(s.workers.size(),cpu_amx_workers()),jobs);
#endif
  s.last_threads=1;
#if RNS8_CPU_AMX_COMPILED
#if defined(RNS8_CPU_PARALLEL_OPENMP) && RNS8_CPU_PARALLEL_OPENMP
#pragma omp parallel num_threads(workers) if(workers>1)
#endif
  {
    rns8_cpu_amx_info info{};
    if(cpu_amx_probe(true,info)!=RNS8_SUCCESS)admitted.store(false,std::memory_order_relaxed);
    unsigned worker=0,team=1;
#if defined(RNS8_CPU_PARALLEL_OPENMP) && RNS8_CPU_PARALLEL_OPENMP
    worker=omp_get_thread_num();team=omp_get_num_threads();
#pragma omp barrier
#pragma omp single
    s.last_threads=team;
#endif
    if(admitted.load(std::memory_order_relaxed))for(uint64_t job=worker;job<jobs;job+=team)tile_job(p,a,s,s.workers[worker],job);
  }
#else
  (void)jobs;(void)workers;admitted=false;
#endif
  if(!admitted)return RNS8_UNSUPPORTED_BACKEND;
  std::copy(s.output.begin(),s.output.end(),c.residues.begin());return RNS8_SUCCESS;
}
rns8_status cpu_amx_workspace_info(const CpuAmxWorkspace& s,rns8_cpu_amx_workspace_info& o) noexcept {
  o.prepared_threads=s.workers.size();o.last_execution_threads=s.last_threads;
  o.packed_b_bytes=s.packed_b.capacity();o.output_staging_bytes=s.output.capacity();
  o.owned_payload_bytes=sizeof(s)+o.packed_b_bytes+o.output_staging_bytes+s.workers.capacity()*sizeof(Worker);
  o.b_matrix_instance=s.b_instance;o.b_generation=s.b_generation;o.b_pack_count=s.b_packs;o.b_reuse_count=s.b_reuses;
  o.layout_version=layout_version;o.safe_k=safe_k;return RNS8_SUCCESS;
}
}
rns8_status rns8_query_cpu_amx(uint32_t request,rns8_cpu_amx_info* out) {
  if(!out || request>1 || !rns8::detail::valid_abi(out->struct_size,out->abi_version,sizeof(*out)))return RNS8_INVALID_ARGUMENT;
  // Query itself succeeds even when execution is unavailable; fields distinguish causes.
  rns8::detail::cpu_amx_probe(request!=0,*out);return RNS8_SUCCESS;
}
rns8_status rns8_get_cpu_amx_workspace_info(const rns8_workspace* workspace,rns8_cpu_amx_workspace_info* out) {
  if(!workspace || !out || !rns8::detail::valid_abi(out->struct_size,out->abi_version,sizeof(*out)))return RNS8_INVALID_ARGUMENT;
  if(workspace->in_flight)return RNS8_BUSY;
  if(workspace->backend!=RNS8_BACKEND_CPU_AMX || !workspace->cpu_amx)return RNS8_UNSUPPORTED_BACKEND;
  const auto size=out->struct_size;const auto abi=out->abi_version;*out={};out->struct_size=size;out->abi_version=abi;
  return rns8::detail::cpu_amx_workspace_info(*workspace->cpu_amx,*out);
}
