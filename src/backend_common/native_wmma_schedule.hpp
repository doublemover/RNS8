#ifndef RNS8_NATIVE_WMMA_SCHEDULE_HPP
#define RNS8_NATIVE_WMMA_SCHEDULE_HPP
#include "native_wmma_queue.hpp"
#if defined(__HIPCC__)
#define RNS8_QUEUE_HD __host__ __device__
#else
#define RNS8_QUEUE_HD
#endif
namespace rns8::detail {
enum class NativeWmmaScheduleMode : uint32_t { explicit_round_robin=0, task_major=1, compact_round_robin=2, ranked_round_robin=3 };
struct NativeWmmaScheduleConfig {
  uint32_t abi_version=1;
  NativeWmmaScheduleMode mode=NativeWmmaScheduleMode::explicit_round_robin;
  uint32_t claim_chunk=1; // 1..32; tail switches to singleton claims
  NativeWmmaDispatchPolicy dispatch_policy=NativeWmmaDispatchPolicy::native_uniform_only;
};
struct NativeWmmaTaskRange { uint32_t begin, end, tile_cols, groups; };
struct NativeWmmaRoundPhase { uint32_t first_work, round_begin, round_end, active_tasks; };
struct NativeWmmaScheduleHeader {
  uint32_t abi_version=1, mode=0, task_count=0, work_count=0;
  uint32_t phase_count=0, claim_chunk=1, reserved0=0, reserved1=0;
};

static_assert(sizeof(NativeWmmaTaskRange)==16 && sizeof(NativeWmmaRoundPhase)==16 && sizeof(NativeWmmaScheduleHeader)==32, "schedule ABI");

RNS8_QUEUE_HD inline bool native_wmma_schedule_header_valid(const NativeWmmaScheduleHeader& h) {
  if(!h.task_count || h.task_count>4096 || !h.work_count || !h.claim_chunk || h.claim_chunk>32 || h.phase_count>h.task_count)return false;
  if(h.mode==3)return h.abi_version==2 && h.phase_count && h.reserved0==(h.task_count+31)/32 && h.reserved1==2*h.reserved0+1;
  return h.abi_version==1 && h.mode<=2 && (h.mode==2?bool(h.phase_count):h.phase_count==0) && !h.reserved0 && !h.reserved1;
}
RNS8_QUEUE_HD inline uint32_t native_wmma_popcount32(uint32_t bits) {
  bits-=((bits>>1)&0x55555555u);bits=(bits&0x33333333u)+((bits>>2)&0x33333333u);
  bits=(bits+(bits>>4))&0x0f0f0f0fu;return (bits*0x01010101u)>>24;
}
struct NativeWmmaDecodeWork {uint64_t range_predicates=0,rank_probes=0;};

// Immutable directories are fully admitted on the host before upload. Decode is
// bounded: task-major uses binary range lookup; fair compact mode binary-searches
// distinct count breakpoints then scans at most task_count active-task bits.
// The explicit ABI2 candidate selects a bitmap word by prefix rank, then a bit
// in five bounded subdivisions; its all-active phase needs no directory lookup.
// No per-work list, host rediscovery/download, spin loop, or wave-width assumption.
RNS8_QUEUE_HD inline bool decode_native_wmma_schedule(const NativeWmmaScheduleHeader& header,
    const NativeWmmaTaskRange* ranges, const NativeWmmaRoundPhase* phases,
    const NativeWmmaTask* tasks, uint32_t ordinal, NativeWmmaWork& result,
    const uint32_t* rank_directory=nullptr,NativeWmmaDecodeWork* work=nullptr) {
  if(!native_wmma_schedule_header_valid(header) || !ranges || !tasks || ordinal>=header.work_count)return false;
  uint32_t task=0,local=0;
  if(header.mode==uint32_t(NativeWmmaScheduleMode::task_major)) {
    uint32_t lo=0,hi=header.task_count;
    while(lo<hi) { const uint32_t mid=lo+(hi-lo)/2; if(ranges[mid].end<=ordinal)lo=mid+1;else hi=mid; }
    if(lo==header.task_count || ordinal<ranges[lo].begin)return false;
    task=lo;local=ordinal-ranges[lo].begin;
  } else if(header.mode==uint32_t(NativeWmmaScheduleMode::compact_round_robin) || header.mode==3) {
    if(!phases || !header.phase_count || header.phase_count>header.task_count)return false;
    uint32_t lo=0,hi=header.phase_count;
    while(lo<hi) {const uint32_t mid=lo+(hi-lo)/2;if(phases[mid].first_work<=ordinal)lo=mid+1;else hi=mid;}
    if(!lo)return false;
    const auto phase=phases[lo-1];
    if(!phase.active_tasks || phase.active_tasks>header.task_count)return false;
    local=phase.round_begin+(ordinal-phase.first_work)/phase.active_tasks;
    if(local>=phase.round_end)return false;
    uint32_t rank=(ordinal-phase.first_work)%phase.active_tasks;
    if(header.mode==3 && phase.active_tasks==header.task_count)task=rank;
    else if(header.mode==3) {
      // Phase zero has every positive-count task active. Only later sparse
      // phases own a [prefix ranks][bitmap words] table.
      if(!rank_directory || lo<2)return false;
      const auto words=header.reserved0;
      const auto* prefix=rank_directory+uint64_t(lo-2)*header.reserved1;
      if(prefix[words]!=phase.active_tasks)return false;
      uint32_t begin=0,end=words;
      while(begin<end) {
        const uint32_t mid=begin+(end-begin)/2;if(work)++work->rank_probes;
        if(prefix[mid+1]<=rank)begin=mid+1;else end=mid;
      }
      if(begin>=words || prefix[begin]>rank)return false;
      uint32_t bits=prefix[words+1+begin];rank-=prefix[begin];
      if(rank>=native_wmma_popcount32(bits))return false;
      uint32_t bit=0;
      for(uint32_t width=16;width;width>>=1) {
        const uint32_t low=bits&((1u<<width)-1),count=native_wmma_popcount32(low);
        if(work)++work->rank_probes;
        if(rank>=count){rank-=count;bits>>=width;bit+=width;}else bits=low;
      }
      task=begin*32+bit;if(task>=header.task_count)return false;
    } else {
      bool found=false;
      for(uint32_t i=0;i<header.task_count;++i) {
        if(work)++work->range_predicates;
        if(ranges[i].end-ranges[i].begin>local) {
          if(!rank){task=i;found=true;break;} --rank;
        }
      }
      if(!found)return false;
    }
  } else return false;
  const auto range=ranges[task];
  if(!range.groups || !range.tile_cols || local>=range.end-range.begin)return false;
  const uint32_t tile=local/range.groups;
  result={task,tile/range.tile_cols,tile%range.tile_cols,(local%range.groups)*tasks[task].group};
  return result.first_plane<tasks[task].prefix;
}
RNS8_QUEUE_HD inline const char* native_wmma_schedule_identity(NativeWmmaScheduleMode mode) {
  return mode==NativeWmmaScheduleMode::explicit_round_robin?"explicit_round_robin_tile_channel_v1":
      mode==NativeWmmaScheduleMode::task_major?"task_major_prefix_directory_tile_channel_v1":
      mode==NativeWmmaScheduleMode::ranked_round_robin?"ranked_round_robin_bitmap_prefix_select_v1":
      mode==NativeWmmaScheduleMode::compact_round_robin?"compact_round_robin_breakpoint_rank_scan_v1":"invalid";
}
}
#undef RNS8_QUEUE_HD
#endif
