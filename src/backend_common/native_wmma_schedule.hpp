#ifndef RNS8_NATIVE_WMMA_SCHEDULE_HPP
#define RNS8_NATIVE_WMMA_SCHEDULE_HPP
#include "native_wmma_queue.hpp"
#if defined(__HIPCC__)
#define RNS8_QUEUE_HD __host__ __device__
#else
#define RNS8_QUEUE_HD
#endif
namespace rns8::detail {
enum class NativeWmmaScheduleMode : uint32_t { explicit_round_robin=0, task_major=1, compact_round_robin=2 };
struct NativeWmmaScheduleConfig {
  uint32_t abi_version=1;
  NativeWmmaScheduleMode mode=NativeWmmaScheduleMode::explicit_round_robin;
  uint32_t claim_chunk=1; // 1..32; tail switches to singleton claims
};
struct NativeWmmaTaskRange { uint32_t begin, end, tile_cols, groups; };
struct NativeWmmaRoundPhase { uint32_t first_work, round_begin, round_end, active_tasks; };
struct NativeWmmaScheduleHeader {
  uint32_t abi_version=1, mode=0, task_count=0, work_count=0;
  uint32_t phase_count=0, claim_chunk=1, reserved0=0, reserved1=0;
};

static_assert(sizeof(NativeWmmaTaskRange)==16 && sizeof(NativeWmmaRoundPhase)==16 && sizeof(NativeWmmaScheduleHeader)==32, "schedule ABI");

// Immutable directories are fully admitted on the host before upload. Decode is
// bounded: task-major uses binary range lookup; fair compact mode binary-searches
// distinct count breakpoints then scans at most task_count active-task bits.
// No per-work list, host rediscovery/download, spin loop, or wave-width assumption.
RNS8_QUEUE_HD inline bool decode_native_wmma_schedule(const NativeWmmaScheduleHeader& header,
    const NativeWmmaTaskRange* ranges, const NativeWmmaRoundPhase* phases,
    const NativeWmmaTask* tasks, uint32_t ordinal, NativeWmmaWork& result) {
  if(header.abi_version!=1 || !ranges || !tasks || !header.task_count || header.task_count>4096 ||
      ordinal>=header.work_count || header.claim_chunk<1 || header.claim_chunk>32) return false;
  uint32_t task=0,local=0;
  if(header.mode==uint32_t(NativeWmmaScheduleMode::task_major)) {
    uint32_t lo=0,hi=header.task_count;
    while(lo<hi) { const uint32_t mid=lo+(hi-lo)/2; if(ranges[mid].end<=ordinal)lo=mid+1;else hi=mid; }
    if(lo==header.task_count || ordinal<ranges[lo].begin)return false;
    task=lo;local=ordinal-ranges[lo].begin;
  } else if(header.mode==uint32_t(NativeWmmaScheduleMode::compact_round_robin)) {
    if(!phases || !header.phase_count || header.phase_count>header.task_count)return false;
    uint32_t lo=0,hi=header.phase_count;
    while(lo<hi) {const uint32_t mid=lo+(hi-lo)/2;if(phases[mid].first_work<=ordinal)lo=mid+1;else hi=mid;}
    if(!lo)return false;
    const auto phase=phases[lo-1];
    if(!phase.active_tasks || phase.active_tasks>header.task_count)return false;
    local=phase.round_begin+(ordinal-phase.first_work)/phase.active_tasks;
    if(local>=phase.round_end)return false;
    uint32_t rank=(ordinal-phase.first_work)%phase.active_tasks;
    bool found=false;
    for(uint32_t i=0;i<header.task_count;++i) {
      if(ranges[i].end-ranges[i].begin>local) {
        if(!rank){task=i;found=true;break;} --rank;
      }
    }
    if(!found)return false;
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
      mode==NativeWmmaScheduleMode::compact_round_robin?"compact_round_robin_breakpoint_rank_scan_v1":"invalid";
}
}
#undef RNS8_QUEUE_HD
#endif
