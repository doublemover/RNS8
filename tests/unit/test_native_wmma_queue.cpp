#include "core/internal.hpp"
#include "backend_common/native_wmma_queue_plan.hpp"
#include "backend_common/submission_topology.hpp"
#include "core/host_control_observations.hpp"
#include <catch2/catch_test_macros.hpp>
#include <set>
#include <tuple>
#include <thread>

using namespace rns8::detail;

TEST_CASE("host-control observers restore nested calling-thread scopes on normal and exceptional exit") {
  HostControlObservations outer,inner;
  CHECK(current_host_control_observer()==nullptr);
  {
    HostControlObservationScope observation(outer);
    record_host_control(&HostControlObservations::submission_proofs);
    bool foreign_empty=false;HostControlObservations other_thread;
    std::thread worker([&](){foreign_empty=current_host_control_observer()==nullptr;
      HostControlObservationScope local(other_thread);record_host_control(&HostControlObservations::event_queries);});
    worker.join();CHECK(foreign_empty);CHECK(current_host_control_observer()==&outer);
    CHECK(other_thread.event_queries==(host_control_diagnostics_compiled()?1:0));CHECK(outer.event_queries==0);
    try {HostControlObservationScope nested(inner);record_host_control(&HostControlObservations::event_waits);throw 1;}catch(int){}
    CHECK(current_host_control_observer()==&outer);
    record_host_control(&HostControlObservations::submission_proofs);
    outer.event_records=UINT64_MAX;record_host_control(&HostControlObservations::event_records);CHECK(outer.event_records==UINT64_MAX);
  }
  CHECK(current_host_control_observer()==nullptr);
  CHECK(outer.submission_proofs==(host_control_diagnostics_compiled()?2:0));
  CHECK(inner.event_waits==(host_control_diagnostics_compiled()?1:0));
  record_host_control(&HostControlObservations::submission_proofs);
  CHECK(outer.submission_proofs==(host_control_diagnostics_compiled()?2:0));
}

TEST_CASE("one batch admits its borrowed stream once and never retains that authority across batches") {
  int context=0,other_context=0,stream=0,other_stream=0;unsigned queries=0;
  const auto check=[&](){++queries;return RNS8_SUCCESS;};
  BorrowedStreamAdmission batch;
  for(unsigned child=0;child<64;++child)REQUIRE(batch.admit(&context,&stream,check)==RNS8_SUCCESS);
  CHECK(queries==1);
  CHECK(batch.admit(&other_context,&stream,check)==RNS8_INVALID_ARGUMENT);
  CHECK(batch.admit(&context,&other_stream,check)==RNS8_INVALID_ARGUMENT);CHECK(queries==1);
  BorrowedStreamAdmission next;REQUIRE(next.admit(&context,&stream,check)==RNS8_SUCCESS);CHECK(queries==2);
  BorrowedStreamAdmission failed;
  CHECK(failed.admit(&context,&stream,[&](){++queries;return RNS8_UNSUPPORTED_BACKEND;})==RNS8_UNSUPPORTED_BACKEND);
  REQUIRE(failed.admit(&context,&stream,check)==RNS8_SUCCESS);CHECK(queries==4);
  BorrowedStreamAdmission exceptional;
  CHECK_THROWS(exceptional.admit(&context,&stream,[]()->rns8_status{throw 1;}));
  REQUIRE(exceptional.admit(&context,&stream,check)==RNS8_SUCCESS);CHECK(queries==5);
  BorrowedStreamAdmission empty;
  CHECK(empty.admit(nullptr,&stream,check)==RNS8_INVALID_ARGUMENT);
  CHECK(empty.admit(&context,nullptr,check)==RNS8_INVALID_ARGUMENT);CHECK(queries==5);
}
TEST_CASE("compiled group reader multiplicities retain exact shared-reader overflow admission") {
  struct Reader {uint64_t in_flight_readers=0;};
  Reader a[64],shared;std::vector<Reader*> bindings;
  for(auto& reader:a){bindings.push_back(&reader);bindings.push_back(&shared);}
  const auto counts=compile_reader_multiplicities(std::move(bindings));REQUIRE(counts.size()==65);
  uint64_t total=0;for(const auto& entry:counts){total+=entry.second;CHECK(entry.second==(entry.first==&shared?64:1));}
  CHECK(total==128);CHECK(reader_multiplicities_fit(counts));
  shared.in_flight_readers=UINT64_MAX-64;CHECK(reader_multiplicities_fit(counts));
  ++shared.in_flight_readers;CHECK_FALSE(reader_multiplicities_fit(counts));
  shared.in_flight_readers=0;a[17].in_flight_readers=UINT64_MAX;CHECK_FALSE(reader_multiplicities_fit(counts));
  const auto single=compile_reader_multiplicities(std::vector<Reader*>{&shared});
  REQUIRE(single.size()==1);CHECK(single[0].second==1); // A==B is reserved only once by each child.
  CHECK_FALSE(reader_multiplicities_fit(compile_reader_multiplicities(std::vector<Reader*>{nullptr})));
}

TEST_CASE("native queue specialization binds group signedness representation and explicit policy") {
  using Policy=NativeWmmaDispatchPolicy;
  CHECK(native_wmma_uniform_hint({}).group==0);
  uint32_t encoded_storage=0;
  for(uint32_t group:{1u,2u,4u})for(uint32_t u:{0u,1u})for(bool encoded:{false,true}) {
    NativeWmmaTask first{},second{};first.group=second.group=group;
    first.unsigned_inputs=second.unsigned_inputs=u;
    first.m=17;second.m=3;first.prefix=20;second.prefix=1;
    if(encoded)first.encoded_b=second.encoded_b=&encoded_storage;
    for(auto policy:{Policy::native_uniform_only,Policy::uniform_native_or_encoded,Policy::generic_task_dispatch}) {
      const bool specialized=policy!=Policy::generic_task_dispatch && (!encoded || policy==Policy::uniform_native_or_encoded);
      const auto hint=native_wmma_uniform_hint({first,second},policy);
      CHECK(hint.group==(specialized?group:0));CHECK(hint.unsigned_inputs==(specialized?u:0));
      CHECK(hint.encoded_b==uint32_t(specialized && encoded));
    }
    const auto generic=[&](){CHECK(native_wmma_uniform_hint({first,second},Policy::uniform_native_or_encoded).group==0);};
    second.unsigned_inputs^=1;generic();second.unsigned_inputs=u;
    second.group=group==1?2:1;generic();second.group=group;
    second.encoded_b=encoded?nullptr:&encoded_storage;generic();second.encoded_b=first.encoded_b;
    first.group=3;generic();first.group=group;first.unsigned_inputs=2;generic();
  }
  CHECK(native_wmma_uniform_hint({NativeWmmaTask{}},static_cast<Policy>(UINT32_MAX)).group==0);
}

TEST_CASE("native queue bounds offset K fragments through the last accessed element") {
  uint64_t allocation[12]{}, b = 1;
  NativeWmmaTask task{allocation + 3, &b, reinterpret_cast<int8_t*>(allocation + 8),
                      2, 1, 1, 4, 1, 1, 1, 1, 0};
  std::vector<NativeWmmaWork> work;
  uint64_t bytes = 0;
  REQUIRE(build_native_wmma_queue({task}, 1, UINT64_MAX, work, bytes) == RNS8_SUCCESS);
  task.c = reinterpret_cast<int8_t*>(allocation + 7);
  REQUIRE(build_native_wmma_queue({task}, 1, UINT64_MAX, work, bytes) == RNS8_INVALID_ARGUMENT);
}

TEST_CASE("native WMMA queue covers heterogeneous tiles exactly once") {
  std::vector<uint64_t> a(33 * 19), b(19 * 35);
  std::vector<int8_t> c(20 * 33 * 35), d(c.size()), e(c.size());
  std::vector<NativeWmmaTask> tasks = {
      {a.data(), b.data(), c.data(), 33, 35, 19, 19, 35, 35, 20, 4, 0},
      {a.data(), b.data(), d.data(), 17, 21, 19, 19, 35, 35, 3, 2, 1},
      {a.data(), b.data(), e.data(), 1, 1, 19, 19, 35, 35, 1, 1, 0}};
  std::vector<NativeWmmaWork> work;
  uint64_t bytes = 0;
  REQUIRE(build_native_wmma_queue(tasks, 1024, UINT64_MAX, work, bytes) == RNS8_SUCCESS);
  REQUIRE(work.size() == 45 + 8 + 1);
  CHECK(work[0].task == 0);
  CHECK(work[1].task == 1);
  CHECK(work[2].task == 2);
  std::set<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>> expected, actual;
  for (uint32_t id = 0; id < tasks.size(); ++id) {
    const auto& t = tasks[id];
    for (uint32_t row = 0; row < (t.m + 15) / 16; ++row)
      for (uint32_t col = 0; col < (t.n + 15) / 16; ++col)
        for (uint32_t p = 0; p < t.prefix; p += t.group) expected.emplace(id, row, col, p);
  }
  for (const auto& w : work) REQUIRE(actual.emplace(w.task, w.tile_row, w.tile_col, w.first_plane).second);
  CHECK(actual == expected);
  CHECK(bytes == tasks.size() * sizeof(NativeWmmaTask) + work.size() * sizeof(NativeWmmaWork) + 4);
  const auto original_bytes = bytes;
  REQUIRE(build_native_wmma_queue(tasks, 1, bytes - 1, work, bytes) == RNS8_WORKSPACE_TOO_SMALL);
  CHECK(bytes == original_bytes);
  CHECK(work.size() == expected.size());
  for (uint32_t workers : {0u, 1025u})
    CHECK(build_native_wmma_queue(tasks, workers, UINT64_MAX, work, bytes) == RNS8_INVALID_ARGUMENT);
  tasks[1].c = tasks[0].c + 1;
  CHECK(build_native_wmma_queue(tasks, 1, UINT64_MAX, work, bytes) == RNS8_INVALID_ARGUMENT);
  tasks[1].c = reinterpret_cast<int8_t*>(a.data());
  CHECK(build_native_wmma_queue(tasks, 1, UINT64_MAX, work, bytes) == RNS8_INVALID_ARGUMENT);
  tasks[1].c = d.data();
  tasks[0].group = 3;
  CHECK(build_native_wmma_queue(tasks, 1, UINT64_MAX, work, bytes) == RNS8_INVALID_ARGUMENT);
  tasks[0].group = 4;
  tasks[0].lda = INT64_MAX;
  CHECK(build_native_wmma_queue(tasks, 1, UINT64_MAX, work, bytes) == RNS8_UNSUPPORTED_BACKEND);
  CHECK(bytes == original_bytes);
  CHECK(work.size() == expected.size());
}

TEST_CASE("native queue admits encoded spans and rejects encoded output hazards") {
  uint64_t a=1,b=2;
  int8_t c=0;
  uint32_t encoded[5]{};
  NativeWmmaTask task{&a,&b,&c,1,1,1,1,1,1,1,1,0,encoded,4};
  std::vector<NativeWmmaWork> work; uint64_t bytes=0;
  CHECK(native_wmma_uniform_hint({task}).group==0);
  REQUIRE(build_native_wmma_queue({task},1,UINT64_MAX,work,bytes)==RNS8_SUCCESS);
  task.encoded_k_words=3;
  CHECK(build_native_wmma_queue({task},1,UINT64_MAX,work,bytes)==RNS8_INVALID_ARGUMENT);
  task.encoded_k_words=4;
  task.c=reinterpret_cast<int8_t*>(encoded)+16; // retained capability word also belongs to cache
  CHECK(build_native_wmma_queue({task},1,UINT64_MAX,work,bytes)==RNS8_INVALID_ARGUMENT);
  task.c=&c; task.encoded_b=nullptr;
  CHECK(build_native_wmma_queue({task},1,UINT64_MAX,work,bytes)==RNS8_INVALID_ARGUMENT);
}

namespace {
auto queue_key(const NativeWmmaWork& w){return std::make_tuple(w.task,w.tile_row,w.tile_col,w.first_plane);}
NativeWmmaTask address_only_task(uint32_t id,int64_t m,int64_t n,int64_t k,uint32_t prefix,uint32_t group) {
  // Never dereferenced: address-domain/metadata tests need no operand allocation.
  const uintptr_t base=uintptr_t{1}<<40;
  const uintptr_t a=base*(id+1),b=a+(uintptr_t{1}<<37),c=a+(uintptr_t{1}<<38);
  return {reinterpret_cast<const void*>(a),reinterpret_cast<const void*>(b),reinterpret_cast<int8_t*>(c),
      m,n,k,k,n,n,prefix,group,id%2};
}
}
TEST_CASE("procedural native schedules preserve sets and compressed fair sequences") {
  for(uint32_t prefix=1;prefix<=20;++prefix)for(uint32_t group:{1u,2u,4u})
    for(int64_t edge:{1,15,16,17,33}) {
      std::vector<NativeWmmaTask> tasks={address_only_task(0,edge,17,19,prefix,group),
          address_only_task(1,17,edge,1,21-prefix,4),address_only_task(2,1,1,65537,1,2)};
      NativeWmmaSchedulePlan explicit_plan;
      REQUIRE(build_native_wmma_schedule(tasks,128,UINT64_MAX,{},explicit_plan)==RNS8_SUCCESS);
      std::set<decltype(queue_key(NativeWmmaWork{}))> expected;
      for(auto w:explicit_plan.explicit_work)REQUIRE(expected.insert(queue_key(w)).second);
      for(auto mode:{NativeWmmaScheduleMode::task_major,NativeWmmaScheduleMode::compact_round_robin,NativeWmmaScheduleMode::ranked_round_robin})
        for(uint32_t chunk:{1u,4u,32u}) {
          NativeWmmaSchedulePlan plan;
          REQUIRE(build_native_wmma_schedule(tasks,128,UINT64_MAX,{1,mode,chunk},plan)==RNS8_SUCCESS);
          REQUIRE(validate_native_wmma_schedule(tasks,128,plan)==RNS8_SUCCESS);
          CHECK(plan.header.work_count==explicit_plan.header.work_count);
          CHECK(plan.explicit_work.empty());
          std::set<decltype(queue_key(NativeWmmaWork{}))> actual;
          for(uint32_t i=0;i<plan.header.work_count;++i) {
            NativeWmmaWork w{};
            REQUIRE(decode_native_wmma_schedule(plan.header,plan.ranges.data(),plan.phases.data(),tasks.data(),i,w,plan.rank_directory.data()));
            REQUIRE(actual.insert(queue_key(w)).second);
            if(mode!=NativeWmmaScheduleMode::task_major)CHECK(queue_key(w)==queue_key(explicit_plan.explicit_work[i]));
          }
          CHECK(actual==expected);
          NativeWmmaWork sentinel{99,99,99,99};
          CHECK_FALSE(decode_native_wmma_schedule(plan.header,plan.ranges.data(),plan.phases.data(),tasks.data(),plan.header.work_count,sentinel));
          CHECK(sentinel.task==99);
        }
    }
}
TEST_CASE("ranked fair queues remove scalar directory scans while preserving complete phase order") {
  for(uint32_t count:{1u,64u,512u,4096u})for(uint32_t shape:{0u,1u,2u}) {
    std::vector<NativeWmmaTask> tasks;
    for(uint32_t i=0;i<count;++i) {
      const uint32_t tiles=shape==0?1:shape==1?(i+1==count?37:1):1+(i%31);
      tasks.push_back(address_only_task(i,int64_t(tiles)*16,1,1,1,1));
    }
    NativeWmmaSchedulePlan ranked,explicit_plan,scan;
    REQUIRE(build_native_wmma_schedule(tasks,128,UINT64_MAX,{1,NativeWmmaScheduleMode::ranked_round_robin,32},ranked)==RNS8_SUCCESS);
    REQUIRE(validate_native_wmma_schedule(tasks,128,ranked)==RNS8_SUCCESS);
    REQUIRE(build_native_wmma_schedule(tasks,128,UINT64_MAX,{},explicit_plan)==RNS8_SUCCESS);
    REQUIRE(build_native_wmma_schedule(tasks,128,UINT64_MAX,{1,NativeWmmaScheduleMode::compact_round_robin,32},scan)==RNS8_SUCCESS);
    CHECK(ranked.header.abi_version==2);CHECK(ranked.header.work_count==explicit_plan.header.work_count);
    CHECK(ranked.rank_directory.size()==(ranked.phases.size()-1)*ranked.header.reserved1);
    CHECK(ranked.device_bytes==scan.device_bytes+ranked.rank_directory.size()*4);
    NativeWmmaDecodeWork ranked_work,scan_work;
    for(uint32_t ordinal=0;ordinal<ranked.header.work_count;++ordinal) {
      NativeWmmaWork output{};
      REQUIRE(decode_native_wmma_schedule(ranked.header,ranked.ranges.data(),ranked.phases.data(),tasks.data(),ordinal,output,ranked.rank_directory.data(),&ranked_work));
      REQUIRE(queue_key(output)==queue_key(explicit_plan.explicit_work[ordinal]));
      if(shape==0)REQUIRE(decode_native_wmma_schedule(scan.header,scan.ranges.data(),scan.phases.data(),tasks.data(),ordinal,output,nullptr,&scan_work));
    }
    CHECK(ranked_work.range_predicates==0);CHECK(ranked_work.rank_probes<=uint64_t(ranked.header.work_count)*12);
    if(shape==0) {
      CHECK(ranked.rank_directory.empty());CHECK(ranked_work.rank_probes==0);
      CHECK(scan_work.range_predicates==uint64_t(count)*(count+1)/2);
    }
    const auto original=ranked;
    CHECK(build_native_wmma_schedule(tasks,128,ranked.device_bytes-1,{1,NativeWmmaScheduleMode::ranked_round_robin,32},ranked)==RNS8_WORKSPACE_TOO_SMALL);
    CHECK(ranked.device_bytes==original.device_bytes);CHECK(ranked.rank_directory==original.rank_directory);
    if(!ranked.rank_directory.empty()) {
      NativeWmmaWork sentinel{99,99,99,99};
      CHECK_FALSE(decode_native_wmma_schedule(ranked.header,ranked.ranges.data(),ranked.phases.data(),tasks.data(),ranked.phases[1].first_work,sentinel));
      CHECK(sentinel.task==99);
      auto bad=ranked;bad.rank_directory.pop_back();CHECK(validate_native_wmma_schedule(tasks,128,bad)==RNS8_INVALID_ARGUMENT);
      bad=ranked;bad.rank_directory.back()^=1;CHECK(validate_native_wmma_schedule(tasks,128,bad)==RNS8_INVALID_ARGUMENT);
      bad=ranked;++bad.rank_directory[1];CHECK(validate_native_wmma_schedule(tasks,128,bad)==RNS8_INVALID_ARGUMENT);
    }
    auto bad=ranked;bad.header.abi_version=1;CHECK(validate_native_wmma_schedule(tasks,128,bad)==RNS8_INVALID_ARGUMENT);
    bad=ranked;++bad.header.reserved1;CHECK(validate_native_wmma_schedule(tasks,128,bad)==RNS8_INVALID_ARGUMENT);
  }
}
TEST_CASE("procedural metadata rejects corrupt directories before publication") {
  std::vector<NativeWmmaTask> tasks={address_only_task(0,33,17,19,20,1),address_only_task(1,1,1,7,3,2)};
  NativeWmmaSchedulePlan plan;
  REQUIRE(build_native_wmma_schedule(tasks,2,UINT64_MAX,{1,NativeWmmaScheduleMode::compact_round_robin,4},plan)==RNS8_SUCCESS);
  const auto original=plan;
  for(unsigned mutation=0;mutation<16;++mutation) {
    auto bad=original;
    switch(mutation) {
      case 0:bad.header.abi_version=2;break;case 1:bad.header.mode=3;break;
      case 2:bad.header.claim_chunk=0;break;case 3:bad.header.claim_chunk=33;break;
      case 4:++bad.header.work_count;break;case 5:--bad.header.task_count;break;
      case 6:bad.header.reserved0=1;break;case 7:bad.ranges[1]=bad.ranges[0];break;
      case 8:++bad.ranges[0].end;break;case 9:bad.ranges[0].tile_cols=0;break;
      case 10:bad.ranges.erase(bad.ranges.begin());break;case 11:bad.phases[0].first_work=1;break;
      case 12:bad.phases[0].active_tasks=0;break;case 13:--bad.phases[0].round_end;break;
      case 14:bad.phases.pop_back();break;case 15:++bad.device_bytes;break;
    }
    CHECK(validate_native_wmma_schedule(tasks,2,bad)!=RNS8_SUCCESS);
  }
  CHECK(build_native_wmma_schedule(tasks,2,plan.device_bytes-1,{1,NativeWmmaScheduleMode::compact_round_robin,4},plan)==RNS8_WORKSPACE_TOO_SMALL);
  CHECK(plan.device_bytes==original.device_bytes);
  CHECK(validate_native_wmma_schedule(tasks,2,plan)==RNS8_SUCCESS);
  CHECK(build_native_wmma_schedule({},2,UINT64_MAX,{},plan)==RNS8_INVALID_ARGUMENT);
  tasks[0].m=INT64_MAX;
  CHECK(build_native_wmma_schedule(tasks,2,UINT64_MAX,{},plan)==RNS8_UNSUPPORTED_BACKEND);
}
TEST_CASE("compact queue admits capacity without expanding millions of work records") {
  const std::vector<NativeWmmaTask> tasks={address_only_task(0,8192,8192,1,20,1)};
  NativeWmmaSchedulePlan plan;
  REQUIRE(build_native_wmma_schedule(tasks,1024,1024,{1,NativeWmmaScheduleMode::compact_round_robin,32},plan)==RNS8_SUCCESS);
  CHECK(plan.header.work_count==5242880);
  CHECK(uint64_t(plan.header.work_count)*sizeof(NativeWmmaWork)==80*1024*1024);
  CHECK(plan.device_bytes==sizeof(NativeWmmaTask)+sizeof(NativeWmmaTaskRange)+sizeof(NativeWmmaRoundPhase)+4);
  CHECK(build_native_wmma_schedule(tasks,1024,1024,{},plan)==RNS8_WORKSPACE_TOO_SMALL);
  CHECK(plan.header.mode==2);
}
TEST_CASE("procedural queue bounds 4096 tasks and maximum claim overshoot") {
  std::vector<NativeWmmaTask> tasks;
  for(uint32_t i=0;i<4096;++i)tasks.push_back(address_only_task(i,1,1,1,1,1));
  NativeWmmaSchedulePlan plan;
  REQUIRE(build_native_wmma_schedule(tasks,1024,UINT64_MAX,{1,NativeWmmaScheduleMode::compact_round_robin,32},plan)==RNS8_SUCCESS);
  CHECK(plan.header.work_count==4096);CHECK(plan.phases.size()==1);
  for(uint32_t i=0;i<4096;++i){NativeWmmaWork w{};REQUIRE(decode_native_wmma_schedule(plan.header,plan.ranges.data(),plan.phases.data(),tasks.data(),i,w));CHECK(w.task==i);}
  tasks.push_back(address_only_task(4096,1,1,1,1,1));
  CHECK(build_native_wmma_schedule(tasks,1,UINT64_MAX,{},plan)==RNS8_INVALID_ARGUMENT);
  tasks.clear();
  const int64_t largest_m=INT32_MAX-65536;
  const uint64_t per_task=uint64_t(largest_m+15)/16,target=UINT32_MAX-1024u*32u;
  const uint32_t full=uint32_t(target/per_task),tail=uint32_t(target%per_task);
  for(uint32_t i=0;i<full;++i)tasks.push_back(address_only_task(i,largest_m,1,1,1,1));
  REQUIRE(tail>0);tasks.push_back(address_only_task(full,int64_t(tail)*16,1,1,1,1));
  REQUIRE(build_native_wmma_schedule(tasks,1024,UINT64_MAX,{1,NativeWmmaScheduleMode::task_major,32},plan)==RNS8_SUCCESS);
  CHECK(plan.header.work_count==target);
  ++tasks.back().m;
  CHECK(build_native_wmma_schedule(tasks,1024,UINT64_MAX,{1,NativeWmmaScheduleMode::task_major,32},plan)==RNS8_RANGE_ERROR);
  CHECK(plan.header.work_count==target);
}

#include "backend_common/native_locality.hpp"
#include <random>
TEST_CASE("native locality is a bijection over disjoint fragments and partial modulus groups") {
  constexpr int m=17,n=19,k=23,prefix=7;
  std::vector<int64_t> a(m*k),b(k*n);for(size_t i=0;i<a.size();++i)a[i]=int(i%11)-5;for(size_t i=0;i<b.size();++i)b[i]=int(i%13)-6;
  for(uint32_t group:{1u,2u,4u})for(uint32_t fragments:{1u,2u,3u,7u}) {
    std::vector<int8_t> output(uint64_t(fragments)*prefix*m*n);
    std::vector<NativeWmmaTask> tasks;
    for(uint32_t f=0;f<fragments;++f){const int begin=k*f/fragments,end=k*(f+1)/fragments;
      tasks.push_back({a.data()+begin,b.data()+begin*n,output.data()+uint64_t(f)*prefix*m*n,m,n,end-begin,k,n,n,prefix,group,0});}
    std::vector<NativeWmmaWork> original;uint64_t bytes=0;
    REQUIRE(build_native_wmma_queue(tasks,8,UINT64_MAX,original,bytes)==RNS8_SUCCESS);
    const auto identities=[](const auto& list){std::set<std::tuple<uint32_t,uint32_t,uint32_t,uint32_t>> result;
      for(const auto& w:list)REQUIRE(result.emplace(w.task,w.tile_row,w.tile_col,w.first_plane).second);return result;};
    for(auto order:{RNS8_LOCALITY_ORDINARY,RNS8_LOCALITY_OUTPUT_NEIGHBOR,RNS8_LOCALITY_MODULUS,RNS8_LOCALITY_SAME_OUTPUT}) {
      rns8_locality_desc d{sizeof(d),RNS8_ABI_VERSION,order,0,1,1,8,1024};rns8_locality_info info{};uint64_t transient=0;auto work=original;
      REQUIRE(lower_native_locality(tasks,work,d,"gfx1100",info,transient)==RNS8_SUCCESS);
      CHECK(identities(work)==identities(original));CHECK(info.unique_source_bytes==(m+n)*k*8);CHECK(info.physical_affinity_established==0);
      CHECK(info.output_read_write_bytes==uint64_t(fragments)*prefix*m*n);CHECK(info.directory_bytes==work.size()*sizeof(NativeWmmaWork));
      std::shuffle(work.begin(),work.end(),std::mt19937(3637));std::fill(output.begin(),output.end(),99);
      for(auto w:work){const auto& t=tasks[w.task];for(int r=w.tile_row*16;r<std::min<int64_t>(t.m,w.tile_row*16+16);++r)
        for(int c=w.tile_col*16;c<std::min<int64_t>(t.n,w.tile_col*16+16);++c)for(uint32_t p=w.first_plane;p<std::min(t.prefix,w.first_plane+t.group);++p){
          int64_t sum=0;for(int q=0;q<t.k;++q)sum+=static_cast<const int64_t*>(t.a)[r*t.lda+q]*static_cast<const int64_t*>(t.b)[q*t.ldb+c];
          int value=int(sum%kDefaultModuli[p]);if(value<0)value+=kDefaultModuli[p];t.c[(p*m+r)*n+c]=int8_t(value>=int((kDefaultModuli[p]+1)/2)?value-kDefaultModuli[p]:value);}}
      for(int r=0;r<m;++r)for(int c=0;c<n;++c)for(uint32_t p=0;p<prefix;++p){int64_t expected=0,merged=0;
        for(int q=0;q<k;++q)expected+=a[r*k+q]*b[q*n+c];for(uint32_t f=0;f<fragments;++f)merged+=output[((f*prefix+p)*m+r)*n+c];
        CHECK((merged-expected)%kDefaultModuli[p]==0);}
      auto fallback=original;REQUIRE(lower_native_locality(tasks,fallback,d,"unknown",info,transient)==RNS8_SUCCESS);CHECK(info.applied==0);
      for(size_t i=0;i<original.size();++i)CHECK(std::tie(fallback[i].task,fallback[i].tile_row,fallback[i].tile_col,fallback[i].first_plane)==std::tie(original[i].task,original[i].tile_row,original[i].tile_col,original[i].first_plane));
    }
  }
}
