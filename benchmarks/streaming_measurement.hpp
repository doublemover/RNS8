#pragma once
#include <algorithm>
#include <cstdint>
namespace rns8::bench {
struct StreamingMeasurementPlan {
  uint32_t warmup_calls=0,measured_calls=0,batches=0;
  uint64_t total_calls=0;
};
inline bool streaming_measurement_plan(uint32_t warmups,uint32_t calls,uint32_t batches,StreamingMeasurementPlan& out) {
  if(warmups>1024 || !calls || calls>65536 || batches<3 || batches>64 || uint64_t(calls)*batches>1048576)return false;
  const auto warm=std::max(2u,calls);
  out={warm,calls,batches,uint64_t(warmups)*warm+uint64_t(calls)*batches};return true;
}
struct StreamingBatchObservation {
  uint32_t calls=0;
  double duration_us=0,setup_us=0;
  uint64_t first_source_version=0,device_allocate_calls=0,device_allocated_bytes=0;
};
}
