#pragma once
#include "stream_panel_plan.hpp"
#include <cmath>
#include <vector>
namespace rns8::detail {
// Uses the admitted runtime geometry and deterministic direct-mapped evictions.
// This is an endpoint list-schedule estimate, never observed overlap or evidence.
inline double estimate_stream_panel_call(const StreamPanelProblem& problem,const StreamPanelResources& resources,
    const rns8_stream_gemm_cost_profile& profile,bool upper,std::vector<uint64_t>& tags) {
  const auto& rates=upper?profile.stages.upper_ns_per_unit:profile.stages.lower_ns_per_unit;
  const double launch=upper?profile.stages.upper_launch_ns:profile.stages.lower_launch_ns;
  const double host_rate=upper?profile.upper_host_copy_ns_per_byte:profile.lower_host_copy_ns_per_byte;
  const double boundary=upper?profile.upper_boundary_ns:profile.lower_boundary_ns;
  const bool overlap=problem.schedule==RNS8_STREAM_GEMM_OVERLAPPED;
  const double duplex=overlap?(upper?profile.upper_duplex_multiplier:profile.lower_duplex_multiplier):1;
  const double scan_rate=upper?profile.upper_input_scan_ns_per_cell:profile.lower_input_scan_ns_per_cell;
  double host=(double(problem.m)*problem.k+double(problem.k)*problem.n)*scan_rate;
  double upload=0,compute=0,download=0,uploaded[2]{},consumed[2]{},downloaded[2]{},output_bytes[2]{};
  uint64_t tiles=0;
  StreamPanelCoverage coverage(problem,resources,1,1,1,2,1);StreamPanelStep step;
  while(coverage.next(step)) {
    const unsigned slot=unsigned(step.index&1);const uint64_t key=step.index%resources.b_panels;
    const bool hit=!tags.empty() && tags[key%tags.size()]==key;
    if(!hit && !tags.empty())tags[key%tags.size()]=key;
    const double input_cells=double(step.rows)*step.depth+(hit?0:double(step.depth)*step.cols);
    const double bytes=input_cells*8,conversions=input_cells*problem.prefix;
    const double upload_ns=bytes*rates[0]*duplex+(hit?1:2)*launch;
    const double compute_ns=conversions*rates[1]+double(step.rows)*step.cols*step.depth*problem.prefix*rates[2]+
        (hit?2:3)*launch+(step.first?launch:0);
    if(overlap) {
      host=std::max(host,uploaded[slot])+bytes*host_rate+(3+(step.index>=2?2:0))*boundary;
      upload=std::max({upload,host,consumed[slot]})+upload_ns;uploaded[slot]=upload;
      compute=std::max({compute,upload,host})+compute_ns;consumed[slot]=compute;
    } else {
      host+=bytes*host_rate+upload_ns+compute_ns+(2*(hit?1:2)+1+(step.first?1:0))*boundary;
      compute=upload=host;
    }
    if(step.last) {
      const unsigned out=unsigned(tiles++&1);const double payload=double(step.rows)*step.cols*8;
      const double crt=double(step.rows)*step.cols*problem.prefix*rates[3]+2*launch;
      const double down=(payload+4)*rates[5]*duplex+2*launch;
      if(overlap) {
        if(output_bytes[out])host=std::max(host,downloaded[out])+output_bytes[out]*host_rate+boundary;
        host+=3*boundary;
        compute=std::max(compute,host)+crt;
        download=std::max(download,compute)+down;downloaded[out]=download;output_bytes[out]=payload;
      } else host+=crt+down+payload*host_rate+3*boundary;
    }
    if(!coverage.acknowledge(step.index,1,1,1,2,1))return NAN;
  }
  if(!coverage.complete())return NAN;
  if(overlap) {
    const unsigned first=output_bytes[0] && output_bytes[1] && downloaded[1]<downloaded[0]?1:0;
    for(unsigned i=0;i<2;++i)if(output_bytes[first^i])
      host=std::max(host,downloaded[first^i])+output_bytes[first^i]*host_rate+boundary;
    host=std::max({host,upload,compute,download})+3*boundary;
  }
  return std::isfinite(host)?host:NAN;
}
inline double estimate_stream_panels(const StreamPanelProblem& p,const StreamPanelResources& r,
    const rns8_stream_gemm_cost_profile& profile,bool upper) {
  if(r.status!=RNS8_SUCCESS || r.total_steps>profile.max_modeled_panels || !profile.reuse_count)return NAN;
  std::vector<uint64_t> tags(size_t(r.b_cache_entries),UINT64_MAX);
  const double cold=estimate_stream_panel_call(p,r,profile,upper,tags);
  const double warm=profile.reuse_count>1?estimate_stream_panel_call(p,r,profile,upper,tags):cold;
  // Averaging this way avoids multiplying a latency by an enormous reuse count.
  const double mean=warm+(cold-warm)/double(profile.reuse_count);
  return std::isfinite(mean) && mean>0?mean:NAN;
}
}
