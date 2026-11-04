#pragma once

#include "checked_arithmetic.hpp"
#include <cstdint>

namespace rns8::detail {

struct ExportLaunch {
  int elements=0,threads=256,blocks=0,quad_blocks=0;
  uint64_t max_grid_x=0;
};

// Wide/quotient-first geometry avoids signed ceil-add overflow. Separately
// prove the largest scalar/quad inactive lane before narrowing kernel indices.
inline bool checked_export_launch(uint64_t rows,uint64_t cols,uint64_t max_grid_x,
    ExportLaunch& output) noexcept {
  uint64_t elements=0;
  if(!rows || !cols || !checked::multiply(rows,cols,elements) || elements>uint64_t(INT32_MAX))
    return false;
  const auto ceil=[](uint64_t n,uint64_t d){return n/d+uint64_t(n%d!=0);};
  const uint64_t blocks=ceil(elements,256),quads=ceil(ceil(elements,4),256);
  if(!max_grid_x || blocks>max_grid_x || quads>max_grid_x ||
      blocks*256-1>uint64_t(INT32_MAX) || quads*1024-1>uint64_t(INT32_MAX))return false;
  output={int(elements),256,int(blocks),int(quads),max_grid_x};
  return true;
}

inline bool checked_export_schedule(uint64_t entries,const ExportLaunch& launch,
    int& total_blocks) noexcept {
  uint64_t blocks=0;
  if(!entries || launch.blocks<=0 || !checked::multiply(entries,uint64_t(launch.blocks),blocks) ||
      blocks>uint64_t(INT32_MAX) || blocks>launch.max_grid_x)return false;
  total_blocks=int(blocks);
  return true;
}

}  // namespace rns8::detail
