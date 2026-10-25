#ifndef RNS8_CPU_AMX_HPP
#define RNS8_CPU_AMX_HPP
#include "rns8/rns8.h"
#include <memory>
#include <cstddef>
struct rns8_plan;
struct rns8_matrix;
namespace rns8::detail {
struct CpuAmxWorkspace;
bool cpu_amx_compiled() noexcept;
rns8_status cpu_amx_probe(bool request_permission, rns8_cpu_amx_info& info) noexcept;
uint32_t cpu_amx_workers() noexcept;
bool cpu_amx_workspace_bytes(int64_t m,int64_t n,int64_t k,uint32_t prefix,uint32_t workers,uint64_t& bytes) noexcept;
rns8_status cpu_amx_prepare(const rns8_plan&,std::shared_ptr<CpuAmxWorkspace>&);
rns8_status cpu_amx_gemm(const rns8_plan&,const rns8_matrix&,const rns8_matrix&,rns8_matrix&,CpuAmxWorkspace&);
rns8_status cpu_amx_workspace_info(const CpuAmxWorkspace&,rns8_cpu_amx_workspace_info&) noexcept;
// Pure admission predicate and B swizzle are also independently testable without AMX.
bool cpu_amx_admitted(const rns8_cpu_amx_info&) noexcept;
void cpu_amx_pack_b_tile(const int8_t* b,int64_t k,int64_t n,int64_t ld,int64_t kb,int64_t nb,int8_t* tile) noexcept;
}
#endif
