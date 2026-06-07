from __future__ import annotations

HOST_NEUTRAL_CORE_COMMANDS = ["cmake", "ninja", "git", "python"]
WINDOWS_CORE_COMMANDS = ["vcpkg"]
CORE_COMMANDS = HOST_NEUTRAL_CORE_COMMANDS + WINDOWS_CORE_COMMANDS
WINDOWS_HIP_COMMANDS = ["hipcc", "hipInfo", "hipconfig"]
LINUX_ROCM_COMMANDS = ["hipcc", "hipconfig", "rocminfo"]
LINUX_SMI_COMMANDS = ["rocm-smi", "amd-smi"]
LINUX_TOPOLOGY_COMMANDS = ["numactl", "lstopo"]
LINUX_PROFILER_COMMANDS = ["rocprofv3", "rocprofv3-avail"]
LINUX_BANDWIDTH_COMMANDS = ["rocm-bandwidth-test"]
RCCL_TEST_COMMANDS = ["all_reduce_perf", "all_gather_perf", "broadcast_perf", "reduce_scatter_perf"]
LINUX_READINESS_COMMANDS = (
    LINUX_ROCM_COMMANDS
    + LINUX_SMI_COMMANDS
    + LINUX_TOPOLOGY_COMMANDS
    + LINUX_PROFILER_COMMANDS
    + LINUX_BANDWIDTH_COMMANDS
    + RCCL_TEST_COMMANDS
)
PYTHON_PACKAGES = ["numpy", "pandas", "matplotlib", "pytest", "scipy"]
CORE_VCPKG_PACKAGES = ["boost-multiprecision", "catch2", "nlohmann-json"]
OPTIONAL_CPP_PACKAGES = ["gmp", "flint", "ntl", "fflas-ffpack", "linbox"]
RADEON_TOOLS = [
    "rga",
    "RadeonGPUProfiler",
    "RadeonDeveloperPanel",
    "RadeonMemoryVisualizer",
    "RadeonDeveloperServiceCLI",
]
SUPPORTED_TARGETS = {
    "gfx1030": {"tier": "W2", "family": "RDNA2", "role": "functional HIP regression"},
    "gfx1100": {"tier": "W0", "family": "RDNA3", "role": "local Windows bring-up and RDNA3 optimization"},
    "gfx1200": {"tier": "W1", "family": "RDNA4", "role": "current consumer matrix-core target"},
    "gfx1201": {"tier": "W1", "family": "RDNA4", "role": "current consumer matrix-core target"},
    "gfx90a": {"tier": "I2", "family": "CDNA2", "role": "supported CDNA2 cluster target"},
    "gfx942": {"tier": "I1", "family": "CDNA3", "role": "previous-generation Instinct production"},
    "gfx950": {"tier": "I0", "family": "CDNA4", "role": "current Instinct production"},
}
LINUX_ROCM_COVERAGE_TARGETS = tuple(SUPPORTED_TARGETS)
LINUX_RDNA_TARGETS = {"gfx1030", "gfx1100", "gfx1200", "gfx1201"}
LINUX_CDNA_TARGETS = {"gfx90a", "gfx942", "gfx950"}
ACCELERATOR_NAMES = ("hipblaslt", "ck", "rocwmma", "amdgpu_builtins")
ACCELERATOR_ENABLE_FLAGS = {
    "hipblaslt": "RNS8_ENABLE_HIPBLASLT",
    "ck": "RNS8_ENABLE_CK",
    "rocwmma": "RNS8_ENABLE_ROCWMMA",
    "amdgpu_builtins": "RNS8_ENABLE_AMDGPU_BUILTINS",
}
ACCELERATOR_ENABLE_POLICY = "fail_fast_until_real_exact_correctness_backend"
CHECKER_VALIDATION_SCOPE = (
    "dependency/readiness reporting only; no build, test, smoke, schema, benchmark, "
    "or correctness validation"
)
CANDIDATE_ACCELERATOR_EVIDENCE_CLASS = "candidate_accelerator_evidence_only"
EXPECTED_ROCM_SUBMODULES = {
    "ck": {
        "path": "third_party/rocm/composable_kernel",
        "url": "https://github.com/ROCm/composable_kernel.git",
        "branch": "release/rocm-rel-7.1",
        "sha": "d9272218c4c59a58e41d3d346362cdaa707c30ce",
    },
    "rocwmma": {
        "path": "third_party/rocm/rocWMMA",
        "url": "https://github.com/ROCm/rocWMMA.git",
        "branch": "release/rocm-rel-7.1",
        "sha": "1ab208f49945c38626b79e3f0c284d65ac44a781",
    },
}

ACCELERATOR_PROBE_SOURCES = {
    "hipblaslt": """#include <hipblaslt/hipblaslt.h>

int main() {
  hipblasLtHandle_t handle = nullptr;
  hipblasStatus_t status = hipblasLtCreate(&handle);
  if (status != HIPBLAS_STATUS_SUCCESS) {
    return 2;
  }
  int version = 0;
  (void)hipblasLtGetVersion(handle, &version);
  status = hipblasLtDestroy(handle);
  return status == HIPBLAS_STATUS_SUCCESS ? 0 : 3;
}
""",
    "ck": """#include <ck/ck.hpp>
#include <ck_tile/core.hpp>

__global__ void rns8_ck_dependency_probe_kernel() {}

int main() {
  return 0;
}
""",
    "rocwmma": """#include <rocwmma/rocwmma.hpp>

__global__ void rns8_rocwmma_dependency_probe_kernel() {}

int main() {
  return 0;
}
""",
}

ACCELERATOR_PRIMITIVE_PROBE_SOURCES = {
    "ck": """#include <cstdint>
#include <ck/ck.hpp>
#include <ck/tensor_operation/gpu/device/gemm_specialization.hpp>
#include <ck/tensor_operation/gpu/device/tensor_layout.hpp>
#include <ck/tensor_operation/gpu/device/impl/device_gemm_wmma.hpp>
#include <ck/tensor_operation/gpu/element/element_wise_operation.hpp>

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using DeviceGemmInstance = ck::tensor_operation::device::DeviceGemmWmma_CShuffle<
    Row, Col, Row, int8_t, int8_t, int8_t, int32_t, int32_t, PassThrough, PassThrough, PassThrough,
    ck::tensor_operation::device::GemmSpecialization::MNKPadding,
    1, 128, 64, 128, 64, 2, 16, 16, 2, 4,
    S<4, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 2, 2, true,
    S<4, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 2, 2, true,
    1, 1, S<1, 32, 1, 4>, 8>;

extern "C" float rns8_ck_i8_wmma_primitive_probe(const int8_t* a, const int8_t* b, int8_t* c) {
  auto arg = DeviceGemmInstance::MakeArgument(
      a, b, c, 64, 128, 64, 64, 128, 128, PassThrough{}, PassThrough{}, PassThrough{});
  if (!DeviceGemmInstance::IsSupportedArgument(arg) || !DeviceGemmInstance::IsValidCompilationParameter()) {
    return -1.0f;
  }
  auto invoker = DeviceGemmInstance::MakeInvoker();
  return invoker.Run(arg);
}
""",
    "rocwmma": """#include <cstdint>
#include <rocwmma/rocwmma.hpp>

extern "C" __global__ void rns8_rocwmma_i8_mma_primitive_probe(
    const int8_t* a, const int8_t* b, int32_t* c) {
  using namespace rocwmma;
  using FragA = fragment<matrix_a, 16, 16, 32, int8_t, row_major>;
  using FragB = fragment<matrix_b, 16, 16, 32, int8_t, col_major>;
  using FragAcc = fragment<accumulator, 16, 16, 32, int32_t>;
  FragA frag_a;
  FragB frag_b;
  FragAcc acc;
  fill_fragment(acc, 0);
  load_matrix_sync(frag_a, a, 32);
  load_matrix_sync(frag_b, b, 32);
  mma_sync(acc, frag_a, frag_b, acc);
  store_matrix_sync(c, acc, 16, mem_row_major);
}
""",
}


