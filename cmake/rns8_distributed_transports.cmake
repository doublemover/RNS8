# Optional process transports never become correctness dependencies of the
# ordinary CPU/direct-HIP library. Explicit enablement requires actual APIs.
if(RNS8_ENABLE_DISTRIBUTED_MPI)
  find_package(MPI 3.0 REQUIRED COMPONENTS CXX)
endif()
if(RNS8_ENABLE_RCCL)
  find_path(RNS8_RCCL_INCLUDE_DIR rccl/rccl.h HINTS "${RNS8_HIP_ROOT}/include" /opt/rocm/include REQUIRED)
  find_library(RNS8_RCCL_LIBRARY NAMES rccl HINTS "${RNS8_HIP_ROOT}/lib" /opt/rocm/lib REQUIRED)
  include(CheckCXXSourceCompiles)
  set(CMAKE_REQUIRED_INCLUDES "${RNS8_RCCL_INCLUDE_DIR}" ${RNS8_HIP_INCLUDE_DIRS})
  set(CMAKE_REQUIRED_LIBRARIES "${RNS8_RCCL_LIBRARY}" ${RNS8_HIP_LIBRARIES})
  set(CMAKE_REQUIRED_DEFINITIONS -D__HIP_PLATFORM_AMD__=1)
  check_cxx_source_compiles("#include <rccl/rccl.h>
    int main(){ncclConfig_t c=NCCL_CONFIG_INITIALIZER;c.blocking=0;
    ncclComm_t comm=nullptr;ncclUniqueId id{};ncclResult_t status=ncclInProgress;
    ncclCommInitRankConfig(&comm,2,id,0,&c);ncclCommGetAsyncError(comm,&status);
    ncclGroupStart();ncclSend(nullptr,0,ncclInt8,1,comm,nullptr);
    ncclRecv(nullptr,0,ncclInt8,1,comm,nullptr);ncclGroupEnd();return ncclCommAbort(comm);}"
    RNS8_RCCL_NONBLOCKING_APIS)
  unset(CMAKE_REQUIRED_INCLUDES)
  unset(CMAKE_REQUIRED_LIBRARIES)
  unset(CMAKE_REQUIRED_DEFINITIONS)
  if(NOT RNS8_RCCL_NONBLOCKING_APIS)
    message(FATAL_ERROR "Selected RCCL lacks the linked nonblocking init/progress/abort and matching P2P APIs")
  endif()
  add_library(RNS8::rccl_transport UNKNOWN IMPORTED GLOBAL)
  set_target_properties(RNS8::rccl_transport PROPERTIES IMPORTED_LOCATION "${RNS8_RCCL_LIBRARY}")
endif()
if(RNS8_ENABLE_ROCSHMEM)
  find_path(RNS8_ROCSHMEM_INCLUDE_DIR rocshmem/rocshmem.hpp HINTS "${RNS8_HIP_ROOT}/include" /opt/rocm/include REQUIRED)
  find_library(RNS8_ROCSHMEM_LIBRARY NAMES rocshmem HINTS "${RNS8_HIP_ROOT}/lib" /opt/rocm/lib REQUIRED)
  # A separate fully device-linked shared helper resolves rocSHMEM device code.
  # Merely archiving an RDC object into a host-linked library is insufficient.
  set(RNS8_ROCSHMEM_HELPER "${CMAKE_CURRENT_BINARY_DIR}/transport/librns8-rocshmem-transport.so")
  set(_process_arch_args)
  foreach(_arch IN LISTS RNS8_AMDGPU_TARGETS)
    list(APPEND _process_arch_args "--offload-arch=${_arch}")
  endforeach()
  set(_process_mpi_includes)
  foreach(_include IN LISTS MPI_CXX_INCLUDE_DIRS)
    list(APPEND _process_mpi_includes "-I${_include}")
  endforeach()
  add_custom_command(OUTPUT "${RNS8_ROCSHMEM_HELPER}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/transport"
    COMMAND "${CMAKE_COMMAND}" -E env ${RNS8_HIP_COMPILER_ENVIRONMENT}
      "${RNS8_HIP_HIPCC}" ${_process_arch_args} -std=c++17 -O2 -fgpu-rdc -fPIC -shared
      "-I${RNS8_ROCSHMEM_INCLUDE_DIR}" ${_process_mpi_includes}
      "${CMAKE_CURRENT_SOURCE_DIR}/src/backend_hip_direct/rocshmem_transport.hip"
      "${RNS8_ROCSHMEM_LIBRARY}" ${MPI_CXX_LIBRARIES}
      -o "${RNS8_ROCSHMEM_HELPER}"
    DEPENDS src/backend_hip_direct/rocshmem_transport.hip "${RNS8_ROCSHMEM_LIBRARY}"
    VERBATIM COMMENT "Device-linking real rocSHMEM put/signal/quiet transport")
  add_custom_target(rns8_rocshmem_helper DEPENDS "${RNS8_ROCSHMEM_HELPER}")
  add_library(RNS8::rocshmem_transport SHARED IMPORTED GLOBAL)
  set_target_properties(RNS8::rocshmem_transport PROPERTIES IMPORTED_LOCATION "${RNS8_ROCSHMEM_HELPER}")
  add_dependencies(RNS8::rocshmem_transport rns8_rocshmem_helper)
endif()
