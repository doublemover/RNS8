set(RNS8_ENABLE_HIP ON CACHE BOOL "Enable explicit Linux ROCm HIP integration")
set(RNS8_HIP_ROOT "/opt/rocm" CACHE PATH "Linux ROCm root")
set(RNS8_AMDGPU_TARGETS "gfx942;gfx950" CACHE STRING "Default Instinct ROCm validation targets")

