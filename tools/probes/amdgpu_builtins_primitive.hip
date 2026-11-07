// Compile the production builtin bodies, including sparse signatures where the
// target implements them. An empty specialization cannot certify a primitive.
#include "backend_common/matrix_target.hpp"
#if defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__ &&                                   \
    !(RNS8_BUILTIN_BODY_CDNA3 || RNS8_BUILTIN_BODY_RDNA3 || RNS8_BUILTIN_BODY_RDNA4)
#error RNS8 has no AMDGPU builtin matrix body for this target
#endif
#include "backend_amdgpu_builtins/amdgpu_builtins_kernels.hip"
