
rns8_context* create_cpu_context() {
  rns8_context_options options{};
  options.struct_size = sizeof(options);
  options.abi_version = RNS8_ABI_VERSION;
  options.requested_backend = RNS8_BACKEND_CPU_REFERENCE;
  rns8_context* ctx = nullptr;
  return rns8_create_context(-1, &options, &ctx) == RNS8_SUCCESS ? ctx : nullptr;
}

rns8_context* create_hip_context() {
  rns8_context_options options{};
  options.struct_size = sizeof(options);
  options.abi_version = RNS8_ABI_VERSION;
  options.requested_backend = RNS8_BACKEND_HIP_DIRECT;
  rns8_context* ctx = nullptr;
  return rns8_create_context(0, &options, &ctx) == RNS8_SUCCESS ? ctx : nullptr;
}

rns8_context* create_wrap64_context() {
  rns8_context_options options{};
  options.struct_size = sizeof(options);
  options.abi_version = RNS8_ABI_VERSION;
  options.requested_backend = RNS8_BACKEND_WRAP64_BYTE_LIMB;
  rns8_context* ctx = nullptr;
  return rns8_create_context(-1, &options, &ctx) == RNS8_SUCCESS ? ctx : nullptr;
}

