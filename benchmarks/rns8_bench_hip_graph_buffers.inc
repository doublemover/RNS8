struct DeviceBuffer {
  int device_id = -1;
  void* ptr = nullptr;
  std::size_t bytes = 0;

  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  ~DeviceBuffer() {
    reset();
  }

  void allocate(int device, std::size_t requested_bytes, const char* label) {
    reset();
    device_id = device;
    bytes = requested_bytes;
    const rns8_status status = rns8::detail::hip_direct_allocate(device_id, requested_bytes, &ptr);
    if (status != RNS8_SUCCESS) {
      fail_status(label, status);
    }
  }

  void reset() {
    if (ptr) {
      const rns8_status status = rns8::detail::hip_direct_free(device_id, ptr);
      ptr = nullptr;
      bytes = 0;
      if (status != RNS8_SUCCESS) {
        std::cerr << "hip_direct_free: " << rns8_status_string(status) << "\n";
      }
    }
  }
};

std::string hip_graph_error_text(const char* operation, int code) {
#if RNS8_CONFIGURED_HIP_ENABLED
  const auto error = static_cast<hipError_t>(code);
  const char* name = hipGetErrorName(error);
  const char* text = hipGetErrorString(error);
  std::ostringstream out;
  out << operation << " failed";
  if (name && name[0] != '\0') {
    out << ": " << name;
  }
  if (text && text[0] != '\0') {
    out << " (" << text << ")";
  }
  return out.str();
#else
  (void)operation;
  (void)code;
  return "HIP graph replay requires a HIP-enabled benchmark build";
#endif
}

void destroy_hip_graph_replay_state(HipGraphReplayState& state) {
#if RNS8_CONFIGURED_HIP_ENABLED
  if (state.executable) {
    const hipError_t status = hipGraphExecDestroy(state.executable);
    if (status != hipSuccess) {
      std::cerr << hip_graph_error_text("hipGraphExecDestroy", static_cast<int>(status)) << "\n";
    }
    state.executable = nullptr;
  }
  if (state.graph) {
    const hipError_t status = hipGraphDestroy(state.graph);
    if (status != hipSuccess) {
      std::cerr << hip_graph_error_text("hipGraphDestroy", static_cast<int>(status)) << "\n";
    }
    state.graph = nullptr;
  }
  state.stream.reset();
#else
  (void)state;
#endif
}

rns8_status capture_hip_graph_replay(
    int device_id,
    const std::function<rns8_status(void*)>& capture_body,
    HipGraphReplayState& state,
    uint64_t& capture_us,
    uint64_t& instantiate_us,
    std::string& error_text) {
#if RNS8_CONFIGURED_HIP_ENABLED
  destroy_hip_graph_replay_state(state);
  const hipError_t device_status = hipSetDevice(device_id);
  if (device_status != hipSuccess) {
    error_text = hip_graph_error_text("hipSetDevice", static_cast<int>(device_status));
    return RNS8_BACKEND_FAILURE;
  }
  hipError_t status = state.stream.create_non_blocking();
  if (status != hipSuccess) {
    error_text = hip_graph_error_text("hipStreamCreateWithFlags", static_cast<int>(status));
    return RNS8_BACKEND_FAILURE;
  }

  const auto capture_start = std::chrono::steady_clock::now();
  status = hipStreamBeginCapture(state.stream.get(), hipStreamCaptureModeGlobal);
  if (status != hipSuccess) {
    error_text = hip_graph_error_text("hipStreamBeginCapture", static_cast<int>(status));
    destroy_hip_graph_replay_state(state);
    return RNS8_BACKEND_FAILURE;
  }

  const rns8_status body_status = capture_body(static_cast<void*>(state.stream.get()));
  hipGraph_t captured_graph = nullptr;
  status = hipStreamEndCapture(state.stream.get(), &captured_graph);
  const auto capture_end = std::chrono::steady_clock::now();
  capture_us = elapsed_us(capture_start, capture_end);
  if (body_status != RNS8_SUCCESS) {
    if (captured_graph) {
      (void)hipGraphDestroy(captured_graph);
    }
    error_text = std::string("captured Direct-HIP graph body returned ") + rns8_status_string(body_status);
    return body_status;
  }
  if (status != hipSuccess) {
    error_text = hip_graph_error_text("hipStreamEndCapture", static_cast<int>(status));
    if (captured_graph) {
      (void)hipGraphDestroy(captured_graph);
    }
    return RNS8_BACKEND_FAILURE;
  }
  state.graph = captured_graph;

  const auto instantiate_start = std::chrono::steady_clock::now();
  status = hipGraphInstantiate(&state.executable, state.graph, nullptr, nullptr, 0);
  const auto instantiate_end = std::chrono::steady_clock::now();
  instantiate_us = elapsed_us(instantiate_start, instantiate_end);
  if (status != hipSuccess) {
    error_text = hip_graph_error_text("hipGraphInstantiate", static_cast<int>(status));
    destroy_hip_graph_replay_state(state);
    return RNS8_BACKEND_FAILURE;
  }
  return RNS8_SUCCESS;
#else
  (void)device_id;
  (void)capture_body;
  (void)state;
  capture_us = 0;
  instantiate_us = 0;
  error_text = "HIP graph replay requires a HIP-enabled benchmark build";
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status launch_hip_graph_replay(int device_id, HipGraphReplayState& state, std::string& error_text) {
#if RNS8_CONFIGURED_HIP_ENABLED
  if (!state.executable) {
    error_text = "hip graph executable is not instantiated";
    return RNS8_INVALID_ARGUMENT;
  }
  hipError_t status = hipSetDevice(device_id);
  if (status != hipSuccess) {
    error_text = hip_graph_error_text("hipSetDevice", static_cast<int>(status));
    return RNS8_BACKEND_FAILURE;
  }
  status = hipGraphLaunch(state.executable, state.stream.get());
  if (status != hipSuccess) {
    error_text = hip_graph_error_text("hipGraphLaunch", static_cast<int>(status));
    return RNS8_BACKEND_FAILURE;
  }
  status = hipStreamSynchronize(state.stream.get());
  if (status != hipSuccess) {
    error_text = hip_graph_error_text("hipStreamSynchronize", static_cast<int>(status));
    return RNS8_BACKEND_FAILURE;
  }
  ++state.launch_count;
  return RNS8_SUCCESS;
#else
  (void)device_id;
  (void)state;
  error_text = "HIP graph replay requires a HIP-enabled benchmark build";
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

template <typename T>
uint64_t vector_alu_workspace_bytes(const Args& args) {
  const std::size_t a_bytes = checked_bytes(checked_elements(args.m, args.k, "A"), sizeof(T), "A");
  const std::size_t b_bytes = checked_bytes(checked_elements(args.k, args.n, "B"), sizeof(T), "B");
  const std::size_t c_bytes = checked_bytes(checked_elements(args.m, args.n, "C"), sizeof(T), "C");
  const std::size_t status_bytes = sizeof(uint32_t);
  const std::size_t ab_bytes = checked_add_bytes(a_bytes, b_bytes, "vector workspace A+B");
  const std::size_t abc_bytes = checked_add_bytes(ab_bytes, c_bytes, "vector workspace A+B+C");
  const std::size_t total = checked_add_bytes(abc_bytes, status_bytes, "vector workspace A+B+C+status");
  return static_cast<uint64_t>(total);
}

double average(const std::vector<uint64_t>& values) {
  if (values.empty()) {
    return 0.0;
  }
  uint64_t sum = 0;
  for (const uint64_t value : values) {
    sum += value;
  }
  return static_cast<double>(sum) / static_cast<double>(values.size());
}

double median(std::vector<uint64_t> values) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const std::size_t middle = values.size() / 2;
  if ((values.size() & 1u) != 0u) {
    return static_cast<double>(values[middle]);
  }
  return (static_cast<double>(values[middle - 1]) + static_cast<double>(values[middle])) / 2.0;
}

double percentile(std::vector<uint64_t> values, double p) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const double scaled = p * static_cast<double>(values.size() - 1);
  const auto index = static_cast<std::size_t>(scaled + 0.999999);
  return static_cast<double>(values[index]);
}

double average_double(const std::vector<double>& values) {
  if (values.empty()) {
    return 0.0;
  }
  double sum = 0.0;
  for (const double value : values) {
    sum += value;
  }
  return sum / static_cast<double>(values.size());
}

double percentile_double(std::vector<double> values, double p) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const double scaled = p * static_cast<double>(values.size() - 1);
  const auto index = static_cast<std::size_t>(scaled + 0.999999);
  return values[index];
}

