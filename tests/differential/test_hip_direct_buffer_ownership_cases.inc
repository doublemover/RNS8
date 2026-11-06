TEST_CASE("private HIP buffer moves and rejected replacement preserve allocation ownership") {
  using namespace rns8::detail;
  if (!hip_available())
    SKIP("no HIP device for buffer ownership");
  hip_direct_device_buffer first, second;
  REQUIRE(first.allocate(0, 128) == RNS8_SUCCESS);
  void* original = first.get();
  REQUIRE(first.allocate(0, 0) == RNS8_INVALID_ARGUMENT);
  REQUIRE(first.get() == original);
  REQUIRE(first.bytes() == 128);
  REQUIRE(first.allocate(-1, 256) == RNS8_INVALID_ARGUMENT);
  REQUIRE(first.get() == original);
  hip_direct_device_buffer moved(std::move(first));
  REQUIRE_FALSE(first);
  REQUIRE(first.bytes() == 0);
  REQUIRE(moved.get() == original);
  REQUIRE(moved.bytes() == 128);
  REQUIRE(second.allocate(0, 64) == RNS8_SUCCESS);
  const auto before = hip_direct_allocation_counters_snapshot();
  second = std::move(moved);
  REQUIRE_FALSE(moved);
  REQUIRE(moved.bytes() == 0);
  REQUIRE(second.get() == original);
  REQUIRE(second.bytes() == 128);
  const auto after = hip_direct_allocation_counters_snapshot();
  REQUIRE(after.free_calls - before.free_calls == 1);
  REQUIRE(after.allocate_calls == before.allocate_calls);
  REQUIRE(second.reset() == RNS8_SUCCESS);
  REQUIRE_FALSE(second);
  REQUIRE(second.bytes() == 0);
  REQUIRE(second.reset() == RNS8_SUCCESS);
}

#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
TEST_CASE("pinned host buffer reuse and moves preserve staging ownership") {
  using namespace rns8::detail;
  if (!hip_available())
    SKIP("no HIP device for pinned staging ownership");
  hip_pinned_host_buffer first, second;
  REQUIRE(first.ensure(0, 128) == hipSuccess);
  auto* original = first.get();
  REQUIRE(first.ensure(0, 64) == hipSuccess);
  REQUIRE(first.get() == original);
  REQUIRE(first.capacity() == 128);
  REQUIRE(first.ensure(0, 0) == hipErrorInvalidValue);
  REQUIRE(first.get() == original);
  REQUIRE(first.capacity() == 128);
  REQUIRE(first.ensure(-1, 256) == hipErrorInvalidValue);
  REQUIRE(first.get() == original);
  hip_pinned_host_buffer moved(std::move(first));
  REQUIRE(first.get() == nullptr);
  REQUIRE(first.capacity() == 0);
  REQUIRE(first.device_id() == -1);
  REQUIRE(second.ensure(0, 64) == hipSuccess);
  second = std::move(moved);
  REQUIRE(moved.get() == nullptr);
  REQUIRE(moved.capacity() == 0);
  REQUIRE(second.get() == original);
  REQUIRE(second.capacity() == 128);
  REQUIRE(second.device_id() == 0);
  REQUIRE(second.reset() == hipSuccess);
  REQUIRE(second.reset() == hipSuccess);
  REQUIRE(second.get() == nullptr);
  REQUIRE(second.capacity() == 0);
}

TEST_CASE("synchronous device copies finish before nonblocking stream consumption") {
  using namespace rns8::detail;
  if (!hip_available())
    SKIP("no HIP device for synchronous copy completion");
  constexpr std::size_t bytes = 1024 * 1024;
  hip_direct_device_buffer source, destination;
  REQUIRE(source.allocate(0, bytes) == RNS8_SUCCESS);
  REQUIRE(destination.allocate(0, bytes) == RNS8_SUCCESS);
  std::vector<uint8_t> expected(bytes);
  for (std::size_t i = 0; i < bytes; ++i)
    expected[i] = static_cast<uint8_t>((i * 17 + i / 251) % 256);
  REQUIRE(hip_direct_copy_host_to_device(0, source.get(), expected.data(), bytes) == RNS8_SUCCESS);
  const auto before = hip_direct_allocation_counters_snapshot();
  REQUIRE(hip_direct_copy_device_to_device(0, destination.get(), source.get(), bytes) ==
          RNS8_SUCCESS);
  CHECK(hipStreamQuery(nullptr) == hipSuccess);
  const auto after = hip_direct_allocation_counters_snapshot();
  CHECK(after.d2d_calls - before.d2d_calls == 1);
  CHECK(after.d2d_bytes - before.d2d_bytes == bytes);
  hip_unique_stream stream;
  REQUIRE(stream.create([](hipStream_t* out) noexcept {
    return hipStreamCreateWithFlags(out, hipStreamNonBlocking);
  }) == hipSuccess);
  hip_pinned_host_buffer readback;
  REQUIRE(readback.ensure(0, bytes) == hipSuccess);
  source.mark_pending();
  destination.mark_pending();
  readback.mark_pending();
  REQUIRE(hipMemcpyAsync(readback.get(), destination.get(), bytes, hipMemcpyDeviceToHost,
                         stream.get()) == hipSuccess);
  REQUIRE(hipStreamSynchronize(stream.get()) == hipSuccess);
  source.mark_completed();
  destination.mark_completed();
  readback.mark_completed();
  CHECK(std::equal(expected.begin(), expected.end(), static_cast<const uint8_t*>(readback.get())));
}
#endif

TEST_CASE("HIP readbacks preserve borrowed and retained destinations across staging sizes",
          "[hip][ownership][readback]") {
  using namespace rns8::detail;
  if (!hip_available())
    SKIP("no HIP device for owned readback staging");
  for (size_t bytes :
       {size_t(1), size_t(32), size_t(33), size_t(257), size_t(65536), size_t(1048577)}) {
    INFO("bytes=" << bytes);
    std::vector<uint8_t> input(bytes);
    auto output = std::make_shared<std::vector<uint8_t>>(bytes + 2, 0xa5);
    for (size_t index = 0; index < bytes; ++index)
      input[index] = uint8_t(index * 37 + bytes);
    hip_direct_device_buffer device;
    REQUIRE(device.allocate(0, bytes) == RNS8_SUCCESS);
    device.mark_pending();
    REQUIRE(hip_direct_copy_host_to_device(0, device.get(), input.data(), bytes) == RNS8_SUCCESS);
    device.mark_completed();
    const auto before = hip_direct_allocation_counters_snapshot();
    CHECK(hip_direct_copy_device_to_host(0, output->data() + 1, device.get(), bytes,
                                         static_cast<HostExportDestination>(2)) ==
          RNS8_INVALID_ARGUMENT);
    const auto after = hip_direct_allocation_counters_snapshot();
    CHECK(after.d2h_calls == before.d2h_calls);
    CHECK(*output == std::vector<uint8_t>(bytes + 2, 0xa5));
    for (auto kind : {HostExportDestination::borrowed, HostExportDestination::retained_owner}) {
      std::fill(output->begin(), output->end(), 0xa5);
      HostTransferPayload retained(hip_direct_defer_buffer_cleanup);
      if (kind == HostExportDestination::retained_owner)
        REQUIRE(retained.begin(0, std::shared_ptr<void>(output, output->data() + 1), bytes) ==
                RNS8_SUCCESS);
      device.mark_pending();
      REQUIRE(hip_direct_copy_device_to_host(0, output->data() + 1, device.get(), bytes, kind) ==
              RNS8_SUCCESS);
      device.mark_completed();
      retained.complete();
      CHECK(output->front() == 0xa5);
      CHECK(output->back() == 0xa5);
      CHECK(std::equal(input.begin(), input.end(), output->begin() + 1));
    }
  }
}
