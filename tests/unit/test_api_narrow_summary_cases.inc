TEST_CASE("narrow summary queries expose completed metadata without probing a device",
          "[api][narrow][host]") {
  using namespace rns8::detail;
  // Synthetic metadata exercises the query contract, not GPU qualification.
  rns8_workspace workspace;
  rns8_narrow_summary_info info{};
  info.struct_size = sizeof(info);
  info.abi_version = RNS8_ABI_VERSION;
  const auto unchanged = info;
  CHECK(rns8_get_narrow_summary_info(nullptr, &info) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8_get_narrow_summary_info(&workspace, nullptr) == RNS8_INVALID_ARGUMENT);
  workspace.backend = RNS8_BACKEND_CPU_REFERENCE;
  CHECK(rns8_get_narrow_summary_info(&workspace, &info) == RNS8_UNSUPPORTED_BACKEND);
  CHECK(std::memcmp(&info, &unchanged, sizeof(info)) == 0);
  workspace.backend = RNS8_BACKEND_HIP_DIRECT;
  workspace.hip_device_id = 0;
  REQUIRE(rns8_get_narrow_summary_info(&workspace, &info) == RNS8_SUCCESS);
  CHECK(info.wmma_capability_observed == 0);
  CHECK(info.retained_scan_bytes == 0);
  CHECK(info.retained_product_bytes == 0);
  workspace.narrow_domains[0].identity.instance = 11;
  workspace.narrow_domains[0].identity.generation = 17;
  workspace.narrow_domains[1].identity.instance = 12;
  workspace.narrow_domains[1].identity.generation = 23;
  workspace.narrow_device_scans[0] = UINT64_MAX;
  workspace.narrow_device_scans[1] = 7;
  workspace.narrow_reuses[0] = 13;
  workspace.narrow_reuses[1] = 19;
  workspace.narrow_host_proofs[0] = 29;
  workspace.narrow_host_proofs[1] = 31;
  auto& cap = workspace.narrow_capability_observation;
  cap.provider = &hip_direct_query_native_wmma_capability;
  cap.device_id = 0;
  std::copy_n("gfx1100", 8, cap.target.begin());
  for (bool supported : {false, true}) {
    cap.supported = supported;
    REQUIRE(rns8_get_narrow_summary_info(&workspace, &info) == RNS8_SUCCESS);
    CHECK(info.wmma_capability_observed == 1);
    CHECK(info.wmma_capability_supported == uint32_t(supported));
    CHECK(info.a_instance == 11);
    CHECK(info.b_instance == 12);
    CHECK(info.a_generation == 17);
    CHECK(info.b_generation == 23);
    CHECK(info.a_device_scans == UINT64_MAX);
    CHECK(info.b_device_scans == 7);
    CHECK(info.a_reuses == 13);
    CHECK(info.b_reuses == 19);
    CHECK(info.a_host_proofs == 29);
    CHECK(info.b_host_proofs == 31);
  }
  for (unsigned invalid = 0; invalid < 3; ++invalid) {
    const auto saved = cap;
    if (invalid == 0)
      cap.provider = nullptr;
    if (invalid == 1)
      ++cap.device_id;
    if (invalid == 2)
      cap.target[0] = 0;
    REQUIRE(rns8_get_narrow_summary_info(&workspace, &info) == RNS8_SUCCESS);
    CHECK(info.wmma_capability_observed == 0);
    CHECK(info.wmma_capability_supported == 0);
    cap = saved;
  }
  for (unsigned invalid = 0; invalid < 5; ++invalid) {
    info = unchanged;
    if (invalid == 0)
      --info.struct_size;
    if (invalid == 1)
      ++info.struct_size;
    if (invalid == 2)
      ++info.abi_version;
    if (invalid == 3)
      info.reserved = 1;
    workspace.in_flight = true;
    const auto before = info;
    CHECK(rns8_get_narrow_summary_info(&workspace, &info) ==
          (invalid == 4 ? RNS8_BUSY : RNS8_INVALID_ARGUMENT));
    CHECK(std::memcmp(&info, &before, sizeof(info)) == 0);
  }
  workspace.in_flight = false;
  // Only the mandatory size prefix is readable. ASan must catch any attempted
  // version/reserved read before the short size has been rejected.
  alignas(rns8_narrow_summary_info) uint64_t short_header = sizeof(uint64_t);
  CHECK(rns8_get_narrow_summary_info(&workspace, reinterpret_cast<rns8_narrow_summary_info*>(
                                                     &short_header)) == RNS8_INVALID_ARGUMENT);
  CHECK(short_header == sizeof(uint64_t));
  alignas(rns8_narrow_summary_info) unsigned char misaligned[sizeof(info) + 1]{};
  std::memcpy(misaligned + 1, &unchanged, sizeof(info));
  CHECK(rns8_get_narrow_summary_info(&workspace, reinterpret_cast<rns8_narrow_summary_info*>(
                                                     misaligned + 1)) == RNS8_INVALID_ARGUMENT);
  CHECK(std::memcmp(misaligned + 1, &unchanged, sizeof(info)) == 0);
}
