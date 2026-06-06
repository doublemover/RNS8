void store_u64_limbs(std::vector<uint8_t>& dst, int64_t cell, uint64_t value) {
  for (uint32_t limb = 0; limb < 8; ++limb) {
    dst[static_cast<std::size_t>(cell * 8 + limb)] = static_cast<uint8_t>((value >> (8u * limb)) & 0xffu);
  }
}

uint64_t load_u64_limbs(const std::vector<uint8_t>& src, int64_t cell) {
  uint64_t value = 0;
  for (uint32_t limb = 0; limb < 8; ++limb) {
    value |= static_cast<uint64_t>(src[static_cast<std::size_t>(cell * 8 + limb)]) << (8u * limb);
  }
  return value;
}

void run_wrap64_resident_device_gemm(
    const std::vector<uint8_t>& a_limbs,
    const std::vector<uint8_t>& b_limbs,
    std::vector<uint8_t>& c_limbs,
    int64_t m,
    int64_t n,
    int64_t k) {
  void* device_a = nullptr;
  void* device_b = nullptr;
  void* device_c = nullptr;
  REQUIRE(rns8::detail::hip_direct_allocate(0, a_limbs.size(), &device_a) == RNS8_SUCCESS);
  REQUIRE(rns8::detail::hip_direct_allocate(0, b_limbs.size(), &device_b) == RNS8_SUCCESS);
  REQUIRE(rns8::detail::hip_direct_allocate(0, c_limbs.size(), &device_c) == RNS8_SUCCESS);
  REQUIRE(rns8::detail::hip_direct_copy_host_to_device(0, device_a, a_limbs.data(), a_limbs.size()) ==
          RNS8_SUCCESS);
  REQUIRE(rns8::detail::hip_direct_copy_host_to_device(0, device_b, b_limbs.data(), b_limbs.size()) ==
          RNS8_SUCCESS);
  REQUIRE(rns8::detail::hip_direct_copy_host_to_device(0, device_c, c_limbs.data(), c_limbs.size()) ==
          RNS8_SUCCESS);

  REQUIRE(rns8::detail::wrap64_hip_gemm_byte_limbs_device_resident(0, device_a, device_b, device_c, m, n, k) ==
          RNS8_SUCCESS);
  REQUIRE(rns8::detail::hip_direct_copy_device_to_host(0, c_limbs.data(), device_c, c_limbs.size()) ==
          RNS8_SUCCESS);

  CHECK(rns8::detail::hip_direct_free(0, device_c) == RNS8_SUCCESS);
  CHECK(rns8::detail::hip_direct_free(0, device_b) == RNS8_SUCCESS);
  CHECK(rns8::detail::hip_direct_free(0, device_a) == RNS8_SUCCESS);
}

void run_resident_ring_gemm(
    const std::vector<int8_t>& A,
    const std::vector<int8_t>& B,
    std::vector<int8_t>& C,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    uint16_t modulus,
    uint32_t modulus_index,
    uint32_t selected_prefix) {
  const std::size_t a_bytes = static_cast<std::size_t>(m) * static_cast<std::size_t>(lda) * sizeof(int8_t);
  const std::size_t b_bytes = static_cast<std::size_t>(k) * static_cast<std::size_t>(ldb) * sizeof(int8_t);
  const std::size_t c_bytes = static_cast<std::size_t>(m) * static_cast<std::size_t>(ldc) * sizeof(int8_t);
  REQUIRE(A.size() * sizeof(int8_t) >= a_bytes);
  REQUIRE(B.size() * sizeof(int8_t) >= b_bytes);
  REQUIRE(C.size() * sizeof(int8_t) >= c_bytes);

  void* device_a = nullptr;
  void* device_b = nullptr;
  void* device_c = nullptr;
  REQUIRE(rns8::detail::hip_direct_allocate(0, a_bytes, &device_a) == RNS8_SUCCESS);
  REQUIRE(rns8::detail::hip_direct_allocate(0, b_bytes, &device_b) == RNS8_SUCCESS);
  REQUIRE(rns8::detail::hip_direct_allocate(0, c_bytes, &device_c) == RNS8_SUCCESS);
  REQUIRE(rns8::detail::hip_direct_copy_host_to_device(0, device_a, A.data(), a_bytes) == RNS8_SUCCESS);
  REQUIRE(rns8::detail::hip_direct_copy_host_to_device(0, device_b, B.data(), b_bytes) == RNS8_SUCCESS);
  REQUIRE(rns8::detail::hip_direct_copy_host_to_device(0, device_c, C.data(), c_bytes) == RNS8_SUCCESS);

  REQUIRE(rns8::detail::hip_direct_ring_gemm_i8_device(
              0,
              device_a,
              device_b,
              device_c,
              m,
              n,
              k,
              lda,
              ldb,
              ldc,
              modulus,
              modulus_index,
              selected_prefix) == RNS8_SUCCESS);
  REQUIRE(rns8::detail::hip_direct_copy_device_to_host(0, C.data(), device_c, c_bytes) == RNS8_SUCCESS);

  CHECK(rns8::detail::hip_direct_free(0, device_c) == RNS8_SUCCESS);
  CHECK(rns8::detail::hip_direct_free(0, device_b) == RNS8_SUCCESS);
  CHECK(rns8::detail::hip_direct_free(0, device_a) == RNS8_SUCCESS);
}

std::vector<int8_t> exact_residues_for(boost::multiprecision::cpp_int value, uint32_t prefix) {
  std::vector<int8_t> residues(prefix);
  for (uint32_t p = 0; p < prefix; ++p) {
    residues[p] = rns8::detail::centered_residue(value, rns8::detail::kDefaultModuli[p]);
  }
  return residues;
}

void fill_exact_residue_matrix(rns8_matrix* matrix, const std::vector<boost::multiprecision::cpp_int>& values) {
  REQUIRE(matrix != nullptr);
  REQUIRE(values.size() == static_cast<std::size_t>(matrix->desc.rows * matrix->desc.cols));
  for (uint32_t p = 0; p < matrix->prefix; ++p) {
    for (int64_t row = 0; row < matrix->desc.rows; ++row) {
      for (int64_t col = 0; col < matrix->desc.cols; ++col) {
        const std::size_t value_index = static_cast<std::size_t>(row * matrix->desc.cols + col);
        const auto residues = exact_residues_for(values[value_index], matrix->prefix);
        const std::size_t residue = rns8::detail::residue_index(*matrix, p, row, col);
        matrix->residues[residue] = residues[p];
      }
    }
  }
  rns8::test::set_host_residues_current(*matrix, true);
  rns8::test::set_device_residues_current(*matrix, false);
}

void upload_exact_residues_to_hip(rns8_matrix* matrix) {
  REQUIRE(matrix != nullptr);
  REQUIRE(rns8::detail::hip_direct_copy_host_to_device(
              matrix->hip_device_id, matrix->hip_residues, matrix->residues.data(), matrix->hip_residue_bytes) ==
          RNS8_SUCCESS);
  rns8::test::set_host_residues_current(*matrix, false);
  rns8::test::set_device_residues_current(*matrix, true);
}

struct BoundedHipResidentSnapshot {
  rns8::detail::hip_direct_allocation_counters allocations{};
  void* a_residues = nullptr;
  void* b_residues = nullptr;
  void* c_residues = nullptr;
  void* a_upload = nullptr;
  void* b_upload = nullptr;
  void* c_upload = nullptr;
  void* c_export = nullptr;
  void* c_status = nullptr;
  std::size_t a_residue_bytes = 0;
  std::size_t b_residue_bytes = 0;
  std::size_t c_residue_bytes = 0;
  std::size_t a_upload_bytes = 0;
  std::size_t b_upload_bytes = 0;
  std::size_t c_upload_bytes = 0;
  std::size_t c_export_bytes = 0;
  std::size_t c_status_bytes = 0;
  uint64_t workspace_bound = 0;
  uint32_t workspace_tile_m = 0;
  uint32_t workspace_tile_n = 0;
  uint64_t workspace_schedule_tile_rows = 0;
  uint64_t workspace_schedule_tile_cols = 0;
  uint64_t workspace_schedule_tile_count = 0;
  uint32_t workspace_schedule_min_required_prefix = 0;
  uint32_t workspace_schedule_max_required_prefix = 0;
  uint32_t workspace_schedule_min_selected_prefix = 0;
  uint32_t workspace_schedule_max_selected_prefix = 0;
  uint32_t workspace_schedule_prefix_group_count = 0;
  uint32_t workspace_schedule_range_bit_length = 0;
  uint32_t workspace_schedule_adaptive_prefix_active = 0;
  uint32_t workspace_schedule_adaptive_skip_active = 0;
  uint32_t workspace_schedule_flags = 0;
  uint64_t workspace_schedule_fingerprint = 0;
};

BoundedHipResidentSnapshot capture_bounded_resident_snapshot(
    const rns8_matrix* A,
    const rns8_matrix* B,
    const rns8_matrix* C,
    const rns8_workspace* workspace) {
  REQUIRE(A != nullptr);
  REQUIRE(B != nullptr);
  REQUIRE(C != nullptr);
  REQUIRE(workspace != nullptr);
  BoundedHipResidentSnapshot snapshot{};
  snapshot.allocations = rns8::detail::hip_direct_allocation_counters_snapshot();
  snapshot.a_residues = A->hip_residues;
  snapshot.b_residues = B->hip_residues;
  snapshot.c_residues = C->hip_residues;
  snapshot.a_upload = A->hip_upload_buffer;
  snapshot.b_upload = B->hip_upload_buffer;
  snapshot.c_upload = C->hip_upload_buffer;
  snapshot.c_export = C->hip_export_buffer;
  snapshot.c_status = C->hip_status_buffer;
  snapshot.a_residue_bytes = A->hip_residue_bytes;
  snapshot.b_residue_bytes = B->hip_residue_bytes;
  snapshot.c_residue_bytes = C->hip_residue_bytes;
  snapshot.a_upload_bytes = A->hip_upload_bytes;
  snapshot.b_upload_bytes = B->hip_upload_bytes;
  snapshot.c_upload_bytes = C->hip_upload_bytes;
  snapshot.c_export_bytes = C->hip_export_bytes;
  snapshot.c_status_bytes = C->hip_status_bytes;
  snapshot.workspace_bound = workspace->bound;
  snapshot.workspace_tile_m = workspace->tile_m;
  snapshot.workspace_tile_n = workspace->tile_n;
  snapshot.workspace_schedule_tile_rows = workspace->schedule_tile_rows;
  snapshot.workspace_schedule_tile_cols = workspace->schedule_tile_cols;
  snapshot.workspace_schedule_tile_count = workspace->schedule_tile_count;
  snapshot.workspace_schedule_min_required_prefix = workspace->schedule_min_required_prefix;
  snapshot.workspace_schedule_max_required_prefix = workspace->schedule_max_required_prefix;
  snapshot.workspace_schedule_min_selected_prefix = workspace->schedule_min_selected_prefix;
  snapshot.workspace_schedule_max_selected_prefix = workspace->schedule_max_selected_prefix;
  snapshot.workspace_schedule_prefix_group_count = workspace->schedule_prefix_group_count;
  snapshot.workspace_schedule_range_bit_length = workspace->schedule_range_bit_length;
  snapshot.workspace_schedule_adaptive_prefix_active = workspace->schedule_adaptive_prefix_active;
  snapshot.workspace_schedule_adaptive_skip_active = workspace->schedule_adaptive_skip_active;
  snapshot.workspace_schedule_flags = workspace->schedule_flags;
  snapshot.workspace_schedule_fingerprint = workspace->schedule_fingerprint;
  return snapshot;
}

void check_bounded_resident_snapshot_unchanged(
    const BoundedHipResidentSnapshot& snapshot,
    const rns8_matrix* A,
    const rns8_matrix* B,
    const rns8_matrix* C,
    const rns8_workspace* workspace) {
  const auto repeated = rns8::detail::hip_direct_allocation_counters_snapshot();
  CHECK(repeated.allocate_calls == snapshot.allocations.allocate_calls);
  CHECK(repeated.free_calls == snapshot.allocations.free_calls);
  CHECK(repeated.allocated_bytes == snapshot.allocations.allocated_bytes);
  CHECK(A->hip_residues == snapshot.a_residues);
  CHECK(B->hip_residues == snapshot.b_residues);
  CHECK(C->hip_residues == snapshot.c_residues);
  CHECK(A->hip_upload_buffer == snapshot.a_upload);
  CHECK(B->hip_upload_buffer == snapshot.b_upload);
  CHECK(C->hip_upload_buffer == snapshot.c_upload);
  CHECK(C->hip_export_buffer == snapshot.c_export);
  CHECK(C->hip_status_buffer == snapshot.c_status);
  CHECK(A->hip_residue_bytes == snapshot.a_residue_bytes);
  CHECK(B->hip_residue_bytes == snapshot.b_residue_bytes);
  CHECK(C->hip_residue_bytes == snapshot.c_residue_bytes);
  CHECK(A->hip_upload_bytes == snapshot.a_upload_bytes);
  CHECK(B->hip_upload_bytes == snapshot.b_upload_bytes);
  CHECK(C->hip_upload_bytes == snapshot.c_upload_bytes);
  CHECK(C->hip_export_bytes == snapshot.c_export_bytes);
  CHECK(C->hip_status_bytes == snapshot.c_status_bytes);
  CHECK(workspace->bound == snapshot.workspace_bound);
  CHECK(workspace->tile_m == snapshot.workspace_tile_m);
  CHECK(workspace->tile_n == snapshot.workspace_tile_n);
  CHECK(workspace->schedule_tile_rows == snapshot.workspace_schedule_tile_rows);
  CHECK(workspace->schedule_tile_cols == snapshot.workspace_schedule_tile_cols);
  CHECK(workspace->schedule_tile_count == snapshot.workspace_schedule_tile_count);
  CHECK(workspace->schedule_min_required_prefix == snapshot.workspace_schedule_min_required_prefix);
  CHECK(workspace->schedule_max_required_prefix == snapshot.workspace_schedule_max_required_prefix);
  CHECK(workspace->schedule_min_selected_prefix == snapshot.workspace_schedule_min_selected_prefix);
  CHECK(workspace->schedule_max_selected_prefix == snapshot.workspace_schedule_max_selected_prefix);
  CHECK(workspace->schedule_prefix_group_count == snapshot.workspace_schedule_prefix_group_count);
  CHECK(workspace->schedule_range_bit_length == snapshot.workspace_schedule_range_bit_length);
  CHECK(workspace->schedule_adaptive_prefix_active == snapshot.workspace_schedule_adaptive_prefix_active);
  CHECK(workspace->schedule_adaptive_skip_active == snapshot.workspace_schedule_adaptive_skip_active);
  CHECK(workspace->schedule_flags == snapshot.workspace_schedule_flags);
  CHECK(workspace->schedule_fingerprint == snapshot.workspace_schedule_fingerprint);
}

struct Wrap64HipResidentSnapshot {
  rns8::detail::hip_direct_allocation_counters allocations{};
  void* a_byte_limbs = nullptr;
  void* b_byte_limbs = nullptr;
  void* c_byte_limbs = nullptr;
  void* a_upload = nullptr;
  void* b_upload = nullptr;
  void* c_upload = nullptr;
  void* c_export = nullptr;
  void* c_status = nullptr;
  std::size_t a_byte_limb_bytes = 0;
  std::size_t b_byte_limb_bytes = 0;
  std::size_t c_byte_limb_bytes = 0;
  std::size_t a_upload_bytes = 0;
  std::size_t b_upload_bytes = 0;
  std::size_t c_upload_bytes = 0;
  std::size_t c_export_bytes = 0;
  std::size_t c_status_bytes = 0;
  uint64_t workspace_bound = 0;
  uint32_t workspace_tile_m = 0;
  uint32_t workspace_tile_n = 0;
  uint64_t workspace_schedule_tile_rows = 0;
  uint64_t workspace_schedule_tile_cols = 0;
  uint64_t workspace_schedule_tile_count = 0;
  uint32_t workspace_schedule_min_required_prefix = 0;
  uint32_t workspace_schedule_max_required_prefix = 0;
  uint32_t workspace_schedule_min_selected_prefix = 0;
  uint32_t workspace_schedule_max_selected_prefix = 0;
  uint32_t workspace_schedule_prefix_group_count = 0;
  uint32_t workspace_schedule_range_bit_length = 0;
  uint32_t workspace_schedule_adaptive_prefix_active = 0;
  uint32_t workspace_schedule_adaptive_skip_active = 0;
  uint32_t workspace_schedule_flags = 0;
  uint64_t workspace_schedule_fingerprint = 0;
};

Wrap64HipResidentSnapshot capture_wrap64_resident_snapshot(
    const rns8_matrix* A,
    const rns8_matrix* B,
    const rns8_matrix* C,
    const rns8_workspace* workspace) {
  REQUIRE(A != nullptr);
  REQUIRE(B != nullptr);
  REQUIRE(C != nullptr);
  REQUIRE(workspace != nullptr);
  Wrap64HipResidentSnapshot snapshot{};
  snapshot.allocations = rns8::detail::hip_direct_allocation_counters_snapshot();
  snapshot.a_byte_limbs = A->hip_byte_limbs;
  snapshot.b_byte_limbs = B->hip_byte_limbs;
  snapshot.c_byte_limbs = C->hip_byte_limbs;
  snapshot.a_upload = A->hip_upload_buffer;
  snapshot.b_upload = B->hip_upload_buffer;
  snapshot.c_upload = C->hip_upload_buffer;
  snapshot.c_export = C->hip_export_buffer;
  snapshot.c_status = C->hip_status_buffer;
  snapshot.a_byte_limb_bytes = A->hip_byte_limb_bytes;
  snapshot.b_byte_limb_bytes = B->hip_byte_limb_bytes;
  snapshot.c_byte_limb_bytes = C->hip_byte_limb_bytes;
  snapshot.a_upload_bytes = A->hip_upload_bytes;
  snapshot.b_upload_bytes = B->hip_upload_bytes;
  snapshot.c_upload_bytes = C->hip_upload_bytes;
  snapshot.c_export_bytes = C->hip_export_bytes;
  snapshot.c_status_bytes = C->hip_status_bytes;
  snapshot.workspace_bound = workspace->bound;
  snapshot.workspace_tile_m = workspace->tile_m;
  snapshot.workspace_tile_n = workspace->tile_n;
  snapshot.workspace_schedule_tile_rows = workspace->schedule_tile_rows;
  snapshot.workspace_schedule_tile_cols = workspace->schedule_tile_cols;
  snapshot.workspace_schedule_tile_count = workspace->schedule_tile_count;
  snapshot.workspace_schedule_min_required_prefix = workspace->schedule_min_required_prefix;
  snapshot.workspace_schedule_max_required_prefix = workspace->schedule_max_required_prefix;
  snapshot.workspace_schedule_min_selected_prefix = workspace->schedule_min_selected_prefix;
  snapshot.workspace_schedule_max_selected_prefix = workspace->schedule_max_selected_prefix;
  snapshot.workspace_schedule_prefix_group_count = workspace->schedule_prefix_group_count;
  snapshot.workspace_schedule_range_bit_length = workspace->schedule_range_bit_length;
  snapshot.workspace_schedule_adaptive_prefix_active = workspace->schedule_adaptive_prefix_active;
  snapshot.workspace_schedule_adaptive_skip_active = workspace->schedule_adaptive_skip_active;
  snapshot.workspace_schedule_flags = workspace->schedule_flags;
  snapshot.workspace_schedule_fingerprint = workspace->schedule_fingerprint;
  return snapshot;
}

void check_wrap64_resident_snapshot_unchanged(
    const Wrap64HipResidentSnapshot& snapshot,
    const rns8_matrix* A,
    const rns8_matrix* B,
    const rns8_matrix* C,
    const rns8_workspace* workspace) {
  const auto repeated = rns8::detail::hip_direct_allocation_counters_snapshot();
  CHECK(repeated.allocate_calls == snapshot.allocations.allocate_calls);
  CHECK(repeated.free_calls == snapshot.allocations.free_calls);
  CHECK(repeated.allocated_bytes == snapshot.allocations.allocated_bytes);
  CHECK(A->hip_residues == nullptr);
  CHECK(B->hip_residues == nullptr);
  CHECK(C->hip_residues == nullptr);
  CHECK(A->hip_byte_limbs == snapshot.a_byte_limbs);
  CHECK(B->hip_byte_limbs == snapshot.b_byte_limbs);
  CHECK(C->hip_byte_limbs == snapshot.c_byte_limbs);
  CHECK(A->hip_upload_buffer == snapshot.a_upload);
  CHECK(B->hip_upload_buffer == snapshot.b_upload);
  CHECK(C->hip_upload_buffer == snapshot.c_upload);
  CHECK(C->hip_export_buffer == snapshot.c_export);
  CHECK(C->hip_status_buffer == snapshot.c_status);
  CHECK(A->hip_byte_limb_bytes == snapshot.a_byte_limb_bytes);
  CHECK(B->hip_byte_limb_bytes == snapshot.b_byte_limb_bytes);
  CHECK(C->hip_byte_limb_bytes == snapshot.c_byte_limb_bytes);
  CHECK(A->hip_upload_bytes == snapshot.a_upload_bytes);
  CHECK(B->hip_upload_bytes == snapshot.b_upload_bytes);
  CHECK(C->hip_upload_bytes == snapshot.c_upload_bytes);
  CHECK(C->hip_export_bytes == snapshot.c_export_bytes);
  CHECK(C->hip_status_bytes == snapshot.c_status_bytes);
  CHECK(A->device_byte_limbs_current);
  CHECK(B->device_byte_limbs_current);
  CHECK(C->device_byte_limbs_current);
  CHECK_FALSE(A->host_byte_limbs_current);
  CHECK_FALSE(B->host_byte_limbs_current);
  CHECK_FALSE(C->host_byte_limbs_current);
  CHECK(workspace->bound == snapshot.workspace_bound);
  CHECK(workspace->tile_m == snapshot.workspace_tile_m);
  CHECK(workspace->tile_n == snapshot.workspace_tile_n);
  CHECK(workspace->schedule_tile_rows == snapshot.workspace_schedule_tile_rows);
  CHECK(workspace->schedule_tile_cols == snapshot.workspace_schedule_tile_cols);
  CHECK(workspace->schedule_tile_count == snapshot.workspace_schedule_tile_count);
  CHECK(workspace->schedule_min_required_prefix == snapshot.workspace_schedule_min_required_prefix);
  CHECK(workspace->schedule_max_required_prefix == snapshot.workspace_schedule_max_required_prefix);
  CHECK(workspace->schedule_min_selected_prefix == snapshot.workspace_schedule_min_selected_prefix);
  CHECK(workspace->schedule_max_selected_prefix == snapshot.workspace_schedule_max_selected_prefix);
  CHECK(workspace->schedule_prefix_group_count == snapshot.workspace_schedule_prefix_group_count);
  CHECK(workspace->schedule_range_bit_length == snapshot.workspace_schedule_range_bit_length);
  CHECK(workspace->schedule_adaptive_prefix_active == snapshot.workspace_schedule_adaptive_prefix_active);
  CHECK(workspace->schedule_adaptive_skip_active == snapshot.workspace_schedule_adaptive_skip_active);
  CHECK(workspace->schedule_flags == snapshot.workspace_schedule_flags);
  CHECK(workspace->schedule_fingerprint == snapshot.workspace_schedule_fingerprint);
}

struct FiniteHipResidentSnapshot {
  rns8::detail::hip_direct_allocation_counters allocations{};
  void* a_residues = nullptr;
  void* b_residues = nullptr;
  void* c_residues = nullptr;
  void* a_upload = nullptr;
  void* b_upload = nullptr;
  void* c_upload = nullptr;
  void* c_export = nullptr;
  void* c_status = nullptr;
  std::size_t a_residue_bytes = 0;
  std::size_t b_residue_bytes = 0;
  std::size_t c_residue_bytes = 0;
  std::size_t a_upload_bytes = 0;
  std::size_t b_upload_bytes = 0;
  std::size_t c_upload_bytes = 0;
  std::size_t c_export_bytes = 0;
  std::size_t c_status_bytes = 0;
  uint16_t a_modulus = 0;
  uint16_t b_modulus = 0;
  uint16_t c_modulus = 0;
  uint64_t workspace_bound = 0;
  uint32_t workspace_prefix = 0;
  uint32_t workspace_schedule_min_required_prefix = 0;
  uint32_t workspace_schedule_max_required_prefix = 0;
  uint32_t workspace_schedule_min_selected_prefix = 0;
  uint32_t workspace_schedule_max_selected_prefix = 0;
  uint32_t workspace_schedule_prefix_group_count = 0;
};

FiniteHipResidentSnapshot capture_finite_resident_snapshot(
    const rns8_matrix* A,
    const rns8_matrix* B,
    const rns8_matrix* C,
    const rns8_workspace* workspace) {
  REQUIRE(A != nullptr);
  REQUIRE(B != nullptr);
  REQUIRE(C != nullptr);
  REQUIRE(workspace != nullptr);
  FiniteHipResidentSnapshot snapshot{};
  snapshot.allocations = rns8::detail::hip_direct_allocation_counters_snapshot();
  snapshot.a_residues = A->hip_residues;
  snapshot.b_residues = B->hip_residues;
  snapshot.c_residues = C->hip_residues;
  snapshot.a_upload = A->hip_upload_buffer;
  snapshot.b_upload = B->hip_upload_buffer;
  snapshot.c_upload = C->hip_upload_buffer;
  snapshot.c_export = C->hip_export_buffer;
  snapshot.c_status = C->hip_status_buffer;
  snapshot.a_residue_bytes = A->hip_residue_bytes;
  snapshot.b_residue_bytes = B->hip_residue_bytes;
  snapshot.c_residue_bytes = C->hip_residue_bytes;
  snapshot.a_upload_bytes = A->hip_upload_bytes;
  snapshot.b_upload_bytes = B->hip_upload_bytes;
  snapshot.c_upload_bytes = C->hip_upload_bytes;
  snapshot.c_export_bytes = C->hip_export_bytes;
  snapshot.c_status_bytes = C->hip_status_bytes;
  snapshot.a_modulus = A->finite_modulus;
  snapshot.b_modulus = B->finite_modulus;
  snapshot.c_modulus = C->finite_modulus;
  snapshot.workspace_bound = workspace->bound;
  snapshot.workspace_prefix = workspace->prefix;
  snapshot.workspace_schedule_min_required_prefix = workspace->schedule_min_required_prefix;
  snapshot.workspace_schedule_max_required_prefix = workspace->schedule_max_required_prefix;
  snapshot.workspace_schedule_min_selected_prefix = workspace->schedule_min_selected_prefix;
  snapshot.workspace_schedule_max_selected_prefix = workspace->schedule_max_selected_prefix;
  snapshot.workspace_schedule_prefix_group_count = workspace->schedule_prefix_group_count;
  return snapshot;
}

void check_finite_resident_snapshot_unchanged(
    const FiniteHipResidentSnapshot& snapshot,
    const rns8_matrix* A,
    const rns8_matrix* B,
    const rns8_matrix* C,
    const rns8_workspace* workspace) {
  const auto repeated = rns8::detail::hip_direct_allocation_counters_snapshot();
  CHECK(repeated.allocate_calls == snapshot.allocations.allocate_calls);
  CHECK(repeated.free_calls == snapshot.allocations.free_calls);
  CHECK(repeated.allocated_bytes == snapshot.allocations.allocated_bytes);
  CHECK(A->hip_residues == snapshot.a_residues);
  CHECK(B->hip_residues == snapshot.b_residues);
  CHECK(C->hip_residues == snapshot.c_residues);
  CHECK(A->hip_upload_buffer == snapshot.a_upload);
  CHECK(B->hip_upload_buffer == snapshot.b_upload);
  CHECK(C->hip_upload_buffer == snapshot.c_upload);
  CHECK(C->hip_export_buffer == snapshot.c_export);
  CHECK(C->hip_status_buffer == snapshot.c_status);
  CHECK(A->hip_residue_bytes == snapshot.a_residue_bytes);
  CHECK(B->hip_residue_bytes == snapshot.b_residue_bytes);
  CHECK(C->hip_residue_bytes == snapshot.c_residue_bytes);
  CHECK(A->hip_upload_bytes == snapshot.a_upload_bytes);
  CHECK(B->hip_upload_bytes == snapshot.b_upload_bytes);
  CHECK(C->hip_upload_bytes == snapshot.c_upload_bytes);
  CHECK(C->hip_export_bytes == snapshot.c_export_bytes);
  CHECK(C->hip_status_bytes == snapshot.c_status_bytes);
  CHECK(A->hip_byte_limbs == nullptr);
  CHECK(B->hip_byte_limbs == nullptr);
  CHECK(C->hip_byte_limbs == nullptr);
  CHECK(A->device_residues_current);
  CHECK(B->device_residues_current);
  CHECK(C->device_residues_current);
  CHECK_FALSE(A->host_residues_current);
  CHECK_FALSE(B->host_residues_current);
  CHECK_FALSE(C->host_residues_current);
  CHECK(A->finite_modulus == snapshot.a_modulus);
  CHECK(B->finite_modulus == snapshot.b_modulus);
  CHECK(C->finite_modulus == snapshot.c_modulus);
  CHECK(workspace->bound == snapshot.workspace_bound);
  CHECK(workspace->prefix == snapshot.workspace_prefix);
  CHECK(workspace->schedule_min_required_prefix == snapshot.workspace_schedule_min_required_prefix);
  CHECK(workspace->schedule_max_required_prefix == snapshot.workspace_schedule_max_required_prefix);
  CHECK(workspace->schedule_min_selected_prefix == snapshot.workspace_schedule_min_selected_prefix);
  CHECK(workspace->schedule_max_selected_prefix == snapshot.workspace_schedule_max_selected_prefix);
  CHECK(workspace->schedule_prefix_group_count == snapshot.workspace_schedule_prefix_group_count);
}

