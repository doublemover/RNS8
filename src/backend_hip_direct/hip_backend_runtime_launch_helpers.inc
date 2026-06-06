namespace {

#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
template <typename Fn>
hipError_t timed_hip_operation(const char* label, Fn&& fn) {
  if (!hip_direct_timing_enabled()) {
    return fn();
  }

  hip_unique_event_pair events;
  hipError_t event_status = events.create_and_record_start();
  if (event_status != hipSuccess) {
    return fn();
  }

  const hipError_t op_status = fn();
  if (op_status == hipSuccess) {
    event_status = events.record_stop();
    if (event_status == hipSuccess) {
      hip_direct_timing_record_pending_event(label, events.release_start(), events.release_stop());
      return op_status;
    }
  }
  return op_status;
}

std::string env_value(const char* name) {
#if defined(_MSC_VER)
  char* buffer = nullptr;
  std::size_t length = 0;
  if (_dupenv_s(&buffer, &length, name) != 0 || !buffer) {
    return {};
  }
  std::string value(buffer);
  std::free(buffer);
  return value;
#else
  const char* value = std::getenv(name);
  return value ? std::string(value) : std::string{};
#endif
}

bool env_flag_disabled(const char* name) {
  const std::string value = env_value(name);
  return value == "0" || value == "false" || value == "FALSE" || value == "off" ||
         value == "OFF" || value == "no" || value == "NO";
}

bool env_flag_enabled(const char* name) {
  const std::string value = env_value(name);
  return value == "1" || value == "true" || value == "TRUE" || value == "on" ||
         value == "ON" || value == "yes" || value == "YES";
}

bool pinned_export_staging_enabled(std::size_t bytes, bool padded_destination, bool default_padded_staging) {
  if (bytes < kPinnedExportStagingMinBytes || env_flag_disabled("RNS8_HIP_PINNED_EXPORT_STAGING")) {
    return false;
  }
  return (default_padded_staging && padded_destination) || env_flag_enabled("RNS8_HIP_PINNED_EXPORT_STAGING");
}

bool checked_i32_shape(int64_t rows, int64_t cols, int64_t ld, uint32_t prefix) {
  if (rows <= 0 || cols <= 0 || ld < cols || prefix == 0 || prefix > RNS8_DEFAULT_MODULUS_COUNT) {
    return false;
  }
  return rows <= std::numeric_limits<int>::max() && cols <= std::numeric_limits<int>::max() &&
         ld <= std::numeric_limits<int>::max() && prefix <= static_cast<uint32_t>(std::numeric_limits<int>::max());
}

rns8_status set_hip_device(int device_id) {
  if (device_id < 0) {
    return RNS8_INVALID_ARGUMENT;
  }
  const hipError_t err = hipSetDevice(device_id);
  return err == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
}

bool checked_pack_elements(int64_t rows, int64_t cols, uint32_t prefix) {
  const uint64_t max_pack_elements = static_cast<uint64_t>(std::numeric_limits<int>::max()) * 256u;
  return static_cast<uint64_t>(rows) <= max_pack_elements / static_cast<uint64_t>(cols) / prefix;
}

bool checked_grouped_source_bytes(
    uint32_t task_count,
    int64_t rows,
    int64_t ld,
    std::size_t element_size,
    std::size_t* out_bytes,
    int64_t* out_task_stride) {
  if (task_count == 0 || rows <= 0 || ld <= 0 || element_size == 0 || !out_bytes || !out_task_stride) {
    return false;
  }
  const auto max_size = std::numeric_limits<std::size_t>::max();
  const uint64_t row_count = static_cast<uint64_t>(rows);
  const uint64_t row_elements = static_cast<uint64_t>(ld);
  if (row_count > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / row_elements) {
    return false;
  }
  const uint64_t task_stride = row_count * row_elements;
  if (task_stride > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    return false;
  }
  if (static_cast<uint64_t>(task_count) > static_cast<uint64_t>(max_size / element_size) / task_stride) {
    return false;
  }
  *out_task_stride = static_cast<int64_t>(task_stride);
  *out_bytes = static_cast<std::size_t>(task_stride) * static_cast<std::size_t>(task_count) * element_size;
  return true;
}

bool checked_grouped_source_versions(uint64_t first_source_version, uint32_t task_count) {
  return task_count != 0 &&
         first_source_version <= std::numeric_limits<uint64_t>::max() - static_cast<uint64_t>(task_count - 1);
}

bool checked_matrix_elements_i32(int64_t rows, int64_t cols) {
  if (rows <= 0 || cols <= 0 || rows > std::numeric_limits<int>::max() || cols > std::numeric_limits<int>::max()) {
    return false;
  }
  return static_cast<uint64_t>(rows) <=
         static_cast<uint64_t>(std::numeric_limits<int>::max()) / static_cast<uint64_t>(cols);
}

void release_pinned_export_staging() {
  g_pinned_export_staging.reset();
}

void* ensure_pinned_export_staging(int device_id, std::size_t bytes) {
  if (bytes == 0) {
    return nullptr;
  }
  if (g_pinned_export_staging.ensure(device_id, bytes) != hipSuccess) {
    return nullptr;
  }
  return g_pinned_export_staging.get();
}

bool checked_output_bytes(int64_t rows, int64_t cols, std::size_t element_size) {
  if (rows <= 0 || cols <= 0 || element_size == 0) {
    return false;
  }
  const auto max_size = std::numeric_limits<std::size_t>::max();
  return static_cast<uint64_t>(rows) <=
         static_cast<uint64_t>(max_size / element_size / static_cast<std::size_t>(cols));
}

void scatter_compact_host_matrix(
    void* dst,
    int64_t dst_ld,
    const void* src,
    int64_t rows,
    int64_t cols,
    std::size_t cell_bytes) {
  const std::size_t compact_row_bytes = static_cast<std::size_t>(cols) * cell_bytes;
  if (dst_ld == cols) {
    std::memcpy(dst, src, static_cast<std::size_t>(rows) * compact_row_bytes);
    return;
  }
  auto* dst_bytes = static_cast<std::uint8_t*>(dst);
  const auto* src_bytes = static_cast<const std::uint8_t*>(src);
  const std::size_t dst_row_bytes = static_cast<std::size_t>(dst_ld) * cell_bytes;
  for (int64_t row = 0; row < rows; ++row) {
    std::memcpy(
        dst_bytes + static_cast<std::size_t>(row) * dst_row_bytes,
        src_bytes + static_cast<std::size_t>(row) * compact_row_bytes,
        compact_row_bytes);
  }
}

void record_export_host_staging_copy_sample(
    const std::chrono::steady_clock::time_point& start,
    const std::chrono::steady_clock::time_point& stop) {
  if (!hip_direct_timing_enabled()) {
    return;
  }
  const auto nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
  if (nanos >= 0) {
    hip_direct_timing_record_sample("export_host_staging_copy", static_cast<double>(nanos) / 1000.0);
  }
}

hipError_t copy_compact_matrix_device_to_host_direct(
    void* dst,
    int64_t dst_ld,
    const void* src,
    int64_t rows,
    int64_t cols,
    std::size_t cell_bytes) {
  const std::size_t compact_row_bytes = static_cast<std::size_t>(cols) * cell_bytes;
  if (dst_ld == cols) {
    return hipMemcpy(
        dst,
        src,
        static_cast<std::size_t>(rows) * compact_row_bytes,
        hipMemcpyDeviceToHost);
  }
  return hipMemcpy2D(
      dst,
      static_cast<std::size_t>(dst_ld) * cell_bytes,
      src,
      compact_row_bytes,
      compact_row_bytes,
      static_cast<std::size_t>(rows),
      hipMemcpyDeviceToHost);
}

hipError_t copy_compact_matrix_device_to_host(
    int device_id,
    const char* label,
    void* dst,
    int64_t dst_ld,
    const void* src,
    int64_t rows,
    int64_t cols,
    std::size_t cell_bytes,
    bool default_padded_staging = true) {
  if (!dst || !src || dst_ld < cols || !checked_output_bytes(rows, cols, cell_bytes) ||
      !checked_output_bytes(rows, dst_ld, cell_bytes)) {
    return hipErrorInvalidValue;
  }
  const std::size_t compact_row_bytes = static_cast<std::size_t>(cols) * cell_bytes;
  const std::size_t compact_bytes = static_cast<std::size_t>(rows) * compact_row_bytes;
  if (pinned_export_staging_enabled(compact_bytes, dst_ld != cols, default_padded_staging)) {
    void* pinned = ensure_pinned_export_staging(device_id, compact_bytes);
    if (pinned) {
      hipError_t err = timed_hip_operation(label, [&]() {
        return hipMemcpy(pinned, src, compact_bytes, hipMemcpyDeviceToHost);
      });
      if (err != hipSuccess) {
        return err;
      }
      const auto start = std::chrono::steady_clock::now();
      scatter_compact_host_matrix(dst, dst_ld, pinned, rows, cols, cell_bytes);
      const auto stop = std::chrono::steady_clock::now();
      record_export_host_staging_copy_sample(start, stop);
      return hipSuccess;
    }
  }
  return timed_hip_operation(label, [&]() {
    return copy_compact_matrix_device_to_host_direct(dst, dst_ld, src, rows, cols, cell_bytes);
  });
}

bool checked_limb_export_pitch(int64_t ld, uint32_t limb_count) {
  if (ld <= 0 || limb_count == 0) {
    return false;
  }
  const auto max_size = std::numeric_limits<std::size_t>::max();
  return static_cast<uint64_t>(ld) <=
         static_cast<uint64_t>(max_size / sizeof(uint64_t) / static_cast<std::size_t>(limb_count));
}

bool exact_wide_signed_export_requires_status(uint32_t limb_count) {
  return limb_count < 3;
}

bool exact_wide_unsigned_export_requires_status(uint32_t limb_count) {
  return limb_count < 3;
}

rns8_status validate_grouped_rns_matrices(
    rns8_matrix* const* matrices,
    uint32_t task_count,
    rns8_semantics expected_semantics,
    int64_t rows,
    int64_t cols,
    bool require_current,
    int* out_device_id,
    uint32_t* out_prefix,
    std::vector<const int8_t*>* out_residue_ptrs) {
  if (!matrices || task_count == 0 || !out_device_id || !out_prefix) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (rows != 0 || cols != 0) {
    if (!checked_matrix_elements_i32(rows, cols)) {
      return RNS8_INVALID_ARGUMENT;
    }
  }
  int device_id = -1;
  uint32_t prefix = 0;
  if (out_residue_ptrs) {
    out_residue_ptrs->clear();
    out_residue_ptrs->reserve(task_count);
  }
  for (uint32_t index = 0; index < task_count; ++index) {
    const rns8_matrix* matrix = matrices[index];
    if (!matrix || matrix->backend != RNS8_BACKEND_HIP_DIRECT || matrix->desc.semantics != expected_semantics ||
        !matrix->hip_residues || matrix->hip_device_id < 0) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (rows != 0 && (matrix->desc.rows != rows || matrix->desc.cols != cols)) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (require_current && !matrix->device_residues_current) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (index == 0) {
      device_id = matrix->hip_device_id;
      prefix = matrix->prefix;
    } else if (matrix->hip_device_id != device_id || matrix->prefix != prefix) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (out_residue_ptrs) {
      out_residue_ptrs->push_back(static_cast<const int8_t*>(matrix->hip_residues));
    }
  }
  *out_device_id = device_id;
  *out_prefix = prefix;
  return RNS8_SUCCESS;
}

bool checked_tile_entry(const rns8_plan_tile_schedule_entry& entry, int64_t rows, int64_t cols) {
  if (entry.struct_size != sizeof(rns8_plan_tile_schedule_entry) || entry.abi_version != RNS8_ABI_VERSION ||
      (entry.flags & ~kKnownTileScheduleFlags) != 0 || entry.row_offset < 0 || entry.col_offset < 0 ||
      entry.row_extent <= 0 || entry.col_extent <= 0 || entry.required_prefix == 0 || entry.selected_prefix == 0 ||
      entry.required_prefix > entry.selected_prefix || entry.selected_prefix > RNS8_MAX_SUPPORTED_PREFIX) {
    return false;
  }
  if (entry.row_offset > rows || entry.col_offset > cols) {
    return false;
  }
  return entry.row_extent <= rows - entry.row_offset && entry.col_extent <= cols - entry.col_offset &&
         entry.row_offset <= std::numeric_limits<int>::max() && entry.col_offset <= std::numeric_limits<int>::max() &&
         entry.row_extent <= std::numeric_limits<int>::max() && entry.col_extent <= std::numeric_limits<int>::max();
}

struct checked_tile_schedule {
  std::vector<uint64_t> row_offsets;
  std::vector<uint64_t> row_extents;
  std::vector<uint64_t> col_offsets;
  std::vector<uint64_t> col_extents;
  std::vector<uint32_t> selected_prefix_groups;
};

bool collect_unique_sorted_u64(std::vector<uint64_t>& values, uint64_t value) {
  if (std::find(values.begin(), values.end(), value) == values.end()) {
    values.push_back(value);
  }
  return values.size() <= static_cast<std::size_t>(std::numeric_limits<int>::max());
}

bool collect_unique_sorted_u32(std::vector<uint32_t>& values, uint32_t value) {
  if (std::find(values.begin(), values.end(), value) == values.end()) {
    values.push_back(value);
  }
  return values.size() <= static_cast<std::size_t>(std::numeric_limits<uint32_t>::max());
}

bool checked_tile_axis_coverage(const std::vector<uint64_t>& offsets, const std::vector<uint64_t>& extents, uint64_t total) {
  uint64_t expected_offset = 0;
  for (std::size_t index = 0; index < offsets.size(); ++index) {
    if (expected_offset > total || offsets[index] != expected_offset || extents[index] == 0 ||
        extents[index] > total - expected_offset) {
      return false;
    }
    expected_offset += extents[index];
  }
  return expected_offset == total;
}

bool checked_tile_schedule_contract(
    const rns8_plan_tile_schedule_entry* entries,
    uint64_t entry_count,
    int64_t rows,
    int64_t cols,
    checked_tile_schedule* out) {
  if (!entries || entry_count == 0 || entry_count > static_cast<uint64_t>(std::numeric_limits<std::size_t>::max()) ||
      rows <= 0 || cols <= 0) {
    return false;
  }
  std::vector<uint64_t> row_ids;
  std::vector<uint64_t> col_ids;
  std::vector<uint32_t> selected_prefixes;
  for (uint64_t index = 0; index < entry_count; ++index) {
    const auto& entry = entries[static_cast<std::size_t>(index)];
    if (!checked_tile_entry(entry, rows, cols) || !collect_unique_sorted_u64(row_ids, entry.tile_row) ||
        !collect_unique_sorted_u64(col_ids, entry.tile_col) ||
        !collect_unique_sorted_u32(selected_prefixes, entry.selected_prefix)) {
      return false;
    }
  }
  std::sort(row_ids.begin(), row_ids.end());
  std::sort(col_ids.begin(), col_ids.end());
  std::sort(selected_prefixes.begin(), selected_prefixes.end());
  if (row_ids.empty() || col_ids.empty() ||
      row_ids.size() > static_cast<std::size_t>(std::numeric_limits<uint64_t>::max() / col_ids.size()) ||
      static_cast<uint64_t>(row_ids.size() * col_ids.size()) != entry_count) {
    return false;
  }
  for (std::size_t row = 0; row < row_ids.size(); ++row) {
    if (row_ids[row] != row) {
      return false;
    }
  }
  for (std::size_t col = 0; col < col_ids.size(); ++col) {
    if (col_ids[col] != col) {
      return false;
    }
  }
  checked_tile_schedule schedule{};
  schedule.row_offsets.assign(row_ids.size(), 0);
  schedule.row_extents.assign(row_ids.size(), 0);
  schedule.col_offsets.assign(col_ids.size(), 0);
  schedule.col_extents.assign(col_ids.size(), 0);
  schedule.selected_prefix_groups = selected_prefixes;
  std::vector<uint8_t> seen(static_cast<std::size_t>(entry_count), 0);
  for (uint64_t index = 0; index < entry_count; ++index) {
    const auto& entry = entries[static_cast<std::size_t>(index)];
    const std::size_t tile_row = static_cast<std::size_t>(entry.tile_row);
    const std::size_t tile_col = static_cast<std::size_t>(entry.tile_col);
    if (tile_row != static_cast<std::size_t>(index / static_cast<uint64_t>(col_ids.size())) ||
        tile_col != static_cast<std::size_t>(index % static_cast<uint64_t>(col_ids.size()))) {
      return false;
    }
    const std::size_t linear = tile_row * col_ids.size() + tile_col;
    if (linear >= seen.size() || seen[linear] != 0) {
      return false;
    }
    seen[linear] = 1;
    const uint64_t row_offset = static_cast<uint64_t>(entry.row_offset);
    const uint64_t col_offset = static_cast<uint64_t>(entry.col_offset);
    const uint64_t row_extent = static_cast<uint64_t>(entry.row_extent);
    const uint64_t col_extent = static_cast<uint64_t>(entry.col_extent);
    if (schedule.row_extents[tile_row] == 0) {
      schedule.row_offsets[tile_row] = row_offset;
      schedule.row_extents[tile_row] = row_extent;
    } else if (schedule.row_offsets[tile_row] != row_offset || schedule.row_extents[tile_row] != row_extent) {
      return false;
    }
    if (schedule.col_extents[tile_col] == 0) {
      schedule.col_offsets[tile_col] = col_offset;
      schedule.col_extents[tile_col] = col_extent;
    } else if (schedule.col_offsets[tile_col] != col_offset || schedule.col_extents[tile_col] != col_extent) {
      return false;
    }
    const auto group = std::lower_bound(selected_prefixes.begin(), selected_prefixes.end(), entry.selected_prefix);
    if (group == selected_prefixes.end() || *group != entry.selected_prefix ||
        entry.group_index != static_cast<uint32_t>(group - selected_prefixes.begin())) {
      return false;
    }
  }
  if (std::find(seen.begin(), seen.end(), uint8_t{0}) != seen.end() ||
      !checked_tile_axis_coverage(schedule.row_offsets, schedule.row_extents, static_cast<uint64_t>(rows)) ||
      !checked_tile_axis_coverage(schedule.col_offsets, schedule.col_extents, static_cast<uint64_t>(cols))) {
    return false;
  }
  if (out) {
    *out = std::move(schedule);
  }
  return true;
}

bool scheduled_tile_block_shape(
    const rns8_plan_tile_schedule_entry* entries,
    uint64_t entry_count,
    int* max_row_blocks,
    int* max_col_blocks) {
  if (!entries || entry_count == 0 || !max_row_blocks || !max_col_blocks) {
    return false;
  }
  uint64_t row_blocks = 0;
  uint64_t col_blocks = 0;
  for (uint64_t index = 0; index < entry_count; ++index) {
    const auto& entry = entries[static_cast<std::size_t>(index)];
    const uint64_t entry_row_blocks = (static_cast<uint64_t>(entry.row_extent) + 15u) / 16u;
    const uint64_t entry_col_blocks = (static_cast<uint64_t>(entry.col_extent) + 15u) / 16u;
    row_blocks = std::max(row_blocks, entry_row_blocks);
    col_blocks = std::max(col_blocks, entry_col_blocks);
  }
  if (row_blocks == 0 || col_blocks == 0 || row_blocks > static_cast<uint64_t>(std::numeric_limits<int>::max()) ||
      col_blocks > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
    return false;
  }
  *max_row_blocks = static_cast<int>(row_blocks);
  *max_col_blocks = static_cast<int>(col_blocks);
  return true;
}

bool scheduled_tile_max_elements(
    const rns8_plan_tile_schedule_entry* entries,
    uint64_t entry_count,
    int* max_elements) {
  if (!entries || entry_count == 0 || !max_elements) {
    return false;
  }
  uint64_t elements = 0;
  for (uint64_t index = 0; index < entry_count; ++index) {
    const auto& entry = entries[static_cast<std::size_t>(index)];
    const uint64_t row_extent = static_cast<uint64_t>(entry.row_extent);
    const uint64_t col_extent = static_cast<uint64_t>(entry.col_extent);
    if (col_extent != 0 && row_extent > static_cast<uint64_t>(std::numeric_limits<int>::max()) / col_extent) {
      return false;
    }
    elements = std::max(elements, row_extent * col_extent);
  }
  if (elements == 0 || elements > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
    return false;
  }
  *max_elements = static_cast<int>(elements);
  return true;
}

bool schedule_has_zero_output_tiles(const rns8_plan_tile_schedule_entry* entries, uint64_t entry_count) {
  if (!entries) {
    return false;
  }
  for (uint64_t index = 0; index < entry_count; ++index) {
    if ((entries[static_cast<std::size_t>(index)].flags & RNS8_TILE_SCHEDULE_ZERO_OUTPUT) != 0) {
      return true;
    }
  }
  return false;
}

bool schedule_has_zero_row_col_products(const rns8_plan_tile_schedule_entry* entries, uint64_t entry_count) {
  if (!entries) {
    return false;
  }
  for (uint64_t index = 0; index < entry_count; ++index) {
    if ((entries[static_cast<std::size_t>(index)].flags & RNS8_TILE_SCHEDULE_ZERO_ROW_COL_PRODUCT) != 0) {
      return true;
    }
  }
  return false;
}

bool schedule_all_zero_output_tiles_uniform_prefix(
    const rns8_plan_tile_schedule_entry* entries,
    uint64_t entry_count,
    uint32_t& selected_prefix) {
  if (!entries || entry_count == 0) {
    return false;
  }
  selected_prefix = 0;
  for (uint64_t index = 0; index < entry_count; ++index) {
    const auto& entry = entries[static_cast<std::size_t>(index)];
    if ((entry.flags & RNS8_TILE_SCHEDULE_ZERO_OUTPUT) == 0) {
      return false;
    }
    if (selected_prefix == 0) {
      selected_prefix = entry.selected_prefix;
    } else if (selected_prefix != entry.selected_prefix) {
      return false;
    }
  }
  return selected_prefix != 0;
}

struct hip_rns_modulus_launch {
  const int8_t* a = nullptr;
  const int8_t* b = nullptr;
  int8_t* c = nullptr;
  int64_t m = 0;
  int64_t n = 0;
  int64_t k = 0;
  int64_t lda = 0;
  int64_t ldb = 0;
  int64_t ldc = 0;
  uint16_t modulus = 0;
  uint32_t modulus_index = 0;
  uint32_t selected_prefix = 0;
  void* stream = nullptr;
};

struct hip_rns_scheduled_modulus_launch {
  const int8_t* a = nullptr;
  const int8_t* b = nullptr;
  int8_t* c = nullptr;
  const rns8_plan_tile_schedule_entry* device_entries = nullptr;
  const uint8_t* zero_a_rows = nullptr;
  const uint8_t* zero_b_cols = nullptr;
  uint64_t entry_count = 0;
  int max_tile_row_blocks = 0;
  int max_tile_col_blocks = 0;
  int64_t k = 0;
  int64_t lda = 0;
  int64_t ldb = 0;
  int64_t ldc = 0;
  uint16_t modulus = 0;
  uint32_t modulus_index = 0;
  uint32_t selected_prefix = 0;
};

uint32_t modulus_reciprocal_u32(uint16_t modulus) {
  return static_cast<uint32_t>((uint64_t{1} << 32u) / static_cast<uint32_t>(modulus));
}

bool checked_rns_modulus_launch(const hip_rns_modulus_launch& launch) {
  if (!launch.a || !launch.b || !launch.c || launch.m <= 0 || launch.n <= 0 || launch.k <= 0 ||
      launch.lda < launch.k || launch.ldb < launch.n || launch.ldc < launch.n || launch.selected_prefix == 0 ||
      launch.selected_prefix > RNS8_MAX_SUPPORTED_PREFIX || launch.modulus_index >= launch.selected_prefix ||
      launch.modulus_index >= RNS8_DEFAULT_MODULUS_COUNT ||
      launch.modulus != kDefaultModuli[launch.modulus_index]) {
    return false;
  }
  return launch.m <= std::numeric_limits<int>::max() && launch.n <= std::numeric_limits<int>::max() &&
         launch.k <= std::numeric_limits<int>::max() && launch.lda <= std::numeric_limits<int>::max() &&
         launch.ldb <= std::numeric_limits<int>::max() && launch.ldc <= std::numeric_limits<int>::max() &&
         RNS8_SAFE_INT32_K_BLOCK <= static_cast<uint32_t>(std::numeric_limits<int>::max());
}

bool checked_scheduled_modulus_launch(const hip_rns_scheduled_modulus_launch& launch) {
  if (!launch.a || !launch.b || !launch.c || !launch.device_entries || launch.entry_count == 0 ||
      launch.max_tile_row_blocks <= 0 || launch.max_tile_col_blocks <= 0 || launch.k <= 0 ||
      launch.lda < launch.k || launch.ldb <= 0 || launch.ldc <= 0 || launch.selected_prefix == 0 ||
      launch.selected_prefix > RNS8_MAX_SUPPORTED_PREFIX || launch.modulus_index >= launch.selected_prefix ||
      launch.modulus_index >= RNS8_DEFAULT_MODULUS_COUNT ||
      launch.modulus != kDefaultModuli[launch.modulus_index]) {
    return false;
  }
  return launch.entry_count <= static_cast<uint64_t>(std::numeric_limits<int>::max()) &&
         launch.max_tile_row_blocks <= std::numeric_limits<int>::max() &&
         launch.max_tile_col_blocks <= std::numeric_limits<int>::max() &&
         launch.k <= std::numeric_limits<int>::max() && launch.lda <= std::numeric_limits<int>::max() &&
         launch.ldb <= std::numeric_limits<int>::max() && launch.ldc <= std::numeric_limits<int>::max() &&
         RNS8_SAFE_INT32_K_BLOCK <= static_cast<uint32_t>(std::numeric_limits<int>::max());
}

hipError_t launch_rns_modulus_gemm(const hip_rns_modulus_launch& launch) {
  if (!checked_rns_modulus_launch(launch)) {
    return hipErrorInvalidValue;
  }
  const int code = rns8_hip_direct_ring_gemm_i8_device_on_stream(
      launch.a,
      launch.b,
      launch.c,
      static_cast<int>(launch.m),
      static_cast<int>(launch.n),
      static_cast<int>(launch.k),
      static_cast<int>(launch.lda),
      static_cast<int>(launch.ldb),
      static_cast<int>(launch.ldc),
      static_cast<int>(launch.modulus),
      modulus_reciprocal_u32(launch.modulus),
      static_cast<int>(launch.modulus_index),
      static_cast<int>(launch.selected_prefix),
      static_cast<int>(RNS8_SAFE_INT32_K_BLOCK),
      launch.stream);
  return code == static_cast<int>(hipSuccess) ? hipSuccess : static_cast<hipError_t>(code);
}

hipError_t launch_rns_scheduled_modulus_gemm(const hip_rns_scheduled_modulus_launch& launch) {
  if (!checked_scheduled_modulus_launch(launch)) {
    return hipErrorInvalidValue;
  }
  const int code = rns8_hip_direct_ring_gemm_i8_scheduled_device(
      launch.a,
      launch.b,
      launch.c,
      launch.device_entries,
      launch.zero_a_rows,
      launch.zero_b_cols,
      static_cast<int>(launch.entry_count),
      launch.max_tile_row_blocks,
      launch.max_tile_col_blocks,
      static_cast<int>(launch.k),
      static_cast<int>(launch.lda),
      static_cast<int>(launch.ldb),
      static_cast<int>(launch.ldc),
      static_cast<int>(launch.modulus),
      modulus_reciprocal_u32(launch.modulus),
      static_cast<int>(launch.modulus_index),
      static_cast<int>(launch.selected_prefix),
      static_cast<int>(RNS8_SAFE_INT32_K_BLOCK));
  return code == static_cast<int>(hipSuccess) ? hipSuccess : static_cast<hipError_t>(code);
}

hipError_t launch_rns_grouped_prefix_gemm(
    const int8_t* a,
    const int8_t* b,
    int8_t* c,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    uint32_t prefix,
    void* stream = nullptr) {
  if (!a || !b || !c || m <= 0 || n <= 0 || k <= 0 || lda < k || ldb < n || ldc < n ||
      prefix == 0 || prefix > RNS8_MAX_SUPPORTED_PREFIX ||
      m > std::numeric_limits<int>::max() || n > std::numeric_limits<int>::max() ||
      k > std::numeric_limits<int>::max() || lda > std::numeric_limits<int>::max() ||
      ldb > std::numeric_limits<int>::max() || ldc > std::numeric_limits<int>::max() ||
      RNS8_SAFE_INT32_K_BLOCK > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
    return hipErrorInvalidValue;
  }
  const int code = rns8_hip_direct_ring_gemm_i8_grouped_prefix_device_on_stream(
      a,
      b,
      c,
      static_cast<int>(m),
      static_cast<int>(n),
      static_cast<int>(k),
      static_cast<int>(lda),
      static_cast<int>(ldb),
      static_cast<int>(ldc),
      static_cast<int>(prefix),
      static_cast<int>(RNS8_SAFE_INT32_K_BLOCK),
      stream);
  return code == static_cast<int>(hipSuccess) ? hipSuccess : static_cast<hipError_t>(code);
}

hipError_t launch_rns_grouped_task_prefix_gemm(
    const int8_t* const* a_ptrs,
    const int8_t* const* b_ptrs,
    int8_t* const* c_ptrs,
    int64_t task_count,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    uint32_t prefix) {
  if (!a_ptrs || !b_ptrs || !c_ptrs || task_count <= 0 || m <= 0 || n <= 0 || k <= 0 || lda < k || ldb < n ||
      ldc < n || prefix == 0 || prefix > RNS8_MAX_SUPPORTED_PREFIX ||
      task_count > std::numeric_limits<int>::max() || m > std::numeric_limits<int>::max() ||
      n > std::numeric_limits<int>::max() || k > std::numeric_limits<int>::max() ||
      lda > std::numeric_limits<int>::max() || ldb > std::numeric_limits<int>::max() ||
      ldc > std::numeric_limits<int>::max() ||
      RNS8_SAFE_INT32_K_BLOCK > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
    return hipErrorInvalidValue;
  }
  const int code = rns8_hip_direct_ring_gemm_i8_grouped_task_prefix_device(
      a_ptrs,
      b_ptrs,
      c_ptrs,
      static_cast<int>(task_count),
      static_cast<int>(m),
      static_cast<int>(n),
      static_cast<int>(k),
      static_cast<int>(lda),
      static_cast<int>(ldb),
      static_cast<int>(ldc),
      static_cast<int>(prefix),
      static_cast<int>(RNS8_SAFE_INT32_K_BLOCK));
  return code == static_cast<int>(hipSuccess) ? hipSuccess : static_cast<hipError_t>(code);
}
#endif

}  // namespace

