#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
extern "C" int rns8_hip_materialize_ragged(
    const int8_t*, const RaggedResidueTile*, int, int, int, bool, int8_t*, ResidueLayout, int*);
#endif
rns8_status hip_direct_materialize_ragged_residues(
    int device_id, const void* source, const RaggedResidueLayout& layout,
    void* destination, const ResidueLayout& target, uint64_t instance, uint64_t generation,
    uint64_t max_scratch_bytes, bool& commit_started,
    void* staged, void* directory, void* device_status,bool prepared_directory_current) {
  commit_started = false;
  if (!compatible_ragged_destination(layout, target, instance, generation) ||
      !nonoverlapping_byte_ranges(source, layout.bytes, destination, target.bytes)) return RNS8_INVALID_ARGUMENT;
  if (layout.bytes > INT32_MAX || target.bytes > INT32_MAX ||
      layout.tiles.size() > INT32_MAX / sizeof(RaggedResidueTile)) return RNS8_UNSUPPORTED_BACKEND;
  const uint64_t metadata_bytes = layout.tiles.size() * sizeof(RaggedResidueTile);
  if (target.bytes + metadata_bytes + sizeof(int) > max_scratch_bytes) return RNS8_WORKSPACE_TOO_SMALL;
  const std::pair<const void*,uint64_t> spans[]={{source,layout.bytes},{destination,target.bytes},
      {staged,target.bytes},{directory,metadata_bytes},{device_status,sizeof(int)}};
  if(reinterpret_cast<uintptr_t>(directory)%alignof(RaggedResidueTile) ||
      reinterpret_cast<uintptr_t>(device_status)%alignof(int)) return RNS8_INVALID_ARGUMENT;
  for(unsigned i=0;i<5;++i) for(unsigned j=i+1;j<5;++j)
    if(!nonoverlapping_byte_ranges(spans[i].first,spans[i].second,spans[j].first,spans[j].second))
      return RNS8_INVALID_ARGUMENT;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  auto status = set_hip_device(device_id);
  if (status != RNS8_SUCCESS) return status;
  // Only the generation-bound immutable public owner supplies true. Borrowed
  // independent callers keep the ordinary upload-on-call contract by default.
  if(!prepared_directory_current){status=hip_direct_copy_host_to_device(device_id,directory,layout.tiles.data(),metadata_bytes);if(status!=RNS8_SUCCESS)return status;}
  // No compact destination byte survives a successful materialization. Retain
  // the old-output precopy only when physical padding really must be preserved.
  const bool overwritten=target.kind==ResidueLayoutKind::modulus_major?target.ld==target.cols:
      target.rows%target.tile_rows==0 && target.cols%target.tile_cols==0;
  if(!overwritten){status=hip_direct_copy_device_to_device(device_id,staged,destination,target.bytes);if(status!=RNS8_SUCCESS)return status;}
  if (hipMemset(device_status, 0, sizeof(int)) != hipSuccess) return RNS8_BACKEND_FAILURE;
  const auto error = timed_hip_operation("ragged_exact_materialization_kernel", [&]() {
    const auto launched = static_cast<hipError_t>(rns8_hip_materialize_ragged(
        static_cast<const int8_t*>(source), static_cast<const RaggedResidueTile*>(directory),
        static_cast<int>(layout.tile_rows), static_cast<int>(layout.tile_cols), static_cast<int>(layout.grid_cols),
        layout.signed_value, static_cast<int8_t*>(staged), target, static_cast<int*>(device_status)));
    return launched == hipSuccess ? hipDeviceSynchronize() : launched;
  });
  if (error != hipSuccess) return RNS8_BACKEND_FAILURE;
  int failure = 0;
  status = hip_direct_copy_device_to_host(device_id, &failure, device_status, sizeof(failure));
  if (status != RNS8_SUCCESS) return status;
  if (failure) return failure == 2 ? RNS8_RANGE_ERROR : RNS8_INVALID_ARGUMENT;
  commit_started = true;
  status=hip_direct_copy_device_to_device(device_id, destination, staged, target.bytes);
  const auto drain=hip_direct_synchronize(device_id);
  return status==RNS8_SUCCESS?drain:status;
#else
  (void)device_id;(void)prepared_directory_current;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
extern "C" int rns8_hip_discover_ragged_bounds(const int8_t*, int, int, int, int, int, int,
    bool, uint64_t, uint64_t, uint64_t, RaggedResidueTile*, int*);
extern "C" int rns8_hip_compact_ragged(const int8_t*, int, int, const RaggedResidueTile*, int, int, int, int8_t*);
#endif
// Scratch is caller-owned so failed drains never destroy live kernel resources.
rns8_status hip_direct_discover_ragged_bounds(int device_id, const void* source,
    uint32_t prefix, const uint64_t bound[3], RaggedResidueLayout& layout,
    void* device_directory, void* device_status) {
  if (!source || !bound || !device_directory || !device_status || !layout.rows || !layout.cols ||
      !layout.tile_rows || !layout.tile_cols || layout.tile_rows > 128 || layout.tile_cols > 128 ||
      layout.grid_rows != 1 + (layout.rows-1)/layout.tile_rows ||
      layout.grid_cols != 1 + (layout.cols-1)/layout.tile_cols ||
      layout.tiles.size() != layout.grid_rows*layout.grid_cols ||
      !prefix || prefix > 20 || layout.tiles.size() > INT32_MAX/sizeof(RaggedResidueTile) ||
      !qualified_i32_matrix_address(layout.rows,layout.cols,layout.cols,prefix)) return RNS8_INVALID_ARGUMENT;
  const uint64_t directory_bytes=layout.tiles.size()*sizeof(RaggedResidueTile);
  const uint64_t source_bytes=layout.rows*layout.cols*prefix;
  if(reinterpret_cast<uintptr_t>(device_directory)%alignof(RaggedResidueTile) ||
      reinterpret_cast<uintptr_t>(device_status)%alignof(int) ||
      !nonoverlapping_byte_ranges(source,source_bytes,device_directory,directory_bytes) ||
      !nonoverlapping_byte_ranges(source,source_bytes,device_status,sizeof(int)) ||
      !nonoverlapping_byte_ranges(device_directory,directory_bytes,device_status,sizeof(int)))
    return RNS8_INVALID_ARGUMENT;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  auto status = set_hip_device(device_id);
  if (status != RNS8_SUCCESS) return status;
  if (hipMemset(device_status,0,sizeof(int)) != hipSuccess) return RNS8_BACKEND_FAILURE;
  const auto error = timed_hip_operation("ragged_device_bound_discovery", [&]() {
    const auto launched = static_cast<hipError_t>(rns8_hip_discover_ragged_bounds(
        static_cast<const int8_t*>(source), int(layout.rows), int(layout.cols), int(prefix),
        int(layout.tile_rows), int(layout.tile_cols), int(layout.grid_cols), layout.signed_value,
        bound[0],bound[1],bound[2],static_cast<RaggedResidueTile*>(device_directory),static_cast<int*>(device_status)));
    return launched == hipSuccess ? hipDeviceSynchronize() : launched;
  });
  if (error != hipSuccess) return RNS8_BACKEND_FAILURE;
  int failure=0;
  status=hip_direct_copy_device_to_host(device_id,&failure,device_status,sizeof(failure));
  if(status!=RNS8_SUCCESS) return status;
  if(failure) return failure==2?RNS8_RANGE_ERROR:RNS8_INVALID_ARGUMENT;
  return hip_direct_copy_device_to_host(device_id,layout.tiles.data(),device_directory,
      layout.tiles.size()*sizeof(RaggedResidueTile));
#else
  (void)device_id;return RNS8_UNSUPPORTED_BACKEND;
#endif
}
rns8_status hip_direct_compact_ragged_residues(int device_id, const void* source,
    uint32_t prefix, const RaggedResidueLayout& layout, void* destination, void* device_directory) {
  if (!device_directory || !valid_ragged_residue_layout(layout,layout.value_instance,layout.value_generation,prefix) ||
      !qualified_i32_matrix_address(layout.rows,layout.cols,layout.cols,prefix) ||
      layout.tiles.size()>INT32_MAX/sizeof(RaggedResidueTile) ||
      !nonoverlapping_byte_ranges(source,layout.rows*layout.cols*prefix,destination,layout.bytes))
    return RNS8_INVALID_ARGUMENT;
  const uint64_t directory_bytes=layout.tiles.size()*sizeof(RaggedResidueTile);
  if(reinterpret_cast<uintptr_t>(device_directory)%alignof(RaggedResidueTile) ||
      !nonoverlapping_byte_ranges(source,layout.rows*layout.cols*prefix,device_directory,directory_bytes) ||
      !nonoverlapping_byte_ranges(destination,layout.bytes,device_directory,directory_bytes))
    return RNS8_INVALID_ARGUMENT;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  auto status = set_hip_device(device_id);
  if(status!=RNS8_SUCCESS) return status;
  status=hip_direct_copy_host_to_device(device_id,device_directory,layout.tiles.data(),layout.tiles.size()*sizeof(RaggedResidueTile));
  if(status!=RNS8_SUCCESS) return status;
  const auto error=timed_hip_operation("ragged_device_payload_compaction",[&]() {
    const auto launched=static_cast<hipError_t>(rns8_hip_compact_ragged(static_cast<const int8_t*>(source),
        int(layout.rows),int(layout.cols),static_cast<const RaggedResidueTile*>(device_directory),
        int(layout.tile_rows),int(layout.tile_cols),int(layout.grid_cols),static_cast<int8_t*>(destination)));
    return launched==hipSuccess?hipDeviceSynchronize():launched;
  });
  return error==hipSuccess?RNS8_SUCCESS:RNS8_BACKEND_FAILURE;
#else
  (void)device_id;return RNS8_UNSUPPORTED_BACKEND;
#endif
}
