#include "core/api_internal.hpp"
#include "core/hip_resources.hpp"
using namespace rns8::detail;
using namespace rns8::detail::api;
struct rns8_ragged_materialization {
  rns8_context* ctx=nullptr;
  const rns8_matrix* source=nullptr;
  RaggedResidueLayout layout;
  cpp_int bound;
  std::vector<int8_t> host;
  hip_direct_device_buffer device;
};
rns8_status rns8_prepare_ragged_materialization(rns8_context* ctx,const rns8_matrix* source,
    uint32_t tile_rows,uint32_t tile_cols,uint64_t budget,rns8_ragged_materialization** out) {
  return guard_api([&]() -> rns8_status {
    if(!ctx || !source || !out || !tile_rows || !tile_cols || tile_rows>128 || tile_cols>128)
      return RNS8_INVALID_ARGUMENT;
    if(ctx->hip_failure_poisoned) return RNS8_BACKEND_FAILURE;
    if(ctx->backend!=RNS8_BACKEND_CPU_REFERENCE && ctx->backend!=RNS8_BACKEND_HIP_DIRECT)
      return RNS8_UNSUPPORTED_BACKEND;
    if(source->in_flight_writer) return RNS8_BUSY;
    if(source->backend!=ctx->backend || !exact_wide_semantics(source->desc.semantics) ||
        !source->prefix || source->prefix>20 || source->desc.rows<=0 || source->desc.cols<=0 ||
        source->generation_exhausted || !matrix_descriptor_matches(*source,source->desc.semantics,
            RNS8_BOUND_NONE,source->desc.rows,source->desc.cols,source->prefix,0,0) ||
        !rns_matrix_storage_matches(*source,ctx->backend,source->desc.rows,source->desc.cols,source->prefix) ||
        !rns_residue_state_current_for_backend(*source,ctx->backend) || source->exact_range_prefix<source->prefix ||
        !exact_range_fits(source->desc.semantics,source->exact_max_magnitude,modulus_product(source->prefix)))
      return RNS8_INVALID_ARGUMENT;
    if(ctx->prepared_references==UINT64_MAX || source->prepared_references==UINT64_MAX) return RNS8_RANGE_ERROR;
    const bool gpu=ctx->backend==RNS8_BACKEND_HIP_DIRECT;
    if(gpu && source->hip_device_id!=ctx->device_id) return RNS8_INVALID_ARGUMENT;
    AddressDomain address;
    if(!checked_host_plane_address(source->desc.rows,source->desc.cols,source->desc.cols,source->prefix,1,address))
      return RNS8_RANGE_ERROR;
    auto owner=std::make_unique<rns8_ragged_materialization>();
    auto& layout=owner->layout;
    layout.rows=uint64_t(source->desc.rows);layout.cols=uint64_t(source->desc.cols);
    layout.tile_rows=tile_rows;layout.tile_cols=tile_cols;
    layout.grid_rows=1+(layout.rows-1)/tile_rows;layout.grid_cols=1+(layout.cols-1)/tile_cols;
    layout.value_instance=source->matrix_instance_id;layout.value_generation=source->content_generation;
    layout.signed_value=source->desc.semantics==RNS8_EXACT_WIDE_SIGNED;
    const uint64_t limit=uint64_t(PTRDIFF_MAX);
    if(layout.grid_rows>limit/layout.grid_cols/sizeof(RaggedResidueTile)) return RNS8_RANGE_ERROR;
    const uint64_t metadata=layout.grid_rows*layout.grid_cols*sizeof(RaggedResidueTile);
    const uint64_t snapshot=gpu?source->hip_residue_bytes:0;
    if(snapshot>limit || metadata>limit-snapshot) return RNS8_RANGE_ERROR;
    const uint64_t base=metadata+snapshot;
    if(base>budget) return RNS8_WORKSPACE_TOO_SMALL;
    layout.tiles.resize(static_cast<size_t>(layout.grid_rows*layout.grid_cols));
    std::vector<int8_t> copied;
    const int8_t* uniform=source->residues.data();
    if(gpu) {
      copied.resize(static_cast<size_t>(snapshot));
      const auto status=hip_direct_copy_device_to_host(ctx->device_id,copied.data(),source->hip_residues,copied.size());
      if(status!=RNS8_SUCCESS) return status;
      uniform=copied.data();
    }
    const uint64_t cells=layout.rows*layout.cols,area=uint64_t(tile_rows)*tile_cols;
    const cpp_int product=modulus_product(source->prefix),mask=(cpp_int(1)<<64)-1;
    std::vector<int8_t> residues(source->prefix);
    for(uint64_t tr=0;tr<layout.grid_rows;++tr) for(uint64_t tc=0;tc<layout.grid_cols;++tc) {
      auto& tile=layout.tiles[tr*layout.grid_cols+tc];cpp_int bound=0;
      for(uint64_t row=tr*tile_rows;row<std::min(layout.rows,(tr+1)*tile_rows);++row)
        for(uint64_t col=tc*tile_cols;col<std::min(layout.cols,(tc+1)*tile_cols);++col) {
          for(uint32_t p=0;p<source->prefix;++p) {
            const int value=uniform[p*cells+row*layout.cols+col],q=kDefaultModuli[p];
            if(value<-(q/2) || value>(q-1)/2) return RNS8_INVALID_ARGUMENT;
            residues[p]=static_cast<int8_t>(value);
          }
          cpp_int value=reconstruct_canonical(residues,source->prefix);
          if(layout.signed_value && value>=(product+1)/2) value-=product;
          const cpp_int magnitude=boost::multiprecision::abs(value);
          if(magnitude>source->exact_max_magnitude) return RNS8_RANGE_ERROR;
          if(magnitude>bound) bound=magnitude;
        }
      tile.prefix=1;
      while(tile.prefix<source->prefix && !exact_range_fits(source->desc.semantics,bound,modulus_product(tile.prefix)))
        ++tile.prefix;
      tile.byte_offset=layout.bytes;
      const uint64_t bytes=area*tile.prefix;
      if(layout.bytes>limit-bytes) return RNS8_RANGE_ERROR;
      layout.bytes+=bytes;
      for(unsigned limb=0;limb<3;++limb) tile.magnitude_bound[limb]=((bound>>(64*limb))&mask).convert_to<uint64_t>();
      if(bound>owner->bound) owner->bound=bound;
    }
    if(!valid_ragged_residue_layout(layout,layout.value_instance,layout.value_generation,source->prefix))
      return RNS8_INVALID_ARGUMENT;
    const uint64_t copies=gpu?2:1;
    if(layout.bytes>(limit-base)/copies) return RNS8_RANGE_ERROR;
    if(layout.bytes>(budget-base)/copies) return RNS8_WORKSPACE_TOO_SMALL;
    owner->host.assign(static_cast<size_t>(layout.bytes),0);
    for(uint64_t row=0;row<layout.rows;++row) for(uint64_t col=0;col<layout.cols;++col) {
      const auto& tile=layout.tiles[(row/tile_rows)*layout.grid_cols+col/tile_cols];
      const uint64_t local=(row%tile_rows)*tile_cols+col%tile_cols;
      for(uint32_t p=0;p<tile.prefix;++p) owner->host[tile.byte_offset+p*area+local]=uniform[p*cells+row*layout.cols+col];
    }
    if(gpu) {
      auto status=owner->device.allocate(ctx->device_id,static_cast<size_t>(layout.bytes));
      if(status!=RNS8_SUCCESS) return status;
      status=hip_direct_copy_host_to_device(ctx->device_id,owner->device.get(),owner->host.data(),owner->host.size());
      if(status!=RNS8_SUCCESS) return status;
      std::vector<int8_t>().swap(owner->host);
    }
    owner->ctx=ctx;owner->source=source;
    ++ctx->prepared_references;++source->prepared_references;
    *out=owner.release();return RNS8_SUCCESS;
  });
}
rns8_status rns8_get_ragged_resources(const rns8_ragged_materialization* owner,rns8_ragged_resources* out) {
  if(!owner || !out || !valid_abi(out->struct_size,out->abi_version,sizeof(*out))) return RNS8_INVALID_ARGUMENT;
  rns8_ragged_resources r{};r.struct_size=sizeof(r);r.abi_version=RNS8_ABI_VERSION;
  r.minimum_prefix=20;
  for(const auto& tile:owner->layout.tiles) {r.minimum_prefix=std::min(r.minimum_prefix,tile.prefix);r.maximum_prefix=std::max(r.maximum_prefix,tile.prefix);}
  r.tile_count=owner->layout.tiles.size();r.payload_bytes=owner->layout.bytes;
  r.directory_bytes=r.tile_count*sizeof(RaggedResidueTile);
  r.source_instance=owner->layout.value_instance;r.source_generation=owner->layout.value_generation;
  *out=r;return RNS8_SUCCESS;
}
rns8_status rns8_destroy_ragged_materialization(rns8_ragged_materialization* owner) {
  if(owner) {--owner->ctx->prepared_references;--owner->source->prepared_references;delete owner;}
  return RNS8_SUCCESS;
}
rns8_status rns8_materialize_ragged_exact(rns8_context* ctx,const rns8_ragged_materialization* owner,
    rns8_matrix* output,uint64_t budget) {
  return guard_api([&]() -> rns8_status {
    if(!ctx || !owner || owner->ctx!=ctx || !output || output==owner->source) return RNS8_INVALID_ARGUMENT;
    if(ctx->hip_failure_poisoned) return RNS8_BACKEND_FAILURE;
    const auto& source=*owner->source;
    if(source.in_flight_writer || output->in_flight_writer || output->in_flight_readers) return RNS8_BUSY;
    if(source.generation_exhausted || source.matrix_instance_id!=owner->layout.value_instance ||
        source.content_generation!=owner->layout.value_generation || !rns_residue_state_current_for_backend(source,ctx->backend))
      return RNS8_INVALID_ARGUMENT;
    if(output->generation_exhausted) return RNS8_RANGE_ERROR;
    if(output->backend!=ctx->backend || !output->prefix || output->prefix>20 || !matrix_descriptor_matches(*output,source.desc.semantics,RNS8_BOUND_NONE,
        source.desc.rows,source.desc.cols,output->prefix,0,0) ||
        !rns_matrix_storage_matches(*output,ctx->backend,source.desc.rows,source.desc.cols,output->prefix) ||
        (ctx->backend==RNS8_BACKEND_HIP_DIRECT && output->hip_device_id!=ctx->device_id)) return RNS8_INVALID_ARGUMENT;
    if(!exact_range_fits(source.desc.semantics,owner->bound,modulus_product(output->prefix))) return RNS8_RANGE_ERROR;
    ResidueLayout target;
    if(!make_residue_layout(ResidueLayoutKind::modulus_major,source.desc.rows,source.desc.cols,source.desc.cols,
        output->prefix,0,0,target)) return RNS8_INVALID_ARGUMENT;
    cpp_int bound=owner->bound; // stage all fallible proof work before publication
    bool started=false;
    struct Guard { rns8_matrix& output;bool& started;bool published=false;
      ~Guard(){if(started && !published) invalidate_output_currentness(output);} } guard{*output,started};
    rns8_status status;
    if(ctx->backend==RNS8_BACKEND_HIP_DIRECT) {
      status=hip_direct_materialize_ragged_residues(ctx->device_id,owner->device.get(),owner->layout,
          output->hip_residues,target,source.matrix_instance_id,source.content_generation,budget,started);
      if(status!=RNS8_SUCCESS) return status;
      mark_output_device_residues_current(*output);
    } else {
      status=materialize_ragged_residues_host(owner->host.data(),owner->layout,output->residues.data(),target,
          source.matrix_instance_id,source.content_generation,budget);
      if(status!=RNS8_SUCCESS) return status;
      mark_output_host_residues_current(*output);
    }
    output->exact_max_magnitude.swap(bound);output->exact_axis_ranges.rows.clear();output->exact_axis_ranges.cols.clear();
    output->exact_range_prefix=output->prefix;guard.published=true;
    return RNS8_SUCCESS;
  });
}
