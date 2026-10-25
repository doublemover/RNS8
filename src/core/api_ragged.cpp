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
  void *device_payload=nullptr,*device_directory=nullptr,*device_status=nullptr,*staged_output=nullptr;
  uint64_t staged_capacity=0;
  rns8_matrix* active_output=nullptr;
  bool pending=false,source_reader=false,context_reference=true,cleanup_started=false;
  rns8_ragged_materialization* failed_cleanup_next=nullptr;
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
    if(ctx->prepared_references==UINT64_MAX || source->prepared_references==UINT64_MAX || source->in_flight_readers==UINT64_MAX) return RNS8_RANGE_ERROR;
    const bool gpu=ctx->backend==RNS8_BACKEND_HIP_DIRECT;
    if(gpu && source->hip_device_id!=ctx->device_id) return RNS8_INVALID_ARGUMENT;
    AddressDomain address;
    if(!checked_host_plane_address(source->desc.rows,source->desc.cols,source->desc.cols,source->prefix,1,address))
      return RNS8_RANGE_ERROR;
    const auto discard=[](rns8_ragged_materialization* owner) noexcept {
      if(!owner) return;
      auto* context=owner->ctx;
      if(rns8_destroy_ragged_materialization(owner)==RNS8_SUCCESS) return;
      context->hip_failure_poisoned=true;
      --context->prepared_references;owner->context_reference=false;
      owner->failed_cleanup_next=context->failed_ragged_cleanup;
      context->failed_ragged_cleanup=owner;
    };
    std::unique_ptr<rns8_ragged_materialization,decltype(discard)> owner(new rns8_ragged_materialization,discard);
    owner->ctx=ctx;owner->source=source;
    ++ctx->prepared_references;++source->prepared_references;
    if(gpu) {++source->in_flight_readers;owner->source_reader=true;}
    auto& layout=owner->layout;
    layout.rows=uint64_t(source->desc.rows);layout.cols=uint64_t(source->desc.cols);
    layout.tile_rows=tile_rows;layout.tile_cols=tile_cols;
    layout.grid_rows=1+(layout.rows-1)/tile_rows;layout.grid_cols=1+(layout.cols-1)/tile_cols;
    layout.value_instance=source->matrix_instance_id;layout.value_generation=source->content_generation;
    layout.signed_value=source->desc.semantics==RNS8_EXACT_WIDE_SIGNED;
    const uint64_t limit=uint64_t(PTRDIFF_MAX);
    if(layout.grid_rows>limit/layout.grid_cols/sizeof(RaggedResidueTile)) return RNS8_RANGE_ERROR;
    const uint64_t metadata=layout.grid_rows*layout.grid_cols*sizeof(RaggedResidueTile);
    const uint64_t snapshot=gpu?metadata+sizeof(int):0; // device directory and proof status
    if(snapshot>limit || metadata>limit-snapshot) return RNS8_RANGE_ERROR;
    const uint64_t base=metadata+snapshot;
    if(base>budget) return RNS8_WORKSPACE_TOO_SMALL;
    layout.tiles.resize(static_cast<size_t>(layout.grid_rows*layout.grid_cols));
    const int8_t* uniform=source->residues.data();
    if(gpu) {
      auto status=hip_direct_allocate(ctx->device_id,metadata,&owner->device_directory);
      if(status!=RNS8_SUCCESS) return status;
      status=hip_direct_allocate(ctx->device_id,sizeof(int),&owner->device_status);
      if(status!=RNS8_SUCCESS) return status;
      uint64_t bound[3]{};const cpp_int mask=(cpp_int(1)<<64)-1;
      for(unsigned limb=0;limb<3;++limb) bound[limb]=((source->exact_max_magnitude>>(64*limb))&mask).convert_to<uint64_t>();
      owner->pending=true;
      status=hip_direct_discover_ragged_bounds(ctx->device_id,source->hip_residues,source->prefix,bound,layout,
          owner->device_directory,owner->device_status);
      if(status!=RNS8_SUCCESS) return status;
      owner->pending=false;
    }
    const uint64_t cells=layout.rows*layout.cols,area=uint64_t(tile_rows)*tile_cols;
    const cpp_int product=modulus_product(source->prefix),mask=(cpp_int(1)<<64)-1;
    std::vector<int8_t> residues(source->prefix);
    for(uint64_t tr=0;tr<layout.grid_rows;++tr) for(uint64_t tc=0;tc<layout.grid_cols;++tc) {
      auto& tile=layout.tiles[tr*layout.grid_cols+tc];cpp_int bound=0;
      if(gpu) bound=cpp_int(tile.magnitude_bound[0])+(cpp_int(tile.magnitude_bound[1])<<64)+(cpp_int(tile.magnitude_bound[2])<<128);
      else for(uint64_t row=tr*tile_rows;row<std::min(layout.rows,(tr+1)*tile_rows);++row)
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
    const uint64_t copies=1; // payload stays on its execution backend
    if(layout.bytes>(limit-base)/copies) return RNS8_RANGE_ERROR;
    if(layout.bytes>(budget-base)/copies) return RNS8_WORKSPACE_TOO_SMALL;
    if(!gpu) owner->host.assign(static_cast<size_t>(layout.bytes),0);
    if(!gpu) for(uint64_t row=0;row<layout.rows;++row) for(uint64_t col=0;col<layout.cols;++col) {
      const auto& tile=layout.tiles[(row/tile_rows)*layout.grid_cols+col/tile_cols];
      const uint64_t local=(row%tile_rows)*tile_cols+col%tile_cols;
      for(uint32_t p=0;p<tile.prefix;++p) owner->host[tile.byte_offset+p*area+local]=uniform[p*cells+row*layout.cols+col];
    }
    if(gpu) {
      auto status=hip_direct_allocate(ctx->device_id,layout.bytes,&owner->device_payload);
      if(status!=RNS8_SUCCESS) return status;
      owner->pending=true;
      status=hip_direct_compact_ragged_residues(ctx->device_id,source->hip_residues,source->prefix,layout,
          owner->device_payload,owner->device_directory);
      if(status!=RNS8_SUCCESS) return status;
      owner->pending=false;
      --source->in_flight_readers;owner->source_reader=false;
      // This finalized immutable directory is retained and already current on device;
      // later materializations must not upload the same records again.
    }
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
  r.device_metadata_bytes=(owner->device_directory?r.directory_bytes:0)+(owner->device_status?sizeof(int):0);
  r.materialization_capacity_bytes=owner->staged_output?owner->staged_capacity:0;
  r.source_instance=owner->layout.value_instance;r.source_generation=owner->layout.value_generation;
  *out=r;return RNS8_SUCCESS;
}
rns8_status rns8_destroy_ragged_materialization(rns8_ragged_materialization* owner) {
  if(owner) {
    owner->cleanup_started=true;
    if(owner->pending) {
      if(hip_direct_synchronize(owner->ctx->device_id)!=RNS8_SUCCESS) return RNS8_BACKEND_FAILURE;
      owner->pending=false;
    }
    if(owner->active_output) {
      owner->active_output->in_flight_writer=false;
      --owner->active_output->prepared_references;owner->active_output=nullptr;
    }
    if(owner->source_reader) {--owner->source->in_flight_readers;owner->source_reader=false;}
    rns8_status cleanup=RNS8_SUCCESS;
    for(auto** pointer:{&owner->staged_output,&owner->device_payload,&owner->device_directory,&owner->device_status}) {
      if(!*pointer) continue;
      const auto status=hip_direct_free(owner->ctx->device_id,*pointer);
      if(status==RNS8_SUCCESS) *pointer=nullptr;
      else if(cleanup==RNS8_SUCCESS) cleanup=status;
    }
    if(cleanup!=RNS8_SUCCESS) return cleanup;
    if(owner->context_reference) --owner->ctx->prepared_references;
    --owner->source->prepared_references;delete owner;
  }
  return RNS8_SUCCESS;
}
rns8_status rns8_materialize_ragged_exact(rns8_context* ctx,const rns8_ragged_materialization* owner,
    rns8_matrix* output,uint64_t budget) {
  return guard_api([&]() -> rns8_status {
    if(!ctx || !owner || owner->ctx!=ctx || owner->cleanup_started || !output || output==owner->source) return RNS8_INVALID_ARGUMENT;
    if(ctx->hip_failure_poisoned) return RNS8_BACKEND_FAILURE;
    const auto& source=*owner->source;
    if(source.in_flight_writer || output->in_flight_writer || output->in_flight_readers) return RNS8_BUSY;
    if(source.generation_exhausted || source.matrix_instance_id!=owner->layout.value_instance ||
        source.content_generation!=owner->layout.value_generation || !rns_residue_state_current_for_backend(source,ctx->backend))
      return RNS8_INVALID_ARGUMENT;
    if(output->generation_exhausted || output->prepared_references==UINT64_MAX) return RNS8_RANGE_ERROR;
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
      ~Guard(){if(started && !published) {
        clear_residue_current(output);clear_byte_limb_current(output);clear_native_current(output);
      }} } guard{*output,started};
    rns8_status status;
    if(ctx->backend==RNS8_BACKEND_HIP_DIRECT) {
      auto& execution=*const_cast<rns8_ragged_materialization*>(owner);
      const uint64_t metadata=owner->layout.tiles.size()*sizeof(RaggedResidueTile)+sizeof(int);
      const uint64_t capacity=std::max(target.bytes,execution.staged_capacity);
      if(capacity>UINT64_MAX-metadata || capacity+metadata>budget) return RNS8_WORKSPACE_TOO_SMALL;
      if(execution.staged_capacity<target.bytes) {
        if(execution.staged_output) {
          status=hip_direct_free(ctx->device_id,execution.staged_output);
          if(status!=RNS8_SUCCESS) return status;
          execution.staged_output=nullptr;execution.staged_capacity=0;
        }
        status=hip_direct_allocate(ctx->device_id,target.bytes,&execution.staged_output);
        if(status!=RNS8_SUCCESS) return status;
        execution.staged_capacity=target.bytes;
      }
      execution.active_output=output;++output->prepared_references;output->in_flight_writer=true;
      execution.pending=true;
      struct Reservation {
        rns8_ragged_materialization& owner;bool& started;bool completed=false;
        void release() noexcept {
          owner.pending=false;owner.active_output->in_flight_writer=false;
          --owner.active_output->prepared_references;owner.active_output=nullptr;
        }
        ~Reservation() noexcept {
          if(completed) return;
          if(started) {
            clear_residue_current(*owner.active_output);clear_byte_limb_current(*owner.active_output);
            clear_native_current(*owner.active_output);
          }
          if(hip_direct_synchronize(owner.ctx->device_id)==RNS8_SUCCESS) release();
          else owner.ctx->hip_failure_poisoned=true;
        }
      } reservation{execution,started};
      status=hip_direct_materialize_ragged_residues(ctx->device_id,owner->device_payload,owner->layout,
          output->hip_residues,target,source.matrix_instance_id,source.content_generation,budget,started,
          execution.staged_output,execution.device_directory,execution.device_status,true);
      if(status!=RNS8_SUCCESS) return status;
      mark_output_device_residues_current(*output);
      reservation.release();reservation.completed=true;
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

rns8_status rns8::detail::api::drain_failed_ragged_cleanup(rns8_context& ctx) {
  while(ctx.failed_ragged_cleanup) {
    auto* owner=ctx.failed_ragged_cleanup;
    auto* next=owner->failed_cleanup_next;
    const auto status=rns8_destroy_ragged_materialization(owner);
    if(status!=RNS8_SUCCESS) return status;
    ctx.failed_ragged_cleanup=next;
  }
  return RNS8_SUCCESS;
}
