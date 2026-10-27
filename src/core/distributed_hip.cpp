#include "distributed_hip.hpp"
#include "backend_common/distributed_device.hpp"
#include "backend_common/distributed_integrity.hpp"
#include "backend_common/arithmetic_certificate.hpp"
#include "core/hip_resources.hpp"
#include <atomic>
#include <chrono>
#include <cstring>
#include <thread>

#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
extern "C" int rns8_hip_distributed_pack(const uint64_t*, const uint64_t*, int8_t*, int8_t*,
    uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, bool, uint32_t, uint64_t, void*);
extern "C" int rns8_hip_distributed_reduce_partials(const int8_t*,int8_t*,const rns8::detail::distributed::DeviceTile*,uint64_t,uint64_t,uint32_t,uint32_t,void*);
extern "C" int rns8_hip_distributed_export(const int8_t*, uint64_t*, int, int, uint32_t,
    uint32_t, const rns8::detail::distributed::DeviceTile*, bool, bool, int*, void*);
extern "C" int rns8_hip_distributed_materialize(const uint64_t*, int8_t*, int, uint32_t, bool, void*);
extern "C" int rns8_hip_distributed_integrity(const uint64_t*,const uint64_t*,const uint64_t*,
    uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,bool,int*,void*);
extern "C" int rns8_hip_direct_ring_gemm_i8_device_on_stream(const int8_t*, const int8_t*, int8_t*,
    int, int, int, int, int, int, int, uint32_t, int, int, int, void*);
#endif

namespace rns8::detail::distributed {
namespace {
uint64_t monotonic_ns() {
  return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count());
}
uint64_t deadline_after(uint64_t duration) {
  const auto now = monotonic_ns(); return duration > UINT64_MAX - now ? UINT64_MAX : now + duration;
}
}

struct HipExecution::Impl {
  std::shared_ptr<const Plan> plan;
  std::shared_ptr<const Inputs> inputs;
  HipProgress progress;
  uint64_t readers = 0, deadline = 0;
  bool issued = false, final_queued = false, closed = false;
  bool initialization_pending = false;
  bool drain_markers_queued = false;
  std::vector<std::string> certificates;
  Impl* quarantine_next = nullptr;
  inline static std::atomic<Impl*> quarantined{nullptr};
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  struct Slot {
    hip_unique_stream stream;
    hip_unique_event ready;
    hip_direct_device_buffer packed;
    uint64_t work = UINT64_MAX;
    bool ready_observed = false;
  };
  struct Device {
    int ordinal = -1;
    std::vector<uint64_t> work;
    size_t next = 0;
    uint64_t packed_bytes = 0;
    std::vector<Slot> slots;
    hip_direct_device_buffer a, b, output;
    hip_unique_event staged;
  };
  std::vector<Device> devices;
  hip_unique_stream gather_stream;
  hip_unique_event completion, stage_arrived;
  hip_direct_device_buffer gather, partials, result, directory, status;
  hip_direct_device_buffer uniform, uniform_pending;
  uint32_t uniform_prefix = 0;
  hip_pinned_host_buffer staging, host_status;
  std::vector<uint64_t> export_staging;
  // Retained through failed initialization drains, just like native snapshots.
  std::vector<DeviceTile> tile_records;
  std::vector<Phase> phases;
  std::vector<uint8_t> queued;
  uint64_t staged_work = UINT64_MAX;
  uint32_t stage_phase = 0; // 1 source D2H pending; 2 destination H2D pending

  Admission error(hipError_t code, const char* phase, uint32_t rank = UINT32_MAX,
                  uint64_t work = UINT64_MAX) {
    if (code == hipSuccess) return {};
    if (progress.operation) {
      progress.operation = reject(RNS8_BACKEND_FAILURE, phase, rank, work);
      progress.runtime_error = int(code);
    }
    progress.output_current = false;
    progress.state = issued ? State::draining : State::failed;
    return progress.operation;
  }
  void pending(bool value) {
    const auto set = [value](auto& buffer) { if (value) buffer.mark_pending(); else buffer.mark_completed(); };
    for (auto& device : devices) {
      set(device.a); set(device.b); set(device.output);
      for (auto& slot : device.slots) set(slot.packed);
    }
    set(gather); set(partials); set(result); set(directory); set(status); set(staging); set(host_status); set(uniform); set(uniform_pending);
  }
  Admission allocate(hip_direct_device_buffer& buffer, int device, uint64_t bytes) {
    if (!bytes) return {};
    if (bytes > SIZE_MAX) return reject(RNS8_RANGE_ERROR, "device_allocation_extent");
    const auto status_code = buffer.allocate(device, size_t(bytes));
    return status_code == RNS8_SUCCESS ? Admission{} : reject(status_code, "device_allocation");
  }
  static hipError_t event_create(hip_unique_event& event) {
    hipEvent_t handle = nullptr;
    const auto code = hipEventCreateWithFlags(&handle, hipEventDisableTiming);
    if (code == hipSuccess) event.reset(handle);
    return code;
  }
  Admission initialize() {
    const auto& r = plan->request();
    if (r.m > INT32_MAX || r.n > INT32_MAX || r.k > INT32_MAX ||
        r.lda > INT32_MAX || r.ldb > INT32_MAX || r.m * r.n > INT32_MAX)
      return reject(RNS8_UNSUPPORTED_BACKEND, "local_kernel_address_width");
    std::vector<Rank> observed; std::vector<Edge> edges;
    std::vector<uint32_t> ordinals;
    for (const auto& rank : r.ranks) ordinals.push_back(rank.ordinal);
    auto admission = HipExecution::discover(ordinals, observed, edges);
    if (!admission) return admission;
    for (size_t rank = 0; rank < observed.size(); ++rank)
      if (observed[rank].physical_id != r.ranks[rank].physical_id ||
          observed[rank].target != r.ranks[rank].target)
        return reject(RNS8_INVALID_ARGUMENT, "runtime_device_identity_changed", uint32_t(rank));
    const auto plan_hash = rns8::detail::sha256(plan->identity());
    for (const auto& rank : observed) {
      arithmetic::Certificate c;
      c.semantics = uint32_t(r.semantics); c.signed_output = plan->signed_value();
      c.requires_range_guard = c.bound_known = true;
      for (const auto& tile : plan->tiles()) c.output_bound = std::max(c.output_bound, tile.magnitude);
      for (uint32_t plane = 0; plane < r.prefix; ++plane) c.basis.push_back(garner::kConstants.moduli[plane]);
      auto& instruction = c.instruction;
      instruction.target = rank.target; instruction.opcode = "rns8_ring_gemm_i8_i32_tiled_kernel";
#ifdef RNS8_BUILD_MANIFEST_JSON
      instruction.compiler_build = RNS8_BUILD_MANIFEST_JSON;
#else
      instruction.compiler_build = "unavailable-direct-build-manifest";
#endif
      instruction.compiler_flags = instruction.compiler_build;
      instruction.lowering_identity = "distributed-axes-v2;integer-modular-K-reduction;native-centered-reducer;three-limb-gather-crt;plan=" + plan_hash;
      instruction.safe_k = 65536; instruction.lhs_abs = instruction.rhs_abs = 128; instruction.prior_abs = 128;
      for (auto modulus : c.basis) {
        const auto base = uint32_t(c.dag.size());
        arithmetic::Node input; input.bits = 8; input.equality = arithmetic::Equality::modular_value;
        input.input = {-Integer(modulus / 2), Integer((modulus - 1) / 2), 1, modulus};
        c.dag.push_back(input); c.dag.push_back(input); input.bits = 32; c.dag.push_back(input);
        arithmetic::Node dot; dot.operation = arithmetic::Operation::dot; dot.equality = arithmetic::Equality::modular_value;
        dot.left = base; dot.right = base + 1; dot.prior = base + 2; dot.count = instruction.safe_k;
        c.dag.push_back(dot);
        arithmetic::Node reduction; reduction.operation = arithmetic::Operation::centered_reduce;
        reduction.equality = arithmetic::Equality::modular_value; reduction.left = base + 3;
        reduction.modulus = modulus; reduction.bits = 8; c.dag.push_back(reduction);
      }
      const auto checked = arithmetic::verify(c);
      if (!checked.admitted()) return reject(RNS8_RANGE_ERROR, "shard_arithmetic_certificate");
      certificates.push_back(checked.sha256); // layout/target execution verdicts remain not-run
    }
    const int root = int(r.ranks[r.owner].ordinal);
    for (const auto& work : plan->work()) if (work.transport == Transport::peer) {
      const auto found = std::find_if(edges.begin(), edges.end(), [&](const Edge& edge) {
        return edge.source == work.rank && edge.destination == r.owner;
      });
      if (found == edges.end() || found->peer != Support::supported)
        return reject(RNS8_UNSUPPORTED_BACKEND, "runtime_peer_access_unavailable", work.rank, work.id);
    }
    devices.resize(r.ranks.size()); phases.resize(plan->work().size(), Phase::expected);
    queued.resize(plan->work().size(), 0); export_staging.resize(size_t(r.m * r.n * 3));
    for (uint32_t rank = 0; rank < devices.size(); ++rank) devices[rank].ordinal = int(r.ranks[rank].ordinal);
    for (const auto& work : plan->work()) devices[work.rank].work.push_back(work.id);
    auto& tiles = tile_records;
    tiles.reserve(plan->tiles().size());
    uint64_t partial_offset=0;
    for (const auto& tile : plan->tiles()) {
      DeviceTile record;record.partial_offset=partial_offset;record.fragments=plan->k_fragments();
      partial_offset+=tile.rows*tile.cols*tile.prefix*record.fragments;
      record.row = tile.row; record.col = tile.col; record.rows = tile.rows; record.cols = tile.cols; record.prefix = tile.prefix;
      Integer magnitude = tile.magnitude;
      for (auto& word : record.bound) { word = static_cast<uint64_t>(magnitude & UINT64_MAX); magnitude >>= 64; }
      tiles.push_back(record);
    }
    for (uint32_t rank = 0; rank < devices.size(); ++rank) {
      auto& device = devices[rank]; hip_scoped_device selected(device.ordinal);
      if (selected.status() != hipSuccess) return error(selected.status(), "device_selection", rank);
      size_t free = 0, total = 0;
      auto code = hipMemGetInfo(&free, &total);
      if (code != hipSuccess) return error(code, "memory_inventory", rank);
      if (plan->resources().ranks[rank].peak > free)
        return reject(RNS8_WORKSPACE_TOO_SMALL, "runtime_free_memory", rank);
      if (device.work.empty() && !(r.native_integrity && rank==r.owner)) continue;
      device.slots.resize(r.pipeline_slots);
      for (auto id : device.work) {
        const auto& tile = plan->tiles()[plan->work()[id].tile];
        device.packed_bytes = std::max(device.packed_bytes, (tile.rows + tile.cols) * (plan->work()[id].k_end-plan->work()[id].k_begin));
      }
      if (!(admission = allocate(device.a, device.ordinal, inputs->lhs().size() * 8)) ||
          !(admission = allocate(device.b, device.ordinal, inputs->rhs().size() * 8)) ||
          !(admission = allocate(device.output, device.ordinal, plan->resources().ranks[rank].local_outputs))) return admission;
      // Preparation is explicitly synchronous; submit reuses these immutable
      // resident native snapshots and never performs a pageable upload.
      // A failed copy/synchronization is not proof that initialization DMA has
      // stopped. Retain the complete owner until its default streams are safe.
      initialization_pending = true; issued = true; pending(true);
      code = hipMemcpy(device.a.get(), inputs->lhs().data(), inputs->lhs().size() * 8, hipMemcpyHostToDevice);
      if (code == hipSuccess) code = hipMemcpy(device.b.get(), inputs->rhs().data(), inputs->rhs().size() * 8, hipMemcpyHostToDevice);
      if (code == hipSuccess) code = hipStreamSynchronize(nullptr);
      if (code != hipSuccess) return error(code, "native_replication", rank);
      pending(false); issued = false; initialization_pending = false;
      if ((code = event_create(device.staged)) != hipSuccess) return error(code, "staging_event", rank);
      for (auto& slot : device.slots) {
        if (!(admission = allocate(slot.packed, device.ordinal, device.packed_bytes))) return admission;
        if ((code = slot.stream.create_non_blocking()) != hipSuccess || (code = event_create(slot.ready)) != hipSuccess)
          return error(code, "rank_stream_event", rank);
      }
    }
    hip_scoped_device selected(root);
    if (selected.status() != hipSuccess) return error(selected.status(), "owner_device");
    for (uint32_t rank = 0; rank < devices.size(); ++rank) if (rank != r.owner && r.transport == Transport::peer && !devices[rank].work.empty()) {
      const auto code = hipDeviceEnablePeerAccess(devices[rank].ordinal, 0);
      if (code == hipErrorPeerAccessAlreadyEnabled) (void)hipGetLastError();
      else if (code != hipSuccess) return error(code, "peer_enable", rank);
      // Peer access is primary-context state; never disable another user's access.
    }
    if (!(admission=allocate(partials,root,plan->resources().ranks[r.owner].partial_gather)) ||
        !(admission = allocate(gather, root, plan->resources().ranks[r.owner].gather)) ||
        !(admission = allocate(result, root, r.m * r.n * 24)) ||
        !(admission = allocate(directory, root, tiles.size() * sizeof(DeviceTile))) ||
        !(admission = allocate(status, root, 4))) return admission;
    initialization_pending = true; issued = true; pending(true);
    auto code = hipMemcpy(directory.get(), tiles.data(), tiles.size() * sizeof(DeviceTile), hipMemcpyHostToDevice);
    if (code == hipSuccess) code = hipStreamSynchronize(nullptr);
    if (code != hipSuccess) return error(code, "directory_upload");
    pending(false); issued = false; initialization_pending = false;
    if (code == hipSuccess) code = gather_stream.create_non_blocking();
    if (code == hipSuccess) code = event_create(completion);
    if (code == hipSuccess) code = event_create(stage_arrived);
    if (code == hipSuccess) code = host_status.ensure(root, 4);
    if (code == hipSuccess && plan->resources().pinned_staging > 4)
      code = staging.ensure(root, size_t(plan->resources().pinned_staging - 4), true);
    return code == hipSuccess ? Admission{} : error(code, "owner_resources");
  }
  Admission issue(uint32_t rank, Slot& slot) {
    auto& device = devices[rank];
    if (device.next == device.work.size()) return {};
    const auto& r = plan->request();
    const auto id = device.work[device.next]; const auto& work = plan->work()[id]; const auto& tile = plan->tiles()[work.tile];
    hip_scoped_device selected(device.ordinal);
    if (selected.status() != hipSuccess) return error(selected.status(), "rank_selection", rank, id);
    slot.work = id; slot.ready_observed = false;
    const uint64_t local_k=work.k_end-work.k_begin;
    auto* a = static_cast<int8_t*>(slot.packed.get()); auto* b = a + tile.rows * local_k;
    auto* c = static_cast<int8_t*>(device.output.get()) + work.byte_offset;
    auto code = hipError_t(rns8_hip_distributed_pack(static_cast<const uint64_t*>(device.a.get()),
        static_cast<const uint64_t*>(device.b.get()), a, b, tile.row, tile.col, tile.rows, tile.cols,
        local_k, r.lda, r.ldb, plan->signed_value(), work.modulus,work.k_begin, slot.stream.get()));
    if (code == hipSuccess) code = hipError_t(rns8_hip_direct_ring_gemm_i8_device_on_stream(a, b, c,
        int(tile.rows), int(tile.cols), int(local_k), int(local_k), int(tile.cols), int(tile.cols), int(work.modulus),
        uint32_t((uint64_t{1} << 32) / work.modulus), int(work.logical_plane), int(r.prefix), 65536, slot.stream.get()));
    if (code == hipSuccess) code = hipEventRecord(slot.ready.get(), slot.stream.get());
    if (code != hipSuccess) return error(code, "shard_enqueue", rank, id);
    ++device.next; return {};
  }
  Admission copy(uint64_t id) {
    const auto& r = plan->request(); const auto& work = plan->work()[id]; const auto& tile = plan->tiles()[work.tile];
    auto& device = devices[work.rank];
    const auto* source = static_cast<const int8_t*>(device.output.get()) + work.byte_offset;
    if (work.transport == Transport::pinned_host) {
      if (staged_work != UINT64_MAX) return reject(RNS8_BUSY, "staging_slot_retained");
      hip_scoped_device selected(device.ordinal);
      if (selected.status() != hipSuccess) return error(selected.status(), "staging_source_device", work.rank, id);
      const auto stream = device.slots[0].stream.get();
      auto code = hipMemcpyAsync(staging.get(), source, size_t(work.bytes), hipMemcpyDeviceToHost, stream);
      if (code == hipSuccess) code = hipEventRecord(device.staged.get(), stream);
      if (code != hipSuccess) return error(code, "staging_source_copy", work.rank, id);
      staged_work = id; stage_phase = 1; phases[id] = Phase::host_staged; return {};
    }
    const int root = int(r.ranks[r.owner].ordinal); hip_scoped_device selected(root);
    if (selected.status() != hipSuccess) return error(selected.status(), "gather_device", work.rank, id);
    auto* destination = partials?static_cast<int8_t*>(partials.get())+work.gather_offset:
        static_cast<int8_t*>(gather.get())+uint64_t(work.logical_plane)*r.m*r.n+tile.row*r.n+tile.col;
    const uint64_t destination_pitch=partials?tile.cols:r.n;
    hipError_t code = hipSuccess;
    if (work.transport == Transport::local)
      code = hipMemcpy2DAsync(destination, size_t(destination_pitch), source, size_t(tile.cols), size_t(tile.cols),
          size_t(tile.rows), hipMemcpyDeviceToDevice, gather_stream.get());
    else for (uint64_t row = 0; row < tile.rows && code == hipSuccess; ++row)
      code = hipMemcpyPeerAsync(destination + row * destination_pitch, root, source + row * tile.cols,
          device.ordinal, size_t(tile.cols), gather_stream.get());
    if (code != hipSuccess) return error(code, "gather_copy", work.rank, id);
    queued[id] = 1; ++progress.transfers_queued; return {};
  }
  Admission advance_staging() {
    if (staged_work == UINT64_MAX) return {};
    const auto& r = plan->request(); const auto& work = plan->work()[staged_work]; const auto& tile = plan->tiles()[work.tile];
    const int root = int(r.ranks[r.owner].ordinal);
    if (stage_phase == 1) {
      hip_scoped_device selected(devices[work.rank].ordinal);
      if (selected.status() != hipSuccess) return error(selected.status(), "staging_event_device", work.rank, work.id);
      auto code = hipEventQuery(devices[work.rank].staged.get());
      if (code == hipErrorNotReady) return {};
      if (code != hipSuccess) return error(code, "staging_source_completion", work.rank, work.id);
      hip_scoped_device owner(root);
      if (owner.status() != hipSuccess) return error(owner.status(), "staging_destination_device", work.rank, work.id);
      auto* destination = partials?static_cast<int8_t*>(partials.get())+work.gather_offset:
        static_cast<int8_t*>(gather.get())+uint64_t(work.logical_plane)*r.m*r.n+tile.row*r.n+tile.col;
    const uint64_t destination_pitch=partials?tile.cols:r.n;
      code = hipMemcpy2DAsync(destination, size_t(destination_pitch), staging.get(), size_t(tile.cols),
          size_t(tile.cols), size_t(tile.rows), hipMemcpyHostToDevice, gather_stream.get());
      if (code == hipSuccess) code = hipEventRecord(stage_arrived.get(), gather_stream.get());
      if (code != hipSuccess) return error(code, "staging_destination_copy", work.rank, work.id);
      queued[work.id] = 1; ++progress.transfers_queued; stage_phase = 2;
    }
    if (stage_phase == 2) {
      hip_scoped_device selected(root);
      if (selected.status() != hipSuccess) return error(selected.status(), "staging_destination_event", work.rank, work.id);
      const auto code = hipEventQuery(stage_arrived.get());
      if (code == hipErrorNotReady) return {};
      if (code != hipSuccess) return error(code, "staging_destination_completion", work.rank, work.id);
      staged_work = UINT64_MAX; stage_phase = 0;
    }
    return {};
  }
  void drain() {
    bool ready = true;
    for (auto& device : devices) {
      if (!device.a && !device.b && !device.output && !device.staged &&
          std::all_of(device.slots.begin(), device.slots.end(), [](const Slot& slot) {
            return !slot.packed && !slot.ready && !slot.stream;
          })) continue;
      hip_scoped_device selected(device.ordinal);
      if (selected.status() != hipSuccess) { progress.state = State::quarantined; return; }
      if (initialization_pending) {
        const auto code = hipStreamQuery(nullptr);
        if (code == hipErrorNotReady) ready = false;
        else if (code != hipSuccess) { progress.state = State::quarantined; return; }
      }
      for (auto& slot : device.slots) {
        if (!slot.stream) continue;
        // Use an explicit terminal event for cancellation too. In particular,
        // Windows HIP command batching must observe a recorded completion
        // point, rather than only querying an otherwise unflushed stream.
        if (!drain_markers_queued && slot.ready &&
            hipEventRecord(slot.ready.get(), slot.stream.get()) != hipSuccess) {
          progress.state = State::quarantined; return;
        }
        const auto code = slot.ready ? hipEventQuery(slot.ready.get()) : hipStreamQuery(slot.stream.get());
        if (code == hipErrorNotReady) ready = false;
        else if (code != hipSuccess) { progress.state = State::quarantined; return; }
      }
    }
    hip_scoped_device selected(int(plan->request().ranks[plan->request().owner].ordinal));
    if (selected.status() != hipSuccess) { progress.state = State::quarantined; return; }
    if (!drain_markers_queued && completion &&
        hipEventRecord(completion.get(), gather_stream.get()) != hipSuccess) {
      progress.state = State::quarantined; return;
    }
    const auto code = completion ? hipEventQuery(completion.get()) : hipStreamQuery(gather_stream.get());
    drain_markers_queued = true;
    if (code == hipErrorNotReady) ready = false;
    else if (code != hipSuccess) { progress.state = State::quarantined; return; }
    if (!ready) return;
    pending(false); issued = false; initialization_pending = false;
    progress.state = progress.operation.status == RNS8_CANCELED ? State::canceled :
        progress.operation.status == RNS8_TIMEOUT ? State::timed_out : State::failed;
  }
  Admission poll() {
    if (progress.state == State::draining || progress.state == State::quarantined) { drain(); return {}; }
    if (progress.state != State::pending) return {};
    if (monotonic_ns() >= deadline) {
      progress.operation = reject(RNS8_TIMEOUT, "operation_deadline"); progress.state = State::draining; drain(); return {};
    }
    if (!advance_staging()) return {};
    for (uint32_t rank = 0; rank < devices.size() && progress.state == State::pending; ++rank) {
      auto& device = devices[rank]; hip_scoped_device selected(device.ordinal);
      if (selected.status() != hipSuccess) { error(selected.status(), "rank_poll", rank); break; }
      for (auto& slot : device.slots) if (slot.work != UINT64_MAX) {
        if (!slot.ready_observed) {
          const auto code = hipEventQuery(slot.ready.get());
          if (code == hipErrorNotReady) continue;
          if (code != hipSuccess) { error(code, "shard_completion", rank, slot.work); break; }
          slot.ready_observed = true; phases[slot.work] = Phase::local_complete; ++progress.local_completed;
        }
        const auto copied = copy(slot.work);
        if (copied.status == RNS8_BUSY) continue;
        if (!copied) break;
        slot.work = UINT64_MAX;
        if (!issue(rank, slot)) break;
      }
    }
    if (progress.state != State::pending) return {};
    const auto& r = plan->request(); hip_scoped_device selected(int(r.ranks[r.owner].ordinal));
    if (selected.status() != hipSuccess) { error(selected.status(), "owner_poll"); return {}; }
    if (!final_queued && progress.transfers_queued == plan->work().size()) {
      auto code=hipSuccess;
      if(partials)code=hipError_t(rns8_hip_distributed_reduce_partials(static_cast<const int8_t*>(partials.get()),static_cast<int8_t*>(gather.get()),
          static_cast<const DeviceTile*>(directory.get()),r.m,r.n,uint32_t(plan->tiles().size()),r.prefix,gather_stream.get()));
      if(code==hipSuccess)code = hipError_t(rns8_hip_distributed_export(static_cast<const int8_t*>(gather.get()),
          static_cast<uint64_t*>(result.get()), int(r.m * r.n), int(r.n), r.tile_rows, r.tile_cols,
          static_cast<const DeviceTile*>(directory.get()), plan->signed_value(), plan->bounded(),
          static_cast<int*>(status.get()), gather_stream.get()));
      if(code==hipSuccess && r.native_integrity) {
        const auto& source=devices[r.owner];
        code=hipError_t(rns8_hip_distributed_integrity(static_cast<const uint64_t*>(source.a.get()),
            static_cast<const uint64_t*>(source.b.get()),static_cast<const uint64_t*>(result.get()),
            r.m,r.n,r.k,r.lda,r.ldb,plan->signed_value(),static_cast<int*>(status.get()),gather_stream.get()));
      }
      if (code == hipSuccess) code = hipMemcpyAsync(host_status.get(), status.get(), 4, hipMemcpyDeviceToHost, gather_stream.get());
      if (code == hipSuccess) code = hipEventRecord(completion.get(), gather_stream.get());
      if (code != hipSuccess) { error(code, "global_reconstruction"); return {}; }
      final_queued = true;
    }
    if (final_queued) {
      const auto code = hipEventQuery(completion.get());
      if (code == hipErrorNotReady) return {};
      if (code != hipSuccess) { error(code, "global_completion"); return {}; }
      // Every copy and reconstruction is before this owner-stream event. Rank
      // source events were observed before each copy; no cross-device clock math.
      pending(false); issued = false; staged_work = UINT64_MAX; stage_phase = 0;
      const uint32_t flags=*static_cast<const uint32_t*>(host_status.get());
      progress.integrity_failure_mask=flags&~1u;
      progress.integrity_completed=r.native_integrity && !(flags&1u);
      const auto committed=distributed_commit_status(flags);
      if (committed!=RNS8_SUCCESS) {
        progress.operation = reject(committed, flags&1u?"global_export_range":"global_integrity_mismatch"); progress.state = State::failed;
      } else {
        std::fill(phases.begin(), phases.end(), Phase::proof_ready);
        progress.published_epoch = progress.epoch; progress.output_current = true; progress.state = State::published;
      }
    }
    return {};
  }
  Admission cleanup() {
    if (issued || readers) return reject(RNS8_BUSY, "distributed_readers_or_work_retained");
    progress.output_current = false; // a partial cleanup cannot leave a borrowable freed result
    // Destroy under each owning device and retain failed handles for retry.
    const auto event = [](hip_unique_event& value) {
      if (!value) return hipSuccess;
      const auto code = hipEventDestroy(value.get()); if (code == hipSuccess) (void)value.release(); return code;
    };
    const auto stream = [](hip_unique_stream& value) {
      if (!value) return hipSuccess;
      const auto code = hipStreamDestroy(value.get()); if (code == hipSuccess) (void)value.release(); return code;
    };
    for (auto& device : devices) {
      if (!device.a && !device.b && !device.output && !device.staged &&
          std::all_of(device.slots.begin(), device.slots.end(), [](const Slot& slot) {
            return !slot.packed && !slot.ready && !slot.stream;
          })) continue;
      hip_scoped_device selected(device.ordinal);
      if (selected.status() != hipSuccess) return reject(RNS8_BACKEND_FAILURE, "cleanup_device");
      for (auto& slot : device.slots) {
        if (event(slot.ready) != hipSuccess || stream(slot.stream) != hipSuccess)
          return reject(RNS8_BACKEND_FAILURE, "cleanup_rank_event_stream");
        if (slot.packed.reset() != RNS8_SUCCESS) return reject(RNS8_BACKEND_FAILURE, "cleanup_packed_storage");
      }
      if (event(device.staged) != hipSuccess) return reject(RNS8_BACKEND_FAILURE, "cleanup_staging_event");
      for (auto* buffer : {&device.a, &device.b, &device.output})
        if (buffer->reset() != RNS8_SUCCESS) return reject(RNS8_BACKEND_FAILURE, "cleanup_rank_storage");
    }
    if (!gather && !partials && !result && !directory && !status && !uniform && !uniform_pending &&
        !gather_stream && !completion && !stage_arrived && !staging.get() && !host_status.get()) {
      closed = true; return {};
    }
    hip_scoped_device selected(int(plan->request().ranks[plan->request().owner].ordinal));
    if (selected.status() != hipSuccess) return reject(RNS8_BACKEND_FAILURE, "cleanup_owner_device");
    if (event(completion) != hipSuccess || event(stage_arrived) != hipSuccess || stream(gather_stream) != hipSuccess)
      return reject(RNS8_BACKEND_FAILURE, "cleanup_owner_event_stream");
    if (staging.reset() != hipSuccess || host_status.reset() != hipSuccess)
      return reject(RNS8_BACKEND_FAILURE, "cleanup_host_storage");
    for (auto* buffer : {&gather, &partials, &result, &directory, &status, &uniform, &uniform_pending})
      if (buffer->reset() != RNS8_SUCCESS) return reject(RNS8_BACKEND_FAILURE, "cleanup_owner_storage");
    closed = true; return {};
  }
#endif
};

bool HipExecution::compiled() noexcept {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  return true;
#else
  return false;
#endif
}
Admission HipExecution::discover(const std::vector<uint32_t>& ordinals, std::vector<Rank>& ranks, std::vector<Edge>& edges) {
  if (ordinals.empty() || ordinals.size() > 64) return reject(RNS8_INVALID_ARGUMENT, "selected_device_count");
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  try {
    std::vector<Rank> discovered; std::vector<Edge> connections;
    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess || !count) return reject(RNS8_UNSUPPORTED_BACKEND, "no_hip_devices");
    for (auto ordinal : ordinals) {
      if (ordinal >= uint32_t(count) || std::count(ordinals.begin(), ordinals.end(), ordinal) != 1)
        return reject(RNS8_INVALID_ARGUMENT, "duplicate_or_missing_device");
      hipDeviceProp_t prop{}; char bdf[64]{};
      if (hipGetDeviceProperties(&prop, int(ordinal)) != hipSuccess ||
          hipDeviceGetPCIBusId(bdf, sizeof(bdf), int(ordinal)) != hipSuccess)
        return reject(RNS8_BACKEND_FAILURE, "device_identity_query");
      Rank device; device.ordinal = ordinal; device.physical_id = "uuid=";
      static constexpr char hex[] = "0123456789abcdef";
      for (unsigned char byte : prop.uuid.bytes) {
        device.physical_id += hex[byte >> 4]; device.physical_id += hex[byte & 15];
      }
      device.physical_id += ";bdf="; device.physical_id += bdf;
      device.physical_id += ";visible=" + std::to_string(ordinal);
      device.target = prop.gcnArchName;
      device.memory_budget = prop.totalGlobalMem; discovered.push_back(std::move(device));
    }
    for (uint32_t source = 0; source < ordinals.size(); ++source)
      for (uint32_t destination = 0; destination < ordinals.size(); ++destination) if (source != destination) {
        int forward = 0, reverse = 0;
        const auto a = hipDeviceCanAccessPeer(&forward, int(ordinals[source]), int(ordinals[destination]));
        const auto b = hipDeviceCanAccessPeer(&reverse, int(ordinals[destination]), int(ordinals[source]));
        connections.push_back({source, destination,
            a != hipSuccess || b != hipSuccess ? Support::unknown : forward && reverse ? Support::supported : Support::unsupported,
            Support::supported}); // portable pinned API path; actual allocation is checked at preparation
      }
    ranks.swap(discovered); edges.swap(connections); return {};
  } catch (const std::bad_alloc&) { return reject(RNS8_INTERNAL_ERROR, "discovery_allocation"); }
#else
  (void)ranks; (void)edges; return reject(RNS8_UNSUPPORTED_BACKEND, "hip_not_compiled");
#endif
}
HipExecution::HipExecution(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
const std::vector<std::string>& HipExecution::arithmetic_certificates() const { return impl_->certificates; }
HipExecution::~HipExecution() {
  if (impl_ && !close()) {
    // Allocation-free quarantine: the preexisting owner is its own queue node.
    // Events, input snapshots and both ends of DMA remain owned together.
    auto* item = impl_.release(); auto* head = Impl::quarantined.load();
    do { item->quarantine_next = head; } while (!Impl::quarantined.compare_exchange_weak(head, item));
  }
}
Admission HipExecution::create(std::shared_ptr<const Plan> plan, std::shared_ptr<const Inputs> inputs,
                               std::unique_ptr<HipExecution>& output) {
  if (!plan || !inputs) return reject(RNS8_INVALID_ARGUMENT, "plan_and_input_required");
  if (output) return reject(RNS8_BUSY, "output_owner_already_present");
  if (!compiled()) return reject(RNS8_UNSUPPORTED_BACKEND, "hip_not_compiled");
  try {
    const auto proof = validate_inputs(*plan, *inputs); if (!proof) return proof;
    auto state = std::make_unique<Impl>(); state->plan = std::move(plan); state->inputs = std::move(inputs);
    auto execution = std::unique_ptr<HipExecution>(new HipExecution(std::move(state)));
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
    const auto prepared = execution->impl_->initialize(); if (!prepared) return prepared;
#endif
    output = std::move(execution); return {};
  } catch (const std::bad_alloc&) { return reject(RNS8_INTERNAL_ERROR, "preparation_allocation"); }
}
Admission HipExecution::submit(uint64_t epoch, uint64_t timeout_ns) {
  auto& p = *impl_;
  if (p.closed || !epoch || epoch == UINT64_MAX || epoch <= p.progress.epoch || !timeout_ns)
    return reject(RNS8_INVALID_ARGUMENT, "fresh_epoch_and_deadline_required");
  if (p.issued || p.readers || p.progress.state == State::quarantined) return reject(RNS8_BUSY, "previous_epoch_retained");
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  const auto& r = p.plan->request(); hip_scoped_device selected(int(r.ranks[r.owner].ordinal));
  if (selected.status() != hipSuccess) return reject(RNS8_BACKEND_FAILURE, "submit_owner_preflight");
  p.progress = {}; p.progress.epoch = epoch; p.progress.expected = p.plan->work().size();
  p.progress.integrity_requested=r.native_integrity;
  p.progress.state = State::pending; p.deadline = deadline_after(timeout_ns);
  p.final_queued = false; p.staged_work = UINT64_MAX; p.stage_phase = 0;
  p.drain_markers_queued = false;
  p.uniform_prefix = 0;
  std::fill(p.phases.begin(), p.phases.end(), Phase::expected); std::fill(p.queued.begin(), p.queued.end(), 0);
  for (auto& device : p.devices) { device.next = 0; for (auto& slot : device.slots) slot.work = UINT64_MAX; }
  // Ownership becomes pending before the first possible write, including a
  // failed enqueue. The accepted epoch remains observable on partial failure.
  p.pending(true); p.issued = true;
  // Selected planes are completely overwritten before CRT; zero-prefix tiles
  // read no residue bytes. Missing coverage must never be masked by zero fill.
  auto code = hipMemsetAsync(p.status.get(), 0, 4, p.gather_stream.get());
  if (code != hipSuccess) return p.error(code, "epoch_initialization");
  for (uint32_t rank = 0; rank < p.devices.size(); ++rank)
    for (auto& slot : p.devices[rank].slots) { const auto issued = p.issue(rank, slot); if (!issued) return issued; }
  return {};
#else
  return reject(RNS8_UNSUPPORTED_BACKEND, "hip_not_compiled");
#endif
}
Admission HipExecution::query(uint64_t epoch, HipProgress& output) {
  auto& p = *impl_;
  if (p.closed || !epoch || epoch != p.progress.epoch) return reject(RNS8_INVALID_ARGUMENT, "foreign_or_stale_epoch");
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  const auto polled = p.poll(); if (!polled) return polled;
#endif
  output = p.progress; return {};
}
Admission HipExecution::wait(uint64_t epoch, uint64_t timeout_ns, HipProgress& output) {
  if (!timeout_ns) return reject(RNS8_INVALID_ARGUMENT, "positive_observation_timeout");
  const auto end = deadline_after(timeout_ns);
  do {
    const auto polled = query(epoch, output); if (!polled) return polled;
    if (output.state != State::pending && output.state != State::draining) return {};
    if (monotonic_ns() >= end) return reject(RNS8_TIMEOUT, "observation_timeout_reservations_retained");
    std::this_thread::sleep_for(std::chrono::microseconds(50));
  } while (true);
}
Admission HipExecution::cancel(uint64_t epoch) {
  auto& p = *impl_;
  if (!epoch || epoch != p.progress.epoch) return reject(RNS8_INVALID_ARGUMENT, "foreign_or_stale_epoch");
  if (p.progress.state != State::pending && p.progress.state != State::draining) return {};
  if (p.progress.operation) p.progress.operation = reject(RNS8_CANCELED, "caller_canceled");
  p.progress.state = State::draining; p.progress.output_current = false; return {};
}
Admission HipExecution::retain(uint64_t epoch) {
  auto& p = *impl_;
  if (p.closed || !epoch || epoch != p.progress.published_epoch || !p.progress.output_current || p.readers == UINT64_MAX)
    return reject(RNS8_INVALID_ARGUMENT, "published_epoch_required");
  ++p.readers; return {};
}
Admission HipExecution::release(uint64_t epoch) {
  auto& p = *impl_;
  if (!epoch || epoch != p.progress.published_epoch || !p.readers) return reject(RNS8_INVALID_ARGUMENT, "retained_epoch_required");
  --p.readers; return {};
}
Admission HipExecution::resident_limbs(uint64_t epoch, int& device, const void*& pointer) const {
  const auto& p = *impl_;
  if (p.closed || !epoch || epoch != p.progress.published_epoch || !p.progress.output_current || !p.readers)
    return reject(RNS8_BUSY, "retained_published_result_required");
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  device = int(p.plan->request().ranks[p.plan->request().owner].ordinal); pointer = p.result.get(); return {};
#else
  (void)device; (void)pointer; return reject(RNS8_UNSUPPORTED_BACKEND, "hip_not_compiled");
#endif
}
Admission HipExecution::export_limbs(uint64_t epoch, uint64_t* destination, uint64_t ld) {
  auto& p = *impl_; const auto& r = p.plan->request();
  AddressDomain span;
  if (!destination || ld < r.n || !checked_host_plane_address(r.m, r.n, ld, 1, 24, span))
    return reject(RNS8_INVALID_ARGUMENT, "destination_address");
  if (!nonoverlapping_byte_ranges(destination, span.byte_extent, p.inputs->lhs().data(), p.inputs->lhs().size() * 8) ||
      !nonoverlapping_byte_ranges(destination, span.byte_extent, p.inputs->rhs().data(), p.inputs->rhs().size() * 8))
    return reject(RNS8_INVALID_ARGUMENT, "destination_aliases_native_snapshot");
  if (p.closed || !epoch || epoch != p.progress.published_epoch || !p.progress.output_current)
    return reject(RNS8_BUSY, "complete_published_result_required");
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  hip_scoped_device selected(int(r.ranks[r.owner].ordinal));
  if (selected.status() != hipSuccess) return reject(RNS8_BACKEND_FAILURE, "export_device");
  if (hipMemcpy(p.export_staging.data(), p.result.get(), p.export_staging.size() * 8, hipMemcpyDeviceToHost) != hipSuccess)
    return reject(RNS8_BACKEND_FAILURE, "export_copy_preserved_caller");
  for (uint64_t row = 0; row < r.m; ++row)
    std::copy_n(p.export_staging.data() + row * r.n * 3, size_t(r.n * 3), destination + row * ld * 3);
  return {};
#else
  return reject(RNS8_UNSUPPORTED_BACKEND, "hip_not_compiled");
#endif
}
Admission HipExecution::materialize(uint64_t epoch, uint32_t prefix, uint64_t max_bytes) {
  auto& p = *impl_; const auto& r = p.plan->request();
  if (p.closed || !epoch || epoch != p.progress.published_epoch || !p.progress.output_current || p.readers)
    return reject(RNS8_BUSY, "unborrowed_published_result_required");
  if (!prefix || prefix > 20) return reject(RNS8_INVALID_ARGUMENT, "materialization_prefix");
  try {
    Integer product = 1;
    for (uint32_t plane = 0; plane < prefix; ++plane) product *= garner::kConstants.moduli[plane];
    for (const auto& tile : p.plan->tiles())
      if (product <= tile.magnitude * (p.plan->signed_value() ? 2 : 1))
        return reject(RNS8_RANGE_ERROR, "materialization_basis_insufficient");
  } catch (const std::bad_alloc&) { return reject(RNS8_INTERNAL_ERROR, "materialization_proof_allocation"); }
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  const uint64_t bytes = r.m * r.n * prefix;
  if (bytes > max_bytes) return reject(RNS8_WORKSPACE_TOO_SMALL, "materialization_budget");
  if (p.uniform_prefix == prefix) return {};
  uint64_t reserved = p.plan->resources().ranks[r.owner].peak;
  if (!add_bytes(reserved, p.uniform.bytes()) || !add_bytes(reserved, bytes) || reserved > r.ranks[r.owner].memory_budget)
    return reject(RNS8_WORKSPACE_TOO_SMALL, "owner_materialization_peak");
  if (p.uniform_pending.reset() != RNS8_SUCCESS) return reject(RNS8_BACKEND_FAILURE, "previous_materialization_cleanup");
  auto admitted = p.allocate(p.uniform_pending, int(r.ranks[r.owner].ordinal), bytes); if (!admitted) return admitted;
  hip_scoped_device selected(int(r.ranks[r.owner].ordinal));
  if (selected.status() != hipSuccess) return reject(RNS8_BACKEND_FAILURE, "materialization_device");
  p.pending(true); p.issued = true;
  auto code = hipError_t(rns8_hip_distributed_materialize(static_cast<const uint64_t*>(p.result.get()),
      static_cast<int8_t*>(p.uniform_pending.get()), int(r.m * r.n), prefix, p.plan->signed_value(), p.gather_stream.get()));
  const auto drained = hipStreamSynchronize(p.gather_stream.get());
  if (drained != hipSuccess) { p.error(code != hipSuccess ? code : drained, "materialization_failed_drain"); return p.progress.operation; }
  p.pending(false); p.issued = false;
  if (code != hipSuccess) return reject(RNS8_BACKEND_FAILURE, "materialization_launch");
  if (p.uniform.reset() != RNS8_SUCCESS) return reject(RNS8_BACKEND_FAILURE, "materialization_replacement_free");
  p.uniform = std::move(p.uniform_pending); p.uniform_prefix = prefix; return {};
#else
  (void)max_bytes; return reject(RNS8_UNSUPPORTED_BACKEND, "hip_not_compiled");
#endif
}
Admission HipExecution::resident_residues(uint64_t epoch, int& device, const void*& pointer, uint32_t& prefix) const {
  const auto& p = *impl_;
  if (p.closed || !epoch || epoch != p.progress.published_epoch || !p.progress.output_current || !p.readers)
    return reject(RNS8_BUSY, "retained_published_result_required");
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if (!p.uniform_prefix) return reject(RNS8_INVALID_ARGUMENT, "uniform_materialization_required");
  device = int(p.plan->request().ranks[p.plan->request().owner].ordinal); pointer = p.uniform.get(); prefix = p.uniform_prefix;
  return {};
#else
  (void)device; (void)pointer; (void)prefix; return reject(RNS8_UNSUPPORTED_BACKEND, "hip_not_compiled");
#endif
}
Admission HipExecution::close() {
  if (!impl_ || impl_->closed) return {};
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  return impl_->cleanup();
#else
  impl_->closed = true; return {};
#endif
}
void HipExecution::reap_quarantined() noexcept {
  auto* pending = Impl::quarantined.exchange(nullptr);
  while (pending) {
    auto* next = pending->quarantine_next;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
    if (pending->issued) pending->drain();
    if (pending->cleanup()) { delete pending; pending = next; continue; }
#endif
    auto* head = Impl::quarantined.load();
    do { pending->quarantine_next = head; } while (!Impl::quarantined.compare_exchange_weak(head, pending));
    pending = next;
  }
}
} // namespace rns8::detail::distributed
