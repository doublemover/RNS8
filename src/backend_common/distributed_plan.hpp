#pragma once

// Shared hardware-independent contract for the CPU mock and real HIP owner.
// Runtime discovery, transport and target qualification remain with the owner.
#include "address_domain.hpp"
#include "garner_constants.hpp"
#include "rns8/rns8.h"
#include <boost/multiprecision/cpp_int.hpp>
#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace rns8::detail::distributed {
using Integer = boost::multiprecision::cpp_int;
enum class Axis { modulus, rows, columns, contraction, hybrid };
enum class Transport { local, peer, pinned_host };
enum class Support { unknown, unsupported, supported };
enum class HostStorage { exact_reference, limbs_only };
enum class Phase { expected, local_complete, host_staged, transfer_complete, visible, proof_ready };
enum class State { idle, pending, published, draining, failed, canceled, timed_out, quarantined };
struct Admission {
  rns8_status status = RNS8_SUCCESS;
  const char* reason = "admitted";
  uint32_t rank = UINT32_MAX;
  uint64_t work = UINT64_MAX;
  explicit operator bool() const { return status == RNS8_SUCCESS; }
};
inline Admission reject(rns8_status status, const char* reason,
                        uint32_t rank = UINT32_MAX, uint64_t work = UINT64_MAX) {
  return {status, reason, rank, work};
}
inline bool add_bytes(uint64_t& value, uint64_t increment) {
  if (increment > UINT64_MAX - value) return false;
  value += increment; return true;
}
inline bool multiply(uint64_t a, uint64_t b, uint64_t& result) {
  if (b && a > UINT64_MAX / b) return false;
  result = a * b; return true;
}
inline bool aligned_bytes(uint64_t bytes, uint64_t& result) {
  if (bytes > UINT64_MAX - 255) return false;
  result = (bytes + 255) & ~uint64_t{255}; return true;
}
inline bool digest(const std::string& text) {
  return text.size() == 64 && std::all_of(text.begin(), text.end(), [](char c) {
    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
  });
}
inline bool ascii_label(const std::string& text) {
  return std::all_of(text.begin(), text.end(), [](unsigned char c) { return c < 128; });
}
struct ValueIdentity {
  uint64_t instance = 0, generation = 0;
  bool operator==(const ValueIdentity& other) const {
    return instance == other.instance && generation == other.generation;
  }
  bool valid() const { return instance && generation && generation != UINT64_MAX; }
};
struct Rank {
  uint32_t ordinal = 0;
  std::string physical_id; // explicit selected identity; never inferred from rank
  std::string target;
  std::string physical_package_id = "unknown"; // partitions may share a package
  std::string compute_partition = "unknown", memory_partition = "unknown";
  uint32_t weight = 1; // caller supplied relative plane cost; not a measured rate
  uint64_t memory_budget = 0, scratch_bytes = 0, overhead_reserve = 4096;
};
struct Edge {
  uint32_t source = 0, destination = 0;
  Support peer = Support::unknown, pinned_host = Support::unknown;
};
struct TileProof {
  uint32_t prefix = 0;
  Integer magnitude = 0;
};
struct Request {
  uint64_t operation = 0;
  Axis axis = Axis::modulus;
  rns8_semantics semantics = RNS8_EXACT_WIDE_SIGNED;
  uint64_t m = 0, n = 0, k = 0, lda = 0, ldb = 0;
  uint32_t prefix = 0, tile_rows = 16, tile_cols = 16, owner = 0;
  ValueIdentity a, b;
  std::string certificate_sha256, layout_sha256;
  Integer magnitude_bound = 0;
  std::vector<TileProof> tile_proofs; // empty means uniform prefix/global bound
  std::vector<Rank> ranks;
  std::vector<Edge> edges;
  std::vector<uint32_t> plane_costs; // optional named-cohort relative costs
  std::string cost_profile_sha256;
  Transport transport = Transport::pinned_host; // explicit; no automatic fallback
  uint64_t host_budget = 0, pinned_budget = 0, host_overhead_reserve = 1u << 20;
  uint64_t maximum_work_items = 1u << 20;
  uint32_t pipeline_slots = 2; // bounded per-rank pack/compute windows
  HostStorage host_storage = HostStorage::exact_reference;
  bool native_integrity = false; // independent original-input checks before publication
};
struct Tile {
  uint64_t row = 0, col = 0, rows = 0, cols = 0;
  uint32_t prefix = 0;
  Integer magnitude = 0;
};
struct Work {
  uint64_t id = 0, tile = 0;
  uint32_t rank = 0, logical_plane = 0, modulus = 0;
  uint64_t k_begin = 0, k_end = 0, byte_offset = 0, bytes = 0;
  Transport transport = Transport::local;
  bool operator==(const Work& rhs) const {
    return id == rhs.id && tile == rhs.tile && rank == rhs.rank &&
        logical_plane == rhs.logical_plane && modulus == rhs.modulus &&
        k_begin == rhs.k_begin && k_end == rhs.k_end && byte_offset == rhs.byte_offset &&
        bytes == rhs.bytes && transport == rhs.transport;
  }
};
struct RankResources {
  uint64_t native_inputs = 0, packed_inputs = 0, local_outputs = 0;
  uint64_t gather = 0, crt_workspace = 0, export_output = 0;
  uint64_t descriptors = 0, scratch = 0, overhead_reserve = 0, peak = 0;
};
struct Resources {
  std::vector<RankResources> ranks;
  uint64_t host_inputs = 0, host_model = 0, host_overhead_reserve = 0;
  uint64_t pinned_staging = 0, transfer_bytes = 0, descriptor_bytes = 0;
  uint64_t contract_metadata = 0;
  // These are checked buffer reservations, not a measured allocator/RCCL peak.
  bool allocator_overhead_known = false;
};

class Plan {
 public:
  const Request& request() const { return request_; }
  const std::vector<Tile>& tiles() const { return tiles_; }
  const std::vector<Work>& work() const { return work_; }
  const Resources& resources() const { return resources_; }
  const std::string& identity() const { return identity_; }
  bool signed_value() const {
    return request_.semantics == RNS8_BOUNDED_I64 || request_.semantics == RNS8_EXACT_WIDE_SIGNED;
  }
  bool bounded() const {
    return request_.semantics == RNS8_BOUNDED_I64 || request_.semantics == RNS8_BOUNDED_U64;
  }
  // Construction stages everything and preserves a prior plan on rejection.
  static Admission create(const Request& request, std::shared_ptr<const Plan>& output) {
    if (request.ranks.size() > 64 || request.edges.size() > 64 * 63 ||
        request.tile_proofs.size() > (1u << 20) || request.plane_costs.size() > 20)
      return reject(RNS8_INVALID_ARGUMENT, "request_descriptor_limit");
    try {
      auto plan = std::shared_ptr<Plan>(new Plan(request));
      const auto result = plan->build();
      if (result) { plan->seal_identity(); output = std::move(plan); }
      return result;
    } catch (const std::bad_alloc&) { return reject(RNS8_INTERNAL_ERROR, "allocation_failed"); }
  }
 private:
  explicit Plan(const Request& request) : request_(request) {}
  Request request_;
  std::vector<Tile> tiles_;
  std::vector<Work> work_;
  Resources resources_;
  std::string identity_;
  void seal_identity() {
    // An exact structural key, not a truncated hash or an arithmetic proof.
    std::ostringstream key;
    const auto& r = request_;
    const auto text = [&](const std::string& value) { key << value.size() << ':' << value << ';'; };
    key << "distributed_modulus_v1;" << r.operation << ';' << unsigned(r.semantics) << ';'
        << r.m << ';' << r.n << ';' << r.k << ';' << r.lda << ';' << r.ldb << ';'
        << r.prefix << ';' << r.owner << ';' << r.a.instance << ';' << r.a.generation << ';'
        << r.b.instance << ';' << r.b.generation << ';' << unsigned(r.transport) << ';'
        << r.pipeline_slots << ';' << unsigned(r.host_storage) << ';' << r.native_integrity << ';';
    text(r.certificate_sha256); text(r.layout_sha256); text(r.cost_profile_sha256);
    for (const auto& rank : r.ranks) {
      key << rank.ordinal << ';' << rank.weight << ';';
      text(rank.physical_id); text(rank.target); text(rank.physical_package_id);
      text(rank.compute_partition); text(rank.memory_partition);
    }
    for (const auto& tile : tiles_)
      key << tile.row << ';' << tile.col << ';' << tile.rows << ';' << tile.cols << ';'
          << tile.prefix << ';' << tile.magnitude << ';';
    for (const auto& work : work_)
      key << work.id << ';' << work.rank << ';' << work.tile << ';' << work.logical_plane << ';'
          << work.modulus << ';' << work.k_begin << ';' << work.k_end << ';' << work.byte_offset << ';'
          << work.bytes << ';' << unsigned(work.transport) << ';';
    identity_ = key.str();
  }
  Admission build() {
    const auto& r = request_;
    if (r.axis != Axis::modulus) return reject(RNS8_UNSUPPORTED_BACKEND, "only_modulus_split_implemented");
    if (r.semantics != RNS8_BOUNDED_I64 && r.semantics != RNS8_BOUNDED_U64 &&
        r.semantics != RNS8_EXACT_WIDE_SIGNED && r.semantics != RNS8_EXACT_WIDE_UNSIGNED)
      return reject(RNS8_UNSUPPORTED_BACKEND, "integer_semantics_required");
    if (!r.operation || !r.a.valid() || !r.b.valid() || !digest(r.certificate_sha256) ||
        !digest(r.layout_sha256) || !r.m || !r.n || !r.k || r.lda < r.k || r.ldb < r.n ||
        !r.prefix || r.prefix > 20 || !r.tile_rows || !r.tile_cols || r.tile_rows > 256 ||
        r.tile_cols > 256 || r.ranks.empty() || r.ranks.size() > 64 || r.owner >= r.ranks.size() ||
        !r.maximum_work_items || r.maximum_work_items > (1u << 20) || r.magnitude_bound < 0 ||
        !r.pipeline_slots || r.pipeline_slots > 3 || unsigned(r.host_storage) > 1)
      return reject(RNS8_INVALID_ARGUMENT, "malformed_contract");
    if (r.magnitude_bound >= (Integer(1) << 160)) return reject(RNS8_RANGE_ERROR, "proof_width_limit");
    if (r.transport != Transport::peer && r.transport != Transport::pinned_host &&
        !(r.transport == Transport::local && r.ranks.size() == 1))
      return reject(RNS8_INVALID_ARGUMENT, "explicit_transport_required");
    AddressDomain a_span, b_span, c_span;
    if (!checked_host_plane_address(r.m, r.k, r.lda, 1, 8, a_span) ||
        !checked_host_plane_address(r.k, r.n, r.ldb, 1, 8, b_span) ||
        !checked_host_plane_address(r.m, r.n, r.n, r.prefix, 1, c_span))
      return reject(RNS8_RANGE_ERROR, "address_span_overflow");
    const uint64_t grid_rows = 1 + (r.m - 1) / r.tile_rows;
    const uint64_t grid_cols = 1 + (r.n - 1) / r.tile_cols;
    uint64_t tile_count = 0, max_work = 0;
    if (!multiply(grid_rows, grid_cols, tile_count) || !multiply(tile_count, r.prefix, max_work) ||
        max_work > r.maximum_work_items || tile_count > SIZE_MAX)
      return reject(RNS8_RANGE_ERROR, "coverage_descriptor_limit");
    if (!r.tile_proofs.empty() && r.tile_proofs.size() != tile_count)
      return reject(RNS8_INVALID_ARGUMENT, "tile_proof_count");
    if ((!r.cost_profile_sha256.empty() && !digest(r.cost_profile_sha256)) ||
        (!r.plane_costs.empty() && (r.plane_costs.size() != r.prefix || !digest(r.cost_profile_sha256))) ||
        std::any_of(r.plane_costs.begin(), r.plane_costs.end(), [](uint32_t cost) { return !cost; }))
      return reject(RNS8_INVALID_ARGUMENT, "plane_cost_profile");
    for (uint32_t rank = 0; rank < r.ranks.size(); ++rank) {
      const auto& item = r.ranks[rank];
      if (!item.weight || item.weight > 65535 || item.physical_id.empty() ||
          item.physical_id.size() > 128 || item.target.empty() || item.target.size() > 64 ||
          item.physical_package_id.size() > 128 || item.compute_partition.size() > 64 || item.memory_partition.size() > 64 ||
          !ascii_label(item.physical_id) || !ascii_label(item.target) || !ascii_label(item.physical_package_id) ||
          !ascii_label(item.compute_partition) || !ascii_label(item.memory_partition))
        return reject(RNS8_INVALID_ARGUMENT, "rank_identity_or_weight", rank);
      for (uint32_t prior = 0; prior < rank; ++prior)
        if (item.ordinal == r.ranks[prior].ordinal || item.physical_id == r.ranks[prior].physical_id)
          return reject(RNS8_INVALID_ARGUMENT, "duplicate_selected_device", rank);
    }
    for (std::size_t i = 0; i < r.edges.size(); ++i) {
      const auto& e = r.edges[i];
      if (e.source >= r.ranks.size() || e.destination >= r.ranks.size() || e.source == e.destination ||
          static_cast<unsigned>(e.peer) > 2 || static_cast<unsigned>(e.pinned_host) > 2)
        return reject(RNS8_INVALID_ARGUMENT, "malformed_transport_edge");
      for (std::size_t prior = 0; prior < i; ++prior)
        if (r.edges[prior].source == e.source && r.edges[prior].destination == e.destination)
          return reject(RNS8_INVALID_ARGUMENT, "duplicate_transport_edge");
    }
    resources_.ranks.resize(r.ranks.size());
    std::vector<Integer> load(r.ranks.size(), 0);
    std::vector<uint64_t> counts(r.ranks.size(), 0);
    for (uint64_t row = 0; row < grid_rows; ++row) {
      for (uint64_t col = 0; col < grid_cols; ++col) {
        const uint64_t id = row * grid_cols + col;
        const auto proof = r.tile_proofs.empty() ? TileProof{r.prefix, r.magnitude_bound} : r.tile_proofs[id];
        if (proof.prefix > r.prefix || (proof.prefix == 0 && proof.magnitude != 0) ||
            proof.magnitude < 0 || proof.magnitude > r.magnitude_bound)
          return reject(RNS8_INVALID_ARGUMENT, "tile_proof_contract");
        Integer product = 1;
        for (uint32_t plane = 0; plane < proof.prefix; ++plane) product *= garner::kConstants.moduli[plane];
        if (product <= (signed_value() ? Integer(2 * proof.magnitude) : proof.magnitude))
          return reject(RNS8_RANGE_ERROR, "insufficient_global_basis");
        if (bounded() && proof.magnitude > (signed_value() ? Integer(Integer(1) << 63) : Integer(UINT64_MAX)))
          return reject(RNS8_RANGE_ERROR, "bounded_export_range");
        Tile tile{row * r.tile_rows, col * r.tile_cols,
            std::min<uint64_t>(r.tile_rows, r.m - row * r.tile_rows),
            std::min<uint64_t>(r.tile_cols, r.n - col * r.tile_cols), proof.prefix, proof.magnitude};
        tiles_.push_back(tile);
        const uint64_t cells = tile.rows * tile.cols;
        for (uint32_t plane = 0; plane < tile.prefix; ++plane) {
          uint32_t selected = 0;
          // Compare rational normalized loads exactly, stable lower-rank tie.
          // Arbitrary precision keeps weighted comparisons exact for long K.
          for (uint32_t rank = 1; rank < r.ranks.size(); ++rank)
            if (load[rank] * r.ranks[selected].weight < load[selected] * r.ranks[rank].weight) selected = rank;
          load[selected] += Integer(cells) * r.k * (r.plane_costs.empty() ? 1 : r.plane_costs[plane]);
          ++counts[selected];
          auto& memory = resources_.ranks[selected];
          Transport transport = selected == r.owner ? Transport::local : r.transport;
          if (transport != Transport::local) {
            const auto edge = std::find_if(r.edges.begin(), r.edges.end(), [&](const Edge& e) {
              return e.source == selected && e.destination == r.owner;
            });
            const Support support = edge == r.edges.end() ? Support::unknown :
                (transport == Transport::peer ? edge->peer : edge->pinned_host);
            if (support != Support::supported)
              return reject(RNS8_UNSUPPORTED_BACKEND, support == Support::unknown ?
                  "transport_capability_unknown" : "transport_unsupported", selected);
          }
          const uint64_t offset = memory.local_outputs;
          uint64_t packed_a = 0, packed_b = 0;
          if (!multiply(tile.rows, r.k, packed_a) || !multiply(r.k, tile.cols, packed_b) ||
              !add_bytes(packed_a, packed_b) ||
              !add_bytes(memory.local_outputs, cells))
            return reject(RNS8_RANGE_ERROR, "rank_buffer_overflow", selected);
          memory.packed_inputs = std::max(memory.packed_inputs, packed_a);
          work_.push_back({work_.size(), id, selected, plane, garner::kConstants.moduli[plane],
              0, r.k, offset, cells, transport});
          if (transport != Transport::local) {
            if (!add_bytes(resources_.transfer_bytes, cells)) return reject(RNS8_RANGE_ERROR, "transfer_overflow");
            if (transport == Transport::pinned_host) resources_.pinned_staging =
                std::max(resources_.pinned_staging, cells); // one explicitly serialized staging slot
          }
        }
      }
    }
    uint64_t native = a_span.byte_extent;
    if (!add_bytes(native, b_span.byte_extent)) return reject(RNS8_RANGE_ERROR, "native_input_overflow");
    uint64_t cells = 0;
    if (!multiply(r.m, r.n, cells)) return reject(RNS8_RANGE_ERROR, "output_overflow");
    for (uint32_t rank = 0; rank < r.ranks.size(); ++rank) {
      auto& memory = resources_.ranks[rank];
      if (!multiply(memory.packed_inputs, r.pipeline_slots, memory.packed_inputs))
        return reject(RNS8_RANGE_ERROR, "pipeline_buffer_overflow", rank);
      memory.native_inputs = counts[rank] || (r.native_integrity && rank==r.owner) ? native : 0;
      memory.scratch = counts[rank] ? r.ranks[rank].scratch_bytes : 0;
      memory.overhead_reserve = r.ranks[rank].overhead_reserve;
      if (!multiply(counts[rank], sizeof(Work) + 128, memory.descriptors))
        return reject(RNS8_RANGE_ERROR, "descriptor_overflow", rank);
      if (rank == r.owner) {
        memory.gather = c_span.byte_extent;
        uint64_t tile_metadata = 0;
        if (!multiply(tile_count, 64, tile_metadata) || !add_bytes(tile_metadata, 4) ||
            !add_bytes(memory.descriptors, tile_metadata))
          return reject(RNS8_RANGE_ERROR, "owner_descriptor_overflow", rank);
        if (!multiply(cells, 24, memory.crt_workspace) ||
            !multiply(cells, bounded() ? 8 : 24, memory.export_output))
          return reject(RNS8_RANGE_ERROR, "owner_output_overflow", rank);
      }
      for (auto bytes : {memory.native_inputs, memory.packed_inputs, memory.local_outputs,
          memory.gather, memory.crt_workspace, memory.export_output, memory.descriptors,
          memory.scratch, memory.overhead_reserve}) {
        uint64_t aligned = 0;
        if (!aligned_bytes(bytes, aligned) || !add_bytes(memory.peak, aligned))
          return reject(RNS8_RANGE_ERROR, "rank_peak_overflow", rank);
      }
      if (memory.peak > r.ranks[rank].memory_budget)
        return reject(RNS8_WORKSPACE_TOO_SMALL, "rank_memory_budget", rank);
      if (!add_bytes(resources_.descriptor_bytes, memory.descriptors))
        return reject(RNS8_RANGE_ERROR, "descriptor_overflow", rank);
    }
    if (!add_bytes(resources_.pinned_staging, 4))
      return reject(RNS8_RANGE_ERROR, "pinned_status_overflow");
    resources_.host_inputs = native;
    resources_.host_overhead_reserve = r.host_overhead_reserve;
    // Mock owns native bytes, fragment payloads, gathered bytes, cpp_int output
    // staging and state/identities. Outputs/proofs are <160 bits. The full
    // native oracle is <192 bits: K<=PTRDIFF_MAX/8 and each product<2^128.
    // 512/cell reserves these objects/temporaries; allocator overhead is explicit.
    uint64_t integer_bytes = 0;
    if (!multiply(cells, r.host_storage == HostStorage::exact_reference ? 512 : 24, integer_bytes))
      return reject(RNS8_RANGE_ERROR, "host_model_overflow");
    resources_.host_model = native;
    // Exact structural keys, tile proofs and the transient fragment stamp are
    // bounded but not fixed-size Work objects. Reserve separately for them.
    if (!multiply(work_.size(), 2048, resources_.contract_metadata))
      return reject(RNS8_RANGE_ERROR, "contract_metadata_overflow");
    for (auto bytes : {integer_bytes, r.host_storage == HostStorage::exact_reference ? c_span.byte_extent : uint64_t{0}, resources_.descriptor_bytes,
        resources_.contract_metadata, resources_.pinned_staging, r.host_overhead_reserve})
      if (!add_bytes(resources_.host_model, bytes)) return reject(RNS8_RANGE_ERROR, "host_model_overflow");
    if (r.host_storage == HostStorage::exact_reference) for (const auto& memory : resources_.ranks)
      if (!add_bytes(resources_.host_model, memory.local_outputs)) return reject(RNS8_RANGE_ERROR, "host_model_overflow");
    if (resources_.pinned_staging > r.pinned_budget) return reject(RNS8_WORKSPACE_TOO_SMALL, "pinned_memory_budget");
    if (resources_.host_model > r.host_budget) return reject(RNS8_WORKSPACE_TOO_SMALL, "host_memory_budget");
    return {};
  }
};
} // namespace rns8::detail::distributed
