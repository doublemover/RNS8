#pragma once
#include "distributed_inputs.hpp"
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace rns8::detail::distributed {
struct Stamp {
  uint64_t operation = 0, epoch = 0;
  ValueIdentity a, b;
  uint32_t basis_version = 1;
  std::string certificate_sha256, layout_sha256;
  std::string plan_identity;
  bool operator==(const Stamp& rhs) const {
    return operation == rhs.operation && epoch == rhs.epoch && a == rhs.a && b == rhs.b &&
        basis_version == rhs.basis_version && certificate_sha256 == rhs.certificate_sha256 &&
        layout_sha256 == rhs.layout_sha256 && plan_identity == rhs.plan_identity;
  }
};
enum class ResidueEncoding { canonical_u8, centered_i8 };
struct Fragment {
  Stamp stamp;
  Work work;
  // Canonical [0,m) bytes: byte256 is not needed even for modulus256.
  // Centered signed bytes are intentionally a different, rejected encoding.
  std::vector<uint8_t> canonical;
  ResidueEncoding encoding = ResidueEncoding::canonical_u8;
};
struct ExecutionReport {
  const char* schema = "rns8_distributed_mock_v1";
  const char* execution_kind = "cpu_mock_modulus_split";
  bool hardware_executed = false, production_eligible = false;
  State state = State::idle;
  uint64_t operation = 0, epoch = 0, expected = 0, local_complete = 0;
  uint64_t transfer_complete = 0, visible = 0, proof_ready = 0, published_cells = 0;
  uint64_t deadline_ns = 0, elapsed_ns = 0, retained_payload_bytes = 0;
  uint64_t result_readers = 0;
  uint64_t pending_host_stage_work = UINT64_MAX;
  Admission outcome;
  std::vector<uint64_t> missing_work;
  std::vector<uint32_t> missing_rank_acknowledgments, undrained_ranks;
};

class MockExecution {
 public:
  explicit MockExecution(std::shared_ptr<const Plan> plan) : plan_(std::move(plan)) {}
  State state() const { return state_; }
  uint64_t epoch() const { return epoch_; }
  const std::shared_ptr<const Plan>& plan() const { return plan_; }
  const std::vector<Integer>* result() const { return state_ == State::published ? &result_ : nullptr; }
  ValueIdentity result_identity() const {
    return state_ == State::published ? ValueIdentity{plan_->request().operation, epoch_} : ValueIdentity{};
  }
  // Explicit host materialization from the globally complete unique integer.
  // It cannot fabricate suffix planes from one rank's insufficient subset.
  Admission materialize(uint32_t prefix, uint64_t budget, std::vector<uint8_t>& output) const {
    if (state_ != State::published) return reject(RNS8_BUSY, "published_result_required");
    if (!prefix || prefix > 20) return reject(RNS8_INVALID_ARGUMENT, "target_prefix");
    const auto& r = plan_->request();
    Integer product = 1;
    for (uint32_t plane = 0; plane < prefix; ++plane) product *= garner::kConstants.moduli[plane];
    for (const auto& tile : plan_->tiles())
      if (product <= (plan_->signed_value() ? Integer(2 * tile.magnitude) : tile.magnitude))
        return reject(RNS8_RANGE_ERROR, "target_basis_insufficient");
    uint64_t bytes = 0;
    if (!multiply(result_.size(), prefix, bytes)) return reject(RNS8_RANGE_ERROR, "materialization_overflow");
    if (bytes > budget || bytes > SIZE_MAX) return reject(RNS8_WORKSPACE_TOO_SMALL, "materialization_budget");
    try {
      std::vector<uint8_t> staged(static_cast<std::size_t>(bytes));
      for (uint32_t plane = 0; plane < prefix; ++plane) {
        const auto modulus = garner::kConstants.moduli[plane];
        for (std::size_t cell = 0; cell < result_.size(); ++cell) {
          Integer residue = result_[cell] % modulus;
          if (residue < 0) residue += modulus;
          staged[plane * result_.size() + cell] = residue.convert_to<uint8_t>();
        }
      }
      output.swap(staged); return {};
    } catch (const std::bad_alloc&) { return reject(RNS8_INTERNAL_ERROR, "allocation_failed"); }
  }

  Admission begin(uint64_t epoch, std::shared_ptr<const Inputs> inputs, uint64_t now_ns, uint64_t deadline_ns) {
    if (!plan_) return reject(RNS8_INVALID_ARGUMENT, "plan_required");
    if (plan_->request().host_storage != HostStorage::exact_reference)
      return reject(RNS8_INVALID_ARGUMENT, "mock_reference_storage_required");
    if (state_ == State::pending || state_ == State::draining || state_ == State::quarantined || readers_)
      return reject(RNS8_BUSY, "previous_epoch_retained");
    if (!epoch || epoch == UINT64_MAX || epoch <= epoch_ || deadline_ns <= now_ns || !inputs)
      return reject(RNS8_INVALID_ARGUMENT, "epoch_or_deadline");
    const auto& r = plan_->request();
    AddressDomain a, b;
    if (!(inputs->a() == r.a) || !(inputs->b() == r.b))
      return reject(RNS8_INVALID_ARGUMENT, "input_generation_mismatch");
    if (!checked_host_plane_address(r.m, r.k, r.lda, 1, 8, a) ||
        !checked_host_plane_address(r.k, r.n, r.ldb, 1, 8, b) ||
        inputs->lhs().size() != a.byte_extent / 8 || inputs->rhs().size() != b.byte_extent / 8)
      return reject(RNS8_INVALID_ARGUMENT, "input_storage_extent");
    try {
      std::vector<Slot> slots(plan_->work().size());
      std::vector<bool> ranks(r.ranks.size(), false), drained(r.ranks.size(), false);
      std::vector<uint8_t> gather(static_cast<std::size_t>(r.m * r.n * r.prefix), 0);
      std::vector<uint8_t> staging(static_cast<std::size_t>(plan_->resources().pinned_staging));
      // All allocations precede replacing a previous completed result/epoch.
      slots_.swap(slots); rank_done_.swap(ranks); drained_.swap(drained); gather_.swap(gather);
      staging_.swap(staging); staging_owner_ = UINT64_MAX;
      inputs_ = std::move(inputs); result_.clear();
      epoch_ = epoch; start_ns_ = last_ns_ = now_ns; deadline_ns_ = deadline_ns;
      state_ = State::pending; outcome_ = {}; return {};
    } catch (const std::bad_alloc&) { return reject(RNS8_INTERNAL_ERROR, "allocation_failed"); }
  }

  // The reference producer is intentionally synchronous and CPU-only. It emits
  // one explicitly mapped plane; a local shard is never exported as a prefix.
  Admission produce(uint64_t id, Fragment& output) const {
    if (state_ != State::pending || id >= plan_->work().size())
      return reject(RNS8_INVALID_ARGUMENT, "pending_work_required");
    try {
      const auto& r = plan_->request();
      const auto& work = plan_->work()[id];
      const auto& tile = plan_->tiles()[work.tile];
      Fragment fragment{stamp(), work, std::vector<uint8_t>(static_cast<std::size_t>(work.bytes))};
      for (uint64_t row = 0; row < tile.rows; ++row)
        for (uint64_t col = 0; col < tile.cols; ++col) {
          uint32_t sum = 0;
          for (uint64_t k = work.k_begin; k < work.k_end; ++k) {
            const auto a = native_residue(inputs_->lhs()[(tile.row + row) * r.lda + k], plan_->signed_value(), work.modulus);
            const auto b = native_residue(inputs_->rhs()[k * r.ldb + tile.col + col], plan_->signed_value(), work.modulus);
            sum = (sum + a * b) % work.modulus; // <=255^2+255; independent of K
          }
          fragment.canonical[row * tile.cols + col] = static_cast<uint8_t>(sum);
        }
      output = std::move(fragment); return {};
    } catch (const std::bad_alloc&) { return reject(RNS8_INTERNAL_ERROR, "allocation_failed"); }
  }
  Admission local_complete(Fragment fragment, uint64_t now_ns) {
    const auto clock = advance(now_ns); if (!clock) return clock;
    if (!(fragment.stamp == stamp())) return reject(RNS8_INVALID_ARGUMENT, "stale_or_foreign_fragment");
    if (fragment.work.id >= slots_.size()) return reject(RNS8_INVALID_ARGUMENT, "unexpected_work");
    const auto id = fragment.work.id;
    const auto& expected = plan_->work()[id];
    if (fragment.encoding != ResidueEncoding::canonical_u8)
      return reject(RNS8_INVALID_ARGUMENT, "residue_encoding_mismatch", expected.rank, id);
    if (!(fragment.work == expected) || fragment.canonical.size() != expected.bytes)
      return reject(RNS8_INVALID_ARGUMENT, "fragment_membership_or_extent", expected.rank, id);
    if (slots_[id].phase != Phase::expected)
      return reject(RNS8_INVALID_ARGUMENT, "duplicate_fragment", expected.rank, id);
    if (std::any_of(fragment.canonical.begin(), fragment.canonical.end(), [&](uint8_t value) { return value >= expected.modulus; }))
      return reject(RNS8_INVALID_ARGUMENT, "noncanonical_fragment", expected.rank, id);
    slots_[id].payload = std::move(fragment.canonical);
    slots_[id].phase = Phase::local_complete; return {};
  }
  Admission stage_host(uint64_t id, uint64_t now_ns) {
    const auto clock = advance(now_ns); if (!clock) return clock;
    if (id >= slots_.size() || slots_[id].phase != Phase::local_complete)
      return reject(RNS8_INVALID_ARGUMENT, "source_not_ready", UINT32_MAX, id);
    if (plan_->work()[id].transport != Transport::pinned_host)
      return reject(RNS8_INVALID_ARGUMENT, "host_staging_not_selected", UINT32_MAX, id);
    if (staging_owner_ != UINT64_MAX) return reject(RNS8_BUSY, "pinned_slot_in_use", UINT32_MAX, id);
    std::copy(slots_[id].payload.begin(), slots_[id].payload.end(), staging_.begin());
    staging_owner_ = id; slots_[id].phase = Phase::host_staged; return {};
  }
  Admission transfer_complete(uint64_t id, uint64_t now_ns) {
    const auto clock = advance(now_ns); if (!clock) return clock;
    if (id >= slots_.size()) return reject(RNS8_INVALID_ARGUMENT, "unexpected_work", UINT32_MAX, id);
    const auto& r = plan_->request(); const auto& work = plan_->work()[id];
    const bool staged = work.transport == Transport::pinned_host;
    if ((staged && (slots_[id].phase != Phase::host_staged || staging_owner_ != id)) ||
        (!staged && slots_[id].phase != Phase::local_complete))
      return reject(RNS8_INVALID_ARGUMENT, "source_or_host_stage_not_ready", work.rank, id);
    const auto& source = staged ? staging_ : slots_[id].payload;
    const auto& tile = plan_->tiles()[work.tile];
    for (uint64_t row = 0; row < tile.rows; ++row)
      for (uint64_t col = 0; col < tile.cols; ++col)
        {auto& destination=gather_[(work.logical_plane * r.m + tile.row + row) * r.n + tile.col + col];
          destination=plan_->request().axis==Axis::contraction?uint8_t((uint32_t(destination)+source[row*tile.cols+col])%work.modulus):source[row*tile.cols+col];}
    // A staging slot is reusable only after its destination copy completed.
    if (staged) staging_owner_ = UINT64_MAX;
    slots_[id].phase = Phase::transfer_complete; return {};
  }
  Admission destination_visible(uint64_t id, uint64_t now_ns) {
    const auto clock = advance(now_ns); if (!clock) return clock;
    if (id >= slots_.size() || slots_[id].phase != Phase::transfer_complete)
      return reject(RNS8_INVALID_ARGUMENT, "transfer_not_complete", UINT32_MAX, id);
    slots_[id].phase = Phase::visible; return {};
  }
  Admission proof_complete(uint64_t id, uint64_t now_ns) {
    const auto clock = advance(now_ns); if (!clock) return clock;
    if (id >= slots_.size() || slots_[id].phase != Phase::visible)
      return reject(RNS8_INVALID_ARGUMENT, "destination_not_visible", UINT32_MAX, id);
    // Proof/status completion is a distinct mock event. Publication below
    // independently validates the claimed bounds and every integer output.
    slots_[id].phase = Phase::proof_ready; return {};
  }
  Admission acknowledge_rank(uint32_t rank, uint64_t now_ns) {
    const auto clock = advance(now_ns); if (!clock) return clock;
    if (rank >= rank_done_.size() || rank_done_[rank]) return reject(RNS8_INVALID_ARGUMENT, "rank_acknowledgment", rank);
    for (const auto& work : plan_->work())
      if (work.rank == rank && slots_[work.id].phase != Phase::proof_ready)
        return reject(RNS8_BUSY, "rank_work_incomplete", rank, work.id);
    rank_done_[rank] = true; return {}; // includes explicitly idle ranks
  }
  Admission publish(uint64_t now_ns) {
    const auto clock = advance(now_ns); if (!clock) return clock;
    if (std::any_of(rank_done_.begin(), rank_done_.end(), [](bool done) { return !done; }))
      return reject(RNS8_BUSY, "rank_acknowledgment_missing");
    if (std::any_of(slots_.begin(), slots_.end(), [](const Slot& slot) { return slot.phase != Phase::proof_ready; }))
      return reject(RNS8_BUSY, "expected_coverage_missing");
    try {
      const auto& r = plan_->request();
      std::vector<Integer> staged(static_cast<std::size_t>(r.m * r.n));
      for (const auto& tile : plan_->tiles()) {
        Integer product = 1;
        for (uint32_t plane = 0; plane < tile.prefix; ++plane) product *= garner::kConstants.moduli[plane];
        for (uint64_t row = tile.row; row < tile.row + tile.rows; ++row)
          for (uint64_t col = tile.col; col < tile.col + tile.cols; ++col) {
            Integer value = 0, prefix_product = 1;
            for (uint32_t plane = 0; plane < tile.prefix; ++plane) {
              const uint32_t modulus = garner::kConstants.moduli[plane];
              const uint32_t residue = gather_[(plane * r.m + row) * r.n + col];
              const uint32_t previous = (value % modulus).convert_to<uint32_t>();
              const uint32_t digit = ((residue + modulus - previous) * garner::kConstants.inverses[plane]) % modulus;
              value += prefix_product * digit; prefix_product *= modulus;
            }
            if (plan_->signed_value() && value >= (product + 1) / 2) value -= product;
            // Independent full integer dot proves the input-bound declaration,
            // so a too-small basis cannot publish a matching modular alias.
            Integer reference = 0;
            for (uint64_t k = 0; k < r.k; ++k)
              reference += native_integer(inputs_->lhs()[row * r.lda + k], plan_->signed_value()) *
                  native_integer(inputs_->rhs()[k * r.ldb + col], plan_->signed_value());
            if (boost::multiprecision::abs(reference) > tile.magnitude)
              return fail(RNS8_RANGE_ERROR, "actual_value_exceeds_proof");
            if (value != reference) return fail(RNS8_VERIFICATION_FAILED, "reconstructed_value_mismatch");
            if (plan_->bounded() && (reference > (plan_->signed_value() ? Integer(INT64_MAX) : Integer(UINT64_MAX)) ||
                reference < (plan_->signed_value() ? Integer(INT64_MIN) : Integer(0))))
              return fail(RNS8_RANGE_ERROR, "actual_bounded_export_range");
            staged[row * r.n + col] = std::move(value);
          }
      }
      result_.swap(staged); state_ = State::published; outcome_ = {}; return {};
    } catch (const std::bad_alloc&) { return fail(RNS8_INTERNAL_ERROR, "allocation_failed"); }
  }

  Admission poll(uint64_t now_ns) { return advance(now_ns); }
  Admission cancel() {
    if (state_ != State::pending) return reject(RNS8_INVALID_ARGUMENT, "no_pending_epoch");
    return fail(RNS8_CANCELED, "cancel_requested");
  }
  Admission fail_rank(uint32_t rank) {
    if (state_ != State::pending || rank >= rank_done_.size()) return reject(RNS8_INVALID_ARGUMENT, "no_pending_rank");
    return fail(RNS8_BACKEND_FAILURE, "mock_rank_failure", rank);
  }
  Admission acknowledge_cleanup(uint32_t rank, bool drained) {
    if ((state_ != State::draining && state_ != State::quarantined) || rank >= drained_.size() || drained_[rank])
      return reject(RNS8_INVALID_ARGUMENT, "cleanup_acknowledgment", rank);
    if (!drained) {
      state_ = State::quarantined;
      return reject(RNS8_BACKEND_FAILURE, "failed_drain_retains_resources", rank);
    }
    drained_[rank] = true;
    // A failed drain is terminal quarantine for this owner. Later acknowledgments
    // do not undo it or silently make the old buffers reusable.
    if (state_ == State::draining && std::all_of(drained_.begin(), drained_.end(), [](bool value) { return value; })) {
      for (auto& slot : slots_) std::vector<uint8_t>{}.swap(slot.payload);
      std::vector<uint8_t>{}.swap(gather_); std::vector<uint8_t>{}.swap(staging_);
      staging_owner_ = UINT64_MAX; inputs_.reset();
      state_ = outcome_.status == RNS8_CANCELED ? State::canceled :
          outcome_.status == RNS8_TIMEOUT ? State::timed_out : State::failed;
    }
    return {};
  }
  Admission retain_result() {
    if (state_ != State::published || readers_ == UINT64_MAX) return reject(RNS8_INVALID_ARGUMENT, "published_result_required");
    ++readers_; return {};
  }
  Admission release_result() {
    if (!readers_) return reject(RNS8_INVALID_ARGUMENT, "result_reader_missing");
    --readers_; return {};
  }
  ExecutionReport report() const {
    ExecutionReport report;
    if(plan_ && plan_->request().axis!=Axis::modulus)report.execution_kind="cpu_mock_explicit_axes";
    report.state = state_; report.epoch = epoch_; report.operation = plan_ ? plan_->request().operation : 0;
    report.expected = plan_ ? plan_->work().size() : 0;
    report.outcome = outcome_; report.deadline_ns = deadline_ns_;
    report.elapsed_ns = last_ns_ - start_ns_; report.result_readers = readers_;
    report.pending_host_stage_work = staging_owner_;
    report.published_cells = state_ == State::published ? result_.size() : 0;
    for (uint64_t id = 0; id < report.expected; ++id) {
      const auto phase = id < slots_.size() ? slots_[id].phase : Phase::expected;
      report.local_complete += phase >= Phase::local_complete;
      report.transfer_complete += phase >= Phase::transfer_complete;
      report.visible += phase >= Phase::visible;
      report.proof_ready += phase >= Phase::proof_ready;
      if (phase != Phase::proof_ready) report.missing_work.push_back(id);
    }
    for (uint32_t rank = 0; rank < rank_done_.size(); ++rank) {
      if (!rank_done_[rank]) report.missing_rank_acknowledgments.push_back(rank);
      if ((state_ == State::draining || state_ == State::quarantined) && !drained_[rank]) report.undrained_ranks.push_back(rank);
    }
    for (const auto& slot : slots_) report.retained_payload_bytes += slot.payload.size();
    report.retained_payload_bytes += gather_.size();
    report.retained_payload_bytes += staging_.size();
    return report;
  }
 private:
  struct Slot { Phase phase = Phase::expected; std::vector<uint8_t> payload; };
  std::shared_ptr<const Plan> plan_;
  std::shared_ptr<const Inputs> inputs_;
  std::vector<Slot> slots_;
  std::vector<bool> rank_done_, drained_;
  std::vector<uint8_t> gather_, staging_;
  uint64_t staging_owner_ = UINT64_MAX;
  std::vector<Integer> result_;
  State state_ = State::idle;
  uint64_t epoch_ = 0, start_ns_ = 0, last_ns_ = 0, deadline_ns_ = 0, readers_ = 0;
  Admission outcome_;
  Stamp stamp() const {
    const auto& r = plan_->request();
    return {r.operation, epoch_, r.a, r.b, 1, r.certificate_sha256, r.layout_sha256, plan_->identity()};
  }
  Admission fail(rns8_status status, const char* reason, uint32_t rank = UINT32_MAX) {
    outcome_ = reject(status, reason, rank); state_ = State::draining; return outcome_;
  }
  Admission advance(uint64_t now_ns) {
    if (state_ != State::pending) return reject(RNS8_INVALID_ARGUMENT, "pending_epoch_required");
    if (now_ns < last_ns_) return reject(RNS8_INVALID_ARGUMENT, "monotonic_clock_required");
    last_ns_ = now_ns;
    if (now_ns >= deadline_ns_) return fail(RNS8_TIMEOUT, "operation_deadline");
    return {};
  }
};

inline const char* state_name(State state) {
  switch (state) {
    case State::idle: return "idle";
    case State::pending: return "pending";
    case State::published: return "published";
    case State::draining: return "draining";
    case State::failed: return "failed";
    case State::canceled: return "canceled";
    case State::timed_out: return "timed_out";
    case State::quarantined: return "quarantined";
  }
  return "unknown";
}
// Small report adapter, not benchmark-schema evidence. Free-form strings use
// JSON escapes; no external dependency or hand-built fake timing is necessary.
inline std::string json_string(const std::string& value) {
  std::ostringstream out; out << '"';
  for (unsigned char c : value) {
    if (c == '"' || c == '\\') out << '\\' << c;
    else if (c < 32 || c >= 127) out << "\\u00" << std::hex << std::setw(2) << std::setfill('0') << unsigned(c) << std::dec;
    else out << c;
  }
  out << '"'; return out.str();
}
inline std::string report_json(const MockExecution& execution) {
  if (!execution.plan()) throw std::invalid_argument("plan_required");
  const auto report = execution.report(); const auto& plan = *execution.plan(); const auto& r = plan.request();
  std::ostringstream out;
  out << "{\"schema\":\"rns8_distributed_mock_v1\",\"execution_kind\":\"cpu_mock_modulus_split\","
      << "\"hardware_executed\":false,\"production_eligible\":false,\"axis\":\"modulus\","
      << "\"state\":" << json_string(state_name(report.state)) << ",\"operation\":" << report.operation
      << ",\"epoch\":" << report.epoch << ",\"shape\":[" << r.m << ',' << r.n << ',' << r.k << ']'
      << ",\"input_identities\":[[" << r.a.instance << ',' << r.a.generation << "],[" << r.b.instance << ',' << r.b.generation << "]]"
      << ",\"certificate_sha256\":" << json_string(r.certificate_sha256)
      << ",\"layout_sha256\":" << json_string(r.layout_sha256)
      << ",\"plan_identity\":" << json_string(plan.identity())
      << ",\"semantics\":" << unsigned(r.semantics)
      << ",\"distribution_axis\":" << unsigned(r.axis) << ",\"k_fragments\":" << plan.k_fragments()
      << ",\"rank_grid_rows\":" << r.rank_grid_rows << ",\"rank_grid_cols\":" << r.rank_grid_cols
      << ",\"basis\":\"rns8_default_ordered_v1\",\"residue_encoding\":\"canonical_u8\",\"prefix\":" << r.prefix
      << ",\"expected\":" << report.expected << ",\"local_complete\":" << report.local_complete
      << ",\"transfer_complete\":" << report.transfer_complete << ",\"destination_visible\":" << report.visible
      << ",\"proof_ready\":" << report.proof_ready << ",\"published_cells\":" << report.published_cells
      << ",\"status\":" << unsigned(report.outcome.status) << ",\"reason\":" << json_string(report.outcome.reason)
      << ",\"failed_rank\":" << report.outcome.rank << ",\"failed_work\":" << report.outcome.work
      << ",\"elapsed_ns\":" << report.elapsed_ns << ",\"deadline_ns\":" << report.deadline_ns
      << ",\"clock\":\"caller_supplied_mock_monotonic_ns\",\"performance_timing\":null"
      << ",\"host_model_bytes\":" << plan.resources().host_model
      << ",\"pinned_staging_bytes\":" << plan.resources().pinned_staging
      << ",\"transfer_bytes\":" << plan.resources().transfer_bytes
      << ",\"retained_payload_bytes\":" << report.retained_payload_bytes
      << ",\"result_readers\":" << report.result_readers
      << ",\"pending_host_stage_work\":" << report.pending_host_stage_work
      << ",\"allocator_overhead_known\":false,\"basis_moduli\":[";
  for (uint32_t plane = 0; plane < r.prefix; ++plane) { if (plane) out << ','; out << garner::kConstants.moduli[plane]; }
  out << "],\"tiles\":[";
  for (std::size_t i = 0; i < plan.tiles().size(); ++i) {
    if (i) out << ',';
    const auto& tile = plan.tiles()[i];
    out << "{\"tile\":" << i << ",\"row\":" << tile.row << ",\"col\":" << tile.col
        << ",\"rows\":" << tile.rows << ",\"cols\":" << tile.cols << ",\"prefix\":" << tile.prefix
        << ",\"magnitude_bound\":" << json_string(tile.magnitude.str()) << '}';
  }
  out << "],\"expected_work\":[";
  for (const auto& work : plan.work()) {
    if (work.id) out << ',';
    out << "{\"id\":" << work.id << ",\"rank\":" << work.rank << ",\"tile\":" << work.tile
        << ",\"logical_plane\":" << work.logical_plane << ",\"modulus\":" << work.modulus
        << ",\"k_begin\":" << work.k_begin << ",\"k_end\":" << work.k_end
        << ",\"byte_offset\":" << work.byte_offset << ",\"bytes\":" << work.bytes
        << ",\"gather_offset\":" << work.gather_offset
        << ",\"transport\":" << json_string(work.transport == Transport::local ? "local" :
            work.transport == Transport::peer ? "peer" : "pinned_host") << '}';
  }
  out << "],\"ranks\":[";
  for (uint32_t rank = 0; rank < r.ranks.size(); ++rank) {
    if (rank) out << ',';
    uint64_t count = 0; for (const auto& work : plan.work()) count += work.rank == rank;
    out << "{\"rank\":" << rank << ",\"ordinal\":" << r.ranks[rank].ordinal
        << ",\"physical_id\":" << json_string(r.ranks[rank].physical_id)
        << ",\"target\":" << json_string(r.ranks[rank].target)
        << ",\"physical_package_id\":" << json_string(r.ranks[rank].physical_package_id)
        << ",\"compute_partition\":" << json_string(r.ranks[rank].compute_partition)
        << ",\"memory_partition\":" << json_string(r.ranks[rank].memory_partition)
        << ",\"expected_work\":" << count << ",\"idle\":" << (count ? "false" : "true")
        << ",\"reserved_peak_bytes\":" << plan.resources().ranks[rank].peak << '}';
  }
  out << "],\"missing_work\":[";
  for (std::size_t i = 0; i < report.missing_work.size(); ++i) { if (i) out << ','; out << report.missing_work[i]; }
  out << "],\"missing_rank_acknowledgments\":[";
  for (std::size_t i = 0; i < report.missing_rank_acknowledgments.size(); ++i) { if (i) out << ','; out << report.missing_rank_acknowledgments[i]; }
  out << "],\"undrained_ranks\":[";
  for (std::size_t i = 0; i < report.undrained_ranks.size(); ++i) { if (i) out << ','; out << report.undrained_ranks[i]; }
  out << "]}"; return out.str();
}
} // namespace rns8::detail::distributed
