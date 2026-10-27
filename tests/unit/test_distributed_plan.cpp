#include "backend_common/distributed_mock.hpp"
#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>
#include <numeric>
#include <random>
#include <set>

namespace {
namespace d = rns8::detail::distributed;
using boost::multiprecision::cpp_int;

d::Request distributed_request(uint32_t ranks = 3) {
  d::Request r;
  r.operation = 71; r.m = 3; r.n = 5; r.k = 7; r.lda = 9; r.ldb = 8;
  r.prefix = 20; r.tile_rows = 2; r.tile_cols = 3;
  r.a = {17, 3}; r.b = {29, 8};
  r.certificate_sha256 = std::string(64, 'a');
  r.layout_sha256 = std::string(64, 'b');
  r.magnitude_bound = cpp_int(1) << 140;
  r.host_budget = uint64_t{1} << 30; r.pinned_budget = 1u << 20;
  r.transport = ranks == 1 ? d::Transport::local : d::Transport::pinned_host;
  for (uint32_t rank = 0; rank < ranks; ++rank) {
    d::Rank selected;
    selected.ordinal = 2 + rank * 3; // deliberately noncontiguous runtime ordinals
    selected.physical_id = "mock-device-" + std::to_string(rank);
    selected.target = "mock-only";
    selected.memory_budget = uint64_t{1} << 30;
    r.ranks.push_back(selected);
    if (rank) r.edges.push_back({rank, 0, d::Support::unknown, d::Support::supported});
  }
  return r;
}
std::shared_ptr<const d::Plan> distributed_plan(const d::Request& request) {
  std::shared_ptr<const d::Plan> plan;
  const auto admission = d::Plan::create(request, plan);
  REQUIRE(admission.status == RNS8_SUCCESS);
  REQUIRE(plan);
  return plan;
}
std::shared_ptr<const d::Inputs> distributed_inputs(const d::Request& r) {
  std::vector<uint64_t> a((r.m - 1) * r.lda + r.k, UINT64_C(0xdeadbeef));
  std::vector<uint64_t> b((r.k - 1) * r.ldb + r.n, UINT64_C(0xfeedface));
  const uint64_t values[] = {0, 1, UINT64_MAX, uint64_t{1} << 63,
      (uint64_t{1} << 63) - 1, UINT64_C(0xffffffff00000001), 17};
  for (uint64_t row = 0; row < r.m; ++row)
    for (uint64_t k = 0; k < r.k; ++k) a[row * r.lda + k] = values[(row * 3 + k) % 7];
  for (uint64_t k = 0; k < r.k; ++k)
    for (uint64_t col = 0; col < r.n; ++col) b[k * r.ldb + col] = values[(k * 2 + col + 1) % 7];
  return std::make_shared<const d::Inputs>(r.a, r.b, std::move(a), std::move(b));
}
std::vector<cpp_int> distributed_oracle(const d::Request& r, const d::Inputs& inputs) {
  // Independent classical full product: no shard producer, Garner constants or
  // mock reconstruction helper is used to certify its own output.
  const bool signed_value = r.semantics == RNS8_BOUNDED_I64 || r.semantics == RNS8_EXACT_WIDE_SIGNED;
  const auto decode = [&](uint64_t bits) -> cpp_int {
    return signed_value && bits > uint64_t(INT64_MAX) ? cpp_int(bits) - (cpp_int(1) << 64) : cpp_int(bits);
  };
  std::vector<cpp_int> output(r.m * r.n);
  for (uint64_t row = 0; row < r.m; ++row)
    for (uint64_t col = 0; col < r.n; ++col)
      for (uint64_t k = 0; k < r.k; ++k)
        output[row * r.n + col] += decode(inputs.lhs()[row * r.lda + k]) * decode(inputs.rhs()[k * r.ldb + col]);
  return output;
}
void deliver(d::MockExecution& execution, uint64_t id, uint64_t& clock) {
  d::Fragment fragment;
  REQUIRE(execution.produce(id, fragment).status == RNS8_SUCCESS);
  REQUIRE(execution.local_complete(std::move(fragment), ++clock).status == RNS8_SUCCESS);
  if (execution.plan()->work()[id].transport == d::Transport::pinned_host)
    REQUIRE(execution.stage_host(id, ++clock).status == RNS8_SUCCESS);
  REQUIRE(execution.transfer_complete(id, ++clock).status == RNS8_SUCCESS);
  REQUIRE(execution.destination_visible(id, ++clock).status == RNS8_SUCCESS);
  REQUIRE(execution.proof_complete(id, ++clock).status == RNS8_SUCCESS);
}
void finish(d::MockExecution& execution, uint64_t& clock, uint32_t seed = 13) {
  std::vector<uint64_t> order(execution.plan()->work().size());
  std::iota(order.begin(), order.end(), 0);
  std::mt19937 generator(seed);
  std::shuffle(order.begin(), order.end(), generator);
  for (auto id : order) deliver(execution, id, clock);
  for (uint32_t rank = 0; rank < execution.plan()->request().ranks.size(); ++rank)
    REQUIRE(execution.acknowledge_rank(rank, ++clock).status == RNS8_SUCCESS);
}
void drain(d::MockExecution& execution) {
  for (uint32_t rank = 0; rank < execution.plan()->request().ranks.size(); ++rank)
    REQUIRE(execution.acknowledge_cleanup(rank, true).status == RNS8_SUCCESS);
}
}

TEST_CASE("distributed modulus mock matches independent full-width Boost product in every rank order", "[distributed][cpu]") {
  for (uint32_t ranks : {1u, 2u, 3u, 4u})
    for (auto semantics : {RNS8_EXACT_WIDE_SIGNED, RNS8_EXACT_WIDE_UNSIGNED}) {
      auto r = distributed_request(ranks); r.semantics = semantics;
      auto plan = distributed_plan(r); auto inputs = distributed_inputs(r);
      const auto expected = distributed_oracle(r, *inputs);
      for (uint32_t seed : {3u, 11u, 97u}) {
        d::MockExecution execution(plan); uint64_t clock = 10;
        REQUIRE(execution.begin(1, inputs, clock, 1000000).status == RNS8_SUCCESS);
        CHECK(execution.result() == nullptr);
        finish(execution, clock, seed);
        REQUIRE(execution.publish(++clock).status == RNS8_SUCCESS);
        REQUIRE(execution.result());
        CHECK(*execution.result() == expected);
        CHECK(execution.report().missing_work.empty());
        CHECK(execution.report().published_cells == r.m * r.n);
      }
    }
}

TEST_CASE("distributed ordered bases retain complete identity through every axis and exact reconstruction", "[distributed][cpu]") {
  for(auto axis:{d::Axis::modulus,d::Axis::rows,d::Axis::columns,d::Axis::hybrid,d::Axis::contraction}) {
    auto r=distributed_request(3);r.axis=axis;r.magnitude_bound=cpp_int(1)<<135;
    r.basis={9,16,25,251,241,239,233,229,227,223,211,199,197,193,191,181,179,173,167,163};
    if(axis==d::Axis::hybrid){r.rank_grid_rows=1;r.rank_grid_cols=3;}
    auto plan=distributed_plan(r);const auto identity=plan->identity();
    auto inputs=distributed_inputs(r);d::MockExecution execution(plan);uint64_t clock=1;
    REQUIRE(execution.begin(1,inputs,clock,1000000));finish(execution,clock);REQUIRE(execution.publish(++clock));
    CHECK(*execution.result()==distributed_oracle(r,*inputs));
    for(const auto& work:plan->work())CHECK(work.modulus==r.basis[work.logical_plane]);
    std::reverse(r.basis.begin(),r.basis.end());auto reordered=distributed_plan(r);CHECK(reordered->identity()!=identity);
    CHECK(reordered->basis_packet().identity!=plan->basis_packet().identity);
    std::vector<uint8_t> materialized;REQUIRE(execution.materialize(20,1000000,materialized));
    for(unsigned plane=0;plane<20;++plane)for(size_t cell=0;cell<execution.result()->size();++cell){
      cpp_int residue=(*execution.result())[cell]%plan->request().basis[plane];if(residue<0)residue+=plan->request().basis[plane];
      CHECK(materialized[plane*execution.result()->size()+cell]==residue.convert_to<unsigned>());}
    r.basis[1]=r.basis[0];std::shared_ptr<const d::Plan> rejected;
    CHECK(d::Plan::create(r,rejected).status==RNS8_INVALID_ARGUMENT);
  }
}

TEST_CASE("distributed output and contraction axes own exact complete fragment coverage", "[distributed][cpu]") {
  for(uint32_t ranks:{1u,2u,3u,4u})for(auto axis:{d::Axis::rows,d::Axis::columns,d::Axis::hybrid,d::Axis::contraction})
  for(auto semantics:{RNS8_EXACT_WIDE_SIGNED,RNS8_EXACT_WIDE_UNSIGNED}) {
    auto r=distributed_request(ranks);r.axis=axis;r.semantics=semantics;
    if(axis==d::Axis::hybrid){r.rank_grid_rows=ranks==4?2:1;r.rank_grid_cols=ranks/r.rank_grid_rows;}
    if(axis==d::Axis::contraction)r.k_fragments=3;
    auto plan=distributed_plan(r);const auto inputs=distributed_inputs(r);const auto expected=distributed_oracle(r,*inputs);
    for(uint64_t tile=0;tile<plan->tiles().size();++tile)for(uint32_t plane=0;plane<r.prefix;++plane){
      uint64_t end=0;uint32_t count=0,owner=UINT32_MAX;
      for(const auto& work:plan->work())if(work.tile==tile && work.logical_plane==plane){
        CHECK(work.k_begin==end);CHECK(work.k_end>work.k_begin);end=work.k_end;++count;
        if(axis!=d::Axis::contraction){if(owner==UINT32_MAX)owner=work.rank;CHECK(work.rank==owner);}
        if(axis==d::Axis::contraction)CHECK(work.gather_offset+work.bytes<=plan->resources().ranks[r.owner].partial_gather);
      }
      CHECK(end==r.k);CHECK(count==plan->k_fragments());
    }
    for(uint32_t seed:{3u,97u}){d::MockExecution execution(plan);uint64_t clock=10;
      REQUIRE(execution.begin(1,inputs,clock,1000000));finish(execution,clock,seed);
      REQUIRE(execution.publish(++clock));REQUIRE(execution.result());CHECK(*execution.result()==expected);}
    CHECK(plan->identity()!=distributed_plan(distributed_request(ranks))->identity());
  }
}
TEST_CASE("distributed K fragments reduce enormous canceling partials without using the final bound", "[distributed][cpu]") {
  auto r=distributed_request(2);r.m=r.n=1;r.k=2;r.lda=2;r.ldb=1;r.tile_rows=r.tile_cols=1;r.axis=d::Axis::contraction;r.k_fragments=2;r.magnitude_bound=0;
  auto plan=distributed_plan(r);auto inputs=std::make_shared<const d::Inputs>(r.a,r.b,
      std::vector<uint64_t>{uint64_t(INT64_MAX),uint64_t(INT64_MAX)},std::vector<uint64_t>{uint64_t(INT64_MAX),uint64_t(-INT64_MAX)});
  d::MockExecution execution(plan);uint64_t clock=10;REQUIRE(execution.begin(1,inputs,clock,1000000));
  d::Fragment fragment;REQUIRE(execution.produce(0,fragment));auto altered=fragment;altered.work.k_end=2;
  CHECK(execution.local_complete(std::move(altered),++clock).status==RNS8_INVALID_ARGUMENT);
  finish(execution,clock,7);REQUIRE(execution.publish(++clock));REQUIRE(execution.result());CHECK(execution.result()->at(0)==0);
}

TEST_CASE("distributed immutable coverage owns each tile plane and full K exactly once", "[distributed][cpu]") {
  auto r = distributed_request(4);
  r.prefix = 4; r.magnitude_bound = 100;
  r.tile_proofs = {{1, 100}, {2, 100}, {3, 100}, {4, 100}};
  r.plane_costs = {1, 7, 2, 9}; r.cost_profile_sha256 = std::string(64, 'c');
  r.ranks[1].weight = 3;
  auto plan = distributed_plan(r);
  std::set<std::pair<uint64_t, uint32_t>> coverage;
  std::vector<uint64_t> next(r.ranks.size());
  for (const auto& work : plan->work()) {
    REQUIRE(coverage.emplace(work.tile, work.logical_plane).second);
    CHECK(work.logical_plane < plan->tiles()[work.tile].prefix);
    CHECK(work.k_begin == 0); CHECK(work.k_end == r.k);
    CHECK(work.byte_offset == next[work.rank]); next[work.rank] += work.bytes;
    CHECK(work.bytes == plan->tiles()[work.tile].rows * plan->tiles()[work.tile].cols);
  }
  CHECK(coverage.size() == 10);
  auto key = plan->identity();
  r.tile_proofs[0].prefix = 4;
  r.ranks[0].physical_id = "caller-mutated";
  CHECK(plan->identity() == key);
  CHECK(plan->tiles()[0].prefix == 1);
  CHECK(plan->request().ranks[0].physical_id == "mock-device-0");
  CHECK(distributed_plan(plan->request())->identity() == key);
}

TEST_CASE("distributed idle ranks must acknowledge and local completion is not publication", "[distributed][cpu]") {
  auto r = distributed_request(4);
  r.m = r.n = r.k = r.lda = r.ldb = 1; r.prefix = 1; r.magnitude_bound = 1;
  auto plan = distributed_plan(r);
  auto inputs = std::make_shared<const d::Inputs>(r.a, r.b, std::vector<uint64_t>{1}, std::vector<uint64_t>{1});
  d::MockExecution execution(plan); uint64_t clock = 1;
  REQUIRE(execution.begin(4, inputs, clock, 1000).status == RNS8_SUCCESS);
  d::Fragment fragment; REQUIRE(execution.produce(0, fragment).status == RNS8_SUCCESS);
  CHECK(execution.destination_visible(0, ++clock).status == RNS8_INVALID_ARGUMENT);
  REQUIRE(execution.local_complete(fragment, ++clock).status == RNS8_SUCCESS);
  CHECK(execution.local_complete(fragment, ++clock).status == RNS8_INVALID_ARGUMENT);
  CHECK(execution.report().visible == 0); CHECK(execution.result() == nullptr);
  CHECK(execution.acknowledge_rank(0, ++clock).status == RNS8_BUSY);
  REQUIRE(execution.transfer_complete(0, ++clock).status == RNS8_SUCCESS);
  CHECK(execution.report().visible == 0);
  REQUIRE(execution.destination_visible(0, ++clock).status == RNS8_SUCCESS);
  CHECK(execution.proof_complete(0, clock - 1).status == RNS8_INVALID_ARGUMENT);
  REQUIRE(execution.proof_complete(0, ++clock).status == RNS8_SUCCESS);
  REQUIRE(execution.acknowledge_rank(0, ++clock).status == RNS8_SUCCESS);
  CHECK(execution.publish(++clock).status == RNS8_BUSY);
  CHECK(execution.report().missing_rank_acknowledgments == std::vector<uint32_t>{1, 2, 3});
  for (uint32_t rank = 1; rank < 4; ++rank) REQUIRE(execution.acknowledge_rank(rank, ++clock).status == RNS8_SUCCESS);
  REQUIRE(execution.publish(++clock).status == RNS8_SUCCESS);
  CHECK((*execution.result())[0] == 1);
}

TEST_CASE("distributed membership rejects stale foreign duplicate and misencoded fragments", "[distributed][cpu]") {
  auto r = distributed_request(3); auto plan = distributed_plan(r);
  d::MockExecution execution(plan); uint64_t clock = 1;
  REQUIRE(execution.begin(9, distributed_inputs(r), clock, 1000000).status == RNS8_SUCCESS);
  d::Fragment original; REQUIRE(execution.produce(1, original).status == RNS8_SUCCESS);
  const auto reject = [&](d::Fragment changed) {
    CHECK(execution.local_complete(std::move(changed), ++clock).status == RNS8_INVALID_ARGUMENT);
    CHECK(execution.report().local_complete == 0);
    CHECK(execution.state() == d::State::pending);
  };
  for (auto field : {&d::Stamp::operation, &d::Stamp::epoch}) {
    auto changed = original; ++(changed.stamp.*field); reject(changed);
  }
  auto changed = original; ++changed.stamp.a.generation; reject(changed);
  changed = original; ++changed.stamp.b.instance; reject(changed);
  changed = original; changed.stamp.basis_version = 2; reject(changed);
  changed = original; changed.stamp.certificate_sha256[0] = 'f'; reject(changed);
  changed = original; changed.stamp.layout_sha256[0] = 'f'; reject(changed);
  changed = original; changed.stamp.plan_identity += "foreign"; reject(changed);
  for (auto field : {&d::Work::tile, &d::Work::k_begin, &d::Work::k_end, &d::Work::byte_offset, &d::Work::bytes}) {
    changed = original; ++(changed.work.*field); reject(changed);
  }
  changed = original; changed.work.rank = 63; reject(changed);
  changed = original; changed.work.logical_plane = 19; reject(changed);
  changed = original; changed.work.modulus = 127; reject(changed);
  changed = original; changed.canonical.pop_back(); reject(changed);
  changed = original; changed.canonical[0] = 255; reject(changed); // plane1 modulus255
  changed = original; changed.encoding = d::ResidueEncoding::centered_i8; reject(changed);
  REQUIRE(execution.local_complete(original, ++clock).status == RNS8_SUCCESS);
  CHECK(execution.local_complete(original, ++clock).status == RNS8_INVALID_ARGUMENT);
  CHECK(execution.report().local_complete == 1);
  CHECK(execution.publish(++clock).status == RNS8_BUSY);
}

TEST_CASE("distributed cancellation deadline and failed drain retain ownership until safe cleanup", "[distributed][cpu]") {
  auto r = distributed_request(2); auto plan = distributed_plan(r); auto inputs = distributed_inputs(r);
  for (auto status : {RNS8_CANCELED, RNS8_TIMEOUT, RNS8_BACKEND_FAILURE}) {
    d::MockExecution execution(plan); uint64_t clock = 1;
    REQUIRE(execution.begin(1, inputs, clock, 1000).status == RNS8_SUCCESS);
    deliver(execution, 0, clock);
    const auto failed = status == RNS8_CANCELED ? execution.cancel() :
        status == RNS8_TIMEOUT ? execution.poll(1000) : execution.fail_rank(1);
    CHECK(failed.status == status);
    CHECK(execution.state() == d::State::draining);
    CHECK(execution.result() == nullptr);
    CHECK(execution.report().retained_payload_bytes > 0);
    CHECK(execution.begin(2, inputs, 1001, 2000).status == RNS8_BUSY);
    drain(execution);
    CHECK(execution.report().retained_payload_bytes == 0);
    CHECK(execution.report().proof_ready == 1); // cleanup does not erase history
    CHECK(execution.begin(1, inputs, 1001, 2000).status == RNS8_INVALID_ARGUMENT);
    REQUIRE(execution.begin(2, inputs, 1001, 1000000).status == RNS8_SUCCESS);
    d::Fragment fresh; REQUIRE(execution.produce(0, fresh).status == RNS8_SUCCESS);
    --fresh.stamp.epoch;
    CHECK(execution.local_complete(fresh, 1002).status == RNS8_INVALID_ARGUMENT);
  }
  d::MockExecution poisoned(plan); uint64_t clock = 1;
  REQUIRE(poisoned.begin(1, inputs, clock, 1000).status == RNS8_SUCCESS);
  deliver(poisoned, 0, clock);
  REQUIRE(poisoned.cancel().status == RNS8_CANCELED);
  CHECK(poisoned.acknowledge_cleanup(0, false).status == RNS8_BACKEND_FAILURE);
  CHECK(poisoned.state() == d::State::quarantined);
  CHECK(poisoned.report().retained_payload_bytes > 0);
  CHECK(poisoned.begin(2, inputs, 1001, 2000).status == RNS8_BUSY);
}

TEST_CASE("distributed complete coverage still cannot publish corrupt residues or aliased range proofs", "[distributed][cpu]") {
  for (bool corrupt : {false, true}) {
    auto r = distributed_request(2);
    r.m = r.n = r.k = r.lda = r.ldb = 1; r.prefix = corrupt ? 2 : 1;
    r.magnitude_bound = corrupt ? 1000 : 1;
    auto plan = distributed_plan(r);
    // A false prefix1 bound would reconstruct513 as1; full integer check rejects.
    auto inputs = std::make_shared<const d::Inputs>(r.a, r.b,
        std::vector<uint64_t>{corrupt ? 3u : 513u}, std::vector<uint64_t>{1});
    d::MockExecution execution(plan); uint64_t clock = 1;
    REQUIRE(execution.begin(1, inputs, clock, 1000).status == RNS8_SUCCESS);
    for (const auto& work : plan->work()) {
      d::Fragment fragment; REQUIRE(execution.produce(work.id, fragment).status == RNS8_SUCCESS);
      if (corrupt && work.id == 0) fragment.canonical[0] = 4;
      REQUIRE(execution.local_complete(std::move(fragment), ++clock).status == RNS8_SUCCESS);
      if (work.transport == d::Transport::pinned_host)
        REQUIRE(execution.stage_host(work.id, ++clock).status == RNS8_SUCCESS);
      REQUIRE(execution.transfer_complete(work.id, ++clock).status == RNS8_SUCCESS);
      REQUIRE(execution.destination_visible(work.id, ++clock).status == RNS8_SUCCESS);
      REQUIRE(execution.proof_complete(work.id, ++clock).status == RNS8_SUCCESS);
    }
    for (uint32_t rank = 0; rank < 2; ++rank) REQUIRE(execution.acknowledge_rank(rank, ++clock).status == RNS8_SUCCESS);
    CHECK(execution.publish(++clock).status == (corrupt ? RNS8_VERIFICATION_FAILED : RNS8_RANGE_ERROR));
    CHECK(execution.result() == nullptr);
    CHECK(execution.state() == d::State::draining);
  }
}

TEST_CASE("distributed transactional plan admission rejects malformed coverage transport and budgets", "[distributed][cpu]") {
  auto valid = distributed_request(2); auto output = distributed_plan(valid); const auto original = output;
  const auto rejected = [&](d::Request request, rns8_status expected) {
    const auto admission = d::Plan::create(request, output);
    CHECK(admission.status == expected); CHECK(output == original);
  };
  {auto invalid=valid;invalid.axis=d::Axis::hybrid;rejected(invalid,RNS8_INVALID_ARGUMENT);}
  {auto invalid=valid;invalid.k_fragments=2;rejected(invalid,RNS8_INVALID_ARGUMENT);}
  {auto invalid=valid;invalid.axis=d::Axis::contraction;invalid.k_fragments=65;rejected(invalid,RNS8_INVALID_ARGUMENT);}
  for (auto semantics : {RNS8_FINITE_RING_U8, RNS8_FINITE_FIELD_U8, RNS8_WRAP_U64_MOD_2_64}) {
    auto r = valid; r.semantics = semantics; rejected(r, RNS8_UNSUPPORTED_BACKEND);
  }
  auto r = valid; r.ranks[1].ordinal = r.ranks[0].ordinal; rejected(r, RNS8_INVALID_ARGUMENT);
  r = valid; r.ranks[1].physical_id = r.ranks[0].physical_id; rejected(r, RNS8_INVALID_ARGUMENT);
  r = valid; r.owner = 2; rejected(r, RNS8_INVALID_ARGUMENT);
  r = valid; r.a.generation = UINT64_MAX; rejected(r, RNS8_INVALID_ARGUMENT);
  r = valid; r.certificate_sha256 = "unknown"; rejected(r, RNS8_INVALID_ARGUMENT);
  r = valid; r.layout_sha256.clear(); rejected(r, RNS8_INVALID_ARGUMENT);
  r = valid; r.edges.clear(); rejected(r, RNS8_UNSUPPORTED_BACKEND);
  r = valid; r.transport = d::Transport::peer; rejected(r, RNS8_UNSUPPORTED_BACKEND);
  r = valid; r.edges[0].pinned_host = d::Support::unsupported; rejected(r, RNS8_UNSUPPORTED_BACKEND);
  r = valid; r.edges[0].source = 0; r.edges[0].destination = 1; rejected(r, RNS8_UNSUPPORTED_BACKEND); // direction matters
  r = valid; r.edges.push_back(r.edges[0]); rejected(r, RNS8_INVALID_ARGUMENT);
  r = valid; r.prefix = 1; rejected(r, RNS8_RANGE_ERROR);
  r = valid; r.tile_proofs = {{0, 0}}; rejected(r, RNS8_INVALID_ARGUMENT);
  r = valid; r.m = UINT64_MAX; rejected(r, RNS8_RANGE_ERROR);
  r = valid; r.maximum_work_items = 1; rejected(r, RNS8_RANGE_ERROR);
  r = valid; r.ranks[0].memory_budget = 1; rejected(r, RNS8_WORKSPACE_TOO_SMALL);
  r = valid; r.host_budget = 1; rejected(r, RNS8_WORKSPACE_TOO_SMALL);
  r = valid; r.pinned_budget = 0; rejected(r, RNS8_WORKSPACE_TOO_SMALL);
  r = valid; r.ranks[0].scratch_bytes = UINT64_MAX; rejected(r, RNS8_RANGE_ERROR);
  r = valid; r.plane_costs = {1}; rejected(r, RNS8_INVALID_ARGUMENT);
  r = valid; r.plane_costs.assign(r.prefix, 0); r.cost_profile_sha256 = std::string(64, 'c'); rejected(r, RNS8_INVALID_ARGUMENT);
  auto one = distributed_request(1); auto bound = distributed_plan(one);
  one.ranks[0].memory_budget = bound->resources().ranks[0].peak;
  one.host_budget = bound->resources().host_model;
  REQUIRE(d::Plan::create(one, output).status == RNS8_SUCCESS);
  --one.host_budget;
  CHECK(d::Plan::create(one, output).status == RNS8_WORKSPACE_TOO_SMALL);
}

TEST_CASE("distributed input identity result readers and fresh epoch preserve a prior completed value", "[distributed][cpu]") {
  auto r = distributed_request(1); auto plan = distributed_plan(r); auto inputs = distributed_inputs(r);
  d::MockExecution execution(plan); uint64_t clock = 1;
  REQUIRE(execution.begin(7, inputs, clock, 1000000).status == RNS8_SUCCESS);
  finish(execution, clock);
  REQUIRE(execution.publish(++clock).status == RNS8_SUCCESS);
  const auto expected = *execution.result();
  auto identity = r.a; ++identity.generation;
  auto changed = std::make_shared<const d::Inputs>(identity, r.b, inputs->lhs(), inputs->rhs());
  CHECK(execution.begin(8, changed, ++clock, 1000000).status == RNS8_INVALID_ARGUMENT);
  CHECK(*execution.result() == expected);
  REQUIRE(execution.retain_result().status == RNS8_SUCCESS);
  CHECK(execution.begin(8, inputs, ++clock, 1000000).status == RNS8_BUSY);
  REQUIRE(execution.release_result().status == RNS8_SUCCESS);
  REQUIRE(execution.begin(8, inputs, ++clock, 1000000).status == RNS8_SUCCESS);
  CHECK(execution.result() == nullptr);
}

TEST_CASE("distributed report is explicit mock evidence with exact membership and no performance claim", "[distributed][cpu]") {
  auto r = distributed_request(3); r.ranks[0].physical_id = "mock\"device\\zero\n";
  auto plan = distributed_plan(r); d::MockExecution execution(plan); uint64_t clock = 1;
  REQUIRE(execution.begin(2, distributed_inputs(r), clock, 1000000).status == RNS8_SUCCESS);
  deliver(execution, 0, clock);
  const auto report = nlohmann::json::parse(d::report_json(execution));
  CHECK(report.at("schema") == "rns8_distributed_mock_v1");
  CHECK(report.at("hardware_executed") == false);
  CHECK(report.at("production_eligible") == false);
  CHECK(report.at("performance_timing").is_null());
  CHECK(report.at("expected_work").size() == plan->work().size());
  CHECK(report.at("proof_ready") == 1);
  CHECK(report.at("published_cells") == 0);
  CHECK(report.at("missing_work").size() == plan->work().size() - 1);
  CHECK(report.at("ranks")[0].at("physical_id") == r.ranks[0].physical_id);
  CHECK(report.at("basis_moduli")[1] == 255);
  CHECK(report.at("plan_identity") == plan->identity());
  CHECK(report.at("allocator_overhead_known") == false);
  CHECK(report.at("tiles").size() == plan->tiles().size());
  d::MockExecution missing(nullptr);
  CHECK(missing.begin(1, nullptr, 1, 2).status == RNS8_INVALID_ARGUMENT);
  CHECK_THROWS_AS(d::report_json(missing), std::invalid_argument);
}

TEST_CASE("distributed adaptive completion materializes missing planes before a changed-generation chain", "[distributed][cpu]") {
  auto first = distributed_request(3);
  first.semantics = RNS8_EXACT_WIDE_UNSIGNED;
  first.m = 2; first.n = first.k = first.lda = first.ldb = 1;
  first.prefix = 2; first.tile_rows = first.tile_cols = 1;
  first.magnitude_bound = 2000; first.tile_proofs = {{1, 100}, {2, 2000}};
  auto plan = distributed_plan(first);
  auto values = std::make_shared<const d::Inputs>(first.a, first.b,
      std::vector<uint64_t>{100, 2000}, std::vector<uint64_t>{1});
  d::MockExecution execution(plan); uint64_t clock = 1;
  std::vector<uint8_t> expanded{99};
  CHECK(execution.materialize(4, 1000, expanded).status == RNS8_BUSY);
  REQUIRE(execution.begin(1, values, clock, 1000000).status == RNS8_SUCCESS);
  finish(execution, clock);
  REQUIRE(execution.publish(++clock).status == RNS8_SUCCESS);
  const auto identity = execution.result_identity();
  CHECK(execution.materialize(1, 1000, expanded).status == RNS8_RANGE_ERROR);
  CHECK(expanded == std::vector<uint8_t>{99});
  CHECK(execution.materialize(4, 7, expanded).status == RNS8_WORKSPACE_TOO_SMALL);
  REQUIRE(execution.materialize(4, 8, expanded).status == RNS8_SUCCESS);
  for (uint32_t plane = 0; plane < 4; ++plane) {
    const uint32_t modulus = rns8::detail::garner::kConstants.moduli[plane];
    CHECK(expanded[plane * 2] == 100 % modulus);
    CHECK(expanded[plane * 2 + 1] == 2000 % modulus);
  }
  CHECK(execution.result_identity() == identity);
  auto second = first;
  second.operation = 72; second.prefix = 4; second.tile_proofs.clear();
  second.magnitude_bound = 2000000000; second.a = identity;
  second.b = {51, 1};
  auto next_inputs = std::make_shared<const d::Inputs>(second.a, second.b,
      std::vector<uint64_t>{100, 2000}, std::vector<uint64_t>{1000000});
  d::MockExecution next(distributed_plan(second));
  REQUIRE(next.begin(1, next_inputs, clock, 1000000).status == RNS8_SUCCESS);
  finish(next, clock);
  REQUIRE(next.publish(++clock).status == RNS8_SUCCESS);
  CHECK(*next.result() == distributed_oracle(second, *next_inputs));
}

TEST_CASE("distributed signed boundary and cancellation do not require safe full integer partial sums", "[distributed][cpu]") {
  auto r = distributed_request(2);
  r.semantics = RNS8_BOUNDED_I64;
  r.m = r.n = 1; r.k = 3; r.lda = 3; r.ldb = 1; r.prefix = 9;
  r.magnitude_bound = cpp_int(1) << 63;
  // The first products largely cancel after a >64-bit intermediate. Each
  // modular dot still reduces at every term, so no unsafe machine sum exists.
  auto inputs = std::make_shared<const d::Inputs>(r.a, r.b,
      std::vector<uint64_t>{uint64_t{1} << 63, uint64_t{1} << 63, uint64_t{1} << 63},
      std::vector<uint64_t>{uint64_t{1} << 63, uint64_t(INT64_MAX), 2});
  d::MockExecution execution(distributed_plan(r)); uint64_t clock = 1;
  REQUIRE(execution.begin(1, inputs, clock, 1000000).status == RNS8_SUCCESS);
  finish(execution, clock);
  REQUIRE(execution.publish(++clock).status == RNS8_SUCCESS);
  CHECK((*execution.result())[0] == cpp_int(INT64_MIN));
  CHECK(*execution.result() == distributed_oracle(r, *inputs));
}

TEST_CASE("distributed proven zero tiles retain explicit idle inventory without fabricated planes", "[distributed][cpu]") {
  auto r = distributed_request(4);
  r.m = r.n = r.k = r.lda = r.ldb = 1; r.prefix = 3;
  r.magnitude_bound = 0; r.tile_proofs = {{0, 0}};
  auto inputs = std::make_shared<const d::Inputs>(r.a, r.b,
      std::vector<uint64_t>{0}, std::vector<uint64_t>{UINT64_MAX});
  d::MockExecution execution(distributed_plan(r)); uint64_t clock = 1;
  REQUIRE(execution.plan()->work().empty());
  REQUIRE(execution.begin(1, inputs, clock, 1000).status == RNS8_SUCCESS);
  CHECK(execution.publish(++clock).status == RNS8_BUSY);
  for (uint32_t rank = 0; rank < 4; ++rank)
    REQUIRE(execution.acknowledge_rank(rank, ++clock).status == RNS8_SUCCESS);
  REQUIRE(execution.publish(++clock).status == RNS8_SUCCESS);
  CHECK((*execution.result())[0] == 0);
}

TEST_CASE("distributed pinned mock slot follows D2H then H2D completion before reuse", "[distributed][cpu]") {
  auto r = distributed_request(2); auto plan = distributed_plan(r);
  d::MockExecution execution(plan); uint64_t clock = 1;
  REQUIRE(execution.begin(1, distributed_inputs(r), clock, 1000000).status == RNS8_SUCCESS);
  std::vector<uint64_t> remote;
  for (const auto& work : plan->work()) if (work.transport == d::Transport::pinned_host) remote.push_back(work.id);
  REQUIRE(remote.size() >= 2);
  for (uint32_t i = 0; i < 2; ++i) {
    d::Fragment fragment;
    REQUIRE(execution.produce(remote[i], fragment).status == RNS8_SUCCESS);
    REQUIRE(execution.local_complete(std::move(fragment), ++clock).status == RNS8_SUCCESS);
  }
  CHECK(execution.transfer_complete(remote[0], ++clock).status == RNS8_INVALID_ARGUMENT);
  REQUIRE(execution.stage_host(remote[0], ++clock).status == RNS8_SUCCESS);
  CHECK(execution.report().pending_host_stage_work == remote[0]);
  CHECK(execution.stage_host(remote[1], ++clock).status == RNS8_BUSY);
  REQUIRE(execution.transfer_complete(remote[0], ++clock).status == RNS8_SUCCESS);
  CHECK(execution.report().visible == 0);
  REQUIRE(execution.stage_host(remote[1], ++clock).status == RNS8_SUCCESS);
  REQUIRE(execution.destination_visible(remote[0], ++clock).status == RNS8_SUCCESS);
  CHECK(execution.report().visible == 1);
}

TEST_CASE("independent commit checking charges native inputs to an otherwise idle owner", "[distributed][cpu]") {
  auto request=distributed_request(2);request.owner=1;request.m=request.n=request.k=request.lda=request.ldb=1;
  request.prefix=1;request.tile_rows=request.tile_cols=1;request.magnitude_bound=1;
  request.edges.push_back({0,1,d::Support::unknown,d::Support::supported});
  const auto ordinary=distributed_plan(request);REQUIRE(ordinary->resources().ranks[1].native_inputs==0);
  request.native_integrity=true;const auto checked=distributed_plan(request);
  CHECK(checked->resources().ranks[1].native_inputs==16);
  uint64_t aligned_native=0; REQUIRE(d::aligned_bytes(16,aligned_native));
  CHECK(checked->resources().ranks[1].peak==ordinary->resources().ranks[1].peak+aligned_native);
  request.ranks[1].memory_budget=checked->resources().ranks[1].peak-1;
  std::shared_ptr<const d::Plan> rejected;CHECK(d::Plan::create(request,rejected).status==RNS8_WORKSPACE_TOO_SMALL);
}
