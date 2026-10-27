#include <catch2/catch_test_macros.hpp>
#include "core/distributed_hip.hpp"
#include "backend_hip_direct/hip_backend.hpp"
#include "backend_common/distributed_integrity.hpp"
#include "core/hip_resources.hpp"
#include <cstdlib>
#include <string>

namespace {
namespace d = rns8::detail::distributed;
d::Request live_request() {
  d::Request r;
  r.operation = 501; r.m = 3; r.n = 5; r.k = 65; r.lda = 67; r.ldb = 8;
  r.prefix = 20; r.tile_rows = 2; r.tile_cols = 3; r.magnitude_bound = d::Integer(1) << 148;
  r.a = {17, 3}; r.b = {29, 8}; r.certificate_sha256 = std::string(64, 'a'); r.layout_sha256 = std::string(64, 'b');
  r.host_budget = uint64_t{1} << 30; r.pinned_budget = 1u << 20;
  r.host_storage = d::HostStorage::limbs_only;
  return r;
}
std::shared_ptr<const d::Plan> make_plan(const d::Request& r) {
  std::shared_ptr<const d::Plan> plan; REQUIRE(d::Plan::create(r, plan)); return plan;
}
std::shared_ptr<const d::Inputs> make_inputs(const d::Request& r, bool narrow = false, bool zero_rows = false, bool full_bounded = false) {
  std::vector<uint64_t> a((r.m - 1) * r.lda + r.k, 99), b((r.k - 1) * r.ldb + r.n, 99);
  const uint64_t edges[] = {0, 1, UINT64_MAX, uint64_t{1} << 63, uint64_t(INT64_MAX), 17};
  for (uint64_t row = 0; row < r.m; ++row) for (uint64_t k = 0; k < r.k; ++k)
    a[row * r.lda + k] = zero_rows && row < 2 ? 0 : full_bounded ?
        (r.semantics == RNS8_BOUNDED_U64 ? UINT64_MAX : (row + k + 1) % 2 ? uint64_t{1} + (uint64_t{1} << 63) : uint64_t(INT64_MAX)) : narrow ?
        (r.semantics == RNS8_BOUNDED_U64 || row % 2 == 0 ? 1 : UINT64_MAX) : edges[(row + k) % 6];
  for (uint64_t k = 0; k < r.k; ++k) for (uint64_t col = 0; col < r.n; ++col)
    b[k * r.ldb + col] = full_bounded && r.semantics == RNS8_BOUNDED_U64 ? uint64_t(k == 0) : narrow ? 1 : edges[(k + 2 * col + 1) % 6];
  return std::make_shared<const d::Inputs>(r.a, r.b, std::move(a), std::move(b));
}
void mock_inventory(d::Request& r) {
  r.transport = d::Transport::local;
  d::Rank rank; rank.physical_id = "cpu-contract-only"; rank.target = "cpu-contract-only";
  rank.memory_budget = uint64_t{1} << 30; r.ranks = {rank};
}
}

TEST_CASE("distributed native admission proves actual uniqueness before HIP work", "[distributed][cpu]") {
  auto r = live_request(); mock_inventory(r);
  auto plan = make_plan(r); auto inputs = make_inputs(r);
  REQUIRE(d::validate_inputs(*plan, *inputs));
  r.magnitude_bound = 1; plan = make_plan(r);
  CHECK(d::validate_inputs(*plan, *inputs)); // a sufficient basis lets export check cancellation/range
  r.prefix = 1; plan = make_plan(r);
  CHECK(d::validate_inputs(*plan, *inputs).status == RNS8_RANGE_ERROR);
  auto stale = std::make_shared<const d::Inputs>(d::ValueIdentity{17, 4}, r.b, inputs->lhs(), inputs->rhs());
  CHECK(d::validate_inputs(*plan, *stale).status == RNS8_INVALID_ARGUMENT);
  if (!d::HipExecution::compiled()) {
    std::unique_ptr<d::HipExecution> execution;
    CHECK(d::HipExecution::create(plan, inputs, execution).status == RNS8_UNSUPPORTED_BACKEND);
    CHECK_FALSE(execution);
  }
}
TEST_CASE("distributed pack capacity follows bounded slots rather than total output tiles", "[distributed][cpu]") {
  auto r = live_request(); mock_inventory(r); r.k = 65537; r.lda = r.k;
  r.m = r.n = 32; r.ldb = r.n; r.tile_rows = r.tile_cols = 16;
  const auto small = make_plan(r);
  r.m = r.n = 128; r.ldb = r.n;
  const auto large = make_plan(r);
  CHECK(large->resources().ranks[0].packed_inputs == small->resources().ranks[0].packed_inputs);
  CHECK(large->resources().ranks[0].local_outputs > small->resources().ranks[0].local_outputs);
  r.pipeline_slots = 0; std::shared_ptr<const d::Plan> rejected;
  CHECK(d::Plan::create(r, rejected).status == RNS8_INVALID_ARGUMENT);
  r.pipeline_slots = 4; CHECK(d::Plan::create(r, rejected).status == RNS8_INVALID_ARGUMENT);
}
TEST_CASE("distributed integrity distinguishes native checks from range and invisible errors", "[distributed][cpu]") {
  auto r=live_request();mock_inventory(r);r.m=r.n=r.k=r.lda=r.ldb=1;
  r.native_integrity=true;auto plan=make_plan(r);
  d::Inputs inputs(r.a,r.b,{7},{9});uint64_t value[3]={63,0,0};
  CHECK(d::reference_integrity_cell(*plan,inputs,0,0,value)==0);
  value[0]=64;CHECK(d::reference_integrity_cell(*plan,inputs,0,0,value)==12);
  value[0]=63+257*263;CHECK(d::reference_integrity_cell(*plan,inputs,0,0,value)==0);
  CHECK(d::distributed_commit_status(1)==RNS8_RANGE_ERROR);
  CHECK(d::distributed_commit_status(4)==RNS8_VERIFICATION_FAILED);
  CHECK(d::distributed_commit_status(8)==RNS8_VERIFICATION_FAILED);
  CHECK(d::distributed_commit_status(0)==RNS8_SUCCESS);
  r.native_integrity=false;auto unchecked=make_plan(r);CHECK(plan->identity()!=unchecked->identity());
}
static void check_real_modulus_shards(uint64_t k,d::Axis axis=d::Axis::modulus,bool custom_basis=false) {
  if (!d::HipExecution::compiled()) SKIP("direct HIP not compiled");
  rns8_device_info info{};
  if (rns8::detail::hip_direct_probe(0, info) != RNS8_SUCCESS) SKIP("HIP device unavailable");
  // Default is the explicit one-device kernel/owner differential. A future
  // qualification host sets 0,1 (or other visible ordinals), never fabricated ranks.
  std::vector<uint32_t> ordinals;
  const char* configured = std::getenv("RNS8_DISTRIBUTED_DEVICES");
  std::string selected = configured ? configured : "0";
  for (size_t begin = 0; begin < selected.size();) {
    const auto end = selected.find(',', begin); const auto item = selected.substr(begin, end - begin);
    REQUIRE(!item.empty()); REQUIRE(item.find_first_not_of("0123456789") == std::string::npos);
    const auto ordinal = std::stoull(item); REQUIRE(ordinal <= UINT32_MAX); ordinals.push_back(uint32_t(ordinal));
    if (end == std::string::npos) break;
    begin = end + 1; REQUIRE(begin < selected.size());
  }
  REQUIRE(!ordinals.empty());
  const char* requested_transport = std::getenv("RNS8_DISTRIBUTED_TRANSPORT");
  const bool peer = requested_transport && std::string(requested_transport) == "peer";
  REQUIRE((!requested_transport || peer || std::string(requested_transport) == "pinned"));
  for (auto semantics : {RNS8_BOUNDED_I64, RNS8_BOUNDED_U64, RNS8_EXACT_WIDE_SIGNED, RNS8_EXACT_WIDE_UNSIGNED}) for (bool ragged : {false, true}) {
    CAPTURE(semantics, selected, peer, ragged);
    auto r = live_request(); r.semantics = semantics; r.k = k; r.lda = k + 2;r.axis=axis;
    if(custom_basis){r.basis={9,16,25,251,241,239,233,229,227,223,211,199,197,193,191,181,179,173,167,163};r.magnitude_bound=d::Integer(1)<<135;}
    if(axis==d::Axis::contraction)r.k_fragments=3;
    const bool narrow = semantics == RNS8_BOUNDED_I64 || semantics == RNS8_BOUNDED_U64;
    if (narrow) { r.prefix = k < 128 ? (custom_basis?2:1) : 9; r.magnitude_bound = r.k; }
    const bool full_bounded = narrow && ragged;
    if (full_bounded) { r.prefix = 20; r.magnitude_bound = semantics == RNS8_BOUNDED_I64 ? d::Integer(INT64_MAX) : d::Integer(UINT64_MAX); }
    REQUIRE(d::HipExecution::discover(ordinals, r.ranks, r.edges));
    if(axis==d::Axis::hybrid){r.rank_grid_rows=r.ranks.size()==4?2:1;r.rank_grid_cols=uint32_t(r.ranks.size())/r.rank_grid_rows;}
    r.transport = ordinals.size() == 1 ? d::Transport::local : peer ? d::Transport::peer : d::Transport::pinned_host;
    if (peer && ordinals.size() > 1) for (const auto& edge : r.edges)
      if (edge.peer != d::Support::supported) SKIP("explicit peer transport unsupported on selected devices");
    if (ragged) r.tile_proofs = {{0, 0}, {0, 0}, {r.prefix, r.magnitude_bound}, {r.prefix, r.magnitude_bound}};
    r.native_integrity=true;
    auto plan = make_plan(r); auto inputs = make_inputs(r, narrow, ragged, full_bounded);
    std::unique_ptr<d::HipExecution> execution;
    REQUIRE(d::HipExecution::create(plan, inputs, execution));
    REQUIRE(execution->arithmetic_certificates().size() == ordinals.size());
    std::vector<uint64_t> observed(r.m * (r.n + 2) * 3, 99);
    REQUIRE(execution->submit(1, UINT64_C(30000000000)));
    CHECK(execution->export_limbs(1, observed.data(), r.n + 2).status == RNS8_BUSY);
    CHECK(observed == std::vector<uint64_t>(observed.size(), 99));
    d::HipProgress progress;
    REQUIRE(execution->wait(1, UINT64_C(30000000000), progress)); REQUIRE(progress.operation);
    REQUIRE(progress.output_current); REQUIRE(progress.local_completed == plan->work().size());
    CHECK(progress.integrity_requested);CHECK(progress.integrity_completed);CHECK(progress.integrity_failure_mask==0);
    REQUIRE(progress.transfers_queued == plan->work().size());
    REQUIRE(execution->export_limbs(1, observed.data(), r.n + 2));
    CHECK(execution->export_limbs(1, const_cast<uint64_t*>(inputs->lhs().data()), r.n + 2).status == RNS8_INVALID_ARGUMENT);
    const bool signed_value = semantics == RNS8_BOUNDED_I64 || semantics == RNS8_EXACT_WIDE_SIGNED;
    const d::Integer mask = (d::Integer(1) << 192) - 1;
    for (uint64_t row = 0; row < r.m; ++row) for (uint64_t col = 0; col < r.n + 2; ++col) {
      d::Integer exact = 0;
      if (col < r.n) for (uint64_t k = 0; k < r.k; ++k) {
        const auto decode = [signed_value](uint64_t bits) -> d::Integer {
          return signed_value && bits > uint64_t(INT64_MAX) ? d::Integer(bits) - (d::Integer(1) << 64) : d::Integer(bits);
        };
        exact += decode(inputs->lhs()[row * r.lda + k]) * decode(inputs->rhs()[k * r.ldb + col]);
      }
      exact &= mask;
      for (unsigned limb = 0; limb < 3; ++limb)
        CHECK(observed[(row * (r.n + 2) + col) * 3 + limb] == (col < r.n ? uint64_t((exact >> (64 * limb)) & UINT64_MAX) : 99));
    }
    const uint32_t materialized_prefix = r.prefix == 1 ? 2 : 20;
    CHECK(execution->materialize(1, materialized_prefix, r.m * r.n * materialized_prefix - 1).status == RNS8_WORKSPACE_TOO_SMALL);
    REQUIRE(execution->materialize(1, materialized_prefix, r.m * r.n * materialized_prefix));
    REQUIRE(execution->retain(1));
    int residue_device = -1; const void* residue_pointer = nullptr; uint32_t residue_prefix = 0;
    const rns8::detail::BasisPacket* residue_basis=nullptr;
    REQUIRE(execution->resident_residues(1, residue_device, residue_pointer, residue_prefix,residue_basis));
    REQUIRE(residue_basis!=nullptr);CHECK(residue_basis->identity==plan->basis_packet().identity);
    CHECK(residue_prefix == materialized_prefix);
    std::vector<int8_t> residues(size_t(r.m * r.n * residue_prefix));
    REQUIRE(rns8::detail::hip_direct_copy_device_to_host(residue_device, residues.data(), residue_pointer, residues.size()) == RNS8_SUCCESS);
    for (uint32_t plane = 0; plane < residue_prefix; ++plane) for (uint64_t row = 0; row < r.m; ++row)
      for (uint64_t col = 0; col < r.n; ++col) {
        d::Integer exact = 0;
        for (unsigned limb = 0; limb < 3; ++limb) exact += d::Integer(observed[(row * (r.n + 2) + col) * 3 + limb]) << (64 * limb);
        if (signed_value && (observed[(row * (r.n + 2) + col) * 3 + 2] >> 63)) exact -= d::Integer(1) << 192;
        const auto modulus = residue_basis->moduli[plane]; d::Integer reduced = exact % modulus;
        if (reduced < 0) reduced += modulus;
        if (reduced >= (modulus + 1) / 2) reduced -= modulus;
        CHECK(residues[plane * r.m * r.n + row * r.n + col] == reduced.convert_to<int>());
      }
    int device = -1; const void* resident = nullptr; REQUIRE(execution->resident_limbs(1, device, resident));
    CHECK(device == int(r.ranks[r.owner].ordinal)); CHECK(resident != nullptr);
    CHECK(execution->submit(2, UINT64_C(30000000000)).status == RNS8_BUSY);
    CHECK(execution->close().status == RNS8_BUSY); REQUIRE(execution->release(1));
    REQUIRE(execution->submit(2, UINT64_C(30000000000))); REQUIRE(execution->cancel(2));
    REQUIRE(execution->wait(2, UINT64_C(30000000000), progress)); CHECK(progress.operation.status == RNS8_CANCELED);
    CHECK_FALSE(progress.output_current); REQUIRE(execution->close());
  }
}
TEST_CASE("real HIP modulus shards gather and publish one exact generation", "[distributed][hip]") {
  check_real_modulus_shards(65);
}
TEST_CASE("real HIP distributed packing and GEMM cross the safe K slab boundary", "[distributed][hip]") {
  check_real_modulus_shards(65537);
}
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
extern "C" int rns8_hip_distributed_integrity(const uint64_t*,const uint64_t*,const uint64_t*,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,bool,int*,void*);
TEST_CASE("HIP distributed integrity compares complete limbs against independent native products", "[distributed][hip]") {
  using namespace rns8::detail;rns8_device_info info{};
  if(hip_direct_probe(0,info)!=RNS8_SUCCESS)SKIP("HIP device unavailable");
  hip_direct_device_buffer a,b,c,status;
  REQUIRE(a.allocate(0,8)==RNS8_SUCCESS);REQUIRE(b.allocate(0,8)==RNS8_SUCCESS);
  REQUIRE(c.allocate(0,24)==RNS8_SUCCESS);REQUIRE(status.allocate(0,4)==RNS8_SUCCESS);
  for(bool sign:{false,true})for(uint64_t error:{uint64_t{0},uint64_t{1},uint64_t{257*263}}) {
    const uint64_t av=sign?UINT64_MAX:7,bv=9;
    const d::Integer integer=(sign?d::Integer(-9):d::Integer(63))+error;
    d::Integer encoded=integer&((d::Integer(1)<<192)-1);uint64_t limbs[3]{};
    for(unsigned i=0;i<3;++i)limbs[i]=uint64_t((encoded>>(64*i))&UINT64_MAX);
    const uint32_t zero=0;
    REQUIRE(hip_direct_copy_host_to_device(0,a.get(),&av,8)==RNS8_SUCCESS);
    REQUIRE(hip_direct_copy_host_to_device(0,b.get(),&bv,8)==RNS8_SUCCESS);
    REQUIRE(hip_direct_copy_host_to_device(0,c.get(),limbs,24)==RNS8_SUCCESS);
    REQUIRE(hip_direct_copy_host_to_device(0,status.get(),&zero,4)==RNS8_SUCCESS);
    for(auto* buffer:{&a,&b,&c,&status})buffer->mark_pending();
    const auto launched=rns8_hip_distributed_integrity(static_cast<const uint64_t*>(a.get()),static_cast<const uint64_t*>(b.get()),
        static_cast<const uint64_t*>(c.get()),1,1,1,1,1,sign,static_cast<int*>(status.get()),nullptr);
    REQUIRE(hip_direct_synchronize(0)==RNS8_SUCCESS);
    for(auto* buffer:{&a,&b,&c,&status})buffer->mark_completed();REQUIRE(launched==0);
    uint32_t flags=0;REQUIRE(hip_direct_copy_device_to_host(0,&flags,status.get(),4)==RNS8_SUCCESS);
    CHECK(flags==(error==1?12u:0u));
  }
}
#endif

TEST_CASE("real HIP explicit output and contraction distributions retain complete exact results", "[distributed][hip]") {
  for(auto axis:{d::Axis::rows,d::Axis::columns,d::Axis::hybrid,d::Axis::contraction})check_real_modulus_shards(65,axis);
}
TEST_CASE("real HIP distributed ordered bases retain small composite and reordered channel identities", "[distributed][hip]") {
  for(auto axis:{d::Axis::modulus,d::Axis::rows,d::Axis::columns,d::Axis::hybrid,d::Axis::contraction})check_real_modulus_shards(65,axis,true);
}
