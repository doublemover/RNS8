namespace {
rns8::detail::arithmetic::Certificate certificate_fixture() {
  using namespace rns8::detail::arithmetic;
  Certificate c;c.basis={251,253};c.requires_range_guard=true;c.bound_known=true;c.output_bound=100;
  c.instruction={"cpu","signed-i8-dot","test-build","strict","test-lowering",8,32,true,true,65536,255,128,128};
  Node a;a.input={-128,127,1,256};a.bits=8;a.equality=Equality::modular_value;c.dag.push_back(a);c.dag.push_back(a);
  Node d;d.operation=Operation::dot;d.equality=Equality::modular_value;d.left=0;d.right=1;d.count=65536;c.dag.push_back(d);
  return c;
}
}
TEST_CASE("arithmetic identity uses standard SHA256 and length framed assumptions","[certificate]") {
  CHECK(rns8::detail::sha256("")=="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  CHECK(rns8::detail::sha256("abc")=="ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  CHECK(rns8::detail::sha256("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq")=="248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
  using namespace rns8::detail::arithmetic;
  const auto c=certificate_fixture();const auto good=verify(c);REQUIRE(good.admitted());
  CHECK(good.target==Verdict::not_run);CHECK(good.layout==Verdict::not_run);CHECK(good.floating_rewrite==Verdict::unsupported);
  auto changed=c;changed.instruction.compiler_flags+=" -ffast-math";CHECK(verify(changed).sha256!=good.sha256);
  changed=c;std::reverse(changed.basis.begin(),changed.basis.end());CHECK(verify(changed).sha256!=good.sha256);
  changed=c;changed.instruction.instruction_evidence="measured";CHECK(verify(changed).target==Verdict::not_run);
}
TEST_CASE("typed arithmetic rejects bad physical intermediates and domains","[certificate]") {
  using namespace rns8::detail::arithmetic;const auto c=certificate_fixture();
  auto bad=c;bad.instruction.safe_k=131072;CHECK(verify(bad).failure=="instruction_slab_plus_prior_overflow");
  bad=c;bad.instruction.signed_operands=false;CHECK_FALSE(verify(bad).admitted());
  bad=c;bad.basis={251,251};CHECK(verify(bad).failure=="ordered_basis_not_coprime");
  bad=c;bad.output_bound=251*253;CHECK(verify(bad).failure=="strict_integer_range");
  bad=c;bad.dag[2].count=131072;CHECK(verify(bad).failure=="physical_intermediate_overflow");
  bad=c;bad.dag[2].equality=Equality::integer_value;CHECK(verify(bad).failure=="arithmetic_equality_kind");
  bad=c;Node combine;combine.operation=Operation::add;combine.equality=Equality::modular_value;combine.left=2;combine.right=2;
  bad.dag.push_back(combine);CHECK(verify(bad).failure=="physical_intermediate_overflow");
  // Reducing both safe dots first makes the fusion safe, without cancellation.
  bad=c;Node reduce;reduce.operation=Operation::canonical_reduce;reduce.equality=Equality::modular_value;reduce.left=2;reduce.modulus=256;
  bad.dag.push_back(reduce);combine.left=3;combine.right=3;bad.dag.push_back(combine);CHECK(verify(bad).admitted());
  Node divide;divide.operation=Operation::exact_divide;divide.left=3;divide.divisor=2;bad.dag.push_back(divide);
  CHECK(verify(bad).failure=="unreduced_exact_division");
  bad=c;bad.dag[0].input.high=128;CHECK_FALSE(verify(bad).admitted());
  bad=c;bad.dag[2].left=3;CHECK_FALSE(verify(bad).admitted());
}
TEST_CASE("layout and unsigned modular nodes never imply integer equivalence","[certificate]") {
  using namespace rns8::detail::arithmetic;auto c=certificate_fixture();
  Node n;n.operation=Operation::layout;n.equality=Equality::physical_layout;n.left=0;n.bits=8;n.source_extent=4;n.destination_extent=4;
  n.source_offsets={0,1,2,3};n.destination_offsets={2,0,3,1};c.dag.push_back(n);
  CHECK(verify(c).layout==Verdict::proven);c.dag.back().destination_offsets={2,0,2,1};CHECK(verify(c).failure=="layout_not_injective");
  c=certificate_fixture();c.representation=Representation::wrap_byte_limbs;c.dag.resize(2);
  for(auto& a:c.dag){a.input={0,255,1,0};a.is_signed=false;a.equality=Equality::integer_value;}
  n={};n.operation=Operation::wrap_multiply;n.equality=Equality::modular_value;n.left=0;n.right=1;n.bits=8;n.is_signed=false;c.dag.push_back(n);CHECK(verify(c).admitted());
  n.operation=Operation::add;n.equality=Equality::integer_value;n.left=2;n.right=0;n.bits=32;c.dag.push_back(n);
  CHECK(verify(c).failure=="arithmetic_domain_or_order");
}
TEST_CASE("public plans seal conditional arithmetic identity into cache and workspace","[api][certificate]") {
  auto* ctx=create_cpu_context();
  for(const auto semantics:{RNS8_EXACT_WIDE_SIGNED,RNS8_EXACT_WIDE_UNSIGNED})for(const int64_t k:{65535,65536,65537}) {
    auto desc=exact_desc(semantics,1,1,k);rns8_plan* plan=nullptr;
    REQUIRE(rns8_create_plan(ctx,&desc,&plan)==RNS8_SUCCESS);
    rns8_arithmetic_certificate_info info{};info.struct_size=sizeof(info);info.abi_version=RNS8_ABI_VERSION;
    REQUIRE(rns8_get_plan_arithmetic_certificate(plan,&info)==RNS8_SUCCESS);
    CHECK(info.algebraic_verdict==RNS8_PROOF_PROVEN);CHECK(info.finite_exact_verdict==RNS8_PROOF_PROVEN);
    CHECK(info.target_execution_verdict==RNS8_PROOF_NOT_RUN);CHECK(info.layout_verdict==RNS8_PROOF_NOT_RUN);
    CHECK(info.requires_current_range_guard==1);CHECK(info.safe_k==static_cast<uint64_t>(std::min<int64_t>(k,65536)));
    CHECK(std::string(info.certificate_sha256).size()==64);CHECK(plan->backend_autotune_key.find(info.certificate_sha256)!=std::string::npos);
    CHECK(plan->validated_contract->arithmetic_guard==plan->arithmetic_guard);
    const auto sha=std::string(info.certificate_sha256);rns8_plan* other=nullptr;REQUIRE(rns8_create_plan(ctx,&desc,&other)==RNS8_SUCCESS);
    REQUIRE(rns8_get_plan_arithmetic_certificate(other,&info)==RNS8_SUCCESS);CHECK(sha==info.certificate_sha256);
    CHECK(plan->validated_contract->canonical_workspace_key==other->validated_contract->canonical_workspace_key);
    rns8_destroy_plan(other);rns8_destroy_plan(plan);
  }
  rns8_destroy_context(ctx);
}
