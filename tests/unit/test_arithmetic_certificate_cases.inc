namespace {
rns8::detail::arithmetic::Certificate certificate_fixture() {
  using namespace rns8::detail::arithmetic;
  Certificate c;
  c.basis = {251, 253};
  c.requires_range_guard = true;
  c.bound_known = true;
  c.output_bound = 100;
  c.instruction = {"cpu",
                   "signed-i8-dot",
                   "test-build",
                   "strict",
                   "test-lowering",
                   8,
                   32,
                   true,
                   true,
                   65536,
                   255,
                   128,
                   128};
  Node a;
  a.input = {-128, 127, 1, 256};
  a.bits = 8;
  a.equality = Equality::modular_value;
  c.dag.push_back(a);
  c.dag.push_back(a);
  Node d;
  d.operation = Operation::dot;
  d.equality = Equality::modular_value;
  d.left = 0;
  d.right = 1;
  d.count = 65536;
  c.dag.push_back(d);
  return c;
}
} // namespace
TEST_CASE("arithmetic identity uses standard SHA256 and length framed assumptions",
          "[certificate]") {
  CHECK(rns8::detail::sha256("") ==
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  CHECK(rns8::detail::sha256("abc") ==
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  CHECK(rns8::detail::sha256("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq") ==
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
  using namespace rns8::detail::arithmetic;
  const auto c = certificate_fixture();
  const auto good = verify(c);
  REQUIRE(good.admitted());
  CHECK(good.target == Verdict::not_run);
  CHECK(good.layout == Verdict::not_run);
  CHECK(good.floating_rewrite == Verdict::unsupported);
  auto changed = c;
  changed.instruction.compiler_flags += " -ffast-math";
  CHECK(verify(changed).sha256 != good.sha256);
  changed = c;
  std::reverse(changed.basis.begin(), changed.basis.end());
  CHECK(verify(changed).sha256 != good.sha256);
  changed = c;
  changed.instruction.instruction_evidence = "measured";
  CHECK(verify(changed).target == Verdict::not_run);
}
TEST_CASE("typed arithmetic rejects bad physical intermediates and domains", "[certificate]") {
  using namespace rns8::detail::arithmetic;
  const auto c = certificate_fixture();
  auto bad = c;
  bad.instruction.safe_k = 131072;
  CHECK(verify(bad).failure == "instruction_slab_plus_prior_overflow");
  bad = c;
  bad.instruction.signed_operands = false;
  CHECK_FALSE(verify(bad).admitted());
  bad = c;
  bad.basis = {251, 251};
  CHECK(verify(bad).failure == "ordered_basis_not_coprime");
  bad = c;
  bad.output_bound = 251 * 253;
  CHECK(verify(bad).failure == "strict_integer_range");
  bad = c;
  bad.dag[2].count = 131072;
  CHECK(verify(bad).failure == "physical_intermediate_overflow");
  bad = c;
  bad.dag[2].equality = Equality::integer_value;
  CHECK(verify(bad).failure == "arithmetic_equality_kind");
  bad = c;
  Node combine;
  combine.operation = Operation::add;
  combine.equality = Equality::modular_value;
  combine.left = 2;
  combine.right = 2;
  bad.dag.push_back(combine);
  CHECK(verify(bad).failure == "physical_intermediate_overflow");
  // Reducing both safe dots first makes the fusion safe, without cancellation.
  bad = c;
  Node reduce;
  reduce.operation = Operation::canonical_reduce;
  reduce.equality = Equality::modular_value;
  reduce.left = 2;
  reduce.modulus = 256;
  bad.dag.push_back(reduce);
  combine.left = 3;
  combine.right = 3;
  bad.dag.push_back(combine);
  CHECK(verify(bad).admitted());
  Node divide;
  divide.operation = Operation::exact_divide;
  divide.left = 3;
  divide.divisor = 2;
  bad.dag.push_back(divide);
  CHECK(verify(bad).failure == "unreduced_exact_division");
  bad = c;
  bad.dag[0].input.high = 128;
  CHECK_FALSE(verify(bad).admitted());
  bad = c;
  bad.dag[2].left = 3;
  CHECK_FALSE(verify(bad).admitted());
}
TEST_CASE("layout and unsigned modular nodes never imply integer equivalence", "[certificate]") {
  using namespace rns8::detail::arithmetic;
  auto c = certificate_fixture();
  Node n;
  n.operation = Operation::layout;
  n.equality = Equality::physical_layout;
  n.left = 0;
  n.bits = 8;
  n.source_extent = 4;
  n.destination_extent = 4;
  n.source_offsets = {0, 1, 2, 3};
  n.destination_offsets = {2, 0, 3, 1};
  c.dag.push_back(n);
  CHECK(verify(c).layout == Verdict::proven);
  c.dag.back().destination_offsets = {2, 0, 2, 1};
  CHECK(verify(c).failure == "layout_not_injective");
  c = certificate_fixture();
  c.representation = Representation::wrap_byte_limbs;
  c.dag.resize(2);
  for (auto& a : c.dag) {
    a.input = {0, 255, 1, 0};
    a.is_signed = false;
    a.equality = Equality::integer_value;
  }
  n = {};
  n.operation = Operation::wrap_multiply;
  n.equality = Equality::modular_value;
  n.left = 0;
  n.right = 1;
  n.bits = 8;
  n.is_signed = false;
  c.dag.push_back(n);
  CHECK(verify(c).admitted());
  n.operation = Operation::add;
  n.equality = Equality::integer_value;
  n.left = 2;
  n.right = 0;
  n.bits = 32;
  c.dag.push_back(n);
  CHECK(verify(c).failure == "arithmetic_domain_or_order");
}
TEST_CASE("public plans seal conditional arithmetic identity into cache and workspace",
          "[api][certificate]") {
  auto* ctx = create_cpu_context();
  for (const auto semantics : {RNS8_EXACT_WIDE_SIGNED, RNS8_EXACT_WIDE_UNSIGNED})
    for (const int64_t k : {65535, 65536, 65537}) {
      auto desc = exact_desc(semantics, 1, 1, k);
      rns8_plan* plan = nullptr;
      REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
      rns8_arithmetic_certificate_info info{};
      info.struct_size = sizeof(info);
      info.abi_version = RNS8_ABI_VERSION;
      REQUIRE(rns8_get_plan_arithmetic_certificate(plan, &info) == RNS8_SUCCESS);
      CHECK(info.algebraic_verdict == RNS8_PROOF_PROVEN);
      CHECK(info.finite_exact_verdict == RNS8_PROOF_PROVEN);
      CHECK(info.target_execution_verdict == RNS8_PROOF_NOT_RUN);
      CHECK(info.layout_verdict == RNS8_PROOF_NOT_RUN);
      CHECK(info.requires_current_range_guard == 1);
      CHECK(info.safe_k == static_cast<uint64_t>(std::min<int64_t>(k, 65536)));
      CHECK(std::string(info.certificate_sha256).size() == 64);
      CHECK(plan->backend_autotune_key.find(info.certificate_sha256) != std::string::npos);
      CHECK(plan->validated_contract->arithmetic_guard == plan->arithmetic_guard);
      const auto sha = std::string(info.certificate_sha256);
      rns8_plan* other = nullptr;
      REQUIRE(rns8_create_plan(ctx, &desc, &other) == RNS8_SUCCESS);
      REQUIRE(rns8_get_plan_arithmetic_certificate(other, &info) == RNS8_SUCCESS);
      CHECK(sha == info.certificate_sha256);
      CHECK(plan->validated_contract->canonical_workspace_key ==
            other->validated_contract->canonical_workspace_key);
      rns8_destroy_plan(other);
      rns8_destroy_plan(plan);
    }
  rns8_destroy_context(ctx);
}

TEST_CASE("CK certificates describe physical alphabets and post epilogue carries",
          "[certificate][ck]") {
  using namespace rns8::detail;
  using namespace arithmetic;
  std::vector<std::string> identities;
  for (const auto descriptor : {make_ck_implementation(0, 1, 64), make_ck_implementation(0, 0, 64),
                                make_ck_implementation(1, 1, 128)}) {
    REQUIRE(descriptor.valid());
    identities.push_back(descriptor.lowering_identity());
    for (uint32_t q = 2; q <= 256; ++q) {
      Certificate c;
      c.instruction.target = "host-adapter-only";
      c.instruction.opcode = descriptor.kernel_family();
      c.instruction.compiler_build = "fixture";
      c.instruction.compiler_flags = "fixture";
      c.instruction.lowering_identity = "fixture";
      REQUIRE(append_ck_arithmetic_contract(c, descriptor, {q}, descriptor.slab_k));
      const auto result = verify(c);
      INFO(q);
      INFO(descriptor.kernel_family());
      REQUIRE(result.admitted());
      CHECK(result.target == Verdict::not_run);
      CHECK(c.instruction.prior_abs == 0);
      const uint32_t dot_index = descriptor.signed_operands ? 2 : 4;
      const auto& dot = c.dag[dot_index];
      CHECK(dot.prior == UINT32_MAX);
      const auto& physical = result.intervals[dot.left];
      for (int value = -128; value <= 127; ++value) {
        if (value < -int(q / 2) || value > int((q - 1) / 2))
          continue;
        const auto byte =
            ck_encode_matrix_input(static_cast<int8_t>(value), q, descriptor.signed_operands);
        const int encoded =
            descriptor.signed_operands ? int(byte) : int(static_cast<uint8_t>(byte));
        CHECK(Integer(encoded) >= physical.low);
        CHECK(Integer(encoded) <= physical.high);
        CHECK((encoded - value) % int(q) == 0);
      }
      CHECK(result.intervals.back().low == -Integer(q / 2));
      CHECK(result.intervals.back().high == Integer((q - 1) / 2));
      auto bad = c;
      bad.instruction.signed_operands = !bad.instruction.signed_operands;
      CHECK_FALSE(verify(bad).admitted());
      if (!descriptor.signed_operands) {
        bad = c;
        bad.instruction.operand_encoding = OperandEncoding::centered_signed;
        CHECK_FALSE(verify(bad).admitted());
        bad = c;
        bad.dag[dot.left].is_signed = true;
        CHECK_FALSE(verify(bad).admitted());
        bad = c;
        bad.instruction.operand_encoding = OperandEncoding::centered_signed;
        bad.instruction.signed_operands = true;
        CHECK_FALSE(verify(bad).admitted());
      }
    }
    auto bad = descriptor;
    bad.signed_operands = !bad.signed_operands;
    CHECK_FALSE(bad.valid());
    bad = descriptor;
    bad.alphabet = CkOperandAlphabet::unknown;
    CHECK_FALSE(bad.valid());
    bad = descriptor;
    bad.family =
        descriptor.family == CkMatrixFamily::wmma ? CkMatrixFamily::xdl : CkMatrixFamily::wmma;
    CHECK_FALSE(bad.valid());
    bad = descriptor;
    bad.zero_initialized_dot = false;
    CHECK_FALSE(bad.valid());
  }
  CHECK(identities[0] != identities[1]);
  CHECK(identities[0] != identities[2]);
  CHECK_FALSE(make_ck_implementation(0, -1, 64).valid());
  CHECK_FALSE(make_ck_implementation(1, 0, 128).valid());
}

TEST_CASE("CK unsigned zero initialized dot boundary is independent of production cap",
          "[certificate][ck]") {
  using namespace rns8::detail;
  using namespace arithmetic;
  const auto d = make_ck_implementation(0, 0, 64);
  Certificate c;
  c.instruction.target = "host-boundary-proof";
  c.instruction.opcode = d.kernel_family();
  c.instruction.compiler_build = "fixture";
  c.instruction.compiler_flags = "fixture";
  c.instruction.lowering_identity = "boundary-experiment";
  REQUIRE(append_ck_arithmetic_contract(c, d, {256}, 32768));
  REQUIRE(verify(c).admitted());
  CHECK(d.slab_k == 32768);
  // Deliberately enlarge only this host proof; no dispatch cap is changed.
  c.instruction.safe_k = 33025;
  c.dag[4].count = 33025;
  auto result = verify(c);
  REQUIRE(result.admitted());
  CHECK(result.intervals[4].high == Integer(33025) * 255 * 255);
  CHECK(result.intervals[7].low == -256);
  CHECK(result.intervals[7].high == 254);
  c.instruction.safe_k = 33026;
  c.dag[4].count = 33026;
  CHECK_FALSE(verify(c).admitted());
  c.instruction.safe_k = 33025;
  c.dag[4].count = 33025;
  c.dag[4].prior = 6;
  CHECK_FALSE(verify(c).admitted()); // Future prior node cannot enter the dot.
}
