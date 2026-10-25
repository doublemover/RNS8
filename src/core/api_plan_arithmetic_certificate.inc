// One exact-integer check at plan construction. The guard proves physical
// bounds for this declared lowering; it does not attest generated ISA/layout.
static std::shared_ptr<const arithmetic::Guard> make_plan_arithmetic_guard(const rns8_plan& plan) {
  using namespace arithmetic;
  Certificate c;
  c.semantics=uint32_t(plan.desc.semantics);
  c.signed_output=plan.desc.semantics==RNS8_BOUNDED_I64 || plan.desc.semantics==RNS8_EXACT_WIDE_SIGNED;
  c.requires_range_guard=uses_rns_storage(plan.desc.semantics);
  if(c.requires_range_guard)for(uint32_t p=0;p<plan.prefix;++p)c.basis.push_back(garner::kConstants.moduli[p]);
  c.bound_known=c.requires_range_guard && plan.desc.bound_kind!=RNS8_BOUND_NONE;
  c.output_bound=plan.desc.bound;
  if(is_per_tile_bound_kind(plan.desc.bound_kind))for(auto bound:plan.tile_bounds)c.output_bound=std::max(c.output_bound,Integer(bound));
  auto& i=c.instruction;
  i.target=plan.backend_target_id.empty()?"unresolved-target":plan.backend_target_id;
  i.opcode=plan.backend_selected_kernel;
#ifdef RNS8_BUILD_MANIFEST_JSON
  i.compiler_build=RNS8_BUILD_MANIFEST_JSON;
#else
  i.compiler_build="unavailable-direct-build-manifest";
#endif
  // The complete manifest (including flags) is bound, never an inferred default.
  i.compiler_flags=i.compiler_build;
  i.lowering_identity="ordinary-plan-arithmetic-v1;backend="+std::to_string(plan.backend)+";kernel="+i.opcode+
    ";accumulator="+plan.backend_accumulator_type+";K="+std::to_string(plan.desc.k)+
    ";epilogue="+plan.backend_epilogue_mode+";library="+plan.backend_library_version;
  i.safe_k=plan.backend_accumulator_k_block_size;
  i.lhs_abs=static_cast<uint32_t>(plan.backend_accumulator_max_lhs_abs);
  i.rhs_abs=static_cast<uint32_t>(plan.backend_accumulator_max_rhs_abs);
  const auto input=[&](Integer lo,Integer hi,uint32_t bits,bool sign,uint32_t modulus=0){
    Node n;n.input={std::move(lo),std::move(hi),1,modulus};n.bits=bits;n.is_signed=sign;
    n.equality=modulus?Equality::modular_value:Equality::integer_value;
    c.dag.push_back(std::move(n));return static_cast<uint32_t>(c.dag.size()-1);
  };
  if(plan.desc.semantics==RNS8_WRAP_U64_MOD_2_64) {
    c.representation=Representation::wrap_byte_limbs;i.signed_operands=false;i.signed_accumulator=false;i.accumulator_bits=64;
    const auto a=input(0,255,8,false),b=input(0,255,8,false);
    Node dot;dot.left=a;dot.right=b;dot.bits=64;dot.is_signed=false;dot.count=static_cast<uint64_t>(plan.desc.k);
    if(plan.backend==RNS8_BACKEND_HIP_DIRECT && wrap64_hip_uses_u32_accumulator(plan.desc.k)) {
      // Longest low diagonal contains eight byte products per K. Carry is
      // widened to uint64 before addition and extraction, never uint32.
      dot.operation=Operation::dot;dot.count*=8;dot.bits=32;i.accumulator_bits=32;
      c.dag.push_back(dot);
      const auto carry=input(0,UINT32_MAX,64,false);
      Node add;add.operation=Operation::add;add.left=2;add.right=carry;add.bits=64;add.is_signed=false;c.dag.push_back(add);
    } else {dot.operation=Operation::wrap_dot;dot.equality=Equality::modular_value;c.dag.push_back(dot);}
  } else if(plan.backend==RNS8_BACKEND_HIP_VECTOR_ALU_INT64) {
    c.representation=Representation::native_exact_limbs;i.operand_bits=64;i.accumulator_bits=192;i.signed_operands=c.signed_output;i.signed_accumulator=c.signed_output;
    const Integer hi=c.signed_output?(Integer(1)<<63)-1:(Integer(1)<<64)-1;
    const Integer lo=c.signed_output?-(Integer(1)<<63):Integer(0);
    const auto a=input(lo,hi,64,c.signed_output),b=input(lo,hi,64,c.signed_output);
    Node n;n.operation=Operation::dot;n.left=a;n.right=b;n.count=static_cast<uint64_t>(plan.desc.k);n.bits=192;n.is_signed=c.signed_output;c.dag.push_back(n);
  } else {
    c.representation=uses_finite_storage(plan.desc.semantics)?Representation::canonical_finite_u8:Representation::centered_rns_i8;
    i.prior_abs=255;
    std::vector<uint32_t> moduli=c.basis;
    if(uses_finite_storage(plan.desc.semantics))moduli={plan.desc.finite_modulus};
    for(auto modulus:moduli) {
      const auto a=input(-Integer(modulus/2),Integer((modulus-1)/2),8,true,modulus);
      const auto b=input(-Integer(modulus/2),Integer((modulus-1)/2),8,true,modulus);
      const auto prior=input(0,modulus-1,32,true,modulus);
      Node dot;dot.operation=Operation::dot;dot.equality=Equality::modular_value;dot.left=a;dot.right=b;dot.prior=prior;dot.count=i.safe_k;c.dag.push_back(dot);
      Node reduce;reduce.operation=Operation::canonical_reduce;reduce.equality=Equality::modular_value;reduce.left=static_cast<uint32_t>(c.dag.size()-1);reduce.modulus=modulus;reduce.bits=8;reduce.is_signed=false;c.dag.push_back(reduce);
      // The invariant after every slab is again a canonical residue. The
      // already-checked prior node covers all later slab carries, not just 0.
    }
  }
  return std::make_shared<const Guard>(std::move(c));
}
