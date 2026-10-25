#pragma once
#include "sha256.hpp"
#include <boost/multiprecision/cpp_int.hpp>
#include <algorithm>
#include <array>
#include <numeric>
#include <string>
#include <vector>
namespace rns8::detail::arithmetic {
using Integer=boost::multiprecision::cpp_int;
enum class Verdict:uint32_t { unknown=0,proven=1,refuted=2,unsupported=3,not_run=4 };
enum class Equality:uint32_t { integer_value=0,modular_value=1,physical_layout=2 };
enum class Operation:uint32_t { input=0,add=1,multiply=2,dot=3,canonical_reduce=4,centered_reduce=5,exact_divide=6,wrap_reduce=7,layout=8,wrap_add=9,wrap_multiply=10,wrap_dot=11 };
enum class Representation:uint32_t { centered_rns_i8=1,canonical_finite_u8=2,wrap_byte_limbs=3,native_exact_limbs=4,signed_finite_i4=5 };
struct Interval {Integer low=0,high=0,multiple=1;uint32_t modulus=0,wrapped_bits=0;};
struct Node {
  Operation operation=Operation::input;Equality equality=Equality::integer_value;
  uint32_t left=UINT32_MAX,right=UINT32_MAX,prior=UINT32_MAX,bits=32;bool is_signed=true;
  uint64_t count=1,divisor=1;uint32_t modulus=0;
  Interval input;
  std::vector<uint64_t> source_offsets,destination_offsets;
  uint64_t source_extent=0,destination_extent=0;
};
struct Instruction {
  std::string target,opcode,compiler_build,compiler_flags,lowering_identity;
  uint32_t operand_bits=8,accumulator_bits=32;bool signed_operands=true,signed_accumulator=true;
  uint64_t safe_k=0,prior_abs=0;uint32_t lhs_abs=0,rhs_abs=0;
  // Receipts are identities only. This host checker never sets target execution
  // or layout verdicts from an evidence string, target name or format label.
  std::string instruction_evidence,layout_evidence;
};
struct Certificate {
  uint32_t version=1;Representation representation=Representation::centered_rns_i8;
  std::vector<uint32_t> basis;std::vector<Node> dag;Instruction instruction;
  uint32_t semantics=0;bool requires_range_guard=false,signed_output=false;
  Integer output_bound=0;bool bound_known=false;
};
struct Checked {
  Verdict algebra=Verdict::unknown,finite_exact=Verdict::unknown,layout=Verdict::not_run,target=Verdict::not_run;
  Verdict floating_rewrite=Verdict::unsupported;
  std::string sha256,lowering_sha256,failure;
  std::vector<Interval> intervals;
  uint32_t failed_node=UINT32_MAX;uint64_t safe_k=0;bool requires_range_guard=false;
  bool admitted()const{return algebra==Verdict::proven && finite_exact==Verdict::proven;}
};
inline Integer gcd(Integer a,Integer b){if(a<0)a=-a;if(b<0)b=-b;while(b!=0){Integer r=a%b;a=b;b=r;}return a;}
inline bool fits(const Interval& x,uint32_t bits,bool sign) {
  if(!bits || bits>256 || x.low>x.high || x.multiple<=0)return false;
  const Integer limit=Integer(1)<<(bits-(sign?1:0));return x.low>=(sign?-limit:Integer(0)) && x.high<limit;
}
inline Interval multiply(const Interval& a,const Interval& b) {
  std::array<Integer,4> endpoints{{a.low*b.low,a.low*b.high,a.high*b.low,a.high*b.high}};
  return {*std::min_element(endpoints.begin(),endpoints.end()),*std::max_element(endpoints.begin(),endpoints.end()),a.multiple*b.multiple,a.modulus?a.modulus:b.modulus};
}
// Length-prefixed canonical integer/text encoding avoids delimiter ambiguity.
inline std::string canonical(const Certificate& c) {
  std::string out="rns8-arithmetic-runtime-v1";
  const auto text=[&](const std::string& value){out+='|';out+=std::to_string(value.size());out+=':';out+=value;};
  const auto integer=[&](const Integer& value){text(value.str());};
  integer(c.version);integer(uint32_t(c.representation));integer(c.semantics);integer(c.requires_range_guard);integer(c.signed_output);integer(c.bound_known);integer(c.output_bound);
  integer(c.basis.size());for(auto p:c.basis)integer(p);
  const auto& i=c.instruction;for(const auto* s:{&i.target,&i.opcode,&i.compiler_build,&i.compiler_flags,&i.lowering_identity,&i.instruction_evidence,&i.layout_evidence})text(*s);
  for(uint64_t v:{uint64_t(i.operand_bits),uint64_t(i.accumulator_bits),uint64_t(i.signed_operands),uint64_t(i.signed_accumulator),i.safe_k,i.prior_abs,uint64_t(i.lhs_abs),uint64_t(i.rhs_abs)})integer(v);
  integer(c.dag.size());for(const auto& n:c.dag){for(uint64_t v:{uint64_t(n.operation),uint64_t(n.equality),uint64_t(n.left),uint64_t(n.right),uint64_t(n.prior),uint64_t(n.bits),uint64_t(n.is_signed),n.count,n.divisor,uint64_t(n.modulus)})integer(v);
    integer(n.input.low);integer(n.input.high);integer(n.input.multiple);integer(n.input.modulus);integer(n.input.wrapped_bits);integer(n.source_extent);integer(n.destination_extent);
    integer(n.source_offsets.size());for(auto v:n.source_offsets)integer(v);integer(n.destination_offsets.size());for(auto v:n.destination_offsets)integer(v);}
  return out;
}
inline Checked verify(const Certificate& c) {
  Checked result;result.requires_range_guard=c.requires_range_guard;result.safe_k=c.instruction.safe_k;
  const auto fail=[&](const char* reason,uint32_t node=UINT32_MAX){result.failure=reason;result.failed_node=node;result.algebra=Verdict::refuted;result.finite_exact=Verdict::unknown;return result;};
  if(uint32_t(c.representation)<1 || uint32_t(c.representation)>5 || c.version!=1 || c.dag.empty() || c.dag.size()>4096 || c.basis.size()>20)return fail("certificate_structure");
  const auto& instruction=c.instruction;
  if(instruction.target.empty() || instruction.opcode.empty() || instruction.compiler_build.empty() || instruction.compiler_flags.empty() || instruction.lowering_identity.empty())return fail("instruction_identity");
  Integer product=1;for(auto p:c.basis){if(p<2 || p>256 || gcd(product,p)!=1)return fail("ordered_basis_not_coprime");product*=p;}
  if(c.bound_known && (c.output_bound<0 || (c.requires_range_guard && (c.basis.empty() || product<=c.output_bound*(c.signed_output?2:1)))))return fail("strict_integer_range");
  if(c.representation==Representation::centered_rns_i8 || c.representation==Representation::canonical_finite_u8 || c.representation==Representation::signed_finite_i4) {
    const bool nibble=c.representation==Representation::signed_finite_i4;
    if(instruction.operand_bits!=(nibble?4u:8u) || instruction.accumulator_bits!=32 || !instruction.signed_operands || !instruction.signed_accumulator ||
        !instruction.safe_k || instruction.lhs_abs>(nibble?8u:128u) || instruction.rhs_abs>(nibble?8u:128u))return fail("signed_i8_instruction_contract");
    if(Integer(instruction.safe_k)*instruction.lhs_abs*instruction.rhs_abs+instruction.prior_abs>INT32_MAX)return fail("instruction_slab_plus_prior_overflow");
  }
  result.intervals.reserve(c.dag.size());bool has_layout=false;
  for(uint32_t index=0;index<c.dag.size();++index){const auto& n=c.dag[index];Interval value;
    if(!n.bits || n.bits>256)return fail("physical_width",index);
    const auto input=[&](uint32_t id)->const Interval*{return id<index?&result.intervals[id]:nullptr;};
    const auto* a=input(n.left);const auto* b=input(n.right);
    if(n.operation==Operation::input){if(n.left!=UINT32_MAX || n.right!=UINT32_MAX || n.prior!=UINT32_MAX)return fail("input_arity",index);value=n.input;
      if(value.wrapped_bits || value.modulus==1 || (value.modulus && n.equality!=Equality::modular_value) || (!value.modulus && n.equality!=Equality::integer_value))return fail("input_domain",index);}
    else if(n.operation==Operation::wrap_add || n.operation==Operation::wrap_multiply || n.operation==Operation::wrap_dot){
      if(!a || !b || a->low<0 || b->low<0 || a->modulus || b->modulus || (a->wrapped_bits && a->wrapped_bits!=n.bits) || (b->wrapped_bits && b->wrapped_bits!=n.bits) || n.is_signed || n.equality!=Equality::modular_value || !n.count || n.prior!=UINT32_MAX)return fail("defined_unsigned_arithmetic",index);
      value={0,(Integer(1)<<n.bits)-1,1,0,n.bits};
    }
    else if(n.operation==Operation::add || n.operation==Operation::multiply || n.operation==Operation::dot){
      if(!a || !b || a->wrapped_bits || b->wrapped_bits || (a->modulus && b->modulus && a->modulus!=b->modulus))return fail("arithmetic_domain_or_order",index);
      if(n.operation==Operation::add)value={a->low+b->low,a->high+b->high,gcd(a->multiple,b->multiple),a->modulus?a->modulus:b->modulus};
      else {value=multiply(*a,*b);if(n.operation==Operation::dot){if(!n.count)return fail("dot_count",index);
          value.low*=n.count;value.high*=n.count;value.low=std::min(Integer(0),value.low);value.high=std::max(Integer(0),value.high);}}
      if(n.equality!=(value.modulus?Equality::modular_value:Equality::integer_value))return fail("arithmetic_equality_kind",index);
      if(n.prior!=UINT32_MAX){const auto* prior=input(n.prior);if(n.operation!=Operation::dot || !prior || prior->modulus!=value.modulus)return fail("accumulator_carry_domain",index);
        value.low+=prior->low;value.high+=prior->high;value.multiple=gcd(value.multiple,prior->multiple);}
    } else {
      if(!a || n.right!=UINT32_MAX || n.prior!=UINT32_MAX)return fail("unary_arity_or_order",index);value=*a;
      if(n.operation==Operation::canonical_reduce || n.operation==Operation::centered_reduce){
        if(a->wrapped_bits || n.modulus<2 || (a->modulus && a->modulus!=n.modulus) || n.equality!=Equality::modular_value)return fail("named_reduction_domain",index);
        value.low=n.operation==Operation::centered_reduce?-Integer(n.modulus/2):Integer(0);
        value.high=n.operation==Operation::centered_reduce?Integer((n.modulus-1)/2):Integer(n.modulus-1);
        value.modulus=n.modulus;value.multiple=gcd(a->multiple,n.modulus);
      }else if(n.operation==Operation::exact_divide){
        if(!n.divisor || a->modulus || a->wrapped_bits || a->multiple%n.divisor!=0 || n.equality!=Equality::integer_value)return fail("unreduced_exact_division",index);
        value.low/=n.divisor;value.high/=n.divisor;value.multiple/=n.divisor;
      }else if(n.operation==Operation::wrap_reduce){
        if(a->modulus || n.equality!=Equality::modular_value || n.is_signed)return fail("unsigned_wrap_domain",index);
        value.low=0;value.high=(Integer(1)<<n.bits)-1;value.multiple=1;value.modulus=0;value.wrapped_bits=n.bits;
      }else if(n.operation==Operation::layout){
        if(n.equality!=Equality::physical_layout || !n.source_extent || !n.destination_extent || n.source_offsets.empty() || n.source_offsets.size()!=n.destination_offsets.size() || n.source_offsets.size()>1048576)return fail("layout_domain",index);
        auto source=n.source_offsets,destination=n.destination_offsets;std::sort(source.begin(),source.end());std::sort(destination.begin(),destination.end());
        if(source.back()>=n.source_extent || destination.back()>=n.destination_extent || std::adjacent_find(source.begin(),source.end())!=source.end() || std::adjacent_find(destination.begin(),destination.end())!=destination.end())return fail("layout_not_injective",index);
        has_layout=true;
      }else return fail("unsupported_proof_node",index);
    }
    if(!fits(value,n.bits,n.is_signed))return fail("physical_intermediate_overflow",index);
    result.intervals.push_back(std::move(value));
  }
  result.algebra=Verdict::proven;result.finite_exact=Verdict::proven;
  result.layout=has_layout?Verdict::proven:Verdict::not_run;
  result.lowering_sha256=rns8::detail::sha256(c.instruction.lowering_identity);
  result.sha256=rns8::detail::sha256(canonical(c));return result;
}
struct Guard { Certificate certificate; Checked checked; explicit Guard(Certificate c):certificate(std::move(c)),checked(verify(certificate)){} };
}
