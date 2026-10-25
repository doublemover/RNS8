#pragma once
#include "rank23_slp.hpp"

namespace rns8::detail::rank23::october55 {
#include "rank23_october55_data.hpp"
inline constexpr auto u=slp::compile(raw_u()), v=slp::compile(raw_v()), w=slp::compile(raw_w());
static_assert(u.valid && v.valid && w.valid,"October 55-add DAG has a valid liveness schedule");
static_assert(u.source_additions==13 && v.source_additions==14 && w.source_additions==28,
              "October artifact must execute exactly 13+14+28 binary additions");
struct Coefficients { int u[23][9]{},v[23][9]{},w[9][23]{}; };
constexpr Coefficients expand() {
  Coefficients out{};
  const slp::Plan plans[]{u,v,w};
  for(int stage=0;stage<3;++stage) {
    const auto& plan=plans[stage];
    for(int input=0;input<plan.inputs;++input) {
      int slots[slp::capacity]{}; slots[input]=1;
      for(int i=0;i<plan.count;++i) {
        const auto op=plan.steps[i];
        const int value=slots[op.left]+(op.subtract?-slots[op.right]:slots[op.right]);
        slots[op.target]=op.negate?-value:value;
      }
      for(int output=0;output<plan.outputs;++output) {
        const int value=slots[plan.results[output]];
        if(stage==0)out.u[output][input]=value;
        else if(stage==1)out.v[output][input]=value;
        else out.w[output][input]=value;
      }
    }
  }
  return out;
}
inline constexpr auto coefficients=expand();
constexpr bool exact_ordered_tensor() {
  for(int output=0;output<9;++output)for(int a=0;a<9;++a)for(int b=0;b<9;++b) {
    int actual=0;
    for(int product=0;product<23;++product)
      actual+=coefficients.u[product][a]*coefficients.v[product][b]*coefficients.w[output][product];
    const int expected=a/3==output/3 && b%3==output%3 && a%3==b/3;
    if(actual!=expected)return false;
  }
  return true;
}
static_assert(exact_ordered_tensor(),"all 729 ordered coefficients equal classical 3x3 matrix multiplication");
// Integer-linear evaluation order bounds, before node-wise modular reduction.
constexpr int maximum_l1(const slp::Plan& plan) {
  int slots[slp::capacity][23]{};
  for(int i=0;i<plan.inputs;++i)slots[i][i]=1;
  int maximum=1;
  for(int i=0;i<plan.count;++i) {
    const auto op=plan.steps[i]; int values[23]{},norm=0;
    for(int j=0;j<plan.inputs;++j) {
      const int value=slots[op.left][j]+(op.subtract?-slots[op.right][j]:slots[op.right][j]);
      values[j]=op.negate?-value:value; norm+=values[j]<0?-values[j]:values[j];
    }
    for(int j=0;j<plan.inputs;++j)slots[op.target][j]=values[j];
    if(norm>maximum)maximum=norm;
  }
  return maximum;
}
static_assert(maximum_l1(u)==8 && maximum_l1(v)==6 && maximum_l1(w)==11,"exact artifact coefficient bounds");
} // namespace rns8::detail::rank23::october55
