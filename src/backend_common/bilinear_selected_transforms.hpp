#pragma once
#include "selected_linear_plan.hpp"
#include "rank23_october55.hpp"
#include "rank48_slp.hpp"

namespace rns8::detail {
constexpr LinearDag october55_input_dag(bool lhs) {
  const auto raw=lhs?rank23::october55::raw_u():rank23::october55::raw_v();
  LinearDag dag;dag.inputs=unsigned(raw.inputs);dag.count=unsigned(raw.count);dag.outputs=unsigned(raw.outputs);
  for(unsigned index=0;index<dag.count;++index) {
    const auto node=raw.assignments[index];const int sign=node.negate?-1:1;
    dag.nodes[index]={node.left,node.right,int8_t(sign),int8_t(node.subtract?-sign:sign),false};
  }
  for(unsigned index=0;index<dag.outputs;++index)dag.results[index]=unsigned(raw.results[index]);
  return dag;
}
constexpr LinearDag rank48_input_dag(bool lhs) {
  const auto& canonical=lhs?rank48::slp::u:rank48::slp::v;
  LinearDag dag;dag.inputs=canonical.inputs;dag.count=canonical.count;dag.outputs=canonical.outputs;
  // Recover an immutable SSA definition at EVERY assignment. Tracking versions
  // prevents the earlier producer of a reused slot from being pruned away.
  unsigned version[48]{};
  for(unsigned index=0;index<dag.inputs;++index)version[index]=index;
  for(unsigned index=0;index<dag.count;++index) {
    const auto step=canonical.steps[index];
    dag.nodes[index]={int(version[step.left]),step.operation<2?int(version[step.right]):-1,
      int8_t(step.operation==3?-1:1),int8_t(step.operation==1?-1:1),step.operation==2};
    version[step.target]=dag.inputs+index;
  }
  for(unsigned index=0;index<dag.outputs;++index)dag.results[index]=version[canonical.results[index]];
  return dag;
}
template<bool Rank48,bool Lhs,unsigned Product>
inline constexpr auto selected_bilinear_plan=select_linear_output(Rank48?rank48_input_dag(Lhs):october55_input_dag(Lhs),Product);
}
