#pragma once

#include <cstdint>

#include "rns8_bench_types.hpp"

namespace rns8::bench {

bool finite_benchmark_semantics(BenchSemantics semantics);
bool bounded_benchmark_semantics(BenchSemantics semantics);
bool exact_wide_benchmark_semantics(BenchSemantics semantics);
bool rns_chain_benchmark_semantics(BenchSemantics semantics);
bool rns_residue_chain_requested(const Args& args);
bool residue_current_output_mode(const Args& args);
bool residue_chain_final_export_requested(const Args& args);
bool residue_chain_independent_final_export_requested(const Args& args);
bool exact_wide_export_status_check_required(const Args& args);

bool valid_finite_field_modulus(uint16_t modulus);
bool valid_finite_modulus(BenchSemantics semantics, uint16_t modulus);

const char* next_op_hint_name(NextOpHint hint);
const char* semantics_name(BenchSemantics semantics);
rns8_semantics c_semantics(BenchSemantics semantics);
const char* c_semantics_name(rns8_semantics semantics);
rns8_bound_kind global_bound_kind(BenchSemantics semantics);
rns8_bound_kind bound_kind(const Args& args);
const char* bound_kind_name(const Args& args);
const char* bound_kind_name(rns8_bound_kind bound_kind);
const char* bound_mode_name(BoundMode mode);

}  // namespace rns8::bench
