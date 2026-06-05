#include "rns8_bench_modes.hpp"

namespace rns8::bench {

bool finite_benchmark_semantics(BenchSemantics semantics) {
  return semantics == BenchSemantics::FiniteRingU8 || semantics == BenchSemantics::FiniteFieldU8;
}

bool bounded_benchmark_semantics(BenchSemantics semantics) {
  return semantics == BenchSemantics::BoundedI64 || semantics == BenchSemantics::BoundedU64;
}

bool exact_wide_benchmark_semantics(BenchSemantics semantics) {
  return semantics == BenchSemantics::ExactWideSigned || semantics == BenchSemantics::ExactWideUnsigned;
}

bool rns_chain_benchmark_semantics(BenchSemantics semantics) {
  return bounded_benchmark_semantics(semantics) || exact_wide_benchmark_semantics(semantics);
}

bool rns_residue_chain_requested(const Args& args) {
  return rns_chain_benchmark_semantics(args.semantics) && args.residue_chain_length > 1;
}

bool residue_current_output_mode(const Args& args) {
  return rns_residue_chain_requested(args) &&
         !(args.residue_chain_final_export || args.residue_chain_independent_final_export);
}

bool residue_chain_final_export_requested(const Args& args) {
  return rns_residue_chain_requested(args) &&
         (args.residue_chain_final_export || args.residue_chain_independent_final_export);
}

bool residue_chain_independent_final_export_requested(const Args& args) {
  return rns_residue_chain_requested(args) && args.residue_chain_independent_final_export;
}

bool exact_wide_export_status_check_required(const Args& args) {
  if (args.semantics == BenchSemantics::ExactWideUnsigned) {
    return args.exact_wide_limb_count < 3;
  }
  if (args.semantics == BenchSemantics::ExactWideSigned) {
    return args.exact_wide_limb_count < 3;
  }
  return true;
}

bool valid_finite_field_modulus(uint16_t modulus) {
  if (modulus < 2 || modulus > 251) {
    return false;
  }
  for (uint16_t divisor = 2; divisor * divisor <= modulus; ++divisor) {
    if (modulus % divisor == 0) {
      return false;
    }
  }
  return true;
}

bool valid_finite_modulus(BenchSemantics semantics, uint16_t modulus) {
  if (semantics == BenchSemantics::FiniteRingU8) {
    return modulus >= 2 && modulus <= 256;
  }
  if (semantics == BenchSemantics::FiniteFieldU8) {
    return valid_finite_field_modulus(modulus);
  }
  return true;
}

const char* next_op_hint_name(NextOpHint hint) {
  switch (hint) {
    case NextOpHint::Auto:
      return "auto";
    case NextOpHint::FinalExport:
      return "final-export";
    case NextOpHint::RnsGemm:
      return "rns-gemm";
    case NextOpHint::NativeGemm:
      return "native-gemm";
    case NextOpHint::NativeToRns:
      return "native-to-rns";
    case NextOpHint::ReuseB:
      return "reuse-b";
  }
  return "unknown";
}

const char* semantics_name(BenchSemantics semantics) {
  switch (semantics) {
    case BenchSemantics::BoundedI64:
      return "bounded_i64";
    case BenchSemantics::BoundedU64:
      return "bounded_u64";
    case BenchSemantics::ExactWideSigned:
      return "exact_wide_signed";
    case BenchSemantics::ExactWideUnsigned:
      return "exact_wide_unsigned";
    case BenchSemantics::WrapU64Mod2_64:
      return "wrap_u64_mod_2_64";
    case BenchSemantics::FiniteRingU8:
      return "finite_ring_u8";
    case BenchSemantics::FiniteFieldU8:
      return "finite_field_u8";
  }
  return "unknown";
}

rns8_semantics c_semantics(BenchSemantics semantics) {
  switch (semantics) {
    case BenchSemantics::BoundedI64:
      return RNS8_BOUNDED_I64;
    case BenchSemantics::BoundedU64:
      return RNS8_BOUNDED_U64;
    case BenchSemantics::ExactWideSigned:
      return RNS8_EXACT_WIDE_SIGNED;
    case BenchSemantics::ExactWideUnsigned:
      return RNS8_EXACT_WIDE_UNSIGNED;
    case BenchSemantics::WrapU64Mod2_64:
      return RNS8_WRAP_U64_MOD_2_64;
    case BenchSemantics::FiniteRingU8:
      return RNS8_FINITE_RING_U8;
    case BenchSemantics::FiniteFieldU8:
      return RNS8_FINITE_FIELD_U8;
  }
  return RNS8_BOUNDED_I64;
}

const char* c_semantics_name(rns8_semantics semantics) {
  switch (semantics) {
    case RNS8_BOUNDED_I64:
      return "bounded_i64";
    case RNS8_BOUNDED_U64:
      return "bounded_u64";
    case RNS8_EXACT_WIDE_SIGNED:
      return "exact_wide_signed";
    case RNS8_EXACT_WIDE_UNSIGNED:
      return "exact_wide_unsigned";
    case RNS8_WRAP_U64_MOD_2_64:
      return "wrap_u64_mod_2_64";
    case RNS8_FINITE_RING_U8:
      return "finite_ring_u8";
    case RNS8_FINITE_FIELD_U8:
      return "finite_field_u8";
    default:
      return "unknown";
  }
}

rns8_bound_kind global_bound_kind(BenchSemantics semantics) {
  switch (semantics) {
    case BenchSemantics::BoundedI64:
      return RNS8_BOUND_GLOBAL_MAX_ABS;
    case BenchSemantics::BoundedU64:
      return RNS8_BOUND_GLOBAL_MAX_UNSIGNED;
    case BenchSemantics::WrapU64Mod2_64:
    case BenchSemantics::ExactWideSigned:
    case BenchSemantics::ExactWideUnsigned:
    case BenchSemantics::FiniteRingU8:
    case BenchSemantics::FiniteFieldU8:
      return RNS8_BOUND_NONE;
  }
  return RNS8_BOUND_NONE;
}

rns8_bound_kind bound_kind(const Args& args) {
  if (args.bound_mode == BoundMode::PerTile) {
    switch (args.semantics) {
      case BenchSemantics::BoundedI64:
        return RNS8_BOUND_PER_TILE_MAX_ABS;
      case BenchSemantics::BoundedU64:
        return RNS8_BOUND_PER_TILE_MAX_UNSIGNED;
      case BenchSemantics::ExactWideSigned:
      case BenchSemantics::ExactWideUnsigned:
      case BenchSemantics::WrapU64Mod2_64:
      case BenchSemantics::FiniteRingU8:
      case BenchSemantics::FiniteFieldU8:
        return RNS8_BOUND_NONE;
    }
  }
  return global_bound_kind(args.semantics);
}

const char* bound_kind_name(const Args& args) {
  if (args.bound_mode == BoundMode::PerTile) {
    switch (args.semantics) {
      case BenchSemantics::BoundedI64:
        return "per_tile_max_abs";
      case BenchSemantics::BoundedU64:
        return "per_tile_max_unsigned";
      case BenchSemantics::ExactWideSigned:
      case BenchSemantics::ExactWideUnsigned:
      case BenchSemantics::WrapU64Mod2_64:
      case BenchSemantics::FiniteRingU8:
      case BenchSemantics::FiniteFieldU8:
        return "none";
    }
  }
  switch (args.semantics) {
    case BenchSemantics::BoundedI64:
      return "global_max_abs";
    case BenchSemantics::BoundedU64:
      return "global_max_unsigned";
    case BenchSemantics::ExactWideSigned:
    case BenchSemantics::ExactWideUnsigned:
    case BenchSemantics::WrapU64Mod2_64:
    case BenchSemantics::FiniteRingU8:
    case BenchSemantics::FiniteFieldU8:
      return "none";
  }
  return "unknown";
}

const char* bound_kind_name(rns8_bound_kind bound_kind) {
  switch (bound_kind) {
    case RNS8_BOUND_NONE:
      return "none";
    case RNS8_BOUND_GLOBAL_MAX_ABS:
      return "global_max_abs";
    case RNS8_BOUND_GLOBAL_MAX_UNSIGNED:
      return "global_max_unsigned";
    case RNS8_BOUND_PER_TILE_MAX_ABS:
      return "per_tile_max_abs";
    case RNS8_BOUND_PER_TILE_MAX_UNSIGNED:
      return "per_tile_max_unsigned";
    case RNS8_BOUND_INPUT_RANGE_AND_K:
      return "input_range_and_k";
    default:
      return "unknown";
  }
}

const char* bound_mode_name(BoundMode mode) {
  switch (mode) {
    case BoundMode::Global:
      return "global";
    case BoundMode::PerTile:
      return "per_tile";
  }
  return "unknown";
}

}  // namespace rns8::bench
