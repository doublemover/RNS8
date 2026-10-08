#include "core/api_internal.hpp"

using namespace rns8::detail::api;

rns8_status rns8_lift_exact_wide_cpu(rns8_context* ctx, rns8_matrix* matrix, uint32_t target_prefix,
                                     uint64_t max_staged_residue_bytes) {
  return guard_api([&]() -> rns8_status {
    if (!ctx || !matrix || ctx->backend != matrix->backend) return RNS8_INVALID_ARGUMENT;
    // Lifting is owned by the resident backend. Never stage a hidden device download.
    if (ctx->backend != RNS8_BACKEND_CPU_REFERENCE) return RNS8_UNSUPPORTED_BACKEND;
    if (!rns8::detail::exact_wide_semantics(matrix->desc.semantics) || target_prefix == 0 ||
        target_prefix > RNS8_MAX_SUPPORTED_PREFIX || target_prefix > matrix->prefix ||
        matrix->desc.max_prefix != matrix->prefix ||
        rns8::detail::validate_matrix_desc(matrix->desc, matrix->prefix) != RNS8_SUCCESS) {
      return RNS8_INVALID_ARGUMENT;
    }
    std::size_t cells = 0, capacity = 0;
    if (!matrix_cell_count(matrix->desc.rows, matrix->desc.cols, cells) ||
        !rns_residue_count(matrix->desc.rows, matrix->desc.cols, matrix->prefix, capacity)) {
      return RNS8_RANGE_ERROR;
    }
    if (!rns_matrix_storage_matches(*matrix, ctx->backend, matrix->desc.rows, matrix->desc.cols,
                                    matrix->prefix) ||
        !rns_residue_state_current_for_backend(*matrix, ctx->backend)) {
      return RNS8_INVALID_ARGUMENT;
    }
    const uint32_t source_prefix = matrix->exact_range_prefix;
    if (source_prefix == 0 || source_prefix > matrix->prefix) return RNS8_RANGE_ERROR;
    const auto product = rns8::detail::modulus_product(source_prefix);
    // The source range must already identify an integer uniquely. Additional
    // moduli cannot recover an earlier alias, even if the target range is large.
    if (!rns8::detail::exact_range_fits(matrix->desc.semantics, matrix->exact_max_magnitude, product)) {
      return RNS8_RANGE_ERROR;
    }
    if (target_prefix <= source_prefix) return RNS8_SUCCESS;

    // Full capacity was checked above, so both products are representable.
    const std::size_t source_count = cells * source_prefix;
    const std::size_t staged_count = cells * (target_prefix - source_prefix);
    if (staged_count > max_staged_residue_bytes) return RNS8_WORKSPACE_TOO_SMALL;
    std::vector<int8_t> staged;
    if (staged_count > staged.max_size()) return RNS8_RANGE_ERROR;
    staged.resize(staged_count);
    std::vector<int8_t> cell_residues(source_prefix);
    const bool signed_value = matrix->desc.semantics == RNS8_EXACT_WIDE_SIGNED;
    const boost::multiprecision::cpp_int threshold = (product + 1) / 2;
    for (std::size_t cell = 0; cell < cells; ++cell) {
      for (uint32_t p = 0; p < source_prefix; ++p) {
        const int value = matrix->residues[cells * p + cell];
        const int modulus = rns8::detail::kDefaultModuli[p];
        if (value < -(modulus / 2) || value > (modulus - 1) / 2) return RNS8_INVALID_ARGUMENT;
        cell_residues[p] = static_cast<int8_t>(value);
      }
      auto value = rns8::detail::reconstruct_canonical(cell_residues, source_prefix);
      if (signed_value && value >= threshold) value -= product;
      if (boost::multiprecision::abs(value) > matrix->exact_max_magnitude) return RNS8_RANGE_ERROR;
      for (uint32_t p = source_prefix; p < target_prefix; ++p) {
        staged[cells * (p - source_prefix) + cell] =
            rns8::detail::centered_residue(value, rns8::detail::kDefaultModuli[p]);
      }
    }
    // No allocations, CRT work, or other throwing operations after the first
    // resident write. Existing planes and logical-value metadata stay intact.
    std::copy(staged.begin(), staged.end(), matrix->residues.begin() + source_count);
    matrix->exact_range_prefix = target_prefix;
    return RNS8_SUCCESS;
  });
}
