#include "arithmetic_case.hpp"
#include <cstdio>
#include <cstdlib>
#include <iostream>

namespace {
using namespace rns8_fuzz;
void check(rns8_status actual, rns8_status expected, const char* phase) {
  if (actual != expected) {
    std::fprintf(stderr, "arithmetic-case-v1 %s: status %u expected %u\n", phase, unsigned(actual), unsigned(expected));
    std::abort();
  }
}
template<class T> T abi() {
  T value{}; value.struct_size = sizeof(value); value.abi_version = RNS8_ABI_VERSION; return value;
}
struct Owner {
  rns8_context* ctx = nullptr; rns8_plan* plan = nullptr; rns8_workspace* work = nullptr;
  rns8_matrix *a = nullptr, *b = nullptr, *c = nullptr;
  rns8_prepared_execution* prepared = nullptr;
  ~Owner() {
    check(rns8_destroy_prepared_execution(prepared), RNS8_SUCCESS, "destroy prepared");
    check(rns8_destroy_workspace(work), RNS8_SUCCESS, "destroy workspace");
    for (auto matrix : {c, b, a}) check(rns8_destroy_matrix(matrix), RNS8_SUCCESS, "destroy matrix");
    check(rns8_destroy_plan(plan), RNS8_SUCCESS, "destroy plan");
    check(rns8_destroy_context(ctx), RNS8_SUCCESS, "destroy context");
  }
};
template<class T> std::vector<T> padded(const std::vector<uint64_t>& raw, uint32_t rows,
                                       uint32_t cols, uint32_t padding, bool is_signed) {
  const std::size_t ld = cols + padding;
  std::vector<T> result((rows-1)*ld+cols+2, T(91));
  for (uint32_t row = 0; row < rows; ++row) for (uint32_t col = 0; col < cols; ++col) {
    const auto item = raw[std::size_t(row)*cols+col];
    result[1+row*ld+col] = is_signed ? T(signed_bits(item)) : T(item);
  }
  return result;
}
void pack(Owner& owner, const ArithmeticCase& value) {
  if (value.finite()) {
    auto a = padded<uint8_t>(value.a, value.m, value.k, value.padding_a, false);
    auto b = padded<uint8_t>(value.b, value.k, value.n, value.padding_b, false);
    check(rns8_pack_finite_u8(owner.ctx, owner.a, value.modulus, a.data()+1, value.k+value.padding_a, 0), RNS8_SUCCESS, "pack finite A");
    check(rns8_pack_finite_u8(owner.ctx, owner.b, value.modulus, b.data()+1, value.n+value.padding_b, 0), RNS8_SUCCESS, "pack finite B");
  } else if (value.signed_input()) {
    auto a = padded<int64_t>(value.a, value.m, value.k, value.padding_a, true);
    auto b = padded<int64_t>(value.b, value.k, value.n, value.padding_b, true);
    check(rns8_pack_i64(owner.ctx, owner.a, a.data()+1, value.k+value.padding_a, 0), RNS8_SUCCESS, "pack signed A");
    check(rns8_pack_i64(owner.ctx, owner.b, b.data()+1, value.n+value.padding_b, 0), RNS8_SUCCESS, "pack signed B");
  } else {
    auto a = padded<uint64_t>(value.a, value.m, value.k, value.padding_a, false);
    auto b = padded<uint64_t>(value.b, value.k, value.n, value.padding_b, false);
    check(rns8_pack_u64(owner.ctx, owner.a, a.data()+1, value.k+value.padding_a, 0), RNS8_SUCCESS, "pack unsigned A");
    check(rns8_pack_u64(owner.ctx, owner.b, b.data()+1, value.n+value.padding_b, 0), RNS8_SUCCESS, "pack unsigned B");
  }
}
rns8_status execute(Owner& owner, const ArithmeticCase& value) {
  if (value.finite()) return rns8_gemm_finite_u8(owner.ctx, owner.plan, value.modulus, owner.a, owner.b, owner.c, owner.work);
  if (value.wrap()) return rns8_gemm_wrap_u64(owner.ctx, owner.plan, owner.a, owner.b, owner.c, owner.work);
  return rns8_gemm_rns(owner.ctx, owner.plan, owner.a, owner.b, owner.c, owner.work);
}
template<class T, class Export> void compare(const ArithmeticCase& value, const Expected& expected,
                                            uint32_t limbs, Export export_values) {
  const std::size_t ld = value.n+value.padding_c;
  // Minimal final-row extent plus two guards; no assumed last-row padding.
  std::vector<T> output(((value.m-1)*ld+value.n)*limbs+2, T(91));
  if (value.operations & 2) {
    const auto untouched = output;
    check(export_values(output.data()+1, int64_t(value.n)-1), RNS8_INVALID_ARGUMENT, "invalid export leading dimension");
    if (output != untouched) { std::fputs("invalid export changed destination\n", stderr); std::abort(); }
  }
  check(export_values(output.data()+1, int64_t(ld)), RNS8_SUCCESS, "export");
  std::vector<T> wanted(output.size(), T(91));
  for (uint32_t row = 0; row < value.m; ++row) for (uint32_t col = 0; col < value.n; ++col) {
    cpp_int cell = expected.cells[std::size_t(row)*value.n+col];
    if (value.wide() && cell < 0) cell += cpp_int(1) << (64*limbs);
    for (uint32_t limb = 0; limb < limbs; ++limb) {
      const cpp_int piece = value.wide() ? cpp_int((cell >> (64*limb)) & UINT64_MAX) : cell;
      wanted[1+(row*ld+col)*limbs+limb] = piece.convert_to<T>();
    }
  }
  if (output != wanted) {
    const auto first = std::mismatch(output.begin(), output.end(), wanted.begin());
    std::cerr << "arithmetic-case-v1 differing storage element " << (first.first-output.begin())
              << " actual " << +*first.first << " expected " << +*first.second << '\n';
    std::abort();
  }
}
void compare(Owner& owner, const ArithmeticCase& value, const Expected& expected) {
  if (value.finite()) compare<uint8_t>(value, expected, 1, [&](auto* out, int64_t ld) {
    return rns8_export_finite_u8(owner.ctx, owner.plan, value.modulus, owner.c, out, ld); });
  else if (value.wide()) compare<uint64_t>(value, expected, 3, [&](auto* out, int64_t ld) {
    return value.signed_input() ? rns8_export_exact_wide_signed_limbs(owner.ctx, owner.plan, owner.c, out, ld, 3)
                               : rns8_export_exact_wide_unsigned_limbs(owner.ctx, owner.plan, owner.c, out, ld, 3); });
  else if (value.signed_input()) compare<int64_t>(value, expected, 1, [&](auto* out, int64_t ld) {
    return rns8_export_i64(owner.ctx, owner.plan, owner.c, out, ld); });
  else compare<uint64_t>(value, expected, 1, [&](auto* out, int64_t ld) {
    return value.wrap() ? rns8_export_wrap_u64(owner.ctx, owner.plan, owner.c, out, ld)
                        : rns8_export_u64(owner.ctx, owner.plan, owner.c, out, ld); });
}
void replay(ArithmeticCase value, const Expected& expected) {
  Owner owner;
  const auto backend = value.wrap() ? RNS8_BACKEND_WRAP64_BYTE_LIMB : RNS8_BACKEND_CPU_REFERENCE;
  auto options = abi<rns8_context_options>(); options.requested_backend = backend;
  check(rns8_create_context(-1, &options, &owner.ctx), RNS8_SUCCESS, "create context");
  auto desc = abi<rns8_gemm_desc>(); desc.requested_backend = backend; desc.semantics = value.semantics();
  desc.bound_kind = value.semantic == 0 ? RNS8_BOUND_GLOBAL_MAX_ABS :
                    value.semantic == 1 ? RNS8_BOUND_GLOBAL_MAX_UNSIGNED : RNS8_BOUND_NONE;
  desc.m = value.m; desc.n = value.n; desc.k = value.k; desc.max_prefix = value.prefix;
  desc.finite_modulus = value.modulus; desc.bound = expected.bound;
  if (value.wide()) desc.flags = RNS8_PLAN_FORCE_FIXED_PREFIX | RNS8_PLAN_EXACT_VALUE_BOUND;
  check(rns8_create_plan(owner.ctx, &desc, &owner.plan), RNS8_SUCCESS, "create plan");
  auto matrix = [&](uint32_t rows, uint32_t cols, uint32_t padding, rns8_matrix** out) {
    auto md = abi<rns8_matrix_desc>(); md.rows = rows; md.cols = cols; md.logical_ld = cols+padding;
    md.semantics = desc.semantics; md.bound_kind = desc.bound_kind;
    md.logical_layout = RNS8_LAYOUT_ROW_MAJOR; md.max_prefix = value.prefix;
    check(rns8_create_matrix(owner.ctx, &md, out), RNS8_SUCCESS, "create matrix");
  };
  matrix(value.m, value.k, value.padding_a, &owner.a);
  matrix(value.k, value.n, value.padding_b, &owner.b);
  matrix(value.m, value.n, value.padding_c, &owner.c);
  check(rns8_create_workspace(owner.ctx, owner.plan, &owner.work), RNS8_SUCCESS, "create workspace");
  pack(owner, value); check(execute(owner, value), RNS8_SUCCESS, "GEMM"); compare(owner, value, expected);
  if (value.operations & 1) {
    check(execute(owner, value), RNS8_SUCCESS, "repeat GEMM"); compare(owner, value, expected);
  }
  if (value.operations & 4) {
    check(rns8_prepare_gemm(owner.ctx, owner.plan, owner.a, owner.b, owner.c, owner.work, &owner.prepared), RNS8_SUCCESS, "prepare");
    std::fill(value.a.begin(), value.a.end(), 0); pack(owner, value);
    rns8_completion_token token{};
    check(rns8_execute_prepared(owner.prepared, &token), RNS8_INVALID_ARGUMENT, "stale prepared inputs");
    compare(owner, value, expected);
    check(rns8_refresh_prepared_inputs(owner.prepared), RNS8_SUCCESS, "refresh prepared");
    check(rns8_execute_prepared(owner.prepared, &token), RNS8_SUCCESS, "execute refreshed prepared");
    compare(owner, value, oracle(value));
  }
}
} // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, std::size_t size) {
  ArithmeticCase value;
  if (!decode(data, size, value)) return 0;
  const auto expected = oracle(value);
  // Outside-envelope cases are not valid arithmetic executions. Descriptor
  // admission/status mutations have a separate harness.
  if (expected.admitted) replay(std::move(value), expected);
  return 0;
}
