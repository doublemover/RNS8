#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <type_traits>
#include <vector>

#include "../support/allocation_fault.hpp"
#include "../support/noninteractive_errors.hpp"
#include "core/internal.hpp"
#include "rns8/rns8.hpp"

namespace {
using boost::multiprecision::cpp_int;
namespace fault = rns8::test::allocation_fault;

const char* current_case = "setup";
std::size_t current_point = std::numeric_limits<std::size_t>::max();

void report_termination() noexcept {
  const auto result = fault::end();
  std::fprintf(stderr, "FAILED: terminate case=%s point=%zu calls=%zu bytes=%zu triggered=%d\n", current_case,
               current_point, result.calls, result.bytes, result.triggered);
#if defined(_WIN32)
  void* frames[32]{};
  const auto count = CaptureStackBackTrace(0, 32, frames, nullptr);
  const auto base = reinterpret_cast<std::uintptr_t>(GetModuleHandleW(nullptr));
  for (USHORT i = 0; i < count; ++i)
    std::fprintf(stderr, "frame[%u]=0x%llx executable_rva=0x%llx\n", static_cast<unsigned>(i),
                 static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(frames[i])),
                 static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(frames[i]) - base));
#endif
  std::_Exit(86);  // A terminating test is always a failure, never an accepted allocation error.
}

void require(bool condition, const char* message) {
  if (!condition) {
    std::fprintf(stderr, "FAILED: %s\n", message);
    std::exit(2);
  }
}

rns8_gemm_desc gemm_desc(rns8_semantics semantics, int64_t m, int64_t n, int64_t k, bool fixed = false) {
  rns8_gemm_desc d{};
  d.struct_size = sizeof(d);
  d.abi_version = RNS8_ABI_VERSION;
  d.semantics = semantics;
  d.bound_kind = RNS8_BOUND_NONE;
  d.requested_backend = RNS8_BACKEND_CPU_REFERENCE;
  d.m = m;
  d.n = n;
  d.k = k;
  d.max_prefix = fixed ? 17 : 20;
  d.flags = fixed ? RNS8_PLAN_FORCE_FIXED_PREFIX : 0;
  return d;
}

rns8_matrix_desc matrix_desc(rns8_semantics semantics, int64_t rows, int64_t cols) {
  rns8_matrix_desc d{};
  d.struct_size = sizeof(d);
  d.abi_version = RNS8_ABI_VERSION;
  d.semantics = semantics;
  d.bound_kind = RNS8_BOUND_NONE;
  d.rows = rows;
  d.cols = cols;
  d.logical_ld = cols;
  d.logical_layout = RNS8_LAYOUT_ROW_MAJOR;
  d.max_prefix = 20;
  return d;
}

bool same_axis(const rns8_exact_axis_range& a, const rns8_exact_axis_range& b) {
  return a.max_magnitude == b.max_magnitude && a.absolute_sum == b.absolute_sum && a.sum == b.sum &&
         a.minimum == b.minimum && a.maximum == b.maximum && a.deviation_sum == b.deviation_sum &&
         a.sum_known == b.sum_known;
}

bool same_axes(const std::vector<rns8_exact_axis_range>& a, const std::vector<rns8_exact_axis_range>& b) {
  if (a.size() != b.size()) return false;
  for (std::size_t i = 0; i < a.size(); ++i)
    if (!same_axis(a[i], b[i])) return false;
  return true;
}

struct MatrixSnapshot {
  rns8_matrix value;
  const int8_t* residues;
  const rns8_exact_axis_range* rows;
  const rns8_exact_axis_range* cols;
  explicit MatrixSnapshot(const rns8_matrix& m)
      : value(m),
        residues(m.residues.data()),
        rows(m.exact_axis_ranges.rows.data()),
        cols(m.exact_axis_ranges.cols.data()) {}
  void check(const rns8_matrix& m) const {
    require(m.residues == value.residues && m.residues.data() == residues,
            "residue content/storage rollback");
    require(m.matrix_instance_id == value.matrix_instance_id && m.source_version == value.source_version &&
                m.prefix == value.prefix && m.backend == value.backend &&
                m.exact_range_prefix == value.exact_range_prefix &&
                m.exact_max_magnitude == value.exact_max_magnitude,
            "matrix identity/version/range rollback");
    require(m.desc.rows == value.desc.rows && m.desc.cols == value.desc.cols &&
                m.desc.max_prefix == value.desc.max_prefix && m.desc.semantics == value.desc.semantics,
            "descriptor rollback");
    require(m.host_residues_current == value.host_residues_current &&
                m.device_residues_current == value.device_residues_current &&
                m.host_native_current == value.host_native_current &&
                m.device_native_current == value.device_native_current &&
                m.host_byte_limbs_current == value.host_byte_limbs_current &&
                m.device_byte_limbs_current == value.device_byte_limbs_current &&
                m.native_i64 == value.native_i64 && m.native_u64 == value.native_u64,
            "native/currentness rollback");
    require(same_axes(m.exact_axis_ranges.rows, value.exact_axis_ranges.rows) &&
                same_axes(m.exact_axis_ranges.cols, value.exact_axis_ranges.cols) &&
                m.exact_axis_ranges.rows.data() == rows && m.exact_axis_ranges.cols.data() == cols,
            "axis proof content/storage rollback");
  }
};

struct Fixture {
  rns8_semantics semantics;
  int64_t m, n, k;
  bool shared, lift_only;
  rns8::Context ctx;
  rns8::Plan plan;
  rns8::Workspace workspace;
  rns8::Matrix a, b, c;
  std::vector<cpp_int> a_values, b_values, expected;
  rns8_matrix saved_a, saved_b, saved_c;

  Fixture(rns8_semantics sem, bool alias = false, bool asymmetric = false, bool lift = false)
      : semantics(sem),
        m(2),
        n(alias ? 2 : 13),
        k(alias ? 2 : 1),
        shared(alias),
        lift_only(lift),
        ctx(-1, RNS8_BACKEND_CPU_REFERENCE),
        plan(ctx, gemm_desc(sem, m, n, k)),
        workspace(ctx, plan),
        a(ctx, matrix_desc(sem, m, k)),
        b(ctx, matrix_desc(sem, k, n)),
        c(ctx, matrix_desc(sem, m, n)) {
    if (sem == RNS8_EXACT_WIDE_SIGNED) {
      produce<int64_t>(a, m, k, a_values);
      produce<int64_t>(b, k, n, b_values);
      std::vector<int64_t> sentinel(static_cast<std::size_t>(m * n), 42);
      rns8::check(rns8_pack_i64(ctx.get(), c.get(), sentinel.data(), n, 99));
    } else {
      produce<uint64_t>(a, m, k, a_values);
      produce<uint64_t>(b, k, n, b_values);
      std::vector<uint64_t> sentinel(static_cast<std::size_t>(m * n), 42);
      rns8::check(rns8_pack_u64(ctx.get(), c.get(), sentinel.data(), n, 99));
    }
    if (asymmetric) rns8::check(rns8_lift_exact_wide_cpu(ctx.get(), b.get(), 20, UINT64_MAX));
    const auto& rhs = shared ? a_values : b_values;
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        cpp_int dot = 0;
        for (int64_t inner = 0; inner < k; ++inner) dot += a_values[row * k + inner] * rhs[inner * n + col];
        expected.push_back(dot);
      }
    }
    saved_a = *a.get();
    saved_b = *b.get();
    saved_c = *c.get();
  }

  // Produce wide inputs through public packing/GEMM, never manufacture a proof
  // from raw CRT residues. The independent full-integer oracle is native*1024.
  template <typename T>
  void produce(rns8::Matrix& out, int64_t rows, int64_t cols, std::vector<cpp_int>& values) {
    rns8::Plan producer(ctx, gemm_desc(semantics, rows, cols, cols, true));
    rns8::Workspace scratch(ctx, producer);
    rns8::Matrix native(ctx, matrix_desc(semantics, rows, cols));
    rns8::Matrix scale(ctx, matrix_desc(semantics, cols, cols));
    std::vector<T> input(static_cast<std::size_t>(rows * cols));
    std::vector<T> diagonal(static_cast<std::size_t>(cols * cols), 0);
    for (std::size_t i = 0; i < input.size(); ++i) {
      if constexpr (std::is_same_v<T, int64_t>)
        input[i] = (i % 2) ? INT64_MAX : INT64_MIN;
      else
        input[i] = UINT64_MAX - static_cast<uint64_t>(i);
      values.push_back(cpp_int(input[i]) * 1024);
    }
    for (int64_t i = 0; i < cols; ++i) diagonal[i * cols + i] = 1024;
    if constexpr (std::is_same_v<T, int64_t>) {
      rns8::check(rns8_pack_i64(ctx.get(), native.get(), input.data(), cols, 11));
      rns8::check(rns8_pack_i64(ctx.get(), scale.get(), diagonal.data(), cols, 12));
    } else {
      rns8::check(rns8_pack_u64(ctx.get(), native.get(), input.data(), cols, 11));
      rns8::check(rns8_pack_u64(ctx.get(), scale.get(), diagonal.data(), cols, 12));
    }
    rns8::check(
        rns8_gemm_rns(ctx.get(), producer.get(), native.get(), scale.get(), out.get(), scratch.get()));
    require(out.get()->exact_range_prefix == 17, "public producer prefix");
  }

  void restore() {
    *a.get() = saved_a;
    *b.get() = saved_b;
    *c.get() = saved_c;
  }

  rns8_status call(uint32_t& selected) {
    if (lift_only) {
      const auto status = rns8_lift_exact_wide_cpu(ctx.get(), a.get(), 20, UINT64_MAX);
      if (status == RNS8_SUCCESS) selected = 20;
      return status;
    }
    return rns8_gemm_exact_wide_cpu_auto(ctx.get(), plan.get(), a.get(), shared ? a.get() : b.get(), c.get(),
                                         workspace.get(), UINT64_MAX, &selected);
  }

  void verify(uint32_t selected) {
    if (lift_only) {
      require(a.get()->exact_range_prefix == 20, "successful lift prefix");
      for (uint32_t p = 0; p < 20; ++p)
        for (std::size_t cell = 0; cell < a_values.size(); ++cell)
          require(a.get()->residues[a_values.size() * p + cell] ==
                      rns8::detail::centered_residue(a_values[cell], rns8::detail::kDefaultModuli[p]),
                  "successful lift integer preservation");
      return;
    }
    auto desc = gemm_desc(semantics, m, n, k);
    desc.max_prefix = selected;
    desc.flags = RNS8_PLAN_FORCE_FIXED_PREFIX;
    rns8::Plan exporter(ctx, desc);
    std::vector<uint64_t> limbs(expected.size() * 3);
    rns8::check(
        semantics == RNS8_EXACT_WIDE_SIGNED
            ? rns8_export_exact_wide_signed_limbs(ctx.get(), exporter.get(), c.get(), limbs.data(), n, 3)
            : rns8_export_exact_wide_unsigned_limbs(ctx.get(), exporter.get(), c.get(), limbs.data(), n, 3));
    for (std::size_t cell = 0; cell < expected.size(); ++cell) {
      cpp_int value = expected[cell];
      if (value < 0) value += cpp_int(1) << 192;
      for (uint32_t limb = 0; limb < 3; ++limb) {
        require(limbs[3 * cell + limb] == static_cast<uint64_t>(value & cpp_int(UINT64_MAX)),
                "independent full-integer limb oracle");
        value >>= 64;
      }
    }
  }
};

std::size_t exercise(const char* name, Fixture& f, bool profile_only) {
  current_case = name;
  fault::Profile measured{};
  uint32_t chosen = 0;
  for (unsigned repeat = 0; repeat < 2; ++repeat) {
    f.restore();
    uint32_t selected = 777;
    fault::begin();
    const auto status = f.call(selected);
    const auto result = fault::end();
    require(status == RNS8_SUCCESS && !result.triggered, "profile control succeeds");
    f.verify(selected);
    if (repeat == 0) {
      measured = result;
      chosen = selected;
    } else
      require(result.calls == measured.calls && result.bytes == measured.bytes && selected == chosen,
              "repeatable allocation profile");
  }
  require(measured.calls > 0 && measured.calls <= 4096, "bounded observed allocation points");
  if (!profile_only) {
    for (std::size_t point = 0; point < measured.calls; ++point) {
      f.restore();
      const MatrixSnapshot a(*f.a.get()), b(*f.b.get()), c(*f.c.get());
      const auto prefix = f.plan.get()->prefix;
      const auto product = f.plan.get()->modulus_product;
      const auto fingerprint = f.workspace.get()->schedule_fingerprint;
      const auto workspace_prefix = f.workspace.get()->prefix;
      const auto live = fault::live_allocations();
      uint32_t selected = 777;
      current_point = point;
      fault::begin(point);
      const auto status = f.call(selected);
      const auto result = fault::end();
      if (status != RNS8_INTERNAL_ERROR || !result.triggered) {
        std::fprintf(stderr, "case=%s point=%zu calls=%zu status=%u triggered=%d\n", name, point,
                     result.calls, static_cast<unsigned>(status), result.triggered);
        require(false, "every observed allocation failure is returned through the C API");
      }
      require(fault::live_allocations() == live, "failed transaction releases temporary C++ allocations");
      require(selected == 777, "selected-prefix output rollback");
      a.check(*f.a.get());
      b.check(*f.b.get());
      c.check(*f.c.get());
      require(f.plan.get()->prefix == prefix && f.plan.get()->modulus_product == product &&
                  f.workspace.get()->prefix == workspace_prefix &&
                  f.workspace.get()->schedule_fingerprint == fingerprint,
              "plan/workspace rollback");
      // A successful retry after every failure establishes recoverable state.
      require(f.call(selected) == RNS8_SUCCESS && selected == chosen, "successful retry after fault");
      f.verify(selected);
    }
  }
  std::printf("case=%s selected=%u allocations=%zu requested_bytes=%zu fault_points=%zu\n", name, chosen,
              measured.calls, measured.bytes, profile_only ? 0 : measured.calls);
  return profile_only ? 0 : measured.calls;
}

void allocator_self_check() {
  const auto live = fault::live_allocations();
  fault::begin(0);
  void* rejected = ::operator new(8, std::nothrow);
  const auto result = fault::end();
  require(rejected == nullptr && result.triggered && result.calls == 1, "nothrow fault hook");
  void* aligned = ::operator new(8, std::align_val_t(64));
  require(reinterpret_cast<std::uintptr_t>(aligned) % 64 == 0, "aligned allocation hook");
  ::operator delete(aligned, std::align_val_t(64));
  void* array = ::operator new[](8);
  ::operator delete[](array, std::size_t(8));
  require(fault::live_allocations() == live, "hook paired deallocation accounting");
}

void reused_scratch_boundary_check() {
  constexpr int64_t m = 2, n = 3, k = 2 * RNS8_SAFE_INT32_K_BLOCK + 1, ldb = 4, ldc = 4;
  std::vector<int8_t> a(m * k), b(k * ldb), c(m * ldc);
  std::vector<int32_t> scratch(n, INT32_MAX);
  auto* storage = scratch.data();
  for (uint16_t modulus : std::array<uint16_t, 3>{256, 255, 253}) {
    const int8_t low = static_cast<int8_t>(-static_cast<int>(modulus / 2));
    const int8_t high = static_cast<int8_t>((modulus - 1) / 2);
    std::fill(a.begin(), a.begin() + k, low);
    std::fill(a.begin() + k, a.end(), high);
    for (int64_t inner = 0; inner < k; ++inner) {
      b[inner * ldb] = low;
      b[inner * ldb + 1] = high;
      b[inner * ldb + 2] = inner % 2 ? low : high;
      b[inner * ldb + 3] = 127;  // Padded column is ignored, including modulus 253.
    }
    for (unsigned repeat = 0; repeat < 2; ++repeat) {
      std::fill(c.begin(), c.end(), int8_t(99));
      fault::begin();
      rns8::detail::ring_gemm_modulus_serial(a.data(), b.data(), c.data(), m, n, k, k, ldb, ldc, modulus,
                                             scratch);
      const auto profile = fault::end();
      require(profile.calls == 0 && scratch.data() == storage,
              "serial primitive reuses scratch without allocation");
      for (int64_t row = 0; row < m; ++row) {
        for (int64_t col = 0; col < n; ++col) {
          int64_t exact = 0;
          for (int64_t inner = 0; inner < k; ++inner)
            exact += static_cast<int64_t>(a[row * k + inner]) * b[inner * ldb + col];
          int64_t reduced = exact % modulus;
          if (reduced < 0) reduced += modulus;
          if (reduced >= (modulus + 1) / 2) reduced -= modulus;
          require(c[row * ldc + col] == reduced, "independent multi-block dot oracle with reused scratch");
        }
        require(c[row * ldc + n] == 99, "padded output preservation");
      }
    }
  }
  std::puts("PASS scratch_cases=6 K=131073 moduli=256,255,253 internal_allocations=0");
}

}  // namespace

int main(int argc, char** argv) {
  rns8::test::configure_noninteractive_errors();
  std::set_terminate(report_termination);
  const bool profile_only = argc == 2 && std::strcmp(argv[1], "--profile-only") == 0;
  if (argc > 2 || (argc == 2 && !profile_only)) return 1;
#if defined(_ITERATOR_DEBUG_LEVEL) && _ITERATOR_DEBUG_LEVEL > 0
  if (!profile_only) {
    std::fprintf(stderr,
                 "SKIP: fault sweep requires _ITERATOR_DEBUG_LEVEL=0; Debug STL proxy allocation in "
                 "noexcept constructors can terminate before the API guard. Use RelWithDebInfo/Release.\n");
    return 77;
  }
#endif
  allocator_self_check();
  reused_scratch_boundary_check();
  std::size_t points = 0;
  Fixture signed_dual(RNS8_EXACT_WIDE_SIGNED);
  points += exercise("signed-dual-2x13", signed_dual, profile_only);
  Fixture unsigned_dual(RNS8_EXACT_WIDE_UNSIGNED);
  points += exercise("unsigned-dual-2x13", unsigned_dual, profile_only);
  Fixture asymmetric(RNS8_EXACT_WIDE_SIGNED, false, true);
  points += exercise("signed-asymmetric-2x13", asymmetric, profile_only);
  Fixture shared(RNS8_EXACT_WIDE_UNSIGNED, true);
  points += exercise("unsigned-shared-2x2", shared, profile_only);
  Fixture signed_lift(RNS8_EXACT_WIDE_SIGNED, false, false, true);
  points += exercise("signed-lift", signed_lift, profile_only);
  Fixture unsigned_lift(RNS8_EXACT_WIDE_UNSIGNED, false, false, true);
  points += exercise("unsigned-lift", unsigned_lift, profile_only);
  std::printf("PASS scenarios=6 allocation_failures=%zu mode=%s\n", points,
              profile_only ? "profile" : "faults");
}
