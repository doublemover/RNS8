#include <catch2/catch_test_macros.hpp>
#include <boost/multiprecision/cpp_int.hpp>
#include "backend_common/gram_plan.hpp"
#include <array>
#include <vector>

TEST_CASE("Gram production directory and mirror retain independent ownership and exact values",
          "[gram][cpu][mutation]") {
  using namespace rns8::detail;
  using boost::multiprecision::cpp_int;
  const std::array<uint16_t, 3> moduli{256, 255, 253};
  for (int n : {1, 15, 16, 17, 31, 32, 33})
    for (auto orientation :
         {GramOrientation::transpose_times_self, GramOrientation::self_times_transpose}) {
      GramProblem p;
      p.semantics = RNS8_EXACT_WIDE_SIGNED;
      p.rows = orientation == GramOrientation::transpose_times_self ? 3 : n;
      p.cols = orientation == GramOrientation::transpose_times_self ? n : 3;
      p.source_ld = p.cols + 3;
      p.output_ld = n + 2;
      p.source_instance = 1;
      p.prefix = 3;
      const auto support = query_gram_support(p, UINT64_MAX, false, false);
      REQUIRE(support.status == RNS8_SUCCESS);
      const auto jobs = gram_tiles(support);
      const uint64_t q = (n + 15) / 16;
      const auto expected_jobs = q * (q + 1) / 2;
      CHECK(jobs.size() == expected_jobs);
      std::vector<int> owners(n * n, 0);
      for (auto job : jobs)
        for (int r = 0; r < 16; ++r)
          for (int col = 0; col < 16; ++col) {
            const auto row = uint64_t(job.row) * 16 + r, column = uint64_t(job.col) * 16 + col;
            if (row >= uint64_t(n) || column >= uint64_t(n) || row > column)
              continue;
            ++owners[row * n + column];
            if (row != column)
              ++owners[column * n + row];
          }
      for (auto count : owners)
        CHECK(count == 1);
      const auto source_plane = p.rows * p.source_ld;
      std::vector<int8_t> source(size_t(source_plane * p.prefix), 101);
      for (uint32_t plane = 0; plane < p.prefix; ++plane)
        for (int64_t row = 0; row < p.rows; ++row)
          for (int64_t col = 0; col < p.cols; ++col)
            source[size_t(plane * source_plane + row * p.source_ld + col)] =
                int8_t((3 * row + 5 * col) % 17 - 8);
      const auto output_plane = n * p.output_ld;
      for (bool separate : {false, true}) {
        std::vector<int8_t> result(size_t(output_plane * p.prefix + 64), 103);
        gram_residues_cpu(source.data(), result.data(), p, jobs, moduli.data(), separate);
        for (uint32_t plane = 0; plane < p.prefix; ++plane)
          for (int row = 0; row < n; ++row)
            for (int col = 0; col < n; ++col) {
              cpp_int sum = 0;
              for (int k = 0; k < 3; ++k) {
                const auto a = orientation == GramOrientation::transpose_times_self
                                   ? k * p.source_ld + row
                                   : row * p.source_ld + k;
                const auto b = orientation == GramOrientation::transpose_times_self
                                   ? k * p.source_ld + col
                                   : col * p.source_ld + k;
                sum += cpp_int(source[size_t(plane * source_plane + a)]) *
                       source[size_t(plane * source_plane + b)];
              }
              cpp_int centered = sum % moduli[plane];
              if (centered < 0)
                centered += moduli[plane];
              if (centered > (moduli[plane] - 1) / 2)
                centered -= moduli[plane];
              const auto expected_residue = centered.convert_to<int>();
              CHECK(result[size_t(plane * output_plane + row * p.output_ld + col)] ==
                    expected_residue);
            }
        for (uint32_t plane = 0; plane < p.prefix; ++plane)
          for (int row = 0; row < n; ++row)
            for (int col = n; col < p.output_ld; ++col)
              CHECK(result[size_t(plane * output_plane + row * p.output_ld + col)] == 103);
        for (size_t index = size_t(output_plane * p.prefix); index < result.size(); ++index)
          CHECK(result[index] == 103);
      }
    }
}
