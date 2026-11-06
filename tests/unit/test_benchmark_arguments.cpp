#include "../../benchmarks/rns8_bench_args.hpp"
#include <catch2/catch_test_macros.hpp>
#include <string>
#include <vector>

namespace {
rns8::bench::Args parse_benchmark_arguments(std::vector<std::string> arguments) {
  std::vector<char*> pointers;
  pointers.reserve(arguments.size());
  for (auto& argument : arguments)
    pointers.push_back(argument.data());
  return rns8::bench::parse_args(static_cast<int>(pointers.size()), pointers.data());
}
} // namespace

TEST_CASE("benchmark reference requests retain ordinary integer operation admission",
          "[benchmark-arguments]") {
  for (const auto semantics : {"bounded-i64", "bounded-u64", "finite-u8-ring", "finite-u8-field",
                               "wrap-u64", "exact-wide-signed", "exact-wide-unsigned"}) {
    const auto args = parse_benchmark_arguments({
        "rns8-bench",
        "--backend",
        std::string(semantics) == "wrap-u64" ? "wrap64-byte-limb" : "cpu",
        "--semantics",
        semantics,
        "--modulus",
        "251",
        "--m",
        "4",
        "--n",
        "4",
        "--k",
        "4",
        "--warmups",
        "0",
        "--repeats",
        "1",
        "--verify-native-output",
    });
    CHECK(args.verify_native_output);
    CHECK_FALSE(args.vector_to_rns_chain);
    CHECK(args.residue_chain_length == 1);
  }
}

TEST_CASE(
    "bounded vector handoff CLI admits complete chain verification and reference cache requests",
    "[benchmark-arguments]") {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  for (const auto semantics : {"bounded-i64", "bounded-u64"}) {
    for (const auto mode : {"--vector-to-rns-chain", "--vector-to-rns-chain-host-repack-control"}) {
      const auto verified = parse_benchmark_arguments({
          "rns8-bench",
          "--semantics",
          semantics,
          "--m",
          "4",
          "--n",
          "4",
          "--k",
          "4",
          mode,
          "--verify-native-output",
          "--warmups",
          "0",
          "--repeats",
          "1",
      });
      CHECK(verified.vector_to_rns_chain);
      CHECK(verified.verify_native_output);
      CHECK(verified.vector_to_rns_chain_host_repack_control ==
            (std::string(mode) == "--vector-to-rns-chain-host-repack-control"));
      const auto cached = parse_benchmark_arguments({
          "rns8-bench",
          "--semantics",
          semantics,
          "--m",
          "4",
          "--n",
          "4",
          "--k",
          "4",
          mode,
          "--reference-cache-dir",
          "temp/parser-reference-cache",
          "--populate-reference-cache",
      });
      CHECK(cached.reference_cache_directory == "temp/parser-reference-cache");
      CHECK(cached.populate_reference_cache);
      CHECK(cached.vector_to_rns_chain);
    }
  }
#else
  SKIP("vector handoff argument admission requires the actual HIP-enabled build configuration");
#endif
}

TEST_CASE("resident chain CLI admits complete bounded and wide references",
          "[benchmark-arguments]") {
  for (const auto semantics :
       {"bounded-i64", "bounded-u64", "exact-wide-signed", "exact-wide-unsigned"})
    for (const auto mode :
         {"--residue-chain-final-export", "--residue-chain-independent-final-export"}) {
      const auto args = parse_benchmark_arguments({
          "rns8-bench",
          "--backend",
          "cpu",
          "--semantics",
          semantics,
          "--m",
          "4",
          "--n",
          "4",
          "--k",
          "4",
          "--residue-chain-length",
          "3",
          mode,
          "--verify-native-output",
          "--reference-cache-dir",
          "temp/parser-reference-cache",
          "--populate-reference-cache",
      });
      CHECK(args.residue_chain_length == 3);
      CHECK(args.residue_chain_final_export);
      CHECK(args.residue_chain_independent_final_export ==
            (std::string(mode) == "--residue-chain-independent-final-export"));
      CHECK(args.verify_native_output);
      CHECK(args.populate_reference_cache);
      CHECK(args.reference_cache_directory == "temp/parser-reference-cache");
    }
}
