#include "backend_common/sha256.hpp"
#include "tool_common/execution_host_identity.hpp"
#include <catch2/catch_test_macros.hpp>
#include <sstream>

TEST_CASE("application identity digests match RFC 4231 HMAC-SHA256 vectors", "[host-identity]") {
  // https://www.rfc-editor.org/rfc/rfc4231, cases 1, 2 and 6.
  CHECK(rns8::detail::hmac_sha256(std::string(20, '\x0b'), "Hi There") ==
        "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7");
  CHECK(rns8::detail::hmac_sha256("Jefe", "what do ya want for nothing?") ==
        "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843");
  CHECK(rns8::detail::hmac_sha256(std::string(131, static_cast<char>(0xaa)),
                                  "Test Using Larger Than Block-Size Key - Hash Key First") ==
        "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54");
}

TEST_CASE("host OS identities normalize material and publish only scoped derived identifiers",
          "[host-identity]") {
  using rns8::tools::ExecutionHostSnapshot;
  const auto first = ExecutionHostSnapshot::from_material(
      "windows", "native_registry_machine_guid_64_view", "01234567-89AB-CDEF-0123-456789ABCDEF");
  const auto same = ExecutionHostSnapshot::from_material(
      "windows", "native_registry_machine_guid_64_view", "{01234567-89ab-cdef-0123-456789abcdef}");
  const auto other = ExecutionHostSnapshot::from_material(
      "windows", "native_registry_machine_guid_64_view", "1123456789abcdef0123456789abcdef");
  const auto linux = ExecutionHostSnapshot::from_material("linux", "/etc/machine-id",
                                                          "0123456789abcdef0123456789abcdef\n");
  CHECK(std::string(first.status) == "observed");
  CHECK(first.os_instance_sha256.size() == 64);
  CHECK(first.os_instance_sha256 == same.os_instance_sha256);
  CHECK(first.os_instance_sha256 != other.os_instance_sha256);
  CHECK(first.os_instance_sha256 != linux.os_instance_sha256);
  std::ostringstream output;
  first.print(output);
  const auto value = nlohmann::json::parse(output.str());
  CHECK(value["scope"] == "observed_os_instance");
  CHECK(value["os_instance_sha256"] == first.os_instance_sha256);
  CHECK(output.str().find("01234567-89AB-CDEF-0123-456789ABCDEF") == std::string::npos);
  CHECK(output.str().find("0123456789abcdef0123456789abcdef") == std::string::npos);
  for (const auto material :
       {"", "uninitialized", "00000000000000000000000000000000", "ffffffffffffffffffffffffffffffff",
        "01234567-89ab-cdef-0123-456789abcdeg", "0123456789abcdef0123456789abcdefextra"}) {
    const auto missing = ExecutionHostSnapshot::from_material("linux", "/etc/machine-id", material);
    CHECK(std::string(missing.status) == "unavailable");
    CHECK(missing.os_instance_sha256.empty());
  }
}
