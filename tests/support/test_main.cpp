#include <catch2/catch_session.hpp>

#include "noninteractive_errors.hpp"
#include "tool_common/build_identity.hpp"
#include <iostream>
#if defined(RNS8_ENABLE_HOST_RUNTIME_WITNESS)
#include "host_runtime_witness.hpp"
#endif

int main(int argc, char** argv) {
  const int identity_status = rns8::tools::emit_build_identity(argc, argv, std::cout);
  if (identity_status >= 0)
    return identity_status;
  rns8::test::configure_noninteractive_errors();
#if defined(RNS8_ENABLE_HOST_RUNTIME_WITNESS)
  rns8::test::HostRuntimeWitness witness(argc, argv);
#endif
  return Catch::Session().run(argc, argv);
}
