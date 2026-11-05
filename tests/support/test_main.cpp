#include <catch2/catch_session.hpp>

#include "noninteractive_errors.hpp"
#include "tool_common/build_identity.hpp"
#include <iostream>

int main(int argc, char** argv) {
  const int identity_status = rns8::tools::emit_build_identity(argc, argv, std::cout);
  if (identity_status >= 0)
    return identity_status;
  rns8::test::configure_noninteractive_errors();
  return Catch::Session().run(argc, argv);
}
