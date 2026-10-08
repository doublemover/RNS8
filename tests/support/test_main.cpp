#include <catch2/catch_session.hpp>

#include "noninteractive_errors.hpp"

int main(int argc, char** argv) {
  rns8::test::configure_noninteractive_errors();
  return Catch::Session().run(argc, argv);
}
