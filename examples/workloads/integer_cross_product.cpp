#include "workload_integer_cross_product.hpp"

int main(int argc, char** argv) {
  return rns8_workload::program(argc, argv, [](const rns8_workload::Options& options) {
    rns8_workload::integer_cross_product(options);
  });
}
