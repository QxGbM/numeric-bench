#include <blr_commons.hpp>

int main(int argc, char* argv[]) try {
  MPI_Init(&argc, &argv);
  const bool passed = runBenchmark<BlockedDense>(argc, argv);
  MPI_Finalize();
  return passed ? EXIT_SUCCESS : EXIT_FAILURE;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  MPI_Abort(MPI_COMM_WORLD, 1);
  return 1;
}
