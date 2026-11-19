#include <hss_common.hpp>

static void runBenchmark(int argc, char* argv[]) {
  const long long Nbody = argc > 1 ? std::atoll(argv[1]) : 2048;
  const long long leaf_size = argc > 2 ? std::atoll(argv[2]) : 512;
  const double epi = argc > 3 ? std::atof(argv[3]) : 1e-10;
  if (!std::isfinite(epi) || epi < 0.)
    throw std::invalid_argument("Require finite nonnegative tolerance");
  runHssBenchmark<DenseMatrix>(Nbody, leaf_size,
    [epi](DenseMatrix& matrix) { matrix.eps = epi; },
    [epi](const DenseMatrix&) {
      std::cout << " HSS  Matvec Target accuracy: " << epi << std::endl;
    });
}

int main(int argc, char* argv[]) try {
  MPI_Init(&argc, &argv);
  runBenchmark(argc, argv);
  MPI_Finalize();
  return 0;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  MPI_Abort(MPI_COMM_WORLD, 1);
  return 1;
}
