#include <hss_common.hpp>

template <class T>
static void runBenchmark(long long Nbody, long long leaf_size, double eps) {
  runHssBenchmark<DenseMatrix<T>>(Nbody, leaf_size,
    [eps](DenseMatrix<T>& matrix) { matrix.eps = eps; },
    [eps](const DenseMatrix<T>&) {
      std::cout << " HSS  Matvec Target accuracy: " << eps << std::endl;
    });
}

static void runBenchmark(int argc, char* argv[]) {
  long long Nbody = 2048, leaf_size = 512;
  double eps = 1e-10;
  const char precision = parseHssArguments(argc, argv,
    hssOption("N", Nbody), hssOption("leaf_size", leaf_size), hssOption("eps", eps));
  if (!std::isfinite(eps) || eps < 0.)
    throw std::invalid_argument("Require finite nonnegative tolerance");
  if (precision == 'D') runBenchmark<double>(Nbody, leaf_size, eps);
  else runBenchmark<std::complex<double>>(Nbody, leaf_size, eps);
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
