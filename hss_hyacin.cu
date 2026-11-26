#include <hss_common.hpp>
#include <hyacin.h>

namespace {

template <class T>
class DenseMatrixCustom : public DenseMatrix<T> {
  using Base = DenseMatrix<T>;
  using DeviceScalar = HssDeviceScalar<T>;
  using Base::blas;
  using Base::compressWith;
protected:
  hyacinHandle_t handle{};

public:
  using Base::stream;

  DenseMatrixCustom(long long localM, long long N, ncclComm_t ncclPrime, ncclComm_t ncclSub,
                    cublasHandle_t blas, cusolverDnHandle_t solver, cudaStream_t stream)
    : Base(localM, N, ncclPrime, ncclSub, blas, solver, stream) {
    handle.cudaStream = stream;
    handle.cublasHandle = blas;
    handle.cusolverHandle = solver;
    // Each compression callback fits a complete local block. HSS handles the
    // changing NCCL communicator groups; Hyacin must not all-reduce these blocks.
    handle.ncclComm = nullptr;
    try {
      hyacinCreate(&handle);
    } catch (...) {
      hyacinDestroy(&handle);
      throw;
    }
    if (handle.cudaStream != stream || handle.cublasHandle != blas || handle.cusolverHandle != solver) {
      hyacinDestroy(&handle);
      throw std::runtime_error("Hyacin replaced the borrowed HSS stream or library handles");
    }
  }

  ~DenseMatrixCustom() override { hyacinDestroy(&handle); }
  DenseMatrixCustom(const DenseMatrixCustom&) = delete;
  DenseMatrixCustom& operator=(const DenseMatrixCustom&) = delete;

  void compress(std::vector<long long>& recipe, std::vector<T>& basis,
                ncclComm_t ncclPrime, ncclComm_t ncclSub) override {
    if (!std::isfinite(this->eps) || this->eps < 0.)
      throw std::invalid_argument("DenseMatrix::eps must be finite and nonnegative");
    thrust::device_vector<double> singularValues;
    thrust::device_vector<DeviceScalar> left;
    compressWith(recipe, basis, ncclPrime, ncclSub,
      [&](int rows, int cols, DeviceScalar* block, thrust::device_vector<DeviceScalar>& deviceBasis) {
        const int maxRank = std::min(rows, cols);
        const auto offset = deviceBasis.size();
        deviceBasis.resize(offset + static_cast<size_t>(cols) * maxRank);
        auto* right = deviceData(deviceBasis) + offset;
        singularValues.resize(maxRank);
        // A and V are distinct allocations, so Hyacin preserves the input block.
        // The returned rank already reflects Hyacin's internal truncation.
        const int rank = hyacinTsvdFitSegmentN(&handle, this->eps, 'N', rows, cols, maxRank,
          block, rows, deviceData(singularValues), right, cols, rows, hssComplex<T> ? HYACIN_F64_COMPLEX : HYACIN_F64);
        if (rank < 0 || rank > maxRank)
          throw std::runtime_error("Hyacin returned a rank outside [0, min(M,N)]");
        if (rank > 0) {
          left.resize(static_cast<size_t>(rows) * rank);
          multiplyRightBasis(blas, rows, cols, rank, block, right, deviceData(left));
          copyDevice(block, deviceData(left), left.size(), stream);
        }
        deviceBasis.resize(offset + static_cast<size_t>(cols) * rank);
        return rank;
      });
    // Clear optional Hyacin timer events as well as synchronizing its work.
    hyacinSync_TimerSegments(&handle, nullptr, 0);
  }
};

template <class T>
static void runBenchmark(long long Nbody, long long leaf_size, double eps) {
  runHssBenchmark<DenseMatrixCustom<T>>(Nbody, leaf_size,
    [eps](DenseMatrixCustom<T>& matrix) { matrix.eps = eps; },
    [eps](const DenseMatrixCustom<T>&) {
      std::cout << " HSS Hyacin Matvec Target accuracy: " << eps << std::endl;
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

} // namespace

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
