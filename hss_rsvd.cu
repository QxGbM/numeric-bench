#include <hss_common.hpp>

namespace {

template <class T>
class DenseMatrixCustom : public DenseMatrix<T> {
  using Base = DenseMatrix<T>;
  using DeviceScalar = HssDeviceScalar<T>;
  using Base::blas;
  using Base::stream;
  using Base::compressWith;
  using Base::solver;
  cusolverDnParams_t params = nullptr;

  struct Workspace {
    thrust::device_vector<DeviceScalar> input, left;
    thrust::device_vector<double> singularValues;
    thrust::device_vector<unsigned char> deviceWork;
    thrust::device_vector<int> info;
    std::vector<unsigned char> hostWork;

    Workspace() : info(1) {}
    Workspace(const Workspace&) = delete;
    Workspace& operator=(const Workspace&) = delete;
  };

public:
  int K = 200;
  int oversampling = 0;
  int power_iters = 0;
  double alpha = 1.15;

  DenseMatrixCustom(long long localM, long long N, ncclComm_t ncclPrime, ncclComm_t ncclSub,
                    cublasHandle_t blas, cusolverDnHandle_t solver, cudaStream_t stream)
    : Base(localM, N, ncclPrime, ncclSub, blas, solver, stream) {
    CUSOLVER_CHECK(cusolverDnCreateParams(&params));
  }

  ~DenseMatrixCustom() override { cusolverDnDestroyParams(params); }
  DenseMatrixCustom(const DenseMatrixCustom&) = delete;
  DenseMatrixCustom& operator=(const DenseMatrixCustom&) = delete;

  void iterateTolerance() override {
    if (!std::isfinite(alpha) || alpha <= 0.)
      throw std::invalid_argument("RSVD alpha must be finite and positive");
    const double next = std::floor(double(K) * alpha);
    if (next < 1. || next > double(std::numeric_limits<int>::max()))
      throw std::overflow_error("Iterated RSVD rank must fit a positive int");
    K = static_cast<int>(next);
  }

  void compress(std::vector<long long>& recipe, std::vector<T>& basis,
                ncclComm_t ncclPrime, ncclComm_t ncclSub) override {
    if (K <= 0 || oversampling < 0 || power_iters < 0)
      throw std::invalid_argument("RSVD requires positive K and nonnegative oversampling/power_iters");
    // Deliberately do not consult DenseMatrix::eps or read the singular values.
    std::unique_ptr<Workspace> scratch;
    compressWith(recipe, basis, ncclPrime, ncclSub,
      [&](int rows, int cols, DeviceScalar* block, thrust::device_vector<DeviceScalar>& deviceBasis) {
        if (static_cast<long long>(K) + oversampling >= std::min(rows, cols))
          throw std::invalid_argument("RSVD requires K + oversampling < min(block rows, block cols); K=" +
            std::to_string(K) + ", oversampling=" + std::to_string(oversampling) +
            ", rows=" + std::to_string(rows) + ", cols=" + std::to_string(cols));
        if (!scratch) scratch = std::make_unique<Workspace>();
        auto& work = *scratch;
        // GESVDR overwrites its input; preserve A for the A*V product.
        work.input.resize(static_cast<size_t>(rows) * cols);
        copyDevice(deviceData(work.input), block, work.input.size(), stream);
        work.singularValues.resize(std::min(rows, cols)); // API output storage; values are never read.
        work.left.resize(static_cast<size_t>(rows) * K);
        const auto offset = deviceBasis.size();
        deviceBasis.resize(offset + static_cast<size_t>(cols) * K);
        auto* right = deviceData(deviceBasis) + offset;
        size_t deviceBytes = 0, hostBytes = 0;
        CUSOLVER_CHECK(cusolverDnXgesvdr_bufferSize(solver, params, 'N', 'S', rows, cols, K,
          oversampling, power_iters, hssCudaType<T>, deviceData(work.input), rows,
          CUDA_R_64F, deviceData(work.singularValues), hssCudaType<T>, nullptr, rows,
          hssCudaType<T>, right, cols, hssCudaType<T>, &deviceBytes, &hostBytes));
        work.deviceWork.resize(deviceBytes);
        work.hostWork.resize(hostBytes);
        CUSOLVER_CHECK(cusolverDnXgesvdr(solver, params, 'N', 'S', rows, cols, K,
          oversampling, power_iters, hssCudaType<T>, deviceData(work.input), rows,
          CUDA_R_64F, deviceData(work.singularValues), hssCudaType<T>, nullptr, rows,
          hssCudaType<T>, right, cols, hssCudaType<T>, deviceData(work.deviceWork), deviceBytes,
          work.hostWork.data(), hostBytes, deviceData(work.info)));
        int info = 0;
        copyToHost(&info, deviceData(work.info), sizeof(info), stream);
        if (info != 0) throw std::runtime_error("cusolverDnXgesvdr returned info=" + std::to_string(info));
        multiplyRightBasis(blas, rows, cols, K, block, right, deviceData(work.left));
        copyDevice(block, deviceData(work.left), work.left.size(), stream);
        return K; // Fixed rank, independent of the (potentially inaccurate) S.
      });
  }
};

template <class T>
static void runBenchmark(long long Nbody, long long leaf_size, int rank_leaves, int oversampling, int power_iters, double alpha) {
  runHssBenchmark<DenseMatrixCustom<T>>(Nbody, leaf_size,
    [=](DenseMatrixCustom<T>& matrix) {
      matrix.K = rank_leaves;
      matrix.oversampling = oversampling;
      matrix.power_iters = power_iters;
      matrix.alpha = alpha;
    },
    [=](const DenseMatrixCustom<T>& matrix) {
      std::cout << " HSS RSVD Leaf rank: " << rank_leaves << " Final rank parameter: " << matrix.K << std::endl;
      std::cout << " HSS RSVD Oversampling: " << oversampling << " Power iterations: " << power_iters
                << " Alpha: " << alpha << std::endl;
    });
}

static void runBenchmark(int argc, char* argv[]) {
  long long Nbody = 2048, leaf_size = 512;
  int rank_leaves = 200, oversampling = 0, power_iters = 0;
  double alpha = 1.15;
  const char precision = parseHssArguments(argc, argv,
    hssOption("N", Nbody), hssOption("leaf_size", leaf_size), hssOption("K", rank_leaves), hssOption("oversampling", oversampling),
    hssOption("piters", power_iters), hssOption("alpha", alpha));
  if (rank_leaves <= 0 || oversampling < 0 || power_iters < 0 || !std::isfinite(alpha) || alpha <= 0.)
    throw std::invalid_argument("Require positive rank_leaves/alpha and nonnegative oversampling/power_iters");
  if (precision == 'D') runBenchmark<double>(Nbody, leaf_size, rank_leaves, oversampling, power_iters, alpha);
  else runBenchmark<std::complex<double>>(Nbody, leaf_size, rank_leaves, oversampling, power_iters, alpha);
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
