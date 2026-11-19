#include <hss_common.hpp>

namespace {

class DenseMatrixCustom : public DenseMatrix {
  cusolverDnParams_t params = nullptr;

  struct Workspace {
    thrust::device_vector<cuDoubleComplex> input, left;
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
    : DenseMatrix(localM, N, ncclPrime, ncclSub, blas, solver, stream) {
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

  void compress(std::vector<long long>& recipe, std::vector<std::complex<double>>& basis,
                ncclComm_t ncclPrime, ncclComm_t ncclSub) override {
    if (K <= 0 || oversampling < 0 || power_iters < 0)
      throw std::invalid_argument("RSVD requires positive K and nonnegative oversampling/power_iters");
    // Deliberately do not consult DenseMatrix::eps or read the singular values.
    std::unique_ptr<Workspace> scratch;
    compressWith(recipe, basis, ncclPrime, ncclSub,
      [&](int rows, int cols, cuDoubleComplex* block, thrust::device_vector<cuDoubleComplex>& deviceBasis) {
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
          oversampling, power_iters, CUDA_C_64F, deviceData(work.input), rows,
          CUDA_R_64F, deviceData(work.singularValues), CUDA_C_64F, nullptr, rows,
          CUDA_C_64F, right, cols, CUDA_C_64F, &deviceBytes, &hostBytes));
        work.deviceWork.resize(deviceBytes);
        work.hostWork.resize(hostBytes);
        CUSOLVER_CHECK(cusolverDnXgesvdr(solver, params, 'N', 'S', rows, cols, K,
          oversampling, power_iters, CUDA_C_64F, deviceData(work.input), rows,
          CUDA_R_64F, deviceData(work.singularValues), CUDA_C_64F, nullptr, rows,
          CUDA_C_64F, right, cols, CUDA_C_64F, deviceData(work.deviceWork), deviceBytes,
          work.hostWork.data(), hostBytes, deviceData(work.info)));
        int info = 0;
        copyToHost(&info, deviceData(work.info), sizeof(info), stream);
        if (info != 0) throw std::runtime_error("cusolverDnXgesvdr returned info=" + std::to_string(info));
        const cuDoubleComplex one = make_cuDoubleComplex(1., 0.), zero = make_cuDoubleComplex(0., 0.);
        CUBLAS_CHECK(cublasZgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, rows, K, cols,
          &one, block, rows, right, cols, &zero, deviceData(work.left), rows));
        copyDevice(block, deviceData(work.left), work.left.size(), stream);
        return K; // Fixed rank, independent of the (potentially inaccurate) S.
      });
  }
};

static void runBenchmark(int argc, char* argv[]) {
  const long long Nbody = argc > 1 ? std::atoll(argv[1]) : 2048;
  const long long leaf_size = argc > 2 ? std::atoll(argv[2]) : 512;
  const int rank_leaves = argc > 3 ? std::atoi(argv[3]) : 200;
  const int oversampling = argc > 4 ? std::atoi(argv[4]) : 0;
  const int power_iters = argc > 5 ? std::atoi(argv[5]) : 0;
  const double alpha = argc > 6 ? std::atof(argv[6]) : 1.15;
  if (rank_leaves <= 0 || oversampling < 0 || power_iters < 0 || !std::isfinite(alpha) || alpha <= 0.)
    throw std::invalid_argument("Require positive rank_leaves/alpha and nonnegative oversampling/power_iters");
  runHssBenchmark<DenseMatrixCustom>(Nbody, leaf_size,
    [=](DenseMatrixCustom& matrix) {
      matrix.K = rank_leaves;
      matrix.oversampling = oversampling;
      matrix.power_iters = power_iters;
      matrix.alpha = alpha;
    },
    [=](const DenseMatrixCustom& matrix) {
      std::cout << " HSS RSVD Leaf rank: " << rank_leaves << " Final rank parameter: " << matrix.K << std::endl;
      std::cout << " HSS RSVD Oversampling: " << oversampling << " Power iterations: " << power_iters
                << " Alpha: " << alpha << std::endl;
    });
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
