#include <hss_common.hpp>
#include <hyacin.h>

namespace {

class DenseMatrixCustom : public DenseMatrix {
protected:
  hyacinHandle_t handle{};

public:
  DenseMatrixCustom(long long localM, long long N, ncclComm_t ncclPrime, ncclComm_t ncclSub,
                    cublasHandle_t blas, cusolverDnHandle_t solver, cudaStream_t stream)
    : DenseMatrix(localM, N, ncclPrime, ncclSub, blas, solver, stream) {
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

  void compress(std::vector<long long>& recipe, std::vector<std::complex<double>>& basis,
                ncclComm_t ncclPrime, ncclComm_t ncclSub) override {
    if (!std::isfinite(eps) || eps < 0.)
      throw std::invalid_argument("DenseMatrix::eps must be finite and nonnegative");
    thrust::device_vector<double> singularValues;
    thrust::device_vector<cuDoubleComplex> left;
    compressWith(recipe, basis, ncclPrime, ncclSub,
      [&](int rows, int cols, cuDoubleComplex* block, thrust::device_vector<cuDoubleComplex>& deviceBasis) {
        const int maxRank = std::min(rows, cols);
        const auto offset = deviceBasis.size();
        deviceBasis.resize(offset + static_cast<size_t>(cols) * maxRank);
        auto* right = deviceData(deviceBasis) + offset;
        singularValues.resize(maxRank);
        // A and V are distinct allocations, so Hyacin preserves the input block.
        // The returned rank already reflects Hyacin's internal truncation.
        const int rank = hyacinTsvdFitSegmentN(&handle, eps, 'N', rows, cols, maxRank,
          block, rows, deviceData(singularValues), right, cols, rows, HYACIN_F64_COMPLEX);
        if (rank < 0 || rank > maxRank)
          throw std::runtime_error("Hyacin returned a rank outside [0, min(M,N)]");
        if (rank > 0) {
          left.resize(static_cast<size_t>(rows) * rank);
          const cuDoubleComplex one = make_cuDoubleComplex(1., 0.), zero = make_cuDoubleComplex(0., 0.);
          CUBLAS_CHECK(cublasZgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, rows, rank, cols,
            &one, block, rows, right, cols, &zero, deviceData(left), rows));
          copyDevice(block, deviceData(left), left.size(), stream);
        }
        deviceBasis.resize(offset + static_cast<size_t>(cols) * rank);
        return rank;
      });
    // Clear optional Hyacin timer events as well as synchronizing its work.
    hyacinSync_TimerSegments(&handle, nullptr, 0);
  }
};

static void runBenchmark(int argc, char* argv[]) {
  const long long Nbody = argc > 1 ? std::atoll(argv[1]) : 2048;
  const long long leaf_size = argc > 2 ? std::atoll(argv[2]) : 512;
  const double epi = argc > 3 ? std::atof(argv[3]) : 1e-10;
  if (!std::isfinite(epi) || epi < 0.)
    throw std::invalid_argument("Require finite nonnegative tolerance");
  runHssBenchmark<DenseMatrixCustom>(Nbody, leaf_size,
    [epi](DenseMatrixCustom& matrix) { matrix.eps = epi; },
    [epi](const DenseMatrixCustom&) {
      std::cout << " HSS Hyacin Matvec Target accuracy: " << epi << std::endl;
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
