#include <hss_common.hpp>
#include <hyacin.h>

// Exercise the private implementation without introducing a production header.
#define main hss_hyacin_benchmark_main
#include "../hss_hyacin.cu"
#undef main

namespace {

static void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

class TestMatrix : public DenseMatrixCustom {
public:
  using DenseMatrixCustom::DenseMatrixCustom;
  bool borrows(cudaStream_t expectedStream, cublasHandle_t expectedBlas, cusolverDnHandle_t expectedSolver) const {
    return handle.cudaStream == expectedStream && handle.cublasHandle == expectedBlas &&
           handle.cusolverHandle == expectedSolver && handle.ncclComm == nullptr;
  }
  void setStored(const std::vector<std::complex<double>>& values) {
    require(values.size() == A.size(), "Invalid test matrix dimensions");
    CUDA_CHECK(cudaMemcpyAsync(deviceData(A), values.data(), values.size() * sizeof(cuDoubleComplex),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }
  std::vector<std::complex<double>> stored() const {
    std::vector<std::complex<double>> result(A.size());
    copyToHost(result.data(), deviceData(A), result.size() * sizeof(cuDoubleComplex), stream);
    return result;
  }
};

static void checkCompression(int rows, int cols, int blocks, int rank, ncclComm_t comm,
    cublasHandle_t blas, cusolverDnHandle_t solver, cudaStream_t stream) {
  TestMatrix matrix(cols * blocks, rows, comm, nullptr, blas, solver, stream);
  require(matrix.eps == 0., "Base tolerance must initialize to zero");
  require(matrix.borrows(stream, blas, solver), "Hyacin must reuse the HSS CUDA resources");
  DenseMatrix& polymorphic = matrix;
  matrix.eps = 1.e-8;
  polymorphic.iterateTolerance();
  require(matrix.eps == 1.e-8, "Hyacin must retain the base tolerance across levels");
  std::vector<std::complex<double>> original(static_cast<size_t>(rows) * cols * blocks);
  for (int b = 0; b < blocks; ++b) for (int j = 0; j < cols; ++j) for (int i = 0; i < rows; ++i) {
    // A rank-four diagonal block exercises overlapping compaction when blocks=2.
    if (rank == 4)
      original[i + static_cast<size_t>(b * cols + j) * rows] = i == j && i < rank ? 4. - i : 0.;
    else {
      const std::complex<double> u(1. + .2 * i, .1 * i);
      const std::complex<double> v(.5 + .3 * j, -.2 * j);
      const std::complex<double> x(std::sin(i + .2), std::cos(i + .3));
      const std::complex<double> y(std::cos(j + .4), std::sin(j + .5));
      original[i + static_cast<size_t>(b * cols + j) * rows] = u * v + .1 * x * y;
    }
  }
  matrix.setStored(original);
  std::vector<long long> recipe(blocks, cols);
  std::vector<std::complex<double>> basis;
  polymorphic.compress(recipe, basis, comm, nullptr);
  require(std::all_of(recipe.begin(), recipe.end(), [rank](long long r) { return r == rank; }),
          "Recipe must contain the internally truncated Hyacin rank");
  require(basis.size() == static_cast<size_t>(blocks) * cols * rank, "Right basis has the wrong dimensions");
  const auto compact = matrix.stored();
  require(compact.size() == static_cast<size_t>(blocks) * rows * rank, "Compact factor has the wrong dimensions");
  double productError = 0., productNorm = 0., residual = 0., inputNorm = 0.;
  for (int b = 0; b < blocks; ++b) {
    for (int k = 0; k < rank; ++k) for (int i = 0; i < rows; ++i) {
      std::complex<double> expected = 0.;
      for (int j = 0; j < cols; ++j)
        expected += original[i + static_cast<size_t>(b * cols + j) * rows] * basis[j + static_cast<size_t>(b * rank + k) * cols];
      productError += std::norm(compact[i + static_cast<size_t>(b * rank + k) * rows] - expected);
      productNorm += std::norm(expected);
    }
    for (int j = 0; j < cols; ++j) for (int i = 0; i < rows; ++i) {
      std::complex<double> reconstructed = 0.;
      for (int k = 0; k < rank; ++k)
        reconstructed += compact[i + static_cast<size_t>(b * rank + k) * rows] *
          std::conj(basis[j + static_cast<size_t>(b * rank + k) * cols]);
      const auto entry = original[i + static_cast<size_t>(b * cols + j) * rows];
      residual += std::norm(entry - reconstructed); inputNorm += std::norm(entry);
    }
  }
  require(std::sqrt(productError / productNorm) < 1.e-12, "Compact factor must equal A*V, including overlapping block compaction");
  require(std::sqrt(residual / inputNorm) < 1.e-8, "Hyacin must reconstruct this low-rank input");
  std::cout << "Hyacin block " << rows << "x" << cols << ", blocks=" << blocks << ", rank=" << rank
            << ": A*V error=" << std::sqrt(productError / productNorm)
            << ", reconstruction error=" << std::sqrt(residual / inputNorm) << '\n';
}

} // namespace

int main(int argc, char* argv[]) try {
  MPI_Init(&argc, &argv);
  selectCudaDevice(MPI_COMM_WORLD);
  ncclComm_t comm = bootstrapNccl(MPI_COMM_WORLD);
  cudaStream_t stream;
  cublasHandle_t blas;
  cusolverDnHandle_t solver;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUBLAS_CHECK(cublasCreate(&blas)); CUBLAS_CHECK(cublasSetStream(blas, stream));
  CUSOLVER_CHECK(cusolverDnCreate(&solver)); CUSOLVER_CHECK(cusolverDnSetStream(solver, stream));
  checkCompression(10, 6, 1, 2, comm, blas, solver, stream); // Tall input.
  checkCompression(6, 10, 1, 2, comm, blas, solver, stream); // Wide input, no adjoint workaround.
  checkCompression(10, 6, 2, 4, comm, blas, solver, stream); // Overlapping packing of successive blocks.
  {
    DenseMatrixCustom matrix(6, 10, comm, nullptr, blas, solver, stream);
    matrix.eps = std::numeric_limits<double>::quiet_NaN();
    std::vector<long long> recipe{6}; std::vector<std::complex<double>> basis;
    bool rejected = false;
    try { matrix.compress(recipe, basis, comm, nullptr); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "Invalid tolerance must be rejected");
  }
  // Matrix teardown must leave borrowed resources alive and bound to the same stream.
  cudaStream_t blasStream = nullptr, solverStream = nullptr;
  CUBLAS_CHECK(cublasGetStream(blas, &blasStream));
  CUSOLVER_CHECK(cusolverDnGetStream(solver, &solverStream));
  require(blasStream == stream && solverStream == stream, "Hyacin teardown changed borrowed streams");
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUSOLVER_CHECK(cusolverDnDestroy(solver)); CUBLAS_CHECK(cublasDestroy(blas));
  CUDA_CHECK(cudaStreamDestroy(stream)); NCCL_CHECK(ncclCommDestroy(comm));
  MPI_Finalize();
  std::cout << "HSS Hyacin GPU checks passed\n";
  return 0;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  MPI_Abort(MPI_COMM_WORLD, 1);
  return 1;
}
