#include <hss_common.hpp>

// Exercise the private implementation without introducing a production header.
#define main hss_rsvd_benchmark_main
#include "../hss_rsvd.cu"
#undef main

namespace {

static void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

template <class T>
class TestMatrix : public DenseMatrixCustom<T> {
  using TestBackend = DenseMatrixCustom<T>;
  using TestBackend::A;

public:
  using TestBackend::TestBackend;
  void setStored(const std::vector<T>& values) {
    require(values.size() == A.size(), "Invalid test matrix dimensions");
    CUDA_CHECK(cudaMemcpyAsync(deviceData(A), values.data(), values.size() * sizeof(T),
                               cudaMemcpyHostToDevice, this->stream));
    CUDA_CHECK(cudaStreamSynchronize(this->stream));
  }
  std::vector<T> stored() const {
    std::vector<T> result(A.size());
    copyToHost(result.data(), deviceData(A), result.size() * sizeof(T), this->stream);
    return result;
  }
};

template <class T>
static T testValue(double real, double imag) {
  if constexpr (hssComplex<T>) return T(real, imag);
  else return real;
}

template <class T>
static T adjointValue(T value) {
  if constexpr (hssComplex<T>) return std::conj(value);
  else return value;
}

template <class T>
static void checkCompression(int rows, int cols, int blocks, int rank, ncclComm_t comm,
    cublasHandle_t blas, cusolverDnHandle_t solver, cudaStream_t stream) {
  TestMatrix<T> matrix(cols * blocks, rows, comm, nullptr, blas, solver, stream);
  require(matrix.eps == 0. && matrix.K == 200 && matrix.oversampling == 0 &&
          matrix.power_iters == 0 && matrix.alpha == 1.15, "RSVD defaults must match the benchmark");
  DenseMatrix<T>& polymorphic = matrix;
  matrix.K = 8; matrix.alpha = 1.5;
  polymorphic.iterateTolerance();
  require(matrix.K == 12, "HSS level hook must dispatch to floor(K*alpha)");
  matrix.K = rank;
  matrix.eps = std::numeric_limits<double>::quiet_NaN(); // RSVD must never consult eps.
  std::vector<T> original(static_cast<size_t>(rows) * cols * blocks);
  for (int b = 0; b < blocks; ++b) for (int j = 0; j < cols; ++j) for (int i = 0; i < rows; ++i) {
    const T u = testValue<T>(1. + .2 * i, .1 * i);
    const T v = testValue<T>(.5 + .3 * j, -.2 * j);
    const T x = testValue<T>(std::sin(i + .2), std::cos(i + .3));
    const T y = testValue<T>(std::cos(j + .4), std::sin(j + .5));
    original[i + static_cast<size_t>(b * cols + j) * rows] = u * v + .1 * x * y;
  }
  matrix.setStored(original);
  std::vector<long long> recipe(blocks, cols);
  std::vector<T> basis;
  polymorphic.compress(recipe, basis, comm, nullptr);
  require(std::all_of(recipe.begin(), recipe.end(), [rank](long long r) { return r == rank; }),
          "Recipe must retain K even for an exactly rank-two input");
  require(basis.size() == static_cast<size_t>(blocks) * cols * rank, "Right basis has the wrong dimensions");
  const auto compact = matrix.stored();
  require(compact.size() == static_cast<size_t>(blocks) * rows * rank, "Compact factor has the wrong dimensions");
  double productError = 0., productNorm = 0., residual = 0., inputNorm = 0.;
  for (int b = 0; b < blocks; ++b) {
    for (int k = 0; k < rank; ++k) for (int i = 0; i < rows; ++i) {
      T expected = 0.;
      for (int j = 0; j < cols; ++j)
        expected += original[i + static_cast<size_t>(b * cols + j) * rows] * basis[j + static_cast<size_t>(b * rank + k) * cols];
      productError += std::norm(compact[i + static_cast<size_t>(b * rank + k) * rows] - expected);
      productNorm += std::norm(expected);
    }
    for (int j = 0; j < cols; ++j) for (int i = 0; i < rows; ++i) {
      T reconstructed = 0.;
      for (int k = 0; k < rank; ++k)
        reconstructed += compact[i + static_cast<size_t>(b * rank + k) * rows] *
          adjointValue(basis[j + static_cast<size_t>(b * rank + k) * cols]);
      const auto entry = original[i + static_cast<size_t>(b * cols + j) * rows];
      residual += std::norm(entry - reconstructed); inputNorm += std::norm(entry);
    }
  }
  require(std::sqrt(productError / productNorm) < 1.e-12, "Compact factor must equal A*V, including overlapping block compaction");
  require(std::sqrt(residual / inputNorm) < 1.e-8, "Fixed-rank RSVD must reconstruct this low-rank input");
  std::cout << "RSVD block " << rows << "x" << cols << ", blocks=" << blocks << ", K=" << rank
            << ", data=" << (hssComplex<T> ? 'Z' : 'D')
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
  checkCompression<double>(10, 6, 1, 3, comm, blas, solver, stream);
  checkCompression<std::complex<double>>(10, 6, 1, 3, comm, blas, solver, stream); // Tall input.
  checkCompression<double>(6, 10, 1, 3, comm, blas, solver, stream);
  checkCompression<std::complex<double>>(6, 10, 1, 3, comm, blas, solver, stream); // Wide input, no adjoint workaround.
  checkCompression<double>(10, 6, 2, 4, comm, blas, solver, stream);
  checkCompression<std::complex<double>>(10, 6, 2, 4, comm, blas, solver, stream); // Overlapping packing of successive blocks.
  {
    DenseMatrixCustom<std::complex<double>> matrix(6, 10, comm, nullptr, blas, solver, stream);
    matrix.K = 5; matrix.oversampling = 1;
    std::vector<long long> recipe{6}; std::vector<std::complex<double>> basis;
    bool rejected = false;
    try { matrix.compress(recipe, basis, comm, nullptr); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "Invalid K+oversampling must be rejected without silently lowering K");
  }
  CUSOLVER_CHECK(cusolverDnDestroy(solver)); CUBLAS_CHECK(cublasDestroy(blas));
  CUDA_CHECK(cudaStreamDestroy(stream)); NCCL_CHECK(ncclCommDestroy(comm));
  MPI_Finalize();
  std::cout << "HSS RSVD GPU checks passed\n";
  return 0;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  MPI_Abort(MPI_COMM_WORLD, 1);
  return 1;
}
