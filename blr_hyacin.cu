#include <blr_commons.hpp>
#include <hyacin.h>

namespace {

class BlockedDenseCustom : public BlockedDense {
  hyacinHandle_t handle{};

public:
  BlockedDenseCustom(std::int32_t tileRows, std::int32_t tileCols, std::int32_t localCells,
                     std::int32_t leafCells, cublasHandle_t blas, cusolverDnHandle_t solver,
                     cudaStream_t stream)
    : BlockedDense(tileRows, tileCols, localCells, leafCells, blas, solver, stream) {
    handle.cudaStream = stream;
    handle.cublasHandle = blas;
    handle.cusolverHandle = solver;
    // Each rank compresses its own tiles locally.
    handle.ncclComm = nullptr;
    try {
      hyacinCreate(&handle);
    } catch (...) {
      hyacinDestroy(&handle);
      throw;
    }
    if (handle.cudaStream != stream || handle.cublasHandle != blas || handle.cusolverHandle != solver) {
      hyacinDestroy(&handle);
      throw std::runtime_error("Hyacin replaced the borrowed BLR stream or library handles");
    }
  }

  ~BlockedDenseCustom() override { hyacinDestroy(&handle); }
  BlockedDenseCustom(const BlockedDenseCustom&) = delete;
  BlockedDenseCustom& operator=(const BlockedDenseCustom&) = delete;

  BLR toBLR(const std::vector<long long>& leafOffsets, long long firstCell, double tolerance) override {
    thrust::device_vector<double> singularValues;
    thrust::device_vector<cuDoubleComplex> basis;
    auto result = toBLRWith(leafOffsets, firstCell, tolerance,
      [&](int rows, int cols, const cuDoubleComplex* tile, double eps) {
        const int fullRank = std::min(rows, cols);
        singularValues.resize(fullRank);
        basis.resize(static_cast<std::size_t>(cols) * fullRank);
        return hyacinTsvdFitSegmentN(&handle, eps, 'N', rows, cols, fullRank, tile, tile_rows,
          deviceData(singularValues), deviceData(basis), cols, rows, HYACIN_F64_COMPLEX);
      },
      [&](int rows, int cols, int rank, const cuDoubleComplex* tile, cuDoubleComplex* left,
          cuDoubleComplex* right) {
        const auto* v = deviceData(basis);
        const cuDoubleComplex one = make_cuDoubleComplex(1., 0.), zero{};
        checkBlas(cublasZgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, rows, rank, cols, &one,
                              tile, tile_rows, v, cols, &zero, left, rows));
        copyDevice(right, v, static_cast<std::size_t>(cols) * rank, stream);
      });
    hyacinSync_TimerSegments(&handle, nullptr, 0);
    return result;
  }
};

}  // namespace

int main(int argc, char* argv[]) try {
  MPI_Init(&argc, &argv);
  const bool passed = runBenchmark<BlockedDenseCustom>(argc, argv);
  MPI_Finalize();
  return passed ? EXIT_SUCCESS : EXIT_FAILURE;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  MPI_Abort(MPI_COMM_WORLD, 1);
  return 1;
}
