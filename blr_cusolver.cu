#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>
#include <mpi.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

// Both complex types store two adjacent doubles in real, imaginary order.
static_assert(sizeof(std::complex<double>) == sizeof(cuDoubleComplex));
static_assert(sizeof(cuDoubleComplex) == 2 * sizeof(double));

template <class T>
static inline T* deviceData(thrust::device_vector<T>& data) {
  return thrust::raw_pointer_cast(data.data());
}

template <class T>
static inline const T* deviceData(const thrust::device_vector<T>& data) {
  return thrust::raw_pointer_cast(data.data());
}

static inline int checkedInt(long long n) {
  if (n < 0 || n > std::numeric_limits<int>::max())
    throw std::overflow_error("Dense matrix dimension exceeds the library's int range");
  return static_cast<int>(n);
}

static inline void checkCuda(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}

static inline void checkBlas(cublasStatus_t status) {
  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error(cublasGetStatusString(status));
}

static inline void checkSolver(cusolverStatus_t status, const char* operation) {
  if (status != CUSOLVER_STATUS_SUCCESS)
    throw std::runtime_error(std::string(operation) + " failed with status " + std::to_string(status));
}

static inline void copyToHost(void* dst, const void* src, size_t bytes, cudaStream_t stream) {
  if (bytes == 0) return;
  checkCuda(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, stream));
  checkCuda(cudaStreamSynchronize(stream));
}

static inline void copyDevice(cuDoubleComplex* dst, const cuDoubleComplex* src, size_t count, cudaStream_t stream) {
  if (count > 0 && dst != src)
    checkCuda(cudaMemcpyAsync(dst, src, count * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice, stream));
}

static inline void copyRectangle(void* dst, long long dstStride, const void* src, long long srcStride,
                                 long long rows, long long cols, cudaMemcpyKind kind, cudaStream_t stream) {
  checkCuda(cudaMemcpy2DAsync(dst, dstStride * sizeof(cuDoubleComplex), src,
                               srcStride * sizeof(cuDoubleComplex), rows * sizeof(cuDoubleComplex), cols, kind, stream));
}

// Both matrices are viewed in column-major order.
static inline void transposeBlock(cublasHandle_t blas, const cuDoubleComplex* src,
                                  long long rows, long long cols, long long srcStride,
                                  cuDoubleComplex* dst, long long dstStride) {
  const cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0);
  const cuDoubleComplex zero = make_cuDoubleComplex(0.0, 0.0);
  checkBlas(cublasZgeam_64(blas, CUBLAS_OP_T, CUBLAS_OP_N, cols, rows, &one, src, srcStride,
                             &zero, dst, dstStride, dst, dstStride));
}

struct SvdWorkspace {
  thrust::device_vector<cuDoubleComplex> vh, work, diagonal;
  thrust::device_vector<double> singularValues;
  thrust::device_vector<int> info;
  std::vector<double> singularValuesHost, tails;
  std::vector<std::complex<double>> diagonalHost;

  SvdWorkspace() : info(1) {}
};

// GESVD overwrites input with the left singular vectors of the packed m-by-n matrix.
static inline void factorizeSvd(cusolverDnHandle_t solver, int m, int n, cuDoubleComplex* input,
                         SvdWorkspace& scratch, cudaStream_t stream) {
  int lwork = 0;
  checkSolver(cusolverDnZgesvd_bufferSize(solver, m, n, &lwork), "cusolverDnZgesvd_bufferSize");
  scratch.vh.resize(static_cast<size_t>(n) * n);
  scratch.work.resize(lwork);
  scratch.singularValues.resize(n);
  scratch.singularValuesHost.resize(n);
  checkSolver(cusolverDnZgesvd(solver, 'O', 'S', m, n, input, m, deviceData(scratch.singularValues), nullptr, 1,
                           deviceData(scratch.vh), n, deviceData(scratch.work), lwork, nullptr,
                           deviceData(scratch.info)), "cusolverDnZgesvd");
  int info;
  checkCuda(cudaMemcpyAsync(&info, deviceData(scratch.info), sizeof(info), cudaMemcpyDeviceToHost, stream));
  copyToHost(scratch.singularValuesHost.data(), deviceData(scratch.singularValues), n * sizeof(double), stream);
  if (info != 0)
    throw std::runtime_error("cusolverDnZgesvd returned info " + std::to_string(info));
}

static inline void scaleSvdColumns(cublasHandle_t blas, int rows, int rank, const cuDoubleComplex* input,
                            cuDoubleComplex* output, SvdWorkspace& scratch, cudaStream_t stream) {
  const auto& values = scratch.singularValuesHost;
  scratch.diagonalHost.assign(values.begin(), values.begin() + rank);
  scratch.diagonal.resize(rank);
  auto* diagonal = deviceData(scratch.diagonal);
  checkCuda(cudaMemcpyAsync(diagonal, scratch.diagonalHost.data(), rank * sizeof(cuDoubleComplex),
                            cudaMemcpyHostToDevice, stream));
  checkBlas(cublasZdgmm(blas, CUBLAS_SIDE_RIGHT, rows, rank, input, rows, diagonal, 1, output, rows));
}

class BLR {
public:
  const std::int32_t gridM, gridN;
  std::vector<std::array<std::int32_t, 3>> dims;
  // Tiles are U * transpose(VT). At full rank U holds the dense tile and VT is empty.
  std::vector<thrust::device_vector<cuDoubleComplex>> U, VT;

  // The caller owns both handles and keeps cuBLAS bound to the stream.
  BLR(std::int32_t localCells, std::int32_t leafCells, cublasHandle_t blas, cudaStream_t stream);
  void matVec(const std::vector<long long>& leafOffsets, long long firstCell,
              const std::complex<double>* x, std::complex<double>* localY) const;

private:
  const cublasHandle_t blas;
  const cudaStream_t stream;
};

class BlockedDense {
public:
  const std::int32_t tile_rows, tile_cols, gridM, gridN;
  // Zero-padded column-major tiles, stored in tile-row order with gridN tiles per row.
  thrust::device_vector<cuDoubleComplex> A;

  // The caller owns the handles and keeps cuBLAS and cuSOLVER bound to the stream.
  BlockedDense(std::int32_t tileRows, std::int32_t tileCols, std::int32_t localCells, std::int32_t leafCells,
               cublasHandle_t blas, cusolverDnHandle_t solver, cudaStream_t stream);
  std::size_t tileSize() const;
  void matVec(const std::vector<long long>& leafOffsets, long long firstCell,
              const std::complex<double>* x, std::complex<double>* localY) const;
  // Consumes the dense tile storage after building the BLR matrix.
  BLR toBLR(const std::vector<long long>& leafOffsets, long long firstCell, double tolerance);

private:
  const cublasHandle_t blas;
  const cusolverDnHandle_t solver;
  const cudaStream_t stream;
};

BLR::BLR(std::int32_t localCells, std::int32_t leafCells, cublasHandle_t blas, cudaStream_t stream)
  : gridM(localCells), gridN(leafCells), dims(static_cast<std::size_t>(gridM) * gridN),
    U(dims.size()), VT(dims.size()), blas(blas), stream(stream) {}

void BLR::matVec(const std::vector<long long>& leafOffsets, long long firstCell,
                 const std::complex<double>* x, std::complex<double>* localY) const {
  if (gridM == 0) return;

  int maxRank = 0;
  for (const auto& [rows, cols, rank] : dims)
    if (rank < std::min(rows, cols))
      maxRank = std::max(maxRank, rank);

  const long long firstRow = leafOffsets[firstCell];
  thrust::device_vector<cuDoubleComplex> deviceX(leafOffsets.back()), intermediate(maxRank);
  thrust::device_vector<cuDoubleComplex> deviceY(leafOffsets[firstCell + gridM] - firstRow, cuDoubleComplex{});
  checkCuda(cudaMemcpyAsync(deviceData(deviceX), x, deviceX.size() * sizeof(cuDoubleComplex),
                            cudaMemcpyHostToDevice, stream));

  const cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0);
  const cuDoubleComplex zero = make_cuDoubleComplex(0.0, 0.0);
  std::size_t index = 0;
  for (int i = 0; i < gridM; ++i) {
    auto* y = deviceData(deviceY) + (leafOffsets[firstCell + i] - firstRow);
    for (int j = 0; j < gridN; ++j, ++index) {
      const auto [rows, cols, rank] = dims[index];
      if (rank == 0) continue;

      const auto* blockX = deviceData(deviceX) + leafOffsets[j];
      const auto* left = deviceData(U[index]);
      if (rank == std::min(rows, cols)) {
        checkBlas(cublasZgemv(blas, CUBLAS_OP_N, rows, cols, &one, left, rows, blockX, 1, &one, y, 1));
      } else {
        auto* tmp = deviceData(intermediate);
        checkBlas(cublasZgemv(blas, CUBLAS_OP_T, cols, rank, &one, deviceData(VT[index]), cols,
                              blockX, 1, &zero, tmp, 1));
        checkBlas(cublasZgemv(blas, CUBLAS_OP_N, rows, rank, &one, left, rows, tmp, 1, &one, y, 1));
      }
    }
  }
  copyToHost(localY, deviceData(deviceY), deviceY.size() * sizeof(cuDoubleComplex), stream);
}

BlockedDense::BlockedDense(std::int32_t tileRows, std::int32_t tileCols, std::int32_t localCells,
                           std::int32_t leafCells, cublasHandle_t blas, cusolverDnHandle_t solver, cudaStream_t stream)
  : tile_rows(tileRows), tile_cols(tileCols), gridM(localCells), gridN(leafCells),
    A(tileSize() * gridN * gridM, make_cuDoubleComplex(0.0, 0.0)), blas(blas), solver(solver), stream(stream) {}

std::size_t BlockedDense::tileSize() const {
  return static_cast<std::size_t>(tile_rows) * tile_cols;
}

void BlockedDense::matVec(const std::vector<long long>& leafOffsets, long long firstCell,
                         const std::complex<double>* x, std::complex<double>* localY) const {
  if (gridM == 0) return;
  if (A.empty())
    throw std::logic_error("BlockedDense tile storage has already been consumed");

  const long long firstRow = leafOffsets[firstCell];
  thrust::device_vector<cuDoubleComplex> deviceX(leafOffsets.back());
  thrust::device_vector<cuDoubleComplex> deviceY(leafOffsets[firstCell + gridM] - firstRow);
  checkCuda(cudaMemcpyAsync(deviceData(deviceX), x, deviceX.size() * sizeof(cuDoubleComplex),
                            cudaMemcpyHostToDevice, stream));

  const cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0);
  const cuDoubleComplex zero = make_cuDoubleComplex(0.0, 0.0);
  const auto* tile = deviceData(A);
  for (int i = 0; i < gridM; ++i) {
    const long long start = leafOffsets[firstCell + i];
    const int rows = checkedInt(leafOffsets[firstCell + i + 1] - start);
    auto* y = deviceData(deviceY) + (start - firstRow);
    for (int j = 0; j < gridN; ++j, tile += tileSize()) {
      const int cols = checkedInt(leafOffsets[j + 1] - leafOffsets[j]);
      checkBlas(cublasZgemv(blas, CUBLAS_OP_N, rows, cols, &one, tile, tile_rows,
                            deviceData(deviceX) + leafOffsets[j], 1, j == 0 ? &zero : &one, y, 1));
    }
  }
  copyToHost(localY, deviceData(deviceY), deviceY.size() * sizeof(cuDoubleComplex), stream);
}

BLR BlockedDense::toBLR(const std::vector<long long>& leafOffsets, long long firstCell, double tolerance) {
  if (gridM > 0 && A.empty())
    throw std::logic_error("BlockedDense tile storage has already been consumed");
  if (!std::isfinite(tolerance) || tolerance < 0.)
    throw std::invalid_argument("SVD tolerance must be finite and nonnegative");

  BLR result(gridM, gridN, blas, stream);
  if (gridM == 0) return result;

  thrust::device_vector<cuDoubleComplex> svdInput;
  SvdWorkspace scratch;
  const auto* tile = deviceData(A);
  std::size_t index = 0;
  for (int i = 0; i < gridM; ++i) {
    const long long globalRow = firstCell + i;
    const int rows = checkedInt(leafOffsets[globalRow + 1] - leafOffsets[globalRow]);
    for (int j = 0; j < gridN; ++j, ++index, tile += tileSize()) {
      const int cols = checkedInt(leafOffsets[j + 1] - leafOffsets[j]);
      const int fullRank = std::min(rows, cols);
      const std::size_t denseSize = static_cast<std::size_t>(rows) * cols;
      auto& left = result.U[index];
      auto& right = result.VT[index];
      int rank = fullRank;
      const bool wide = rows < cols;

      if (globalRow != j) {
        // GESVD requires m >= n. Transposing a wide tile preserves the U * transpose(VT) contract.
        svdInput.resize(denseSize);
        auto* input = deviceData(svdInput);
        if (wide)
          transposeBlock(blas, tile, rows, cols, tile_rows, input, cols);
        else
          copyRectangle(input, rows, tile, tile_rows, rows, cols, cudaMemcpyDeviceToDevice, stream);
        factorizeSvd(solver, std::max(rows, cols), fullRank, input, scratch, stream);

        const auto& singularValues = scratch.singularValuesHost;
        const double scale = singularValues.front();
        rank = 0;
        if (scale > 0.) {
          auto& tails = scratch.tails;
          tails.resize(fullRank);
          std::transform_inclusive_scan(singularValues.rbegin(), singularValues.rend(), tails.rbegin(),
                                        std::plus<double>(), [scale](double value) {
            const double normalized = value / scale;
            return normalized * normalized;
          });
          const double total = tails.front();
          rank = static_cast<int>(std::find_if(tails.begin(), tails.end(), [total, tolerance](double tail) {
            return std::sqrt(tail / total) <= tolerance;
          }) - tails.begin());
        }
        if ((static_cast<std::size_t>(rows) + cols) * rank >= denseSize)
          rank = fullRank;
      }

      result.dims[index] = {rows, cols, rank};
      if (rank == fullRank) {
        left.resize(denseSize);
        copyRectangle(deviceData(left), rows, tile, tile_rows, rows, cols, cudaMemcpyDeviceToDevice, stream);
      } else if (rank > 0) {
        left.resize(static_cast<std::size_t>(rows) * rank);
        right.resize(static_cast<std::size_t>(cols) * rank);
        const auto* vh = deviceData(scratch.vh);
        const auto* input = deviceData(svdInput);
        if (wide) {
          transposeBlock(blas, vh, rank, rows, fullRank, deviceData(left), rows);
          scaleSvdColumns(blas, rows, rank, deviceData(left), deviceData(left), scratch, stream);
          copyDevice(deviceData(right), input, right.size(), stream);
        } else {
          scaleSvdColumns(blas, rows, rank, input, deviceData(left), scratch, stream);
          transposeBlock(blas, vh, rank, cols, fullRank, deviceData(right), cols);
        }
      }
      checkCuda(cudaStreamSynchronize(stream));
    }
  }
  thrust::device_vector<cuDoubleComplex>{}.swap(A);
  return result;
}

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct Cell {
  std::array<long long, 2> N{};
  std::array<double, 3> R{};
};

class MatrixAccessor {
public:
  virtual std::complex<double> operator()(double d) const = 0;
};

class Laplace3D : public MatrixAccessor {
public:
  double singularity;
  Laplace3D (double s) : singularity(1. / s) {}
  std::complex<double> operator()(double d) const override {
    if (d == 0.)
      return std::complex<double>(singularity, 0.);
    else
      return std::complex<double>(-1. / d, 0.);
  }
};

class Yukawa3D : public MatrixAccessor {
public:
  double singularity, alpha;
  Yukawa3D (double s, double a) : singularity(1. / s), alpha(a) {}
  std::complex<double> operator()(double d) const override {
    if (d == 0.)
      return std::complex<double>(singularity, 0.);
    else
      return std::complex<double>(std::exp(-alpha * d) / d, 0.);
  }
};

class Gaussian : public MatrixAccessor {
public:
  double alpha;
  Gaussian (double a) : alpha(a) {}
  std::complex<double> operator()(double d) const override {
    if (d == 0.)
      return std::complex<double>(1.0001, 0.);
    return std::complex<double>(std::exp(- alpha * d * d), 0.);
  }
};

class Helmholtz3D : public MatrixAccessor {
public:
  double k;
  double singularity;
  Helmholtz3D(double wave_number, double s) : k(wave_number), singularity(1. / s) {}
  std::complex<double> operator()(double d) const override {
    if (d == 0.)
      return std::complex<double>(singularity, 0.);
    else
      return std::complex<double>(-std::cos(k * d), std::sin(k * d)) / d;
  }
};

[[maybe_unused]] static inline void uniform_unit_cube(double* bodies, long long nbodies, long long dim) {
  long long side = std::ceil(std::pow(nbodies, 1. / dim));
  long long lens[3] = { dim > 0 ? side : 1, dim > 1 ? side : 1, dim > 2 ? side : 1 };
  double step = 1. / side;

  for (long long i = 0; i < lens[0]; ++i)
    for (long long j = 0; j < lens[1]; ++j)
       for (long long k = 0; k < lens[2]; ++k) {
    long long x = k + lens[2] * (j + lens[1] * i);
    if (x < nbodies) {
      bodies[x * 3] = i * step;
      bodies[x * 3 + 1] = j * step;
      bodies[x * 3 + 2] = k * step;
    }
  }
}

[[maybe_unused]] static inline void uniform_unit_cube_rnd(double* bodies, long long nbodies, long long dim,
                                                        unsigned long long seed) {
  std::mt19937_64 gen(seed);
  std::uniform_real_distribution<double> uniform_dist(0., 1.);

  std::array<double, 3>* b3 = reinterpret_cast<std::array<double, 3>*>(bodies);
  std::array<double, 3>* b3_end = reinterpret_cast<std::array<double, 3>*>(&bodies[3 * nbodies]);
  std::for_each(b3, b3_end, [&](std::array<double, 3>& body) {
    for (int i = 0; i < 3; i++)
      body[i] = i < dim ? uniform_dist(gen) : 0.;
  });
}

static inline void mesh_sphere(double* bodies, long long nbodies) {
  const double phi = M_PI * (3. - std::sqrt(5.));  // golden angle in radians

  for (long long i = 0; i < nbodies; ++i) {
    const double y = nbodies == 1 ? 1. : 1. - 2. * ((double)i / (double)(nbodies - 1));

    // Note: setting constant radius = 1 will produce a cylindrical shape
    const double radius = std::sqrt(1. - y * y);  // radius at y
    const double theta = (double)i * phi;

    const double x = radius * std::cos(theta);
    const double z = radius * std::sin(theta);
    bodies[i * 3] = x;
    bodies[i * 3 + 1] = y;
    bodies[i * 3 + 2] = z;
  }
}

static inline void gen_matrix(const MatrixAccessor& eval, long long m, long long n, const double* bi, const double* bj, std::complex<double> Aij[]) {
  const std::array<double, 3>* bi3 = reinterpret_cast<const std::array<double, 3>*>(bi);
  const std::array<double, 3>* bi3_end = reinterpret_cast<const std::array<double, 3>*>(&bi[3 * m]);
  const std::array<double, 3>* bj3 = reinterpret_cast<const std::array<double, 3>*>(bj);
  const std::array<double, 3>* bj3_end = reinterpret_cast<const std::array<double, 3>*>(&bj[3 * n]);

  std::for_each(bj3, bj3_end, [&](const std::array<double, 3>& j) -> void {
    long long ix = std::distance(bj3, &j);
    std::for_each(bi3, bi3_end, [&](const std::array<double, 3>& i) -> void {
      long long iy = std::distance(bi3, &i);
      std::array<double, 3> d({ i[0] - j[0], i[1] - j[1], i[2] - j[2] });
      double d_nrm = std::sqrt(std::inner_product(d.begin(), d.end(), d.begin(), 0.));
      Aij[iy + ix * m] = eval(d_nrm);
    });
  });
}

static inline void getBounds(const double* bodies, long long nbodies, std::array<double, 3>& radii) {
  const auto* first = reinterpret_cast<const std::array<double, 3>*>(bodies);
  const auto* last = first + nbodies;
  for (int i = 0; i < 3; ++i) {
    const auto bounds = std::minmax_element(first, last, [i](const auto& x, const auto& y) {
      return x[i] < y[i];
    });
    const double min = (*bounds.first)[i], max = (*bounds.second)[i];
    radii[i] = (min == max && min == 0.) ? 0. : (max - min) * 0.5 + 1.e-8;
  }
}

static inline void buildBinaryTree(Cell* cells, double* bodies, long long nbodies, long long levels) {
  cells[0].N[0] = 0;
  cells[0].N[1] = nbodies;
  getBounds(bodies, nbodies, cells[0].R);

  long long nleaf = 1ll << levels;
  for (long long i = 0; i < nleaf - 1; i++) {
    Cell& ci = cells[i];
    long long sdim = std::distance(ci.R.begin(), std::max_element(ci.R.begin(), ci.R.end()));
    long long i_begin = ci.N[0];
    long long i_end = ci.N[1];

    std::array<double, 3>* bodies3 = reinterpret_cast<std::array<double, 3>*>(&bodies[i_begin * 3]);
    std::array<double, 3>* bodies3_end = reinterpret_cast<std::array<double, 3>*>(&bodies[i_end * 3]);
    std::sort(bodies3, bodies3_end,
      [=](std::array<double, 3>& i, std::array<double, 3>& j) { return i[sdim] < j[sdim]; });

    long long len = (i << 1) + 1;
    Cell& c0 = cells[len];
    Cell& c1 = cells[len + 1];
    long long loc = i_begin + (i_end - i_begin) / 2;
    c0.N[0] = i_begin;
    c0.N[1] = loc;
    c1.N[0] = loc;
    c1.N[1] = i_end;

    getBounds(bodies + i_begin * 3, loc - i_begin, c0.R);
    getBounds(bodies + loc * 3, i_end - loc, c1.R);
  }
}

static inline void selectCudaDevice(MPI_Comm world) {
  MPI_Comm local;
  MPI_Comm_split_type(world, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local);
  int rank;
  MPI_Comm_rank(local, &rank);
  MPI_Comm_free(&local);

  int deviceCount;
  checkCuda(cudaGetDeviceCount(&deviceCount));
  if (deviceCount == 0)
    throw std::runtime_error("No visible CUDA devices");
  checkCuda(cudaSetDevice(rank % deviceCount));
}

static inline bool runBenchmark(int argc, char* argv[]) {
  long long Nbody = argc > 1 ? std::atoll(argv[1]) : 2048;
  long long leaf_size = argc > 2 ? std::atoll(argv[2]) : 256;
  const double tolerance = argc > 3 ? std::atof(argv[3]) : 1e-10;
  if (Nbody <= 0 || Nbody > std::numeric_limits<int>::max() || leaf_size <= 0)
    throw std::invalid_argument("Matrix order must fit in a positive int and leaf size must be positive");
  if (!std::isfinite(tolerance) || tolerance < 0.)
    throw std::invalid_argument("SVD tolerance must be finite and nonnegative");

  leaf_size = Nbody < leaf_size ? Nbody : leaf_size;
  long long levels = 0;
  for (long long leafCount = Nbody / leaf_size; leafCount > 1; leafCount /= 2)
    ++levels;
  long long Nleaf = (long long)1 << levels;
  long long ncells = Nleaf + Nleaf - 1;

  const MPI_Comm world = MPI_COMM_WORLD;
  int mpi_rank, mpi_size;
  MPI_Comm_rank(world, &mpi_rank);
  MPI_Comm_size(world, &mpi_size);
  selectCudaDevice(world);

  cudaStream_t streamHandle;
  checkCuda(cudaStreamCreateWithFlags(&streamHandle, cudaStreamNonBlocking));
  const std::unique_ptr<std::remove_pointer_t<cudaStream_t>, decltype(&cudaStreamDestroy)>
      stream(streamHandle, cudaStreamDestroy);

  cublasHandle_t blasHandle;
  checkBlas(cublasCreate(&blasHandle));
  const std::unique_ptr<std::remove_pointer_t<cublasHandle_t>, decltype(&cublasDestroy)>
      blas(blasHandle, cublasDestroy);
  checkBlas(cublasSetStream(blas.get(), stream.get()));

  cusolverDnHandle_t solverHandle;
  checkSolver(cusolverDnCreate(&solverHandle), "cusolverDnCreate");
  const std::unique_ptr<std::remove_pointer_t<cusolverDnHandle_t>, decltype(&cusolverDnDestroy)>
      solver(solverHandle, cusolverDnDestroy);
  checkSolver(cusolverDnSetStream(solver.get(), stream.get()), "cusolverDnSetStream");

  //Laplace3D eval(1.);
  //Yukawa3D eval(1, 1.);
  //Gaussian eval(0.005);
  Helmholtz3D eval(4., 1.E0);

  std::vector<double> body(Nbody * 3);
  std::vector<std::complex<double>> Xbody(Nbody);
  std::vector<Cell> cells(ncells);

  mesh_sphere(&body[0], Nbody);
  //uniform_unit_cube_rnd(&body[0], Nbody, 3, 999);
  //uniform_unit_cube(&body[0], Nbody, 3);
  buildBinaryTree(cells.data(), body.data(), Nbody, levels);
  cells.erase(cells.begin(), cells.begin() + (Nleaf - 1));

  const long long cellsPerRank = Nleaf / mpi_size;
  const long long remainder = Nleaf % mpi_size;
  const long long localCellBegin = mpi_rank * cellsPerRank + std::min<long long>(mpi_rank, remainder);
  const long long localCellCount = cellsPerRank + (mpi_rank < remainder);
  const long long localCellEnd = localCellBegin + localCellCount;

  std::vector<long long> leafOffsets(Nleaf + 1);
  std::int32_t tileRows = 0, tileCols = 0;
  for (long long j = 0; j < Nleaf; ++j) {
    leafOffsets[j] = cells[j].N[0];
    const auto bodyCount = static_cast<std::int32_t>(cells[j].N[1] - cells[j].N[0]);
    tileCols = std::max(tileCols, bodyCount);
    if (j >= localCellBegin && j < localCellEnd)
      tileRows = std::max(tileRows, bodyCount);
  }
  leafOffsets.back() = cells.back().N[1];
  const long long localRowBegin = leafOffsets[localCellBegin];
  const long long localRowEnd = leafOffsets[localCellEnd];
  const long long localRowCount = localRowEnd - localRowBegin;

  std::mt19937_64 gen;
  std::uniform_real_distribution<double> uniform_dist(0., 1.);
  std::generate(Xbody.begin(), Xbody.end(), [&]() { return std::complex<double>(uniform_dist(gen), uniform_dist(gen)); });

  std::vector<std::complex<double>> blockedY(localRowCount), blrY(localRowCount);

  BlockedDense blockedA(tileRows, tileCols, localCellCount, Nleaf, blas.get(), solver.get(), stream.get());
  auto* tile = deviceData(blockedA.A);
  std::vector<std::complex<double>> hostTile(blockedA.tileSize());
  for (long long i = localCellBegin; i < localCellEnd; ++i) {
    const long long rows = leafOffsets[i + 1] - leafOffsets[i];
    for (long long j = 0; j < Nleaf; ++j, tile += blockedA.tileSize()) {
      const long long cols = leafOffsets[j + 1] - leafOffsets[j];
      gen_matrix(eval, rows, cols, body.data() + 3 * leafOffsets[i], body.data() + 3 * leafOffsets[j],
                 hostTile.data());
      copyRectangle(tile, blockedA.tile_rows, hostTile.data(), rows, rows, cols, cudaMemcpyHostToDevice, stream.get());
      checkCuda(cudaStreamSynchronize(stream.get()));
    }
  }
  MPI_Barrier(world);
  const double blockedStart = MPI_Wtime();
  blockedA.matVec(leafOffsets, localCellBegin, Xbody.data(), blockedY.data());
  const double blockedTime = MPI_Wtime() - blockedStart;

  MPI_Barrier(world);
  const double compressionStart = MPI_Wtime();
  const BLR blr = blockedA.toBLR(leafOffsets, localCellBegin, tolerance);
  const double compressionTime = MPI_Wtime() - compressionStart;

  MPI_Barrier(world);
  const double blrStart = MPI_Wtime();
  blr.matVec(leafOffsets, localCellBegin, Xbody.data(), blrY.data());
  const double blrTime = MPI_Wtime() - blrStart;

  double norms[2] = {};
  for (long long i = 0; i < localRowCount; ++i) {
    norms[0] += std::norm(blrY[i] - blockedY[i]);
    norms[1] += std::norm(blockedY[i]);
  }
  MPI_Allreduce(MPI_IN_PLACE, norms, 2, MPI_DOUBLE, MPI_SUM, world);
  const double error = norms[1] == 0. ? std::sqrt(norms[0]) : std::sqrt(norms[0] / norms[1]);

  long long localStats[2] = {};
  for (std::size_t index = 0; index < blr.dims.size(); ++index) {
    const auto [rows, cols, rank] = blr.dims[index];
    localStats[0] += rank < std::min(rows, cols);
    localStats[1] += blr.U[index].size() + blr.VT[index].size();
  }
  long long stats[2];
  MPI_Reduce(localStats, stats, 2, MPI_LONG_LONG, MPI_SUM, 0, world);

  const double localTimes[] = {blockedTime, compressionTime, blrTime};
  double maxTimes[3];
  MPI_Reduce(localTimes, maxTimes, 3, MPI_DOUBLE, MPI_MAX, 0, world);

  if (mpi_rank == 0) {
    const long long denseScalars = Nbody * Nbody;
    std::ostringstream storagePercent;
    storagePercent << std::fixed << std::setprecision(2) << 100. * stats[1] / denseScalars;
    std::cout << "MPI processes: " << mpi_size << '\n'
              << "Matrix order: " << Nbody << " Leaf size: " << leaf_size << " Leaf cells: " << Nleaf << '\n'
              << "Leaf cells per rank: " << cellsPerRank << '-' << cellsPerRank + (remainder != 0) << '\n'
              << "SVD relative Frobenius tolerance: " << tolerance << '\n'
              << "Compressed tiles: " << stats[0] << " / " << Nleaf * Nleaf << '\n'
              << "BLR scalars stored: " << stats[1] << " / " << denseScalars
              << " (" << storagePercent.str() << "%)" << '\n'
              << "BLR matvec relative error: " << error << '\n'
              << "Blocked dense matvec time: " << maxTimes[0] << '\n'
              << "BLR construction time: " << maxTimes[1] << '\n'
              << "BLR matvec time: " << maxTimes[2] << std::endl;
  }

  return std::isfinite(error);
}

}  // namespace

int main(int argc, char* argv[]) try {
  MPI_Init(&argc, &argv);
  const bool passed = runBenchmark(argc, argv);
  MPI_Finalize();
  return passed ? EXIT_SUCCESS : EXIT_FAILURE;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  MPI_Abort(MPI_COMM_WORLD, 1);
  return 1;
}
