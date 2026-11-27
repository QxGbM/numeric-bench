#pragma once

#include <complex>
#include <vector>
#include <array>
#include <map>
#include <set>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <memory>
#include <functional>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <cassert>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>
#include <nccl.h>
#include <mpi.h>

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/inner_product.h>
#include <thrust/scan.h>
#include <thrust/system/cuda/execution_policy.h>

// Status checks must remain active in Release builds as well.
template <class Status>
static inline void checkHssStatus(Status status, Status success, const char* call, const char* file, int line) {
  if (status != success)
    throw std::runtime_error(std::string(file) + ":" + std::to_string(line) + ": " + call +
                             " returned status=" + std::to_string(static_cast<int>(status)));
}
#define CUDA_CHECK(call) checkHssStatus((call), cudaSuccess, #call, __FILE__, __LINE__)
#define CUBLAS_CHECK(call) checkHssStatus((call), CUBLAS_STATUS_SUCCESS, #call, __FILE__, __LINE__)
#define CUSOLVER_CHECK(call) checkHssStatus((call), CUSOLVER_STATUS_SUCCESS, #call, __FILE__, __LINE__)
#define NCCL_CHECK(call) checkHssStatus((call), ncclSuccess, #call, __FILE__, __LINE__)

template <class T>
constexpr bool hssComplex = std::is_same_v<T, std::complex<double>>;

template <class T>
using HssDeviceScalar = std::conditional_t<hssComplex<T>, cuDoubleComplex, double>;

template <class T>
constexpr cudaDataType hssCudaType = hssComplex<T> ? CUDA_C_64F : CUDA_R_64F;

template <class T>
constexpr size_t hssDoublesPerScalar = sizeof(T) / sizeof(double);

template <class T>
class DenseMatrix {
protected:
  using DeviceScalar = HssDeviceScalar<T>;
  // Each local row is stored as one conjugated column with leading dimension N.
  thrust::device_vector<DeviceScalar> A;
  cublasHandle_t blas;
  cusolverDnHandle_t solver;
  cudaStream_t stream;

  template <class CompressBlock>
  void compressWith(std::vector<long long>& recipe, std::vector<T>& basis,
                    ncclComm_t ncclPrime, ncclComm_t ncclSub, CompressBlock&& compressBlock);

public:
  using Scalar = T;
  static_assert(std::is_same_v<T, double> || hssComplex<T>);

  thrust::device_vector<long long> rowOffsets;
  long long N; int prime_rank;
  double eps; // Public, caller-adjustable construction tolerance; zero retains all singular values.
  // The caller owns the stream and both handles, and binds the handles to the stream.
  DenseMatrix(long long localM, long long N, ncclComm_t ncclPrime, ncclComm_t ncclSub,
              cublasHandle_t blas, cusolverDnHandle_t solver, cudaStream_t stream);
  void matVecA(const DenseMatrix& matA, T* B, ncclComm_t ncclPrime, ncclComm_t ncclSub) const;
  void adjointA(long long localN, ncclComm_t ncclPrime, ncclComm_t ncclSub);
  virtual void compress(std::vector<long long>& recipe, std::vector<T>& basis,
                ncclComm_t ncclPrime, ncclComm_t ncclSub);
  // Each new replica group contains complete old groups, including their representatives.
  void redistribute(long long localM, ncclComm_t ncclPrimeOld, ncclComm_t ncclPrimeNew, ncclComm_t ncclSubNew);

  void AijSetBlock(long long mA, long long nA, long long iA, long long jA, const T* A_in, long long stride);
  void AijExtractBlock(long long mA, long long nA, long long iA, long long jA, T* A_out, long long stride);
  virtual void iterateTolerance() {}
  virtual ~DenseMatrix() = default;

};

class Cell {
public:
  std::array<long long, 2> Child, N;
  std::array<double, 3> R, C;

  Cell() : Child(std::array<long long, 2>{ -1, -1 }), N(std::array<long long, 2>{ -1, -1 }), R(std::array<double, 3>{ 0., 0., 0. }), C(std::array<double, 3>{ 0., 0., 0. }) {}
};

class CSR {
public:
  long long N;
  std::vector<long long> RowIndex, ColIndex;

  CSR(const std::vector<Cell>& cells);
  CSR(const CSR& global, long long lbegin, long long lend, const std::vector<std::array<long long, 3>>& boxes);
};

class HierarchicalComm {
public:
  long long Proc, NeighborComms, NeighborLoc, LocalChild;
  std::vector<std::array<long long, 3>> Boxes;
  // A color of -1 means the rank is excluded from the group.
  int colorPrime, colorSub;

  HierarchicalComm(const std::vector<Cell>& cells, std::vector<std::pair<long long, long long>>& Mapping,
                   const long long CRows[], const long long CCols[], int world_rank);

  long long oLocal() const { return Boxes[Proc][2] - Boxes[Proc][1]; }
  long long oGlobal() const { return Boxes[Proc][0]; }
  long long lenLocal() const { return Boxes[Proc][1]; }
  long long lenNeighbors() const { return Boxes.back()[2]; }
  long long oLocalChild(const HierarchicalComm& lower) const { return lower.oLocal() - LocalChild; }

  void bcastNeighbor(void* data, const long long noffsets[], size_t size, MPI_Comm prime, MPI_Comm sub) const;
};

template <class T>
class HSSLevel {
public:
  std::vector<long long> URowPrefixes;
  std::vector<long long> UColPrefixes;
  std::vector<long long> VRowPrefixes;
  std::vector<long long> VColPrefixes;

  std::vector<long long> UPrefixes;
  std::vector<long long> VPrefixes;
  std::vector<long long> APrefixes;
  std::vector<long long> CPrefixes;

  std::vector<T> U;
  std::vector<T> V;
  std::vector<T> A;
  std::vector<T> C;

  void eval(DenseMatrix<T>& matA, const long long localM[], const long long localN[],
            const CSR& rels, const HierarchicalComm& comm, MPI_Comm prime, MPI_Comm sub,
            ncclComm_t ncclPrime, ncclComm_t ncclSub, long long localNoffset = -1ll);
  void matVecUpwardPass(const T* X_in, T* Z,
                        const HierarchicalComm& comm, MPI_Comm prime, MPI_Comm sub) const;
  void matVecHorizontalandDownwardPass(T* Y_out, const T* Z, T* Y, const CSR& rels, const HierarchicalComm& comm) const;
  void matVecLeafHorizontalPass(T* X_io, const T* Z, T* Y,
                                const CSR& rels, const HierarchicalComm& comm, MPI_Comm prime, MPI_Comm sub) const;
};

template <class T>
class HSS {
public:
  long long levels;
  std::vector<HierarchicalComm> comm;
  std::vector<CSR> Far_local;
  // Owned communicators, stored as prime/sub pairs by level.
  std::vector<MPI_Comm> mpi_communicators;
  std::vector<ncclComm_t> nccl_communicators;

  std::vector<HSSLevel<T>> A;
  std::pair<long long, long long> local_bodies;

  HSS(const std::vector<Cell>& cells, long long levels, MPI_Comm world, ncclComm_t ncclWorld);
  MPI_Comm prime(long long level) const { return mpi_communicators[2 * level]; }
  MPI_Comm sub(long long level) const { return mpi_communicators[2 * level + 1]; }
  ncclComm_t ncclPrime(long long level) const { return nccl_communicators[2 * level]; }
  ncclComm_t ncclSub(long long level) const { return nccl_communicators[2 * level + 1]; }
  void evalMatVecA(DenseMatrix<T>& matA, const Cell cells[]);
  void allocRHSwork(std::vector<long long>& iwork, std::vector<T>& cwork) const;

  void matVecMul(T X[]) const;
  void matVecMul(T X[], const long long iwork[], T cwork[]) const;
  double solveRelErr(const T X[], const T ref[]);

  // Free the owned communicators before MPI_Finalize.
  void free_all_comms();
};

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

static inline void copyToHost(void* dst, const void* src, size_t bytes, cudaStream_t stream) {
  if (bytes == 0) return;
  CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

template <class T>
static inline void copyDevice(T* dst, const T* src, size_t count, cudaStream_t stream) {
  if (count > 0 && dst != src)
    CUDA_CHECK(cudaMemcpyAsync(dst, src, count * sizeof(T), cudaMemcpyDeviceToDevice, stream));
}

template <class T, class U>
static inline void copyRectangle(T* dst, long long dstStride, const U* src, long long srcStride,
                                 long long rows, long long cols, cudaMemcpyKind kind, cudaStream_t stream) {
  static_assert(sizeof(T) == sizeof(U));
  CUDA_CHECK(cudaMemcpy2DAsync(dst, dstStride * sizeof(T), src,
                               srcStride * sizeof(T), rows * sizeof(T), cols, kind, stream));
}

// Both matrices are viewed in column-major order.
template <class T>
static inline void adjointBlock(cublasHandle_t blas, const T* src, long long rows, long long cols,
                                long long srcStride, T* dst, long long dstStride) {
  if constexpr (std::is_same_v<T, double>) {
    const double one = 1., zero = 0.;
    CUBLAS_CHECK(cublasDgeam_64(blas, CUBLAS_OP_T, CUBLAS_OP_N, cols, rows, &one, src, srcStride,
                               &zero, dst, dstStride, dst, dstStride));
  } else {
    const cuDoubleComplex one = make_cuDoubleComplex(1., 0.), zero{};
    CUBLAS_CHECK(cublasZgeam_64(blas, CUBLAS_OP_C, CUBLAS_OP_N, cols, rows, &one, src, srcStride,
                               &zero, dst, dstStride, dst, dstStride));
  }
}

// Compact left factor used by the right-basis-only backends.
template <class T>
static inline void multiplyRightBasis(cublasHandle_t blas, int rows, int cols, int rank,
                                      const T* block, const T* basis, T* result) {
  if constexpr (std::is_same_v<T, double>) {
    const double one = 1., zero = 0.;
    CUBLAS_CHECK(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, rows, rank, cols,
                            &one, block, rows, basis, cols, &zero, result, rows));
  } else {
    const cuDoubleComplex one = make_cuDoubleComplex(1., 0.), zero{};
    CUBLAS_CHECK(cublasZgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, rows, rank, cols,
                            &one, block, rows, basis, cols, &zero, result, rows));
  }
}

static inline void bcastDevice(void* buffer, size_t count, ncclDataType_t type, ncclComm_t comm, cudaStream_t stream) {
  if (count == 0) return;
  NCCL_CHECK(ncclBroadcast(buffer, buffer, count, type, 0, comm, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

static inline long long maxPartition(const thrust::device_vector<long long>& offsets, cudaStream_t stream) {
  return thrust::inner_product(thrust::cuda::par.on(stream), offsets.begin() + 1, offsets.end(), offsets.begin(), 0ll,
                               thrust::maximum<long long>{}, thrust::minus<long long>{});
}

static inline void allGatherRowOffsets(thrust::device_vector<long long>& offsets, long long localRows, int primeRank, ncclComm_t prime,
                                       cudaStream_t stream) {
  offsets[primeRank + 1] = localRows;
  // Leave offsets[0] at zero and place each rank's count in its receive slot.
  auto* counts = deviceData(offsets) + 1;
  NCCL_CHECK(ncclAllGather(counts + primeRank, counts, 1, ncclInt64, prime, stream));
  thrust::inclusive_scan(thrust::cuda::par.on(stream), offsets.begin() + 1, offsets.end(), offsets.begin() + 1);
}

static inline thrust::device_vector<long long> gatherRowOffsets(long long localM, ncclComm_t prime, ncclComm_t sub, int& primeRank,
                                          cudaStream_t stream) {
  thrust::device_vector<long long> offsets(2);
  primeRank = 0;
  if (prime != nullptr) {
    int size;
    NCCL_CHECK(ncclCommUserRank(prime, &primeRank));
    NCCL_CHECK(ncclCommCount(prime, &size));
    offsets.resize(static_cast<size_t>(size) + 1);
    allGatherRowOffsets(offsets, localM, primeRank, prime, stream);
  } else if (sub == nullptr) {
    offsets[1] = localM;
  }
  if (sub != nullptr) {
    thrust::device_vector<int> layout(2);
    if (prime != nullptr) {
      layout[0] = primeRank;
      layout[1] = checkedInt(offsets.size());
    }
    bcastDevice(deviceData(layout), layout.size(), ncclInt32, sub, stream);
    if (prime == nullptr) {
      primeRank = layout[0];
      offsets.resize(layout[1]);
    }
    bcastDevice(deviceData(offsets), offsets.size(), ncclInt64, sub, stream);
  }
  return offsets;
}

template <class T>
struct SvdWorkspace {
  using DeviceScalar = HssDeviceScalar<T>;
  thrust::device_vector<DeviceScalar> adjoint, vh, work, diagonal;
  thrust::device_vector<double> singularValues;
  thrust::device_vector<int> info;
  std::vector<double> singularValuesHost, tails;
  std::vector<T> diagonalHost;

  SvdWorkspace() : info(1) {}
};

template <class T>
static inline int compressBlock(cusolverDnHandle_t solver, cublasHandle_t blas, double eps, int rows, int cols,
                                HssDeviceScalar<T>* block, SvdWorkspace<T>& scratch, thrust::device_vector<HssDeviceScalar<T>>& basis, cudaStream_t stream) {
  using DeviceScalar = HssDeviceScalar<T>;
  const bool wide = rows < cols;
  const int m = std::max(rows, cols), n = std::min(rows, cols);
  DeviceScalar* input = block;
  if (wide) {
    scratch.adjoint.resize(static_cast<size_t>(m) * n);
    input = deviceData(scratch.adjoint);
    adjointBlock(blas, block, rows, cols, rows, input, m);
  }

  int lwork = 0;
  if constexpr (hssComplex<T>)
    CUSOLVER_CHECK(cusolverDnZgesvd_bufferSize(solver, m, n, &lwork));
  else
    CUSOLVER_CHECK(cusolverDnDgesvd_bufferSize(solver, m, n, &lwork));
  scratch.vh.resize(static_cast<size_t>(n) * n);
  scratch.work.resize(lwork);
  scratch.singularValues.resize(n);
  auto* vh = deviceData(scratch.vh);
  auto* singularValues = deviceData(scratch.singularValues);
  if constexpr (hssComplex<T>)
    CUSOLVER_CHECK(cusolverDnZgesvd(solver, 'O', 'S', m, n, input, m, singularValues, nullptr, 1,
                                  vh, n, deviceData(scratch.work), lwork, nullptr, deviceData(scratch.info)));
  else
    CUSOLVER_CHECK(cusolverDnDgesvd(solver, 'O', 'S', m, n, input, m, singularValues, nullptr, 1,
                                  vh, n, deviceData(scratch.work), lwork, nullptr, deviceData(scratch.info)));
  int info;
  copyToHost(&info, deviceData(scratch.info), sizeof(info), stream);
  if (info != 0)
    throw std::runtime_error("cuSOLVER GESVD returned info=" + std::to_string(info));

  auto& values = scratch.singularValuesHost;
  auto& tails = scratch.tails;
  values.resize(n);
  tails.resize(n);
  copyToHost(values.data(), singularValues, n * sizeof(double), stream);
  std::transform_inclusive_scan(values.rbegin(), values.rend(), tails.rbegin(), std::plus<double>(),
                                [](double value) { return value * value; });
  const double cutoff = tails.front() * std::pow(eps, 2);
  const int rank = static_cast<int>(std::find_if(tails.begin(), tails.end(),
      [cutoff](double tail) { return tail < cutoff; }) - tails.begin());
  if (rank == 0) return rank;

  // For a wide block, the adjoint's U is the original V and its VH is the original UH.
  const long long basisCount = static_cast<long long>(cols) * rank;
  const auto offset = basis.size();
  basis.resize(offset + basisCount);
  auto* right = deviceData(basis) + offset;
  if (wide) {
    adjointBlock(blas, vh, rank, rows, n, block, rows);
    copyDevice(right, input, basisCount, stream);
  } else {
    adjointBlock(blas, vh, rank, cols, n, right, cols);
  }
  if constexpr (hssComplex<T>) {
    scratch.diagonalHost.assign(values.begin(), values.begin() + rank);
    scratch.diagonal.resize(rank);
    auto* diagonal = deviceData(scratch.diagonal);
    CUDA_CHECK(cudaMemcpyAsync(diagonal, scratch.diagonalHost.data(), rank * sizeof(DeviceScalar),
                               cudaMemcpyHostToDevice, stream));
    CUBLAS_CHECK(cublasZdgmm(blas, CUBLAS_SIDE_RIGHT, rows, rank, block, rows, diagonal, 1, block, rows));
  } else {
    CUBLAS_CHECK(cublasDdgmm(blas, CUBLAS_SIDE_RIGHT, rows, rank, block, rows, singularValues, 1, block, rows));
  }

  return rank;
}

template <class T>
inline DenseMatrix<T>::DenseMatrix(long long localM, long long N, ncclComm_t ncclPrime, ncclComm_t ncclSub,
                         cublasHandle_t blas, cusolverDnHandle_t solver, cudaStream_t stream)
  : blas(blas), solver(solver), stream(stream), N(N), eps(0.) {
  rowOffsets = gatherRowOffsets(localM, ncclPrime, ncclSub, prime_rank, stream);
  A.resize((rowOffsets[prime_rank + 1] - rowOffsets[prime_rank]) * N);
}

template <class T>
inline void DenseMatrix<T>::matVecA(const DenseMatrix& matA, T* B, ncclComm_t ncclPrime, ncclComm_t ncclSub) const {
  assert(rowOffsets.back() == matA.N && N == 1ll);
  assert(prime_rank == matA.prime_rank && rowOffsets.size() == matA.rowOffsets.size());
  if (ncclPrime == nullptr && ncclSub == nullptr) return;

  const auto outputSize = matA.rowOffsets[matA.prime_rank + 1] - matA.rowOffsets[matA.prime_rank];
  thrust::device_vector<DeviceScalar> X, Y(outputSize, DeviceScalar{});
  auto* y = deviceData(Y);
  if (ncclPrime != nullptr) {
    const int maxN = checkedInt(maxPartition(rowOffsets, stream));
    const long long primeSize = rowOffsets.size() - 1;
    if (maxN > 0) {
      X.resize(primeSize * maxN);
      auto* x = deviceData(X);
      auto* localX = x + static_cast<long long>(prime_rank) * maxN;
      // Conjugate the stored RHS directly into its padded all-gather slot.
      if (!A.empty())
        adjointBlock(blas, deviceData(A), A.size(), 1, A.size(), localX, 1);
      NCCL_CHECK(ncclAllGather(localX, x, hssDoublesPerScalar<T> * static_cast<size_t>(maxN), ncclDouble, ncclPrime, stream));

      if (outputSize > 0) {
        const thrust::host_vector<long long> offsets = rowOffsets;
        const int outputs = checkedInt(outputSize), lda = checkedInt(matA.N);
        const auto* matrix = deviceData(matA.A);
        bool first = true;
        for (long long rank = 0; rank < primeSize; ++rank) {
          const int inputs = checkedInt(offsets[rank + 1] - offsets[rank]);
          if (inputs == 0) continue;
          if constexpr (hssComplex<T>) {
            const cuDoubleComplex one = make_cuDoubleComplex(1., 0.);
            const cuDoubleComplex beta = first ? cuDoubleComplex{} : one;
            CUBLAS_CHECK(cublasZgemv(blas, CUBLAS_OP_C, inputs, outputs, &one,
                                    matrix + offsets[rank], lda, x + rank * maxN, 1, &beta, y, 1));
          } else {
            const double one = 1., beta = first ? 0. : 1.;
            CUBLAS_CHECK(cublasDgemv(blas, CUBLAS_OP_T, inputs, outputs, &one,
                                    matrix + offsets[rank], lda, x + rank * maxN, 1, &beta, y, 1));
          }
          first = false;
        }
      } else {
        CUDA_CHECK(cudaStreamSynchronize(stream));
      }
    }
  }
  if (ncclSub != nullptr)
    bcastDevice(y, hssDoublesPerScalar<T> * Y.size(), ncclDouble, ncclSub, stream);
  copyToHost(B, y, Y.size() * sizeof(DeviceScalar), stream);
}

template <class T>
inline void DenseMatrix<T>::adjointA(long long localN, ncclComm_t ncclPrime, ncclComm_t ncclSub) {
  if (ncclPrime != nullptr) {
    thrust::device_vector<long long> newOffsets(rowOffsets.size());
    allGatherRowOffsets(newOffsets, localN, prime_rank, ncclPrime, stream);
    const thrust::host_vector<long long> oldHostOffsets = rowOffsets, newHostOffsets = newOffsets;
    assert(newHostOffsets.back() == N);

    const long long maxM = maxPartition(rowOffsets, stream), maxN = maxPartition(newOffsets, stream);
    const long long primeSize = oldHostOffsets.size() - 1;
    const long long localM = oldHostOffsets[prime_rank + 1] - oldHostOffsets[prime_rank];
    const long long newN = oldHostOffsets.back();
    if (maxM > 0 && maxN > 0) {
      const size_t maxElements = std::numeric_limits<size_t>::max() / sizeof(DeviceScalar);
      if (static_cast<size_t>(maxM) > maxElements / maxN / primeSize)
        throw std::overflow_error("Dense adjoint exchange exceeds the addressable size");
      const size_t count = static_cast<size_t>(maxM) * maxN;
      thrust::device_vector<DeviceScalar> exchange(primeSize * count);
      auto* packed = deviceData(exchange);
      if (localM > 0) {
        const auto* matrix = deviceData(A);
        for (long long rank = 0; rank < primeSize; ++rank) {
          const long long cols = newHostOffsets[rank + 1] - newHostOffsets[rank];
          if (cols > 0)
            adjointBlock(blas, matrix + newHostOffsets[rank], cols, localM, N, packed + rank * count, maxM);
        }
      }

      // Reuse A for the out-of-place receive, then unpack back into the send buffer.
      CUDA_CHECK(cudaStreamSynchronize(stream));
      A.clear();
      A.resize(exchange.size());
      auto* received = deviceData(A);
      NCCL_CHECK(ncclAlltoAll(packed, received, hssDoublesPerScalar<T> * count, ncclDouble, ncclPrime, stream));
      if (localN > 0) {
        for (long long rank = 0; rank < primeSize; ++rank) {
          const long long rows = oldHostOffsets[rank + 1] - oldHostOffsets[rank];
          if (rows > 0)
            copyRectangle(packed + oldHostOffsets[rank], newN, received + rank * count, maxM, rows, localN,
                          cudaMemcpyDeviceToDevice, stream);
        }
      }
      CUDA_CHECK(cudaStreamSynchronize(stream));
      exchange.resize(localN * newN);
      A.swap(exchange);
    } else {
      A.clear();
    }

    N = newN;
    rowOffsets = std::move(newOffsets);
  }
  if (ncclSub != nullptr) {
    if (ncclPrime != nullptr)
      rowOffsets[0] = N;
    bcastDevice(deviceData(rowOffsets), rowOffsets.size(), ncclInt64, ncclSub, stream);
    N = rowOffsets[0];
    rowOffsets[0] = 0;
    A.resize((rowOffsets[prime_rank + 1] - rowOffsets[prime_rank]) * N);
    bcastDevice(deviceData(A), hssDoublesPerScalar<T> * A.size(), ncclDouble, ncclSub, stream);
  }
}

template <class T>
inline void DenseMatrix<T>::compress(std::vector<long long>& recipe, std::vector<T>& basis,
                           ncclComm_t ncclPrime, ncclComm_t ncclSub) {
  if (!std::isfinite(eps) || eps < 0.)
    throw std::invalid_argument("DenseMatrix::eps must be finite and nonnegative");
  std::unique_ptr<SvdWorkspace<T>> scratch;
  compressWith(recipe, basis, ncclPrime, ncclSub,
    [&](int rows, int cols, DeviceScalar* block, thrust::device_vector<DeviceScalar>& deviceBasis) {
      if (!scratch) scratch = std::make_unique<SvdWorkspace<T>>();
      return compressBlock(solver, blas, eps, rows, cols, block, *scratch, deviceBasis, stream);
    });
}

template <class T>
template <class CompressBlock>
inline void DenseMatrix<T>::compressWith(std::vector<long long>& recipe, std::vector<T>& basis,
                               ncclComm_t ncclPrime, ncclComm_t ncclSub, CompressBlock&& compressBlock) {
  basis.clear();
  if (N == 0) return;
  assert(std::reduce(recipe.begin(), recipe.end()) == rowOffsets[prime_rank + 1] - rowOffsets[prime_rank]);

  thrust::device_vector<DeviceScalar> deviceBasis, compact;
  deviceBasis.reserve(std::transform_reduce(recipe.begin(), recipe.end(), 0ll, std::plus<long long>(), [](long long r) { return r * r; }));
  if (ncclPrime != nullptr) {
    long long inputRow = 0, outputRow = 0;
    if (!A.empty()) {
      const int storedRows = checkedInt(N);
      auto* matrix = deviceData(A);
      for (long long& blockRows : recipe) {
        if (blockRows == 0) continue;
        auto* src = matrix + inputRow * N;
        const int rank = compressBlock(storedRows, checkedInt(blockRows), src, deviceBasis);
        if (rank > 0 && outputRow < inputRow) {
          const long long count = rank * N;
          if (outputRow + rank > inputRow) {
            compact.resize(count);
            copyDevice(deviceData(compact), src, count, stream);
            src = deviceData(compact);
          }
          copyDevice(matrix + outputRow * N, src, count, stream);
        }
        // Finish using the Thrust work buffers before resizing or reusing them.
        CUDA_CHECK(cudaStreamSynchronize(stream));
        inputRow += blockRows;
        outputRow += rank;
        blockRows = rank;
      }
    }

    A.resize(outputRow * N);
    allGatherRowOffsets(rowOffsets, outputRow, prime_rank, ncclPrime, stream);
  }
  if (ncclSub != nullptr) {
    const auto offsetCount = rowOffsets.size();
    const auto offsetBytes = (offsetCount - 1) * sizeof(long long);
    const auto recipeBytes = recipe.size() * sizeof(long long);
    thrust::device_vector<long long> metadata(offsetCount + recipe.size());
    auto* packed = deviceData(metadata);
    // Slot zero carries the basis length; the remaining offsets are followed by the recipe.
    if (ncclPrime != nullptr) {
      metadata[0] = deviceBasis.size();
      CUDA_CHECK(cudaMemcpyAsync(packed + 1, deviceData(rowOffsets) + 1, offsetBytes, cudaMemcpyDeviceToDevice, stream));
      if (!recipe.empty())
        CUDA_CHECK(cudaMemcpyAsync(packed + offsetCount, recipe.data(), recipeBytes, cudaMemcpyHostToDevice, stream));
    }
    bcastDevice(packed, metadata.size(), ncclInt64, ncclSub, stream);
    if (ncclPrime == nullptr) {
      deviceBasis.resize(metadata[0]);
      CUDA_CHECK(cudaMemcpyAsync(deviceData(rowOffsets) + 1, packed + 1, offsetBytes, cudaMemcpyDeviceToDevice, stream));
      if (!recipe.empty())
        CUDA_CHECK(cudaMemcpyAsync(recipe.data(), packed + offsetCount, recipeBytes, cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    bcastDevice(deviceData(deviceBasis), hssDoublesPerScalar<T> * deviceBasis.size(), ncclDouble, ncclSub, stream);
    A.resize((rowOffsets[prime_rank + 1] - rowOffsets[prime_rank]) * N);
    bcastDevice(deviceData(A), hssDoublesPerScalar<T> * A.size(), ncclDouble, ncclSub, stream);
  }
  basis.resize(deviceBasis.size());
  copyToHost(basis.data(), deviceData(deviceBasis), basis.size() * sizeof(DeviceScalar), stream);
  A.shrink_to_fit();
}

template <class T>
inline void DenseMatrix<T>::redistribute(long long localM, ncclComm_t ncclPrimeOld, ncclComm_t ncclPrimeNew, ncclComm_t ncclSubNew) {
  int primeRankNew;
  auto rowOffsetsNew = gatherRowOffsets(localM, ncclPrimeNew, ncclSubNew, primeRankNew, stream);
  assert(rowOffsets.back() == rowOffsetsNew.back());

  const bool sameLayout = prime_rank == primeRankNew && rowOffsets == rowOffsetsNew;
  if (!sameLayout) {
    const long long srcBegin = rowOffsets[prime_rank];
    const long long dstBegin = rowOffsetsNew[primeRankNew];
    const long long dstEnd = rowOffsetsNew[primeRankNew + 1];
    const long long lo = std::max(srcBegin, dstBegin);
    const long long hi = std::min<long long>(rowOffsets[prime_rank + 1], dstEnd);
    const long long count = std::max(0ll, hi - lo) * N;
    thrust::device_vector<DeviceScalar> B((dstEnd - dstBegin) * N, DeviceScalar{});
    if (count > 0 && (ncclSubNew == nullptr || ncclPrimeOld != nullptr))
      copyDevice(deviceData(B) + (lo - dstBegin) * N, deviceData(A) + (lo - srcBegin) * N, count, stream);
    if (ncclSubNew != nullptr && !B.empty()) {
      auto* buffer = deviceData(B);
      NCCL_CHECK(ncclAllReduce(buffer, buffer, hssDoublesPerScalar<T> * B.size(), ncclDouble, ncclSum, ncclSubNew, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    A.swap(B);
    rowOffsets = std::move(rowOffsetsNew);
    prime_rank = primeRankNew;
  }
}

template <class T>
inline void DenseMatrix<T>::AijSetBlock(long long mA, long long nA, long long iA, long long jA,
                              const T* A_in, long long stride) {
  const long long rowBegin = std::max<long long>(rowOffsets[prime_rank], iA);
  const long long colBegin = std::max(0ll, jA);
  const long long rows = std::min<long long>(rowOffsets[prime_rank + 1], iA + mA) - rowBegin;
  const long long cols = std::min(N, jA + nA) - colBegin;
  if (rows <= 0 || cols <= 0) return;

  thrust::device_vector<DeviceScalar> block(rows * cols);
  const auto* src = A_in + (rowBegin - iA) + (colBegin - jA) * stride;
  auto* dst = deviceData(A) + (rowBegin - rowOffsets[prime_rank]) * N + colBegin;
  copyRectangle(deviceData(block), rows, src, stride, rows, cols, cudaMemcpyHostToDevice, stream);
  adjointBlock(blas, deviceData(block), rows, cols, rows, dst, N);
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

template <class T>
inline void DenseMatrix<T>::AijExtractBlock(long long mA, long long nA, long long iA, long long jA,
                                  T* A_out, long long stride) {
  const long long rowBegin = std::max<long long>(rowOffsets[prime_rank], iA);
  const long long colBegin = std::max(0ll, jA);
  const long long rows = std::min<long long>(rowOffsets[prime_rank + 1], iA + mA) - rowBegin;
  const long long cols = std::min(N, jA + nA) - colBegin;
  if (rows <= 0 || cols <= 0) return;

  thrust::device_vector<DeviceScalar> block(rows * cols);
  auto* src = deviceData(A) + (rowBegin - rowOffsets[prime_rank]) * N + colBegin;
  auto* dst = A_out + (rowBegin - iA) + (colBegin - jA) * stride;
  adjointBlock(blas, src, cols, rows, N, deviceData(block), rows);
  copyRectangle(dst, stride, deviceData(block), rows, rows, cols, cudaMemcpyDeviceToHost, stream);
  CUDA_CHECK(cudaMemset2DAsync(src, N * sizeof(DeviceScalar), 0, cols * sizeof(DeviceScalar), rows, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

inline CSR::CSR(const std::vector<Cell>& cells) : N(cells.size()) {
  std::vector<std::pair<long long, long long>> rels;
  auto getList = [&](long long i, long long j, auto&& self) -> void {
    bool admis = i != j;
    if (admis) { rels.emplace_back(i, j); }
    if (!admis && cells[i].Child[0] >= 0 && cells[j].Child[0] >= 0)
      for (long long k = cells[i].Child[0]; k < cells[i].Child[1]; ++k)
        for (long long l = cells[j].Child[0]; l < cells[j].Child[1]; ++l)
          self(k, l, self);
  };
  getList(0, 0, getList);
  std::sort(rels.begin(), rels.end());

  long long len = rels.size();
  RowIndex.resize(N + 1ll);
  ColIndex.resize(len);

  RowIndex[0] = 0;
  std::transform(rels.begin(), rels.end(), ColIndex.begin(), [](const std::pair<long long, long long>& i) { return i.second; });
  std::for_each(rels.begin(), rels.end(), [&](const std::pair<long long, long long>& i) { ++RowIndex[i.first + 1]; });
  std::inclusive_scan(RowIndex.begin() + 1, RowIndex.end(), RowIndex.begin() + 1);
}

inline CSR::CSR(const CSR& global, long long lbegin, long long lend, const std::vector<std::array<long long, 3>>& boxes) : N(boxes.back()[2]) {
  std::map<long long, long long> local_map;
  for (const std::array<long long, 3>& box : boxes)
    for (long long i = 0; i < box[1]; ++i)
      local_map.emplace(box[0] + i, box[2] - box[1] + i);

  long long offset = global.RowIndex[lbegin];
  RowIndex.resize(lend - lbegin + 1);
  ColIndex.resize(global.RowIndex[lend] - global.RowIndex[lbegin]);
  std::transform(&(global.RowIndex)[lbegin], &(global.RowIndex)[lend + 1], RowIndex.begin(), [=](long long i) { return i - offset; });
  std::transform(&(global.ColIndex)[global.RowIndex[lbegin]], &(global.ColIndex)[global.RowIndex[lend]], ColIndex.begin(), [&](long long j) { return local_map[j]; });
}

inline HierarchicalComm::HierarchicalComm(const std::vector<Cell>& cells, std::vector<std::pair<long long, long long>>& Mapping,
                                   const long long CRows[], const long long CCols[], int world_rank) {
  long long pbegin = Mapping[world_rank].first, pend = Mapping[world_rank].second;
  long long p = std::distance(Mapping.begin(), std::find(Mapping.begin(), Mapping.begin() + world_rank, Mapping[world_rank]));
  long long lenp = std::distance(Mapping.begin() + p,
    std::find_if_not(Mapping.begin() + p, Mapping.end(), [&](const std::pair<long long, long long>& i) { return i == Mapping[world_rank]; }));

  std::map<long long, long long> col_to_mpi_rank;
  std::for_each(Mapping.begin(), Mapping.end(), [&](std::pair<long long, long long>& i) {
    for (long long col = i.first; col < i.second; ++col) col_to_mpi_rank.emplace(col, (long long)std::distance(&Mapping[0], &i)); });

  std::set<long long> cols; cols.insert(col_to_mpi_rank[pbegin]);
  std::for_each(&CCols[CRows[pbegin]], &CCols[CRows[pend]], [&](long long col) { cols.insert(col_to_mpi_rank[col]); });

  Proc = std::distance(cols.begin(), std::find(cols.begin(), cols.end(), p));
  NeighborComms = (p == world_rank) ? (cols.size() - 1ll) : 0ll;
  NeighborLoc = -Proc; const std::pair<long long, long long>* curr = &Mapping[world_rank];
  std::for_each(Mapping.begin(), Mapping.begin() + world_rank, [&](const std::pair<long long, long long>& i) { if (*curr != i) { curr = &i; ++NeighborLoc; }});
  Boxes.reserve(cols.size()); long long lenBoxes = 0; for (long long i : cols)
  { long long ibegin = Mapping[i].first, iend = Mapping[i].second; Boxes.emplace_back(std::array<long long, 3>({ ibegin, iend - ibegin, lenBoxes += iend - ibegin })); }

  colorPrime = world_rank == p ? 0 : -1;
  colorSub = lenp > 1 ? static_cast<int>(p) : -1;

  long long i = 0;
  std::vector<std::pair<long long, long long>> MappingNext(Mapping.size(), std::make_pair(-1ll, -1ll));
  while (i < (long long)Mapping.size()) {
    long long lenP = std::distance(Mapping.begin() + i, std::find_if_not(Mapping.begin() + i, Mapping.end(), [&](std::pair<long long, long long> a) { return a == Mapping[i]; }));
    long long child = cells[Mapping[i].first].Child[0], lenC = cells[Mapping[i].second - 1ll].Child[1] - child;
    if (child >= 0 && lenC > 0)
      for (long long j = 0; j < lenP; ++j) {
        long long c0 = j * lenC / lenP;
        long long c1 = (j + 1ll) * lenC / lenP;
        c1 = std::max(c1, c0 + 1ll);
        MappingNext[i + j] = std::make_pair(child + c0, child + c1);
      }
    i += lenP;
  }
  Mapping = std::move(MappingNext);
  LocalChild = Mapping[world_rank].first - cells[pbegin].Child[0];
}

inline void HierarchicalComm::bcastNeighbor(void* data, const long long noffsets[], size_t size, MPI_Comm prime, MPI_Comm sub) const {
  std::vector<MPI_Request> requests(NeighborComms * 2);

  long long send_loc = noffsets[Boxes[Proc][2] - Boxes[Proc][1]];
  long long send_len = (noffsets[Boxes[Proc][2]] - send_loc) * size;
  unsigned char* send_start = &reinterpret_cast<unsigned char*>(data)[send_loc * size];

  for (long long p = 0; p < NeighborComms; ++p) {
    long long dst = p + (long long)(Proc <= p);
    long long recv_loc = noffsets[Boxes[dst][2] - Boxes[dst][1]];
    long long recv_len = (noffsets[Boxes[dst][2]] - recv_loc) * size;
    unsigned char* recv_start = &reinterpret_cast<unsigned char*>(data)[recv_loc * size];

    long long dst_rank = NeighborLoc + dst;
    MPI_Isend(send_start, send_len, MPI_BYTE, dst_rank, 0, prime, &requests[p << 1]);
    MPI_Irecv(recv_start, recv_len, MPI_BYTE, dst_rank, 0, prime, &requests[(p << 1) | 1]);
  }

  if (0 < requests.size())
    MPI_Waitall(requests.size(), &requests[0], MPI_STATUSES_IGNORE);
  if (sub != MPI_COMM_NULL)
    MPI_Bcast(data, noffsets[Boxes.back()[2]] * size, MPI_BYTE, 0, sub);
}

template <class T>
inline void HSSLevel<T>::eval(DenseMatrix<T>& matA, const long long localM[], const long long localN[],
                    const CSR& rels, const HierarchicalComm& comm, MPI_Comm prime, MPI_Comm sub,
                    ncclComm_t ncclPrime, ncclComm_t ncclSub, long long localNoffset) {
  long long ibegin = comm.oLocal(), nodes = comm.lenLocal(), xlen = comm.lenNeighbors();
  URowPrefixes.resize(xlen + 1); UColPrefixes.resize(xlen + 1);
  VRowPrefixes.resize(xlen + 1); VColPrefixes.resize(xlen + 1);
  UPrefixes.resize(nodes + 1); VPrefixes.resize(nodes + 1);
  CPrefixes.resize(rels.RowIndex[nodes] + 1);

  if (0ll <= localNoffset) {
    APrefixes.resize(nodes + 1);
    std::transform(localM, &localM[nodes], localN, APrefixes.begin() + 1, std::multiplies<long long>());
    std::inclusive_scan(APrefixes.begin() + 1, APrefixes.end(), APrefixes.begin() + 1);
    APrefixes[0] = 0ll; A.resize(APrefixes.back());

    long long localMoffset = matA.rowOffsets[matA.prime_rank];
    for (long long i = 0ll; i < nodes; ++i) {
      long long M = localM[i], N = localN[i];
      matA.AijExtractBlock(M, N, localMoffset, localNoffset, &A[APrefixes[i]], M);
      localMoffset += M; localNoffset += N;
    }
  }

  if (rels.RowIndex[nodes]) {
    URowPrefixes[0] = UColPrefixes[0] = VRowPrefixes[0] = VColPrefixes[0] = UPrefixes[0] = VPrefixes[0] = CPrefixes[0] = 0ll;
    std::vector<long long> DimsMLr(&localM[0], &localM[nodes]), DimsNLr(&localN[0], &localN[nodes]);
    matA.compress(DimsMLr, U, ncclPrime, ncclSub);
    matA.adjointA(std::reduce(DimsNLr.begin(), DimsNLr.end()), ncclPrime, ncclSub);
    matA.compress(DimsNLr, V, ncclPrime, ncclSub);
    long long colOffset = matA.rowOffsets[matA.prime_rank];
    matA.adjointA(std::reduce(DimsMLr.begin(), DimsMLr.end()), ncclPrime, ncclSub);

    std::transform(&localM[0], &localM[nodes], &DimsMLr[0], UPrefixes.begin() + 1, std::multiplies<long long>());
    std::transform(&localN[0], &localN[nodes], &DimsNLr[0], VPrefixes.begin() + 1, std::multiplies<long long>());
    std::inclusive_scan(UPrefixes.begin() + 1, UPrefixes.end(), UPrefixes.begin() + 1);
    std::inclusive_scan(VPrefixes.begin() + 1, VPrefixes.end(), VPrefixes.begin() + 1);

    std::vector<std::array<long long, 4>> dims(xlen + 1);
    for (long long i = 0ll; i < nodes; ++i) { dims[i + ibegin] = std::array<long long, 4>{ localM[i], localN[i], DimsMLr[i], DimsNLr[i] }; }
    std::vector<long long> seq(xlen + 1); std::iota(seq.begin(), seq.end(), 0ll);
    comm.bcastNeighbor(dims.data(), seq.data(), sizeof(std::array<long long, 4>), prime, sub);
    std::inclusive_scan(dims.begin(), dims.end(), dims.begin(),
      [](const std::array<long long, 4>& a, const std::array<long long, 4>& b) { return std::array<long long, 4>{ a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3] }; });

    for (long long i = 0ll; i < xlen; ++i) {
      long long i1 = i + 1ll; const auto& [m, n, k, l] = dims[i];
      URowPrefixes[i1] = m; VRowPrefixes[i1] = n; UColPrefixes[i1] = k; VColPrefixes[i1] = l;
    }

    for (long long i = 0ll; i < nodes; ++i)
      std::transform(rels.ColIndex.begin() + rels.RowIndex[i], rels.ColIndex.begin() + rels.RowIndex[i + 1], CPrefixes.begin() + rels.RowIndex[i] + 1,
        [&](long long col) { return (UColPrefixes[i + ibegin + 1] - UColPrefixes[i + ibegin]) * (VColPrefixes[col + 1] - VColPrefixes[col]); });
    std::inclusive_scan(CPrefixes.begin() + 1, CPrefixes.end(), CPrefixes.begin() + 1);
    C.resize(CPrefixes.back());

    long long rowOffset = matA.rowOffsets[matA.prime_rank] - UColPrefixes[ibegin]; colOffset -= VColPrefixes[ibegin];
    for (long long i = 0; i < nodes; ++i) {
      long long mC = UColPrefixes[i + ibegin + 1] - UColPrefixes[i + ibegin];
      long long y = rowOffset + UColPrefixes[ibegin + i];
      for (long long ij = rels.RowIndex[i]; ij < rels.RowIndex[i + 1] && 0 < mC; ++ij) {
        long long j = rels.ColIndex[ij];
        long long nC = VColPrefixes[j + 1] - VColPrefixes[j];
        long long x = colOffset + VColPrefixes[j];
        if (nC > 0) matA.AijExtractBlock(mC, nC, y, x, &C[CPrefixes[ij]], mC);
      }
    }
  } else {
    std::vector<std::array<long long, 2>> dims(xlen + 1);
    for (long long i = 0ll; i < nodes; ++i) { dims[i + ibegin] = std::array<long long, 2>{ localM[i], localN[i] }; }
    std::vector<long long> seq(xlen + 1); std::iota(seq.begin(), seq.end(), 0ll);
    comm.bcastNeighbor(dims.data(), seq.data(), sizeof(std::array<long long, 2>), prime, sub);
    std::inclusive_scan(dims.begin(), dims.end(), dims.begin(),
      [](const std::array<long long, 2>& a, const std::array<long long, 2>& b) { return std::array<long long, 2>{ a[0] + b[0], a[1] + b[1] }; });

    URowPrefixes[0] = VRowPrefixes[0] = CPrefixes[0] = 0ll;
    for (long long i = 0ll; i < xlen; ++i) {
      long long i1 = i + 1ll; const auto& [m, n] = dims[i];
      URowPrefixes[i1] = m; VRowPrefixes[i1] = n;
    }

    std::fill(UColPrefixes.begin(), UColPrefixes.end(), 0ll);
    std::fill(VColPrefixes.begin(), VColPrefixes.end(), 0ll);
    std::fill(UPrefixes.begin(), UPrefixes.end(), 0ll);
    std::fill(VPrefixes.begin(), VPrefixes.end(), 0ll);
  }
}

extern "C" void zgemv_(const char*, const int*, const int*, const std::complex<double>*, const std::complex<double>*, const int*, const std::complex<double>*, const int*, const std::complex<double>*, std::complex<double>*, const int*) noexcept;
extern "C" void dgemv_(const char*, const int*, const int*, const double*, const double*, const int*, const double*, const int*, const double*, double*, const int*) noexcept;

template <class T>
static inline void hssHostGemv(char trans, int m, int n, const T* A, const T* x, T beta, T* y) {
  const int inc = 1;
  const T one(1.);
  if constexpr (hssComplex<T>)
    zgemv_(&trans, &m, &n, &one, A, &m, x, &inc, &beta, y, &inc);
  else
    dgemv_(&trans, &m, &n, &one, A, &m, x, &inc, &beta, y, &inc);
}

template <class T>
inline void HSSLevel<T>::matVecUpwardPass(const T* X_in, T* Z,
                                const HierarchicalComm& comm, MPI_Comm prime, MPI_Comm sub) const {
  long long ibegin = comm.oLocal(), nodes = comm.lenLocal();
  auto gemv = [](int m, int n, const T* a, const T* x, T* b) {
    hssHostGemv(hssComplex<T> ? 'C' : 'T', n, m, a, x, T{}, b);
  };

  for (long long i = 0ll; i < nodes; ++i) {
    long long M = VColPrefixes[i + ibegin + 1] - VColPrefixes[i + ibegin];
    long long N = VRowPrefixes[i + ibegin + 1] - VRowPrefixes[i + ibegin];
    if (0 < M && 0 < N)
    { gemv(M, N, &V[VPrefixes[i]], &X_in[VRowPrefixes[i + ibegin] - VRowPrefixes[ibegin]], &Z[VColPrefixes[i + ibegin]]); }
  }
  comm.bcastNeighbor(Z, VColPrefixes.data(), sizeof(T), prime, sub);
}

template <class T>
inline void HSSLevel<T>::matVecHorizontalandDownwardPass(T* Y_out, const T* Z, T* Y, const CSR& rels, const HierarchicalComm& comm) const {
  long long ibegin = comm.oLocal(), nodes = comm.lenLocal();
  long long lenX = URowPrefixes[ibegin + nodes] - URowPrefixes[ibegin];
  std::fill(Y_out, &Y_out[lenX], T{});
  auto gemv_b1 = [](int m, int n, const T* a, const T* x, T* b) {
    hssHostGemv('N', m, n, a, x, T(1.), b);
  };
  auto gemv_b0 = [](int m, int n, const T* a, const T* x, T* b) {
    hssHostGemv('N', m, n, a, x, T{}, b);
  };

  for (long long i = 0ll; i < nodes; ++i) {
    long long M = URowPrefixes[i + ibegin + 1] - URowPrefixes[i + ibegin];
    long long K = UColPrefixes[i + ibegin + 1] - UColPrefixes[i + ibegin];
    if (0 < K) {
      T* y = &Y[UColPrefixes[i + ibegin]];
      for (long long ij = rels.RowIndex[i]; ij < rels.RowIndex[i + 1]; ++ij) {
        long long j = rels.ColIndex[ij];
        long long N = VColPrefixes[j + 1] - VColPrefixes[j];
        if (N > 0) gemv_b1(K, N, &C[CPrefixes[ij]], &Z[VColPrefixes[j]], y);
      }
      gemv_b0(M, K, &U[UPrefixes[i]], y, &Y_out[URowPrefixes[i + ibegin] - URowPrefixes[ibegin]]);
    }
  }
}

template <class T>
inline void HSSLevel<T>::matVecLeafHorizontalPass(T* X_io, const T* Z, T* Y,
                                        const CSR& rels, const HierarchicalComm& comm, MPI_Comm prime, MPI_Comm sub) const {
  long long ibegin = comm.oLocal(), nodes = comm.lenLocal();
  long long lenX = VRowPrefixes[ibegin + nodes] - VRowPrefixes[ibegin];

  std::vector<T> X(VRowPrefixes.back(), T{});
  std::copy(&X_io[0], &X_io[lenX], X.begin() + VRowPrefixes[ibegin]);
  comm.bcastNeighbor(X.data(), VRowPrefixes.data(), sizeof(T), prime, sub);
  std::fill(&X_io[0], &X_io[lenX], T{});
  auto gemv = [](int m, int n, const T* a, const T* x, T* b) {
    hssHostGemv('N', m, n, a, x, T(1.), b);
  };

  for (long long i = 0ll; i < nodes; ++i) {
    long long M = URowPrefixes[i + ibegin + 1] - URowPrefixes[i + ibegin];
    long long K = UColPrefixes[i + ibegin + 1] - UColPrefixes[i + ibegin];
    T* w = &X_io[URowPrefixes[i + ibegin] - URowPrefixes[ibegin]];

    if (0 < K) {
      T* y = &Y[UColPrefixes[i + ibegin]];
      for (long long ij = rels.RowIndex[i]; ij < rels.RowIndex[i + 1]; ++ij) {
        long long j = rels.ColIndex[ij];
        long long N = VColPrefixes[j + 1] - VColPrefixes[j];
        if (N > 0) gemv(K, N, &C[CPrefixes[ij]], &Z[VColPrefixes[j]], y);
      }
      gemv(M, K, &U[UPrefixes[i]], y, w);
    }

    long long N = VRowPrefixes[i + ibegin + 1] - VRowPrefixes[i + ibegin];
    gemv(M, N, &A[APrefixes[i]], &X[VRowPrefixes[i + ibegin]], w);
  }
}

template <class T>
inline HSS<T>::HSS(const std::vector<Cell>& cells, long long levels, MPI_Comm world, ncclComm_t ncclWorld) :
  levels(levels), mpi_communicators(2 * (levels + 1), MPI_COMM_NULL),
  nccl_communicators(2 * (levels + 1), nullptr), A(levels + 1) {
  int world_rank, world_size;
  MPI_Comm_rank(world, &world_rank);
  MPI_Comm_size(world, &world_size);

  std::vector<std::pair<long long, long long>> mapping(world_size, std::make_pair(0, 1));
  CSR Far(cells);
  for (long long i = 0ll; i <= levels; ++i) {
    comm.emplace_back(cells, mapping, Far.RowIndex.data(), Far.ColIndex.data(), world_rank);
    const int colorPrime = comm[i].colorPrime == -1 ? MPI_UNDEFINED : comm[i].colorPrime;
    const int colorSub = comm[i].colorSub == -1 ? MPI_UNDEFINED : comm[i].colorSub;
    MPI_Comm_split(world, colorPrime, world_rank, &mpi_communicators[2 * i]);
    MPI_Comm_split(world, colorSub, world_rank, &mpi_communicators[2 * i + 1]);
    const int ncclColorPrime = comm[i].colorPrime == -1 ? NCCL_SPLIT_NOCOLOR : comm[i].colorPrime;
    const int ncclColorSub = comm[i].colorSub == -1 ? NCCL_SPLIT_NOCOLOR : comm[i].colorSub;
    NCCL_CHECK(ncclCommSplit(ncclWorld, ncclColorPrime, world_rank, &nccl_communicators[2 * i], nullptr));
    NCCL_CHECK(ncclCommSplit(ncclWorld, ncclColorSub, world_rank, &nccl_communicators[2 * i + 1], nullptr));
    Far_local.emplace_back(Far, comm[i].oGlobal(), comm[i].oGlobal() + comm[i].lenLocal(), comm[i].Boxes);
  }

  long long llen = comm[levels].lenLocal(), gbegin = comm[levels].oGlobal();
  local_bodies = std::make_pair(cells[gbegin].N[0], cells[gbegin + llen - 1].N[1]);
}

template <class T>
inline void HSS<T>::evalMatVecA(DenseMatrix<T>& matA, const Cell cells[]) {
  long long llen = comm[levels].lenLocal(), gbegin = comm[levels].oGlobal();
  std::vector<long long> localM, localN(llen);
  std::transform(&cells[gbegin], &cells[gbegin + llen], localN.begin(), [](const Cell& c) { return c.N[1] - c.N[0]; });
  A[levels].eval(matA, &localN[0], &localN[0], Far_local[levels], comm[levels],
                 prime(levels), sub(levels), ncclPrime(levels), ncclSub(levels), cells[gbegin].N[0]);

  for (long long l = levels - 1ll; l >= 0ll; --l) {
    matA.redistribute(A[l + 1].UColPrefixes.back(), ncclPrime(l + 1), ncclPrime(l), ncclSub(l));
    llen = comm[l].lenLocal(); gbegin = comm[l].oGlobal(); localM.resize(llen); localN.resize(llen);
    for (long long i = 0ll; i < llen; ++i) {
      long long cbegin = cells[gbegin + i].Child[0] - cells[gbegin].Child[0], cend = cells[gbegin + i].Child[1] - cells[gbegin].Child[0];
      localM[i] = A[l + 1].UColPrefixes[cend] - A[l + 1].UColPrefixes[cbegin];
      localN[i] = A[l + 1].VColPrefixes[cend] - A[l + 1].VColPrefixes[cbegin];
    }
    matA.iterateTolerance();
    A[l].eval(matA, &localM[0], &localN[0], Far_local[l], comm[l], prime(l), sub(l), ncclPrime(l), ncclSub(l));
  }
}

template <class T>
inline void HSS<T>::allocRHSwork(std::vector<long long>& iwork, std::vector<T>& cwork) const {
  iwork.resize((levels * 4ll) + 4ll);
  long long* lenX = &iwork[0], *lenY = &iwork[levels + 2ll], *locV = &iwork[(levels * 2ll) + 4ll], *locU = &iwork[(levels * 3ll) + 4ll];
  lenX[0] = lenY[0] = 0;
  for (long long l = 0; l < levels; ++l) {
    lenX[l + 1] = 0ll < l ? A[l].VRowPrefixes.back() : 0ll;
    lenY[l + 1] = 0ll < l ? A[l].URowPrefixes.back() : 0ll;
    locV[l] = A[l + 1].VColPrefixes[comm[l].oLocalChild(comm[l + 1])];
    locU[l] = A[l + 1].UColPrefixes[comm[l].oLocalChild(comm[l + 1])];
  }

  lenX[levels + 1] = A[levels].VRowPrefixes.back();
  lenY[levels + 1] = A[levels].URowPrefixes.back();
  std::inclusive_scan(&lenX[1], locV, &lenX[1]);
  cwork.resize(lenY[levels + 1]);
}

template <class T>
inline void HSS<T>::matVecMul(T X[]) const {
  std::vector<long long> iworkA;
  std::vector<T> cworkA;
  allocRHSwork(iworkA, cworkA);
  matVecMul(X, iworkA.data(), cworkA.data());
}

template <class T>
inline void HSS<T>::matVecMul(T X[], const long long iwork[], T cwork[]) const {
  const long long* lenX = iwork, *lenY = &iwork[levels + 2ll], *locV = &iwork[(levels * 2ll) + 4ll], *locU = &iwork[(levels * 3ll) + 4ll];
  if (0 < levels)
    A[levels].matVecUpwardPass(X, &cwork[lenX[levels]], comm[levels], prime(levels), sub(levels));
  for (long long l = levels - 1; l >= 0; --l)
    A[l].matVecUpwardPass(&cwork[lenX[l + 1] + locV[l]], &cwork[lenX[l]], comm[l], prime(l), sub(l));

  for (long long l = 0; l < levels; ++l)
    A[l].matVecHorizontalandDownwardPass(&cwork[lenY[l + 1] + locU[l]], &cwork[lenX[l]], &cwork[lenY[l]], Far_local[l], comm[l]);
  A[levels].matVecLeafHorizontalPass(X, &cwork[lenX[levels]], &cwork[lenY[levels]], Far_local[levels],
                                     comm[levels], prime(levels), sub(levels));
}

template <class T>
inline double HSS<T>::solveRelErr(const T X[], const T ref[]) {
  double err[2] = { 0., 0. };
  const MPI_Comm leafPrime = prime(levels), leafSub = sub(levels);
  if (leafPrime != MPI_COMM_NULL) {
    const long long lenX = local_bodies.second - local_bodies.first;
    for (long long i = 0; i < lenX; ++i) {
      err[0] += std::norm(X[i] - ref[i]);
      err[1] += std::norm(ref[i]);
    }
    MPI_Allreduce(MPI_IN_PLACE, err, 2, MPI_DOUBLE, MPI_SUM, leafPrime);
  }
  if (leafSub != MPI_COMM_NULL)
    MPI_Bcast(err, 2, MPI_DOUBLE, 0, leafSub);
  return std::sqrt(err[0] / err[1]);
}

template <class T>
inline void HSS<T>::free_all_comms() {
  for (ncclComm_t handle : nccl_communicators)
    if (handle != nullptr)
      NCCL_CHECK(ncclCommDestroy(handle));
  nccl_communicators.clear();
  for (MPI_Comm& handle : mpi_communicators)
    if (handle != MPI_COMM_NULL)
      MPI_Comm_free(&handle);
  mpi_communicators.clear();
}

class Helmholtz3D {
public:
  double k;
  double singularity;
  Helmholtz3D(double wave_number, double s) : k(wave_number), singularity(1. / s) {}
  std::complex<double> operator()(double d) const {
    if (d == 0.)
      return std::complex<double>(singularity, 0.);
    else
      return std::complex<double>(-std::cos(k * d), std::sin(k * d)) / d;
  }
};

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static inline void mesh_sphere(double* bodies, long long nbodies) {
  const double phi = 2.39996322972865332223; // golden angle in radians
  const double rx = 1., ry = 1., rz = 1., dn = 2. / double((nbodies - int64_t(1)) ?: int64_t(1));
  for (int64_t i = 0; i < nbodies; ++i) {
    double di = double(i);
    double x = 1. - (di * dn);  // x goes from -r to r.
    double radius = std::sqrt(1. - std::min(1., x * x)); // radius at x
    bodies[i * 3] = x * rx;
    bodies[i * 3 + 1] = radius * std::cos(di * phi) * ry;
    bodies[i * 3 + 2] = radius * std::sin(di * phi) * rz;
  }
}

template <class T, class Evaluate>
static inline void fillHelmholtzBlock(long long m, long long n, const double* bi, const double* bj,
                                     T* Aij, Evaluate evaluate) {
  for (long long j = 0; j < n; ++j)
    for (long long i = 0; i < m; ++i) {
      const double distance = std::hypot(bi[3*i] - bj[3*j], bi[3*i+1] - bj[3*j+1], bi[3*i+2] - bj[3*j+2]);
      Aij[i + j*m] = evaluate(distance);
    }
}

static inline void gen_matrix(const Helmholtz3D& eval, long long m, long long n,
                              const double* bi, const double* bj, std::complex<double>* Aij) {
  fillHelmholtzBlock(m, n, bi, bj, Aij, eval);
}

static inline void gen_matrix(const Helmholtz3D& eval, long long m, long long n,
                              const double* bi, const double* bj, double* Aij) {
  fillHelmholtzBlock(m, n, bi, bj, Aij, [&](double distance) {
    return distance == 0. ? eval.singularity : -std::cos(eval.k * distance) / distance;
  });
}

static inline void buildBinaryTree(Cell* cells, double* bodies, long long nbodies, long long levels) {
  cells[0].N[0] = 0;
  cells[0].N[1] = nbodies;
  auto get_bounds = [=](double* bodies, long long nbodies, double R[], double C[]) {
    const std::array<double, 3>* bodies3 = (const std::array<double, 3>*)(&bodies[0]);
    const std::array<double, 3>* bodies3_end = (const std::array<double, 3>*)(&bodies[nbodies * 3]);
    double Xmin[3], Xmax[3];
    for (int i = 0; i < 3; i++) {
      auto minmax = std::minmax_element(bodies3, bodies3_end,
        [=](const std::array<double, 3>& x, const std::array<double, 3>& y) { return x[i] < y[i]; });
      Xmin[i] = (*minmax.first)[i]; Xmax[i] = (*minmax.second)[i];
    }
    std::transform(Xmin, &Xmin[3], Xmax, C, [](double min, double max) { return (min + max) * 0.5; });
    std::transform(Xmin, &Xmin[3], Xmax, R, [](double min, double max) { return (min == max && min == 0.) ? 0. : ((max - min) * 0.5 + 1.e-8); });
  };
  get_bounds(bodies, nbodies, cells[0].R.data(), cells[0].C.data());

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
    ci.Child[0] = len;
    ci.Child[1] = len + 2;

    long long loc = i_begin + (i_end - i_begin) / 2;
    c0.N[0] = i_begin;
    c0.N[1] = loc;
    c1.N[0] = loc;
    c1.N[1] = i_end;

    get_bounds(&bodies[i_begin * 3], loc - i_begin, c0.R.data(), c0.C.data());
    get_bounds(&bodies[loc * 3], i_end - loc, c1.R.data(), c1.C.data());
  }
}

static inline void selectCudaDevice(MPI_Comm world) {
  MPI_Comm local;
  MPI_Comm_split_type(world, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local);
  int rank;
  MPI_Comm_rank(local, &rank);
  MPI_Comm_free(&local);

  int deviceCount;
  const auto countStatus = cudaGetDeviceCount(&deviceCount);
  if (countStatus != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(countStatus));
  if (deviceCount == 0)
    throw std::runtime_error("No visible CUDA devices");
  const auto selectStatus = cudaSetDevice(rank % deviceCount);
  if (selectStatus != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(selectStatus));
}

static inline ncclComm_t bootstrapNccl(MPI_Comm world) {
  int rank, size;
  MPI_Comm_rank(world, &rank);
  MPI_Comm_size(world, &size);
  ncclUniqueId id;
  if (rank == 0)
    NCCL_CHECK(ncclGetUniqueId(&id));
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, world);

  ncclComm_t comm;
  NCCL_CHECK(ncclCommInitRank(&comm, size, id, rank));
  return comm;
}


template <class T>
static void parseHssValue(std::string_view text, T& value) {
  const std::string input(text);
  size_t consumed = 0;
  if constexpr (std::is_same_v<T, double>) {
    value = std::stod(input, &consumed);
  } else {
    const long long parsed = std::stoll(input, &consumed);
    if (parsed < std::numeric_limits<T>::min() || parsed > std::numeric_limits<T>::max())
      throw std::out_of_range("Integer argument out of range: " + input);
    value = static_cast<T>(parsed);
  }
  if (consumed != input.size()) throw std::invalid_argument("Invalid numeric argument: " + input);
}

template <class... T>
static char parseHssArguments(int argc, char* argv[], std::pair<std::string_view, T&>... options) {
  char precision = 'Z';
  size_t positional = 0;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg(argv[i]);
    const auto equals = arg.find('=');
    const auto key = arg.substr(0, equals);
    if (key == "data" && equals != std::string_view::npos) {
      const auto value = arg.substr(equals + 1);
      if (value != "D" && value != "Z") throw std::invalid_argument("Supported precisions: data=D or data=Z");
      precision = value.front();
      continue;
    }
    size_t index = 0;
    auto parse = [&](auto option) {
      const size_t current = index++;
      if (equals == std::string_view::npos ? current != positional : key != option.first) return false;
      parseHssValue(equals == std::string_view::npos ? arg : arg.substr(equals + 1), option.second);
      return true;
    };
    if (!(parse(options) || ...)) throw std::invalid_argument("Unknown HSS argument: " + std::string(arg));
    if (equals == std::string_view::npos) ++positional;
  }
  return precision;
}

template <class T>
static auto hssOption(std::string_view name, T& value) {
  return std::pair<std::string_view, T&>(name, value);
}

template <class Matrix, class Configure, class Report>
static void runHssBenchmark(long long Nbody, long long leaf_size, Configure configure, Report report) {
  using T = typename Matrix::Scalar;
  if (Nbody <= 0 || leaf_size <= 0)
    throw std::invalid_argument("Require positive matrix and leaf sizes");

  leaf_size = Nbody < leaf_size ? Nbody : leaf_size;
  long long levels = (long long)std::log2((double)Nbody / leaf_size);
  long long Nleaf = (long long)1 << levels;
  long long ncells = Nleaf + Nleaf - 1;

  int mpi_rank = 0;
  MPI_Comm world = MPI_COMM_NULL;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_split(MPI_COMM_WORLD, mpi_rank < Nleaf ? 1 : MPI_UNDEFINED, mpi_rank, &world);

  if (world == MPI_COMM_NULL) return;
  selectCudaDevice(world);
  const ncclComm_t ncclWorld = bootstrapNccl(world);

  cudaStream_t streamHandle;
  const auto streamStatus = cudaStreamCreateWithFlags(&streamHandle, cudaStreamNonBlocking);
  if (streamStatus != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(streamStatus));
  const std::unique_ptr<std::remove_pointer_t<cudaStream_t>, decltype(&cudaStreamDestroy)>
      stream(streamHandle, cudaStreamDestroy);

  cublasHandle_t blasHandle;
  const auto blasStatus = cublasCreate(&blasHandle);
  if (blasStatus != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error(cublasGetStatusString(blasStatus));
  const std::unique_ptr<std::remove_pointer_t<cublasHandle_t>, decltype(&cublasDestroy)>
      blas(blasHandle, cublasDestroy);
  const auto blasStreamStatus = cublasSetStream(blas.get(), stream.get());
  if (blasStreamStatus != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error(cublasGetStatusString(blasStreamStatus));

  cusolverDnHandle_t solverHandle;
  if (cusolverDnCreate(&solverHandle) != CUSOLVER_STATUS_SUCCESS)
    throw std::runtime_error("Could not create the cuSOLVER handle");
  const std::unique_ptr<std::remove_pointer_t<cusolverDnHandle_t>, decltype(&cusolverDnDestroy)>
      solver(solverHandle, cusolverDnDestroy);
  if (cusolverDnSetStream(solver.get(), stream.get()) != CUSOLVER_STATUS_SUCCESS)
    throw std::runtime_error("Could not set the cuSOLVER stream");

  std::vector<double> body(Nbody * 3);
  std::vector<T> Xbody(Nbody);
  std::vector<Cell> cell(ncells);

  mesh_sphere(&body[0], Nbody);
  buildBinaryTree(&cell[0], &body[0], Nbody, levels);

  std::mt19937_64 gen;
  std::uniform_real_distribution<double> uniform_dist(0., 1.);
  std::generate(Xbody.begin(), Xbody.end(), [&]() {
    if constexpr (hssComplex<T>) return T(uniform_dist(gen), uniform_dist(gen));
    else return uniform_dist(gen);
  });

  /*cell.erase(cell.begin() + 1, cell.begin() + Nleaf - 1);
  cell[0].Child[0] = 1; cell[0].Child[1] = Nleaf + 1;
  ncells = Nleaf + 1;
  levels = 1;*/

  HSS<T> matA(cell, levels, world, ncclWorld);
  const ncclComm_t leafNcclPrime = matA.ncclPrime(levels), leafNcclSub = matA.ncclSub(levels);
  long long lenX = matA.local_bodies.second - matA.local_bodies.first;
  std::vector<T> X1(lenX, T{});
  std::vector<T> X2(lenX, T{});

  Helmholtz3D eval(4., 1.E0);
  Matrix denseA(lenX, Nbody, leafNcclPrime, leafNcclSub, blas.get(), solver.get(), stream.get());
  for (long long i = matA.local_bodies.first; i < matA.local_bodies.second; i += 512ll)
    for (long long j = 0; j < Nbody; j += 512ll) {
      long long rows = std::min(matA.local_bodies.second - i, 512ll);
      long long cols = std::min(Nbody - j, 512ll);
      std::vector<T> A(rows * cols);
      gen_matrix(eval, rows, cols, &body[3ll * i], &body[3 * j], &A[0]);
      denseA.AijSetBlock(rows, cols, i, j, &A[0], rows);
  }

  DenseMatrix<T> denseX(lenX, 1, leafNcclPrime, leafNcclSub, blas.get(), solver.get(), stream.get());
  denseX.AijSetBlock(lenX, 1, matA.local_bodies.first, 0, &Xbody[matA.local_bodies.first], Nbody);
  double refmatvec_time = MPI_Wtime();
  denseX.matVecA(denseA, &X2[0], leafNcclPrime, leafNcclSub);
  refmatvec_time = MPI_Wtime() - refmatvec_time;

  MPI_Barrier(world);
  double h2_construct_time = MPI_Wtime();
  configure(denseA);
  matA.evalMatVecA(denseA, cell.data());

  MPI_Barrier(world);
  h2_construct_time = MPI_Wtime() - h2_construct_time;

  std::copy(&Xbody[matA.local_bodies.first], &Xbody[matA.local_bodies.second], &X1[0]);
  MPI_Barrier(world);
  double matvec_time = MPI_Wtime();
  matA.matVecMul(&X1[0]);

  MPI_Barrier(world);
  matvec_time = MPI_Wtime() - matvec_time;
  double cerr = matA.solveRelErr(&X1[0], &X2[0]);

  unsigned long long localScalars = 0, storedScalars = 0;
  for (const auto& level : matA.A)
    localScalars += level.U.size() + level.V.size() + level.A.size() + level.C.size();
  MPI_Reduce(&localScalars, &storedScalars, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, world);

  if (mpi_rank == 0) {
    int mpi_size = 1;
    MPI_Comm_size(world, &mpi_size);
    const auto denseScalars = static_cast<unsigned long long>(Nbody) * Nbody;
    std::ostringstream storagePercent;
    storagePercent << std::fixed << std::setprecision(2) << 100. * storedScalars / denseScalars;
    std::cout << "Active MPI processes: " << mpi_size << std::endl;
    std::cout << "Matrix Order: " << Nbody << " Leaf-size: " << leaf_size
              << " Data: " << (hssComplex<T> ? 'Z' : 'D') << std::endl;
    report(denseA);
    std::cout << "HSS scalars stored: " << storedScalars << " / "
              << denseScalars << " (" << storagePercent.str() << "%)" << std::endl;
    std::cout << "Construct Err: " << cerr << std::endl;
    std::cout << " HSS  Construct Time: " << h2_construct_time << std::endl;
    std::cout << " HSS  Matvec Time: " << matvec_time << std::endl;
    std::cout << "Dense Matvec Time: " << refmatvec_time << std::endl;
  }

  matA.free_all_comms();
  NCCL_CHECK(ncclCommDestroy(ncclWorld));
  MPI_Comm_free(&world);
}
