#pragma once

// Infrastructure shared by the Hyacin benchmarks and vendor comparisons.
// HSS intentionally remains independent of these benchmark headers.
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <nccl.h>
#include <mpi.h>
#include <unistd.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <functional>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

template <class T> constexpr bool benchmark_is_complex = false;
template <class R> constexpr bool benchmark_is_complex<std::complex<R>> = true;
template <> constexpr bool benchmark_is_complex<cuDoubleComplex> = true;
template <> constexpr bool benchmark_is_complex<cuComplex> = true;

// Minimal complex construction needed by the shared matrix generator.
// Hyacin extends this for its additional storage types in hyacin_common.hpp.
template <class T> inline T benchmark_from_complex(std::complex<double> value) {
  if constexpr (std::is_same_v<T, cuDoubleComplex>) return make_cuDoubleComplex(value.real(), value.imag());
  else if constexpr (std::is_same_v<T, cuComplex>) return make_cuComplex(float(value.real()), float(value.imag()));
  else return T(value);
}

// Report the failed expression and its source location immediately.
inline void library_check(cudaError_t status, const char* call, const char* file, int line) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(file) + ":" + std::to_string(line) + ": " + call + ": " + cudaGetErrorString(status));
}
inline void library_check(cublasStatus_t status, const char* call, const char* file, int line) {
  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error(std::string(file) + ":" + std::to_string(line) + ": " + call + ": cuBLAS status " + std::to_string(int(status)));
}
inline void library_check(cusolverStatus_t status, const char* call, const char* file, int line) {
  if (status != CUSOLVER_STATUS_SUCCESS)
    throw std::runtime_error(std::string(file) + ":" + std::to_string(line) + ": " + call + ": cuSOLVER status " + std::to_string(int(status)));
}
inline void library_check(ncclResult_t status, const char* call, const char* file, int line) {
  if (status != ncclSuccess)
    throw std::runtime_error(std::string(file) + ":" + std::to_string(line) + ": " + call + ": " + ncclGetErrorString(status));
}
#define LIBRARY_CHECK(call) library_check((call), #call, __FILE__, __LINE__)

inline void check_solver_info(int* device_info, cudaStream_t stream, const char* routine) {
  int info = 0;
  LIBRARY_CHECK(cudaMemcpyAsync(&info, device_info, sizeof(info), cudaMemcpyDeviceToHost, stream));
  LIBRARY_CHECK(cudaStreamSynchronize(stream));
  if (info != 0)
    throw std::runtime_error(std::string(routine) + " returned info=" + std::to_string(info));
}

using blas_int = int; // LP64 for blas
const int32_t kernel_runs = 3;
static_assert(kernel_runs > 0, "kernel_runs must be positive");
inline double kernel_time = 0., rep_time = 0., comm_time = 0.;

// Leading-order real-operation equivalents, independent of storage precision.
template <class T> constexpr double benchmark_flop_factor =
  benchmark_is_complex<T> ? 4. : 1.;

// Full Householder GEQP3 without forming Q, plus the interpolative solve
// R11 * X = R12: R11 is rank-by-rank, R12 is rank-by-(N-rank).
inline double interpolative_flops(double m, double n, double rank, double scalar_factor) {
  const double k = std::min(m, n);
  const double qr = 2. * m * n * k - (2. / 3.) * k * k * k;
  const double trsm = rank * rank * (n - rank);
  return scalar_factor * (qr + trsm);
}

// Leading-order randomized SVD estimate for jobu='N', jobvt='S'.
// l is the sketch width (target rank + oversampling); p is the power count.
// Includes range finding/projection and reduced factorization work, without
// the final multiplication that would form the left singular vectors.
inline double randomized_svd_flops(double m, double n, double l, double p, double scalar_factor) {
  return scalar_factor * (4. * (p + 1.) * m * n * l + l * l * (4. * m + 2. * n));
}

template <class T> struct matrix_generator {
  int64_t gM, gN;
  std::vector<double> bodies;
  const double w;
  matrix_generator(double w, int64_t M, int64_t N) : gM(M), gN(N), bodies(int64_t(3) * (M + N)), w(w) {
    int64_t nbodies = M + N;
    const double phi = 2.39996322972865332223; // golden angle in radians
    const double rx = 100., ry = 10., rz = 10., dn = 2. / double((nbodies - int64_t(1)) ?: int64_t(1));
    for (int64_t i = 0; i < nbodies; ++i) {
      double di = double(i);
      double x = 1. - (di * dn);  // x goes from -r to r.
      double radius = std::sqrt(1. - std::min(1., x * x)); // radius at x
      bodies[i * 3] = x * rx;
      bodies[i * 3 + 1] = radius * std::cos(di * phi) * ry;
      bodies[i * 3 + 2] = radius * std::sin(di * phi) * rz;
    }

    auto get_bounds = [](const double* bodies, int64_t nbodies, double R[], double C[]) {
      const std::array<double, 3>* b3 = reinterpret_cast<const std::array<double, 3>*>(&bodies[0]);
      const std::array<double, 3>* b3_end = reinterpret_cast<const std::array<double, 3>*>(&bodies[nbodies * 3]);
      double Xmin[3], Xmax[3];
      for (int i = 0; i < 3; ++i) {
        auto minmax = std::minmax_element(b3, b3_end, [=](const std::array<double, 3>& x, const std::array<double, 3>& y) { return x[i] < y[i]; });
        Xmin[i] = (*minmax.first)[i]; Xmax[i] = (*minmax.second)[i];
      }
      std::transform(Xmin, &Xmin[3], Xmax, C, [](double min, double max) { return (min + max) * 0.5; });
      std::transform(Xmin, &Xmin[3], Xmax, R, [](double min, double max) { return (min == max && min == 0.) ? 0. : ((max - min) * 0.5 + 1.e-8); });
    };

    struct Cell { std::array<int64_t, 2> Body; std::array<double, 3> R; std::array<double, 3> C; };
    auto nextPowerOf2 = [](uint64_t x) { if (x <= 1llu) { return 1llu; } --x; x |= x >> 1; x |= x >> 2; x |= x >> 4; x |= x >> 8; x |= x >> 16; x |= x >> 32; return x + 1llu; };
    int64_t nleaf = nextPowerOf2(uint64_t(nbodies / 512ll)); std::vector<Cell> cells(nleaf + nleaf - 1);
    cells[0].Body[0] = 0; cells[0].Body[1] = nbodies;
    get_bounds(&bodies[0], nbodies, cells[0].R.data(), cells[0].C.data());

    for (int64_t i = 0; i < nleaf - 1; ++i) {
      Cell& ci = cells[i];
      int64_t sdim = std::distance(ci.R.begin(), std::max_element(ci.R.begin(), ci.R.end()));
      int64_t i_begin = ci.Body[0];
      int64_t i_end = ci.Body[1];

      std::array<double, 3>* bodies3 = reinterpret_cast<std::array<double, 3>*>(&bodies[i_begin * 3]);
      std::array<double, 3>* bodies3_end = reinterpret_cast<std::array<double, 3>*>(&bodies[i_end * 3]);
      std::sort(bodies3, bodies3_end,
        [=](std::array<double, 3>& i, std::array<double, 3>& j) { return i[sdim] < j[sdim]; });

      int64_t len = (i << 1) + 1;
      Cell& c0 = cells[len];
      Cell& c1 = cells[len + 1];
      int64_t loc = i_begin + (i_end - i_begin) / 2;
      c0.Body[0] = i_begin;
      c0.Body[1] = loc;
      c1.Body[0] = loc;
      c1.Body[1] = i_end;

      get_bounds(&bodies[i_begin * 3], loc - i_begin, c0.R.data(), c0.C.data());
      get_bounds(&bodies[loc * 3], i_end - loc, c1.R.data(), c1.C.data());
    }
  }

  inline double eval_real(double d) { return d == 0. ? w : std::cos(w * d) / d; }
  inline std::complex<double> eval_complex(double d) { return std::complex<double>(eval_real(d), d == 0. ? 0. : std::sin(-w * d) / d); }
  //inline double eval_real(double d) { return d == 0. ? w : std::exp((-d * d) / w); }
  //inline std::complex<double> eval_complex(double d) { return std::complex<double>(eval_real(d), 0.); }

  void generate_block(int32_t mb, int32_t nb, T* A, int32_t lda, int32_t grid_row = 0, int32_t grid_col = 0, int32_t tile_m = 1, int32_t tile_n = 1) {
    int64_t row_offset = int64_t(grid_row) * int64_t(mb), col_offset = int64_t(grid_col) * int64_t(nb);
    for (int64_t iA = row_offset, y = 0; iA < gM; iA = row_offset + int64_t(tile_m) * (y += int64_t(mb))) {
      int64_t rows = std::min(gM - iA, int64_t(mb));
      for (int64_t jA = col_offset, x = 0; jA < gN; jA = col_offset + int64_t(tile_n) * (x += int64_t(nb))) {
        int64_t cols = std::min(gN - jA, int64_t(nb));
        for (int64_t j = 0; j < cols; ++j) {
          int64_t j_loc = int64_t(3) * (j + jA + gM);
          double pt_j[3]{ bodies[j_loc], bodies[j_loc + int64_t(1)], bodies[j_loc + int64_t(2)]};
          for (int64_t i = 0; i < rows; ++i) {
            int64_t i_loc = int64_t(3) * (i + iA);
            double diff_x = bodies[i_loc] - pt_j[0];
            double diff_y = bodies[i_loc + int64_t(1)] - pt_j[1];
            double diff_z = bodies[i_loc + int64_t(2)] - pt_j[2];
            double d = std::sqrt(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z);
            int64_t k = (i + y) + (j + x) * int64_t(lda);
            if constexpr(benchmark_is_complex<T>)
            { A[k] = benchmark_from_complex<T>(eval_complex(d)); } else { A[k] = T(eval_real(d)); }
          }
        }
      }
    }
  }
};

extern "C" void dgemm_(const char*, const char*, const blas_int*, const blas_int*, const blas_int*, const double*, const double*, const blas_int*, const double*, const blas_int*, const double*, double*, const blas_int*);
extern "C" void zgemm_(const char*, const char*, const blas_int*, const blas_int*, const blas_int*, const void*, const void*, const blas_int*, const void*, const blas_int*, const void*, void*, const blas_int*);
inline void ncgemm(blas_int M, blas_int N, blas_int K, const double* A, blas_int lda, const double* B, blas_int ldb, double* C, blas_int ldc)
{ char transa = 'N', transb = 'T'; double one = 1., minus_one = -1.; dgemm_(&transa, &transb, &M, &N, &K, &minus_one, A, &lda, B, &ldb, &one, C, &ldc); }
inline void ncgemm(blas_int M, blas_int N, blas_int K, const std::complex<double>* A, blas_int lda, const std::complex<double>* B, blas_int ldb, std::complex<double>* C, blas_int ldc)
{ char transa = 'N', transb = 'C'; std::complex<double> one(1., 0.), minus_one(-1., 0.); zgemm_(&transa, &transb, &M, &N, &K, &minus_one, A, &lda, B, &ldb, &one, C, &ldc); }

// NCCL 2.31+ allows multiple communicator ranks to use the same GPU.
inline void enable_nccl_multi_rank_gpu() {
  if (setenv("NCCL_MULTI_RANK_GPU_ENABLE", "1", 1) != 0)
    throw std::runtime_error("Could not enable NCCL multi-rank GPU support");
}

inline void bootstrap_posix_fork(int32_t& world_rank, int32_t& local_rank, int32_t& world_size, ncclUniqueId& id, int32_t requested_ranks = 1) {
  const char* devices = std::getenv("CUDA_VISIBLE_DEVICES");
  if (devices == nullptr) { throw std::runtime_error("CUDA_VISIBLE_DEVICES is unset"); }
  if (devices[0] == '\0') { throw std::runtime_error("No CUDA device visible"); }

  if (requested_ranks < 0) throw std::invalid_argument("ranks must be nonnegative (zero selects one rank per visible GPU)");
  std::string dev_str(devices);
  world_size = requested_ranks > 0 ? requested_ranks : 1 + std::count(dev_str.begin(), dev_str.end(), ',');
  int32_t rank = 0;
  LIBRARY_CHECK(ncclGetUniqueId(&id));
  for (int32_t i = 1; i < world_size; ++i) if (rank == 0) {
    int32_t pid = ::fork();
    if (pid == -1) { throw std::runtime_error("POSIX Fork Failure"); }
    rank = pid ? 0 : i;
  }
  world_rank = local_rank = rank;
}

inline void bootstrap_mpi(int32_t& world_rank, int32_t& local_rank, int32_t& world_size, ncclUniqueId& id) {
  MPI_Init(nullptr, nullptr);
  MPI_Comm shmcomm; MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shmcomm);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_rank(shmcomm, &local_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_free(&shmcomm);

  if (world_rank == 0) LIBRARY_CHECK(ncclGetUniqueId(&id));
  MPI_Bcast(&id, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);
  MPI_Finalize();
}
