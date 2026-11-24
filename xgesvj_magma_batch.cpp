#include <vendor_common.hpp>
#include <magma_v2.h>
#include <iostream>

namespace {

void check_magma(magma_int_t status, const char* routine) {
  if (status != MAGMA_SUCCESS)
    throw std::runtime_error(std::string(routine) + " returned status=" + std::to_string(status));
}

template <class T>
void check_block_resources(magma_int_t m, magma_int_t n) {
  bool fused;
  magma_int_t nb;
  if constexpr (std::is_same_v<T, double>) {
    fused = magma_dgesvj_batched_use_fused(MagmaVec, MagmaVec, m, n);
    nb = magma_get_dgesvj_batched_nb(std::max(m, n), std::min(m, n));
  } else {
    fused = magma_zgesvj_batched_use_fused(MagmaVec, MagmaVec, m, n);
    nb = magma_get_zgesvj_batched_nb(std::max(m, n), std::min(m, n));
  }
  if (fused) return;
  int sharedBytes = 0;
  LIBRARY_CHECK(cudaDeviceGetAttribute(&sharedBytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
  // MAGMA's blocked driver diagonalizes a 2*nb Gram block in shared memory.
  // Matrix and eigenvectors alone need this much, before padding/workspace.
  // MAGMA 2.10 can otherwise print a resource error yet return success/zero info.
  const size_t order = 2 * static_cast<size_t>(nb);
  const size_t minimumBytes = 2 * order * order * sizeof(T);
  if (minimumBytes > static_cast<size_t>(sharedBytes))
    throw std::runtime_error("MAGMA's default GESVJ block size nb=" + std::to_string(nb) +
      " requires at least " + std::to_string(minimumBytes) + " shared-memory bytes; GPU supports " +
      std::to_string(sharedBytes) + ". Adjust MAGMA's block-size tuning or use a supported matrix shape.");
}

template <class T>
magma_int_t gesvj(magma_queue_t queue, magma_int_t m, magma_int_t n, T* A,
    double* S, T* U, T* V, magma_int_t* info, magma_int_t batchN) {
  const magma_int_t k = std::min(m, n);
  if constexpr (std::is_same_v<T, double>)
    return magma_dgesvj_batched_strided(MagmaVec, MagmaVec,
      m, n, A, m, m*n, S, k, U, m, m*k, V, n, n*k, info, batchN, queue);
  else
    return magma_zgesvj_batched_strided(MagmaVec, MagmaVec,
      m, n, reinterpret_cast<magmaDoubleComplex*>(A), m, m*n, S, k,
      reinterpret_cast<magmaDoubleComplex*>(U), m, m*k,
      reinterpret_cast<magmaDoubleComplex*>(V), n, n*k, info, batchN, queue);
}

template <class T>
cublasStatus_t scale_columns(cublasHandle_t blas, int m, int rank, const T* U,
    const T* S, T* US) {
  if constexpr (std::is_same_v<T, double>)
    return cublasDdgmm(blas, CUBLAS_SIDE_RIGHT, m, rank, U, m, S, 1, US, m);
  else
    return cublasZdgmm(blas, CUBLAS_SIDE_RIGHT, m, rank,
      reinterpret_cast<const cuDoubleComplex*>(U), m,
      reinterpret_cast<const cuDoubleComplex*>(S), 1, reinterpret_cast<cuDoubleComplex*>(US), m);
}

// Match GESVD's reverse cumulative squared singular-value truncation, including
// its strict comparison and zero-norm handling. This is only used for validation.
int validation_rank(const double* S, int count, double eps) {
  std::vector<double> tail(S, S + count);
  std::transform_inclusive_scan(tail.rbegin(), tail.rend(), tail.rbegin(),
    std::plus<double>(), [](double s) { return s * s; });
  if (tail[0] < std::numeric_limits<double>::min()) return 0;
  const double threshold = tail[0] * std::pow(eps, 2);
  return std::distance(tail.begin(), std::find_if(tail.begin(), tail.end(),
    [threshold](double s) { return s < threshold; }));
}

template <class T>
int run(int argc, char* argv[]) {
  int64_t M = 32, N = 32, batchN = 1024;
  double eps = 1.e-12;
  for (int i = 1; i < argc; ++i) {
    if (std::strncmp(argv[i], "M=", 2) == 0) std::sscanf(argv[i], "M=%ld", &M);
    else if (std::strncmp(argv[i], "N=", 2) == 0) std::sscanf(argv[i], "N=%ld", &N);
    else if (std::strncmp(argv[i], "batchN=", 7) == 0) std::sscanf(argv[i], "batchN=%ld", &batchN);
    else if (std::strncmp(argv[i], "eps=", 4) == 0) std::sscanf(argv[i], "eps=%le", &eps);
    else if (std::strncmp(argv[i], "data=", 5) == 0) {}
    else std::cerr << "Ignored parameter: " << argv[i] << '\n';
  }
  const int64_t magmaLimit = std::numeric_limits<magma_int_t>::max();
  if (M <= 0 || M > magmaLimit || N <= 0 || N > magmaLimit || batchN <= 0 ||
      batchN > magmaLimit || eps < 0. || !std::isfinite(eps))
    throw std::invalid_argument("Require positive M,N,batchN within MAGMA integer range and finite eps >= 0");
  if (M > magmaLimit / N)
    throw std::invalid_argument("M*N must fit MAGMA's integer stride type");
  if (M == 1 && N == 1)
    throw std::invalid_argument("MAGMA 2.10's 1x1 GESVJ path fails its internal thread-count assertion");
  if (batchN > std::numeric_limits<int64_t>::max() / (M*N) ||
      static_cast<uint64_t>(M*N) * batchN > std::numeric_limits<size_t>::max() / sizeof(T))
    throw std::overflow_error("Batched matrix allocation size overflows");
  const magma_int_t k = std::min(M, N);
  const int64_t elements = M * N * batchN;
  const size_t bytes = static_cast<size_t>(elements) * sizeof(T);

  LIBRARY_CHECK(cudaSetDevice(0));
  check_magma(magma_init(), "magma_init");
  magma_queue_t queue = nullptr;
  magma_queue_create(0, &queue);
  if (queue == nullptr) throw std::runtime_error("Could not create MAGMA queue");
  const cudaStream_t stream = magma_queue_get_cuda_stream(queue);
  const cublasHandle_t blas = magma_queue_get_cublas_handle(queue);
  check_block_resources<T>(M, N);
  cudaEvent_t start, stop;
  LIBRARY_CHECK(cudaEventCreate(&start));
  LIBRARY_CHECK(cudaEventCreate(&stop));

  // Generate one global matrix; consecutive groups of N columns form the batch.
  std::vector<T> matA(elements);
  matrix_generator<T>(1., M, N * batchN).generate_block(512, 512, matA.data(), M);
  T *dA = nullptr, *dU = nullptr, *dV = nullptr;
  double* dS = nullptr;
  magma_int_t* info = nullptr;
  LIBRARY_CHECK(cudaMalloc((void**)(&dA), bytes));
  LIBRARY_CHECK(cudaMalloc((void**)(&dU), M * k * batchN * sizeof(T)));
  LIBRARY_CHECK(cudaMalloc((void**)(&dV), N * k * batchN * sizeof(T)));
  LIBRARY_CHECK(cudaMalloc((void**)(&dS), k * batchN * sizeof(double)));
  LIBRARY_CHECK(cudaMalloc((void**)(&info), batchN * sizeof(magma_int_t)));

  std::vector<magma_int_t> hostInfo(batchN);
  auto check_info = [&] {
    LIBRARY_CHECK(cudaMemcpyAsync(hostInfo.data(), info, batchN * sizeof(magma_int_t), cudaMemcpyDeviceToHost, stream));
    LIBRARY_CHECK(cudaStreamSynchronize(stream));
    for (int64_t b = 0; b < batchN; ++b)
      if (hostInfo[b] != 0)
        throw std::runtime_error("MAGMA GESVJ batched matrix " + std::to_string(b) +
          " returned info=" + std::to_string(hostInfo[b]));
  };

  // Untimed warmup supplies the singular triplets used for validation.
  LIBRARY_CHECK(cudaMemcpyAsync(dA, matA.data(), bytes, cudaMemcpyHostToDevice, stream));
  check_magma(gesvj(queue, M, N, dA, dS, dU, dV, info, batchN), "magma_gesvj_batched_strided");
  check_info();
  std::vector<double> singularValues(k * batchN);
  LIBRARY_CHECK(cudaMemcpy(singularValues.data(), dS, singularValues.size() * sizeof(double), cudaMemcpyDeviceToHost));
  std::vector<int> ranks(batchN);
  for (int64_t b = 0; b < batchN; ++b)
    ranks[b] = validation_rank(singularValues.data() + b * k, k, eps);

  // DGMM requires a diagonal of the matrix's scalar type, including zero
  // imaginary components for complex singular values.
  std::vector<T> diagonal(singularValues.size());
  std::transform(singularValues.begin(), singularValues.end(), diagonal.begin(),
    [](double s) { return T(s); });
  T *dSigma = nullptr, *dUS = nullptr;
  LIBRARY_CHECK(cudaMalloc((void**)(&dSigma), diagonal.size() * sizeof(T)));
  LIBRARY_CHECK(cudaMalloc((void**)(&dUS), M * k * sizeof(T)));
  LIBRARY_CHECK(cudaMemcpyAsync(dSigma, diagonal.data(), diagonal.size() * sizeof(T), cudaMemcpyHostToDevice, stream));
  LIBRARY_CHECK(cudaMemcpyAsync(dA, matA.data(), bytes, cudaMemcpyHostToDevice, stream));
  double norm = 0., error = 0.;
  LIBRARY_CHECK(vendor_norm(blas, elements, dA, 1, &norm));
  const T one(1.), minus_one(-1.);
  for (int64_t b = 0; b < batchN; ++b) {
    const int rank = ranks[b];
    if (rank == 0) continue; // This block's reconstructed B is zero.
    LIBRARY_CHECK(scale_columns(blas, M, rank, dU + b * M * k, dSigma + b * k, dUS));
    // Accumulate A_b - U_b*S_b*V_b^H in its original block of the global matrix.
    LIBRARY_CHECK(vendor_gemm(blas, CUBLAS_OP_N, CUBLAS_OP_C, M, N, rank,
      &minus_one, dUS, M, dV + b * N * k, N, &one, dA + b * M * N, M));
  }
  LIBRARY_CHECK(vendor_norm(blas, elements, dA, 1, &error));
  error = norm == 0. ? 0. : error / norm;
  LIBRARY_CHECK(cudaFree(dSigma));
  LIBRARY_CHECK(cudaFree(dUS));

  double kernelTime = 0.;
  for (int run = 0; run < kernel_runs; ++run) {
    LIBRARY_CHECK(cudaMemcpyAsync(dA, matA.data(), bytes, cudaMemcpyHostToDevice, stream));
    LIBRARY_CHECK(cudaEventRecord(start, stream));
    check_magma(gesvj(queue, M, N, dA, dS, dU, dV, info, batchN), "magma_gesvj_batched_strided");
    LIBRARY_CHECK(cudaEventRecord(stop, stream));
    check_info();
    float milliseconds = 0.f;
    LIBRARY_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    kernelTime += milliseconds;
  }
  kernelTime /= double(kernel_runs);
  // GESVD-equivalent work for comparison with the other SVD benchmarks.
  // Actual Jacobi work depends on sweeps, which the strided driver does not expose.
  const double flops = double(batchN) * gesvd_flops(double(std::max(M, N)), double(k), vendor_flop_factor<T>);
  const auto rankBounds = std::minmax_element(ranks.begin(), ranks.end());
  const double meanRank = double(std::accumulate(ranks.begin(), ranks.end(), int64_t(0))) / double(batchN);
  std::printf("MAGMA-%cGESVJ-BATCH [M=%ld,N=%ld,batchN=%ld] [eps=%.3le] [err_USVH=%.12le] "
    "[rank_min=%d,rank_mean=%.3f,rank_max=%d] [kernel=%lf ms] [flops=%.6le] [%lf GFlops]\n",
    vendor_precision<T>, M, N, batchN, eps, error, *rankBounds.first, meanRank, *rankBounds.second,
    kernelTime, flops, (1.e-6 * flops) / kernelTime);

  LIBRARY_CHECK(cudaEventDestroy(start));
  LIBRARY_CHECK(cudaEventDestroy(stop));
  LIBRARY_CHECK(cudaFree(info));
  LIBRARY_CHECK(cudaFree(dS));
  LIBRARY_CHECK(cudaFree(dV));
  LIBRARY_CHECK(cudaFree(dU));
  LIBRARY_CHECK(cudaFree(dA));
  magma_queue_destroy(queue);
  check_magma(magma_finalize(), "magma_finalize");
  LIBRARY_CHECK(cudaGetLastError());
  return 0;
}

} // namespace

int main(int argc, char* argv[]) try {
  char precision = 'D';
  for (int i = 1; i < argc; ++i)
    if (std::strncmp(argv[i], "data=", 5) == 0) precision = argv[i][5];
  switch (precision) {
    case 'D': return run<double>(argc, argv);
    case 'Z': return run<std::complex<double>>(argc, argv);
    default: throw std::invalid_argument("Supported precisions: data=D or data=Z");
  }
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return EXIT_FAILURE;
}
