#include <vendor_common.hpp>
#include <kblas.h>
#include <iostream>

namespace {

void check_kblas(int status, const char* routine) {
  if (status != KBLAS_Success)
    throw std::runtime_error(std::string(routine) + ": " + kblasGetErrorString(status) +
      " (status=" + std::to_string(status) + ")");
}

int validation_rank(const std::vector<double>& sortedS, double eps) {
  std::vector<double> tail(sortedS.size());
  std::transform_inclusive_scan(sortedS.rbegin(), sortedS.rend(), tail.rbegin(),
    std::plus<double>(), [](double s) { return s * s; });
  if (tail[0] < std::numeric_limits<double>::min()) return 0;
  const double threshold = tail[0] * std::pow(eps, 2);
  return std::distance(tail.begin(), std::find_if(tail.begin(), tail.end(),
    [threshold](double s) { return s < threshold; }));
}

int run(int argc, char* argv[]) {
  int64_t M = 256, N = 256, K = 64, batchN = 16, seed = 0;
  double eps = 1.e-12;
  char data = 'D';
  for (int i = 1; i < argc; ++i) {
    if (std::strncmp(argv[i], "M=", 2) == 0) std::sscanf(argv[i], "M=%ld", &M);
    else if (std::strncmp(argv[i], "N=", 2) == 0) std::sscanf(argv[i], "N=%ld", &N);
    else if (std::strncmp(argv[i], "K=", 2) == 0) std::sscanf(argv[i], "K=%ld", &K);
    else if (std::strncmp(argv[i], "batchN=", 7) == 0) std::sscanf(argv[i], "batchN=%ld", &batchN);
    else if (std::strncmp(argv[i], "eps=", 4) == 0) std::sscanf(argv[i], "eps=%le", &eps);
    else if (std::strncmp(argv[i], "seed=", 5) == 0) std::sscanf(argv[i], "seed=%ld", &seed);
    else if (std::strncmp(argv[i], "data=", 5) == 0) data = argv[i][5];
    else std::cerr << "Ignored parameter: " << argv[i] << '\n';
  }
  if (data != 'D')
    throw std::invalid_argument("KBLAS GERSVD supports real S/D only; this FP64 benchmark requires data=D (no data=Z API)");
  const int64_t intLimit = std::numeric_limits<int>::max();
  if (M <= 0 || M > intLimit || N <= 0 || N > intLimit || K <= 0 || K > std::min(M, N) || K == N ||
      batchN <= 0 || batchN > std::numeric_limits<int>::max() || eps < 0. || !std::isfinite(eps) ||
      seed < 0 || static_cast<uint64_t>(seed) > std::numeric_limits<unsigned int>::max())
    throw std::invalid_argument("Require positive int M,N,batchN, 1 <= K <= min(M,N) with K < N, finite eps >= 0, and unsigned seed");
  // GERSVD applies QR to M-by-K and N-by-K matrices. Both encounter the
  // installed FP64 QR kernel's 1024-row cap, independently of lda alignment.
  if (M > 1024 || N > 1024)
    throw std::invalid_argument("KBLAS FP64 batched QR has a 1024-row limit; require M,N <= 1024");
  // The strided interface and several workspace intermediates use 32-bit ints.
  if (M > intLimit / N || ((M + 31) / 32 * 32) > intLimit / N ||
      N*K + M*K + K*N + K*K + K > intLimit)
    throw std::invalid_argument("Matrix strides and per-block KBLAS workspace must fit int");
  if (static_cast<uint64_t>(M*N) * batchN > std::numeric_limits<size_t>::max() / sizeof(double))
    throw std::overflow_error("Batched matrix allocation size overflows");
  const int strideA = M * N, strideU = M * K, strideV = N * K;
  const int strideS = std::min(M, N);
  const int64_t elements = int64_t(strideA) * batchN;
  const size_t bytes = static_cast<size_t>(elements) * sizeof(double);

  LIBRARY_CHECK(cudaSetDevice(0));
  cudaStream_t stream;
  LIBRARY_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamDefault));
  kblasHandle_t handle = nullptr;
  check_kblas(kblasCreate(&handle), "kblasCreate");
  kblasSetStream(handle, stream);
  LIBRARY_CHECK(cudaGetLastError());
  const cublasHandle_t blas = kblasGetCublasHandle(handle);
  kblasRandState_t state = nullptr;
  check_kblas(kblasInitRandState(handle, &state, 32768, static_cast<unsigned int>(seed)), "kblasInitRandState");
  kblasDsvd_full_batch_wsquery(handle, M, N, K, batchN, SVD_random);
  check_kblas(kblasAllocateWorkspace(handle), "kblasAllocateWorkspace");

  // One generated M-by-(N*batchN) matrix; each N-column block is one batch member.
  std::vector<double> matA(elements);
  matrix_generator<double>(1., M, N * batchN).generate_block(512, 512, matA.data(), M);
  double *dA = nullptr, *dU = nullptr, *dV = nullptr, *dS = nullptr;
  LIBRARY_CHECK(cudaMalloc((void**)(&dA), bytes));
  LIBRARY_CHECK(cudaMalloc((void**)(&dU), int64_t(strideU) * batchN * sizeof(double)));
  LIBRARY_CHECK(cudaMalloc((void**)(&dV), int64_t(strideV) * batchN * sizeof(double)));
  LIBRARY_CHECK(cudaMalloc((void**)(&dS), int64_t(strideS) * batchN * sizeof(double)));
  LIBRARY_CHECK(cudaMemsetAsync(dS, 0, int64_t(strideS) * batchN * sizeof(double), stream));
  auto factorize = [&] {
    check_kblas(kblasDsvd_full_batch_strided(handle, M, N, K,
      dA, M, strideA, dS, strideS, dU, M, strideU, dV, N, strideV,
      SVD_random, state, batchN), "kblasDsvd_full_batch_strided (SVD_random)");
  };

  // Warmup and validation are outside the measured repetitions.
  LIBRARY_CHECK(cudaMemcpyAsync(dA, matA.data(), bytes, cudaMemcpyHostToDevice, stream));
  factorize();
  LIBRARY_CHECK(cudaStreamSynchronize(stream));
  std::vector<double> singularValues(int64_t(strideS) * batchN);
  LIBRARY_CHECK(cudaMemcpy(singularValues.data(), dS, singularValues.size() * sizeof(double), cudaMemcpyDeviceToHost));
  std::vector<int> ranks(batchN), order(K);
  std::vector<double> sortedS(K);
  double *validationU = nullptr, *validationV = nullptr;
  LIBRARY_CHECK(cudaMalloc((void**)(&validationU), strideU * sizeof(double)));
  LIBRARY_CHECK(cudaMalloc((void**)(&validationV), strideV * sizeof(double)));
  LIBRARY_CHECK(cudaMemcpyAsync(dA, matA.data(), bytes, cudaMemcpyHostToDevice, stream));
  double norm = 0., error = 0.;
  LIBRARY_CHECK(vendor_norm(blas, elements, dA, 1, &norm));
  const double one = 1., minusOne = -1.;
  for (int64_t b = 0; b < batchN; ++b) {
    const double* S = singularValues.data() + b * strideS;
    for (int j = 0; j < K; ++j)
      if (!std::isfinite(S[j]) || S[j] < 0.)
        throw std::runtime_error("KBLAS returned an invalid singular value in batch " + std::to_string(b));
    // The blocked inner SVD need not sort its output. Apply the same permutation
    // to the values and both factors, only for validation.
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [S](int i, int j) { return S[i] > S[j]; });
    for (int j = 0; j < K; ++j) sortedS[j] = S[order[j]];
    const int rank = ranks[b] = validation_rank(sortedS, eps);
    if (rank == 0) continue;
    for (int j = 0; j < rank; ++j) {
      LIBRARY_CHECK(cudaMemcpyAsync(validationU + int64_t(j) * M, dU + b * strideU + int64_t(order[j]) * M,
        M * sizeof(double), cudaMemcpyDeviceToDevice, stream));
      LIBRARY_CHECK(cudaMemcpyAsync(validationV + int64_t(j) * N, dV + b * strideV + int64_t(order[j]) * N,
        N * sizeof(double), cudaMemcpyDeviceToDevice, stream));
    }
    // KBLAS returns U and V=A^T*U. S is already absorbed into V: B=U*V^T.
    LIBRARY_CHECK(vendor_gemm(blas, CUBLAS_OP_N, CUBLAS_OP_T, M, N, rank,
      &minusOne, validationU, M, validationV, N, &one, dA + b * strideA, M));
  }
  LIBRARY_CHECK(vendor_norm(blas, elements, dA, 1, &error));
  error = norm == 0. ? 0. : error / norm;
  if (!std::isfinite(error)) throw std::runtime_error("KBLAS reconstruction produced a nonfinite residual");
  LIBRARY_CHECK(cudaFree(validationV));
  LIBRARY_CHECK(cudaFree(validationU));

  cudaEvent_t start, stop;
  LIBRARY_CHECK(cudaEventCreate(&start));
  LIBRARY_CHECK(cudaEventCreate(&stop));
  double kernelTime = 0.;
  for (int run = 0; run < kernel_runs; ++run) {
    LIBRARY_CHECK(cudaMemcpyAsync(dA, matA.data(), bytes, cudaMemcpyHostToDevice, stream));
    LIBRARY_CHECK(cudaEventRecord(start, stream));
    factorize();
    LIBRARY_CHECK(cudaEventRecord(stop, stream));
    LIBRARY_CHECK(cudaStreamSynchronize(stream));
    float milliseconds = 0.f;
    LIBRARY_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    kernelTime += milliseconds;
  }
  kernelTime /= double(kernel_runs);
  // Leading-order RSVD estimate plus the left-vector rotation and final A^T*U.
  // Inner Jacobi work depends on convergence and is not counted exactly.
  const double perMatrixFlops = randomized_svd_flops(double(M), double(N), double(K), 0., 1.) +
    2.*M*K*K + 2.*M*N*K;
  const double flops = double(batchN) * perMatrixFlops;
  const auto bounds = std::minmax_element(ranks.begin(), ranks.end());
  const double meanRank = double(std::accumulate(ranks.begin(), ranks.end(), int64_t(0))) / double(batchN);
  std::printf("KBLAS-DGERSVD-BATCH [M=%ld,N=%ld,K=%ld,batchN=%ld] [eps=%.3le] [seed=%ld] "
    "[err_UVT=%.12le] [rank_min=%d,rank_mean=%.3f,rank_max=%d] [kernel=%lf ms] [flops=%.6le] [%lf GFlops]\n",
    M, N, K, batchN, eps, seed, error, *bounds.first, meanRank, *bounds.second,
    kernelTime, flops, (1.e-6 * flops) / kernelTime);

  LIBRARY_CHECK(cudaEventDestroy(start));
  LIBRARY_CHECK(cudaEventDestroy(stop));
  LIBRARY_CHECK(cudaFree(dS));
  LIBRARY_CHECK(cudaFree(dV));
  LIBRARY_CHECK(cudaFree(dU));
  LIBRARY_CHECK(cudaFree(dA));
  check_kblas(kblasDestroyRandState(state), "kblasDestroyRandState");
  check_kblas(kblasFreeWorkspace(handle), "kblasFreeWorkspace");
  check_kblas(kblasDestroy(&handle), "kblasDestroy");
  LIBRARY_CHECK(cudaStreamDestroy(stream));
  LIBRARY_CHECK(cudaGetLastError());
  return 0;
}

} // namespace

int main(int argc, char* argv[]) try {
  return run(argc, argv);
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return EXIT_FAILURE;
}
