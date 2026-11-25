
#include <vendor_common.hpp>
#include <magma_v2.h>
#include <iostream>
#include <chrono>

inline void check_magma_info(magma_int_t info, const char* routine) {
  if (info != 0) throw std::runtime_error(std::string(routine) + " returned status/info=" + std::to_string(info));
}
inline void vendor_geqp3(int m, int n, double* A, int lda, int* piv, double* tau,
    double* work, int lwork, double*, int* info) {
  check_magma_info(magma_dgeqp3_gpu(m, n, A, lda, piv, tau, work, lwork, info), "magma_dgeqp3_gpu");
  check_magma_info(*info, "magma_dgeqp3_gpu");
}
inline void vendor_geqp3(int m, int n, std::complex<double>* A, int lda, int* piv, std::complex<double>* tau,
    std::complex<double>* work, int lwork, double* rwork, int* info) {
  check_magma_info(magma_zgeqp3_gpu(m, n, reinterpret_cast<magmaDoubleComplex*>(A), lda, piv,
    reinterpret_cast<magmaDoubleComplex*>(tau), reinterpret_cast<magmaDoubleComplex*>(work), lwork, rwork, info), "magma_zgeqp3_gpu");
  check_magma_info(*info, "magma_zgeqp3_gpu");
}
template <class T>
int64_t geqp3_workspace(int m, int n, magma_queue_t queue) {
  magma_int_t host_bytes = -1, device_bytes = -1, info = 0;
  if constexpr (std::is_same_v<T, double>) {
    check_magma_info(magma_dgeqp3_expert_gpu_work(m, n, nullptr, m, nullptr, nullptr,
      nullptr, &host_bytes, nullptr, &device_bytes, &info, queue), "magma_dgeqp3 workspace query");
  } else {
    check_magma_info(magma_zgeqp3_expert_gpu_work(m, n, nullptr, m, nullptr, nullptr,
      nullptr, &host_bytes, nullptr, &device_bytes, &info, queue), "magma_zgeqp3 workspace query");
  }
  check_magma_info(info, "MAGMA workspace query");
  return (int64_t(device_bytes) + sizeof(T) - 1) / sizeof(T);
}
inline cublasStatus_t vendor_trsm(cublasHandle_t h, int rank, int n, const double* one, double* A, int lda, double* B) {
  return cublasDtrsm(h, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, rank, n, one, A, lda, B, lda);
}
inline cublasStatus_t vendor_trsm(cublasHandle_t h, int rank, int n, const std::complex<double>* one, std::complex<double>* A, int lda, std::complex<double>* B) {
  return cublasZtrsm(h, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, rank, n,
    reinterpret_cast<const cuDoubleComplex*>(one), reinterpret_cast<cuDoubleComplex*>(A), lda, reinterpret_cast<cuDoubleComplex*>(B), lda);
}
inline void vendor_laset(magma_uplo_t uplo, int m, int n, double zero, double one, double* A, int lda, magma_queue_t queue) {
  magmablas_dlaset(uplo, m, n, zero, one, A, lda, queue);
}
inline void vendor_laset(magma_uplo_t uplo, int m, int n, std::complex<double> zero, std::complex<double> one, std::complex<double>* A, int lda, magma_queue_t queue) {
  magmablas_zlaset(uplo, m, n, MAGMA_Z_MAKE(zero.real(), zero.imag()), MAGMA_Z_MAKE(one.real(), one.imag()), reinterpret_cast<magmaDoubleComplex*>(A), lda, queue);
}

template <class T>
double check_answer_lra(int32_t rank, int32_t M, int32_t N, const T* A, int32_t lda, const int32_t* jpiv, const T* R, int32_t ldr) {
  if (rank <= 0 || M <= 0 || N <= 0) { return 0.; }
  constexpr bool Complex = std::is_same_v<T, std::complex<double>>;
  using type = typename std::conditional<Complex, std::complex<double>, double>::type;
  std::vector<type> matB(int64_t(M) * int64_t(N)), matC(int64_t(M) * int64_t(rank)), matR(int64_t(rank) * int64_t(N));
  for (int32_t i = 0; i < rank; ++i)
    vendor_copy2d(M, 1, &A[int64_t(jpiv[i] - 1) * int64_t(lda)], lda, &matC[int64_t(i) * int64_t(M)], M);
  for (int32_t i = 0; i < N; ++i)
    vendor_copy2d(1, rank, &R[int64_t(i) * int64_t(ldr)], 1, &matR[jpiv[i] - 1], N);
  vendor_copy2d(M, N, A, lda, &matB[0], M);
  if constexpr(Complex) { for (std::complex<double>& f : matR) { f = std::conj(f); }}
  ncgemm(M, N, rank, &matC[0], M, &matR[0], N, &matB[0], M);
  double err = std::transform_reduce(matB.begin(), matB.end(), 0., std::plus<double>(), [](auto i) { return std::norm(i); });
  return err;
}

template <class T>
int32_t run(int32_t argc, char* argv[]) {
  int64_t M = 2048, N = 2048;
  double eps = 1.e-12;
  for (int32_t i = 1; i < argc; ++i) {
    if (std::strncmp(argv[i], "M=", 2) == 0) { std::sscanf(argv[i], "M=%ld", &M); }
    else if (std::strncmp(argv[i], "N=", 2) == 0) { std::sscanf(argv[i], "N=%ld", &N); }
    else if (std::strncmp(argv[i], "eps=", 4) == 0) { std::sscanf(argv[i], "eps=%le", &eps); }
    else if (std::strncmp(argv[i], "data=", 5) == 0) {}
    else { std::cerr << "Ignored parameter: " << argv[i] << std::endl; }
  }
  if (M <= 0 || N <= 0 || eps < 0. || !std::isfinite(eps)) throw std::invalid_argument("Require M,N > 0 and finite eps >= 0");
  N = std::min(M, N);

  LIBRARY_CHECK(cudaSetDevice(0));
  check_magma_info(magma_init(), "magma_init");

  magma_queue_t queue = nullptr; magma_queue_create(0, &queue);
  cublasHandle_t cublasH = magma_queue_get_cublas_handle(queue);
  cudaStream_t stream = magma_queue_get_cuda_stream(queue);

  T* dA, *dC; double* dR;
  std::vector<T> tau(N);
  std::vector<magma_int_t> jpvt(N, 0);

  std::vector<T> matA(M * N), matB(M * N);
  matrix_generator<T>(1., M, N).generate_block(512, 512, &matA[0], M);

  /* Timed region start */
  auto host_start = std::chrono::high_resolution_clock::now();

  int64_t Lwork = geqp3_workspace<T>(M, N, queue); magma_int_t info;
  LIBRARY_CHECK(cudaMalloc((void**)&dA, M * N * sizeof(T)));
  LIBRARY_CHECK(cudaMalloc((void**)&dC, Lwork * sizeof(T)));
  LIBRARY_CHECK(cudaMalloc((void**)&dR, 2 * N * sizeof(double)));
  LIBRARY_CHECK(cudaMemcpy(dA, &matA[0], M * N * sizeof(T), cudaMemcpyHostToDevice));

  vendor_geqp3(M, N, dA, M, jpvt.data(), tau.data(), dC, Lwork, dR, &info);
  LIBRARY_CHECK(cudaStreamSynchronize(stream));

  LIBRARY_CHECK(cudaMemcpy(&matB[0], dA, M * N * sizeof(T), cudaMemcpyDeviceToHost));
  std::vector<double> fnorms(N, 0.0), norms_sum(N);
  for (int64_t i = 0; i < N; ++i) for (int64_t j = i; j < N; ++j) { fnorms[i] += std::norm(matB[i + j * M]); }
  for (int64_t i = N - 1; 0 <= i; --i) { norms_sum[i] = i == N - 1 ? fnorms[i] : fnorms[i] + norms_sum[i + 1]; }
  double s0 = std::pow(eps, 2) * norms_sum[0];
  int64_t rank = 0; while (rank < N && s0 < norms_sum[rank]) { ++rank; }

  T one(1.), zero(0.);
  if (rank > 0 && rank < N) LIBRARY_CHECK(vendor_trsm(cublasH, rank, N - rank, &one, dA, M, &dA[rank * M]));
  vendor_laset(MagmaFull, rank, rank, zero, one, dA, M, queue);
  LIBRARY_CHECK(cudaStreamSynchronize(stream));

  LIBRARY_CHECK(cudaMemcpy(&matB[0], dA, M * N * sizeof(T), cudaMemcpyDeviceToHost));
  double nrm = vendor_fnorm(M, N, &matA[0], M);
  double err = nrm == 0. ? std::numeric_limits<double>::quiet_NaN() : std::sqrt(check_answer_lra(rank, M, N, matA.data(), M, jpvt.data(), matB.data(), M) / nrm);

  LIBRARY_CHECK(cudaMemcpy(dA, &matA[0], M * N * sizeof(T), cudaMemcpyHostToDevice));
  std::fill(jpvt.begin(), jpvt.end(), 0);

  cudaEvent_t start, stop;
  LIBRARY_CHECK(cudaEventCreate(&start));
  LIBRARY_CHECK(cudaEventCreate(&stop));

  double kernel_time = 0.;
  for (int32_t run = 0; run < kernel_runs; ++run) {
    LIBRARY_CHECK(cudaMemcpyAsync(dA, matA.data(), M * N * sizeof(T), cudaMemcpyHostToDevice, stream));
    LIBRARY_CHECK(cudaStreamSynchronize(stream)); // MAGMA uses its own internal queue.
    std::fill(jpvt.begin(), jpvt.end(), 0);
    LIBRARY_CHECK(cudaEventRecord(start, stream));
    LIBRARY_CHECK(cudaEventSynchronize(start)); // Order the start before MAGMA's internal queue.
    vendor_geqp3(M, N, dA, M, jpvt.data(), tau.data(), dC, Lwork, dR, &info);
    if (rank > 0 && rank < N) LIBRARY_CHECK(vendor_trsm(cublasH, rank, N - rank, &one, dA, M, &dA[rank * M]));
    vendor_laset(MagmaFull, rank, rank, zero, one, dA, M, queue);
    LIBRARY_CHECK(cudaEventRecord(stop, stream));
    LIBRARY_CHECK(cudaStreamSynchronize(stream));

    float milliseconds = 0.f;
    LIBRARY_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    kernel_time += milliseconds;
  }
  kernel_time /= double(kernel_runs);

  /* Timed region end */
  std::chrono::duration<double, std::milli> host_wtime = std::chrono::high_resolution_clock::now() - host_start;
  double duration = host_wtime.count();
  double flops = interpolative_flops(double(M), double(N), double(rank), vendor_flop_factor<T>);
  printf("MAGMA-%cGEQP3 [M=%ld,N=%ld] [eps=%.1le] [err=%.12le] [rank=%ld] [host=%lf ms] [kernel=%lf ms] [%lf GFlops]\n",
    vendor_precision<T>, M, N, eps, err, rank, duration, kernel_time, (1.e-6 * flops) / kernel_time);

  LIBRARY_CHECK(cudaFree(dA));
  LIBRARY_CHECK(cudaFree(dC));
  LIBRARY_CHECK(cudaFree(dR));

  magma_queue_destroy(queue);
  LIBRARY_CHECK(cudaEventDestroy(start));
  LIBRARY_CHECK(cudaEventDestroy(stop));
  check_magma_info(magma_finalize(), "magma_finalize");

  LIBRARY_CHECK(cudaGetLastError());
  return 0;
}

int32_t main(int32_t argc, char* argv[]) try {
  char precision = 'D';
  for (int32_t i = 1; i < argc; ++i)
    if (std::strncmp(argv[i], "data=", 5) == 0) precision = argv[i][5];
  switch (precision) {
    case 'D': return run<double>(argc, argv);
    case 'Z': return run<std::complex<double>>(argc, argv);
    default: throw std::invalid_argument("Supported precisions: data=D or data=Z");
  }
} catch (const std::exception& error) {
  std::cerr << error.what() << std::endl;
  return EXIT_FAILURE;
}
