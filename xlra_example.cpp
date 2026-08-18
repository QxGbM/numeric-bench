
#include <common.hpp>
#include <iostream>
#include <chrono>

template <class T>
double check_answer_lra(int32_t rank, int32_t M, int32_t N, const T* A, int32_t lda, const int32_t* jpiv, const T* R, int32_t ldr) {
  if (rank <= 0 || M <= 0 || N <= 0) return std::numeric_limits<double>::quiet_NaN();
  if constexpr(std::is_same_v<T, std::complex<double>> || std::is_same_v<T, std::complex<float>> || std::is_same_v<T, __half2>) {
    std::vector<std::complex<double>> matB(int64_t(M) * int64_t(N)), matC(int64_t(M) * int64_t(rank)), matR(int64_t(rank) * int64_t(N));
    for (int32_t i = 0; i < rank; ++i)
      copy2d(M, 1, &A[int64_t(jpiv[i] - 1) * int64_t(lda)], lda, &matC[int64_t(i) * int64_t(M)], M);
    copy2d(M, N, A, lda, &matB[0], M); copy2d(N, rank, R, ldr, &matR[0], N);
    nngemm(M, N, rank, &matC[0], M, &matR[0], N, &matB[0], M);
    double err = std::transform_reduce(matB.begin(), matB.end(), 0., std::plus<double>(), [](auto i) { return std::norm(i); });
    return err;
  }
  else {
    std::vector<double> matB(int64_t(M) * int64_t(N)), matC(int64_t(M) * int64_t(rank)), matR(int64_t(rank) * int64_t(N));
    for (int32_t i = 0; i < rank; ++i)
      copy2d(M, 1, &A[int64_t(jpiv[i] - 1) * int64_t(lda)], lda, &matC[int64_t(i) * int64_t(M)], M);
    copy2d(M, N, A, lda, &matB[0], M); copy2d(N, rank, R, ldr, &matR[0], N);
    nngemm(M, N, rank, &matC[0], M, &matR[0], N, &matB[0], M);
    double err = std::transform_reduce(matB.begin(), matB.end(), 0., std::plus<double>(), [](auto i) { return std::norm(i); });
    return err;
  }
}

template <class T>
int32_t id_hyac(hyacinHandle_t handle, double epi, int32_t M, int32_t N, int32_t K, const T* A, int32_t lda, int32_t* jpiv, T* R, int32_t ldr, char algo) {
  int32_t umax; hyacinPrecision_t precA = __precA<T>(), precC; hyacinAlgorithm_t alg = HYACIN_ALG_AUTO;

  hyacinXsyherk_autoTune(epi, u_extra, &umax, precA, &precC);
  if (algo == 'C') alg = HYACIN_ALG_CRT; else if (algo == 'L') alg = HYACIN_ALG_LIMBS;

  int32_t c_bytes = hyacinXelem('A', &precC);

  void* gram = nullptr, *piv = nullptr;
  cudaMalloc(&gram, int64_t(N) * int64_t(N) * int64_t(c_bytes));
  cudaMalloc(&piv, int64_t(N) * sizeof(int32_t));

  hyacinXsyherk(handle, M, N, umax, precA, A, lda, precC, gram, N, alg);
  int32_t rank = hyacinXGinterp(handle, 'U', epi, N, K, oversampling, precA, R, ldr, (int32_t*)piv, precC, gram, N);

  hyacinSync_TimerSegments(handle, &kernel_time, &comm_time);
  cudaMemcpy(jpiv, piv, sizeof(int32_t) * N, cudaMemcpyDefault);
  cudaFree(gram); cudaFree(piv);
  return rank;
}

template <class T> inline void run(char prec, int64_t M, int64_t N, double epi, char algo) {
  std::vector<T> matA(M * N);
  std::vector<int32_t> ipiv(N);
  matrix_generator<T>(M, N).generate_block(1., 512, 512, &matA[0], M);

  T* d_A = nullptr, * d_X = nullptr;
  cudaMalloc((void**)(&d_A), M * N * sizeof(T));
  cudaMalloc((void**)(&d_X), N * N * sizeof(T));
  cudaMemcpy(d_A, matA.data(), M * N * sizeof(T), cudaMemcpyHostToDevice);

  /* Timed region start */
  auto host_start = std::chrono::high_resolution_clock::now();

  hyacinHandle_t handle;
  hyacinCreate(&handle, 1);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  double err = std::numeric_limits<double>::quiet_NaN();
  if (time_kernel) {
    int32_t rank = id_hyac(handle, epi, M, N, N, d_A, M, ipiv.data(), d_X, N, algo);
    cudaStreamSynchronize(handle.cudaStream);

    std::vector<T> matX(N * N);
    cudaMemcpy(matX.data(), d_X, N * N * sizeof(T), cudaMemcpyDeviceToHost);
    err = std::sqrt(check_answer_lra(rank, M, N, matA.data(), M, ipiv.data(), matX.data(), N) / fnorm(M, N, &matA[0], M));

    std::fill(ipiv.begin(), ipiv.end(), 0);
    cudaMemcpy(d_A, matA.data(), M * N * sizeof(T), cudaMemcpyHostToDevice);
    kernel_time = comm_time = 0.;
  }

  cudaEventRecord(start, handle.cudaStream);
  int32_t rank = id_hyac(handle, epi, M, N, N, d_A, M, ipiv.data(), d_X, N, algo);
  cudaEventRecord(stop, handle.cudaStream);

  cudaStreamSynchronize(handle.cudaStream);
  float milliseconds = 0.0f; cudaEventElapsedTime(&milliseconds, start, stop);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  hyacinDestroy(handle);

  /* Timed region end */
  auto host_end = std::chrono::high_resolution_clock::now();

  cudaFree(d_A);
  cudaFree(d_X);

  std::chrono::duration<double, std::milli> host_wtime = host_end - host_start;
  double duration = time_kernel ? double(milliseconds) : host_wtime.count();

  printf("%c-LRA [M=%ld,N=%ld] [epi=%.1le] [err=%.12le] [rank=%d] [tts=%lf ms] [kernel=%lf ms] [comm=%lf ms]\n",
    prec, M, N, epi, err, rank, duration, kernel_time, comm_time);
}

int32_t main(int32_t argc, char* argv[]) {
  char prec = 'D', algo = 'A';
  int64_t M = 2048, N = 2048;
  double epi = 1.e-12;

  for (int32_t i = 1; i < argc; ++i) {
    if (std::strncmp(argv[i], "M=", 2) == 0) { std::sscanf(argv[i], "M=%ld", &M); }
    else if (std::strncmp(argv[i], "N=", 2) == 0) { std::sscanf(argv[i], "N=%ld", &N); }
    else if (std::strncmp(argv[i], "data=", 5) == 0) { std::sscanf(argv[i], "data=%c", &prec); }
    else if (std::strncmp(argv[i], "epi=", 4) == 0) { std::sscanf(argv[i], "epi=%lf", &epi); }
    else if (std::strncmp(argv[i], "algo=", 5) == 0) { std::sscanf(argv[i], "algo=%c", &algo); }
    else { std::cerr << "Ignored parameter: " << argv[i] << std::endl; }
  }
  N = std::min(M, N);

  auto cu_err = cudaSetDevice(0);
  cudaDeviceReset();
  if (cu_err != cudaSuccess)
  { std::cerr << cudaGetErrorString(cu_err) << std::endl; return -1; }

  switch(prec) {
    case 'D': run<double>(prec, M, N, epi, algo); break;
    case 'S': run<float>(prec, M, N, epi, algo); break;
    case 'H': run<__half>(prec, M, N, epi, algo); break;
    case 'Z': run<std::complex<double>>(prec, M, N, epi, algo); break;
    case 'C': run<std::complex<float>>(prec, M, N, epi, algo); break;
    case 'J': run<__half2>(prec, M, N, epi, algo); break;
    default: break;
  }

  cu_err = cudaGetLastError();
  if (cu_err != cudaSuccess)
    std::cerr << cudaGetErrorString(cu_err) << std::endl;
  return 0;
}
