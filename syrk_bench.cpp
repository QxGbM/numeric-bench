
#include <common.hpp>
#include <iostream>
#include <chrono>

template <class T>
double check_answer_syrk(int32_t M, int32_t N, const T* A, int32_t lda, const T* R, int32_t ldr) {
  if (M <= 0 || N <= 0) { return 0.; }
  constexpr int32_t Complex = std::is_same_v<T, std::complex<double>> || std::is_same_v<T, std::complex<float>> || std::is_same_v<T, __half2>;
  using type = typename std::conditional<Complex, std::complex<double>, double>::type;
  std::vector<type> matB(int64_t(N) * int64_t(M)), matR(int64_t(N) * int64_t(N));
  for (int32_t i = 0; i < N; ++i) { copy2d(1, M, &A[int64_t(i) * int64_t(lda)], 1, &matB[i], N); }
  if constexpr(Complex) { for (std::complex<double>& f : matB) { f = std::conj(f); }}
  copy2d(N, N, R, ldr, &matR[0], N);
  nngemm(N, N, M, &matB[0], N, &matB[0], N, &matR[0], N);
  double err = std::transform_reduce(matR.begin(), matR.end(), 0., std::plus<double>(), [](auto i) { return std::norm(i); });
  return err;
}

template <class T>
void syrk_hyac(hyacinHandle_t handle, char algo, double epi, int32_t u_corr, int32_t batchK, int32_t batchIter, int32_t M, int32_t N, const T* A, int32_t lda, T* R, int32_t ldr) {
  hyacinPrecision_t Atype = __precA<T>();
  int32_t* vexp = nullptr, u, cPanels, lPanels; uint64_t strideC;
  cudaMallocAsync((void**)&vexp, uint64_t(N) * sizeof(int32_t), handle.cudaStream);
  hyacinXGautoType(epi, u_corr, 0, M, N, Atype, &u, &cPanels, &lPanels, &strideC, nullptr, nullptr);
  hyacinXquantizeScale(handle, M, N, Atype, A, lda, u, 0, vexp);

  uint64_t* C = nullptr; cudaMallocAsync((void**)&C, uint64_t(cPanels) * uint64_t(lPanels) * strideC * sizeof(uint64_t), handle.cudaStream);
  void* param = hyacinXherkBatchCreate(handle, algo, epi, u_corr, batchK, N, Atype);

  int32_t beta = 0, iter = param ? batchIter : M; u = param ? HYACIN_QUERY_U : u;
  for (int32_t i = 0; i < M; i += iter)
  { int32_t rows = std::min(M - i, iter); hyacinXherkBatch(handle, algo, rows, N, Atype, &A[i], lda, u, vexp, &beta, lPanels, C, param); }
  hyacinXherkBatchFlush(handle, N, Atype, vexp, beta, lPanels, C, param);
  hyacinXherkBatchDestroy(handle, param);

  hyacinXdequantize(handle, N, lPanels, C, vexp, Atype, R, ldr);
  cudaFreeAsync(vexp, handle.cudaStream); cudaFreeAsync(C, handle.cudaStream);

  double eventMs[2]{ 'D', 'R' }; hyacinSync_TimerSegments(handle, eventMs, 2);
  kernel_time += eventMs[0] + eventMs[1];
}

template <class T> inline void run(char prec, char algo, double epi, int32_t u_corr, int32_t batchK, int32_t batchIter, int64_t M, int64_t N) {
  std::vector<T> matA(M * N);
  matrix_generator<T>(M, N).generate_block(1., 512, 512, &matA[0], M);

  /* Timed region start */
  auto host_start = std::chrono::high_resolution_clock::now();

  T* d_A = nullptr, * d_X = nullptr;
  cudaMalloc((void**)(&d_A), M * N * sizeof(T));
  cudaMalloc((void**)(&d_X), N * N * sizeof(T));
  cudaMemcpy(d_A, matA.data(), M * N * sizeof(T), cudaMemcpyHostToDevice);

  hyacinHandle_t handle;
  hyacinCreate(&handle, 1);

  syrk_hyac(handle, algo, epi, u_corr, batchK, batchIter, M, N, d_A, M, d_X, N);

  std::vector<T> matX(N * N);
  cudaMemcpy(matX.data(), d_X, N * N * sizeof(T), cudaMemcpyDeviceToHost);
  double nrm = fnorm(M, N, &matA[0], M);
  double err = nrm == 0. ? std::numeric_limits<double>::quiet_NaN() : std::sqrt(check_answer_syrk(M, N, matA.data(), M, matX.data(), N) / nrm);

  cudaMemcpy(d_A, matA.data(), M * N * sizeof(T), cudaMemcpyHostToDevice);
  kernel_time = 0.;

  for (int32_t i = 0; i < kernel_runs; ++i)
    syrk_hyac(handle, algo, epi, u_corr, batchK, batchIter, M, N, d_A, M, d_X, N);

  hyacinDestroy(handle);
  cudaFree(d_A);
  cudaFree(d_X);

  /* Timed region end */
  std::chrono::duration<double, std::milli> host_wtime = std::chrono::high_resolution_clock::now() - host_start;
  double duration = host_wtime.count(); kernel_time /= double(kernel_runs);
  uint64_t flops = uint64_t(N + 1) * uint64_t(N) * uint64_t(M);
  if constexpr(std::is_same_v<T, std::complex<double>> || std::is_same_v<T, std::complex<float>> || std::is_same_v<T, __half2>) { flops *= uint64_t(4); }

  printf("%c-SYRK [M=%ld,N=%ld] [epi=%.1le] [err=%.12le] [host=%lf ms] [kernel=%lf ms] [%lf TFlops]\n",
    prec, M, N, epi, err, duration, kernel_time, (1.e-9 * double(flops)) / kernel_time);
}

int32_t main(int32_t argc, char* argv[]) {
  char prec = 'D', algo = 'A';
  int64_t M = 2048, N = 2048; int32_t u_corr = 5, batchK = 65536, batchIter = 2048;
  double epi = 1.e-12;

  for (int32_t i = 1; i < argc; ++i) {
    if (std::strncmp(argv[i], "M=", 2) == 0) { std::sscanf(argv[i], "M=%ld", &M); }
    else if (std::strncmp(argv[i], "N=", 2) == 0) { std::sscanf(argv[i], "N=%ld", &N); }
    else if (std::strncmp(argv[i], "data=", 5) == 0) { std::sscanf(argv[i], "data=%c", &prec); }
    else if (std::strncmp(argv[i], "epi=", 4) == 0) { std::sscanf(argv[i], "epi=%lf", &epi); }
    else if (std::strncmp(argv[i], "algo=", 5) == 0) { std::sscanf(argv[i], "algo=%c", &algo); }
    else if (std::strncmp(argv[i], "u_corr=", 7) == 0) { std::sscanf(argv[i], "u_corr=%d", &u_corr); }
    else if (std::strncmp(argv[i], "batchK=", 7) == 0) { std::sscanf(argv[i], "batchK=%d", &batchK); }
    else if (std::strncmp(argv[i], "batchIter=", 10) == 0) { std::sscanf(argv[i], "batchIter=%d", &batchIter); }
    else { std::cerr << "Ignored parameter: " << argv[i] << std::endl; }
  }
  N = std::min(M, N);

  auto cu_err = cudaSetDevice(0);
  cudaDeviceReset();
  if (cu_err != cudaSuccess)
  { std::cerr << cudaGetErrorString(cu_err) << std::endl; return -1; }

  switch(prec) {
    case 'D': run<double>(prec, algo, epi, u_corr, batchK, batchIter, M, N); break;
    case 'S': run<float>(prec, algo, epi, u_corr, batchK, batchIter, M, N); break;
    case 'H': run<__half>(prec, algo, epi, u_corr, batchK, batchIter, M, N); break;
    case 'Z': run<std::complex<double>>(prec, algo, epi, u_corr, batchK, batchIter, M, N); break;
    case 'C': run<std::complex<float>>(prec, algo, epi, u_corr, batchK, batchIter, M, N); break;
    case 'J': run<__half2>(prec, algo, epi, u_corr, batchK, batchIter, M, N); break;
    default: break;
  }

  cu_err = cudaGetLastError();
  if (cu_err != cudaSuccess)
    std::cerr << cudaGetErrorString(cu_err) << std::endl;
  return 0;
}
