
#include <common.hpp>
#include <iostream>
#include <chrono>

template <class T> void make_2D_oscillatory(double w, int32_t sep, int32_t M, int32_t N, T* A, int32_t lda) {
  constexpr int32_t height = 128;
  auto translate_2d = [](int64_t i) { int64_t x = i / height, y = i - height * x; return std::complex<double>(x, y); };
  sep = height * sep + ((M + height - 1) & (~(height - 1)));

  for (int32_t j = 0; j < N; ++j) {
    auto vj = translate_2d(j + sep);
    for (int32_t i = 0; i < M; ++i) {
      auto vi = translate_2d(i);
      double d = std::abs(vi - vj);
      int64_t k = int64_t(i) + int64_t(j) * int64_t(lda);
      if constexpr(std::is_same_v<T, std::complex<double>> || std::is_same_v<T, std::complex<float>> || std::is_same_v<T, __half2>)
      { A[k] = conv<T>(std::complex<double>(std::cos(w * d) / d, std::sin(w * d) / d)); } else { A[k] = T(std::cos(w * d) / d); }
    }
  }
}

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
  hyacinPrecision_t precA = __precA<T>();
  int32_t* vexp = nullptr, cPanels, lPanels, u_floor;
  cudaMallocAsync((void**)&vexp, int64_t(N) * sizeof(int64_t), handle.cudaStream);
  hyacinXquantizeScale(handle, M, N, precA, A, lda, 0, vexp);
  int32_t u = hyacinXquantizeScaleFinalize(handle, epi, u_corr, M, N, precA, vexp, &cPanels, &lPanels, &u_floor);

  int64_t strideC = (int64_t(N) * int64_t(N + 1)) / int64_t(2);
  uint64_t* C = nullptr; cudaMallocAsync((void**)&C, int64_t(cPanels) * int64_t(lPanels) * int64_t(strideC) * sizeof(uint64_t), handle.cudaStream);
  //hyacinXherk(handle, algo, M, N, precA, A, lda, u, vexp, 0, lPanels, C);

  void* param = nullptr; uint64_t bytesBatch = 0; hyacinXherkBatchCreate(&param, algo, 65536, N, precA, u, u_floor, &bytesBatch);
  int8_t* batch = nullptr; cudaMallocAsync((void**)&batch, bytesBatch, handle.cudaStream);

  int32_t beta = 0;
  for (int32_t i = 0; i < M; i += 2048)
  { int32_t rows = std::min(M - i, 2048); hyacinXherkBatchProcessA(handle, algo, rows, N, precA, &A[i], lda, HYACIN_QUERY_U, vexp, &beta, lPanels, C, param, batch); }
  hyacinXherkBatchFlush(handle, N, precA, vexp, beta, lPanels, C, param, batch);
  hyacinXherkBatchDestroy(param); cudaFreeAsync(batch, handle.cudaStream);

  int32_t gElemBytes; hyacinPrecision_t Gtype = hyacinXGautoType(g_corr, M, precA, u, &gElemBytes);
  void* G = nullptr; cudaMallocAsync((void**)&G, int64_t(N) * int64_t(N) * int64_t(gElemBytes), handle.cudaStream);
  hyacinXdequantize(handle, N, lPanels, C, vexp, Gtype, G, N);
  cudaFreeAsync(vexp, handle.cudaStream); cudaFreeAsync(C, handle.cudaStream);

  int32_t* piv = nullptr; cudaMallocAsync((void**)&piv, int64_t(N) * sizeof(int32_t), handle.cudaStream);
  int32_t rank = hyacinXGinterp(handle, 'A', epi, N, K, oversampling, precA, R, ldr, (int32_t*)piv, Gtype, G, N);
  cudaMemcpyAsync(jpiv, piv, int64_t(N) * sizeof(int32_t), cudaMemcpyDefault, handle.cudaStream);
  cudaFreeAsync(G, handle.cudaStream); cudaFreeAsync(piv, handle.cudaStream);

  hyacinSync_TimerSegments(handle, &kernel_time, &comm_time);
  return rank;
}

template <class T> inline void run(char prec, int64_t M, int64_t N, double epi, char algo) {
  std::vector<T> matA(M * N);
  std::vector<int32_t> ipiv(N);
  matrix_generator<T>(M, N).generate_block(1., 512, 512, &matA[0], M);
  //make_2D_oscillatory(1., 0, M, N, &matA[0], M);

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
