
#include <common.hpp>
#include <iostream>
#include <chrono>

template <class T>
double check_answer_lra(int32_t rank, int32_t M, int32_t N, const T* A, int32_t lda, const int32_t* jpiv, const T* R, int32_t ldr) {
  if (rank <= 0 || M <= 0 || N <= 0) { return 0.; }
  constexpr int32_t Complex = std::is_same_v<T, std::complex<double>> || std::is_same_v<T, std::complex<float>> || std::is_same_v<T, __half2>;
  using type = typename std::conditional<Complex, std::complex<double>, double>::type;
  std::vector<type> matB(int64_t(M) * int64_t(N)), matC(int64_t(M) * int64_t(rank)), matR(int64_t(rank) * int64_t(N));
  for (int32_t i = 0; i < rank; ++i)
    copy2d(M, 1, &A[int64_t(jpiv[i] - 1) * int64_t(lda)], lda, &matC[int64_t(i) * int64_t(M)], M);
  copy2d(M, N, A, lda, &matB[0], M); copy2d(N, rank, R, ldr, &matR[0], N);
  nngemm(M, N, rank, &matC[0], M, &matR[0], N, &matB[0], M);
  double err = std::transform_reduce(matB.begin(), matB.end(), 0., std::plus<double>(), [](auto i) { return std::norm(i); });
  return err;
}

template <class T>
int32_t id_hyac(hyacinHandle_t handle, double epi, int32_t batchIter, int32_t M, int32_t N, int32_t K, const T* A, int32_t lda, int32_t* jpiv, T* R, int32_t ldr) {
  hyacinPrecision_t Atype = hyacin_prec<T>(), Gtype;
  int32_t* vexp = nullptr, u, cPanels, lPanels, gElemBytes; uint64_t strideC, Bbytes;
  cudaMallocFromPoolAsync((void**)&vexp, uint64_t(N) * sizeof(int32_t), handle.mempool, handle.cudaStream);
  hyacinXGautoType(&handle, epi, M, N, Atype, &u, &cPanels, &lPanels, &strideC, &Gtype, &gElemBytes);
  hyacinXquantizeScale(&handle, M, N, Atype, A, lda, u, 0, vexp);

  uint64_t* C = nullptr; cudaMallocFromPoolAsync((void**)&C, uint64_t(cPanels) * uint64_t(lPanels) * strideC * sizeof(uint64_t), handle.mempool, handle.cudaStream);
  if (handle.BatchK <= 0) { hyacinXherk(&handle, M, N, Atype, A, lda, u, vexp, 0, lPanels, C); } else {
    int8_t* Bdata = nullptr; hyacinXherkBatchInit(&handle, epi, N, Atype, &Bbytes);
    cudaMallocFromPoolAsync((void**)&Bdata, Bbytes, handle.mempool, handle.cudaStream);

    int32_t beta = 0, iter = std::min(batchIter, handle.BatchK);
    int64_t strideA = int64_t(lda) * int64_t(sizeof(T)), strideB = int64_t(handle.BatchK) * int64_t(sizeof(T));
    void* Arena = hyacinXherkBatch(&handle, 0, iter, N, Atype, vexp, &beta, lPanels, C, Bdata);
    for (int32_t i = 0; i < M; i += iter) {
      int32_t rows = std::min(M - i, iter);
      cudaMemcpy2DAsync(Arena, strideB, &A[i], strideA, int64_t(rows) * int64_t(sizeof(T)), N, cudaMemcpyDeviceToDevice, handle.cudaStream);
      Arena = hyacinXherkBatch(&handle, rows, iter, N, Atype, vexp, &beta, lPanels, C, Bdata);
    }
    hyacinXherkBatchFlush(&handle, N, Atype, vexp, beta, lPanels, C, Bdata);
    cudaFreeAsync(Bdata, handle.cudaStream);
  }

  void* G = nullptr; cudaMallocFromPoolAsync((void**)&G, uint64_t(N) * uint64_t(N) * uint64_t(gElemBytes), handle.mempool, handle.cudaStream);
  hyacinXdequantize(&handle, N, lPanels, C, vexp, Gtype, G, N);
  cudaFreeAsync(vexp, handle.cudaStream); cudaFreeAsync(C, handle.cudaStream);

  int32_t* piv = nullptr; cudaMallocFromPoolAsync((void**)&piv, uint64_t(N) * sizeof(int32_t), handle.mempool, handle.cudaStream);
  int32_t rank = hyacinXGinterp(&handle, 'A', epi, N, K, Atype, R, ldr, (int32_t*)piv, Gtype, G, N);
  cudaMemcpyAsync(jpiv, piv, int64_t(N) * sizeof(int32_t), cudaMemcpyDefault, handle.cudaStream);
  cudaFreeAsync(G, handle.cudaStream); cudaFreeAsync(piv, handle.cudaStream);

  double eventMs[2]{ 'D', 'R' }; hyacinSync_TimerSegments(&handle, eventMs, 2);
  kernel_time += eventMs[0] + eventMs[1];
  return rank;
}

template <class T> inline void run(char prec, double epi, int32_t batchIter, int64_t M, int64_t N) {
  std::vector<T> matA(M * N);
  std::vector<int32_t> ipiv(N);
  matrix_generator<T>(1., M, N).generate_block(512, 512, &matA[0], M);

  /* Timed region start */
  auto host_start = std::chrono::high_resolution_clock::now();

  T* d_A = nullptr, * d_X = nullptr;
  cudaMalloc((void**)(&d_A), M * N * sizeof(T));
  cudaMalloc((void**)(&d_X), N * N * sizeof(T));
  cudaMemcpy(d_A, matA.data(), M * N * sizeof(T), cudaMemcpyHostToDevice);

  hyacinHandle_t handle;
  hyacinCreate(&handle);

  int32_t rank = id_hyac(handle, epi, batchIter, M, N, N, d_A, M, ipiv.data(), d_X, N);
  cudaStreamSynchronize(handle.cudaStream);

  std::vector<T> matX(N * N);
  cudaMemcpy(matX.data(), d_X, N * N * sizeof(T), cudaMemcpyDeviceToHost);
  double nrm = fnorm(M, N, &matA[0], M);
  double err = nrm == 0. ? std::numeric_limits<double>::quiet_NaN() : std::sqrt(check_answer_lra(rank, M, N, matA.data(), M, ipiv.data(), matX.data(), N) / nrm);

  std::fill(ipiv.begin(), ipiv.end(), 0);
  cudaMemcpy(d_A, matA.data(), M * N * sizeof(T), cudaMemcpyHostToDevice);
  kernel_time = 0.;

  for (int32_t i = 0; i < kernel_runs; ++i)
    rank = id_hyac(handle, epi, batchIter, M, N, N, d_A, M, ipiv.data(), d_X, N);

  hyacinDestroy(&handle);
  cudaFree(d_A);
  cudaFree(d_X);

  /* Timed region end */
  std::chrono::duration<double, std::milli> host_wtime = std::chrono::high_resolution_clock::now() - host_start;
  double duration = host_wtime.count();

  printf("%c-LRA [M=%ld,N=%ld] [epi=%.1le] [err=%.12le] [rank=%d] [host=%lf ms] [kernel=%lf ms]\n",
    prec, M, N, epi, err, rank, duration, kernel_time / double(kernel_runs));
}

int32_t main(int32_t argc, char* argv[]) {
  char prec = 'D', algo = 'A';
  int64_t M = 2048, N = 2048; int32_t u_corr = 6, g_corr = -5, oversampling = 10, batchK = 65536, batchIter = 2048;
  double epi = 1.e-12;

  for (int32_t i = 1; i < argc; ++i) {
    if (std::strncmp(argv[i], "M=", 2) == 0) { std::sscanf(argv[i], "M=%ld", &M); }
    else if (std::strncmp(argv[i], "N=", 2) == 0) { std::sscanf(argv[i], "N=%ld", &N); }
    else if (std::strncmp(argv[i], "data=", 5) == 0) { std::sscanf(argv[i], "data=%c", &prec); }
    else if (std::strncmp(argv[i], "epi=", 4) == 0) { std::sscanf(argv[i], "epi=%lf", &epi); }
    else if (std::strncmp(argv[i], "algo=", 5) == 0) { std::sscanf(argv[i], "algo=%c", &algo); }
    else if (std::strncmp(argv[i], "u_corr=", 7) == 0) { std::sscanf(argv[i], "u_corr=%d", &u_corr); }
    else if (std::strncmp(argv[i], "g_corr=", 7) == 0) { std::sscanf(argv[i], "g_corr=%d", &g_corr); }
    else if (std::strncmp(argv[i], "p=", 2) == 0) { std::sscanf(argv[i], "p=%d", &oversampling); }
    else if (std::strncmp(argv[i], "batchK=", 7) == 0) { std::sscanf(argv[i], "batchK=%d", &batchK); }
    else if (std::strncmp(argv[i], "batchIter=", 10) == 0) { std::sscanf(argv[i], "batchIter=%d", &batchIter); }
    else { std::cerr << "Ignored parameter: " << argv[i] << std::endl; }
  }
  N = std::min(M, N);
  handle_param_environments(algo, batchK, u_corr, g_corr, 0, oversampling);

  auto cu_err = cudaSetDevice(0);
  cudaDeviceReset();
  if (cu_err != cudaSuccess)
  { std::cerr << cudaGetErrorString(cu_err) << std::endl; return -1; }

  switch(prec) {
    case 'D': run<double>(prec, epi, batchIter, M, N); break;
    case 'S': run<float>(prec, epi, batchIter, M, N); break;
    case 'H': run<__half>(prec, epi, batchIter, M, N); break;
    case 'Z': run<std::complex<double>>(prec, epi, batchIter, M, N); break;
    case 'C': run<std::complex<float>>(prec, epi, batchIter, M, N); break;
    case 'J': run<__half2>(prec, epi, batchIter, M, N); break;
    default: break;
  }

  cu_err = cudaGetLastError();
  if (cu_err != cudaSuccess)
    std::cerr << cudaGetErrorString(cu_err) << std::endl;
  return 0;
}
