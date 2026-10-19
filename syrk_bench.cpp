
#include <common.hpp>
#include <iostream>
#include <chrono>

template <class T> inline hyacinPrecision_t hyacin_prec();
template <> inline hyacinPrecision_t hyacin_prec<double>() { return HYACIN_F64; };
template <> inline hyacinPrecision_t hyacin_prec<float>() { return HYACIN_F32; };
template <> inline hyacinPrecision_t hyacin_prec<__half>() { return HYACIN_F16; };
template <> inline hyacinPrecision_t hyacin_prec<std::complex<double>>() { return HYACIN_F64_COMPLEX; };
template <> inline hyacinPrecision_t hyacin_prec<std::complex<float>>() { return HYACIN_F32_COMPLEX; };
template <> inline hyacinPrecision_t hyacin_prec<__half2>() { return HYACIN_F16_COMPLEX; };

template <class T>
void syrk_hyac(hyacinHandle_t handle, double epi, int32_t batchIter, int32_t M, int32_t N, const T* A, int32_t lda, T* R, int32_t ldr) {
  batchIter = batchIter <= 0 ? M : std::min(batchIter, M);
  std::div_t divM = std::div(M, batchIter); int32_t batches = divM.quot + int32_t(0 < divM.rem), lastM = 0 < divM.rem ? divM.rem : batchIter;
  hyacinPrecision_t Atype = hyacin_prec<T>();
  int32_t* vexp = nullptr, u, cPanels, lPanels; uint64_t strideC;
  cudaMallocFromPoolAsync((void**)&vexp, uint64_t(2) * uint64_t(N) * sizeof(int32_t), handle.mempool, handle.cudaStream);
  hyacinXherkBatchInit(&handle, epi, M, N, Atype, &u, &cPanels, &lPanels, &strideC);
  hyacinXquantizeScaleBatch(&handle, batches, batchIter, lastM, N, Atype, A, lda, u, vexp);

  uint64_t* C = nullptr; cudaMallocFromPoolAsync((void**)&C, uint64_t(cPanels) * uint64_t(lPanels) * strideC * sizeof(uint64_t), handle.mempool, handle.cudaStream);
  hyacinXherkBatch(&handle, batches, batchIter, lastM, N, Atype, A, lda, u, vexp, 0, lPanels, C);

  hyacinXdequantize(&handle, N, lPanels, C, vexp, Atype, R, ldr);
  cudaFreeAsync(vexp, handle.cudaStream); cudaFreeAsync(C, handle.cudaStream);
}

template <class T> inline void run(char prec, double epi, int32_t batchIter, int64_t M, int64_t N) {
  std::vector<T> matA(M * N);
  matrix_generator<T>(1., M, N).generate_block(512, 512, &matA[0], M);

  /* Timed region start */
  auto host_start = std::chrono::high_resolution_clock::now();

  T* d_A = nullptr, * d_X = nullptr;
  cudaMalloc((void**)(&d_A), M * N * sizeof(T));
  cudaMalloc((void**)(&d_X), N * N * sizeof(T));
  cudaMemcpy(d_A, matA.data(), M * N * sizeof(T), cudaMemcpyHostToDevice);

  hyacinHandle_t handle;
  hyacinCreate(&handle);

  syrk_hyac(handle, epi, batchIter, M, N, d_A, M, d_X, N);
  hyacinSync_TimerSegments(&handle, nullptr, 0);

  for (int32_t i = 0; i < kernel_runs; ++i)
    syrk_hyac(handle, epi, batchIter, M, N, d_A, M, d_X, N);

  double eventMs[2]{ 'D', 'R' }; hyacinSync_TimerSegments(&handle, eventMs, 2);
  double kernel_time = eventMs[0] + eventMs[1];
  hyacinDestroy(&handle);
  cudaFree(d_A);
  cudaFree(d_X);

  /* Timed region end */
  std::chrono::duration<double, std::milli> host_wtime = std::chrono::high_resolution_clock::now() - host_start;
  double duration = host_wtime.count(); kernel_time /= double(kernel_runs);
  uint64_t flops = uint64_t(N + 1) * uint64_t(N) * uint64_t(M);
  if constexpr(std::is_same_v<T, std::complex<double>> || std::is_same_v<T, std::complex<float>> || std::is_same_v<T, __half2>) { flops *= uint64_t(4); }

  printf("%c-SYRK [M=%ld,N=%ld] [epi=%.1le] [host=%lf ms] [kernel=%lf ms] [%lf TFlops]\n",
    prec, M, N, epi, duration, kernel_time, (1.e-9 * double(flops)) / kernel_time);
}

int32_t main(int32_t argc, char* argv[]) {
  char prec = 'D', algo = 'A';
  int64_t M = 2048, N = 2048; int32_t u_corr = 5, batchIter = 2048;
  double epi = 1.e-12;

  for (int32_t i = 1; i < argc; ++i) {
    if (std::strncmp(argv[i], "M=", 2) == 0) { std::sscanf(argv[i], "M=%ld", &M); }
    else if (std::strncmp(argv[i], "N=", 2) == 0) { std::sscanf(argv[i], "N=%ld", &N); }
    else if (std::strncmp(argv[i], "data=", 5) == 0) { std::sscanf(argv[i], "data=%c", &prec); }
    else if (std::strncmp(argv[i], "epi=", 4) == 0) { std::sscanf(argv[i], "epi=%lf", &epi); }
    else if (std::strncmp(argv[i], "algo=", 5) == 0) { std::sscanf(argv[i], "algo=%c", &algo); }
    else if (std::strncmp(argv[i], "u_corr=", 7) == 0) { std::sscanf(argv[i], "u_corr=%d", &u_corr); }
    else if (std::strncmp(argv[i], "batchIter=", 10) == 0) { std::sscanf(argv[i], "batchIter=%d", &batchIter); }
    else { std::cerr << "Ignored parameter: " << argv[i] << std::endl; }
  }
  N = std::min(M, N);
  handle_param_environments(algo, u_corr, 0, 0, 0);

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
