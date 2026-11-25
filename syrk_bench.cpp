
#include <hyacin_common.hpp>
#include <iostream>
#include <chrono>

template <class T>
void syrk_hyac(const hyacinHandle_t* handle, double eps, int32_t M, int32_t N, const T* A, int32_t lda, T* R, int32_t ldr) {
  int32_t batch = handle->GramSegmentK <= 0 ? M : std::min(handle->GramSegmentK, M);
  std::div_t divM = std::div(M, batch); int32_t batches = divM.quot + int32_t(0 < divM.rem), lastM = 0 < divM.rem ? divM.rem : batch;
  hyacinPrecision_t Atype = hyacin_prec<T>();
  int32_t* vexp = nullptr, u, cPanels, lPanels; uint64_t strideC;
  LIBRARY_CHECK(cudaMallocFromPoolAsync((void**)&vexp, uint64_t(2) * uint64_t(N) * sizeof(int32_t), handle->mempool, handle->cudaStream));
  hyacinXherkBatchInit(handle, eps, M, N, Atype, &u, &cPanels, &lPanels, &strideC);
  hyacinXquantizeScaleBatch(handle, batches, batch, lastM, N, Atype, A, lda, u, vexp);

  uint64_t* C = nullptr; LIBRARY_CHECK(cudaMallocFromPoolAsync((void**)&C, uint64_t(cPanels) * uint64_t(lPanels) * strideC * sizeof(uint64_t), handle->mempool, handle->cudaStream));
  hyacinXherkBatch(handle, batches, batch, lastM, N, Atype, A, lda, u, vexp, 0, lPanels, C);

  hyacinXdequantize(handle, N, lPanels, C, vexp, Atype, R, ldr);
  LIBRARY_CHECK(cudaFreeAsync(vexp, handle->cudaStream)); LIBRARY_CHECK(cudaFreeAsync(C, handle->cudaStream));
}

template <class T> inline void run(char prec, double eps, int64_t M, int64_t N) {
  std::vector<T> matA(M * N);
  matrix_generator<T>(1., M, N).generate_block(512, 512, &matA[0], M);

  /* Timed region start */
  auto host_start = std::chrono::high_resolution_clock::now();

  T* d_A = nullptr, * d_X = nullptr;
  LIBRARY_CHECK(cudaMalloc((void**)(&d_A), M * N * sizeof(T)));
  LIBRARY_CHECK(cudaMalloc((void**)(&d_X), N * N * sizeof(T)));
  LIBRARY_CHECK(cudaMemcpy(d_A, matA.data(), M * N * sizeof(T), cudaMemcpyHostToDevice));

  hyacinHandle_t handle{};
  hyacinCreate(&handle);

  syrk_hyac(&handle, eps, M, N, d_A, M, d_X, N);
  hyacinSync_TimerSegments(&handle, nullptr, 0);

  for (int32_t i = 0; i < kernel_runs; ++i)
    syrk_hyac(&handle, eps, M, N, d_A, M, d_X, N);

  double eventMs[2]{ 'D', 'R' }; hyacinSync_TimerSegments(&handle, eventMs, 2);
  double kernel_time = eventMs[0] + eventMs[1];
  hyacinDestroy(&handle);
  LIBRARY_CHECK(cudaFree(d_A));
  LIBRARY_CHECK(cudaFree(d_X));

  /* Timed region end */
  std::chrono::duration<double, std::milli> host_wtime = std::chrono::high_resolution_clock::now() - host_start;
  double duration = host_wtime.count(); kernel_time /= double(kernel_runs);
  uint64_t flops = uint64_t(N + 1) * uint64_t(N) * uint64_t(M);
  if constexpr(std::is_same_v<T, std::complex<double>> || std::is_same_v<T, std::complex<float>> || std::is_same_v<T, __half2>) { flops *= uint64_t(4); }

  printf("%c-SYRK [M=%ld,N=%ld] [eps=%.1le] [host=%lf ms] [kernel=%lf ms] [%lf TFlops]\n",
    prec, M, N, eps, duration, kernel_time, (1.e-9 * double(flops)) / kernel_time);
}

int32_t main(int32_t argc, char* argv[]) try {
  char prec = 'D', algo = 'A';
  int64_t M = 2048, N = 2048; int32_t u_corr = 7, batch = 8192;
  double eps = 1.e-12;

  for (int32_t i = 1; i < argc; ++i) {
    if (std::strncmp(argv[i], "M=", 2) == 0) { std::sscanf(argv[i], "M=%ld", &M); }
    else if (std::strncmp(argv[i], "N=", 2) == 0) { std::sscanf(argv[i], "N=%ld", &N); }
    else if (std::strncmp(argv[i], "data=", 5) == 0) { std::sscanf(argv[i], "data=%c", &prec); }
    else if (std::strncmp(argv[i], "eps=", 4) == 0) { std::sscanf(argv[i], "eps=%lf", &eps); }
    else if (std::strncmp(argv[i], "algo=", 5) == 0) { std::sscanf(argv[i], "algo=%c", &algo); }
    else if (std::strncmp(argv[i], "u_corr=", 7) == 0) { std::sscanf(argv[i], "u_corr=%d", &u_corr); }
    else if (std::strncmp(argv[i], "batch=", 6) == 0) { std::sscanf(argv[i], "batch=%d", &batch); }
    else { std::cerr << "Ignored parameter: " << argv[i] << std::endl; }
  }
  handle_param_environments(algo, batch, u_corr, 0);

  LIBRARY_CHECK(cudaSetDevice(0));
  LIBRARY_CHECK(cudaDeviceReset());

  switch(prec) {
    case 'D': run<double>(prec, eps, M, N); break;
    case 'S': run<float>(prec, eps, M, N); break;
    case 'H': run<__half>(prec, eps, M, N); break;
    case 'Z': run<std::complex<double>>(prec, eps, M, N); break;
    case 'C': run<std::complex<float>>(prec, eps, M, N); break;
    case 'J': run<__half2>(prec, eps, M, N); break;
    default: break;
  }

  LIBRARY_CHECK(cudaGetLastError());
  return 0;
}

catch (const std::exception& error) {
  fprintf(stderr, "%s\n", error.what());
  return EXIT_FAILURE;
}
