
#include <common.hpp>
#include <iostream>
#include <chrono>

template <class T> inline hyacinPrecision_t hyacin_prec();
template <> inline hyacinPrecision_t hyacin_prec<double>() { return HYACIN_F64; };
template <> inline hyacinPrecision_t hyacin_prec<float>() { return HYACIN_F32; };
template <> inline hyacinPrecision_t hyacin_prec<__half>() { return HYACIN_F16; };
template <> inline hyacinPrecision_t hyacin_prec<cuDoubleComplex>() { return HYACIN_F64_COMPLEX; };
template <> inline hyacinPrecision_t hyacin_prec<cuComplex>() { return HYACIN_F32_COMPLEX; };
template <> inline hyacinPrecision_t hyacin_prec<__half2>() { return HYACIN_F16_COMPLEX; };

template <class T> inline void run(char prec, double epi, int64_t M, int64_t N) {
  std::vector<T> matA(M * N);
  std::vector<int32_t> ipiv(N);
  matrix_generator<T>(1., M, N).generate_block(512, 512, &matA[0], M);

  /* Timed region start */
  auto host_start = std::chrono::high_resolution_clock::now();

  T* d_A = nullptr, *d_X = nullptr; int32_t* jpvt = nullptr;
  cudaMalloc((void**)(&d_A), M * N * sizeof(T));
  cudaMalloc((void**)(&d_X), N * N * sizeof(T));
  cudaMalloc((void**)(&jpvt), N * sizeof(int32_t));
  cudaMemcpy(d_A, matA.data(), M * N * sizeof(T), cudaMemcpyHostToDevice);

  hyacinHandle_t handle{};
  hyacinCreate(&handle);
  int32_t rank = hyacinInterpolateFit(&handle, epi, 'N', M, N, N, d_A, M, jpvt, d_X, N, M, hyacin_prec<T>());
  hyacinSync_TimerSegments(&handle, nullptr, 0);

  for (int32_t i = 0; i < kernel_runs; ++i)
    rank = hyacinInterpolateFit(&handle, epi, 'N', M, N, N, d_A, M, jpvt, d_X, N, M, hyacin_prec<T>());

  double eventMs[2]{ 'D', 'R' }; hyacinSync_TimerSegments(&handle, eventMs, 2);
  double nrm = hyacinCheckNorm(&handle, M, N, d_A, M, 0, nullptr, 0, hyacin_prec<T>());
  double err = nrm == 0. ? std::numeric_limits<double>::quiet_NaN() : std::sqrt(hyacinCheckNorm(&handle, M, N, d_A, M, rank, d_X, N, hyacin_prec<T>()) / nrm);
  double kernel_time = eventMs[0] + eventMs[1];
  hyacinDestroy(&handle);

  cudaFree(d_A);
  cudaFree(d_X);
  cudaFree(jpvt);

  /* Timed region end */
  std::chrono::duration<double, std::milli> host_wtime = std::chrono::high_resolution_clock::now() - host_start;
  double duration = host_wtime.count();

  printf("%c-LRA [M=%ld,N=%ld] [epi=%.1le] [err=%.12le] [rank=%d] [host=%lf ms] [kernel=%lf ms]\n",
    prec, M, N, epi, err, rank, duration, kernel_time / double(kernel_runs));
}

int32_t main(int32_t argc, char* argv[]) {
  char prec = 'D', algo = 'A';
  int64_t M = 2048, N = 2048; int32_t u_corr = 7, batch = 8192;
  double epi = 1.e-12;

  for (int32_t i = 1; i < argc; ++i) {
    if (std::strncmp(argv[i], "M=", 2) == 0) { std::sscanf(argv[i], "M=%ld", &M); }
    else if (std::strncmp(argv[i], "N=", 2) == 0) { std::sscanf(argv[i], "N=%ld", &N); }
    else if (std::strncmp(argv[i], "data=", 5) == 0) { std::sscanf(argv[i], "data=%c", &prec); }
    else if (std::strncmp(argv[i], "epi=", 4) == 0) { std::sscanf(argv[i], "epi=%lf", &epi); }
    else if (std::strncmp(argv[i], "algo=", 5) == 0) { std::sscanf(argv[i], "algo=%c", &algo); }
    else if (std::strncmp(argv[i], "u_corr=", 7) == 0) { std::sscanf(argv[i], "u_corr=%d", &u_corr); }
    else if (std::strncmp(argv[i], "batch=", 6) == 0) { std::sscanf(argv[i], "batch=%d", &batch); }
    else { std::cerr << "Ignored parameter: " << argv[i] << std::endl; }
  }
  N = std::min(M, N);
  handle_param_environments(algo, batch, u_corr, 0);

  auto cu_err = cudaSetDevice(0);
  cudaDeviceReset();
  if (cu_err != cudaSuccess)
  { std::cerr << cudaGetErrorString(cu_err) << std::endl; return -1; }

  switch(prec) {
    case 'D': run<double>(prec, epi, M, N); break;
    case 'S': run<float>(prec, epi, M, N); break;
    case 'H': run<__half>(prec, epi, M, N); break;
    case 'Z': run<cuDoubleComplex>(prec, epi, M, N); break;
    case 'C': run<cuComplex>(prec, epi, M, N); break;
    case 'J': run<__half2>(prec, epi, M, N); break;
    default: break;
  }

  cu_err = cudaGetLastError();
  if (cu_err != cudaSuccess)
    std::cerr << cudaGetErrorString(cu_err) << std::endl;
  return 0;
}
