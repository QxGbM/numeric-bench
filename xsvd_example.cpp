
#include <common.hpp>
#include <iostream>
#include <chrono>

template <class T, class R> inline void run(char prec, int64_t M, int64_t N, int64_t K, char algo, double epi, const std::string& file, const std::string& out) {
  std::vector<T> matA(M * N);
  if (!file.empty())
    matrix_from_row_major_csv(M, N, 512, 512, matA.data(), M, file);
  else
    matrix_generator<T>(M, N).generate_block(1., 512, 512, &matA[0], M);

  /* Timed region start */
  auto host_start = std::chrono::high_resolution_clock::now();

  T* d_A = nullptr, *d_U = nullptr, *d_V = nullptr; R* d_S = nullptr;
  cudaMalloc((void**)(&d_A), M * N * sizeof(T));
  cudaMalloc((void**)(&d_U), M * K * sizeof(T));
  cudaMalloc((void**)(&d_V), K * N * sizeof(T));
  cudaMalloc((void**)(&d_S), K * sizeof(R));
  cudaMemcpy(d_A, matA.data(), M * N * sizeof(T), cudaMemcpyHostToDevice);

  hyacinHandle_t handle;
  hyacinCreate(&handle, 1);

  int32_t rank = 0; double err = std::numeric_limits<double>::quiet_NaN();
  if (1 < kernel_runs) {
    rank = svd_fit_transform(handle, algo, epi, M, M, N, K, d_A, M, d_U, M, d_S, d_V, N, N);

    std::vector<T> matU(M * K), matV(K * N);
    cudaMemcpy(matU.data(), d_U, M * K * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(matV.data(), d_V, K * N * sizeof(T), cudaMemcpyDeviceToHost);

    if (!out.empty())
      write_matrix_to_csv(N, rank, &matV[0], N, out);

    double nrm = fnorm(M, N, &matA[0], M);
    err = nrm == 0. ? std::numeric_limits<double>::quiet_NaN() : std::sqrt(check_answer_svd(M, N, rank, &matU[0], M, &matV[0], N, &matA[0], M) / nrm);
    kernel_time = comm_time = 0.;
  }

  for (int32_t i = 0; i < kernel_runs; ++i)
    rank = svd_fit_transform(handle, algo, epi, M, M, N, K, d_A, M, d_U, M, d_S, d_V, N, N);

  hyacinDestroy(handle);
  std::vector<R> vecS(K);
  cudaMemcpy(vecS.data(), d_S, K * sizeof(R), cudaMemcpyDeviceToHost);
  cudaFree(d_A); cudaFree(d_U); cudaFree(d_S); cudaFree(d_V);

  /* Timed region end */
  std::chrono::duration<double, std::milli> host_wtime = std::chrono::high_resolution_clock::now() - host_start;
  double duration = host_wtime.count();

  printf("%c-SVD [M=%ld,N=%ld,K=%ld] [epi=%.1le] [err=%.12le] [rank=%d] [host=%lf ms] [kernel=%lf ms] [comm=%lf ms]\n",
    prec, M, N, K, epi, err, rank, duration, kernel_time / double(kernel_runs), comm_time / double(kernel_runs));
}

int32_t main(int32_t argc, char* argv[]) {
  char prec = 'D', algo = 'A'; std::string file, out;
  int64_t M = 2048, N = 2048, K = 2048;
  double epi = 1.e-12;

  for (int32_t i = 1; i < argc; ++i) {
    if (std::strncmp(argv[i], "M=", 2) == 0) { std::sscanf(argv[i], "M=%ld", &M); }
    else if (std::strncmp(argv[i], "N=", 2) == 0) { std::sscanf(argv[i], "N=%ld", &N); }
    else if (std::strncmp(argv[i], "K=", 2) == 0) { std::sscanf(argv[i], "K=%ld", &K); }
    else if (std::strncmp(argv[i], "data=", 5) == 0) { std::sscanf(argv[i], "data=%c", &prec); }
    else if (std::strncmp(argv[i], "epi=", 4) == 0) { std::sscanf(argv[i], "epi=%lf", &epi); }
    else if (std::strncmp(argv[i], "file=", 5) == 0) { file.resize(std::strlen(argv[i])); std::sscanf(argv[i], "file=%s", file.data()); }
    else if (std::strncmp(argv[i], "out=", 4) == 0) { out.resize(std::strlen(argv[i])); std::sscanf(argv[i], "out=%s", out.data()); }
    else if (std::strncmp(argv[i], "algo=", 5) == 0) { std::sscanf(argv[i], "algo=%c", &algo); }
    else { std::cerr << "Ignored parameter: " << argv[i] << std::endl; }
  }
  N = std::min(M, N); K = std::min(N, K);

  auto cu_err = cudaSetDevice(0);
  cudaDeviceReset();
  if (cu_err != cudaSuccess)
  { std::cerr << cudaGetErrorString(cu_err) << std::endl; return -1; }

  switch(prec) {
    case 'D': run<double, double>(prec, M, N, K, algo, epi, file, out); break;
    case 'S': run<float, float>(prec, M, N, K, algo, epi, file, out); break;
    case 'H': run<__half, __half>(prec, M, N, K, algo, epi, file, out); break;
    case 'Z': run<std::complex<double>, double>(prec, M, N, K, algo, epi, file, out); break;
    case 'C': run<std::complex<float>, float>(prec, M, N, K, algo, epi, file, out); break;
    case 'J': run<__half2, __half>(prec, M, N, K, algo, epi, file, out); break;
    default: break;
  }

  cu_err = cudaGetLastError();
  if (cu_err != cudaSuccess)
    std::cerr << cudaGetErrorString(cu_err) << std::endl;
  return 0;
}
