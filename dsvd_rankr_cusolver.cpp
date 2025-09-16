
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <vector>
#include <complex>
#include <iostream>
#include <algorithm>

void make_2D_oscillatory(double w, int32_t sep, int32_t M, int32_t N, double* A, int32_t lda) {
  constexpr int32_t height = 128;
  auto translate_2d = [](int64_t i) { int64_t x = i / height, y = i - height * x; return std::complex<double>(x, y); };
  sep = height * sep + ((M + height - 1) & (~(height - 1)));

  for (int32_t j = 0; j < N; ++j) {
    auto vj = translate_2d(j + sep);
    for (int32_t i = 0; i < M; ++i) {
      auto vi = translate_2d(i);
      double d = std::abs(vi - vj);
      A[uint64_t(i) + uint64_t(j) * uint64_t(lda)] = double(std::cos(w * d) / d);
    }
  }
}

int32_t main(int32_t argc, char* argv[]) {
  auto cu_err = cudaSetDevice(0);
  if (cu_err != cudaSuccess)
  { fprintf(stderr, "%s\n", cudaGetErrorString(cu_err)); return -1; }
  
  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

  cublasHandle_t cublasH;
  cublasCreate(&cublasH);
  cublasSetStream(cublasH, stream);

  cusolverDnHandle_t cusolverH;
  cusolverDnCreate(&cusolverH);
  cusolverDnSetStream(cusolverH, stream);

  cusolverDnParams_t params;
  cusolverDnCreateParams(&params);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  int64_t M = 1 < argc ? std::atoi(argv[1]) : 1024;
  int64_t N = 2 < argc ? std::atoi(argv[2]) : 128;
  N = std::min(M, N);

  int64_t k = 3 < argc ? std::atoi(argv[3]) : 128;
  int64_t oversampling = 4 < argc ? std::atoi(argv[4]) : 0;
  int32_t power_iter  = 5 < argc ? std::atoi(argv[5]) : 0;
  k = std::min(k, N);
 
  std::vector<double> matA(M * N, 0.);
  std::vector<double> S(N);
  make_2D_oscillatory(1., 0, M, N, matA.data(), M);

  double* dA = nullptr, *dU = nullptr, *dV = nullptr;
  double* dS = nullptr;
  int32_t* info = nullptr;
  cudaMalloc((void**)(&dA), M * N * sizeof(double));
  cudaMalloc((void**)(&dU), M * N * sizeof(double));
  cudaMalloc((void**)(&dV), N * N * sizeof(double));
  cudaMalloc((void**)(&dS), N * sizeof(double));
  cudaMalloc((void**)(&info), sizeof(int32_t));
  cudaMemcpy(dA, matA.data(), M * N * sizeof(double), cudaMemcpyHostToDevice);

  size_t workspaceInBytesOnDevice, workspaceInBytesOnHost;
  cusolverDnXgesvdr_bufferSize(cusolverH, params, 'S', 'S', M, N, k, oversampling, power_iter,
    CUDA_R_64F, dA, M, CUDA_R_64F, dS, CUDA_R_64F, dU, M, CUDA_R_64F, dV, N, CUDA_R_64F, &workspaceInBytesOnDevice, &workspaceInBytesOnHost);

  void* hWork = std::malloc(workspaceInBytesOnHost), *dWork;
  cudaMalloc(&dWork, workspaceInBytesOnDevice);

  auto status = cusolverDnXgesvdr(cusolverH, params, 'S', 'S', M, N, k, oversampling, power_iter,
    CUDA_R_64F, dA, M, CUDA_R_64F, dS, CUDA_R_64F, dU, M, CUDA_R_64F, dV, N, CUDA_R_64F, dWork, workspaceInBytesOnDevice, hWork, workspaceInBytesOnHost, info);
  cudaDeviceSynchronize();
  int32_t hinfo;
  cudaMemcpy(&hinfo, info, sizeof(int32_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(S.data(), dS, N * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(dA, matA.data(), M * N * sizeof(double), cudaMemcpyHostToDevice);

  double nrm = 0., err = 0.; double minus_one = -1., one = 1.;
  cublasDnrm2_64(cublasH, M * N, dA, int64_t(1), &nrm);
  for (int32_t i = 0; i < k; ++i)
    cublasDscal(cublasH, M, &S[i], &dU[i * M], 1);
  cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_C, M, N, k, &minus_one, dU, M, dV, N, &one, dA, M);
  cublasDnrm2_64(cublasH, M * N, dA, int64_t(1), &err);

  cudaDeviceSynchronize();
  cudaMemcpy(dA, matA.data(), M * N * sizeof(double), cudaMemcpyHostToDevice);
  err = err / nrm;

  cudaEventRecord(start, stream);
  cusolverDnXgesvdr(cusolverH, params, 'S', 'S', M, N, k, oversampling, power_iter,
    CUDA_R_64F, dA, M, CUDA_R_64F, dS, CUDA_R_64F, dU, M, CUDA_R_64F, dV, N, CUDA_R_64F, dWork, workspaceInBytesOnDevice, hWork, workspaceInBytesOnHost, info);
  cudaEventRecord(stop, stream);

  cudaDeviceSynchronize();
  float milliseconds = 0.0f;
  cudaEventElapsedTime(&milliseconds, start, stop);
  int64_t l = k + oversampling;
  int64_t svd_flops = (4 * (power_iter + 1) * M * N * l) + (l * l * (4 * M + 2 * N));
  double gflops = double(svd_flops) * 1.e-6 / milliseconds;
  std::cout << "cusolver-DGESVDR," << M << "," << N << "," << err << "," << k << "," << milliseconds << "," << gflops << "," << ((hinfo == 0 && status == CUSOLVER_STATUS_SUCCESS) ? "OK" : "ERR") << std::endl;

  cudaFree(dA);
  cudaFree(dU);
  cudaFree(dV);
  cudaFree(dS);
  cudaFree(info);
  cudaFree(dWork);
  std::free(hWork);
  cudaStreamDestroy(stream);
  cublasDestroy(cublasH);
  cusolverDnDestroyParams(params);
  cusolverDnDestroy(cusolverH);

  cu_err = cudaGetLastError();
  if (cu_err != cudaSuccess)
    std::cerr << cudaGetErrorString(cu_err) << std::endl;
  return 0;
}
