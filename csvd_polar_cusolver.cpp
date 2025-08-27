
#include <cusolverDn.h>
#include <vector>
#include <complex>
#include <iostream>
#include <algorithm>

void make_2D_oscillatory(double w, int32_t sep, int32_t M, int32_t N, std::complex<float>* A, int32_t lda) {
  auto translate_2d = [](int64_t i) { int64_t x = i / 128, y = i - 128 * x; return std::complex<double>(x, y); };
  sep = 128 * sep + ((M + 127) & (~127));

  for (int32_t j = 0; j < N; ++j) {
    auto vj = translate_2d(j + sep);
    for (int32_t i = 0; i < M; ++i) {
      auto vi = translate_2d(i);
      double d = std::abs(vi - vj);
      A[uint64_t(i) + uint64_t(j) * uint64_t(lda)] = std::complex<float>(std::cos(w * d) / d, std::sin(w * d) / d);
    }
  }
}

int32_t main(int32_t argc, char* argv[]) {
  auto cu_err = cudaSetDevice(0);
  if (cu_err != cudaSuccess)
  { fprintf(stderr, "%s\n", cudaGetErrorString(cu_err)); return -1; }
  
  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

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

  double omega = 3 < argc ? std::atof(argv[3]) : 0.01;
  int32_t sep = 4 < argc ? std::atoi(argv[4]) : 16;

  std::vector<std::complex<float>> matA(M * N);
  make_2D_oscillatory(omega, sep, M, N, matA.data(), M);

  std::complex<float>* dA = nullptr, *dU = nullptr, *dV = nullptr;
  float* dS = nullptr;
  int32_t* info = nullptr;
  cudaMalloc((void**)(&dA), M * N * sizeof(std::complex<float>));
  cudaMalloc((void**)(&dU), M * N * sizeof(std::complex<float>));
  cudaMalloc((void**)(&dV), N * N * sizeof(std::complex<float>));
  cudaMalloc((void**)(&dS), N * sizeof(float));
  cudaMalloc((void**)(&info), sizeof(int32_t));
  cudaMemcpy(dA, matA.data(), M * N * sizeof(std::complex<float>), cudaMemcpyHostToDevice);

  size_t workspaceInBytesOnDevice, workspaceInBytesOnHost;
  cusolverDnXgesvdp_bufferSize(cusolverH, params, CUSOLVER_EIG_MODE_VECTOR, 1, M, N, 
    CUDA_C_32F, dA, M, CUDA_R_32F, dS, CUDA_C_32F, dU, M, CUDA_C_32F, dV, N, CUDA_C_32F, &workspaceInBytesOnDevice, &workspaceInBytesOnHost);

  double h_err = 0.;
  void* hWork = std::malloc(workspaceInBytesOnHost), *dWork;
  cudaMalloc(&dWork, workspaceInBytesOnDevice);

  cusolverDnXgesvdp(cusolverH, params, CUSOLVER_EIG_MODE_VECTOR, 1, M, N, 
    CUDA_C_32F, dA, M, CUDA_R_32F, dS, CUDA_C_32F, dU, M, CUDA_C_32F, dV, N, CUDA_C_32F, dWork, workspaceInBytesOnDevice, hWork, workspaceInBytesOnHost, info, &h_err);
  cudaDeviceSynchronize();
  cudaMemcpy(dA, matA.data(), M * N * sizeof(std::complex<float>), cudaMemcpyHostToDevice);

  cudaEventRecord(start, stream);
  cusolverDnXgesvdp(cusolverH, params, CUSOLVER_EIG_MODE_VECTOR, 1, M, N, 
    CUDA_C_32F, dA, M, CUDA_R_32F, dS, CUDA_C_32F, dU, M, CUDA_C_32F, dV, N, CUDA_C_32F, dWork, workspaceInBytesOnDevice, hWork, workspaceInBytesOnHost, info, &h_err);
  cudaEventRecord(stop, stream);

  cudaDeviceSynchronize();
  float milliseconds = 0.0f;
  cudaEventElapsedTime(&milliseconds, start, stop);
  int64_t svd_flops = N * N * (4 * M + 8 * N);
  double gflops = double(svd_flops) * 1.e-6 / milliseconds;
  std::cout << "cusolver-DGESVD," << M << "," << N << "," << omega << "," << sep << "," << milliseconds << "," << h_err << "," << gflops << std::endl;

  cudaFree(dA);
  cudaFree(dU);
  cudaFree(dV);
  cudaFree(dS);
  cudaFree(info);
  cudaFree(dWork);
  std::free(hWork);
  cudaStreamDestroy(stream);
  cusolverDnDestroyParams(params);
  cusolverDnDestroy(cusolverH);

  cu_err = cudaGetLastError();
  if (cu_err != cudaSuccess)
    std::cerr << cudaGetErrorString(cu_err) << std::endl;
  return 0;
}
