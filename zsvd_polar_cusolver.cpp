
#include <cusolverDn.h>
#include <vector>
#include <complex>
#include <iostream>
#include <algorithm>
#include <random>

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
 
  std::vector<std::complex<double>> matA(M * N, 0.);
  std::mt19937_64 gen(42);
  std::normal_distribution<double> dist(0, 32);
  std::generate(matA.begin(), matA.end(), [&](){ return std::complex<double>(dist(gen), dist(gen)); });

  std::complex<double>* dA = nullptr, *dU = nullptr, *dV = nullptr;
  double* dS = nullptr;
  int32_t* info = nullptr;
  cudaMalloc((void**)(&dA), M * N * sizeof(std::complex<double>));
  cudaMalloc((void**)(&dU), M * N * sizeof(std::complex<double>));
  cudaMalloc((void**)(&dV), N * N * sizeof(std::complex<double>));
  cudaMalloc((void**)(&dS), N * sizeof(double));
  cudaMalloc((void**)(&info), sizeof(int32_t));
  cudaMemcpy(dA, matA.data(), M * N * sizeof(std::complex<double>), cudaMemcpyHostToDevice);

  size_t workspaceInBytesOnDevice, workspaceInBytesOnHost;
  cusolverDnXgesvdp_bufferSize(cusolverH, params, CUSOLVER_EIG_MODE_VECTOR, 1, M, N, 
    CUDA_C_64F, dA, M, CUDA_R_64F, dS, CUDA_C_64F, dU, M, CUDA_C_64F, dV, N, CUDA_C_64F, &workspaceInBytesOnDevice, &workspaceInBytesOnHost);

  double h_err = 0.;
  void* hWork = std::malloc(workspaceInBytesOnHost), *dWork;
  cudaMalloc(&dWork, workspaceInBytesOnDevice);

  auto status = cusolverDnXgesvdp(cusolverH, params, CUSOLVER_EIG_MODE_NOVECTOR, 1, M, N, 
    CUDA_C_64F, dA, M, CUDA_R_64F, dS, CUDA_C_64F, dU, M, CUDA_C_64F, dV, N, CUDA_C_64F, dWork, workspaceInBytesOnDevice, hWork, workspaceInBytesOnHost, info, &h_err);
  cudaDeviceSynchronize();
  int32_t hinfo;
  cudaMemcpy(&hinfo, info, sizeof(int32_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(dA, matA.data(), M * N * sizeof(std::complex<double>), cudaMemcpyHostToDevice);

  cudaEventRecord(start, stream);
  cusolverDnXgesvdp(cusolverH, params, CUSOLVER_EIG_MODE_VECTOR, 1, M, N, 
    CUDA_C_64F, dA, M, CUDA_R_64F, dS, CUDA_C_64F, dU, M, CUDA_C_64F, dV, N, CUDA_C_64F, dWork, workspaceInBytesOnDevice, hWork, workspaceInBytesOnHost, info, &h_err);
  cudaEventRecord(stop, stream);

  cudaDeviceSynchronize();
  float milliseconds = 0.0f;
  cudaEventElapsedTime(&milliseconds, start, stop);
  int64_t svd_flops = N * N * (4 * M + 8 * N);
  double gflops = double(svd_flops) * 1.e-6 / milliseconds;
  std::cout << "cusolver-ZGESVDP," << M << "," << N << "," << milliseconds << "," << gflops << "," << ((hinfo == 0 && status == CUSOLVER_STATUS_SUCCESS) ? "OK" : "ERR") << "," << h_err << std::endl;

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
