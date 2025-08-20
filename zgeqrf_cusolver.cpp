
#include <cusolverDn.h>
#include <vector>
#include <complex>
#include <iostream>
#include <algorithm>
#include <random>

int32_t main(int32_t argc, char* argv[]) {
  auto err = cudaSetDevice(0);
  if (err != cudaSuccess)
  { fprintf(stderr, "%s\n", cudaGetErrorString(err)); return -1; }
  
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
  std::cout << "cusolver ZGEQRF <" << M << ", " << N << ">\n";

  std::mt19937_64 gen(42);
  std::normal_distribution<double> dist(0., 32.);
  std::vector<double> matA(M * N * 2);
  std::generate(matA.begin(), matA.end(), [&]() { return dist(gen); });

  std::complex<double>* dA = nullptr, * dTau = nullptr;
  int32_t* info = nullptr;
  cudaMalloc((void**)(&dA), M * N * sizeof(std::complex<double>));
  cudaMalloc((void**)(&dTau), N * sizeof(std::complex<double>));
  cudaMalloc((void**)(&info), sizeof(int32_t));
  cudaMemcpy(dA, matA.data(), M * N * sizeof(std::complex<double>), cudaMemcpyHostToDevice);

  size_t workspaceInBytesOnDevice, workspaceInBytesOnHost;
  cusolverDnXgeqrf_bufferSize(cusolverH, params, M, N, CUDA_C_64F, dA, M, CUDA_C_64F, dTau, CUDA_C_64F, &workspaceInBytesOnDevice, &workspaceInBytesOnHost);

  void* hWork = std::malloc(workspaceInBytesOnHost), *dWork;
  cudaMalloc(&dWork, workspaceInBytesOnDevice);
  
  cudaEventRecord(start, stream);
  cusolverDnXgeqrf(cusolverH, params, M, N, CUDA_C_64F, dA, M, CUDA_C_64F, dTau, CUDA_C_64F, dWork, workspaceInBytesOnDevice, hWork, workspaceInBytesOnHost, info);
  cudaEventRecord(stop, stream);

  cudaDeviceSynchronize();
  float milliseconds = 0.0f;
  cudaEventElapsedTime(&milliseconds, start, stop);
  int64_t qr_flops = (N * N * N * -2 / 3) + (M * N * N * 2);
  std::cout << "Time: " << milliseconds << " ms\n";
  std::cout << "Total GFLOPs: " << double(qr_flops) * 1.e-6 / milliseconds << "\n" << std::endl;
 
  cudaFree(dA);
  cudaFree(dTau);
  cudaFree(info);
  cudaFree(dWork);
  std::free(hWork);
  cudaStreamDestroy(stream);
  cusolverDnDestroyParams(params);
  cusolverDnDestroy(cusolverH);
  std::cerr << cudaGetErrorString(cudaGetLastError()) << std::endl;
  return 0;
}
