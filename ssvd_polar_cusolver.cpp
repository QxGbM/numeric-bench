
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
  N = std::min(M, N);
  std::cout << "cusolver DGESVD <" << M << ", " << N << ">\n";
 
  std::mt19937_64 gen(42);
  std::normal_distribution<float> dist(0., 32.);
  std::vector<float> matA(M * N);
  std::generate(matA.begin(), matA.end(), [&]() { return dist(gen); });

  float* dA = nullptr, *dU = nullptr, *dV = nullptr;
  float* dS = nullptr;
  int32_t* info = nullptr;
  cudaMalloc((void**)(&dA), M * N * sizeof(float));
  cudaMalloc((void**)(&dU), M * N * sizeof(float));
  cudaMalloc((void**)(&dV), N * N * sizeof(float));
  cudaMalloc((void**)(&dS), N * sizeof(float));
  cudaMalloc((void**)(&info), sizeof(int32_t));
  cudaMemcpy(dA, matA.data(), M * N * sizeof(float), cudaMemcpyHostToDevice);

  size_t workspaceInBytesOnDevice, workspaceInBytesOnHost;
  cusolverDnXgesvdp_bufferSize(cusolverH, params, CUSOLVER_EIG_MODE_VECTOR, 1, M, N, 
    CUDA_R_32F, dA, M, CUDA_R_32F, dS, CUDA_R_32F, dU, M, CUDA_R_32F, dV, N, CUDA_R_32F, &workspaceInBytesOnDevice, &workspaceInBytesOnHost);

  double h_err = 0.;
  void* hWork = std::malloc(workspaceInBytesOnHost), *dWork;
  cudaMalloc(&dWork, workspaceInBytesOnDevice);

  cudaEventRecord(start, stream);
  cusolverDnXgesvdp(cusolverH, params, CUSOLVER_EIG_MODE_VECTOR, 1, M, N, 
    CUDA_R_32F, dA, M, CUDA_R_32F, dS, CUDA_R_32F, dU, M, CUDA_R_32F, dV, N, CUDA_R_32F, dWork, workspaceInBytesOnDevice, hWork, workspaceInBytesOnHost, info, &h_err);
  cudaEventRecord(stop, stream);

  cudaDeviceSynchronize();
  float milliseconds = 0.0f;
  cudaEventElapsedTime(&milliseconds, start, stop);
  int64_t svd_flops = N * N * (4 * M + 8 * N);
  std::cout << "Time: " << milliseconds << " ms\n";
  std::cout << "Polar Peturbation: " << h_err << "\n";
  std::cout << "Total GFLOPs: " << double(svd_flops) * 1.e-6 / milliseconds << "\n" << std::endl;

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
  std::cerr << cudaGetErrorString(cudaGetLastError()) << std::endl;
  return 0;
}
