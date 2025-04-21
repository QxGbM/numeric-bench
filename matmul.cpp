
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <omp.h>
#include <commons.hpp>

#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <mkl.h>

int32_t main() {
  cudaStream_t stream;
  cublasHandle_t handle;
  cudaStreamCreate(&stream);
  cublasCreate(&handle);
  cublasSetStream(handle, stream);

  const int64_t m = 1024, n = m, k = m;

  double* d_A, * d_B, * d_C;
  cudaMallocManaged(reinterpret_cast<void**>(&d_A), m * k * sizeof(double), cudaMemAttachGlobal);
  cudaMallocManaged(reinterpret_cast<void**>(&d_B), k * n * sizeof(double), cudaMemAttachGlobal);
  cudaMallocManaged(reinterpret_cast<void**>(&d_C), m * n * sizeof(double), cudaMemAttachGlobal);

  int64_t flops = m * n * k * 2;
  int32_t loops = 300;
  double gflops = flops * 1.e-9 * loops;
  double alpha = 1., beta = 0.;

  double start = omp_get_wtime();
  for (int32_t i = 0; i < loops; ++i)
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A, m, d_B, k, &beta, d_C, m);
  cudaDeviceSynchronize();
  double end = omp_get_wtime();

  printf("time: %f ms. GFLOPS: %f\n", (end - start) * 1000, gflops / (end - start));

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  cudaStreamDestroy(stream);
  cublasDestroy(handle);
  return 0;
}
