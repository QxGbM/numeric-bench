
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <complex>
#include <cstdint>
#include <omp.h>

int32_t main() {
  auto err = cudaSetDevice(0);
  if (err != cudaSuccess)
  { fprintf(stderr, "%s\n", cudaGetErrorString(err)); return -1; }

  cudaStream_t stream;
  cublasHandle_t handle;
  cudaStreamCreate(&stream);
  cublasCreate(&handle);
  cublasSetStream(handle, stream);

  int64_t m = 4096, n = 4096, k = 16384;
  int32_t iter_k = 4096;

  int8_t* d_A, * d_B;
  cuComplex* d_C;
  cudaMalloc(reinterpret_cast<void**>(&d_A), m * k * sizeof(int8_t));
  cudaMalloc(reinterpret_cast<void**>(&d_B), k * n * sizeof(int8_t));
  cudaMalloc(reinterpret_cast<void**>(&d_C), 4 * m * n * sizeof(int32_t));

  int64_t flops = m * n * k * 2;
  int32_t loops = 10;
  double gflops = flops * 1.e-9 * loops;

  double start = omp_get_wtime();
  int32_t i_alpha = 1, i_beta = 0;
  for (int32_t i = 0; i < loops; ++i)
    cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &i_alpha, d_A, CUDA_R_8I, k, d_B, CUDA_R_8I, k, &i_beta, d_C, CUDA_R_32I, m, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  cudaDeviceSynchronize();
  double lapse = omp_get_wtime() - start;

  printf("<gemmEx with i regular call> time: %f ms. GFLOPS: %f\n", lapse * 1000 / loops, gflops / lapse);

  start = omp_get_wtime();
  i_alpha = 1, i_beta = 0;
  for (int32_t i = 0; i < loops; ++i)
    cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, iter_k, &i_alpha, d_A, CUDA_R_8I, k, iter_k, d_B, CUDA_R_8I, k, iter_k, &i_beta, d_C, CUDA_R_32I, m, m*n, 4, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  cudaDeviceSynchronize();
  lapse = omp_get_wtime() - start;

  printf("<gemmEx with i split-k> time: %f ms. GFLOPS: %f\n", lapse * 1000 / loops, gflops / lapse);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  cudaStreamDestroy(stream);
  cublasDestroy(handle);
  return 0;
}
