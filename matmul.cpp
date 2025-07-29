
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

  const int64_t m = 4096, n = m, k = 1024;

  int8_t* d_A, * d_B;
  cuComplex* d_C;
  cudaMallocManaged(reinterpret_cast<void**>(&d_A), 2 * m * k * sizeof(int8_t), cudaMemAttachGlobal);
  cudaMallocManaged(reinterpret_cast<void**>(&d_B), 2 * k * n * sizeof(int8_t), cudaMemAttachGlobal);
  cudaMallocManaged(reinterpret_cast<void**>(&d_C), m * n * sizeof(cuComplex), cudaMemAttachGlobal);

  int64_t flops = m * n * k * 4;
  int32_t loops = 10;
  double gflops = flops * 1.e-9 * loops;
  std::complex<float> alpha = 1.f, beta = 0.f;

  double start = omp_get_wtime();
  for (int32_t i = 0; i < loops; ++i)
    cublasCgemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_C, m, n, k, (cuComplex*)&alpha, d_A, CUDA_C_8I, m, d_B, CUDA_C_8I, n, (cuComplex*)&beta, d_C, CUDA_C_32F, m);
  cudaDeviceSynchronize();
  double lapse = omp_get_wtime() - start;

  printf("<cgemm> time: %f ms. GFLOPS: %f\n", lapse * 1000 / loops, gflops / lapse);

  start = omp_get_wtime();
  for (int32_t i = 0; i < loops; ++i)
    cublasSgemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, (float*)&alpha, d_A, CUDA_R_8I, m, d_B, CUDA_R_8I, n, (float*)&beta, d_C, CUDA_R_32F, m);
  cudaDeviceSynchronize();
  lapse = omp_get_wtime() - start;

  printf("<sgemm> time: %f ms. GFLOPS: %f\n", lapse * 1000 / loops, gflops / lapse);

  start = omp_get_wtime();
  for (int32_t i = 0; i < loops; ++i)
    cublasGemmEx(handle, CUBLAS_OP_C, CUBLAS_OP_N, m, n, k, &alpha, d_A, CUDA_C_8I, k, d_B, CUDA_C_8I, k, &beta, d_C, CUDA_C_32F, m, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  cudaDeviceSynchronize();
  lapse = omp_get_wtime() - start;

  printf("<gemmEx with c> time: %f ms. GFLOPS: %f\n", lapse * 1000 / loops, gflops / lapse);

  start = omp_get_wtime();
  for (int32_t i = 0; i < loops; ++i)
    cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha, d_A, CUDA_R_8I, k, d_B, CUDA_R_8I, k, &beta, d_C, CUDA_R_32F, m, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  cudaDeviceSynchronize();
  lapse = omp_get_wtime() - start;

  printf("<gemmEx with s> time: %f ms. GFLOPS: %f\n", lapse * 1000 / loops, gflops / lapse);

  start = omp_get_wtime();
  int32_t i_alpha = 1, i_beta = 1;
  for (int32_t i = 0; i < loops; ++i)
    cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &i_alpha, d_A, CUDA_R_8I, k, d_B, CUDA_R_8I, k, &i_beta, d_C, CUDA_R_32I, m, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
  cudaDeviceSynchronize();
  lapse = omp_get_wtime() - start;

  printf("<gemmEx with i unit> time: %f ms. GFLOPS: %f\n", lapse * 1000 / loops, gflops / lapse);

  start = omp_get_wtime();
  i_alpha = -1, i_beta = 1;
  for (int32_t i = 0; i < loops; ++i)
    cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &i_alpha, d_A, CUDA_R_8I, k, 0, d_B, CUDA_R_8I, k, 0, &i_beta, d_C, CUDA_R_32I, m, 0, 1, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
  cudaDeviceSynchronize();
  lapse = omp_get_wtime() - start;

  printf("<gemmEx with i non-unit> time: %f ms. GFLOPS: %f\n", lapse * 1000 / loops, gflops / lapse);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  cudaStreamDestroy(stream);
  cublasDestroy(handle);
  return 0;
}
