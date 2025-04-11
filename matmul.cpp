
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>
#include <omp.h>
#include <magma_v2.h>

int main() {
  magma_init();
  const long long m = 512, n = m, k = m, b = 50;
  magma_queue_t queue = NULL;
  magma_queue_create(0, &queue);

  double** d_A, ** d_B, ** d_C;
  cudaMallocManaged((void**)&d_A, sizeof(double*) * b);
  cudaMallocManaged((void**)&d_B, sizeof(double*) * b);
  cudaMallocManaged((void**)&d_C, sizeof(double*) * b);

  int* rm, *rn, *rk;
  cudaMallocManaged((void**)&rm, sizeof(double*) * b);
  cudaMallocManaged((void**)&rn, sizeof(double*) * b);
  cudaMallocManaged((void**)&rk, sizeof(double*) * b);

  int64_t flops = 0;
  for (int i = 0; i < b; i++) {
    cudaMallocManaged((void**)&d_A[i], sizeof(double) * m * k);
    cudaMallocManaged((void**)&d_B[i], sizeof(double) * n * k);
    cudaMallocManaged((void**)&d_C[i], sizeof(double) * m * n);
    
    rm[i] = m;
    rn[i] = n;
    rk[i] = k;
    flops += 2ll * rm[i] * rn[i] * rk[i];
  }

  int loops = 300;
  double gflops = flops * 1.e-9 * loops;
  double alpha = 1., beta = 0.;

  cublasHandle_t handle = magma_queue_get_cublas_handle(queue);
  magma_queue_sync(queue);

  double start = omp_get_wtime();
  for (int i = 0; i < loops; ++i)
    cublasDgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A, m, d_B, k, &beta, d_C, m, b);
  magma_queue_sync(queue);
  double end = omp_get_wtime();

  printf("time: %f ms. GFLOPS: %f\n", (end - start) * 1000, gflops / (end - start));

  start = omp_get_wtime();
  for (int i = 0; i < loops; ++i)
    magmablas_dgemm_batched(MagmaNoTrans, MagmaNoTrans, m, n, k, alpha, d_A, m, d_B, k, beta, d_C, m, b, queue);
  magma_queue_sync(queue);
  end = omp_get_wtime();

  printf("time: %f ms. GFLOPS: %f\n", (end - start) * 1000, gflops / (end - start));

  start = omp_get_wtime();
  for (int i = 0; i < loops; ++i)
    magmablas_dgemm_vbatched(MagmaNoTrans, MagmaNoTrans, rm, rn, rk, alpha, d_A, rm, d_B, rk, beta, d_C, rm, b, queue);
  magma_queue_sync(queue);
  end = omp_get_wtime();

  printf("time: %f ms. GFLOPS: %f\n", (end - start) * 1000, gflops / (end - start));

  for (int i = 0; i < b; i++) {
    cudaFree(d_A[i]);
    cudaFree(d_B[i]);
    cudaFree(d_C[i]);
  }

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  magma_queue_destroy(queue);
  magma_finalize();
  return 0;
}
