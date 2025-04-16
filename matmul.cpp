
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <omp.h>
#include <magma_v2.h>

int32_t main() {
  magma_init();
  const int64_t m = 1024, n = m, k = m;
  magma_queue_t queue = nullptr;
  magma_queue_create(0, &queue);

  double* d_A, * d_B, * d_C;
  magma_dmalloc(&d_A, m * k);
  magma_dmalloc(&d_B, k * n);
  magma_dmalloc(&d_C, m * n);

  int64_t flops = m * n * k * 2;
  int32_t loops = 300;
  double gflops = flops * 1.e-9 * loops;
  double alpha = 1., beta = 0.;

  double start = omp_get_wtime();
  for (int32_t i = 0; i < loops; ++i)
    magma_dgemm(MagmaNoTrans, MagmaNoTrans, m, n, k, alpha, d_A, m, d_B, k, beta, d_C, m, queue);
  magma_queue_sync(queue);
  double end = omp_get_wtime();

  printf("time: %f ms. GFLOPS: %f\n", (end - start) * 1000, gflops / (end - start));

  start = omp_get_wtime();
  for (int32_t i = 0; i < loops; ++i)
    magmablas_dgemm(MagmaNoTrans, MagmaNoTrans, m, n, k, alpha, d_A, m, d_B, k, beta, d_C, m, queue);
  magma_queue_sync(queue);
  end = omp_get_wtime();

  printf("time: %f ms. GFLOPS: %f\n", (end - start) * 1000, gflops / (end - start));

  magma_free(d_A);
  magma_free(d_B);
  magma_free(d_C);

  magma_queue_destroy(queue);
  magma_finalize();
  return 0;
}
