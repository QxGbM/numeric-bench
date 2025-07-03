
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <omp.h>

#include <vector>
#include <complex>
#include <iostream>
#include <algorithm>
#include <random>
#include <magma_v2.h>
#include <commons.hpp>

int32_t main() {
  magma_init();
  const int64_t m = 2048, n = m;
  magma_queue_t queue = nullptr;
  magma_queue_create(0, &queue);

  int32_t Lwork = -1, info;
  magmaDoubleComplex* d_A, * d_C;
  double* d_R;
  std::vector<magmaDoubleComplex> tau(n);
  std::vector<magma_int_t> jpvt(n, 0);
  magmaDoubleComplex work;

  Eigen::MatrixXcd matA(m, n);
  random_vector(m * n * 2, (double*)matA.data());

  magma_zmalloc(&d_A, m * n);
  magma_dmalloc(&d_R, 2 * n);
  magma_dsetvector(m * n * 2, (double*)matA.data(), 1, (double*)d_A, 1, queue);

  magma_zgeqp3(m, n, nullptr, m, nullptr, nullptr, &work, Lwork, nullptr, &info);
  Lwork = (int32_t)(work.x) * (n + 1);
  printf("%d %d\n", info, Lwork);
  magma_zmalloc(&d_C, Lwork);

  int64_t flops = m * n * 2;
  int32_t loops = 1;
  double gflops = flops * 1.e-9 * loops;

  double start = omp_get_wtime();
  for (int32_t i = 0; i < loops; ++i) {
    magma_zgeqp3_gpu(m, n, d_A, m, jpvt.data(), tau.data(), d_C, Lwork, d_R, &info);
  }
  magma_queue_sync(queue);
  double end = omp_get_wtime();

  printf("time: %lf ms. GFLOPS: %lf\n", (end - start) * 1000, gflops / (end - start));

  magma_free(d_A);
  magma_free(d_C);
  magma_free(d_R);

  magma_queue_destroy(queue);
  magma_finalize();
  return 0;
}

