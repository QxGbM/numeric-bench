
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <vector>
#include <random>
#include <algorithm>
#include <magma_v2.h>
#include <lapacke.h>

int main() {
  magma_init();
  magma_queue_t queue = NULL;
  magma_queue_create(0, &queue);

  int64_t len = 1000000;
  int loops = 100;
  std::vector<double> hostVec(len);

  std::mt19937_64 gen;
  std::uniform_real_distribution<double> uniform_dist(0., 1.);
  std::generate(hostVec.begin(), hostVec.end(), [&]() { return uniform_dist(gen); });

  double* hostVecPin = nullptr, *devVecDouble = nullptr, start, lapse, Gb = 1.e-9 * len * sizeof(double) * loops;
  float* devVecFloat = nullptr, nrm;
  int32_t info;

  magma_dmalloc_pinned(&hostVecPin, len);
  magma_dmalloc(&devVecDouble, len);
  magma_smalloc(&devVecFloat, len);

  std::copy(hostVec.begin(), hostVec.end(), hostVecPin);
  magma_queue_sync(queue);
  start = omp_get_wtime();

  for (int i = 0; i < loops; ++i) {
    magma_dsetvector_async(len, hostVecPin, 1, devVecDouble, 1, queue);
    magmablas_dlag2s(len, 1, hostVecPin, len, devVecFloat, len, queue, &info);
  }

  magma_queue_sync(queue);
  lapse = omp_get_wtime() - start;
  nrm = magma_snrm2(len, devVecFloat, 1, queue);

  printf("<Copy FP64 and GPU conversion> time: %f ms. Gbps: %f, Norm is %f\n", lapse * 1000, Gb / lapse, nrm);

  magma_queue_sync(queue);
  start = omp_get_wtime();

  for (int i = 0; i < loops; ++i)
    magmablas_dlag2s(len, 1, hostVecPin, len, devVecFloat, len, queue, &info);

  magma_queue_sync(queue);
  lapse = omp_get_wtime() - start;
  nrm = magma_snrm2(len, devVecFloat, 1, queue);

  printf("<Direct Access and GPU conversion> time: %f ms. Gbps: %f, Norm is %f\n", lapse * 1000, Gb / lapse, nrm);

  magma_queue_sync(queue);
  start = omp_get_wtime();

  for (int i = 0; i < loops; ++i) {
    LAPACKE_dlag2s(LAPACK_COL_MAJOR, len, 1, hostVec.data(), len, (float*)hostVecPin, len);
    magma_ssetvector_async(len, (float*)hostVecPin, 1, devVecFloat, 1, queue);
  }

  magma_queue_sync(queue);
  lapse = omp_get_wtime() - start;
  nrm = magma_snrm2(len, devVecFloat, 1, queue);

  printf("<CPU conversion and copy FP32> time: %f ms. Gbps: %f, Norm is %f\n", lapse * 1000, Gb / lapse, nrm);

  magma_free_pinned(hostVecPin);
  magma_free(devVecDouble);
  magma_free(devVecFloat);

  magma_queue_destroy(queue);
  magma_finalize();
  return 0;
}
