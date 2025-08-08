
#include <cublas_v2.h>

#include <vector>
#include <complex>
#include <iostream>
#include <algorithm>
#include <random>
#include <magma_v2.h>
#include <commons.hpp>

int32_t main(int32_t argc, char* argv[]) {
  magma_init();
  int64_t M = std::atoi(argv[1]);
  int64_t N = std::atoi(argv[2]);
  magma_queue_t queue = nullptr;
  magma_queue_create(0, &queue);
  cudaStream_t stream = magma_queue_get_cuda_stream(queue);

  int32_t Lwork = -1, info;
  typedef float matrix_t;
  matrix_t* d_A, * d_C;
  std::vector<matrix_t> tau(N);
  std::vector<magma_int_t> jpvt(N, 0);
  matrix_t work;

  Eigen::MatrixXd matA(M, N);
  std::mt19937_64 gen;
  std::normal_distribution<float> dist(0, 32);

  for (int32_t j = 0; j < N; ++j)
    for (int32_t i = 0; i < M; ++i)
      matA(i, j) = dist(gen);

  magma_smalloc(&d_A, M * N);
  magma_ssetvector(M * N, (float*)matA.data(), 1, (float*)d_A, 1, queue);

  magma_sgeqp3(M, N, nullptr, M, nullptr, nullptr, &work, Lwork, &info);
  Lwork = (int32_t)(work) * (N + 1);
  printf("%d %d\n", info, Lwork);
  magma_smalloc(&d_C, Lwork);

  int64_t flops = (M * N * N * 2) - (N * N * N * 2 / 3);
  float mflops = flops * 1.e-6;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, stream);
  magma_sgeqp3_gpu(M, N, d_A, M, jpvt.data(), tau.data(), d_C, Lwork, &info);
  cudaEventRecord(stop, stream);
  magma_queue_sync(queue);

  float milliseconds = 0.0f;
  cudaEventElapsedTime(&milliseconds, start, stop);

  printf("time: %f ms. GFLOPS: %lf\n", milliseconds, mflops / milliseconds);

  magma_free(d_A);
  magma_free(d_C);

  magma_queue_destroy(queue);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  magma_finalize();
  return 0;
}

