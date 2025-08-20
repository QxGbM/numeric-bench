
#include <vector>
#include <complex>
#include <iostream>
#include <algorithm>
#include <random>
#include <magma_v2.h>

int32_t main(int32_t argc, char* argv[]) {
  magma_init();
  int64_t M = 1 < argc ? std::atoi(argv[1]) : 1024;
  int64_t N = 2 < argc ? std::atoi(argv[2]) : 128;
  std::cout << "magma ZGEQP3 <" << M << ", " << N << ">\n";

  magma_queue_t queue = nullptr;
  magma_queue_create(0, &queue);
  cudaStream_t stream = magma_queue_get_cuda_stream(queue);

  magmaDoubleComplex* dA, *dC;
  double* dR;
  std::vector<magmaDoubleComplex> tau(N);
  std::vector<magma_int_t> jpvt(N, 0);

  std::mt19937_64 gen(42);
  std::normal_distribution<double> dist(0., 32.);
  std::vector<double> matA(M * N * 2);
  std::generate(matA.begin(), matA.end(), [&]() { return dist(gen); });
  
  magma_zmalloc(&dA, M * N);
  magma_dmalloc(&dR, 2 * N);
  magma_dsetvector(M * N * 2, (double*)matA.data(), 1, (double*)dA, 1, queue);

  magma_int_t Lwork, info;
  magmaDoubleComplex work;
  magma_zgeqp3(M, N, nullptr, M, jpvt.data(), tau.data(), &work, -1, dR, &info);
  Lwork = 16 * (magma_int_t)(work.x);
  magma_zmalloc(&dC, Lwork);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, stream);
  magma_zgeqp3_gpu(M, N, dA, M, jpvt.data(), tau.data(), dC, Lwork, dR, &info);
  cudaEventRecord(stop, stream);
  magma_queue_sync(queue);

  float milliseconds = 0.0f;
  int64_t qr_flops = (M * N * N * 2) - (N * N * N * 2 / 3);
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "Time: " << milliseconds << " ms\n";
  std::cout << "Total GFLOPs: " << double(qr_flops) * 1.e-6 / milliseconds << "\n" << std::endl;
  
  magma_free(dA);
  magma_free(dC);
  magma_free(dR);

  magma_queue_destroy(queue);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  magma_finalize();
  return 0;
}

