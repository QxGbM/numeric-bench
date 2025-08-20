
#include <vector>
#include <complex>
#include <iostream>
#include <algorithm>
#include <random>
#include <magma_v2.h>
#include <cublas_v2.h>

int32_t main(int32_t argc, char* argv[]) {
  magma_init();
  int64_t M = 1 < argc ? std::atoi(argv[1]) : 1024;
  int64_t N = 2 < argc ? std::atoi(argv[2]) : 128;
  std::cout << "magma SGEQP3 <" << M << ", " << N << ">\n";

  magma_queue_t queue = nullptr;
  magma_queue_create(0, &queue);
  cublasHandle_t cublasH = magma_queue_get_cublas_handle(queue);
  cudaStream_t stream = magma_queue_get_cuda_stream(queue);

  float* dA, *dC;
  std::vector<float> tau(N);
  std::vector<magma_int_t> jpvt(N, 0);

  std::mt19937_64 gen(42);
  std::normal_distribution<float> dist(0., 32.);
  std::vector<float> matA(M * N);
  std::generate(matA.begin(), matA.end(), [&]() { return dist(gen); });
  
  magma_smalloc(&dA, M * N);
  magma_ssetvector(M * N, (float*)matA.data(), 1, (float*)dA, 1, queue);

  magma_int_t Lwork, info;
  float work;
  magma_sgeqp3(M, N, nullptr, M, jpvt.data(), tau.data(), &work, -1, &info);
  Lwork = 16 * (magma_int_t)(work);
  magma_smalloc(&dC, Lwork);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, stream);
  magma_sgeqp3_gpu(M, N, dA, M, jpvt.data(), tau.data(), dC, Lwork, &info);
  cudaEventRecord(stop, stream);
  magma_queue_sync(queue);

  float milliseconds = 0.0f;
  int64_t qr_flops = (M * N * N * 2) - (N * N * N * 2 / 3);
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "Time: " << milliseconds << " ms\n";
  std::cout << "Total GFLOPs: " << double(qr_flops) * 1.e-6 / milliseconds << "\n";

  float nrm = 0.;
  cublasSnrm2_64(cublasH, M * N, dA, int64_t(1), &nrm);
  std::cout << "L2 Nrm: " << nrm << "\n" << std::endl;

  magma_free(dA);
  magma_free(dC);

  magma_queue_destroy(queue);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  magma_finalize();
  return 0;
}

