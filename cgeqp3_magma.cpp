
#include <vector>
#include <complex>
#include <iostream>
#include <algorithm>
#include <random>
#include <magma_v2.h>
#include <cublas_v2.h>

int32_t main(int32_t argc, char* argv[]) {
  auto cu_err = cudaSetDevice(0);
  if (cu_err != cudaSuccess)
  { fprintf(stderr, "%s\n", cudaGetErrorString(cu_err)); return -1; }

  magma_init();
  int64_t M = 1 < argc ? std::atoi(argv[1]) : 1024;
  int64_t N = 2 < argc ? std::atoi(argv[2]) : 128;

  magma_queue_t queue = nullptr;
  magma_queue_create(0, &queue);
  cublasHandle_t cublasH = magma_queue_get_cublas_handle(queue);
  cudaStream_t stream = magma_queue_get_cuda_stream(queue);

  magmaFloatComplex* dA, *dC;
  float* dR;
  std::vector<magmaFloatComplex> tau(N);
  std::vector<magma_int_t> jpvt(N, 0);

  std::mt19937_64 gen(42);
  std::normal_distribution<float> dist(0., 32.);
  std::vector<float> matA(M * N * 2);
  std::generate(matA.begin(), matA.end(), [&]() { return dist(gen); });
  
  magma_cmalloc(&dA, M * N);
  magma_smalloc(&dR, 2 * N);
  magma_ssetvector(M * N * 2, (float*)matA.data(), 1, (float*)dA, 1, queue);

  magma_int_t Lwork, info;
  magmaFloatComplex work;
  magma_cgeqp3(M, N, nullptr, M, jpvt.data(), tau.data(), &work, -1, dR, &info);
  Lwork = 16 * (magma_int_t)(work.x);
  magma_cmalloc(&dC, Lwork);

  magma_cgeqp3_gpu(M, N, dA, M, jpvt.data(), tau.data(), dC, Lwork, dR, &info);
  magma_queue_sync(queue);
  magma_ssetvector(M * N * 2, (float*)matA.data(), 1, (float*)dA, 1, queue);
  std::fill(jpvt.begin(), jpvt.end(), 0);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, stream);
  magma_cgeqp3_gpu(M, N, dA, M, jpvt.data(), tau.data(), dC, Lwork, dR, &info);
  cudaEventRecord(stop, stream);
  magma_queue_sync(queue);

  float nrm = 0.;
  cublasScnrm2_64(cublasH, M * N, (cuComplex*)dA, int64_t(1), &nrm);
  magma_queue_sync(queue);

  float milliseconds = 0.0f;
  cudaEventElapsedTime(&milliseconds, start, stop);
  int64_t qr_flops = (M * N * N * 2) - (N * N * N * 2 / 3);
  double gflops = double(qr_flops) * 1.e-6 / milliseconds;
  std::cout << "magma-CGEQP3," << M << "," << N << "," << nrm << "," << milliseconds << "," << gflops << std::endl;

  magma_free(dA);
  magma_free(dC);
  magma_free(dR);

  magma_queue_destroy(queue);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  magma_finalize();

  cu_err = cudaGetLastError();
  if (cu_err != cudaSuccess)
    std::cerr << cudaGetErrorString(cu_err) << std::endl;
  return 0;
}

