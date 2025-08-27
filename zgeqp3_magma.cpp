
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

  magma_zgeqp3_gpu(M, N, dA, M, jpvt.data(), tau.data(), dC, Lwork, dR, &info);
  magma_queue_sync(queue);
  magma_dsetvector(M * N * 2, (double*)matA.data(), 1, (double*)dA, 1, queue);
  std::fill(jpvt.begin(), jpvt.end(), 0);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, stream);
  magma_zgeqp3_gpu(M, N, dA, M, jpvt.data(), tau.data(), dC, Lwork, dR, &info);
  cudaEventRecord(stop, stream);
  magma_queue_sync(queue);

  double nrm = 0.;
  cublasDznrm2_64(cublasH, M * N, (cuDoubleComplex*)dA, int64_t(1), &nrm);
  magma_queue_sync(queue);

  float milliseconds = 0.0f;
  cudaEventElapsedTime(&milliseconds, start, stop);
  int64_t qr_flops = (M * N * N * 2) - (N * N * N * 2 / 3);
  double gflops = double(qr_flops) * 1.e-6 / milliseconds;
  std::cout << "magma-ZGEQP3," << M << "," << N << "," << nrm << "," << milliseconds << "," << gflops << std::endl;
  
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

