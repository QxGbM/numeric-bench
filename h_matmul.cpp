
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <complex>
#include <cstdint>
#include <random>
#include <omp.h>
#include <cuComplex.h>
#include <hyacinth.hpp>
#include <internal.hpp>

int32_t main() {
  auto err = cudaSetDevice(0);
  if (err != cudaSuccess)
  { fprintf(stderr, "%s\n", cudaGetErrorString(err)); return -1; }

  cudaStream_t stream;
  cublasHandle_t handle;
  cudaStreamCreate(&stream);
  cublasCreate(&handle);
  cublasSetStream(handle, stream);

  const int32_t m = 8000, k = 64;

  std::mt19937_64 gen;
  std::normal_distribution<double> dist(0., 1.);

  std::complex<double>* d_A, * d_C;
  cudaMallocManaged(reinterpret_cast<void**>(&d_A), m * m * sizeof(std::complex<double>), cudaMemAttachGlobal);
  cudaMallocManaged(reinterpret_cast<void**>(&d_C), m * m * sizeof(std::complex<double>), cudaMemAttachGlobal);
  cudaMemset(d_C, 0, m * m * sizeof(std::complex<double>));

  for (int32_t i = 0; i < m * m; ++i)
    d_A[i] = std::complex<double>(dist(gen), dist(gen));

  int64_t flops = int64_t(m) * int64_t(m) * int64_t(k) * 4;
  int32_t loops = 10;
  double gflops = flops * 1.e-9 * loops;
  std::complex<double> one(1., 0.);

  double start = omp_get_wtime();
  for (int32_t i = 0; i < loops; ++i)
    cublasZgemm(handle, CUBLAS_OP_C, CUBLAS_OP_N, m, m, k, (const cuDoubleComplex*)&one, (const cuDoubleComplex*)d_A, m, (const cuDoubleComplex*)d_A, m, (const cuDoubleComplex*)&one, (cuDoubleComplex*)d_C, m);
  cudaDeviceSynchronize();
  double lapse = omp_get_wtime() - start;

  printf("<cublas zgemm> time: %f ms. GFLOPS: %f\n", lapse * 1000 / loops, gflops / lapse);

  start = omp_get_wtime();
  for (int32_t i = 0; i < loops; ++i)
    internal::Cholesky::minus_AHA_gemmk64_double_complex(stream, m, d_A, d_C, m);
  cudaDeviceSynchronize();
  lapse = omp_get_wtime() - start;

  printf("<zgemmk> time: %f ms. GFLOPS: %f\n", lapse * 1000 / loops, gflops / lapse);
  fprintf(stderr, "%s\n", cudaGetErrorString(cudaGetLastError()));

  double nrm;
  cublasDznrm2(handle, m * m, (const cuDoubleComplex*)d_C, 1, &nrm);
  printf("%le\n", nrm / m / m);

  cudaFree(d_A);
  cudaFree(d_C);

  cudaStreamDestroy(stream);
  cublasDestroy(handle);
  return 0;
}
