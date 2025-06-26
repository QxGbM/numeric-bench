
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <complex>
#include <cstdint>
#include <omp.h>
#include <hyacinth.hpp>

int32_t main() {
  auto err = cudaSetDevice(0);
  if (err != cudaSuccess)
  { fprintf(stderr, "%s\n", cudaGetErrorString(err)); return -1; }

  cudaStream_t stream;
  cublasHandle_t handle;
  cudaStreamCreate(&stream);
  cublasCreate(&handle);
  cublasSetStream(handle, stream);

  const int32_t m = 8192, n = m;

  cuDoubleComplex* d_A, * d_B, * d_C, * d_D;
  double* s;
  cudaMallocManaged(reinterpret_cast<void**>(&d_A), m * n * sizeof(cuDoubleComplex), cudaMemAttachGlobal);
  cudaMallocManaged(reinterpret_cast<void**>(&d_B), n * sizeof(cuDoubleComplex), cudaMemAttachGlobal);
  cudaMallocManaged(reinterpret_cast<void**>(&d_C), m * sizeof(cuDoubleComplex), cudaMemAttachGlobal);
  cudaMallocManaged(reinterpret_cast<void**>(&d_D), m * sizeof(cuDoubleComplex), cudaMemAttachGlobal);
  cudaMallocManaged(reinterpret_cast<void**>(&s), sizeof(double), cudaMemAttachGlobal);

  for (int32_t i = 0; i < m; ++i)
    for (int32_t j = 0; j < n; ++j)
      d_A[j + i * m] = make_cuDoubleComplex(i, j);
  
  for (int32_t i = 0; i < n; ++i)
    d_B[i] = make_cuDoubleComplex(i, -i);

  for (int32_t i = 0; i < m; ++i)
    d_C[i] = make_cuDoubleComplex(0., 0.);

  *s = 1.;

  int64_t flops = m * n * 4;
  int32_t loops = 10;
  double gflops = flops * 1.e-9 * loops;
  std::complex<double> alpha = 1., beta = 1.;
  double scale = -1.;

  double start = omp_get_wtime();
  for (int32_t i = 0; i < loops; ++i) {
    //cublasZgemv(handle, CUBLAS_OP_C, m, n, (cuDoubleComplex*)&alpha, d_A, n, d_B, 1, (cuDoubleComplex*)&beta, d_C, 1);
    cublasDgemv(handle, CUBLAS_OP_T, m, n, (double*)&alpha, (const double*)d_A, n, (const double*)d_B, 1, (double*)&beta, (double*)d_C, 1);
    cublasDscal(handle, m, &scale, (double*)d_C, 1);
  }
  cudaDeviceSynchronize();
  double lapse = omp_get_wtime() - start;

  printf("<zgemv> time: %f ms. GFLOPS: %f\n", lapse * 1000 / loops, gflops / lapse);

  start = omp_get_wtime();
  for (int32_t i = 0; i < loops; ++i)
    //minus_adjAx_plusB_scale_double_complex(stream, 1., m, n, (const std::complex<double>*)d_A, n, (const std::complex<double>*)d_B, (std::complex<double>*)d_C, (std::complex<double>*)d_D);
    minus_transAx_plusB_scale_double(stream, s, m, n, (const double*)d_A, n, (const double*)d_B, (double*)d_C, (double*)d_D);
  cudaDeviceSynchronize();
  lapse = omp_get_wtime() - start;

  printf("<zgemv custom> time: %f ms. GFLOPS: %f\n", lapse * 1000 / loops, gflops / lapse);

  double nrm = 0.;
  cublasDznrm2(handle, m, d_C, 1, &nrm);
  cudaDeviceSynchronize();

  printf("err: %e\n", nrm / n);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFree(d_D);
  cudaFree(s);

  cudaStreamDestroy(stream);
  cublasDestroy(handle);
  return 0;
}
