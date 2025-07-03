
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <complex>
#include <cstdint>
#include <omp.h>
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

  const int32_t m = 8192, n = m;

  complex_double2* d_A, * d_C, *s;
  cudaMallocManaged(reinterpret_cast<void**>(&d_A), m * n * sizeof(complex_double2), cudaMemAttachGlobal);
  cudaMallocManaged(reinterpret_cast<void**>(&d_C), m * m * sizeof(complex_double2), cudaMemAttachGlobal);
  cudaMallocManaged(reinterpret_cast<void**>(&s), sizeof(complex_double2), cudaMemAttachGlobal);

  int64_t flops = m * n * 4;
  int32_t loops = 10;
  double gflops = flops * 1.e-9 * loops;

  double start = omp_get_wtime();
  for (int32_t i = 0; i < loops; ++i)
    internal::Cholesky::minus_adjAx_plusB_scale_double_complex(stream, (const double*)s, m, n, (const std::complex<double>*)d_A, n, (std::complex<double>*)d_C);
  cudaDeviceSynchronize();
  double lapse = omp_get_wtime() - start;

  printf("<zgemv> time: %f ms. GFLOPS: %f\n", lapse * 1000 / loops, gflops / lapse);

  start = omp_get_wtime();
  for (int32_t i = 0; i < loops; ++i)
    internal::Cholesky::minus_adjAx_plusB_scale_float_complex(stream, (const float*)s, m, n, (const std::complex<float>*)d_A, n, (std::complex<float>*)d_C);
  cudaDeviceSynchronize();
  lapse = omp_get_wtime() - start;

  printf("<cgemv> time: %f ms. GFLOPS: %f\n", lapse * 1000 / loops, gflops / lapse);

  start = omp_get_wtime();
  for (int32_t i = 0; i < loops; ++i)
    internal::Cholesky::minus_adjAx_plusB_scale_double2_complex(stream, (const double2*)s, m, n, d_A, n, d_C);
  cudaDeviceSynchronize();
  lapse = omp_get_wtime() - start;

  printf("<dd_gemv> time: %f ms. GFLOPS: %f\n", lapse * 1000 / loops, gflops / lapse);

  start = omp_get_wtime();
  for (int32_t i = 0; i < loops; ++i)
    internal::Cholesky::minus_adjAx_plusB_scale_float4_complex(stream, (const float4*)s, m, n, (const complex_float4*)d_A, n, (complex_float4*)d_C);
  cudaDeviceSynchronize();
  lapse = omp_get_wtime() - start;

  printf("<f4_gemv> time: %f ms. GFLOPS: %f\n", lapse * 1000 / loops, gflops / lapse);

  cudaFree(d_A);
  cudaFree(d_C);

  cudaStreamDestroy(stream);
  cublasDestroy(handle);
  return 0;
}
