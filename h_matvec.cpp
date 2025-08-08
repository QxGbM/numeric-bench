
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <complex>
#include <cstdint>
#include <omp.h>
#include <hyacinth.hpp>
#include <internal.hpp>

int32_t main(int32_t argc, char* argv[]) {
  auto err = cudaSetDevice(0);
  if (err != cudaSuccess)
  { fprintf(stderr, "%s\n", cudaGetErrorString(err)); return -1; }

  cudaStream_t stream;
  cublasHandle_t handle;
  cudaStreamCreate(&stream);
  cublasCreate(&handle);
  cublasSetStream(handle, stream);

  int32_t M = std::atoi(argv[1]);
  int32_t N = std::atoi(argv[2]);
  complex_double2* d_A, * d_C;
  cudaMalloc(reinterpret_cast<void**>(&d_A), M * N * sizeof(complex_double2));
  cudaMalloc(reinterpret_cast<void**>(&d_C), M * N * sizeof(complex_double2));

  int64_t flops = M * N * 2;
  int32_t loops = 400;
  double gflops = flops * 1.e-9 * loops;

  double start = omp_get_wtime();
  for (int32_t i = 0; i < loops; ++i)
    internal::Cholesky::minus_adjAx_plusB_scale_double_complex(stream, 1., M, N, (const std::complex<double>*)d_A, N, (std::complex<double>*)d_C);
  cudaDeviceSynchronize();
  double lapse = omp_get_wtime() - start;

  printf("<zgemv> time: %f ms. GFLOPS: %f\n", lapse * 1000 / loops, gflops / lapse);

  start = omp_get_wtime();
  for (int32_t i = 0; i < loops; ++i)
    internal::Cholesky::minus_adjAx_plusB_scale_float_complex(stream, 1.f, M, N, (const std::complex<float>*)d_A, N, (std::complex<float>*)d_C);
  cudaDeviceSynchronize();
  lapse = omp_get_wtime() - start;

  printf("<cgemv> time: %f ms. GFLOPS: %f\n", lapse * 1000 / loops, gflops / lapse);

  start = omp_get_wtime();
  for (int32_t i = 0; i < loops; ++i)
    internal::Cholesky::minus_adjAx_plusB_scale_double2_complex(stream, make_double2(1., 0.), M, N, d_A, N, d_C);
  cudaDeviceSynchronize();
  lapse = omp_get_wtime() - start;

  printf("<dd_gemv> time: %f ms. GFLOPS: %f\n", lapse * 1000 / loops, gflops / lapse);

  start = omp_get_wtime();
  for (int32_t i = 0; i < loops; ++i)
    internal::Cholesky::minus_adjAx_plusB_scale_float4_complex(stream, make_float4(1.f, 0.f, 0.f, 0.f), M, N, (const complex_float4*)d_A, N, (complex_float4*)d_C);
  cudaDeviceSynchronize();
  lapse = omp_get_wtime() - start;

  printf("<f4_gemv> time: %f ms. GFLOPS: %f\n", lapse * 1000 / loops, gflops / lapse);

  std::complex<double> scale(1., 0.), scale2(1., 0.);
  start = omp_get_wtime();
  for (int32_t i = 0; i < loops; ++i)
    cublasZgemv(handle, CUBLAS_OP_C, N, M, (const cuDoubleComplex*)&scale, (const cuDoubleComplex*)d_A, N, (const cuDoubleComplex*)d_A, 1, (const cuDoubleComplex*)&scale2, (cuDoubleComplex*)d_C, 1);
  cudaDeviceSynchronize();
  lapse = omp_get_wtime() - start;

  printf("<zgemv_cublas> time: %f ms. GFLOPS: %f\n", lapse * 1000 / loops, gflops / lapse);

  std::complex<float> scalef(1.f, 0.), scale2f(1.f, 0.);
  start = omp_get_wtime();
  for (int32_t i = 0; i < loops; ++i)
    cublasCgemv(handle, CUBLAS_OP_C, N, M, (const cuComplex*)&scalef, (const cuComplex*)d_A, N, (const cuComplex*)d_A, 1, (const cuComplex*)&scale2f, (cuComplex*)d_C, 1);
  cudaDeviceSynchronize();
  lapse = omp_get_wtime() - start;

  printf("<cgemv_cublas> time: %f ms. GFLOPS: %f\n", lapse * 1000 / loops, gflops / lapse);

  cudaFree(d_A);
  cudaFree(d_C);

  cudaStreamDestroy(stream);
  cublasDestroy(handle);
  fprintf(stderr, "%s\n", cudaGetErrorString(cudaGetLastError()));
  return 0;
}
