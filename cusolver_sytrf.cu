
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <thrust/transform.h>
#include <omp.h>

#include <vector>
#include <complex>
#include <iostream>
#include <algorithm>
#include <random>

int32_t main() {
  auto err = cudaSetDevice(0);
  if (err != cudaSuccess)
  { fprintf(stderr, "%s\n", cudaGetErrorString(err)); return -1; }
  
  cudaStream_t stream;
  cublasHandle_t cublasH;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  cublasCreate(&cublasH);
  cublasSetStream(cublasH, stream);

  cusolverDnHandle_t cusolverH;
  cusolverDnCreate(&cusolverH);
  cusolverDnSetStream(cusolverH, stream);

  int32_t N = 2048;
  int64_t flops = (int64_t)N * (int64_t)N * (int64_t)N / 3;

  std::cout << flops << std::endl;

  std::complex<float>* fmat_dev = nullptr;
  std::complex<double>* dmat_dev = nullptr;

  cudaMallocManaged(reinterpret_cast<void**>(&fmat_dev), N * N * sizeof(std::complex<float>), cudaMemAttachGlobal);
  cudaMallocManaged(reinterpret_cast<void**>(&dmat_dev), N * N * sizeof(std::complex<double>), cudaMemAttachGlobal);

  int32_t fwork_dev_size = 0;
  int32_t dwork_dev_size = 0;
  int32_t* info, *ipiv;

  cusolverDnCsytrf_bufferSize(cusolverH, N, (cuComplex*)fmat_dev, N, &fwork_dev_size);
  cusolverDnZsytrf_bufferSize(cusolverH, N, (cuDoubleComplex*)dmat_dev, N, &dwork_dev_size);

  std::complex<float>* fwork_dev = nullptr;
  std::complex<double>* dwork_dev = nullptr;

  cudaMalloc(reinterpret_cast<void**>(&info), sizeof(int32_t));
  cudaMalloc(reinterpret_cast<void**>(&ipiv), N * sizeof(int32_t));
  cudaMalloc(reinterpret_cast<void**>(&fwork_dev), fwork_dev_size * sizeof(std::complex<float>));
  cudaMalloc(reinterpret_cast<void**>(&dwork_dev), dwork_dev_size * sizeof(std::complex<double>));

  std::mt19937_64 gen;
  std::normal_distribution<float> dist(0.f, 1.f);
  std::generate((float*)fmat_dev, (float*)&fmat_dev[N * N], [&]() { return dist(gen); });
  std::generate((double*)dmat_dev, (double*)&dmat_dev[N * N], [&]() { return dist(gen); });

  int32_t loops = 5;
  double start, lapse, gf = 2.e-9 * flops * loops;
  cudaDeviceSynchronize();
  start = omp_get_wtime();

  for (int32_t i = 0; i < loops; ++i) {
    cusolverDnCsytrf(cusolverH, CUBLAS_FILL_MODE_UPPER, N, (cuComplex*)fmat_dev, N, ipiv, (cuComplex*)fwork_dev, fwork_dev_size, info);
  }

  cudaDeviceSynchronize();
  lapse = omp_get_wtime() - start;

  printf("<csytrf> time: %f ms. Gflops: %f\n", lapse * 1000 / loops, gf / lapse);
  start = omp_get_wtime();

  for (int32_t i = 0; i < loops; ++i) {
    cusolverDnZsytrf(cusolverH, CUBLAS_FILL_MODE_UPPER, N, (cuDoubleComplex*)dmat_dev, N, ipiv, (cuDoubleComplex*)dwork_dev, dwork_dev_size, info);
  }

  cudaDeviceSynchronize();
  lapse = omp_get_wtime() - start;
  printf("<zsytrf> time: %f ms. Gflops: %f\n", lapse * 1000 / loops, gf / lapse);

  std::generate((float*)fmat_dev, (float*)&fmat_dev[N * N], [&]() { return dist(gen); });
  std::generate((double*)dmat_dev, (double*)&dmat_dev[N * N], [&]() { return dist(gen); });

  start = omp_get_wtime();

  for (int32_t i = 0; i < loops; ++i) {
    cusolverDnCpotrf(cusolverH, CUBLAS_FILL_MODE_UPPER, N, (cuComplex*)fmat_dev, N, (cuComplex*)fwork_dev, fwork_dev_size, info);
  }

  cudaDeviceSynchronize();
  lapse = omp_get_wtime() - start;

  printf("<cpotrf> time: %f ms. Gflops: %f\n", lapse * 1000 / loops, gf / lapse);

  start = omp_get_wtime();

  for (int32_t i = 0; i < loops; ++i) {
    cusolverDnZpotrf(cusolverH, CUBLAS_FILL_MODE_UPPER, N, (cuDoubleComplex*)dmat_dev, N, (cuDoubleComplex*)dwork_dev, dwork_dev_size, info);
  }

  cudaDeviceSynchronize();
  lapse = omp_get_wtime() - start;
  printf("<zpotrf> time: %f ms. Gflops: %f\n", lapse * 1000 / loops, gf / lapse);

  cudaFree(fmat_dev);
  cudaFree(dmat_dev);
  cudaFree(fwork_dev);
  cudaFree(dwork_dev);
  cudaFree(ipiv);
  cudaFree(info);

  cudaStreamDestroy(stream);
  cublasDestroy(cublasH);
  cusolverDnDestroy(cusolverH);
  return 0;
}
