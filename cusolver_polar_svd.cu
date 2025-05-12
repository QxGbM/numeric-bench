
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

  int32_t M = 4000, N = std::min(M, 4000);
  int64_t flops = (int64_t)N * (int64_t)N * (4 * (int64_t)M + 8 * (int64_t)N);

  std::cout << flops << std::endl;

  std::complex<float>* fmat_dev = nullptr, *fumat_dev = nullptr, *fvmat_dev = nullptr;
  float* fsvec_dev = nullptr;
  std::complex<double>* dmat_dev = nullptr, *dumat_dev = nullptr, *dvmat_dev = nullptr;
  double* dsvec_dev = nullptr;

  cudaMallocManaged(reinterpret_cast<void**>(&fmat_dev), M * N * sizeof(std::complex<float>), cudaMemAttachGlobal);
  cudaMallocManaged(reinterpret_cast<void**>(&fumat_dev), M * N * sizeof(std::complex<float>), cudaMemAttachGlobal);
  cudaMallocManaged(reinterpret_cast<void**>(&fvmat_dev), N * N * sizeof(std::complex<float>), cudaMemAttachGlobal);
  cudaMallocManaged(reinterpret_cast<void**>(&fsvec_dev), N * sizeof(float), cudaMemAttachGlobal);
  
  cudaMallocManaged(reinterpret_cast<void**>(&dmat_dev), M * N * sizeof(std::complex<double>), cudaMemAttachGlobal);
  cudaMallocManaged(reinterpret_cast<void**>(&dumat_dev), M * N * sizeof(std::complex<double>), cudaMemAttachGlobal);
  cudaMallocManaged(reinterpret_cast<void**>(&dvmat_dev), N * N * sizeof(std::complex<double>), cudaMemAttachGlobal);
  cudaMallocManaged(reinterpret_cast<void**>(&dsvec_dev), N * sizeof(double), cudaMemAttachGlobal);

  cusolverDnParams_t params;
  size_t fwork_dev_size = 0, fwork_host_size = 0;
  size_t dwork_dev_size = 0, dwork_host_size = 0;
  int32_t* info;

  cusolverDnCreateParams(&params);
  cusolverDnXgesvdp_bufferSize(cusolverH, params, CUSOLVER_EIG_MODE_VECTOR, 1, M, N, CUDA_C_32F, fmat_dev, M, CUDA_R_32F, fsvec_dev, CUDA_C_32F, fumat_dev, M, CUDA_C_32F, fvmat_dev, N, CUDA_C_32F, &fwork_dev_size, &fwork_host_size);
  cusolverDnXgesvdp_bufferSize(cusolverH, params, CUSOLVER_EIG_MODE_VECTOR, 1, M, N, CUDA_C_64F, dmat_dev, M, CUDA_R_64F, dsvec_dev, CUDA_C_64F, dumat_dev, M, CUDA_C_64F, dvmat_dev, N, CUDA_C_64F, &dwork_dev_size, &dwork_host_size);

  std::complex<float>* fwork_dev = nullptr, *fwork_host = nullptr;
  std::complex<double>* dwork_dev = nullptr, *dwork_host = nullptr;

  cudaMalloc(reinterpret_cast<void**>(&info), sizeof(int32_t));
  cudaMalloc(reinterpret_cast<void**>(&fwork_dev), fwork_dev_size);
  if (fwork_host_size)
    fwork_host = (std::complex<float>*)malloc(fwork_host_size);
  
  cudaMalloc(reinterpret_cast<void**>(&dwork_dev), dwork_dev_size);
  if (dwork_host_size)
    dwork_host = (std::complex<double>*)malloc(dwork_host_size);

  std::mt19937_64 gen;
  std::normal_distribution<float> dist(0.f, 1.f);
  std::generate((float*)fmat_dev, (float*)&fmat_dev[M * N], [&]() { return dist(gen); });
  std::generate((double*)dmat_dev, (double*)&dmat_dev[M * N], [&]() { return dist(gen); });

  int32_t loops = 5;
  double h_err = 0.;
  double start, lapse, gf = 2.e-9 * flops * loops;
  cudaDeviceSynchronize();
  start = omp_get_wtime();

  for (int32_t i = 0; i < loops; ++i) {
    cusolverDnXgesvdp(cusolverH, params, CUSOLVER_EIG_MODE_VECTOR, 1, M, N, CUDA_C_64F, dmat_dev, M, CUDA_R_64F, dsvec_dev, CUDA_C_64F, dumat_dev, M, CUDA_C_64F, dvmat_dev, N, CUDA_C_64F, dwork_dev, dwork_dev_size, dwork_host, dwork_host_size, info, &h_err);
  }

  cudaDeviceSynchronize();
  lapse = omp_get_wtime() - start;
  printf("<zgesvd> time: %f ms. Gflops: %f, H_ERR: %e\n", lapse * 1000, gf / lapse, h_err);

  start = omp_get_wtime();

  for (int32_t i = 0; i < loops; ++i) {
    cusolverDnXgesvdp(cusolverH, params, CUSOLVER_EIG_MODE_VECTOR, 1, M, N, CUDA_C_32F, fmat_dev, M, CUDA_R_32F, fsvec_dev, CUDA_C_32F, fumat_dev, M, CUDA_C_32F, fvmat_dev, N, CUDA_C_32F, fwork_dev, fwork_dev_size, fwork_host, fwork_host_size, info, &h_err);
  }

  cudaDeviceSynchronize();
  lapse = omp_get_wtime() - start;

  printf("<cgesvd> time: %f ms. Gflops: %f, H_ERR: %e\n", lapse * 1000, gf / lapse, h_err);

  cudaFree(fmat_dev);
  cudaFree(fumat_dev);
  cudaFree(fsvec_dev);
  cudaFree(fvmat_dev);
  cudaFree(dmat_dev);
  cudaFree(dumat_dev);
  cudaFree(dsvec_dev);
  cudaFree(dvmat_dev);

  cudaFree(fwork_dev);
  if (fwork_host)
    free(fwork_host);

  cudaFree(dwork_dev);
  if (dwork_host)
    free(dwork_host);

  cudaStreamDestroy(stream);
  cublasDestroy(cublasH);
  cusolverDnDestroy(cusolverH);
  return 0;
}
