
#include <cublas_v2.h>
#include <omp.h>

#include <vector>
#include <complex>
#include <iostream>
#include <algorithm>
#include <random>

int32_t dp_cheidf_gpu(cublasHandle_t handle, int32_t N, const cuComplex* A, int32_t lda, int32_t* ipiv, cuComplex* X, int32_t ldx, cuComplex* work) {
  int32_t quot = 1 + (((N & 7) - 1) >> 31);
  int32_t ld = ((N >> 3) + quot) << 3;
  if (work == nullptr)
    return ld * (N + 1);

  cudaStream_t stream;
  cublasGetStream(handle, &stream);
  cudaMemcpy2DAsync(work, ld * sizeof(std::complex<float>), A, lda * sizeof(std::complex<float>), N * sizeof(std::complex<float>), N, cudaMemcpyDeviceToDevice, stream);

  int32_t rank = N, iters = 0, sync = 10, *p = nullptr;
  float s0 = 0.f, *s = nullptr;
  cudaMallocManaged((void**)&p, sizeof(int32_t), cudaMemAttachGlobal);
  cudaMallocManaged((void**)&s, sizeof(float), cudaMemAttachGlobal);

  for (int32_t i = 0; i < rank; ++i) {
    cublasIcamax(handle, N, work, ld + 1, p);
    
    std::complex<float> scale(-1.f / work[(*p - 1) * (ld + 1)].x, 0.f);
    cublasCcopy(handle, N, &work[(*p - 1) * ld], 1, &X[i * ldx], 1);
    cublasCgerc(handle, N, N, (cuComplex*)&scale, &X[i * ldx], 1, &X[i * ldx], 1, work, ld);

    if (i == 0)
      s0 = scale.real();
    if (16384.f < (scale.real() / s0))
      rank = i;
  }

  cudaFree(p);
  cudaFree(s);
  return rank;
}

#include <commons.hpp>

int32_t main() {
  auto err = cudaSetDevice(0);
  if (err != cudaSuccess)
  { fprintf(stderr, "%s\n", cudaGetErrorString(err)); return -1; }
  
  cudaStream_t stream;
  cublasHandle_t cublasH;
  cudaStreamCreate(&stream);
  cublasCreate(&cublasH);
  cublasSetStream(cublasH, stream);

  int64_t M = 3000, N = 100;
  Eigen::MatrixXcd matA(M, N);
  random_vector(M * N * 2, (double*)matA.data());

  int64_t flops = (int64_t)N * (int64_t)N * (int64_t)N / 3;
  std::cout << flops << std::endl;

  Eigen::MatrixXcd AAT = matA.adjoint() * matA;
  for (int32_t i = 0; i < 4; ++i) {
    AAT /= AAT.norm();
    AAT = AAT.adjoint() * AAT;
  }

  matA = matA * AAT;
  matA /= matA.norm();
  AAT = matA.adjoint() * matA;

  Eigen::MatrixXcf matAf(N, N);
  for (int32_t i = 0; i < N; ++i)
    for (int32_t j = 0; j < N; ++j)
      matAf(i, j) = std::complex<float>(AAT(i,j).real(), AAT(i, j).imag());

  std::complex<float>* fmat_dev = nullptr, *xmat_dev = nullptr, *work_dev = nullptr;
  int32_t* ipiv = nullptr;
  cudaMalloc((void**)&fmat_dev, N * N * sizeof(std::complex<float>));
  cudaMalloc((void**)&xmat_dev, N * N * sizeof(std::complex<float>));
  cudaMalloc((void**)&ipiv, N * sizeof(int32_t));

  cudaMemcpy(fmat_dev, matAf.data(), N * N * sizeof(std::complex<float>), cudaMemcpyHostToDevice);

  int32_t work = dp_cheidf_gpu(cublasH, N, nullptr, N, nullptr, nullptr, N, nullptr);
  cudaMallocManaged((void**)&work_dev, work * sizeof(std::complex<float>));
  std::cout << work << std::endl;

  int32_t rank = dp_cheidf_gpu(cublasH, N, (cuComplex*)fmat_dev, N, ipiv, (cuComplex*)xmat_dev, N, (cuComplex*)work_dev);
  cudaDeviceSynchronize();
  std::cout << rank << std::endl;

  cudaFree(fmat_dev);
  cudaFree(xmat_dev);
  cudaFree(ipiv);
  cudaFree(work_dev);

  cudaStreamDestroy(stream);
  cublasDestroy(cublasH);
  return 0;
}
