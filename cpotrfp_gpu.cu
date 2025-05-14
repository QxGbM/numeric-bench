
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <thrust/pair.h>
#include <thrust/reduce.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/counting_iterator.h>

#include <vector>
#include <complex>

struct Ismax {
  const int32_t ld;
  const float* ptr;
  Ismax(int32_t ld, const float* ptr) : ld(ld), ptr(ptr) {}
  __device__ thrust::pair<float, int32_t> operator()(int32_t i) {
    return thrust::pair<float, int32_t>(ptr[i * ld], i);
  }
  __device__ thrust::pair<float, int32_t> operator()(thrust::pair<float, int32_t> e1, thrust::pair<float, int32_t> e2) {
    return e1.first < e2.first ? e2 : e1;
  }
};

int32_t cpotrfp_gpu(cublasHandle_t handle, int32_t N, const cuComplex* A, int32_t lda, int32_t* ipiv, cuComplex* X, int32_t ldx, cuComplex* work) {
  int32_t quot = 1 + (((N & 7) - 1) >> 31);
  int32_t ld = ((N >> 3) + quot) << 3;
  if (work == nullptr)
    return ld * (N + 1);

  int32_t rank = N;
  cuComplex* L = &work[N * ld];

  cudaStream_t stream;
  cublasGetStream(handle, &stream);
  cudaMemcpy2DAsync(work, ld * sizeof(std::complex<float>), A, lda * sizeof(std::complex<float>), N * sizeof(std::complex<float>), N, cudaMemcpyDefault, stream);

  const struct Ismax imax_func(2 * (ld + 1), (float*)work);
  auto diag_iter = thrust::make_transform_iterator(thrust::make_counting_iterator(0), imax_func);

  std::vector<int32_t> piv(N);
  float s0 = 0.f;
  const std::complex<float> zero(0.f, 0.f), minus_one(-1.f, 0.f);
  for (int32_t i = 0; i < rank; ++i) {
    thrust::pair<float, int32_t> max = thrust::reduce(thrust::cuda::par_nosync.on(stream), diag_iter, diag_iter + N, thrust::pair<float, int32_t>(0.f, 0), imax_func);
    std::complex<float> scale(1.f / max.first, 0.f);
    piv[i] = max.second;
    
    cublasCcopy(handle, N, &work[piv[i] * ld], 1, L, 1);
    cublasCgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, 1, (const cuComplex*)&scale, &work[piv[i] * ld], ld, (const cuComplex*)&zero, &X[i * ldx], ldx, &X[i * ldx], ldx);
    cublasCgerc(handle, N, N, (const cuComplex*)&minus_one, &X[i * ldx], 1, L, 1, work, ld);

    if (0 == i)
      s0 = scale.real() * 32768.f;
    if (s0 < scale.real())
      rank = i;
  }

  cudaMemcpyAsync(ipiv, &piv[0], N * sizeof(int32_t), cudaMemcpyDefault, stream);
  for (int32_t i = rank - 1; 0 < i; --i) {
    cublasCcopy(handle, i, &X[piv[i]], ldx, L, 1);
    cublasCgeru(handle, N, i, (const cuComplex*)&minus_one, &X[i * ldx], 1, L, 1, X, ldx);
  }
  return rank;
}

#include <commons.hpp>
#include <omp.h>
#include <iostream>
#include <algorithm>
#include <random>

int32_t main() {
  auto err = cudaSetDevice(0);
  if (err != cudaSuccess)
  { fprintf(stderr, "%s\n", cudaGetErrorString(err)); return -1; }
  
  cudaStream_t stream;
  cublasHandle_t cublasH;
  cudaStreamCreate(&stream);
  cublasCreate(&cublasH);
  cublasSetStream(cublasH, stream);

  int64_t M = 4000, N = 1400;
  Eigen::MatrixXcd matA(M, N);
  random_vector(M * N * 2, (double*)matA.data());
  Eigen::MatrixXcf matAf(M, N);
  for (int32_t j = 0; j < N; ++j)
    for (int32_t i = 0; i < M; ++i)
      matAf(i, j) = std::complex<float>(matA(i,j).real(), matA(i, j).imag());

  Eigen::MatrixXcf AAT = matAf.adjoint() * matAf;
  for (int32_t i = 0; i < 5; ++i) {
    AAT /= AAT.norm();
    AAT = AAT.adjoint() * AAT;
  }

  matAf = matAf * AAT;
  matAf /= matAf.norm();
  AAT = matAf.adjoint() * matAf;

  std::complex<float>* fmat_dev = nullptr, *xmat_dev = nullptr, *work_dev = nullptr;
  std::vector<int32_t> ipiv(N);
  cudaMalloc((void**)&fmat_dev, N * N * sizeof(std::complex<float>));
  cudaMalloc((void**)&xmat_dev, N * N * sizeof(std::complex<float>));

  cudaMemcpy(fmat_dev, AAT.data(), N * N * sizeof(std::complex<float>), cudaMemcpyHostToDevice);

  int32_t work = cpotrfp_gpu(cublasH, N, nullptr, N, nullptr, nullptr, N, nullptr);
  cudaMalloc((void**)&work_dev, work * sizeof(std::complex<float>));

  double start, lapse;
  int32_t loops = 5, rank = 0;
  start = omp_get_wtime();

  for (int32_t i = 0; i < loops; ++i)
    rank = cpotrfp_gpu(cublasH, N, (cuComplex*)fmat_dev, N, ipiv.data(), (cuComplex*)xmat_dev, N, (cuComplex*)work_dev);
  
  cudaDeviceSynchronize();
  lapse = omp_get_wtime() - start;
  double gf = 1.e-9 * (N * rank * (N * 4 + rank * 2)) * loops;
  printf("<h-lra> time: %f ms. Gflops: %f\n", lapse * 1000 / loops, gf / lapse);
  std::cout << rank << std::endl;

  Eigen::MatrixXcf Ax(N, rank);
  Eigen::MatrixXcf As(M, rank);
  for (int32_t i = 0; i < rank; ++i)
    As.col(i) = matAf.col(ipiv[i]);
  
  cudaMemcpy(Ax.data(), xmat_dev, N * rank * sizeof(std::complex<float>), cudaMemcpyDeviceToHost);
  std::cout << "id error: " << (matAf - As * Ax.adjoint()).norm() / matAf.norm() << std::endl;

  cudaFree(fmat_dev);
  cudaFree(xmat_dev);
  cudaFree(work_dev);

  cudaStreamDestroy(stream);
  cublasDestroy(cublasH);
  return 0;
}
