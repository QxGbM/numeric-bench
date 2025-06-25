

#include <hyacinth.hpp>
#include <commons.hpp>
#include <omp.h>
#include <iostream>
#include <algorithm>
#include <random>
#include <vector>
#include <complex>

int32_t main() {
  auto err = cudaSetDevice(0);
  if (err != cudaSuccess)
  { fprintf(stderr, "%s\n", cudaGetErrorString(err)); return -1; }
  
  cudaStream_t stream;
  cublasHandle_t cublasH;
  cudaStreamCreate(&stream);
  cublasCreate(&cublasH);
  cublasSetStream(cublasH, stream);

  int64_t M = 4000, N = 800;
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
  printf("rank is: %d\n", rank);

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
