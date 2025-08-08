
#include <commons.hpp>
#include <hyacinth.hpp>

int32_t main() {
  int32_t N = 2048;
  Eigen::MatrixXd matA(N, N), matB(N, N);
  int32_t* ipiv;

  random_vector(N * N, (double*)matA.data());
  for (int32_t i = 0; i < N; ++i) {
    for (int32_t j = 0; j < i; ++j)
      matA(j, i) = matA(i, j);
    matA(i, i) = 5.e4+i;
  }

  //std::cout << matA << std::endl;
  cudaStream_t stream;
  cudaEvent_t start, stop;
  cudaStreamCreate(&stream);
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  double* d_A;
  cudaMallocManaged(reinterpret_cast<void**>(&d_A), (N + 1) * N * sizeof(double), cudaMemAttachGlobal);
  cudaMallocHost(reinterpret_cast<void**>(&ipiv), (N + 8) * sizeof(int32_t));
  cudaMemcpy(d_A, matA.data(), N * N * sizeof(double), cudaMemcpyDefault);

  cudaEventRecord(start, stream);
  int32_t ret = device::Cholesky::dpotrfp(stream, N, d_A, N, ipiv);
  cudaEventRecord(stop, stream);
  cudaDeviceSynchronize();
  std::cout << ret << std::endl;

  cudaMemcpy(matB.data(), d_A, N * N * sizeof(double), cudaMemcpyDefault);
  //for (int32_t i = 0; i < N; ++i)
    //std::cout << ipiv[i] << ", ";

  //std::cout << matA << std::endl << std::endl;
  //std::cout << matB << std::endl << std::endl;

  //std::cout << matB.diagonal() << std::endl;

  Eigen::MatrixXd matUB = matB.triangularView<Eigen::Upper>();
  Eigen::MatrixXd matUBp(N, N);
  for (int32_t i = 0; i < N; ++i)
    matUBp.col(ipiv[i] - 1) = matUB.col(i);

  Eigen::MatrixXd res = matUBp.adjoint() * matUBp;

  std::cout << (matA - res).norm() / matA.norm() << std::endl;
  //std::cout << res << std::endl;

  float milliseconds = 0.0f;
  cudaEventElapsedTime(&milliseconds, start, stop);
  int64_t flops = int64_t(N) * int64_t(N) * int64_t(N) / 3;
  std::cout << "Time: " << milliseconds << " ms\n";
  std::cout << "GFLOPs: " << double(flops) * 1.e-6 / milliseconds << "\n";

  cudaFree(d_A);
  cudaFreeHost(ipiv);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaStreamDestroy(stream);
  return 0;
}
