
#include <commons.hpp>
#include <hyacinth.hpp>

int32_t main() {
  int32_t N = 2048;
  Eigen::MatrixXcd matA(N, N), matB(N, N);
  std::vector<int32_t> ipiv(N);

  random_vector(N * N * 2, (double*)matA.data());
  for (int32_t i = 0; i < N; ++i) {
    for (int32_t j = 0; j < i; ++j)
      matA(j, i) = std::conj(matA(i, j));
    matA(i, i) = std::complex<double>(5.e3+i, 0.);
  }

  //std::cout << matA << std::endl;
  cudaStream_t stream;
  cudaEvent_t start, stop;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  std::complex<double>* d_A;
  cudaMallocManaged(reinterpret_cast<void**>(&d_A), (N + 1) * N * sizeof(std::complex<double>), cudaMemAttachGlobal);
  cudaMemcpy(d_A, matA.data(), N * N * sizeof(std::complex<double>), cudaMemcpyDefault);

  cudaEventRecord(start, stream);
  int32_t ret = zpotrfp_gpu(stream, N, d_A, N, ipiv.data());
  cudaEventRecord(stop, stream);
  cudaDeviceSynchronize();
  std::cout << ret << std::endl;

  cudaMemcpy(matB.data(), d_A, N * N * sizeof(std::complex<double>), cudaMemcpyDefault);
  //for (int32_t i = 0; i < N; ++i)
    //std::cout << ipiv[i] << ", ";

  //std::cout << matA << std::endl << std::endl;
  //std::cout << matB << std::endl << std::endl;

  //std::cout << matB.diagonal() << std::endl;

  Eigen::MatrixXcd matUB = matB.triangularView<Eigen::Upper>();
  Eigen::MatrixXcd matUBp(N, N);
  for (int32_t i = 0; i < N; ++i)
    matUBp.col(ipiv[i] - 1) = matUB.col(i);

  Eigen::MatrixXcd res = matUBp.adjoint() * matUBp;

  std::cout << (matA - res).norm() / matA.norm() << std::endl;
  //std::cout << res << std::endl;

  float milliseconds = 0.0f;
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "Time: " << milliseconds << " ms\n";

  cudaFree(d_A);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaStreamDestroy(stream);
  return 0;
}
