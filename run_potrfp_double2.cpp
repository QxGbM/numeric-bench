
#include <commons.hpp>
#include <hyacinth.hpp>

complex_double2 double_double2(std::complex<double> a) {
  return device::dd::make_complex_double2(make_double2(a.real(), 0.), make_double2(a.imag(), 0.));
}

std::complex<double> double2_double(complex_double2 a) {
  return std::complex<double>(double(a.real.x) + double(a.real.y), double(a.imag.x) + double(a.imag.y));
}

int32_t main() {
  int32_t N = 4096;
  Eigen::MatrixXcd matA(N, N), matB(N, N);
  int32_t* ipiv;
  typedef complex_double2 complex_t;
  std::vector<complex_t> matA_f4(N * N);

  random_vector(N * N * 2, (double*)matA.data());
  for (int32_t i = 0; i < N; ++i) {
    for (int32_t j = 0; j < i; ++j)
      matA(j, i) = std::conj(matA(i, j));
    matA(i, i) = std::complex<double>(5.e3+i, 0.);
  }

  for (int32_t i = 0; i < N * N; ++i) {
    std::complex<double> e = matA.reshaped()[i];
    matA_f4[i] = double_double2(e);
  }

  //std::cout << matA << std::endl;
  cudaStream_t stream;
  cudaEvent_t start, stop;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  complex_t* d_A;
  cudaMallocManaged(reinterpret_cast<void**>(&d_A), (N + 1) * N * sizeof(complex_t), cudaMemAttachGlobal);
  cudaMallocHost(reinterpret_cast<void**>(&ipiv), (N + 8) * sizeof(int32_t));
  cudaMemcpy(d_A, matA_f4.data(), N * N * sizeof(complex_t), cudaMemcpyDefault);

  cudaEventRecord(start, stream);
  int32_t ret = device::Cholesky::complex_double_double_potrfp(stream, N, d_A, N, ipiv);
  cudaEventRecord(stop, stream);
  cudaDeviceSynchronize();
  std::cout << ret << std::endl;

  cudaMemcpy(matA_f4.data(), d_A, N * N * sizeof(complex_t), cudaMemcpyDefault);
  //for (int32_t i = 0; i < N; ++i)
    //std::cout << ipiv[i] << ", ";

  //std::cout << matA << std::endl << std::endl;
  //std::cout << matB << std::endl << std::endl;

  //std::cout << matB.diagonal() << std::endl;

  for (int32_t i = 0; i < N * N; ++i) {
    complex_t e = matA_f4[i];
    matB.reshaped()[i] = double2_double(e);
  }

  Eigen::MatrixXcd matUB = matB.triangularView<Eigen::Upper>();
  Eigen::MatrixXcd matUBp(N, N);
  for (int32_t i = 0; i < N; ++i)
    matUBp.col(ipiv[i] - 1) = matUB.col(i);

  Eigen::MatrixXcd res = matUBp.adjoint() * matUBp;

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
