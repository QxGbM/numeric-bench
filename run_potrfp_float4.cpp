
#include <commons.hpp>
#include <hyacinth.hpp>

float4 double_float4(double a) {
  float x = float(a);
  a -= double(x);
  float y = float(a);
  a -= double(y);
  float z = float(a);
  return float4 { x, y, z, 0.f };
}

double float4_double(float4 a) {
  return double(a.x) + double(a.y) + double(a.z) + double(a.w);
}

int32_t main() {
  int32_t N = 2048;
  Eigen::MatrixXcd matA(N, N), matB(N, N);
  std::vector<int32_t> ipiv(N);
  std::vector<complex_float4> matA_f4(N * N);

  random_vector(N * N * 2, (double*)matA.data());
  for (int32_t i = 0; i < N; ++i) {
    for (int32_t j = 0; j < i; ++j)
      matA(j, i) = std::conj(matA(i, j));
    matA(i, i) = std::complex<double>(5.e3+i, 0.);
  }

  for (int32_t i = 0; i < N * N; ++i) {
    std::complex<double> e = matA.reshaped()[i];
    matA_f4[i] = host::qf::make_complex_float4(double_float4(e.real()), double_float4(e.imag()));
  }

  //std::cout << matA << std::endl;
  cudaStream_t stream;
  cudaEvent_t start, stop;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  complex_float4* d_A;
  cudaMallocManaged(reinterpret_cast<void**>(&d_A), (N + 1) * N * sizeof(complex_float4), cudaMemAttachGlobal);
  cudaMemcpy(d_A, matA_f4.data(), N * N * sizeof(complex_float4), cudaMemcpyDefault);

  cudaEventRecord(start, stream);
  int32_t ret = complex_quad_float_potrfp_gpu(stream, N, d_A, N, ipiv.data());
  cudaEventRecord(stop, stream);
  cudaDeviceSynchronize();
  std::cout << ret << std::endl;

  cudaMemcpy(matA_f4.data(), d_A, N * N * sizeof(complex_float4), cudaMemcpyDefault);
  //for (int32_t i = 0; i < N; ++i)
    //std::cout << ipiv[i] << ", ";

  //std::cout << matA << std::endl << std::endl;
  //std::cout << matB << std::endl << std::endl;

  //std::cout << matB.diagonal() << std::endl;

  for (int32_t i = 0; i < N * N; ++i) {
    complex_float4 e = matA_f4[i];
    matB.reshaped()[i] = std::complex<double>(float4_double(e.real), float4_double(e.imag));
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
  std::cout << "Time: " << milliseconds << " ms\n";

  cudaFree(d_A);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaStreamDestroy(stream);
  return 0;
}
