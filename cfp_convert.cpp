
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <complex>
#include <cstdint>
#include <omp.h>
#include <vector>
#include <random>
#include <numeric>
#include <algorithm>
#include <hyacinth.hpp>
#include <internal.hpp>
#include <int_fp_encode.hpp>
#include <double_double.hpp>
#include <mkl.h>

template<int order> double decode_int8(int8_t (&code)[order], int32_t expon) {
  double res = 0;
  int32_t carry = 0;
  int32_t m7 = order % 7, o7 = order - m7;

  for (int32_t i = 0; i < o7; i += 7) {
    int32_t c[7]{};
    for (int32_t j = 0; j < 7; ++j)
      c[j] = int32_t(code[i+j]);
    int64_t val = device::int8::decode_scaled_7xi32(c, carry);
    res += std::scalbn(double(val), 7*(i+expon));
  }

  int32_t c[7]{};
  for (int32_t i = 0; i < m7; ++i)
    c[i] = code[i+o7];
  int64_t val = device::int8::decode_scaled_7xi32(c, carry);
  res += std::scalbn(double(val), 7*(o7+expon));
  res += std::scalbn(double(carry), 7*(o7+expon+7));
  return res;
}

int32_t main() {
  auto err_cuda = cudaSetDevice(0);
  if (err_cuda != cudaSuccess)
  { fprintf(stderr, "%s\n", cudaGetErrorString(err_cuda)); return -1; }

  cudaStream_t stream;
  cublasHandle_t handle;
  cudaStreamCreate(&stream);
  cublasCreate(&handle);
  cublasSetStream(handle, stream);

  const int32_t M = 7777, N = 999;
  const int32_t ldm = ((M + 15) / 16 * 16), ldn = ((N + 15) / 16 * 16);
  constexpr int32_t order = 7;

  std::vector<std::complex<double>> X(M * N), B(ldm * N), C(ldn * N);
  std::vector<complex_double2> D(ldn * N);
  std::vector<int8_t> iX(2 * ldm * ldn * order);
  std::vector<int32_t> expon(N), iAHA(2 * ldn * ldn * (2 * order - 1));

  std::mt19937_64 gen;
  std::normal_distribution<double> dist(0, 32);
  std::generate(X.begin(), X.end(), [&]() { return std::complex<double>(dist(gen), dist(gen)); });

  std::complex<double>* d_A = nullptr;
  int8_t* d_iA = nullptr;
  int32_t* d_exp = nullptr;
  int32_t* d_AHA = nullptr;
  std::complex<double>* d_C = nullptr;
  complex_double2* d_D = nullptr;
  cudaMallocManaged(reinterpret_cast<void**>(&d_A), M * N * sizeof(std::complex<double>), cudaMemAttachGlobal);
  cudaMallocManaged(reinterpret_cast<void**>(&d_iA), 2 * ldm * ldn * order * sizeof(int8_t), cudaMemAttachGlobal);
  cudaMallocManaged(reinterpret_cast<void**>(&d_exp), N * sizeof(int32_t), cudaMemAttachGlobal);
  cudaMallocManaged(reinterpret_cast<void**>(&d_AHA), 2 * ldn * ldn * (2 * order - 1) * sizeof(int32_t), cudaMemAttachGlobal);
  cudaMallocManaged(reinterpret_cast<void**>(&d_C), ldn * N * sizeof(std::complex<double>), cudaMemAttachGlobal);
  cudaMallocManaged(reinterpret_cast<void**>(&d_D), ldn * N * sizeof(complex_double2), cudaMemAttachGlobal);
  cudaMemset(d_iA, 0, ldm * ldn * order * sizeof(int8_t));

  cudaMemcpy(d_A, X.data(), M * N * sizeof(std::complex<double>), cudaMemcpyDefault);
  for (int32_t i = 0; i < 20; ++i)
    internal::int8::vexp_cf64(stream, order, M, N, d_A, M, d_exp);

  cudaDeviceSynchronize();
  cudaMemcpy(expon.data(), d_exp, N * sizeof(int32_t), cudaMemcpyDefault);
  for (int32_t i = 0; i < N; ++i)
    printf("%d ", expon[i]);
  printf("\n");

  for (int32_t i = 0; i < 20; ++i)
    internal::int8::encode_cf64_order20(stream, order, M, N, d_A, M, d_exp, d_iA, ldm,ldm*ldn);
  cudaDeviceSynchronize();
  cudaMemcpy(iX.data(), d_iA, 2 * ldm * ldn * order * sizeof(int8_t), cudaMemcpyDefault);

  for (int32_t j = 0; j < N; ++j) {
    for (int32_t i = 0; i < M; ++i) {
      int8_t code[order]{};
      for (int32_t k = 0; k < order; ++k)
        code[k] = iX[i + j * ldm + (2 * k) * (ldm * ldn)];
      double im = decode_int8(code, expon[j]);

      for (int32_t k = 0; k < order; ++k)
        code[k] = iX[i + j * ldm + (2 * k + 1) * (ldm * ldn)];
      B[i + j * ldm] = std::complex<double>(decode_int8(code, expon[j]), im);
    }
  }

  double err = 0., nrm = 0.;
  for (int32_t j = 0; j < N; ++j) {
    for (int32_t i = 0; i < M; ++i) {
      err += std::norm(B[i + j * ldm] - X[i + j * M]);
      nrm += std::norm(X[i + j * M]);
    }
    for (int32_t i = M; i < ldm; ++i)
      err += std::norm(B[i + j * ldm]);
  }
  printf("%.20le\n", std::sqrt(err / nrm));

  std::complex<double> one(1., 0.), zero(0., 0.);
  cublasZgemm(handle, CUBLAS_OP_C, CUBLAS_OP_N, N, N, M, (cuDoubleComplex*)&one, (cuDoubleComplex*)d_A, M, (cuDoubleComplex*)d_A, M, (cuDoubleComplex*)&zero, (cuDoubleComplex*)d_C, ldn);
  cudaDeviceSynchronize();
  cudaMemcpy(C.data(), d_C, ldn * N * sizeof(std::complex<double>), cudaMemcpyDefault);

  cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
  internal::int8::strided_c8i_AHA_gemm(handle, order, ldm, ldn, d_iA, d_AHA);
  internal::int8::decode_complex_dd_strided_i32(stream, 2*order-1, N, d_exp, d_AHA, ldn, d_D, ldn);
  cudaDeviceSynchronize();
  cudaMemcpy(iAHA.data(), d_AHA, 2 * ldn * ldn * (2 * order - 1) * sizeof(int32_t), cudaMemcpyDefault);
  cudaMemcpy(D.data(), d_D, ldn * N * sizeof(complex_double2), cudaMemcpyDefault);

  double nrm_AHA = 0.;
  for (int32_t i = 0; i < (2 * ldn * ldn * (2 * order - 1)); ++i)
    nrm_AHA += std::norm((double)iAHA[i]);
  printf("%.40le\n", nrm_AHA);

  err = 0.;
  for (int32_t j = 0; j < N; ++j) {
    for (int32_t i = 0; i < N; ++i) {
      complex_double2 c = D[i + j * ldn];
      double rl = c.real.x + c.real.y;
      double im = c.imag.x + c.imag.y;
      err += std::norm(C[i + j * ldn] - std::complex<double>(rl, im));
    }
  }
  printf("%.20le\n", std::sqrt(err) / nrm);

  cudaFree(d_A);
  cudaFree(d_iA);
  cudaFree(d_exp);
  cudaFree(d_AHA);
  cudaFree(d_C);

  cudaStreamDestroy(stream);
  cublasDestroy(handle);
  fprintf(stderr, "%s\n", cudaGetErrorString(cudaGetLastError()));
  return 0;
}
