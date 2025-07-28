
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
#include <double_double.hpp>
#include <int_fp_encode.hpp>

template<int order> double decode_int(int8_t (&code)[order], int32_t expon) {
  double res = 0;
  for (int32_t i = 0; i < order; ++i)
    res += std::scalbn(double(code[i]), 7*(i+expon));
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

  const int32_t M = 8000, N = 800;
  const int32_t ldm = ((M + 15) / 16 * 16), ldn = ((N + 15) / 16 * 16);
  constexpr int32_t order = 7;

  std::vector<double> X(M * N), B(ldm * N), C(ldn * N), D(ldn * N);
  std::vector<int8_t> iX(ldm * ldn * order);
  std::vector<int32_t> expon(N), iAHA(ldn * N * (2 * order - 1));

  std::mt19937_64 gen;
  std::normal_distribution<double> dist(0, 32);
  std::generate(X.begin(), X.end(), [&]() { return dist(gen); });

  double* d_A = nullptr;
  int8_t* d_iA = nullptr;
  int32_t* d_exp = nullptr;
  int32_t* d_AHA = nullptr;
  double* d_C = nullptr;
  cudaMallocManaged(reinterpret_cast<void**>(&d_A), M * N * sizeof(double), cudaMemAttachGlobal);
  cudaMallocManaged(reinterpret_cast<void**>(&d_iA), ldm * ldn * order * sizeof(int8_t), cudaMemAttachGlobal);
  cudaMallocManaged(reinterpret_cast<void**>(&d_exp), N * sizeof(int32_t), cudaMemAttachGlobal);
  cudaMallocManaged(reinterpret_cast<void**>(&d_AHA), ldn * ldn * (2 * order - 1) * sizeof(int32_t), cudaMemAttachGlobal);
  cudaMallocManaged(reinterpret_cast<void**>(&d_C), ldn * N * sizeof(double), cudaMemAttachGlobal);
  cudaMemset(d_iA, 0, ldm * ldn * order * sizeof(int8_t));

  cudaMemcpy(d_A, X.data(), M * N * sizeof(double), cudaMemcpyDefault);
  for (int32_t i = 0; i < 20; ++i)
    internal::int8::vexp_f64(stream, order, M, N, d_A, M, d_exp);

  cudaDeviceSynchronize();
  cudaMemcpy(expon.data(), d_exp, N * sizeof(int32_t), cudaMemcpyDefault);
  for (int32_t i = 0; i < N; ++i)
    printf("%d ", expon[i]);
  printf("\n");

  for (int32_t i = 0; i < 20; ++i)
    internal::int8::encode_f64_order20(stream, order, M, N, d_A, M, d_exp, d_iA, ldm, ldm*ldn);
  cudaDeviceSynchronize();
  cudaMemcpy(iX.data(), d_iA, ldm * ldn * order * sizeof(int8_t), cudaMemcpyDefault);

  for (int32_t j = 0; j < N; ++j) {
    for (int32_t i = 0; i < M; ++i) {
      int8_t code[order]{};
      for (int32_t k = 0; k < order; ++k)
        code[k] = iX[i + j * ldm + k * (ldm * ldn)];
      B[i + j * ldm] = decode_int(code, expon[j]);
    }
  }

  double err = 0., nrm = 0.;
  for (int32_t j = 0; j < N; ++j) {
    for (int32_t i = 0; i < M; ++i) {
      err += std::pow(B[i + j * ldm] - X[i + j * M], 2);
      nrm += std::pow(X[i + j * M], 2);
    }
    for (int32_t i = M; i < ldm; ++i)
      err += std::pow(B[i + j * ldm], 2);
  }
  printf("%.20le\n", std::sqrt(err / nrm));

  double one = 1., zero = 0.;
  cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, N, M, &one, d_A, M, d_A, M, &zero, d_C, ldn);
  cudaDeviceSynchronize();
  cudaMemcpy(C.data(), d_C, ldn * N * sizeof(double), cudaMemcpyDefault);

  internal::int8::strided_r8i_ATA_gemm(handle, order, ldm, ldn, d_iA, d_AHA);
  internal::int8::decode_f64_strided_i32(stream, 2*order-1, N, d_exp, d_AHA, ldn, d_C, ldn);
  cudaDeviceSynchronize();
  cudaMemcpy(D.data(), d_C, ldn * N * sizeof(double), cudaMemcpyDefault);

  err = 0.;
  for (int32_t j = 0; j < N; ++j) {
    for (int32_t i = 0; i < N; ++i) {
      double c = D[i + j * ldn];
      err += std::pow(C[i + j * ldn] - c, 2);
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
