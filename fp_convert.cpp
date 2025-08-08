
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

template<int32_t base, int32_t order> double decode_int(int8_t (&code)[order], int32_t expon) {
  double res = 0;
  for (int32_t i = 0; i < order; ++i)
    res += std::scalbn(double(code[i]), base*(i+expon));
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

  const int32_t M = 4096, N = 1024;
  const int32_t ldm = ((M + 15) / 16 * 16), ldn = ((N + 15) / 16 * 16);
  constexpr int32_t order = 9;

  std::vector<double> X(M * N), B(ldm * N), C(ldn * N), D(ldn * N);
  std::vector<int8_t> iX(ldm * ldn * order);
  std::vector<int32_t> expon(N), iAHA(ldn * ldn * (2*order));

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
  cudaMallocManaged(reinterpret_cast<void**>(&d_AHA), ldn * ldn * (2*order) * sizeof(int32_t), cudaMemAttachGlobal);
  cudaMallocManaged(reinterpret_cast<void**>(&d_C), ldn * N * sizeof(double), cudaMemAttachGlobal);
  cudaMemset(d_iA, 0, ldm * ldn * order * sizeof(int8_t));

  cudaMemcpy(d_A, X.data(), M * N * sizeof(double), cudaMemcpyDefault);
  for (int32_t i = 0; i < 20; ++i)
    internal::int8::vexp_f64(stream, order, M, N, d_A, M, d_exp);

  cudaDeviceSynchronize();
  cudaMemcpy(expon.data(), d_exp, N * sizeof(int32_t), cudaMemcpyDefault);
  //for (int32_t i = 0; i < N; ++i)
    //printf("%d ", expon[i]);
  //printf("\n");

  for (int32_t i = 0; i < 20; ++i)
    internal::int8::encode_f64(stream, order, M, N, d_A, M, d_exp, d_iA, ldm);
  cudaDeviceSynchronize();
  cudaMemcpy(iX.data(), d_iA, ldm * ldn * order * sizeof(int8_t), cudaMemcpyDefault);

  for (int32_t j = 0; j < N; ++j) {
    for (int32_t i = 0; i < M; ++i) {
      int8_t code[order]{};
      for (int32_t k = 0; k < order; ++k)
        code[k] = iX[i + j * ldm + k * (ldm * ldn)];
      B[i + j * ldm] = decode_int<device::Config::exp_base>(code, expon[j]);
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

  double nrm_iA = 0.;
  for (int32_t i = 0; i < (ldm * ldn * order); ++i)
    nrm_iA += std::norm((double)iX[i]);
  printf("%.40le\n", nrm_iA);

  double one = 1., zero = 0.;
  cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, N, M, &one, d_A, M, d_A, M, &zero, d_C, ldn);
  cudaDeviceSynchronize();
  cudaMemcpy(C.data(), d_C, ldn * N * sizeof(double), cudaMemcpyDefault);

  int32_t orderC = 14;
  internal::int8::r8i_TN_gemm_stridedA(stream, handle, N, 1024, ldn, ldm, d_iA, d_iA, order, d_AHA, orderC);
  for (int32_t i = 0; i < 20; ++i) {
    cudaMemsetAsync(d_C, 0, ldn * N * sizeof(double), stream);
    internal::int8::decode_f64_strided_i32(stream, 2*order-orderC, 2*order, N, d_exp, d_AHA, ldn, d_C, ldn);
  }
  cudaDeviceSynchronize();
  cudaMemcpy(iAHA.data(), d_AHA, ldn * ldn * (2*order) * sizeof(int32_t), cudaMemcpyDefault);
  cudaMemcpy(D.data(), d_C, ldn * N * sizeof(double), cudaMemcpyDefault);

  int64_t abs_AHA = 0, sign_AHA = 0;
  for (int32_t i = 0; i < (ldn * ldn * (2*order)); ++i) {
    abs_AHA += std::abs(iAHA[i]);
    sign_AHA += iAHA[i];
  }
  printf("%lld %lld\n", abs_AHA, sign_AHA);

  err = 0.;
  for (int32_t j = 0; j < N; ++j) {
    for (int32_t i = 0; i < N; ++i) {
      double c = D[i + j * ldn];
      err += std::norm(C[i + j * ldn] - c);
    }
  }
  printf("%.40le\n", std::sqrt(err) / nrm);

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
