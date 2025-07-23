
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

template<int order> double decode_int8(int8_t (&code)[order], int32_t expon) {
  double res = 0;
  for (int i = 0; i < order; ++i)
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

  const int32_t M = 7777, N = 48, LD = 8000;
  constexpr int32_t order = 7;

  std::vector<std::complex<double>> X(M * N), B(LD * N);
  std::vector<int8_t> iX(2 * LD * N * order);
  std::vector<int32_t> expon(N);

  std::mt19937_64 gen;
  std::normal_distribution<double> dist(0, 32);
  std::generate(X.begin(), X.end(), [&]() { return std::complex<double>(dist(gen), dist(gen)); });

  std::complex<double>* d_A = nullptr;
  int8_t* d_iA = nullptr;
  int32_t* d_exp = nullptr;
  cudaMallocManaged(reinterpret_cast<void**>(&d_A), M * N * sizeof(std::complex<double>), cudaMemAttachGlobal);
  cudaMallocManaged(reinterpret_cast<void**>(&d_iA), 2 * LD * N * order * sizeof(int8_t), cudaMemAttachGlobal);
  cudaMallocManaged(reinterpret_cast<void**>(&d_exp), N * sizeof(int32_t), cudaMemAttachGlobal);
  cudaMemset(d_iA, 0, LD * N * order * sizeof(int8_t));

  cudaMemcpy(d_A, X.data(), M * N * sizeof(std::complex<double>), cudaMemcpyDefault);
  for (int32_t i = 0; i < 20; ++i)
    internal::int8::vexp_cf64(stream, order, M, N, d_A, M, d_exp);

  cudaDeviceSynchronize();
  cudaMemcpy(expon.data(), d_exp, N * sizeof(int32_t), cudaMemcpyDefault);
  for (int32_t i = 0; i < N; ++i)
    printf("%d ", expon[i]);
  printf("\n");

  for (int32_t i = 0; i < 20; ++i)
    internal::int8::encode_cf64_order20(stream, order, M, N, d_A, M, d_exp, d_iA, LD, LD * N);
  cudaDeviceSynchronize();
  cudaMemcpy(iX.data(), d_iA, 2 * LD * N * order * sizeof(int8_t), cudaMemcpyDefault);

  for (int32_t j = 0; j < N; ++j) {
    for (int32_t i = 0; i < M; ++i) {
      int8_t code[order]{};
      for (int32_t k = 0; k < order; ++k)
        code[k] = iX[i + j * LD + (2 * k) * (LD * N)];
      double rl = decode_int8(code, expon[j]);

      for (int32_t k = 0; k < order; ++k)
        code[k] = iX[i + j * LD + (2 * k + 1) * (LD * N)];
      B[i + j * LD] = std::complex<double>(rl, decode_int8(code, expon[j]));
    }
  }

  double err = 0., nrm = 0.;
  for (int32_t j = 0; j < N; ++j) {
    for (int32_t i = 0; i < M; ++i) {
      err += std::norm(B[i + j * LD] - X[i + j * M]);
      nrm += std::norm(X[i + j * M]);
    }
    for (int32_t i = M; i < LD; ++i)
      err += std::norm(B[i + j * LD]);
  }
  printf("%.20le\n", std::sqrt(err / nrm));

  cudaFree(d_A);
  cudaFree(d_iA);
  cudaFree(d_exp);

  cudaStreamDestroy(stream);
  cublasDestroy(handle);
  fprintf(stderr, "%s\n", cudaGetErrorString(cudaGetLastError()));
  return 0;
}
