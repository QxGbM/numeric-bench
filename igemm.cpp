
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cstdint>
#include <random>
#include <numeric>
#include <algorithm>
#include <omp.h>

int32_t main(int32_t argc, char* argv[]) {
  auto cu_err = cudaSetDevice(0);
  if (cu_err != cudaSuccess)
  { fprintf(stderr, "%s\n", cudaGetErrorString(cu_err)); return -1; }

  cudaStream_t stream;
  cublasHandle_t handle;
  cudaStreamCreate(&stream);
  cublasCreate(&handle);
  cublasSetStream(handle, stream);

  int32_t M = std::atoi(argv[1]), N = std::atoi(argv[2]), K = std::atoi(argv[3]);
  std::vector<int8_t> i_A(K * M), i_B(K * N);
  std::vector<int32_t> i_C(M * N, 127);
  std::vector<int32_t> i_D(M * N), i_E(M * N);

  std::mt19937_64 gen;
  std::uniform_int_distribution<int32_t> dist(32, 63);
  std::generate(i_A.begin(), i_A.end(), [&]() { return (int8_t)(dist(gen)); });
  std::generate(i_B.begin(), i_B.end(), [&]() { return (int8_t)(dist(gen)); });

  int8_t* d_A, *d_B;
  int32_t* d_C, *d_E;
  cudaMallocManaged(reinterpret_cast<void**>(&d_A), K * M * sizeof(int8_t), cudaMemAttachGlobal);
  cudaMallocManaged(reinterpret_cast<void**>(&d_B), K * N * sizeof(int8_t), cudaMemAttachGlobal);
  cudaMallocManaged(reinterpret_cast<void**>(&d_C), M * N * sizeof(int32_t), cudaMemAttachGlobal);
  cudaMallocManaged(reinterpret_cast<void**>(&d_E), M * N * sizeof(int32_t), cudaMemAttachGlobal);

  int32_t one = 1, zero = 0, minus_one = -1;
  cudaMemcpy(d_A, i_A.data(), K * M * sizeof(int8_t), cudaMemcpyDefault);
  cudaMemcpy(d_B, i_B.data(), K * N * sizeof(int8_t), cudaMemcpyDefault);
  cudaMemcpy(d_C, i_C.data(), M * N * sizeof(int32_t), cudaMemcpyDefault);

  for (int32_t j = 0; j < N; ++j)
    for (int32_t i = 0; i < M; ++i)
      for (int32_t k = 0; k < K; ++k)
        i_C[i + j * M] += int32_t(i_A[k + i * K]) * int32_t(i_B[k + j * K]);

  //for (int32_t k = 0; k < K; k += 1024)
    //cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, 1024, &one, &d_A[k], CUDA_R_8I, K, &d_B[k], CUDA_R_8I, K, &zero, d_C, CUDA_R_32I, M, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &one, d_A, CUDA_R_8I, K, d_B, CUDA_R_8I, K, &one, d_C, CUDA_R_32I, M, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  cudaDeviceSynchronize();
  //cudaMemcpy(d_E, d_C, M * N * sizeof(int32_t), cudaMemcpyDefault);
  //cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &one, d_A, CUDA_R_8I, K, d_B, CUDA_R_8I, K, &one, d_E, CUDA_R_32I, M, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  //cudaDeviceSynchronize();

  cudaMemcpy(i_D.data(), d_C, M * N * sizeof(float), cudaMemcpyDefault);
  cudaMemcpy(i_E.data(), d_E, M * N * sizeof(float), cudaMemcpyDefault);
  
  int64_t abs_err = 0, sign_err = 0, nrm = 0;
  for (int32_t j = 0; j < N; ++j) {
    for (int32_t i = 0; i < M; ++i) {
      int32_t A_ij = int32_t(i_D[i + j * M]) - int32_t(i_E[i + j * M]);
      abs_err += std::abs(A_ij - i_C[i + j * M]);
      sign_err += (A_ij - i_C[i + j * M]);
      nrm += i_C[i + j * M];
      //if (A_ij != i_C[i + j * M])
        printf("err loc: <%d %d> cpu:%d cuda:%d=%d+%d\n", i, j, i_C[i + j * M], A_ij, i_D[i + j * M], -i_E[i + j * M]);
    }
  }
  printf("%lld %lld %lld\n",abs_err,sign_err,nrm);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFree(d_E);

  cudaStreamDestroy(stream);
  cublasDestroy(handle);
  return 0;
}
