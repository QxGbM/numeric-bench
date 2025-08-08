
#include <cublasLt.h>
#include <cuda_runtime_api.h>
#include <cstdint>
#include <random>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <omp.h>

int32_t main(int32_t argc, char* argv[]) {
  auto cu_err = cudaSetDevice(0);
  if (cu_err != cudaSuccess)
  { fprintf(stderr, "%s\n", cudaGetErrorString(cu_err)); return -1; }

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  int32_t M = std::atoi(argv[1]);
  int32_t N = std::atoi(argv[2]);
  int32_t K = std::atoi(argv[3]);
  int32_t batch = std::atoi(argv[4]);

  std::vector<int8_t> i_A(K * M), i_B(K * N);
  std::vector<int32_t> i_C(M * N, 0);
  std::vector<int32_t> i_D(batch * M * N);

  std::mt19937_64 gen;
  std::uniform_int_distribution<int32_t> dist(63, 127);
  std::generate(i_A.begin(), i_A.end(), [&]() { return (int8_t)(dist(gen)); });
  std::generate(i_B.begin(), i_B.end(), [&]() { return (int8_t)(dist(gen)); });

  int8_t* d_A, *d_B;
  int32_t* d_C;
  cudaMallocManaged(reinterpret_cast<void**>(&d_A), K * M * sizeof(int8_t), cudaMemAttachGlobal);
  cudaMallocManaged(reinterpret_cast<void**>(&d_B), K * N * sizeof(int8_t), cudaMemAttachGlobal);
  cudaMallocManaged(reinterpret_cast<void**>(&d_C), batch * M * N * sizeof(int32_t), cudaMemAttachGlobal);

  cudaMemcpy(d_A, i_A.data(), K * M * sizeof(int8_t), cudaMemcpyDefault);
  cudaMemcpy(d_B, i_B.data(), K * N * sizeof(int8_t), cudaMemcpyDefault);

  /*for (int32_t j = 0; j < N; ++j)
    for (int32_t i = 0; i < M; ++i)
      for (int32_t k = 0; k < K; ++k)
        i_C[i + j * M] += int32_t(i_A[k + i * K]) * int32_t(i_B[k + j * K]);*/

  cublasLtHandle_t ltHandle;
  cublasLtCreate(&ltHandle);

  // Matrix descriptors
  cublasLtMatrixLayout_t layoutA, layoutB, layoutC;
  cublasLtOrder_t row_major = CUBLASLT_ORDER_ROW;
  cublasLtMatrixLayoutCreate(&layoutA, CUDA_R_8I, M, K / batch, K / batch);
  cublasLtMatrixLayoutCreate(&layoutB, CUDA_R_8I, K / batch, N, K / batch);
  cublasLtMatrixLayoutCreate(&layoutC, CUDA_R_32I, M, N, M);

  cublasLtMatrixLayoutSetAttribute(layoutA, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_major, sizeof(cublasLtOrder_t));
  if (1 < batch) {
    int64_t strideA = (M * K) / batch, strideB = (K * N) / batch, strideC = int64_t(M) * int64_t(N);
    cublasLtMatrixLayoutSetAttribute(layoutA, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(int32_t));
    cublasLtMatrixLayoutSetAttribute(layoutA, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideA, sizeof(int64_t));

    cublasLtMatrixLayoutSetAttribute(layoutB, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(int32_t));
    cublasLtMatrixLayoutSetAttribute(layoutB, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideB, sizeof(int64_t));

    cublasLtMatrixLayoutSetAttribute(layoutC, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(int32_t));
    cublasLtMatrixLayoutSetAttribute(layoutC, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideC, sizeof(int64_t));
  }

  // Matmul descriptor
  cublasLtMatmulDesc_t matmulDesc;
  cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32I, CUDA_R_32I);

  // Scaling factors
  int32_t alpha = 1;
  int32_t beta = 0;

  // Preference (optional tuning)
  cublasLtMatmulPreference_t preference;
  cublasLtNumericalImplFlags_t numerics = CUBLASLT_NUMERICAL_IMPL_FLAGS_IMMA | CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_32I | CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_8I;
  cublasLtMatmulPreferenceCreate(&preference);
  cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_IMPL_MASK, &numerics, sizeof(cublasLtNumericalImplFlags_t));

  // Workspace (optional, helps performance)
  void* workspace = nullptr;
  size_t workspaceSize = 16 << 20; // 16MB
  cudaMalloc(&workspace, workspaceSize);
  cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(size_t));

  // Heuristic result
  cublasLtMatmulHeuristicResult_t heuristicResult;
  int returnedResults = 0;

  cublasLtMatmulAlgoGetHeuristic(ltHandle, matmulDesc, layoutA, layoutB, layoutC, layoutC, preference, 1, &heuristicResult, &returnedResults);

  if (returnedResults == 0) {
    fprintf(stderr, "No suitable GEMM algorithm found.\n");
    std::exit(EXIT_FAILURE);
  }

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cublasLtMatmul(ltHandle, matmulDesc, &alpha, d_A, layoutA, d_B, layoutB, &beta, d_C, layoutC, d_C, layoutC, &heuristicResult.algo, workspace, workspaceSize, stream);

  // Execute the GEMM
  int32_t loops = 50;
  cudaEventRecord(start, stream);
  for (int32_t i = 0; i < loops; ++i)
    cublasLtMatmul(ltHandle, matmulDesc, &alpha, d_A, layoutA, d_B, layoutB, &beta, d_C, layoutC, d_C, layoutC, &heuristicResult.algo, workspace, workspaceSize, stream);
  cudaEventRecord(stop, stream);
  cudaDeviceSynchronize();
  cudaMemcpy(i_D.data(), d_C, batch * M * N * sizeof(int32_t), cudaMemcpyDefault);

  float milliseconds = 0.0f;
  cudaEventElapsedTime(&milliseconds, start, stop);
  int64_t flops = int64_t(M) * int64_t(N) * int64_t(K) * 2 * int64_t(loops);
  std::cout << "Time: " << milliseconds << " ms\n";
  std::cout << "TFLOPs: " << double(flops) * 1.e-9 / milliseconds << "\n";

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // Clean up
  cudaFree(workspace);
  cublasLtMatmulPreferenceDestroy(preference);
  cublasLtMatmulDescDestroy(matmulDesc);
  cublasLtMatrixLayoutDestroy(layoutA);
  cublasLtMatrixLayoutDestroy(layoutB);
  cublasLtMatrixLayoutDestroy(layoutC);
  cublasLtDestroy(ltHandle);

  /*int64_t abs_err = 0, sign_err = 0, nrm = 0;
  for (int32_t j = 0; j < N; ++j) {
    for (int32_t i = 0; i < M; ++i) {
      int32_t A_ij = i_D[i + j * M] + i_D[i + j * M + M * N] + i_D[i + j * M + 2 * M * N] + i_D[i + j * M + 3 * M * N];
      abs_err += std::abs(A_ij - i_C[i + j * M]);
      sign_err += (A_ij - i_C[i + j * M]);
      nrm += i_C[i + j * M];
      if (A_ij != i_C[i + j * M])
        printf("err loc: <%d %d> cpu:%d cuda:%d\n", i, j, i_C[i + j * M], A_ij);
    }
  }
  printf("%lld %lld %lld\n",abs_err,sign_err,nrm);*/

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  cudaStreamDestroy(stream);
  fprintf(stderr, "%s\n", cudaGetErrorString(cudaGetLastError()));
  return 0;
}
