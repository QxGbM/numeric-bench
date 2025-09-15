
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <vector>
#include <complex>
#include <iostream>
#include <algorithm>

void make_2D_oscillatory(double w, int32_t sep, int32_t M, int32_t N, std::complex<double>* A, int32_t lda) {
  constexpr int32_t height = 128;
  auto translate_2d = [](int64_t i) { int64_t x = i / height, y = i - height * x; return std::complex<double>(x, y); };
  sep = height * sep + ((M + height - 1) & (~(height - 1)));

  for (int32_t j = 0; j < N; ++j) {
    auto vj = translate_2d(j + sep);
    for (int32_t i = 0; i < M; ++i) {
      auto vi = translate_2d(i);
      double d = std::abs(vi - vj);
      A[uint64_t(i) + uint64_t(j) * uint64_t(lda)] = std::complex<double>(std::cos(w * d) / d, std::sin(w * d) / d);
    }
  }
}

int32_t main(int32_t argc, char* argv[]) {
  auto cu_err = cudaSetDevice(0);
  if (cu_err != cudaSuccess)
  { fprintf(stderr, "%s\n", cudaGetErrorString(cu_err)); return -1; }
  
  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

  cublasHandle_t cublasH;
  cublasCreate(&cublasH);
  cublasSetStream(cublasH, stream);

  cusolverDnHandle_t cusolverH;
  cusolverDnCreate(&cusolverH);
  cusolverDnSetStream(cusolverH, stream);

  cusolverDnParams_t params;
  cusolverDnCreateParams(&params);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  int64_t M = 1 < argc ? std::atoi(argv[1]) : 1024;
  int64_t N = 2 < argc ? std::atoi(argv[2]) : 128;
  N = std::min(M, N);
  double epi = 3 < argc ? std::atof(argv[3]) : 1.e-12;

  std::vector<std::complex<double>> matA(M * N, 0.);
  std::vector<double> S(N);
  make_2D_oscillatory(1., 0, M, N, matA.data(), M);

  std::complex<double>* dA = nullptr, *dU = nullptr, *dV = nullptr;
  double* dS = nullptr;
  int32_t* info = nullptr;
  cudaMalloc((void**)(&dA), M * N * sizeof(std::complex<double>));
  cudaMalloc((void**)(&dU), M * N * sizeof(std::complex<double>));
  cudaMalloc((void**)(&dV), N * N * sizeof(std::complex<double>));
  cudaMalloc((void**)(&dS), N * sizeof(double));
  cudaMalloc((void**)(&info), sizeof(int32_t));
  cudaMemcpy(dA, matA.data(), M * N * sizeof(std::complex<double>), cudaMemcpyHostToDevice);

  size_t workspaceInBytesOnDevice, workspaceInBytesOnHost;
  cusolverDnXgesvd_bufferSize(cusolverH, params, 'S', 'S', M, N,
    CUDA_C_64F, dA, M, CUDA_R_64F, dS, CUDA_C_64F, dU, M, CUDA_C_64F, dV, N, CUDA_C_64F, &workspaceInBytesOnDevice, &workspaceInBytesOnHost);

  void* hWork = std::malloc(workspaceInBytesOnHost), *dWork;
  cudaMalloc(&dWork, workspaceInBytesOnDevice);

  auto status = cusolverDnXgesvd(cusolverH, params, 'S', 'S', M, N,
    CUDA_C_64F, dA, M, CUDA_R_64F, dS, CUDA_C_64F, dU, M, CUDA_C_64F, dV, N, CUDA_C_64F, dWork, workspaceInBytesOnDevice, hWork, workspaceInBytesOnHost, info);
  cudaDeviceSynchronize();
  int32_t hinfo;
  cudaMemcpy(&hinfo, info, sizeof(int32_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(S.data(), dS, N * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(dA, matA.data(), M * N * sizeof(std::complex<double>), cudaMemcpyHostToDevice);

  double s0 = S[0] * epi;
  int64_t rank = 0;
  while (rank < N && s0 <= S[rank]) { ++rank; }
  for (int32_t i = 0; i < rank; ++i)
    cublasZdscal(cublasH, M, &S[i], (cuDoubleComplex*)&dU[i * M], 1);
  
  double nrm = 0., err = 0.; std::complex<double> minus_one(-1., 0.), one(1., 0.);
  cublasDznrm2_64(cublasH, M * N, (cuDoubleComplex*)dA, int64_t(1), &nrm);
  cublasZgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, M, N, rank, (cuDoubleComplex*)&minus_one, (const cuDoubleComplex*)dU, M, (const cuDoubleComplex*)dV, N, (cuDoubleComplex*)&one, (cuDoubleComplex*)dA, M);
  cublasDznrm2_64(cublasH, M * N, (cuDoubleComplex*)dA, int64_t(1), &err);

  cudaDeviceSynchronize();
  cudaMemcpy(dA, matA.data(), M * N * sizeof(std::complex<double>), cudaMemcpyHostToDevice);
  err = err / nrm;

  cudaEventRecord(start, stream);
  cusolverDnXgesvd(cusolverH, params, 'S', 'S', M, N,
    CUDA_C_64F, dA, M, CUDA_R_64F, dS, CUDA_C_64F, dU, M, CUDA_C_64F, dV, N, CUDA_C_64F, dWork, workspaceInBytesOnDevice, hWork, workspaceInBytesOnHost, info);
  cudaEventRecord(stop, stream);

  cudaDeviceSynchronize();
  float milliseconds = 0.0f;
  cudaEventElapsedTime(&milliseconds, start, stop);
  int64_t svd_flops = N * N * (4 * M + 8 * N);
  double gflops = double(svd_flops) * 1.e-6 / milliseconds;
  std::cout << "cusolver-ZGESVD," << M << "," << N << "," << err << "," << rank << "," << milliseconds << "," << gflops << "," << ((hinfo == 0 && status == CUSOLVER_STATUS_SUCCESS) ? "OK" : "ERR") << std::endl;

  cudaFree(dA);
  cudaFree(dU);
  cudaFree(dV);
  cudaFree(dS);
  cudaFree(info);
  cudaFree(dWork);
  std::free(hWork);
  cudaStreamDestroy(stream);
  cublasDestroy(cublasH);
  cusolverDnDestroyParams(params);
  cusolverDnDestroy(cusolverH);

  cu_err = cudaGetLastError();
  if (cu_err != cudaSuccess)
    std::cerr << cudaGetErrorString(cu_err) << std::endl;
  return 0;
}
