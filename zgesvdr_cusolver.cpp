
#include <common.hpp>
#include <iostream>

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

  int64_t K = 3 < argc ? std::atoi(argv[3]) : 128;
  int64_t oversampling = 4 < argc ? std::atoi(argv[4]) : 0;
  int32_t power_iter = 5 < argc ? std::atoi(argv[5]) : 0;
  K = std::min(K, N);
 
  std::vector<std::complex<double>> matA(M * N, 0.);
  std::vector<double> S(N);
  matrix_generator<std::complex<double>>(1., M, N).generate_block(512, 512, &matA[0], M);

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
  cusolverDnXgesvdr_bufferSize(cusolverH, params, 'S', 'S', M, N, K, oversampling, power_iter,
    CUDA_C_64F, dA, M, CUDA_R_64F, dS, CUDA_C_64F, dU, M, CUDA_C_64F, dV, N, CUDA_C_64F, &workspaceInBytesOnDevice, &workspaceInBytesOnHost);

  void* hWork = std::malloc(workspaceInBytesOnHost), *dWork;
  cudaMalloc(&dWork, workspaceInBytesOnDevice);

  cusolverDnXgesvdr(cusolverH, params, 'S', 'S', M, N, K, oversampling, power_iter,
    CUDA_C_64F, dA, M, CUDA_R_64F, dS, CUDA_C_64F, dU, M, CUDA_C_64F, dV, N, CUDA_C_64F, dWork, workspaceInBytesOnDevice, hWork, workspaceInBytesOnHost, info);
  cudaDeviceSynchronize();
  int32_t hinfo;
  cudaMemcpy(&hinfo, info, sizeof(int32_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(S.data(), dS, N * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(dA, matA.data(), M * N * sizeof(std::complex<double>), cudaMemcpyHostToDevice);

  double nrm = 0., err = 0.; std::complex<double> minus_one(-1., 0.), one(1., 0.);
  cublasDznrm2_64(cublasH, M * N, (cuDoubleComplex*)dA, int64_t(1), &nrm);
  for (int32_t i = 0; i < K; ++i)
    cublasZdscal(cublasH, M, &S[i], (cuDoubleComplex*)&dU[i * M], 1);
  cublasZgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_C, M, N, K, (cuDoubleComplex*)&minus_one, (const cuDoubleComplex*)dU, M, (const cuDoubleComplex*)dV, N, (cuDoubleComplex*)&one, (cuDoubleComplex*)dA, M);
  cublasDznrm2_64(cublasH, M * N, (cuDoubleComplex*)dA, int64_t(1), &err);

  cudaDeviceSynchronize();
  cudaMemcpy(dA, matA.data(), M * N * sizeof(std::complex<double>), cudaMemcpyHostToDevice);
  err = err / nrm;

  cudaEventRecord(start, stream);
  cusolverDnXgesvdr(cusolverH, params, 'N', 'S', M, N, K, oversampling, power_iter,
    CUDA_C_64F, dA, M, CUDA_R_64F, dS, CUDA_C_64F, dU, M, CUDA_C_64F, dV, N, CUDA_C_64F, dWork, workspaceInBytesOnDevice, hWork, workspaceInBytesOnHost, info);
  cudaEventRecord(stop, stream);

  cudaDeviceSynchronize();
  float kernel_time = 0.0f;
  cudaEventElapsedTime(&kernel_time, start, stop);
  int64_t l = K + oversampling;
  int64_t svd_flops = (4 * (power_iter + 1) * M * N * l) + (l * l * (4 * M + 2 * N));
  printf("CUSOLVER-ZGESVDR [M=%ld,N=%ld] [err=%.12le] [rank=%ld] [kernel=%lf ms] [%lf GFlops]\n",
    M, N, err, K, kernel_time, (1.e-6 * double(svd_flops)) / kernel_time);

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
