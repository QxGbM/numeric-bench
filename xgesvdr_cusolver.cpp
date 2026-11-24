
#include <vendor_common.hpp>
#include <iostream>

template <class T>
int32_t run(int32_t argc, char* argv[]) {
  int64_t M = 2048, N = 2048, K = 1500; int32_t oversampling = 0, power_iter = 0;
  for (int32_t i = 1; i < argc; ++i) {
    if (std::strncmp(argv[i], "M=", 2) == 0) { std::sscanf(argv[i], "M=%ld", &M); }
    else if (std::strncmp(argv[i], "N=", 2) == 0) { std::sscanf(argv[i], "N=%ld", &N); }
    else if (std::strncmp(argv[i], "K=", 2) == 0) { std::sscanf(argv[i], "K=%ld", &K); }
    else if (std::strncmp(argv[i], "oversampling=", 13) == 0) { std::sscanf(argv[i], "oversampling=%d", &oversampling); }
    else if (std::strncmp(argv[i], "piters=", 7) == 0) { std::sscanf(argv[i], "piters=%d", &power_iter); }
    else if (std::strncmp(argv[i], "data=", 5) == 0) {}
    else { std::cerr << "Ignored parameter: " << argv[i] << std::endl; }
  }
  if (M <= 0 || N <= 0 || K <= 0 || oversampling < 0 || power_iter < 0 || oversampling >= std::min(M, N) - 1)
    throw std::invalid_argument("Invalid dimensions, rank, oversampling, or power iterations");
  oversampling = std::max(0, oversampling);
  K = std::min(K, std::min(M, N) - (1 + oversampling));

  LIBRARY_CHECK(cudaSetDevice(0));

  cudaStream_t stream;
  LIBRARY_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamDefault));

  cublasHandle_t cublasH;
  LIBRARY_CHECK(cublasCreate(&cublasH));
  LIBRARY_CHECK(cublasSetStream(cublasH, stream));

  cusolverDnHandle_t cusolverH;
  LIBRARY_CHECK(cusolverDnCreate(&cusolverH));
  LIBRARY_CHECK(cusolverDnSetStream(cusolverH, stream));

  cusolverDnParams_t params;
  LIBRARY_CHECK(cusolverDnCreateParams(&params));

  cudaEvent_t start, stop;
  LIBRARY_CHECK(cudaEventCreate(&start));
  LIBRARY_CHECK(cudaEventCreate(&stop));

  std::vector<T> matA(M * N, 0.);
  matrix_generator<T>(1., M, N).generate_block(512, 512, &matA[0], M);

  T* dA = nullptr, *dU = nullptr, *dV = nullptr;
  double* dS = nullptr; int* info = nullptr;
  LIBRARY_CHECK(cudaMalloc((void**)(&dA), M * N * sizeof(T)));
  LIBRARY_CHECK(cudaMalloc((void**)(&dU), M * K * sizeof(T)));
  LIBRARY_CHECK(cudaMalloc((void**)(&dV), N * K * sizeof(T)));
  LIBRARY_CHECK(cudaMalloc((void**)(&dS), N * sizeof(double)));
  LIBRARY_CHECK(cudaMalloc((void**)(&info), sizeof(int)));
  LIBRARY_CHECK(cudaMemcpy(dA, matA.data(), M * N * sizeof(T), cudaMemcpyHostToDevice));

  size_t workspaceInBytesOnDevice, workspaceInBytesOnHost;
  LIBRARY_CHECK(cusolverDnXgesvdr_bufferSize(cusolverH, params, 'N', 'S', M, N, K, oversampling, power_iter,
    vendor_type<T>, dA, M, CUDA_R_64F, dS, vendor_type<T>, nullptr, M, vendor_type<T>, dV, N, vendor_type<T>, &workspaceInBytesOnDevice, &workspaceInBytesOnHost));
  size_t reconstructionWorkspaceOnDevice, reconstructionWorkspaceOnHost;
  LIBRARY_CHECK(cusolverDnXgesvdr_bufferSize(cusolverH, params, 'S', 'S', M, N, K, oversampling, power_iter,
    vendor_type<T>, dA, M, CUDA_R_64F, dS, vendor_type<T>, dU, M, vendor_type<T>, dV, N, vendor_type<T>, &reconstructionWorkspaceOnDevice, &reconstructionWorkspaceOnHost));
  workspaceInBytesOnDevice = std::max(workspaceInBytesOnDevice, reconstructionWorkspaceOnDevice);
  workspaceInBytesOnHost = std::max(workspaceInBytesOnHost, reconstructionWorkspaceOnHost);

  void* hWork = std::malloc(workspaceInBytesOnHost), *dWork;
  if (workspaceInBytesOnHost && !hWork) throw std::bad_alloc();
  LIBRARY_CHECK(cudaMalloc(&dWork, workspaceInBytesOnDevice));

  LIBRARY_CHECK(cusolverDnXgesvdr(cusolverH, params, 'N', 'S', M, N, K, oversampling, power_iter,
    vendor_type<T>, dA, M, CUDA_R_64F, dS, vendor_type<T>, nullptr, M, vendor_type<T>, dV, N, vendor_type<T>, dWork, workspaceInBytesOnDevice, hWork, workspaceInBytesOnHost, info));
  check_solver_info(info, stream, "GESVDR");
  LIBRARY_CHECK(cudaMemcpy(dA, matA.data(), M * N * sizeof(T), cudaMemcpyHostToDevice));

  double nrm = 0., err = 0.; T minus_one(-1.), one(1.), zero(0.);
  LIBRARY_CHECK(vendor_norm(cublasH, M * N, dA, int64_t(1), &nrm));
  LIBRARY_CHECK(vendor_gemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, M, K, N, &one, dA, M, dV, N, &zero, dU, M));
  LIBRARY_CHECK(vendor_gemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_C, M, N, K, &minus_one, dU, M, dV, N, &one, dA, M));
  LIBRARY_CHECK(vendor_norm(cublasH, M * N, dA, int64_t(1), &err));
  LIBRARY_CHECK(cudaMemcpy(dA, matA.data(), M * N * sizeof(T), cudaMemcpyHostToDevice));
  err = nrm == 0. ? 0. : err / nrm;

  double kernel_time = 0.;
  for (int32_t run = 0; run < kernel_runs; ++run) {
    LIBRARY_CHECK(cudaMemcpyAsync(dA, matA.data(), M * N * sizeof(T), cudaMemcpyHostToDevice, stream));
    LIBRARY_CHECK(cudaEventRecord(start, stream));
    LIBRARY_CHECK(cusolverDnXgesvdr(cusolverH, params, 'N', 'S', M, N, K, oversampling, power_iter,
    vendor_type<T>, dA, M, CUDA_R_64F, dS, vendor_type<T>, nullptr, M, vendor_type<T>, dV, N, vendor_type<T>, dWork, workspaceInBytesOnDevice, hWork, workspaceInBytesOnHost, info));
    LIBRARY_CHECK(cudaEventRecord(stop, stream));
    check_solver_info(info, stream, "GESVDR");
    float milliseconds = 0.f;
    LIBRARY_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    kernel_time += milliseconds;
  }
  kernel_time /= double(kernel_runs);

  // Independently validate the returned U, S, and V. This extra decomposition
  // and its reconstruction are outside the jobu=N benchmark timing.
  LIBRARY_CHECK(cudaMemcpyAsync(dA, matA.data(), M * N * sizeof(T), cudaMemcpyHostToDevice, stream));
  LIBRARY_CHECK(cusolverDnXgesvdr(cusolverH, params, 'S', 'S', M, N, K, oversampling, power_iter,
    vendor_type<T>, dA, M, CUDA_R_64F, dS, vendor_type<T>, dU, M, vendor_type<T>, dV, N, vendor_type<T>, dWork, workspaceInBytesOnDevice, hWork, workspaceInBytesOnHost, info));
  check_solver_info(info, stream, "GESVDR (jobu=S, jobv=S)");

  T* dB = nullptr;
  LIBRARY_CHECK(cudaMalloc((void**)(&dB), M * K * sizeof(T)));
  if constexpr (std::is_same_v<T, double>) {
    LIBRARY_CHECK(cublasDdgmm(cublasH, CUBLAS_SIDE_RIGHT, M, K, dU, M, dS, 1, dB, M));
  } else {
    // ZDGMM takes a complex diagonal; copy real singular values into its real
    // components and leave every imaginary component zero.
    T* dSigma = nullptr;
    LIBRARY_CHECK(cudaMalloc((void**)(&dSigma), K * sizeof(T)));
    LIBRARY_CHECK(cudaMemsetAsync(dSigma, 0, K * sizeof(T), stream));
    LIBRARY_CHECK(cudaMemcpy2DAsync(dSigma, sizeof(T), dS, sizeof(double), sizeof(double), K,
      cudaMemcpyDeviceToDevice, stream));
    LIBRARY_CHECK(cublasZdgmm(cublasH, CUBLAS_SIDE_RIGHT, M, K,
      reinterpret_cast<const cuDoubleComplex*>(dU), M, reinterpret_cast<const cuDoubleComplex*>(dSigma), 1,
      reinterpret_cast<cuDoubleComplex*>(dB), M));
    LIBRARY_CHECK(cudaFree(dSigma));
  }
  LIBRARY_CHECK(cudaMemcpyAsync(dA, matA.data(), M * N * sizeof(T), cudaMemcpyHostToDevice, stream));
  LIBRARY_CHECK(vendor_gemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_C, M, N, K, &minus_one, dB, M, dV, N, &one, dA, M));
  double err_usvh = 0.;
  LIBRARY_CHECK(vendor_norm(cublasH, M * N, dA, int64_t(1), &err_usvh));
  err_usvh = nrm == 0. ? 0. : err_usvh / nrm;
  LIBRARY_CHECK(cudaFree(dB));

  int64_t l = K + oversampling;
  double svd_flops = randomized_svd_flops(double(M), double(N), double(l), double(power_iter), vendor_flop_factor<T>);
  printf("CUSOLVER-%cGESVDR [M=%ld,N=%ld] [err_AVVH=%.12le] [err_USVH=%.12le] [rank=%ld] [kernel=%lf ms] [%lf GFlops]\n",
    vendor_precision<T>, M, N, err, err_usvh, K, kernel_time, (1.e-6 * double(svd_flops)) / kernel_time);

  LIBRARY_CHECK(cudaEventDestroy(start));
  LIBRARY_CHECK(cudaEventDestroy(stop));
  LIBRARY_CHECK(cudaFree(dA));
  LIBRARY_CHECK(cudaFree(dU));
  LIBRARY_CHECK(cudaFree(dV));
  LIBRARY_CHECK(cudaFree(dS));
  LIBRARY_CHECK(cudaFree(info));
  LIBRARY_CHECK(cudaFree(dWork));
  std::free(hWork);
  LIBRARY_CHECK(cudaStreamDestroy(stream));
  LIBRARY_CHECK(cublasDestroy(cublasH));
  LIBRARY_CHECK(cusolverDnDestroyParams(params));
  LIBRARY_CHECK(cusolverDnDestroy(cusolverH));

  LIBRARY_CHECK(cudaGetLastError());
  return 0;
}

int32_t main(int32_t argc, char* argv[]) try {
  char precision = 'D';
  for (int32_t i = 1; i < argc; ++i)
    if (std::strncmp(argv[i], "data=", 5) == 0) precision = argv[i][5];
  switch (precision) {
    case 'D': return run<double>(argc, argv);
    case 'Z': return run<std::complex<double>>(argc, argv);
    default: throw std::invalid_argument("Supported precisions: data=D or data=Z");
  }
} catch (const std::exception& error) {
  std::cerr << error.what() << std::endl;
  return EXIT_FAILURE;
}
