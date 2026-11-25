
#include <vendor_common.hpp>
#include <iostream>

template <class T>
int32_t run(int32_t argc, char* argv[]) {
  int64_t M = 2048, N = 2048; double eps = 1.e-12;
  for (int32_t i = 1; i < argc; ++i) {
    if (std::strncmp(argv[i], "M=", 2) == 0) { std::sscanf(argv[i], "M=%ld", &M); }
    else if (std::strncmp(argv[i], "N=", 2) == 0) { std::sscanf(argv[i], "N=%ld", &N); }
    else if (std::strncmp(argv[i], "eps=", 4) == 0) { std::sscanf(argv[i], "eps=%le", &eps); }
    else if (std::strncmp(argv[i], "data=", 5) == 0) {}
    else { std::cerr << "Ignored parameter: " << argv[i] << std::endl; }
  }
  if (M <= 0 || N <= 0 || eps < 0. || !std::isfinite(eps)) throw std::invalid_argument("Require M,N > 0 and finite eps >= 0");
  N = std::min(M, N);

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
  LIBRARY_CHECK(cudaMalloc((void**)(&dU), M * N * sizeof(T)));
  LIBRARY_CHECK(cudaMalloc((void**)(&dV), N * N * sizeof(T)));
  LIBRARY_CHECK(cudaMalloc((void**)(&dS), N * sizeof(double)));
  LIBRARY_CHECK(cudaMalloc((void**)(&info), sizeof(int)));
  LIBRARY_CHECK(cudaMemcpy(dA, matA.data(), M * N * sizeof(T), cudaMemcpyHostToDevice));

  size_t workspaceInBytesOnDevice, workspaceInBytesOnHost;
  LIBRARY_CHECK(cusolverDnXgesvd_bufferSize(cusolverH, params, 'N', 'S', M, N,
    vendor_type<T>, dA, M, CUDA_R_64F, dS, vendor_type<T>, nullptr, M, vendor_type<T>, dV, N, vendor_type<T>, &workspaceInBytesOnDevice, &workspaceInBytesOnHost));

  void* hWork = std::malloc(workspaceInBytesOnHost), *dWork;
  if (workspaceInBytesOnHost && !hWork) throw std::bad_alloc();
  LIBRARY_CHECK(cudaMalloc(&dWork, workspaceInBytesOnDevice));

  LIBRARY_CHECK(cusolverDnXgesvd(cusolverH, params, 'N', 'S', M, N,
    vendor_type<T>, dA, M, CUDA_R_64F, dS, vendor_type<T>, nullptr, M, vendor_type<T>, dV, N, vendor_type<T>, dWork, workspaceInBytesOnDevice, hWork, workspaceInBytesOnHost, info));
  check_solver_info(info, stream, "GESVD");

  std::vector<double> S(N);
  LIBRARY_CHECK(cudaMemcpy(S.data(), dS, N * sizeof(double), cudaMemcpyDeviceToHost));
  std::transform_inclusive_scan(S.rbegin(), S.rend(), S.rbegin(), std::plus<double>(), [](double x) { return x * x; });
  int64_t rank = 0;
  if (std::numeric_limits<double>::min() <= S[0]) { double s0 = S[0] * std::pow(eps, 2); rank = std::distance(&S[0], std::find_if(&S[0], &S[N], [=](double s) { return s < s0; })); }

  LIBRARY_CHECK(cudaMemcpy(dA, matA.data(), M * N * sizeof(T), cudaMemcpyHostToDevice));
  double nrm = 0., err = 0.; T minus_one(-1.), one(1.), zero(0.);
  LIBRARY_CHECK(vendor_norm(cublasH, M * N, dA, int64_t(1), &nrm));
  LIBRARY_CHECK(vendor_gemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_C, M, rank, N, &one, dA, M, dV, N, &zero, dU, M));
  LIBRARY_CHECK(vendor_gemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, M, N, rank, &minus_one, dU, M, dV, N, &one, dA, M));
  LIBRARY_CHECK(vendor_norm(cublasH, M * N, dA, int64_t(1), &err));
  LIBRARY_CHECK(cudaMemcpy(dA, matA.data(), M * N * sizeof(T), cudaMemcpyHostToDevice));
  err = nrm == 0. ? 0. : err / nrm;

  double kernel_time = 0.;
  for (int32_t run = 0; run < kernel_runs; ++run) {
    LIBRARY_CHECK(cudaMemcpyAsync(dA, matA.data(), M * N * sizeof(T), cudaMemcpyHostToDevice, stream));
    LIBRARY_CHECK(cudaEventRecord(start, stream));
    LIBRARY_CHECK(cusolverDnXgesvd(cusolverH, params, 'N', 'S', M, N,
    vendor_type<T>, dA, M, CUDA_R_64F, dS, vendor_type<T>, nullptr, M, vendor_type<T>, dV, N, vendor_type<T>, dWork, workspaceInBytesOnDevice, hWork, workspaceInBytesOnHost, info));
    LIBRARY_CHECK(cudaEventRecord(stop, stream));
    check_solver_info(info, stream, "GESVD");
    float milliseconds = 0.f;
    LIBRARY_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    kernel_time += milliseconds;
  }
  kernel_time /= double(kernel_runs);
  double svd_flops = gesvd_flops(double(M), double(N), vendor_flop_factor<T>);
  printf("CUSOLVER-%cGESVD [M=%ld,N=%ld] [err=%.12le] [rank=%ld] [kernel=%lf ms] [%lf GFlops]\n",
    vendor_precision<T>, M, N, err, rank, kernel_time, (1.e-6 * double(svd_flops)) / kernel_time);

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
