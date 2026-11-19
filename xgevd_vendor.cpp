
#include <vendor_common.hpp>
#include <iostream>

template <class T>
int32_t run(int32_t argc, char* argv[]) {
  int64_t gM = 2048, N = 2048, K = 1500, mb = 512;
  int32_t ranks = 1;

  for (int32_t i = 1; i < argc; ++i) {
    if (std::strncmp(argv[i], "M=", 2) == 0) { std::sscanf(argv[i], "M=%ld", &gM); }
    else if (std::strncmp(argv[i], "N=", 2) == 0) { std::sscanf(argv[i], "N=%ld", &N); }
    else if (std::strncmp(argv[i], "K=", 2) == 0) { std::sscanf(argv[i], "K=%ld", &K); }
    else if (std::strncmp(argv[i], "ranks=", 6) == 0) { std::sscanf(argv[i], "ranks=%d", &ranks); }
    else if (std::strncmp(argv[i], "mb=", 3) == 0) { std::sscanf(argv[i], "mb=%ld", &mb); }
    else if (std::strncmp(argv[i], "data=", 5) == 0) {}
    else { std::cerr << "Ignored parameter: " << argv[i] << std::endl; }
  }
  if (gM <= 0 || N <= 0 || K <= 0 || mb <= 0) throw std::invalid_argument("Require positive M,N,K,mb");
  N = std::min(gM, N); K = std::min(N, K);

  int32_t world_rank, world_size, local_rank; ncclUniqueId id;
  enable_nccl_multi_rank_gpu();
  bootstrap_posix_fork(world_rank, local_rank, world_size, id, ranks);

  int32_t device_count = 0; LIBRARY_CHECK(cudaGetDeviceCount(&device_count));
  LIBRARY_CHECK(cudaSetDevice(1 < device_count ? local_rank % device_count : 0));
  LIBRARY_CHECK(cudaDeviceReset());

  int32_t grid_row = world_rank, tile_m = world_size;
  int64_t lM = mb * (gM / (mb * tile_m));
  lM += std::max(int64_t(0), std::min(mb, gM - lM * tile_m - mb * grid_row));

  std::vector<T> matA(lM * N);
  matrix_generator<T>(1., gM, N).generate_block(mb, 512, &matA[0], lM, grid_row, 0, tile_m, 1);

  T* d_A = nullptr, *d_V = nullptr, *d_B = nullptr; double* d_S = nullptr;
  void* evd_work_device = nullptr, *evd_work_host = nullptr;
  LIBRARY_CHECK(cudaMalloc((void**)(&d_A), lM * N * sizeof(T)));
  LIBRARY_CHECK(cudaMalloc((void**)(&d_V), N * N * sizeof(T)));
  LIBRARY_CHECK(cudaMalloc((void**)(&d_B), lM * K * sizeof(T)));
  LIBRARY_CHECK(cudaMalloc((void**)(&d_S), N * sizeof(double)));

  cudaStream_t stream;
  cublasHandle_t cublasH;
  cusolverDnHandle_t cusolverH;
  cusolverDnParams_t params;
  ncclComm_t comm;

  LIBRARY_CHECK(cudaStreamCreate(&stream));
  LIBRARY_CHECK(cublasCreate(&cublasH));
  LIBRARY_CHECK(cublasSetStream(cublasH, stream));
  LIBRARY_CHECK(cusolverDnCreate(&cusolverH));
  LIBRARY_CHECK(cusolverDnSetStream(cusolverH, stream));
  LIBRARY_CHECK(cusolverDnCreateParams(&params));
  LIBRARY_CHECK(ncclCommInitRank(&comm, tile_m, id, grid_row));

  size_t workspaceInBytesOnDevice = 0, workspaceInBytesOnHost = 0;
  LIBRARY_CHECK(cusolverDnXsyevd_bufferSize(cusolverH, params, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
    N, vendor_type<T>, d_V, N, CUDA_R_64F, d_S, vendor_type<T>, &workspaceInBytesOnDevice, &workspaceInBytesOnHost));

  LIBRARY_CHECK(cudaMalloc((void**)(&evd_work_device), workspaceInBytesOnDevice));
  evd_work_host = std::malloc(workspaceInBytesOnHost);
  if (workspaceInBytesOnHost && !evd_work_host) throw std::bad_alloc();
  LIBRARY_CHECK(cudaMemcpy(d_A, matA.data(), lM * N * sizeof(T), cudaMemcpyHostToDevice));

  cudaEvent_t start, stop, cstart, cstop;
  LIBRARY_CHECK(cudaEventCreate(&start));
  LIBRARY_CHECK(cudaEventCreate(&stop));
  LIBRARY_CHECK(cudaEventCreate(&cstart));
  LIBRARY_CHECK(cudaEventCreate(&cstop));

  int* info = nullptr; LIBRARY_CHECK(cudaMalloc((void**)&info, sizeof(int)));
  int32_t* d_barrier = nullptr; LIBRARY_CHECK(cudaMalloc((void**)(&d_barrier), 4 * sizeof(double)));
  LIBRARY_CHECK(cudaMemset(d_barrier, 0, 4 * sizeof(double)));
  T one(1.), zero(0.);
  int64_t col_start = N - K;
  LIBRARY_CHECK(cudaMemsetAsync(d_V, 0, N * N * sizeof(T), stream));
  LIBRARY_CHECK(vendor_gram(cublasH, N, lM, d_A, lM, d_V));
  LIBRARY_CHECK(ncclAllReduce(d_V, d_V, int64_t(sizeof(T) / sizeof(double)) * N * N, ncclDouble, ncclSum, comm, stream));
  LIBRARY_CHECK(cusolverDnXsyevd(cusolverH, params, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
    N, vendor_type<T>, d_V, N, CUDA_R_64F, d_S, vendor_type<T>, evd_work_device, workspaceInBytesOnDevice, evd_work_host, workspaceInBytesOnHost, info));
  check_solver_info(info, stream, "SYEVD/HEEVD");
  LIBRARY_CHECK(vendor_gemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, lM, K, N, &one, d_A, lM, &d_V[col_start * N], N, &zero, d_B, lM));
  LIBRARY_CHECK(cudaStreamSynchronize(stream));

  std::vector<T> matU(lM * K), matV(K * N);
  LIBRARY_CHECK(cudaMemcpy(matU.data(), d_B, lM * K * sizeof(T), cudaMemcpyDeviceToHost));
  LIBRARY_CHECK(cudaMemcpy(matV.data(), &d_V[col_start * N], K * N * sizeof(T), cudaMemcpyDeviceToHost));

  LIBRARY_CHECK(cudaMemcpy(d_A, matA.data(), lM * N * sizeof(T), cudaMemcpyHostToDevice));
  LIBRARY_CHECK(ncclAllReduce(d_barrier, d_barrier, 1, ncclInt32, ncclMin, comm, stream));
  LIBRARY_CHECK(cudaStreamSynchronize(stream));
  kernel_time = comm_time = 0.;
  double duration = 0.;
  for (int32_t run = 0; run < kernel_runs; ++run) {
    LIBRARY_CHECK(cudaEventRecord(start, stream));

    LIBRARY_CHECK(vendor_gram(cublasH, N, lM, d_A, lM, d_V));
    LIBRARY_CHECK(cudaEventRecord(cstart, stream));
    LIBRARY_CHECK(ncclAllReduce(d_V, d_V, int64_t(sizeof(T) / sizeof(double)) * N * N, ncclDouble, ncclSum, comm, stream));
    LIBRARY_CHECK(cudaEventRecord(cstop, stream));
    LIBRARY_CHECK(cusolverDnXsyevd(cusolverH, params, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
    N, vendor_type<T>, d_V, N, CUDA_R_64F, d_S, vendor_type<T>, evd_work_device, workspaceInBytesOnDevice, evd_work_host, workspaceInBytesOnHost, info));
    LIBRARY_CHECK(vendor_gemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, lM, K, N, &one, d_A, lM, &d_V[col_start * N], N, &zero, d_B, lM));

    LIBRARY_CHECK(ncclAllReduce(d_barrier, d_barrier, 1, ncclInt32, ncclMin, comm, stream));
    LIBRARY_CHECK(cudaEventRecord(stop, stream));
    LIBRARY_CHECK(cudaStreamSynchronize(stream));
    float milliseconds = 0.0f; LIBRARY_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    float seg1 = 0.0f, seg2 = 0.0f, seg3 = 0.0f; LIBRARY_CHECK(cudaEventElapsedTime(&seg1, start, cstart)); LIBRARY_CHECK(cudaEventElapsedTime(&seg2, cstart, cstop)); LIBRARY_CHECK(cudaEventElapsedTime(&seg3, cstop, stop));
    kernel_time += double(seg1) + double(seg3); comm_time += double(seg2);
    duration += double(milliseconds);
    check_solver_info(info, stream, "SYEVD/HEEVD");
  }
  kernel_time /= double(kernel_runs); comm_time /= double(kernel_runs);
  duration /= double(kernel_runs);

  double ret[4]{ kernel_time, comm_time, vendor_check_answer_svd(lM, N, K, &matU[0], lM, &matV[0], N, &matA[0], lM), vendor_fnorm(lM, N, &matA[0], lM) };
  LIBRARY_CHECK(cudaMemcpy(d_barrier, ret, 4 * sizeof(double), cudaMemcpyHostToDevice));
  LIBRARY_CHECK(ncclAllReduce(d_barrier, d_barrier, 4, ncclDouble, ncclSum, comm, stream));
  LIBRARY_CHECK(cudaStreamSynchronize(stream));
  LIBRARY_CHECK(cudaMemcpy(ret, d_barrier, 4 * sizeof(double), cudaMemcpyDeviceToHost));
  kernel_time = ret[0]; comm_time = ret[1];
  double err = ret[3] == 0. ? std::numeric_limits<double>::quiet_NaN() : std::sqrt(ret[2] / ret[3]);

  LIBRARY_CHECK(cudaFree(info));
  LIBRARY_CHECK(cudaFree(d_barrier));
  LIBRARY_CHECK(cudaEventDestroy(start));
  LIBRARY_CHECK(cudaEventDestroy(stop));
  LIBRARY_CHECK(cudaEventDestroy(cstart));
  LIBRARY_CHECK(cudaEventDestroy(cstop));
  LIBRARY_CHECK(cudaStreamDestroy(stream));
  LIBRARY_CHECK(cublasDestroy(cublasH));
  LIBRARY_CHECK(cusolverDnDestroy(cusolverH));
  LIBRARY_CHECK(cusolverDnDestroyParams(params));
  LIBRARY_CHECK(ncclCommDestroy(comm));

  LIBRARY_CHECK(cudaFree(d_A));
  LIBRARY_CHECK(cudaFree(d_V));
  LIBRARY_CHECK(cudaFree(d_B));
  LIBRARY_CHECK(cudaFree(d_S));
  LIBRARY_CHECK(cudaFree(evd_work_device));
  std::free(evd_work_host);

  printf("%c-SVD#%d [M=%ld,N=%ld,K=%ld] [err=%.12le] [tts=%lf ms] [kernel=%lf ms] [comm=%lf ms]\n",
    vendor_precision<T>, grid_row, gM, N, K, err, duration, kernel_time / double(tile_m), comm_time / double(tile_m));

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
