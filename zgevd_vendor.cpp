
#include <common.hpp>
#include <iostream>

int32_t main(int32_t argc, char* argv[]) {
  std::string file, ref;
  int64_t gM = 2048, N = 2048, K = 1500, mb = 512;

  for (int32_t i = 1; i < argc; ++i) {
    if (std::strncmp(argv[i], "M=", 2) == 0) { std::sscanf(argv[i], "M=%ld", &gM); }
    else if (std::strncmp(argv[i], "N=", 2) == 0) { std::sscanf(argv[i], "N=%ld", &N); }
    else if (std::strncmp(argv[i], "K=", 2) == 0) { std::sscanf(argv[i], "K=%ld", &K); }
    else if (std::strncmp(argv[i], "mb=", 3) == 0) { std::sscanf(argv[i], "mb=%ld", &mb); }
    else if (std::strncmp(argv[i], "file=", 5) == 0) { file.resize(std::strlen(argv[i])); std::sscanf(argv[i], "file=%s", file.data()); }
    else if (std::strncmp(argv[i], "ref=", 4) == 0) { ref.resize(std::strlen(argv[i])); std::sscanf(argv[i], "ref=%s", ref.data()); }
    else { std::cerr << "Ignored parameter: " << argv[i] << std::endl; }
  }
  N = std::min(gM, N); K = std::min(N, K);

  int32_t world_rank, world_size, local_rank; ncclUniqueId id;
  //__bootstrap_mpi(world_rank, world_size, local_rank, id);
  __bootstrap_posix_fork(local_rank, world_size, id); world_rank = local_rank;

  int32_t device_count = 0; cudaGetDeviceCount(&device_count);
  auto cu_err = cudaSetDevice(1 < device_count ? local_rank : 0);
  cudaDeviceReset();
  if (cu_err != cudaSuccess)
  { std::cerr << cudaGetErrorString(cu_err) << std::endl; return -1; }

  int32_t grid_row = world_rank, tile_m = world_size;
  int64_t lM = mb * (gM / (mb * tile_m));
  lM += std::max(int64_t(0), std::min(mb, gM - lM * tile_m - mb * grid_row));

  std::vector<std::complex<double>> matA(lM * N);
  if (!file.empty())
    matrix_from_row_major_csv(gM, N, mb, 512, matA.data(), lM, file, grid_row, 0, tile_m, 1);
  else
    matrix_generator<std::complex<double>>(gM, N).generate_block(1., mb, 512, &matA[0], lM, grid_row, 0, tile_m, 1);

  std::complex<double>* d_A = nullptr, *d_V = nullptr, *d_B = nullptr; double* d_S = nullptr;
  void* evd_work_device = nullptr, *evd_work_host = nullptr;
  cudaMalloc((void**)(&d_A), lM * N * sizeof(std::complex<double>));
  cudaMalloc((void**)(&d_V), N * N * sizeof(std::complex<double>));
  cudaMalloc((void**)(&d_B), lM * K * sizeof(std::complex<double>));
  cudaMalloc((void**)(&d_S), N * sizeof(double));

  hyacinHandle_t handle;
  ncclComm_t comm;

  hyacinCreate(&handle, 0);
  ncclCommInitRank(&comm, tile_m, id, grid_row);
  size_t workspaceInBytesOnDevice = 0, workspaceInBytesOnHost = 0;
  cusolverDnXsyevd_bufferSize(handle.cusolverHandle, handle.cusolverParams, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
    N, CUDA_C_64F, d_V, N, CUDA_R_64F, d_S, CUDA_C_64F, &workspaceInBytesOnDevice, &workspaceInBytesOnHost);

  cudaMalloc((void**)(&evd_work_device), workspaceInBytesOnDevice);
  evd_work_host = std::malloc(workspaceInBytesOnHost);
  cudaMemcpy(d_A, matA.data(), lM * N * sizeof(std::complex<double>), cudaMemcpyHostToDevice);

  cudaEvent_t start, stop, cstart, cstop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventCreate(&cstart);
  cudaEventCreate(&cstop);

  int32_t* d_barrier = nullptr;
  cudaMalloc((void**)(&d_barrier), sizeof(double2));
  cudaMemset(d_barrier, 0xDEADBEEF, sizeof(double2));

  cuDoubleComplex one = make_cuDoubleComplex(1., 0.), zero = make_cuDoubleComplex(0., 0.);
  int64_t col_start = N - K;
  cublasZherk(handle.cublasHandle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_C, N, lM, (double*)&one, (cuDoubleComplex*)d_A, lM, (double*)&zero, (cuDoubleComplex*)d_V, N);
  ncclAllReduce(d_V, d_V, int64_t(2) * N * N, ncclDouble, ncclSum, comm, handle.cudaStream);
  cusolverDnXsyevd(handle.cusolverHandle, handle.cusolverParams, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
    N, CUDA_C_64F, d_V, N, CUDA_R_64F, d_S, CUDA_C_64F, evd_work_device, workspaceInBytesOnDevice, evd_work_host, workspaceInBytesOnHost, nullptr);
  cublasZgemm(handle.cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, lM, K, N, &one, (cuDoubleComplex*)d_A, lM, (cuDoubleComplex*)&d_V[col_start * N], N, &zero, (cuDoubleComplex*)d_B, lM);
  cudaStreamSynchronize(handle.cudaStream);

  std::vector<std::complex<double>> matU(lM * K), matV(K * N);
  cudaMemcpy(matU.data(), d_B, lM * K * sizeof(std::complex<double>), cudaMemcpyDeviceToHost);
  cudaMemcpy(matV.data(), &d_V[col_start * N], K * N * sizeof(std::complex<double>), cudaMemcpyDeviceToHost);

  double ret[2]{ check_answer_svd(lM, N, K, &matU[0], lM, &matV[0], N, &matA[0], lM), fnorm(lM, N, &matA[0], lM) };
  cudaMemcpy(d_barrier, &ret, sizeof(double2), cudaMemcpyHostToDevice);
  ncclAllReduce(d_barrier, d_barrier, 2, ncclDouble, ncclSum, comm, handle.cudaStream);
  cudaStreamSynchronize(handle.cudaStream);
  cudaMemcpy(&ret, d_barrier, sizeof(double2), cudaMemcpyDeviceToHost);
  cudaMemset(d_barrier, 0xDEADBEEF, sizeof(double2));
  double err = std::sqrt(ret[0] / ret[1]), max_elem_err = std::numeric_limits<double>::quiet_NaN();

  if (!ref.empty() && grid_row == 0) {
    std::vector<std::complex<double>> ref_V(N * int64_t(K));
    matrix_from_row_major_csv(N, K, 512, 512, ref_V.data(), N, ref);
    max_elem_err = max_elementwise_relerr(N, K, ref_V.data(), N, matV.data(), N);
  }

  cudaMemcpy(d_A, matA.data(), lM * N * sizeof(std::complex<double>), cudaMemcpyHostToDevice);
  ncclAllReduce(d_barrier, d_barrier, 1, ncclInt32, ncclMin, comm, handle.cudaStream);
  cudaStreamSynchronize(handle.cudaStream);
  kernel_time = comm_time = 0.;
  cudaEventRecord(start, handle.cudaStream);

  cublasZherk(handle.cublasHandle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_C, N, lM, (double*)&one, (cuDoubleComplex*)d_A, lM, (double*)&zero, (cuDoubleComplex*)d_V, N);
  cudaEventRecord(cstart, handle.cudaStream);
  ncclAllReduce(d_V, d_V, int64_t(2) * N * N, ncclDouble, ncclSum, comm, handle.cudaStream);
  cudaEventRecord(cstop, handle.cudaStream);
  cusolverDnXsyevd(handle.cusolverHandle, handle.cusolverParams, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
    N, CUDA_C_64F, d_V, N, CUDA_R_64F, d_S, CUDA_C_64F, evd_work_device, workspaceInBytesOnDevice, evd_work_host, workspaceInBytesOnHost, nullptr);
  cublasZgemm(handle.cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, lM, K, N, &one, (cuDoubleComplex*)d_A, lM, (cuDoubleComplex*)&d_V[col_start * N], N, &zero, (cuDoubleComplex*)d_B, lM);

  ncclAllReduce(d_barrier, d_barrier, 1, ncclInt32, ncclMin, comm, handle.cudaStream);
  cudaEventRecord(stop, handle.cudaStream);
  cudaStreamSynchronize(handle.cudaStream);
  float milliseconds = 0.0f; cudaEventElapsedTime(&milliseconds, start, stop);
  float seg1 = 0.0f, seg2 = 0.0f, seg3 = 0.0f; cudaEventElapsedTime(&seg1, start, cstart); cudaEventElapsedTime(&seg2, cstart, cstop); cudaEventElapsedTime(&seg3, cstop, stop);
  kernel_time = double(seg1) + double(seg3); comm_time = double(seg2);

  cudaFree(d_barrier);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaEventDestroy(cstart);
  cudaEventDestroy(cstop);
  hyacinDestroy(handle);
  ncclCommDestroy(comm);

  std::vector<double> vecS(K);
  cudaMemcpy(vecS.data(), d_S, K * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(d_A);
  cudaFree(d_V);
  cudaFree(d_B);
  cudaFree(d_S);
  cudaFree(evd_work_device);
  std::free(evd_work_host);

  double duration = double(milliseconds);
  printf("Z-SVD#%d [M=%ld,N=%ld,K=%ld] [err=%.12le] [max_elem_err=%.12le] [tts=%lf ms] [kernel=%lf ms] [comm=%lf ms]\n",
    grid_row, gM, N, K, err, max_elem_err, duration, kernel_time, comm_time);

  cu_err = cudaGetLastError();
  if (cu_err != cudaSuccess)
    std::cerr << cudaGetErrorString(cu_err) << std::endl;
  return 0;
}
