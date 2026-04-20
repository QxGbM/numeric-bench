
#include <common.hpp>
#include <cstdlib>
#include <iostream>
#include <chrono>

template <class T, class R> inline void run(char prec, int64_t gM, int64_t gN, int64_t K, int64_t mb, int64_t nb, char algo, double epi, int32_t grid_row, int32_t grid_col, int32_t tile_m, int32_t tile_n, ncclUniqueId id, const std::string& file) {
  int64_t gK = K * tile_n;
  int64_t lM = mb * (gM / (mb * tile_m));
  int64_t lN = nb * (gN / (nb * tile_n));
  lM += std::max(int64_t(0), std::min(mb, gM - lM * tile_m - mb * grid_row));
  lN += std::max(int64_t(0), std::min(nb, gN - lN * tile_n - nb * grid_col));
  
  std::vector<T> matA(lM * lN);
  if (!file.empty())
    matrix_from_row_major_csv(gM, gN, mb, nb, matA.data(), lM, file, grid_row, grid_col, tile_m, tile_n);
  else
    matrix_generator<T>(gM, gN).generate_block(1., mb, nb, &matA[0], lM, grid_row, grid_col, tile_m, tile_n);

  T* d_A = nullptr, *d_V = nullptr; R* d_S = nullptr;
  cudaMalloc((void**)(&d_A), lM * std::max(gK, lN) * sizeof(T));
  cudaMalloc((void**)(&d_V), K * lN * sizeof(T));
  cudaMalloc((void**)(&d_S), K * sizeof(R));
  cudaMemcpy(d_A, matA.data(), lM * lN * sizeof(T), cudaMemcpyHostToDevice);

  /* Timed region start */
  auto host_start = std::chrono::high_resolution_clock::now();

  hyacinHandle_t handle;
  ncclComm_t comm, comm_row, comm_col;

  hyacinCreate(&handle, 1);
  ncclCommInitRank(&comm, tile_m * tile_n, id, grid_row + grid_col * tile_m);
  ncclCommSplit(comm, grid_row, grid_col, &comm_row, nullptr);
  ncclCommSplit(comm, grid_col, grid_row, &comm_col, nullptr);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  int32_t* d_barrier = nullptr;
  int32_t r1, r2, N2, offset;
  if (time_kernel) {
    cudaMalloc((void**)(&d_barrier), sizeof(int32_t));
    cudaMemset(d_barrier, 0xDEADBEEF, sizeof(int32_t));
    r1 = svd_fit_transform_1dr(handle, comm_col, algo, epi, lM, gM, lN, K, d_A, lM, d_S, d_V, lN, lN);
    std::tie(N2, offset) = allgatherv_1dc(handle, comm_row, lM, r1, d_A, lM);
    r2 = svd_fit_transform_1dr(handle, comm_col, algo, epi, lM, gM, N2, K, d_A, lM, d_S, d_V, lN, lN, r1, offset);
    cudaMemcpy(d_A, matA.data(), lM * lN * sizeof(T), cudaMemcpyHostToDevice);
    ncclAllReduce(d_barrier, d_barrier, 1, ncclInt32, ncclMin, comm, handle.cudaStream);
    cudaStreamSynchronize(handle.cudaStream);
    kernel_time = comm_time = 0.;
  }
  cudaEventRecord(start, handle.cudaStream);

  r1 = svd_fit_transform_1dr(handle, comm_col, algo, epi, lM, gM, lN, K, d_A, lM, d_S, d_V, lN, lN);
  std::tie(N2, offset) = allgatherv_1dc(handle, comm_row, lM, r1, d_A, lM);
  r2 = svd_fit_transform_1dr(handle, comm_col, algo, epi, lM, gM, N2, K, d_A, lM, d_S, d_V, lN, lN, r1, offset);

  if (time_kernel)
    ncclAllReduce(d_barrier, d_barrier, 1, ncclInt32, ncclMin, comm, handle.cudaStream);
  cudaEventRecord(stop, handle.cudaStream);
  cudaStreamSynchronize(handle.cudaStream);
  float milliseconds = 0.0f; cudaEventElapsedTime(&milliseconds, start, stop);

  if (time_kernel)
    cudaFree(d_barrier);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  hyacinDestroy(handle);
  ncclCommDestroy(comm);
  ncclCommDestroy(comm_row);
  ncclCommDestroy(comm_col);

  /* Timed region end */
  auto host_end = std::chrono::high_resolution_clock::now();

  std::vector<T> matU(lM * K), matV(K * lN); std::vector<R> vecS(K);
  cudaMemcpy(matU.data(), d_A, lM * K * sizeof(T), cudaMemcpyDeviceToHost);
  cudaMemcpy(matV.data(), d_V, K * lN * sizeof(T), cudaMemcpyDeviceToHost);
  cudaMemcpy(vecS.data(), d_S, K * sizeof(R), cudaMemcpyDeviceToHost);
  cudaFree(d_A);
  cudaFree(d_V);
  cudaFree(d_S);

  /*double* arr = nullptr; cudaMalloc((void**)&arr, 2 * sizeof(double));
  double ret[2]{ check_answer_svd(lM, lN, r2, &matU[0], lM, &matV[0], lN, &matA[0], lM), fnorm(lM, lN, &matA[0], lM) };
  cudaMemcpy(arr, &ret, sizeof(double2), cudaMemcpyHostToDevice);
  ncclAllReduce(arr, arr, 2, ncclDouble, ncclSum, comm, stream);
  cudaStreamSynchronize(stream);
  cudaMemcpy(&ret, arr, sizeof(double2), cudaMemcpyDeviceToHost);
  cudaFree(arr); cudaStreamDestroy(stream); ncclCommDestroy(comm);
  double err = std::sqrt(ret[0] / ret[1]);*/
  double err = std::sqrt(check_answer_svd(lM, lN, r2, &matU[0], lM, &matV[0], lN, &matA[0], lM) / fnorm(lM, lN, &matA[0], lM));

  std::chrono::duration<double, std::milli> host_wtime = host_end - host_start;
  double duration = time_kernel ? double(milliseconds) : host_wtime.count();
  int64_t flops = ((int64_t(gM) + int64_t(gN)) * int64_t(r2) * int64_t(2)) + (int64_t(gM) * int64_t(gN) * int64_t(r2) * int64_t(4));
  double gflops = double(flops) * 1.e-6 / double(duration);
  if (grid_row == 0 && grid_col == 0)
    printf("%lf %lf\n", kernel_time, comm_time);

  printf("%c-SVD#(%d,%d),%ld,%ld,%.1le,%.12le,%d,%d,%lf,%lf\n", prec, grid_row, grid_col, gM, gN, epi, err, r1, r2, duration, gflops);
}

int32_t main(int32_t argc, char* argv[]) {
  char prec = 'D', algo = 'A'; std::string file;
  int32_t tile_m = 1, tile_n = 1;
  int64_t gM = 2048, gN = 2048, K = 2048, mb = 512, nb = 512;
  double epi = 1.e-12;

  for (int32_t i = 1; i < argc; ++i) {
    if (std::strncmp(argv[i], "M=", 2) == 0) { std::sscanf(argv[i], "M=%ld", &gM); }
    else if (std::strncmp(argv[i], "N=", 2) == 0) { std::sscanf(argv[i], "N=%ld", &gN); }
    else if (std::strncmp(argv[i], "K=", 2) == 0) { std::sscanf(argv[i], "K=%ld", &K); }
    else if (std::strncmp(argv[i], "data=", 5) == 0) { std::sscanf(argv[i], "data=%c", &prec); }
    else if (std::strncmp(argv[i], "epi=", 4) == 0) { std::sscanf(argv[i], "epi=%lf", &epi); }
    else if (std::strncmp(argv[i], "mb=", 3) == 0) { std::sscanf(argv[i], "mb=%ld", &mb); }
    else if (std::strncmp(argv[i], "nb=", 3) == 0) { std::sscanf(argv[i], "nb=%ld", &nb); }
    else if (std::strncmp(argv[i], "tilem=", 6) == 0) { std::sscanf(argv[i], "tilem=%d", &tile_m); }
    else if (std::strncmp(argv[i], "tilen=", 6) == 0) { std::sscanf(argv[i], "tilen=%d", &tile_n); }
    else if (std::strncmp(argv[i], "file=", 5) == 0) { file.resize(std::strlen(argv[i])); std::sscanf(argv[i], "file=%s", file.data()); }
    else if (std::strncmp(argv[i], "algo=", 5) == 0) { std::sscanf(argv[i], "algo=%c", &algo); }
    else { std::cerr << "Ignored parameter: " << argv[i] << std::endl; }
  }

  gN = std::min(gM, gN); K = std::min(gN, K);

  int32_t world_rank, world_size, local_rank; ncclUniqueId id;
  //__bootstrap_mpi(world_rank, world_size, local_rank, id);
  __bootstrap_posix_fork(local_rank, world_size, id); world_rank = local_rank;

  if (world_size != tile_m * tile_n)
  { if (world_rank == 0) std::cerr << "Incorrect process grid launch configuration." << std::endl; return -1; }
  int32_t grid_row = world_rank % tile_m, grid_col = world_rank / tile_m;

  int32_t device_count = 0; cudaGetDeviceCount(&device_count);
  auto cu_err = cudaSetDevice(1 < device_count ? local_rank : 0);
  cudaDeviceReset();
  if (cu_err != cudaSuccess)
  { std::cerr << cudaGetErrorString(cu_err) << std::endl; return -1; }

  switch(prec) {
    case 'D': run<double, double>(prec, gM, gN, K, mb, nb, algo, epi, grid_row, grid_col, tile_m, tile_n, id, file); break;
    case 'S': run<float, float>(prec, gM, gN, K, mb, nb, algo, epi, grid_row, grid_col, tile_m, tile_n, id, file); break;
    case 'Z': run<std::complex<double>, double>(prec, gM, gN, K, mb, nb, algo, epi, grid_row, grid_col, tile_m, tile_n, id, file); break;
    case 'C': run<std::complex<float>, float>(prec, gM, gN, K, mb, nb, algo, epi, grid_row, grid_col, tile_m, tile_n, id, file); break;
    default: break;
  }

  cu_err = cudaGetLastError();
  if (cu_err != cudaSuccess)
    std::cerr << cudaGetErrorString(cu_err) << std::endl;
  return 0;
}
