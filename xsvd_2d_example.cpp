
#include <common.hpp>
#include <iostream>
#include <chrono>

template <class T, class R> inline void run(char prec, char algo, char use_evd, double epi, int32_t u_corr, int32_t g_corr, int32_t oversampling, int32_t batchK, int32_t batchIter,
  int64_t gM, int64_t gN, int64_t K, int64_t mb, int64_t nb, int32_t grid_row, int32_t grid_col, int32_t tile_m, int32_t tile_n, ncclUniqueId id, const std::string& file) {
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

  /* Timed region start */
  auto host_start = std::chrono::high_resolution_clock::now();

  T* d_A = nullptr, *d_U = nullptr, *d_V = nullptr; R* d_S = nullptr;
  cudaMalloc((void**)(&d_A), lM * lN * sizeof(T));
  cudaMalloc((void**)(&d_U), lM * gK * sizeof(T));
  cudaMalloc((void**)(&d_V), K * lN * sizeof(T));
  cudaMalloc((void**)(&d_S), K * sizeof(R));
  cudaMemcpy(d_A, matA.data(), lM * lN * sizeof(T), cudaMemcpyHostToDevice);

  hyacinHandle_t handle;
  ncclComm_t comm, comm_row, comm_col;

  ncclCommInitRank(&comm, tile_m * tile_n, id, grid_row + grid_col * tile_m);
  ncclCommSplit(comm, grid_row, grid_col, &comm_row, nullptr);
  ncclCommSplit(comm, grid_col, grid_row, &comm_col, nullptr);
  hyacinCreate2D(&handle, comm_col, comm_row, 1);

  int32_t* d_barrier = nullptr; cudaMalloc((void**)(&d_barrier), 5 * sizeof(double));
  int32_t r1 = svd_fit_transform(handle, algo, use_evd, epi, u_corr, g_corr, oversampling, batchK, batchIter, lM, gM, lN, K, d_A, lM, d_U, lM, d_S, d_V, lN, lN), N2 = r1;
  int32_t offset = hyacinXAllGatherV1Dcol(handle, lM, &N2, int32_t(sizeof(T)), d_U, lM);
  int32_t r2 = svd_fit_transform(handle, algo, use_evd, epi, u_corr, g_corr, oversampling, batchK, batchIter, lM, gM, N2, K, d_U, lM, d_U, lM, d_S, d_V, lN, lN, r1, offset);

  std::vector<T> matU(lM * K), matV(K * lN);
  cudaMemcpy(matU.data(), d_U, lM * K * sizeof(T), cudaMemcpyDeviceToHost);
  cudaMemcpy(matV.data(), d_V, K * lN * sizeof(T), cudaMemcpyDeviceToHost);

  ncclAllReduce(d_barrier, d_barrier, 1, ncclInt32, ncclMin, comm, handle.cudaStream);
  cudaStreamSynchronize(handle.cudaStream);
  kernel_time = rep_time = comm_time = 0.;

  for (int32_t i = 0; i < kernel_runs; ++i) {
    N2 = r1 = svd_fit_transform(handle, algo, use_evd, epi, u_corr, g_corr, oversampling, batchK, batchIter, lM, gM, lN, K, d_A, lM, d_U, lM, d_S, d_V, lN, lN);
    offset = hyacinXAllGatherV1Dcol(handle, lM, &N2, int32_t(sizeof(T)), d_U, lM);
    r2 = svd_fit_transform(handle, algo, use_evd, epi, u_corr, g_corr, oversampling, batchK, batchIter, lM, gM, N2, K, d_U, lM, d_U, lM, d_S, d_V, lN, lN, r1, offset);
  }

  double ret[5]{ kernel_time, rep_time, comm_time, check_answer_svd(lM, lN, r2, &matU[0], lM, &matV[0], lN, &matA[0], lM), fnorm(lM, lN, &matA[0], lM) };
  cudaMemcpy(d_barrier, ret, 5 * sizeof(double), cudaMemcpyHostToDevice);
  ncclAllReduce(d_barrier, d_barrier, 5, ncclDouble, ncclSum, comm, handle.cudaStream);
  cudaStreamSynchronize(handle.cudaStream);
  cudaMemcpy(ret, d_barrier, 5 * sizeof(double), cudaMemcpyDeviceToHost);
  double div = 1. / double(tile_m * tile_n * kernel_runs); kernel_time = ret[0] * div; rep_time = ret[1] * div; comm_time = ret[2] * div;
  double err = ret[3] == 0. ? std::numeric_limits<double>::quiet_NaN() : std::sqrt(ret[3] / ret[4]);

  hyacinDestroy(handle);
  ncclCommDestroy(comm);
  ncclCommDestroy(comm_row);
  ncclCommDestroy(comm_col);
  std::vector<R> vecS(K);
  cudaMemcpy(vecS.data(), d_S, K * sizeof(R), cudaMemcpyDeviceToHost);
  cudaFree(d_A); cudaFree(d_U); cudaFree(d_S); cudaFree(d_V); cudaFree(d_barrier);

  /* Timed region end */
  std::chrono::duration<double, std::milli> host_wtime = std::chrono::high_resolution_clock::now() - host_start;
  double duration = host_wtime.count();

  printf("%c-SVD#(%d,%d) [M=%ld,N=%ld,K=%ld] [epi=%.1le] [err=%.12le] [rank1=%d,rank2=%d] [host=%lf ms] [kernel=%lf ms] [replicate=%lf ms] [comm=%lf ms]\n",
    prec, grid_row, grid_col, gM, gN, K, epi, err, r1, r2, duration, kernel_time, rep_time, comm_time);
}

int32_t main(int32_t argc, char* argv[]) {
  char prec = 'D', algo = 'A', use_evd = 'A'; std::string file;
  int32_t tile_m = 1, tile_n = 1, u_corr = 6, g_corr = -5, oversampling = 10, batchK = 65536, batchIter = 2048;
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
    else if (std::strncmp(argv[i], "evd=", 4) == 0) { std::sscanf(argv[i], "evd=%c", &use_evd); }
    else if (std::strncmp(argv[i], "u_corr=", 7) == 0) { std::sscanf(argv[i], "u_corr=%d", &u_corr); }
    else if (std::strncmp(argv[i], "g_corr=", 7) == 0) { std::sscanf(argv[i], "g_corr=%d", &g_corr); }
    else if (std::strncmp(argv[i], "p=", 2) == 0) { std::sscanf(argv[i], "p=%d", &oversampling); }
    else if (std::strncmp(argv[i], "batchK=", 7) == 0) { std::sscanf(argv[i], "batchK=%d", &batchK); }
    else if (std::strncmp(argv[i], "batchIter=", 10) == 0) { std::sscanf(argv[i], "batchIter=%d", &batchIter); }
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
    case 'D': run<double, double>(prec, algo, use_evd, epi, u_corr, g_corr, oversampling, batchK, batchIter, gM, gN, K, mb, nb, grid_row, grid_col, tile_m, tile_n, id, file); break;
    case 'S': run<float, float>(prec, algo, use_evd, epi, u_corr, g_corr, oversampling, batchK, batchIter, gM, gN, K, mb, nb, grid_row, grid_col, tile_m, tile_n, id, file); break;
    case 'H': run<__half, __half>(prec, algo, use_evd, epi, u_corr, g_corr, oversampling, batchK, batchIter, gM, gN, K, mb, nb, grid_row, grid_col, tile_m, tile_n, id, file); break;
    case 'Z': run<std::complex<double>, double>(prec, algo, use_evd, epi, u_corr, g_corr, oversampling, batchK, batchIter, gM, gN, K, mb, nb, grid_row, grid_col, tile_m, tile_n, id, file); break;
    case 'C': run<std::complex<float>, float>(prec, algo, use_evd, epi, u_corr, g_corr, oversampling, batchK, batchIter, gM, gN, K, mb, nb, grid_row, grid_col, tile_m, tile_n, id, file); break;
    case 'J': run<__half2, __half>(prec, algo, use_evd, epi, u_corr, g_corr, oversampling, batchK, batchIter, gM, gN, K, mb, nb, grid_row, grid_col, tile_m, tile_n, id, file); break;
    default: break;
  }

  cu_err = cudaGetLastError();
  if (cu_err != cudaSuccess)
    std::cerr << cudaGetErrorString(cu_err) << std::endl;
  return 0;
}
