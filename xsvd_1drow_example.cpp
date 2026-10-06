
#include <common.hpp>
#include <iostream>
#include <chrono>

template <class T, class R> inline void run(char prec, char algo, double epi, int32_t jacobi_sweeps, int32_t u_corr, int32_t g_corr, int32_t oversampling, int32_t batchK, int32_t batchIter,
  int64_t gM, int64_t N, int64_t K, int64_t mb, int32_t grid_row, int32_t tile_m, ncclUniqueId id, const std::string& file) {
  int64_t lM = mb * (gM / (mb * tile_m));
  lM += std::max(int64_t(0), std::min(mb, gM - lM * tile_m - mb * grid_row));

  std::vector<T> matA(lM * N);
  if (!file.empty())
    matrix_from_row_major_csv(gM, N, mb, 512, matA.data(), lM, file, grid_row, 0, tile_m, 1);
  else
    matrix_generator<T>(1., gM, N).generate_block(mb, 512, &matA[0], lM, grid_row, 0, tile_m, 1);

  /* Timed region start */
  auto host_start = std::chrono::high_resolution_clock::now();

  T* d_A = nullptr, *d_U = nullptr, *d_V = nullptr; R* d_S = nullptr;
  cudaMalloc((void**)(&d_A), lM * N * sizeof(T));
  cudaMalloc((void**)(&d_U), lM * K * sizeof(T));
  cudaMalloc((void**)(&d_V), K * N * sizeof(T));
  cudaMalloc((void**)(&d_S), K * sizeof(R));
  cudaMemcpy(d_A, matA.data(), lM * N * sizeof(T), cudaMemcpyHostToDevice);

  hyacinHandle_t handle;
  ncclComm_t comm;

  ncclCommInitRank(&comm, tile_m, id, grid_row);
  hyacinCreate2D(&handle, comm, nullptr); handle_param_overwrite(&handle, algo, batchK, u_corr, g_corr, jacobi_sweeps, oversampling);

  int32_t* d_barrier = nullptr; cudaMalloc((void**)(&d_barrier), 5 * sizeof(double));
  int32_t rank = svd_fit_transform(handle, epi, batchIter, lM, gM, N, K, d_A, lM, d_U, lM, d_S, d_V, N, N);

  std::vector<T> matU(lM * K), matV(K * N);
  cudaMemcpy(matU.data(), d_U, lM * K * sizeof(T), cudaMemcpyDeviceToHost);
  cudaMemcpy(matV.data(), d_V, K * N * sizeof(T), cudaMemcpyDeviceToHost);

  ncclAllReduce(d_barrier, d_barrier, 1, ncclInt32, ncclMin, comm, handle.cudaStream);
  cudaStreamSynchronize(handle.cudaStream);
  kernel_time = rep_time = comm_time = 0.;

  for (int32_t i = 0; i < kernel_runs; ++i)
    rank = svd_fit_transform(handle, epi, batchIter, lM, gM, N, K, d_A, lM, d_U, lM, d_S, d_V, N, N);

  double ret[5]{ kernel_time, rep_time, comm_time, check_answer_svd(lM, N, rank, &matU[0], lM, &matV[0], N, &matA[0], lM), fnorm(lM, N, &matA[0], lM) };
  cudaMemcpy(d_barrier, ret, 5 * sizeof(double), cudaMemcpyHostToDevice);
  ncclAllReduce(d_barrier, d_barrier, 5, ncclDouble, ncclSum, comm, handle.cudaStream);
  cudaStreamSynchronize(handle.cudaStream);
  cudaMemcpy(ret, d_barrier, 5 * sizeof(double), cudaMemcpyDeviceToHost);
  double div = 1. / double(tile_m * kernel_runs); kernel_time = ret[0] * div; rep_time = ret[1] * div; comm_time = ret[2] * div;
  double err = ret[4] == 0. ? std::numeric_limits<double>::quiet_NaN() : std::sqrt(ret[3] / ret[4]);

  hyacinDestroy(&handle);
  ncclCommDestroy(comm);
  std::vector<R> vecS(K);
  cudaMemcpy(vecS.data(), d_S, K * sizeof(R), cudaMemcpyDeviceToHost);
  cudaFree(d_A); cudaFree(d_U); cudaFree(d_S); cudaFree(d_V); cudaFree(d_barrier);

  /* Timed region end */
  std::chrono::duration<double, std::milli> host_wtime = std::chrono::high_resolution_clock::now() - host_start;
  double duration = host_wtime.count();

  printf("%c-SVD#%d [M=%ld,N=%ld,K=%ld] [epi=%.1le] [err=%.12le] [rank=%d] [host=%lf ms] [kernel=%lf ms] [replicate=%lf ms] [comm=%lf ms]\n",
    prec, grid_row, gM, N, K, epi, err, rank, duration, kernel_time, rep_time, comm_time);
  //write_matrix_to_csv(rank, 1, &vecS[0], rank, "sv.csv");
}

int32_t main(int32_t argc, char* argv[]) {
  char prec = 'D', algo = 'A'; std::string file;
  int64_t gM = 2048, N = 2048, K = 2048, mb = 512; int32_t jacobi_sweeps = 30, u_corr = 6, g_corr = -5, oversampling = 10, batchK = 65536, batchIter = 2048;
  double epi = 1.e-12;

  for (int32_t i = 1; i < argc; ++i) {
    if (std::strncmp(argv[i], "M=", 2) == 0) { std::sscanf(argv[i], "M=%ld", &gM); }
    else if (std::strncmp(argv[i], "N=", 2) == 0) { std::sscanf(argv[i], "N=%ld", &N); }
    else if (std::strncmp(argv[i], "K=", 2) == 0) { std::sscanf(argv[i], "K=%ld", &K); }
    else if (std::strncmp(argv[i], "data=", 5) == 0) { std::sscanf(argv[i], "data=%c", &prec); }
    else if (std::strncmp(argv[i], "epi=", 4) == 0) { std::sscanf(argv[i], "epi=%lf", &epi); }
    else if (std::strncmp(argv[i], "mb=", 3) == 0) { std::sscanf(argv[i], "mb=%ld", &mb); }
    else if (std::strncmp(argv[i], "file=", 5) == 0) { file.resize(std::strlen(argv[i])); std::sscanf(argv[i], "file=%s", file.data()); }
    else if (std::strncmp(argv[i], "algo=", 5) == 0) { std::sscanf(argv[i], "algo=%c", &algo); }
    else if (std::strncmp(argv[i], "jacobi=", 7) == 0) { std::sscanf(argv[i], "jacobi=%d", &jacobi_sweeps); }
    else if (std::strncmp(argv[i], "u_corr=", 7) == 0) { std::sscanf(argv[i], "u_corr=%d", &u_corr); }
    else if (std::strncmp(argv[i], "g_corr=", 7) == 0) { std::sscanf(argv[i], "g_corr=%d", &g_corr); }
    else if (std::strncmp(argv[i], "p=", 2) == 0) { std::sscanf(argv[i], "p=%d", &oversampling); }
    else if (std::strncmp(argv[i], "batchK=", 7) == 0) { std::sscanf(argv[i], "batchK=%d", &batchK); }
    else if (std::strncmp(argv[i], "batchIter=", 10) == 0) { std::sscanf(argv[i], "batchIter=%d", &batchIter); }
    else { std::cerr << "Ignored parameter: " << argv[i] << std::endl; }
  }
  N = std::min(gM, N); K = std::min(N, K);

  int32_t world_rank, world_size, local_rank; ncclUniqueId id;
  //bootstrap_mpi(world_rank, local_rank, world_size, id);
  bootstrap_posix_fork(world_rank, local_rank, world_size, id);

  int32_t device_count = 0; cudaGetDeviceCount(&device_count);
  auto cu_err = cudaSetDevice(1 < device_count ? local_rank : 0);
  cudaDeviceReset();
  if (cu_err != cudaSuccess)
  { std::cerr << cudaGetErrorString(cu_err) << std::endl; return -1; }

  switch(prec) {
    case 'D': run<double, double>(prec, algo, epi, jacobi_sweeps, u_corr, g_corr, oversampling, batchK, batchIter, gM, N, K, mb, world_rank, world_size, id, file); break;
    case 'S': run<float, float>(prec, algo, epi, jacobi_sweeps, u_corr, g_corr, oversampling, batchK, batchIter, gM, N, K, mb, world_rank, world_size, id, file); break;
    case 'H': run<__half, __half>(prec, algo, epi, jacobi_sweeps, u_corr, g_corr, oversampling, batchK, batchIter, gM, N, K, mb, world_rank, world_size, id, file); break;
    case 'Z': run<std::complex<double>, double>(prec, algo, epi, jacobi_sweeps, u_corr, g_corr, oversampling, batchK, batchIter, gM, N, K, mb, world_rank, world_size, id, file); break;
    case 'C': run<std::complex<float>, float>(prec, algo, epi, jacobi_sweeps, u_corr, g_corr, oversampling, batchK, batchIter, gM, N, K, mb, world_rank, world_size, id, file); break;
    case 'J': run<__half2, __half>(prec, algo, epi, jacobi_sweeps, u_corr, g_corr, oversampling, batchK, batchIter, gM, N, K, mb, world_rank, world_size, id, file); break;
    default: break;
  }

  cu_err = cudaGetLastError();
  if (cu_err != cudaSuccess)
    std::cerr << cudaGetErrorString(cu_err) << std::endl;
  return 0;
}
