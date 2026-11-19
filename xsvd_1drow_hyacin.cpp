
#include <hyacin_common.hpp>
#include <iostream>
#include <chrono>

template <class T, class R> inline void run(char prec, double epi, int64_t gM, int64_t N, int64_t K, int64_t mb, int32_t grid_row, int32_t tile_m, ncclUniqueId id) {
  int64_t lM = mb * (gM / (mb * tile_m));
  lM += std::max(int64_t(0), std::min(mb, gM - lM * tile_m - mb * grid_row));

  std::vector<T> matA(lM * N);
  matrix_generator<T>(1., gM, N).generate_block(mb, 512, &matA[0], lM, grid_row, 0, tile_m, 1);

  /* Timed region start */
  auto host_start = std::chrono::high_resolution_clock::now();

  T* d_A = nullptr, *d_V = nullptr; R* d_S = nullptr;
  LIBRARY_CHECK(cudaMalloc((void**)(&d_A), lM * N * sizeof(T)));
  LIBRARY_CHECK(cudaMalloc((void**)(&d_V), K * N * sizeof(T)));
  LIBRARY_CHECK(cudaMalloc((void**)(&d_S), K * sizeof(R)));
  LIBRARY_CHECK(cudaMemcpy(d_A, matA.data(), lM * N * sizeof(T), cudaMemcpyHostToDevice));
  int32_t* d_barrier = nullptr; LIBRARY_CHECK(cudaMalloc((void**)(&d_barrier), 5 * sizeof(double)));

  hyacinHandle_t handle{};
  ncclComm_t comm;

  LIBRARY_CHECK(ncclCommInitRank(&comm, tile_m, id, grid_row));
  hyacinCreate(&handle); handle.ncclComm = comm;

  int32_t rank = hyacinTsvdFitSegmentN(&handle, epi, 'N', lM, N, K, d_A, lM, d_S, d_V, N, gM, hyacin_prec<T>());
  hyacinSync_TimerSegments(&handle, nullptr, 0);

  LIBRARY_CHECK(ncclAllReduce(d_barrier, d_barrier, 1, ncclInt32, ncclMin, comm, handle.cudaStream));
  LIBRARY_CHECK(cudaStreamSynchronize(handle.cudaStream));
  for (int32_t i = 0; i < kernel_runs; ++i)
    rank = hyacinTsvdFitSegmentN(&handle, epi, 'N', lM, N, K, d_A, lM, d_S, d_V, N, gM, hyacin_prec<T>());

  double eventMs[3]{ 'D', 'R', 'C' }; hyacinSync_TimerSegments(&handle, eventMs, 3);
  double kernel_time = eventMs[0] + eventMs[1] + eventMs[2], rep_time = eventMs[1], comm_time = eventMs[2];
  double ret[5]{ kernel_time, rep_time, comm_time, hyacin_check_norm_fp64<T>(&handle, 'N', lM, N, d_A, lM, rank, d_V, N), hyacin_check_norm_fp64<T>(&handle, 'N', lM, N, d_A, lM, 0, nullptr, 0) };
  LIBRARY_CHECK(cudaMemcpy(d_barrier, ret, 5 * sizeof(double), cudaMemcpyHostToDevice));
  LIBRARY_CHECK(ncclAllReduce(d_barrier, d_barrier, 5, ncclDouble, ncclSum, comm, handle.cudaStream));
  LIBRARY_CHECK(cudaStreamSynchronize(handle.cudaStream));
  LIBRARY_CHECK(cudaMemcpy(ret, d_barrier, 5 * sizeof(double), cudaMemcpyDeviceToHost));
  double div = 1. / double(tile_m * kernel_runs); kernel_time = ret[0] * div; rep_time = ret[1] * div; comm_time = ret[2] * div;
  double err = ret[4] == 0. ? std::numeric_limits<double>::quiet_NaN() : std::sqrt(ret[3] / ret[4]);

  hyacinDestroy(&handle);
  LIBRARY_CHECK(ncclCommDestroy(comm));
  std::vector<R> vecS(K);
  LIBRARY_CHECK(cudaMemcpy(vecS.data(), d_S, K * sizeof(R), cudaMemcpyDeviceToHost));
  LIBRARY_CHECK(cudaFree(d_A)); LIBRARY_CHECK(cudaFree(d_S)); LIBRARY_CHECK(cudaFree(d_V)); LIBRARY_CHECK(cudaFree(d_barrier));

  /* Timed region end */
  std::chrono::duration<double, std::milli> host_wtime = std::chrono::high_resolution_clock::now() - host_start;
  double duration = host_wtime.count();

  // Global equivalent work, divided by the already averaged per-run time.
  const double flops = randomized_svd_flops(double(gM), double(N), double(rank), 0., benchmark_flop_factor<T>);

  printf("%c-SVD#%d [M=%ld,N=%ld,K=%ld] [epi=%.1le] [err=%.12le] [rank=%d] [host=%lf ms] [kernel=%lf ms] [replicate=%lf ms] [comm=%lf ms] [flops=%.6le] [%lf GFlops]\n",
    prec, grid_row, gM, N, K, epi, err, rank, duration, kernel_time, rep_time, comm_time, flops, (1.e-6 * flops) / kernel_time);
}

int32_t main(int32_t argc, char* argv[]) try {
  char prec = 'D', algo = 'A';
  int64_t gM = 2048, N = 2048, K = 2048, mb = 512; int32_t jacobi_sweeps = 5, u_corr = 7, batch = 8192;
  double epi = 1.e-12;
  int32_t ranks = 1;

  for (int32_t i = 1; i < argc; ++i) {
    if (std::strncmp(argv[i], "M=", 2) == 0) { std::sscanf(argv[i], "M=%ld", &gM); }
    else if (std::strncmp(argv[i], "N=", 2) == 0) { std::sscanf(argv[i], "N=%ld", &N); }
    else if (std::strncmp(argv[i], "K=", 2) == 0) { std::sscanf(argv[i], "K=%ld", &K); }
    else if (std::strncmp(argv[i], "data=", 5) == 0) { std::sscanf(argv[i], "data=%c", &prec); }
    else if (std::strncmp(argv[i], "epi=", 4) == 0) { std::sscanf(argv[i], "epi=%lf", &epi); }
    else if (std::strncmp(argv[i], "ranks=", 6) == 0) { std::sscanf(argv[i], "ranks=%d", &ranks); }
    else if (std::strncmp(argv[i], "mb=", 3) == 0) { std::sscanf(argv[i], "mb=%ld", &mb); }
    else if (std::strncmp(argv[i], "algo=", 5) == 0) { std::sscanf(argv[i], "algo=%c", &algo); }
    else if (std::strncmp(argv[i], "jacobi=", 7) == 0) { std::sscanf(argv[i], "jacobi=%d", &jacobi_sweeps); }
    else if (std::strncmp(argv[i], "u_corr=", 7) == 0) { std::sscanf(argv[i], "u_corr=%d", &u_corr); }
    else if (std::strncmp(argv[i], "batch=", 6) == 0) { std::sscanf(argv[i], "batch=%d", &batch); }
    else { std::cerr << "Ignored parameter: " << argv[i] << std::endl; }
  }
  handle_param_environments(algo, batch, u_corr, jacobi_sweeps);

  int32_t world_rank, world_size, local_rank; ncclUniqueId id;
  enable_nccl_multi_rank_gpu();
  bootstrap_posix_fork(world_rank, local_rank, world_size, id, ranks);

  int32_t device_count = 0; LIBRARY_CHECK(cudaGetDeviceCount(&device_count));
  LIBRARY_CHECK(cudaSetDevice(1 < device_count ? local_rank % device_count : 0));
  LIBRARY_CHECK(cudaDeviceReset());

  switch(prec) {
    case 'D': run<double, double>(prec, epi, gM, N, K, mb, world_rank, world_size, id); break;
    case 'S': run<float, float>(prec, epi, gM, N, K, mb, world_rank, world_size, id); break;
    case 'H': run<__half, __half>(prec, epi, gM, N, K, mb, world_rank, world_size, id); break;
    case 'Z': run<cuDoubleComplex, double>(prec, epi, gM, N, K, mb, world_rank, world_size, id); break;
    case 'C': run<cuComplex, float>(prec, epi, gM, N, K, mb, world_rank, world_size, id); break;
    case 'J': run<__half2, __half>(prec, epi, gM, N, K, mb, world_rank, world_size, id); break;
    default: break;
  }

  LIBRARY_CHECK(cudaGetLastError());
  return 0;
}

catch (const std::exception& error) {
  fprintf(stderr, "%s\n", error.what());
  return EXIT_FAILURE;
}
