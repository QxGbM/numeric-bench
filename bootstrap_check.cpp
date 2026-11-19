
#include <benchmark_common.hpp>

int32_t main(int32_t argc, char* argv[]) try {
  int32_t ranks = 1;
  for (int32_t i = 1; i < argc; ++i) {
    if (std::strncmp(argv[i], "ranks=", 6) == 0) { std::sscanf(argv[i], "ranks=%d", &ranks); }
    else { fprintf(stderr, "Ignored parameter: %s\n", argv[i]); }
  }
  int32_t world_rank, world_size, local_rank; ncclUniqueId id;
  enable_nccl_multi_rank_gpu();
  bootstrap_posix_fork(world_rank, local_rank, world_size, id, ranks);

  int32_t device_count = 0; LIBRARY_CHECK(cudaGetDeviceCount(&device_count));
  int32_t device_picked = 1 < device_count ? local_rank % device_count : 0;
  LIBRARY_CHECK(cudaSetDevice(device_picked));
  
  char pciBusId[20];
  LIBRARY_CHECK(cudaDeviceGetPCIBusId(pciBusId, 20, device_picked));
  printf("rank=%d, local_rank=%d, world_size=%d, visible_device=%d, picked=%d, bus=%s\n", 
    world_rank, local_rank, world_size, device_count, device_picked, pciBusId);

  ncclComm_t comm;
  LIBRARY_CHECK(ncclCommInitRank(&comm, world_size, id, world_rank));
  cudaStream_t stream;
  LIBRARY_CHECK(cudaStreamCreate(&stream));

  int32_t* dev = nullptr, one = 1;
  LIBRARY_CHECK(cudaMalloc((void**)&dev, sizeof(int32_t)));
  LIBRARY_CHECK(cudaMemcpy(dev, &one, sizeof(int32_t), cudaMemcpyHostToDevice));
  LIBRARY_CHECK(ncclAllReduce(dev, dev, 1, ncclInt32, ncclSum, comm, stream));
  LIBRARY_CHECK(cudaStreamSynchronize(stream));

  LIBRARY_CHECK(cudaMemcpy(&one, dev, sizeof(int32_t), cudaMemcpyDeviceToHost));
  printf("All-reduce: %d, Expected: %d\n", one, world_size);
  
  LIBRARY_CHECK(cudaFree(dev));
  LIBRARY_CHECK(ncclCommDestroy(comm));
  return 0;
}

catch (const std::exception& error) {
  fprintf(stderr, "%s\n", error.what());
  return EXIT_FAILURE;
}
