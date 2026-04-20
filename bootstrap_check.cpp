
#include <common.hpp>

int32_t main(int32_t, char* []) {
  int32_t world_rank, world_size, local_rank; ncclUniqueId id;
  //__bootstrap_mpi(world_rank, world_size, local_rank, id);
  __bootstrap_posix_fork(local_rank, world_size, id); world_rank = local_rank;

  int32_t device_count = 0; cudaGetDeviceCount(&device_count);
  int32_t device_picked = 1 < device_count ? local_rank : 0;
  cudaSetDevice(device_picked);
  
  char pciBusId[20];
  cudaDeviceGetPCIBusId(pciBusId, 20, device_picked);
  printf("rank=%d, local_rank=%d, world_size=%d, visible_device=%d, picked=%d, bus=%s\n", 
    world_rank, local_rank, world_size, device_count, device_picked, pciBusId);

  ncclComm_t comm;
  ncclCommInitRank(&comm, world_size, id, world_rank);
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  int32_t* dev = nullptr, one = 1;
  cudaMalloc((void**)&dev, sizeof(int32_t));
  cudaMemcpy(dev, &one, sizeof(int32_t), cudaMemcpyHostToDevice);
  ncclAllReduce(dev, dev, 1, ncclInt32, ncclSum, comm, stream);
  cudaStreamSynchronize(stream);

  cudaMemcpy(&one, dev, sizeof(int32_t), cudaMemcpyDeviceToHost);
  printf("All-reduce: %d\n", one);
  
  cudaFree(dev);
  ncclCommDestroy(comm);
  return 0;
}
