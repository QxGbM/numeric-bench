
#include <commons.hpp>
#include <thrust/transform.h>

struct op {
  __device__ float operator()(double a) { return (float)(a); }
};

int32_t main() {
  cudaSetDevice(0);
  cudaStream_t stream;
  cublasHandle_t handle;
  cudaStreamCreate(&stream);
  cublasCreate(&handle);
  cublasSetStream(handle, stream);

  int64_t len = 1000000;
  int32_t loops = 10;
  std::vector<double> hostVec(len);

  random_vector(len, (double*)hostVec.data());

  double* hostVecPin = nullptr, *devVecDouble = nullptr, start, lapse, Gb = 1.e-9 * len * sizeof(double) * loops;
  float* devVecFloat = nullptr, nrm;

  cudaMallocHost(reinterpret_cast<void**>(&hostVecPin), len * sizeof(double));
  cudaMallocManaged(reinterpret_cast<void**>(&devVecDouble), len * sizeof(double), cudaMemAttachGlobal);
  cudaMallocManaged(reinterpret_cast<void**>(&devVecFloat), len * sizeof(float), cudaMemAttachGlobal);

  std::copy(hostVec.begin(), hostVec.end(), hostVecPin);
  cudaDeviceSynchronize();
  start = omp_get_wtime();

  for (int32_t i = 0; i < loops; ++i) {
    cudaMemcpyAsync(devVecDouble, hostVecPin, len * sizeof(double), cudaMemcpyHostToDevice, stream);
    thrust::transform(thrust::cuda::par_nosync.on(stream), devVecDouble, &devVecDouble[len], devVecFloat, op());
  }

  cudaDeviceSynchronize();
  lapse = omp_get_wtime() - start;
  nrm = cblas_snrm2(len, devVecFloat, 1);

  printf("<Copy FP64 and GPU conversion> time: %f ms. Gbps: %f, Norm is %f\n", lapse * 1000, Gb / lapse, nrm);

  cudaDeviceSynchronize();
  start = omp_get_wtime();

  for (int32_t i = 0; i < loops; ++i)
    thrust::transform(thrust::cuda::par_nosync.on(stream), hostVecPin, &hostVecPin[len], devVecFloat, op());

  cudaDeviceSynchronize();
  lapse = omp_get_wtime() - start;
  nrm = cblas_snrm2(len, devVecFloat, 1);

  printf("<Direct Access and GPU conversion> time: %f ms. Gbps: %f, Norm is %f\n", lapse * 1000, Gb / lapse, nrm);

  cudaDeviceSynchronize();
  start = omp_get_wtime();

  for (int32_t i = 0; i < loops; ++i) {
    LAPACKE_dlag2s(LAPACK_COL_MAJOR, len, 1, hostVec.data(), len, (float*)hostVecPin, len);
    cudaMemcpy(devVecFloat, hostVecPin, len * sizeof(float), cudaMemcpyHostToDevice);
  }

  cudaDeviceSynchronize();
  lapse = omp_get_wtime() - start;
  nrm = cblas_snrm2(len, devVecFloat, 1);

  printf("<CPU conversion and copy FP32> time: %f ms. Gbps: %f, Norm is %f\n", lapse * 1000, Gb / lapse, nrm);

  cudaFreeHost(hostVecPin);
  cudaFree(devVecDouble);
  cudaFree(devVecFloat);

  cudaStreamDestroy(stream);
  cublasDestroy(handle);
  return 0;
}
