
#include <commons.hpp>
#include <hyacinth.hpp>
#include <random>

int32_t main(int32_t argc, char* argv[]) {
  auto cu_err = cudaSetDevice(0);
  cudaDeviceReset();
  if (cu_err != cudaSuccess)
  { fprintf(stderr, "%s\n", cudaGetErrorString(cu_err)); return -1; }

  int32_t M = std::atoi(argv[1]);
  int32_t N = std::atoi(argv[2]);
  double epi = std::atof(argv[3]);
  Eigen::MatrixXd matA(M, N);
  int32_t* ipiv;

  std::mt19937_64 gen;
  std::normal_distribution<double> dist(-32, 32);

  for (int32_t j = 0; j < N; ++j)
    for (int32_t i = 0; i < M; ++i)
      matA(i, j) = dist(gen);
  
  /*Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(matA);
  Eigen::MatrixXd R = qr.matrixR().topRows(N).triangularView<Eigen::Upper>();

  for (int32_t i = 0; i < N; ++i)
    if (R(i, i) < 0.)
      R.row(i) = -R.row(i);*/

  device::QR::geqp3_params params;
  device::QR::dgeqp3_ronly_params_query(&params, epi, M, N);
  //device::QR::set_quad_float_as_fp128(&params);
  printf("acc-bits:%d orderA:%d orderC:%d iter_k:%d i8:%lld i32:%lld complex:%lld, complex_bytes:%lld, total_work:%lld\n", params.acc_bits, params.orderA, params.orderC, params.iter_k, params.n_i8, params.n_i32, params.n_elem, params.elem_bytes, params.work_bytes);

  cudaStream_t stream;
  cublasHandle_t handle;
  cudaStreamCreate(&stream);
  cublasCreate(&handle);
  cublasSetStream(handle, stream);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  double* d_A = nullptr;
  void* work = nullptr;
  cudaMalloc((void**)(&d_A), M * N * sizeof(double));
  cudaMalloc(&work, params.work_bytes);
  cudaMallocHost((void**)(&ipiv), (N + 8) * sizeof(int32_t));

  cudaMemcpy(d_A, matA.data(), M * N * sizeof(double), cudaMemcpyHostToDevice);

  cudaEventRecord(start, stream);
  int32_t ret = device::QR::dgeqp3_ronly(stream, handle, params, d_A, M, ipiv, work);
  cudaEventRecord(stop, stream);

  cudaDeviceSynchronize();
  std::cout << ret << std::endl;
  cudaMemcpy(matA.data(), d_A, M * N * sizeof(double), cudaMemcpyDeviceToHost);

  //Eigen::MatrixXd matU = matA.topRows(N).triangularView<Eigen::Upper>();
  //std::cout << (matU - R).norm() / R.norm() << std::endl;

  float milliseconds = 0.0f;
  cudaEventElapsedTime(&milliseconds, start, stop);
  int64_t flops = (int64_t(N) * int64_t(N) * int64_t(N) / 3) + (int64_t(M) * int64_t(N) * int64_t(N) * 2);
  std::cout << "Time: " << milliseconds << " ms\n";
  std::cout << "GFLOPs: " << double(flops) * 1.e-6 / milliseconds << "\n";

  cudaFree(d_A);
  cudaFreeHost(ipiv);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaStreamDestroy(stream);
  cublasDestroy(handle);
  fprintf(stderr, "%s\n", cudaGetErrorString(cudaGetLastError()));
  return 0;
}
