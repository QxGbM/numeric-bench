
#include <vector>
#include <complex>
#include <iostream>
#include <algorithm>
#include <magma_v2.h>
#include <cublas_v2.h>

void make_2D_oscillatory(double w, int32_t sep, int32_t M, int32_t N, double* A, int32_t lda) {
  constexpr int32_t height = 128;
  auto translate_2d = [](int64_t i) { int64_t x = i / height, y = i - height * x; return std::complex<double>(x, y); };
  sep = height * sep + ((M + height - 1) & (~(height - 1)));

  for (int32_t j = 0; j < N; ++j) {
    auto vj = translate_2d(j + sep);
    for (int32_t i = 0; i < M; ++i) {
      auto vi = translate_2d(i);
      double d = std::abs(vi - vj);
      A[uint64_t(i) + uint64_t(j) * uint64_t(lda)] = double(std::cos(w * d) / d);
    }
  }
}

int32_t main(int32_t argc, char* argv[]) {
  auto cu_err = cudaSetDevice(0);
  if (cu_err != cudaSuccess)
  { fprintf(stderr, "%s\n", cudaGetErrorString(cu_err)); return -1; }

  magma_init();
  int64_t M = 1 < argc ? std::atoi(argv[1]) : 1024;
  int64_t N = 2 < argc ? std::atoi(argv[2]) : 128;
  N = std::min(M, N);

  double epi = 3 < argc ? std::atof(argv[3]) : 1.e-12;

  magma_queue_t queue = nullptr;
  magma_queue_create(0, &queue);
  cublasHandle_t cublasH = magma_queue_get_cublas_handle(queue);
  cudaStream_t stream = magma_queue_get_cuda_stream(queue);

  double* dA, *dB, *dC;
  std::vector<double> tau(N);
  std::vector<magma_int_t> jpvt(N, 0);

  std::vector<double> matA(M * N), matB(M * N);
  make_2D_oscillatory(1., 0, M, N, matA.data(), M);
  
  cudaMalloc((void**)&dA, M * N * sizeof(double));
  cudaMalloc((void**)&dB, M * N * sizeof(double));
  cudaMemcpy(dA, &matA[0], M * N * sizeof(double), cudaMemcpyHostToDevice);

  magma_int_t Lwork, info;
  double work;
  magma_dgeqp3(M, N, nullptr, M, jpvt.data(), tau.data(), &work, -1, &info);
  Lwork = std::max(int64_t(16) * (magma_int_t)(work), M * N);
  cudaMalloc((void**)&dC, int64_t(Lwork) * sizeof(double));

  magma_dgeqp3_gpu(M, N, dA, M, jpvt.data(), tau.data(), dC, Lwork, &info);
  cudaStreamSynchronize(stream);
  cudaMemcpy(&matB[0], dA, M * N * sizeof(double), cudaMemcpyDeviceToHost);

  double s0 = epi * std::abs(matB[0]);
  int64_t rank = 0;
  while (rank < N && s0 <= double(std::abs(matB[rank * (M + 1)]))) { ++rank; }

  for (int64_t i = 0; i < rank; ++i) {
    int64_t col = jpvt[i] - 1;
    cudaMemcpy(&dB[i * M], &matA[col * M], M * sizeof(double), cudaMemcpyHostToDevice);
  }

  double one = 1., zero = 0.;
  cublasDtrsm(cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, rank, N - rank, &one, dA, M, &dA[rank * M], M);
  magmablas_dlaset(MagmaFull, rank, rank, zero, one, dA, M, queue);
  cudaStreamSynchronize(stream);

  for (int64_t i = 0; i < N; ++i) {
    int64_t col = jpvt[i] - 1;
    cudaMemcpy(&dC[col * M], &dA[i * M], rank * sizeof(double), cudaMemcpyDeviceToDevice);
  }
  cudaMemcpy(dA, &matA[0], M * N * sizeof(double), cudaMemcpyHostToDevice);

  double minus_one = -1.;
  double nrm = 0., err = 0.;
  cublasDnrm2_64(cublasH, M * N, dA, int64_t(1), &nrm);
  cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, M, N, rank, &minus_one, dB, M, dC, M, &one, dA, M);
  cublasDnrm2_64(cublasH, M * N, dA, int64_t(1), &err);
  cudaStreamSynchronize(stream);
  err = err / nrm;

  cudaMemcpy(dA, &matA[0], M * N * sizeof(double), cudaMemcpyHostToDevice);
  std::fill(jpvt.begin(), jpvt.end(), 0);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, stream);
  magma_dgeqp3_gpu(M, N, dA, M, jpvt.data(), tau.data(), dC, Lwork, &info);
  cublasDtrsm(cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, rank, N - rank, &one, dA, M, &dA[rank * M], M);
  cudaEventRecord(stop, stream);
  cudaStreamSynchronize(stream);

  float milliseconds = 0.0f;
  cudaEventElapsedTime(&milliseconds, start, stop);
  int64_t qr_flops = (M * N * N * 2) - (N * N * N * 2 / 3);
  int64_t trsm_flops = N * rank * rank;
  double gflops = double(qr_flops + trsm_flops) * 1.e-6 / milliseconds;
  std::cout << "magma-DLRA," << M << "," << N << "," << epi << "," << err << "," << rank << "," << milliseconds << "," << gflops << std::endl;
  
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);

  magma_queue_destroy(queue);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  magma_finalize();

  cu_err = cudaGetLastError();
  if (cu_err != cudaSuccess)
    std::cerr << cudaGetErrorString(cu_err) << std::endl;
  return 0;
}

