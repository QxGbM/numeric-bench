
#include <common.hpp>
#include <iostream>
#include <magma_v2.h>

void make_2D_oscillatory(double w, int32_t sep, int32_t M, int32_t N, std::complex<double>* A, int32_t lda) {
  constexpr int32_t height = 128;
  auto translate_2d = [](int64_t i) { int64_t x = i / height, y = i - height * x; return std::complex<double>(x, y); };
  sep = height * sep + ((M + height - 1) & (~(height - 1)));

  for (int32_t j = 0; j < N; ++j) {
    auto vj = translate_2d(j + sep);
    for (int32_t i = 0; i < M; ++i) {
      auto vi = translate_2d(i);
      double d = std::abs(vi - vj);
      A[uint64_t(i) + uint64_t(j) * uint64_t(lda)] = std::complex<double>(std::cos(w * d) / d, std::sin(w * d) / d);
    }
  }
}

template <class T>
double check_answer_lra(int32_t rank, int32_t M, int32_t N, const T* A, int32_t lda, const int32_t* jpiv, const T* R, int32_t ldr) {
  if (rank <= 0 || M <= 0 || N <= 0) { return 0.; }
  constexpr int32_t Complex = std::is_same_v<T, std::complex<double>> || std::is_same_v<T, std::complex<float>> || std::is_same_v<T, __half2>;
  using type = typename std::conditional<Complex, std::complex<double>, double>::type;
  std::vector<type> matB(int64_t(M) * int64_t(N)), matC(int64_t(M) * int64_t(rank)), matR(int64_t(rank) * int64_t(N));
  for (int32_t i = 0; i < rank; ++i)
    copy2d(M, 1, &A[int64_t(jpiv[i] - 1) * int64_t(lda)], lda, &matC[int64_t(i) * int64_t(M)], M);
  for (int32_t i = 0; i < N; ++i)
    copy2d(1, rank, &R[int64_t(i) * int64_t(ldr)], 1, &matR[jpiv[i] - 1], N);
  copy2d(M, N, A, lda, &matB[0], M);
  if constexpr(Complex) { for (std::complex<double>& f : matR) { f = std::conj(f); }}
  nngemm(M, N, rank, &matC[0], M, &matR[0], N, &matB[0], M);
  double err = std::transform_reduce(matB.begin(), matB.end(), 0., std::plus<double>(), [](auto i) { return std::norm(i); });
  return err;
}

int32_t main(int32_t argc, char* argv[]) {
  auto cu_err = cudaSetDevice(0);
  if (cu_err != cudaSuccess)
  { fprintf(stderr, "%s\n", cudaGetErrorString(cu_err)); return -1; }

  magma_init();
  int64_t M = 1 < argc ? std::atoi(argv[1]) : 2048;
  int64_t N = 2 < argc ? std::atoi(argv[2]) : 2048;
  N = std::min(M, N);

  double epi = 3 < argc ? std::atof(argv[3]) : 1.e-12;

  magma_queue_t queue = nullptr;
  magma_queue_create(0, &queue);
  cublasHandle_t cublasH = magma_queue_get_cublas_handle(queue);
  cudaStream_t stream = magma_queue_get_cuda_stream(queue);

  magmaDoubleComplex* dA, *dB, *dC;
  double* dR;
  std::vector<magmaDoubleComplex> tau(N);
  std::vector<magma_int_t> jpvt(N, 0);

  std::vector<std::complex<double>> matA(M * N), matB(M * N);
  matrix_generator<std::complex<double>>(M, N).generate_block(1., 512, 512, &matA[0], M);
  
  cudaMalloc((void**)&dA, M * N * sizeof(std::complex<double>));
  cudaMalloc((void**)&dB, M * N * sizeof(std::complex<double>));
  cudaMalloc((void**)&dR, 2 * N * sizeof(double));
  cudaMemcpy(dA, &matA[0], M * N * sizeof(std::complex<double>), cudaMemcpyHostToDevice);

  magma_int_t Lwork, info;
  magmaDoubleComplex work;
  magma_zgeqp3(M, N, nullptr, M, jpvt.data(), tau.data(), &work, -1, dR, &info);
  Lwork = std::max(int64_t(16) * (magma_int_t)(work.x), M * N);
  cudaMalloc((void**)&dC, int64_t(Lwork) * sizeof(std::complex<double>));

  magma_zgeqp3_gpu(M, N, dA, M, jpvt.data(), tau.data(), dC, Lwork, dR, &info);
  cudaStreamSynchronize(stream);

  cudaMemcpy(&matB[0], dA, M * N * sizeof(std::complex<double>), cudaMemcpyDeviceToHost);
  double s0 = epi * std::abs(matB[0].real());
  int64_t rank = 0; while (rank < N && s0 <= std::abs(matB[rank * (M + 1)].real())) { ++rank; }

  union { std::complex<double> std; magmaDoubleComplex magma; } one{std::complex<double>(1., 0.)}, zero{std::complex<double>(0., 0.)};
  cublasZtrsm(cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, rank, N - rank, (cuDoubleComplex*)&one, dA, M, &dA[rank * M], M);
  magmablas_zlaset(MagmaFull, rank, rank, zero.magma, one.magma, dA, M, queue);
  cudaStreamSynchronize(stream);

  cudaMemcpy(&matB[0], dA, M * N * sizeof(std::complex<double>), cudaMemcpyDeviceToHost);
  double nrm = fnorm(M, N, &matA[0], M);
  double err = nrm == 0. ? std::numeric_limits<double>::quiet_NaN() : std::sqrt(check_answer_lra(rank, M, N, matA.data(), M, jpvt.data(), matB.data(), M) / nrm);

  cudaMemcpy(dA, &matA[0], M * N * sizeof(std::complex<double>), cudaMemcpyHostToDevice);
  std::fill(jpvt.begin(), jpvt.end(), 0);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, stream);
  magma_zgeqp3_gpu(M, N, dA, M, jpvt.data(), tau.data(), dC, Lwork, dR, &info);
  cublasZtrsm(cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, rank, N - rank, (cuDoubleComplex*)&one, dA, M, &dA[rank * M], M);
  cudaEventRecord(stop, stream);
  cudaStreamSynchronize(stream);

  float milliseconds = 0.0f;
  cudaEventElapsedTime(&milliseconds, start, stop);
  int64_t qr_flops = (M * N * N * 2) - (N * N * N * 2 / 3);
  int64_t trsm_flops = N * rank * rank;
  double gflops = double(qr_flops + trsm_flops) * 1.e-6 / milliseconds;
  std::cout << "magma-ZLRA," << M << "," << N << "," << epi << "," << err << "," << rank << "," << milliseconds << "," << gflops << std::endl;
  
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  cudaFree(dR);

  magma_queue_destroy(queue);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  magma_finalize();

  cu_err = cudaGetLastError();
  if (cu_err != cudaSuccess)
    std::cerr << cudaGetErrorString(cu_err) << std::endl;
  return 0;
}
