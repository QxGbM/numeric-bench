
#include <commons.hpp>
#include <hyacinth.hpp>

void d2i(int32_t N, const double Xd[], int8_t Xi[]) {
  std::transform(Xd, &Xd[N], Xi, [](double e) { 
    int32_t i = (int32_t)e; return (int8_t)std::min(std::max(i, -128), 127); });
}

void i2d(int32_t N, const int8_t Xi[], double Xd[]) {
  std::transform(Xi, &Xi[N], Xd, [](float e) { return static_cast<double>(e); });
}

int32_t main() {
  int32_t M = 400, N = 40;
  Eigen::MatrixXcd matA(M, N);
  random_vector(M * N * 2, (double*)matA.data());

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  int32_t ld = align_c_i8(M);

  int8_t* gpu_i8 = nullptr;
  cudaMalloc((void**)&gpu_i8, 2 * ld * N * sizeof(int8_t));

  std::complex<double>* gpu_f64 = nullptr;
  cudaMalloc((void**)&gpu_f64, M * N * sizeof(std::complex<double>));

  Eigen::MatrixXcd ref = matA;

  for (int8_t i = 0; i < 10; ++i) {
    std::vector<int8_t> matAi(M * N * 2);
    cudaMemcpy(gpu_f64, matA.data(), M * N * sizeof(std::complex<double>), cudaMemcpyHostToDevice);
    double scale = c_f64_i8(stream, M, N, (const cuDoubleComplex*)gpu_f64, M, gpu_i8, ld);

    cudaStreamSynchronize(stream);
    cudaMemcpy2D(matAi.data(), 2 * M * sizeof(int8_t), gpu_i8, 2 * ld * sizeof(int8_t), 2 * M * sizeof(int8_t), N, cudaMemcpyDeviceToHost);

    Eigen::MatrixXcd reA(M, N);
    i2d(M * N * 2, matAi.data(), (double*)reA.data());

    matA -= reA * scale;
    printf("iter: %d, nrm: %e, remainder: %e\n", i, scale, matA.norm() / ref.norm());
  }

  cudaFree(gpu_i8);
  cudaFree(gpu_f64);
  cudaStreamDestroy(stream);
  return 0;
}
