
#include <commons.hpp>

void d2i(int64_t N, const double Xd[], int8_t Xi[]) {
  std::transform(Xd, &Xd[N], Xi, [](double e) { 
    int32_t i = (int32_t)e; return (int8_t)std::min(std::max(i, -128), 127); });
}

void i2d(int64_t N, const int8_t Xi[], double Xd[]) {
  std::transform(Xi, &Xi[N], Xd, [](float e) { return static_cast<double>(e); });
}

int32_t main() {
  mkl_verbose(1);

  int64_t M = 400, N = 40;
  Eigen::MatrixXcd matA(M, N);
  random_vector(M * N * 2, (double*)matA.data());

  Eigen::MatrixXcd ref = matA;

  for (int8_t i = 0; i < 10; ++i) {
    double nrm = matA.lpNorm<Eigen::Infinity>();
    Eigen::MatrixXcd matB = matA * 128. / nrm;

    std::vector<int8_t> matAi(M * N * 2);
    d2i(M * N * 2, (double*)matB.data(), matAi.data());

    Eigen::MatrixXcd reA(M, N);
    i2d(M * N * 2, matAi.data(), (double*)reA.data());

    double scale = nrm / 128.;
    matA -= reA * scale;
    printf("iter: %d, nrm: %e, remainder: %e\n", i, scale, matA.norm() / ref.norm());
  }
  return 0;
}
