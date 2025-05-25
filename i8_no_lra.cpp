
#include <commons.hpp>

void d2i(int32_t N, const double Xd[], int8_t Xi[]) {
  std::transform(Xd, &Xd[N], Xi, [](double e) { 
    int32_t i = (int32_t)std::roundf((float)e); return (int8_t)std::min(std::max(i, -127), 127); });
}

void i2d(int32_t N, const int8_t Xi[], double Xd[]) {
  std::transform(Xi, &Xi[N], Xd, [](float e) { return static_cast<double>(e); });
}

int32_t main() {
  mkl_verbose(1);

  int32_t M = 400, N = 40;
  Eigen::MatrixXcd matA(M, N);
  random_vector(M * N * 2, (double*)matA.data());

  Eigen::MatrixXcd ref = matA;
  double nrm = matA.lpNorm<Eigen::Infinity>();

  for (int8_t i = 0; i < 10; ++i) {
    Eigen::MatrixXcd matB = matA * (127. / nrm);

    std::vector<int8_t> matAi(M * N * 2);
    d2i(M * N * 2, (double*)matB.data(), matAi.data());

    Eigen::MatrixXcd reA(M, N);
    i2d(M * N * 2, matAi.data(), (double*)reA.data());

    double scale = nrm / 127.;
    matA -= reA * scale;
    nrm = scale;
    printf("iter: %d, nrm: %e, remainder: %e\n", i, scale, matA.norm() / ref.norm());
  }
  return 0;
}
