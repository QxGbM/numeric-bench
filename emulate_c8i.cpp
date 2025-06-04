
#include <commons.hpp>
#include <array>

void d2i(int32_t N, const double Xd[], int8_t Xi[]) {
  std::transform(Xd, &Xd[N], Xi, [](double e) { 
    int32_t i = (int32_t)e; return (int8_t)std::min(std::max(i, -127), 127); });
}

void conj_c8i(int32_t N, const int8_t X[], int8_t Y[]) {
  std::transform((const int16_t*)X, &((const int16_t*)X)[N], (int16_t*)Y, [](int16_t e) {
    int8_t lo = e & 0xFF;
    int8_t hi = e >> 8;
    return (lo << 8) | (uint8_t)(-hi); });
}

void i2d(int32_t N, const int8_t Xi[], double Xd[]) {
  std::transform(Xi, &Xi[N], Xd, [](float e) { return static_cast<double>(e); });
}

int32_t main() {
  int64_t M = 100, N = 100;
  Eigen::MatrixXcd matA(M, N);
  random_vector(M * N * 2, (double*)matA.data());

  Eigen::MatrixXcd AAT = matA.adjoint() * matA;
  for (int32_t i = 0; i < 7; ++i) {
    AAT /= AAT.norm();
    AAT = AAT.adjoint() * AAT;
  }

  matA = matA * AAT;
  matA /= matA.norm();

  Eigen::MatrixXd matAs(2 * M, N), matAsc(2 * M, N);
  std::copy((double*)matA.data(), (double*)matA.data() + 2 * M * N, matAs.data());
  double scale = matAs.lpNorm<Eigen::Infinity>() / 128.;
  matAs *= 1. / scale;

  std::vector<int8_t> vecAi(M * N * 2), vecAic(M * N * 2);
  d2i(M * N * 2, matAs.data(), vecAi.data());
  conj_c8i(M * N, vecAi.data(), vecAic.data());

  i2d(M * N * 2, vecAi.data(), matAs.data());
  i2d(M * N * 2, vecAic.data(), matAsc.data());

  Eigen::MatrixXd realA = matAs.transpose() * matAs;
  Eigen::MatrixXd imagA = matAsc.transpose() * matAs;
  Eigen::MatrixXcd reA(N, N);

  for (int32_t j = 0; j < N; ++j)
    for (int32_t i = 0; i < N; ++i)
      reA(i, j) = std::complex<double>(realA(i, j), imagA(i, j));

  Eigen::MatrixXcd matAp(M, N);
  std::copy(matAs.data(), matAs.data() + 2 * M * N, (double*)matAp.data());
  Eigen::MatrixXcd ref = matAp.adjoint() * matAp;
  
  std::cout << (ref - reA).norm() / ref.norm() << std::endl;
  return 0;
}
