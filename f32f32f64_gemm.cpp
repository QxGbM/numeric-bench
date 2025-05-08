
#include <commons.hpp>

void d2f(int64_t N, const double Xd[], float Xf[]) {
  std::transform(Xd, &Xd[N], Xf, [](double e) { return static_cast<float>(e); });
}

void f2d(int64_t N, const float Xf[], double Xd[]) {
  std::transform(Xf, &Xf[N], Xd, [](float e) { return static_cast<double>(e); });
}

void trunc_f(int64_t N, float Xf[]) {
  std::transform(Xf, &Xf[N], Xf, [] (float e) {
    uint32_t ie = *((uint32_t*)&e);
    //uint32_t sign = ie >> 31;
    //uint32_t c = (1 < (3 & ie)) << 2;
    uint32_t i = (0xFFFFFFFFF & ie);// + (sign ? 0 : c);
    return *((float*)&i);
  });
}

int32_t main() {
  mkl_verbose(1);
  std::cout.precision(14);
  std::cout << std::scientific;

  int64_t M = 400, N = 400, K = 40;
  Eigen::MatrixXcd matA(M, K), matB(K, N);
  random_vector(M * K * 2, (double*)matA.data());
  random_vector(K * N * 2, (double*)matB.data());

  Eigen::MatrixXcf matAf(M, K), matBf(K, N);
  matA *= std::sqrt(M * K) / matA.norm();
  matB *= std::sqrt(K * N) / matB.norm();

  d2f(M * K * 2, (double*)matA.data(), (float*)matAf.data());
  d2f(K * N * 2, (double*)matB.data(), (float*)matBf.data());

  f2d(M * K * 2, (float*)matAf.data(), (double*)matA.data());
  f2d(K * N * 2, (float*)matBf.data(), (double*)matB.data());

  Eigen::MatrixXcf matCf = matAf * matBf;
  //trunc_f(M * N * 2, (float*)matCf.data());

  Eigen::MatrixXcd matD = matA * matB;
  Eigen::MatrixXcf ref(M, N);
  d2f(M * N * 2, (double*)matD.data(), (float*)ref.data());
  //trunc_f(M * N * 2, (float*)ref.data());

  std::cout << (ref - matCf).norm() / ref.norm() << std::endl;
  return 0;
}
