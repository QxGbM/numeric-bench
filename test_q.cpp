
#include <commons.hpp>

int main() {
  int M = 100;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(0., 1.);

  Eigen::MatrixXcd A(M, M);
  Eigen::VectorXcd x(M);

  random_vector(M * M * 2, (double*)A.data());
  random_vector(M * 2, (double*)x.data());

  std::complex<double> mu(1., 0.);
  auto lu = (A - mu * Eigen::MatrixXcd::Identity(M, M)).lu();
  x = x / x.norm();
  Eigen::VectorXcd y = lu.solve(x);

  for (int i = 0; i < 100; ++i) {
    x = y / y.norm();
    y = lu.solve(x);

    std::complex<double> lambda = x.adjoint() * y;
    std::complex<double> rl = mu + (1. / lambda);

    double err = ((A - rl * Eigen::MatrixXcd::Identity(M, M)) * x).norm();
    std::cout << rl << ", " << err << std::endl;
  }

  std::complex<double> lambda = x.adjoint() * y;
  std::complex<double> rl = mu + (1. / lambda);
  double err = ((A - rl * Eigen::MatrixXcd::Identity(M, M)) * x).norm();
  std::cout << rl << ", " << err << std::endl;

  return 0;
}
