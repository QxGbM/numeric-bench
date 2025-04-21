#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include <complex>
#include <commons.hpp>

#define MKL_COMPLEX_16 std::complex<double>
#define EIGEN_USE_MKL
#include <mkl.h>
#include <eigen3/Eigen/Dense>

// https://math.nist.gov/iml++/gmres.h.txt

template <typename Preconditioner>
int GMRES(const Eigen::MatrixXcd &A, Eigen::VectorXcd &x, const Eigen::VectorXcd &b, const Preconditioner &M, int &m, int &max_iter, double &tol) {
  using Eigen::VectorXcd;
  using Eigen::MatrixXcd;

  double normb = M.solve(b).norm();
  VectorXcd r = M.solve(b - A * x);
  double beta = r.norm();
  
  if (normb == 0.0)
    normb = 1;
  
  double resid = beta / normb;
  std::cout << "0, " << resid << std::endl;
  if (resid <= tol) {
    tol = resid;
    max_iter = 0;
    return 0;
  }

  for (int j = 1; j <= max_iter; j++) {
    MatrixXcd v = MatrixXcd::Zero(A.rows(), m+1);
    v.col(0) = r * (1.0 / beta);
    MatrixXcd H = MatrixXcd::Zero(m+1, m);
    
    for (int i = 0; i < m; i++) {
      VectorXcd w = M.solve(A * v.col(i));
      for (int k = 0; k <= i; k++)
        H(k, i) = v.col(k).adjoint()*w;
      for (int k = 0; k <= i; k++)
        w -= H(k, i) * v.col(k);
      H(i+1, i) = w.norm();
      v.col(i+1) = w * (1.0 / H(i+1, i));
    }

    VectorXcd s = VectorXcd::Zero(m+1);
    s(0) = beta;
    VectorXcd y = H.colPivHouseholderQr().solve(s);
    x += v.leftCols(m) * y;

    r = M.solve(b - A * x);
    beta = r.norm();
    resid = beta / normb;
    std::cout << j << ", " << resid << std::endl;
    if (resid < tol) {
      tol = resid;
      max_iter = j;
      return 0;
    }
  }
  
  tol = resid;
  return 1;
}

int main() {
  // Example usage:
  // Define the matrix A and the vector b
  
  int N = 200;
  Eigen::MatrixXcd A(N, N);
  Eigen::VectorXcd b(N), x = Eigen::VectorXcd::Zero(N);

  std::mt19937_64 gen;
  std::normal_distribution<double> dist(0., 1.);

  random_vector(N * N * 2, (double*)A.data());
  random_vector(N * 2, (double*)b.data());

  auto precond = Eigen::MatrixXcd::Identity(N, N).colPivHouseholderQr();

  int iterations = 40;
  int restarts = 50;
  double residual_norm = 1.e-12;

  GMRES(A, x, b, precond, restarts, iterations, residual_norm);
  std::cout << "Iterations: " << iterations << std::endl;
  std::cout << "Residual Norm: " << (b-A*x).norm() / b.norm() << std::endl;

  return 0;
}