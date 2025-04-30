
#include <complex>
#include <vector>
#include <algorithm>
#include <numeric>
#include <limits>
#include <random>
#include <iostream>
#include <commons.hpp>

#define MKL_COMPLEX_16 std::complex<double>
#define EIGEN_USE_MKL
#include <mkl.h>
#include <eigen3/Eigen/Dense>

int64_t computeBasis(double epi, int64_t M, int64_t N, int64_t K, std::complex<double>* U, std::complex<double>* A) {
  int64_t rank = std::min(M, N);
  int64_t block = std::max(M + K, static_cast<int64_t>(1024));
  std::vector<double> S(M + M);
  std::vector<std::complex<double>> B(M * block);
  const std::complex<double> zero(0., 0.), one(1., 0.), minus_one(-1., 0.);

  if (M < N)
    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasConjTrans, M, M, N, &one, A, M, A, M, &zero, &B[M * K], M);
  else
    std::copy_n(A, M * N, &B[M * K]);

  if (0 < K) {
    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasConjTrans, K, rank, M, &one, U, M, &B[M * K], M, &zero, &B[0], K);
    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, rank, K, &minus_one, U, M, &B[0], K, &one, &B[M * K], M);
    std::copy_n(U, M * K, B.begin());
  }
  LAPACKE_zgesvd(LAPACK_COL_MAJOR, 'O', 'N', M, rank, (lapack_complex_double*)&B[M * K], M, &S[0], nullptr, 1, nullptr, 1, &S[rank]);

  if (M < N) {
    double s0 = S[0] * std::numeric_limits<double>::epsilon();
    std::transform(S.begin(), S.begin() + M, S.begin(), [=](double e) { return s0 < e ? std::sqrt(e) : 0.; });
  }

  if (S[0] < std::numeric_limits<double>::min())
    rank = K;
  else if (1. <= epi)
    rank = std::min(rank, static_cast<int64_t>(epi));
  else {
    double s0 = S[0] * epi;
    rank = K + std::distance(S.begin(), std::find_if(S.begin(), S.begin() + (rank - K), [=](double s) { return s < s0; }));
  }
  std::cout << S[0] << ", " << S[rank - 1] << std::endl;

  std::vector<std::complex<double>> tau(M);
  LAPACKE_zgeqrf(LAPACK_COL_MAJOR, M, rank, (lapack_complex_double*)&B[0], M, (lapack_complex_double*)&tau[0]);
  LAPACKE_zungqr(LAPACK_COL_MAJOR, M, M, rank, (lapack_complex_double*)&B[0], M, (lapack_complex_double*)&tau[0]);
  std::copy_n(&B[M * K], M * (M - K), &U[M * K]);

  for (int64_t col = 0; col < N && 0 < rank; col += block) {
    int64_t L = std::min(N - col, block);
    LAPACKE_zlacpy(LAPACK_COL_MAJOR, 'A', M, L, (const lapack_complex_double*)&A[col * M], M, (lapack_complex_double*)&B[0], M);
    cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, M, L, M, &one, U, M, &B[0], M, &zero, &A[col * M], M);
  }
  return rank;
}

int32_t main() {

  int64_t M = 100, N = 2000;
  std::vector<std::complex<double>> matA(M * N), matU(M * M);
  random_vector(M * N * 2, (double*)matA.data());

  Eigen::MatrixXcd AAT = Eigen::Map<Eigen::MatrixXcd>(matA.data(), M, N) * Eigen::Map<Eigen::MatrixXcd>(matA.data(), M, N).adjoint();
  for (int32_t i = 0; i < 6; ++i) {
    AAT /= AAT.norm();
    AAT = AAT * AAT.adjoint();
  }
  AAT /= AAT.norm();
  Eigen::Map<Eigen::MatrixXcd>(matA.data(), M, N) = AAT * Eigen::Map<Eigen::MatrixXcd>(matA.data(), M, N);

  Eigen::MatrixXcd ref = Eigen::Map<Eigen::MatrixXcd>(matA.data(), M, N);

  double epi = 1.e-10;
  Eigen::JacobiSVD<Eigen::MatrixXcd> svd(Eigen::Map<Eigen::MatrixXcd>(matA.data(), M, N), Eigen::ComputeThinU | Eigen::ComputeThinV);
  svd.setThreshold(epi);
  int64_t rank_svd = svd.rank();
  Eigen::MatrixXcd reA = svd.matrixU().leftCols(rank_svd) * svd.singularValues().topRows(rank_svd).asDiagonal() * svd.matrixV().leftCols(rank_svd).adjoint();
  std::cout << rank_svd << ", " << (reA - ref).norm() / ref.norm() << std::endl;

  int64_t rank = 0;
  Eigen::Map<Eigen::MatrixXcd>(matU.data(), M, M) = Eigen::MatrixXcd::Identity(M, M);

  for (int64_t i = 0; i < 3; ++i) {
    Eigen::MatrixXcd loA = Eigen::Map<Eigen::MatrixXcd>(matA.data(), M, N).bottomRows(M - rank);
    Eigen::MatrixXcd loU(M - rank, M - rank);

    int64_t rank2 = computeBasis(std::pow(epi, 1. / 3.), M - rank, N, 0, loU.data(), loA.data());

    Eigen::Map<Eigen::MatrixXcd>(matU.data(), M, M).rightCols(M - rank) = Eigen::Map<Eigen::MatrixXcd>(matU.data(), M, M).rightCols(M - rank) * loU;
    Eigen::Map<Eigen::MatrixXcd>(matA.data(), M, N).bottomRows(M - rank) = loA;
    rank = rank + rank2;
    std::cout << i << ", " << rank2 << std::endl;
  }

  Eigen::MatrixXcd test = Eigen::Map<Eigen::MatrixXcd>(matU.data(), M, rank) * Eigen::Map<Eigen::MatrixXcd>(matA.data(), M, N);
  Eigen::MatrixXcd test2 = Eigen::Map<Eigen::MatrixXcd>(matU.data(), M, rank) * Eigen::Map<Eigen::MatrixXcd>(matU.data(), M, rank).adjoint() * ref;

  std::cout << rank << ", " << (test - ref).norm() / ref.norm() << ", " << (test2 - ref).norm() / ref.norm() << std::endl;

  return 0;
}
