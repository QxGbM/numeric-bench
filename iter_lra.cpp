
#include <commons.hpp>

int32_t main() {
  mkl_verbose(0);

  int32_t M = 800, N = 100;
  Eigen::MatrixXd matA(M, N);
  random_vector(M * N, (double*)matA.data());

  Eigen::MatrixXd AAT = matA.adjoint() * matA;
  for (int32_t i = 0; i < 6; ++i) {
    AAT /= AAT.norm();
    AAT = AAT.adjoint() * AAT;
  }
  matA = matA * AAT;

  Eigen::MatrixXd ref = matA;

  Eigen::ColPivHouseholderQR<Eigen::MatrixXd> cpqr(matA);
  Eigen::MatrixXd matR = cpqr.matrixQR();
  cpqr.setThreshold(1.e-12);

  printf("true rank: %d \n", (int32_t)cpqr.rank());

  int32_t iters = 6, rank_tot = 0;
  Eigen::MatrixXd Uf(M, iters * N);
  Eigen::MatrixXd Vf(N, iters * N);
  for (int32_t i = 0; i < iters; ++i) {
    Eigen::MatrixXd matB = matA;
    double nrm = 127. / matB.lpNorm<Eigen::Infinity>();

    Eigen::MatrixXf matAi(M, N);
    std::transform(matB.data(), matB.data() + M * N, matAi.data(), [=](double e) { return float(nrm * e); });

    Eigen::JacobiSVD<Eigen::MatrixXf> svd(matAi.transpose() * matAi, Eigen::ComputeThinU);
    svd.setThreshold(1.e-6);

    int32_t rank = svd.rank();
    Eigen::MatrixXf V = svd.matrixU().leftCols(rank);
    Eigen::MatrixXf U = (1. / nrm) * matAi * V;

    Eigen::MatrixXd Vd(N, rank), Ud(M, rank);
    std::transform(V.data(), V.data() + N * rank, Vd.data(), [](float e) { return double(e); });
    std::transform(U.data(), U.data() + M * rank, Ud.data(), [](float e) { return double(e); });

    Vf.middleCols(rank_tot, rank) = Vd;
    Uf.middleCols(rank_tot, rank) = Ud;

    Eigen::MatrixXd appx = Uf.middleCols(rank_tot, rank) * Vf.middleCols(rank_tot, rank).transpose();
    matA -= appx;
    rank_tot += rank;

    printf("iter: %d, low-precision SVD: %e, remainder norm: %e, accumulator rank: %d\n", i + 1, (appx - matB).norm() / matB.norm(), matA.norm() / ref.norm(), rank_tot);
  }

  Eigen::HouseholderQR<Eigen::MatrixXd> qr(Vf.leftCols(rank_tot));
  matR = qr.matrixQR().triangularView<Eigen::Upper>();

  Eigen::JacobiSVD<Eigen::MatrixXd> svd(Uf.leftCols(rank_tot) * matR.transpose());
  svd.setThreshold(1.e-12);
  int32_t rank_svd = svd.rank();
  printf("rank svd: %d, %e\n", rank_svd, (Uf * Vf.transpose() - ref).norm() / ref.norm());

  return 0;
}
