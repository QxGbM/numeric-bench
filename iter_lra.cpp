
#include <commons.hpp>

int32_t main() {
  mkl_verbose(0);

  int32_t M = 200, N = 50;
  Eigen::MatrixXd matA(M, N);
  random_vector(M * N, (double*)matA.data());

  Eigen::MatrixXd AAT = matA.adjoint() * matA;
  for (int32_t i = 0; i < 5; ++i) {
    AAT /= AAT.norm();
    AAT = AAT.adjoint() * AAT;
  }
  matA = matA * AAT;

  Eigen::MatrixXd ref = matA;

  Eigen::ColPivHouseholderQR<Eigen::MatrixXd> cpqr(matA);
  Eigen::MatrixXd qr = cpqr.matrixQR();
  cpqr.setThreshold(1.e-12);

  printf("rank cpqr: %d (%e, %e)\n", cpqr.rank(), qr(cpqr.rank() - 1, cpqr.rank() - 1), qr(0, 0));
  
  int32_t rank_tot = 0;
  for (int32_t i = 0; i < 10; ++i) {
    double nrm = matA.lpNorm<Eigen::Infinity>();
    Eigen::MatrixXd matB = matA * (127. / nrm);

    Eigen::MatrixXf matAi(M, N);
    std::transform(matB.data(), matB.data() + M * N, matAi.data(), [](double e) { return float(e); });

    Eigen::JacobiSVD<Eigen::MatrixXf> svd(matAi.transpose() * matAi, Eigen::ComputeThinU); // fp32
    svd.setThreshold(1.e-6);

    int32_t rank = svd.rank();
    Eigen::MatrixXf V = svd.matrixU().leftCols(rank);
    Eigen::MatrixXf U = matAi * V;

    Eigen::MatrixXd Vd(N, rank), Ud(M, rank);
    std::transform(V.data(), V.data() + N * rank, Vd.data(), [](float e) { return double(e); });
    std::transform(U.data(), U.data() + M * rank, Ud.data(), [](float e) { return double(e); });

    Eigen::MatrixXd appx = Ud * Vd.transpose();
    matA -= (nrm / 127.) * appx;

    printf("%d %e %e %d\n", i, (appx - matB).norm() / matB.norm(), matA.norm(), rank_tot += rank);
  }
  return 0;
}
