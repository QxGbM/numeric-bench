
#include <commons.hpp>

int32_t main() {
  mkl_verbose(0);
  int64_t M = 2400, N = 100;
  Eigen::MatrixXcd matA(M, N);
  random_vector(M * N * 2, (double*)matA.data());

  Eigen::MatrixXcd AAT = matA.adjoint() * matA;
  for (int32_t i = 0; i < 5; ++i) {
    AAT /= AAT.norm();
    AAT = AAT.adjoint() * AAT;
  }

  matA = matA * AAT;
  matA /= matA.norm();
  AAT = matA.adjoint() * matA;

  double epi = 1.e-5;
  Eigen::MatrixXcd ldl = AAT;
  Eigen::MatrixXcd ax(N, N);
  std::vector<int32_t> piv(N);

  int32_t k = std::min(M, N), rank = k;
  double s0 = 0.;
  for (int32_t i = 0; i < rank; ++i) {
    int32_t id = cblas_izamax(k, ldl.data(), k + 1);
    ax.col(i) = ldl.col(id);
    double s = -1. / ldl(id, id).real();
    ldl += s * ax.col(i) * ax.col(i).adjoint();
    piv[i] = id;

    ax.col(i) *= -s;
    //Eigen::MatrixXcd l = ax.row(id).leftCols(i);
    //ax.leftCols(i) -= ax.col(i) * l;

    if (i == 0)
      s0 = -s;
    if (std::sqrt(s0 / -s) < epi)
      rank = i;
  }

  for (int32_t i = rank - 1; 0 < i; --i) {
    int32_t id = piv[i];
    Eigen::MatrixXcd l = ax.row(id).leftCols(i);
    ax.leftCols(i) -= ax.col(i) * l;
  }

  /*Eigen::MatrixXcd l(rank, rank);
  for (int32_t i = 0; i < rank; ++i)
    l.row(i) = ax.row(piv[i]).leftCols(rank);
  std::complex<double> alpha = 1.;
  cblas_ztrsm(CblasColMajor, CblasRight, CblasLower, CblasNoTrans, CblasUnit, k, rank, &alpha, l.data(), rank, ax.data(), k);*/

  Eigen::MatrixXcd As(M, rank);
  for (int32_t i = 0; i < rank; ++i)
    As.col(i) = matA.col(piv[i]);

  std::cout << "id error: " << (matA - As * ax.leftCols(rank).adjoint()).norm() / matA.norm() << std::endl;

  Eigen::ColPivHouseholderQR<Eigen::MatrixXcd> cpqr(matA);
  Eigen::MatrixXcd qr = cpqr.matrixQR();
  auto cp = cpqr.colsPermutation().indices();
  cpqr.setThreshold(epi);

  std::cout << "rank id: " << rank << ", rank cpqr: " << cpqr.rank() << std::endl;

  return 0;
}
