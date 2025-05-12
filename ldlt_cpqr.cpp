
#include <commons.hpp>

int32_t main() {
  mkl_verbose(1);
  int64_t M = 800, N = 100;
  Eigen::MatrixXcd matA(M, N);
  random_vector(M * N * 2, (double*)matA.data());

  matA /= matA.norm();
  Eigen::MatrixXcd AAT = matA.adjoint() * matA;

  for (int32_t i = 0; i < 5; ++i) {
    AAT /= AAT.norm();
    AAT = AAT.adjoint() * AAT;
  }
  matA = matA * AAT;
  AAT = matA.adjoint() * matA;

  double epi = 1.e-7;
  Eigen::LDLT<Eigen::MatrixXcd, Eigen::Upper> ldlt(AAT);
  Eigen::MatrixXcd ldl = ldlt.matrixLDLT();
  auto piv = ldlt.transpositionsP().indices();
  int32_t rank = 0;
  double s0 = ldl(0, 0).real();
  for (int32_t i = 0; i < std::min(M, N); ++i)
    if (epi <= std::sqrt(ldl(i, i).real() / s0))
      rank = i + 1;

  Eigen::ColPivHouseholderQR<Eigen::MatrixXcd> cpqr(matA);
  Eigen::MatrixXcd qr = cpqr.matrixQR();
  auto cp = cpqr.colsPermutation().indices();
  cpqr.setThreshold(epi);

  for (int32_t i = 0; i < N; ++i)
    std::cout << std::sqrt(ldl(i, i).real()) << ", " << piv(i) << ", " << qr(i, i).real() << ", " << cp(i) << std::endl;

  Eigen::FullPivLU<Eigen::MatrixXcd> lu(AAT);
  Eigen::MatrixXcd luA = lu.matrixLU();

  int32_t rank_lu = 0;
  s0 = luA(0, 0).real();
  for (int32_t i = 0; i < std::min(M, N); ++i)
    if (epi <= std::sqrt(luA(i, i).real() / s0))
      rank_lu = i + 1;

  auto piv_i = lu.permutationP().indices();
  auto piv_j = lu.permutationQ().indices();

  for (int32_t i = 0; i < N; ++i)
    std::cout << std::sqrt(luA(i, i).real()) << ", " << qr(i, i).real() << std::endl;

  std::cout << "rank: " << rank << ", " << cpqr.rank() << ", " << rank_lu << std::endl;

  return 0;
}
