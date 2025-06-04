
#include <commons.hpp>

void zpotrfp(int32_t N, std::complex<double>* A, int32_t lda, int32_t* ipiv) {
  const std::complex<double> minus_one(-1., 0.);
  for (int32_t i = 0; i < N; ++i) {
    int32_t id = i + cblas_izamax(N - i, &A[i * (lda + 1)], lda + 1);
    cblas_zswap(N, &A[i], lda, &A[id], lda);
    cblas_zswap(N, &A[i * lda], 1, &A[id * lda], 1);
    double s = std::sqrt(A[i * (lda + 1)].real());

    ipiv[i] = id;
    A[i * (lda + 1)] = s;
    if (i + 1 != N) {
      cblas_zdscal(N - i - 1, 1. / s, &A[i * lda + (i + 1)], 1);
      cblas_zdscal(N - i - 1, 1. / s, &A[(i + 1) * lda + i], lda);
      cblas_zgerc(CblasColMajor, N - i - 1, N - i - 1, &minus_one, &A[i * lda + (i + 1)], 1, &A[i * lda + (i + 1)], 1, &A[(i + 1) * (lda + 1)], lda);
    }
  }
}

int32_t main() {
  int32_t M = 300, N = 100;
  Eigen::MatrixXcd matA(M, N);
  random_vector(M * N * 2, (double*)matA.data());

  Eigen::MatrixXcf matAf(M, N);
  std::transform((double*)matA.data(), (double*)matA.data() + M * N * 2, (float*)matAf.data(), [](double e) { return float(e); });

  Eigen::MatrixXcf AAT = matAf.adjoint() * matAf;
  for (int32_t i = 0; i < 4; ++i) {
    AAT /= AAT.norm();
    AAT = AAT.adjoint() * AAT;
  }

  matAf = matAf * AAT;
  float scale = float(1 << 23) / matAf.lpNorm<Eigen::Infinity>();
  std::transform((float*)matAf.data(), (float*)matAf.data() + M * N * 2, (float*)matAf.data(), [=](float e) { return std::round(scale * e); });

  std::transform((float*)matAf.data(), (float*)matAf.data() + M * N * 2, (double*)matA.data(), [](float e) { return double(e); });

  double epi = 1.e-5;
  Eigen::MatrixXcd ldl = matA.adjoint() * matA;
  std::vector<int32_t> piv(N);

  zpotrfp(N, ldl.data(), N, piv.data());
  ldl = ldl.triangularView<Eigen::Upper>();

  Eigen::MatrixXcf R(N, N);
  std::transform((double*)ldl.data(), (double*)ldl.data() + N * N * 2, (float*)R.data(), [](double e) { return float(std::round(e)); });

  Eigen::ColPivHouseholderQR<Eigen::MatrixXcd> cpqr(matA);
  Eigen::MatrixXcd qr = cpqr.matrixQR().topRows(N).triangularView<Eigen::Upper>();
  auto cp = cpqr.colsPermutation().indices();
  cpqr.setThreshold(epi);

  for (int32_t i = 0; i < N; ++i)
    if (qr(i, i).real() < 0)
      qr.row(i) = -qr.row(i);

  Eigen::MatrixXcf qr_f(N, N);
  std::transform((double*)qr.data(), (double*)qr.data() + N * N * 2, (float*)qr_f.data(), [](double e) { return float(std::round(e)); });

  std::cout << (qr_f - R).norm() / qr_f.norm() << std::endl;
  std::cout <<"rank cpqr: " << cpqr.rank() << std::endl;

  return 0;
}
