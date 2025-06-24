
#include <commons.hpp>

// Complex-Cholesky factorization with diagonal pivoting
void zpotrfp(int32_t N, std::complex<double>* A, int32_t lda, int32_t* ipiv) {
  std::vector<double> diag(N); // copy out diagonals of A, for faster imax
  for (int32_t i = 0; i < N; ++i)
    diag[i] = A[i * (lda + 1)].real();

  for (int32_t i = 0; i < N; ++i) {
    int32_t id = i; // izmax, assumes diagonal are always positive, skip LAPACK pos-def checks
    for (int32_t j = i + 1; j < N; ++j)
      if (diag[id] < diag[j])
        id = j;
    
    double s = 1. / std::sqrt(diag[id]); // rsqrt of diagonal
    ipiv[i] = id; // record pivot

    if (i != id) {
      std::iter_swap(&diag[i], &diag[id]);
      std::iter_swap(&A[i + i * lda], &A[id + i * lda]);
      std::iter_swap(&A[i + id * lda], &A[id + id * lda]); // for diagonal and both columns, exchange only the row element
      for (int32_t j = 0; j < N; ++j) // exchange column i with column id
        std::iter_swap(&A[j + i * lda], &A[j + id * lda]);
      for (int32_t j = 0; j < N; ++j) // write row id with entries inside column id
        A[id + j * lda] = std::conj(A[j + id * lda]);
      // delay write to row i as column i will be updated immediately after
    }

    for (int32_t j = i; j < N; ++j) // left-looking Cholesky factorization, delay update to column i (exchanged) till current iteration
      for (int32_t k = 0; k < i; ++k) // access the upper triangular part of A so that the dot-prod is coalescing
        A[j + i * lda] -= std::conj(A[k + j * lda]) * A[k + i * lda];
      
    for (int32_t j = i; j < N; ++j) // divide by diagonal element
      A[j + i * lda] *= s;

    for (int32_t j = 0; j < N; ++j) // now write row i with updated column i
      A[i + j * lda] = std::conj(A[j + i * lda]);

    for (int32_t j = i + 1; j < N; ++j) { // update the diagonal entries (only real part), skip [0, i] as they will not be in the next pivoting selections
      double rl = A[j + i * lda].real();
      double im = A[j + i * lda].imag();
      diag[j] += -rl * rl - im * im;
    }
  }
}

int32_t main() {
  int32_t N = 10;
  Eigen::MatrixXcd matA(N, N);
  random_vector(N * N * 2, (double*)matA.data());
  for (int32_t i = 0; i < N; ++i) {
    for (int32_t j = 0; j < i; ++j)
      matA(j, i) = std::conj(matA(i, j));
    matA(i, i) = std::complex<double>(1.e3+i, 0.);
  }

  Eigen::MatrixXcd matB = matA;
  std::vector<int32_t> ipiv(N);
  zpotrfp(N, matB.data(), N, ipiv.data());

  Eigen::MatrixXcd matL = matA.llt().matrixLLT().adjoint().triangularView<Eigen::Upper>();
  std::cout << matL.adjoint()*matL << std::endl;

  Eigen::MatrixXcd matLB = matB.triangularView<Eigen::Upper>();
  std::cout << matLB.adjoint()*matLB << std::endl;

  /*Eigen::MatrixXcf matAf(M, N);
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

  double epi = 1.e-6;
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
  std::cout <<"rank cpqr: " << cpqr.rank() << std::endl;*/

  return 0;
}
