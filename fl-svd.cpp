
#include <commons.hpp>

int32_t trunc_i (int32_t i) {
  // 0 <= n128 if i in range, sign = 0; not in range sign = -1
  int32_t s128 = (i + 128) >> 31;
  int32_t s127 = (127 - i) >> 31;
  return (~s128 & ~s127 & i) | (s128 & -128) | (s127 & 127);
}

double round_double(double f, double epi) {
  union { double fp; uint64_t u; } v {f};
  int32_t digits = -std::floor(std::log2(epi));
  int32_t exp = (int32_t(v.u >> 52) & 2047) - 1023;
  int64_t frac = int64_t(std::scalbn(f, digits - exp));
  return std::scalbn(double(frac), exp - digits);
}

int32_t main() {
  int64_t M = 100, N = 800;
  Eigen::MatrixXcd matA(M, N), matU(M, M);
  random_vector(M * N * 2, (double*)matA.data());

  Eigen::MatrixXcd AAT = matA * matA.adjoint();
  for (int32_t i = 0; i < 5; ++i) {
    AAT /= AAT.norm();
    AAT = AAT * AAT.adjoint();
  }
  matA = AAT * matA;

  Eigen::MatrixXcd ref = matA;

  Eigen::JacobiSVD<Eigen::MatrixXcd> svd(matA, Eigen::ComputeThinU | Eigen::ComputeThinV);
  svd.setThreshold(1.e-8);
  int64_t rank_svd = svd.rank();
  Eigen::MatrixXcd reA = svd.matrixU().leftCols(rank_svd) * svd.singularValues().topRows(rank_svd).asDiagonal() * svd.matrixV().leftCols(rank_svd).adjoint();
  std::cout << rank_svd << ", " << (reA - ref).norm() / ref.norm() << std::endl;
  std::cout << svd.singularValues() << std::endl;

  Eigen::MatrixXcd matD(M, N);
  std::transform(&(matA.data())[0], &(matA.data())[M * N], &(matD.data())[0], 
    [](std::complex<double> e) { return std::complex<double>(round_double(e.real(), 1.e-9), round_double(e.imag(), 1.e-9)); });

  svd = Eigen::JacobiSVD<Eigen::MatrixXcd>(matD, Eigen::ComputeThinU | Eigen::ComputeThinV);
  svd.setThreshold(1.e-8);
  rank_svd = svd.rank();
  Eigen::MatrixXcd reD = svd.matrixU().leftCols(rank_svd) * svd.singularValues().topRows(rank_svd).asDiagonal() * svd.matrixV().leftCols(rank_svd).adjoint();
  std::cout << rank_svd << ", " << (reD - ref).norm() / ref.norm() << std::endl;
  std::cout << svd.singularValues() << std::endl;

  double aa = -1.2345678923456789;
  double bb = round_double(aa, 1.e-9);
  printf("%.20le %.20le %.20le\n", aa, bb, std::abs(aa - bb));

  double A_nrm = 32. / matA.lpNorm<Eigen::Infinity>();

  Eigen::MatrixXcf matB(M, N); // fp16
  std::transform(&(matA.data())[0], &(matA.data())[M * N], &(matB.data())[0], 
    [=](std::complex<double> e) { return std::complex<float>(A_nrm * e.real(), A_nrm * e.imag()); });

  Eigen::MatrixXcd matBd(M, N);
  std::transform(&(matB.data())[0], &(matB.data())[M * N], &(matBd.data())[0], 
    [](std::complex<float> e) { return std::complex<double>(e.real(), e.imag()); });
  
  std::cout << "low-prec-matrix norm: " << (matA - matBd / A_nrm).norm() / matA.norm() << std::endl;

  Eigen::JacobiSVD<Eigen::MatrixXcf> svd_b(matB * matB.adjoint(), Eigen::ComputeThinU); // fp32
  svd_b.setThreshold(0.001953125);

  int64_t rank = svd_b.rank();

  Eigen::MatrixXcf Ub = svd_b.matrixU().leftCols(rank);
  Eigen::MatrixXcf Vb = Ub.adjoint() * matB;

  std::cout << "svd err: " << (matB - Ub*Vb).norm() / matB.norm() << std::endl;

  float Ub_nrm = 128.f / Ub.lpNorm<Eigen::Infinity>();
  float Vb_nrm = 128.f / Vb.lpNorm<Eigen::Infinity>();

  std::transform(&(Ub.data())[0], &(Ub.data())[M * rank], &(Ub.data())[0], 
    [=](std::complex<float> e) { return std::complex<float>(trunc_i((int32_t)(Ub_nrm * e.real())), trunc_i((int32_t)(Ub_nrm * e.imag()))); }); // int8
  std::transform(&(Vb.data())[0], &(Vb.data())[rank * N], &(Vb.data())[0], 
    [=](std::complex<float> e) { return std::complex<float>(trunc_i((int32_t)(Vb_nrm * e.real())), trunc_i((int32_t)(Vb_nrm * e.imag()))); }); //int8

  Eigen::MatrixXcf matC = Ub * Vb; // int32 or fp32
  Eigen::MatrixXcd matAsub(M, N);
  double scale = 1. / (A_nrm * (double)Ub_nrm * (double)Vb_nrm);
  std::transform(&(matC.data())[0], &(matC.data())[M * N], &(matAsub.data())[0], 
    [=](std::complex<float> e) { return std::complex<double>(scale * e.real(), scale * e.imag()); });
  
  std::cout << "scale: " << scale << std::endl;
  std::cout << "remainder norm: " << (matA - matAsub).norm() / matA.norm() << std::endl;
  std::cout << "int8-lra err: " << ((1./(Ub_nrm*Vb_nrm))*matC - matB).norm() / matB.norm() << std::endl;
  std::cout << "rank and theoretical error at rank: "<< rank << ", " << svd.singularValues()(rank) / svd.singularValues()(0) << std::endl;

  return 0;
}
