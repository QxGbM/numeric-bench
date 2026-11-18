#pragma once
#include <common.hpp>

// Leading-order real-operation equivalents, independent of storage precision.
template <class T> constexpr double benchmark_flop_factor =
  std::is_same_v<T, std::complex<double>> || std::is_same_v<T, std::complex<float>> ||
  std::is_same_v<T, cuDoubleComplex> || std::is_same_v<T, cuComplex> ||
  std::is_same_v<T, __half2> ? 4. : 1.;

// Full Householder GEQP3 without forming Q, plus the interpolative solve
// R11 * X = R12: R11 is rank-by-rank, R12 is rank-by-(N-rank).
inline double interpolative_flops(double m, double n, double rank, double scalar_factor) {
  const double k = std::min(m, n);
  const double qr = 2. * m * n * k - (2. / 3.) * k * k * k;
  const double trsm = rank * rank * (n - rank);
  return scalar_factor * (qr + trsm);
}

// Leading-order randomized SVD estimate for jobu='N', jobvt='S'.
// l is the sketch width (target rank + oversampling); p is the power count.
// Includes range finding/projection and reduced factorization work, without
// the final multiplication that would form the left singular vectors.
inline double randomized_svd_flops(double m, double n, double l, double p, double scalar_factor) {
  return scalar_factor * (4. * (p + 1.) * m * n * l + l * l * (4. * m + 2. * n));
}
