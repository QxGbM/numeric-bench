#pragma once
#include <benchmark_common.hpp>

// Storage, compute precision, and BLAS operations must all agree.
template <class T> constexpr cudaDataType vendor_type = CUDA_C_64F;
template <> constexpr cudaDataType vendor_type<double> = CUDA_R_64F;
template <class T> constexpr char vendor_precision = 'Z';
template <> constexpr char vendor_precision<double> = 'D';
template <class T> constexpr double vendor_flop_factor = benchmark_flop_factor<T>;

inline cublasStatus_t vendor_gemm(cublasHandle_t h, cublasOperation_t a, cublasOperation_t b,
    int m, int n, int k, const double* alpha, const double* A, int lda,
    const double* B, int ldb, const double* beta, double* C, int ldc) {
  if (a == CUBLAS_OP_C) a = CUBLAS_OP_T;
  if (b == CUBLAS_OP_C) b = CUBLAS_OP_T;
  return cublasDgemm(h, a, b, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
inline cublasStatus_t vendor_gemm(cublasHandle_t h, cublasOperation_t a, cublasOperation_t b,
    int m, int n, int k, const std::complex<double>* alpha, const std::complex<double>* A, int lda,
    const std::complex<double>* B, int ldb, const std::complex<double>* beta, std::complex<double>* C, int ldc) {
  return cublasZgemm(h, a, b, m, n, k, reinterpret_cast<const cuDoubleComplex*>(alpha),
    reinterpret_cast<const cuDoubleComplex*>(A), lda, reinterpret_cast<const cuDoubleComplex*>(B), ldb,
    reinterpret_cast<const cuDoubleComplex*>(beta), reinterpret_cast<cuDoubleComplex*>(C), ldc);
}
inline cublasStatus_t vendor_norm(cublasHandle_t h, int64_t n, const double* A, int64_t inc, double* out) {
  return cublasDnrm2_64(h, n, A, inc, out);
}
inline cublasStatus_t vendor_norm(cublasHandle_t h, int64_t n, const std::complex<double>* A, int64_t inc, double* out) {
  return cublasDznrm2_64(h, n, reinterpret_cast<const cuDoubleComplex*>(A), inc, out);
}
inline cublasStatus_t vendor_gram(cublasHandle_t h, int n, int m, const double* A, int lda, double* C) {
  const double one = 1., zero = 0.;
  return cublasDsyrk(h, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, n, m, &one, A, lda, &zero, C, n);
}
inline cublasStatus_t vendor_gram(cublasHandle_t h, int n, int m, const std::complex<double>* A, int lda, std::complex<double>* C) {
  const double one = 1., zero = 0.;
  return cublasZherk(h, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_C, n, m, &one,
    reinterpret_cast<const cuDoubleComplex*>(A), lda, &zero, reinterpret_cast<cuDoubleComplex*>(C), n);
}

// Conventional Golub-Reinsch work estimate, M >= N. Bidiagonal QR work
// depends on convergence, so this is an estimate rather than a hardware count.
// The rectangular term is M*N^2; no M-by-M left-vector work is requested.
// Complex arithmetic uses the usual four-real-flop scaling.
inline double gesvd_flops(double m, double n, double scalar_factor) {
  return scalar_factor * (4. * m * n * n + 8. * n * n * n + 9. * n * n);
}

// Vendor benchmark storage is already FP64; no mixed-precision conversions.
template <class T> inline void require_vendor_fp64() {
  static_assert(std::is_same_v<T, double> || std::is_same_v<T, std::complex<double>>,
                "Vendor host checks require real or complex FP64");
}
template <class T>
inline void vendor_copy2d(int32_t M, int32_t N, const T* A, int32_t lda, T* B, int32_t ldb) {
  require_vendor_fp64<T>();
  for (int32_t j = 0; j < N; ++j)
    std::copy_n(A + int64_t(j) * lda, M, B + int64_t(j) * ldb);
}
template <class T>
inline double vendor_fnorm(int32_t M, int32_t N, const T* A, int32_t lda) {
  require_vendor_fp64<T>();
  double result = 0.;
  for (int32_t j = 0; j < N; ++j)
    for (int32_t i = 0; i < M; ++i) result += std::norm(A[i + int64_t(j) * lda]);
  return result;
}
template <class T>
inline double vendor_check_answer_svd(int32_t M, int32_t N, int32_t rank, const T* U, int32_t ldu,
    const T* V, int32_t ldv, const T* B, int32_t ldb) {
  require_vendor_fp64<T>();
  if (M <= 0 || N <= 0) return 0.;
  constexpr int32_t block = 512;
  std::vector<T> residual(block * block), left(block * block), right(block * block);
  double result = 0.;
  for (int32_t i = 0; i < M; i += block) {
    const int32_t rows = std::min(M - i, block);
    for (int32_t j = 0; j < N; j += block) {
      const int32_t cols = std::min(N - j, block);
      vendor_copy2d(rows, cols, B + i + int64_t(j) * ldb, ldb, residual.data(), rows);
      for (int32_t k = 0; k < rank; k += block) {
        const int32_t inner = std::min(rank - k, block);
        vendor_copy2d(rows, inner, U + i + int64_t(k) * ldu, ldu, left.data(), block);
        vendor_copy2d(cols, inner, V + j + int64_t(k) * ldv, ldv, right.data(), block);
        ncgemm(rows, cols, inner, left.data(), block, right.data(), block, residual.data(), rows);
      }
      result += vendor_fnorm(rows, cols, residual.data(), rows);
    }
  }
  return result;
}
