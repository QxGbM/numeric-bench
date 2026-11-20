#pragma once
#include <benchmark_common.hpp>
#include <hyacin.h>
#include <cuda_fp16.h>

template <> constexpr bool benchmark_is_complex<__half2> = true;

template <class T, class S> inline T conv(S x) {
  if constexpr (benchmark_is_complex<S>) {
    std::complex<double> value;
    if constexpr (std::is_same_v<S, cuDoubleComplex> || std::is_same_v<S, cuComplex>)
      value = {double(x.x), double(x.y)};
    else if constexpr (std::is_same_v<S, __half2>)
      value = {double(x.x), double(x.y)};
    else value = {double(x.real()), double(x.imag())};
    if constexpr (std::is_same_v<T, __half2>)
      return make_half2(__double2half(value.real()), __double2half(value.imag()));
    else if constexpr (benchmark_is_complex<T>) return benchmark_from_complex<T>(value);
    else return T(value.real());
  } else if constexpr (benchmark_is_complex<T>) {
    const std::complex<double> value(double(x), 0.);
    if constexpr (std::is_same_v<T, __half2>) return make_half2(__double2half(value.real()), __double2half(0.));
    else return benchmark_from_complex<T>(value);
  } else return T(x);
}

template <> inline __half2 benchmark_from_complex<__half2>(std::complex<double> value) {
  return make_half2(__double2half(value.real()), __double2half(value.imag()));
}

template <class T> inline hyacinPrecision_t hyacin_prec();
template <> inline hyacinPrecision_t hyacin_prec<double>() { return HYACIN_F64; }
template <> inline hyacinPrecision_t hyacin_prec<float>() { return HYACIN_F32; }
template <> inline hyacinPrecision_t hyacin_prec<__half>() { return HYACIN_F16; }
template <> inline hyacinPrecision_t hyacin_prec<cuDoubleComplex>() { return HYACIN_F64_COMPLEX; }
template <> inline hyacinPrecision_t hyacin_prec<cuComplex>() { return HYACIN_F32_COMPLEX; }
template <> inline hyacinPrecision_t hyacin_prec<__half2>() { return HYACIN_F16_COMPLEX; }
template <> inline hyacinPrecision_t hyacin_prec<std::complex<double>>() { return HYACIN_F64_COMPLEX; }
template <> inline hyacinPrecision_t hyacin_prec<std::complex<float>>() { return HYACIN_F32_COMPLEX; }

// The library promotes these storage types to FP64 for QR, GEMM, and norms.
template <class T>
inline double hyacin_check_norm_fp64(const hyacinHandle_t* handle, char opA, int32_t M, int32_t N,
    const T* A, int32_t lda, int32_t rank, const T* V, int32_t ldv) {
  return hyacinCheckNorm(handle, opA, M, N, A, lda, rank, V, ldv, hyacin_prec<T>());
}

inline void handle_param_environments(char algo, int32_t batch, int32_t u_corr, int32_t jacobi_sweeps) {
  setenv("HYACIN_GRAM_ALGORITHM", std::string(1, algo).c_str(), 1);
  setenv("HYACIN_GRAM_SEGMENT_K", std::to_string(batch).c_str(), 1);
  setenv("HYACIN_QUANTIZE_BITS_CORR", std::to_string(u_corr).c_str(), 1);
  setenv("HYACIN_JACOBI_SVD_SWEEPS", std::to_string(jacobi_sweeps).c_str(), 1);
  setenv("HYACIN_CREATE_TIMER", "1", 1);
}
