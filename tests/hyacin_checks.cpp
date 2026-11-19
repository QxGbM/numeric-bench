#include <hyacin_common.hpp>
#include <iostream>

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

template <class T>
void check_storage() {
  using fp64 = hyacin_fp64_type<T>;
  static_assert(std::is_same_v<fp64, double> || std::is_same_v<fp64, std::complex<double>>);
  std::vector<T> A, U, V;
  for (int i = 1; i <= 6; ++i) A.push_back(conv<T>(double(i)));
  U = A;
  const fp64 phase = [] {
    if constexpr (benchmark_is_complex<T>) return fp64(0., 1.);
    else return fp64(-1.);
  }();
  for (int i = 0; i < 3; ++i) U[i] = conv<T>(conv<fp64>(U[i]) * phase);
  V = {conv<T>(phase), conv<T>(0.), conv<T>(0.), conv<T>(1.)};
  require(check_answer_svd(3, 2, 2, U.data(), 3, V.data(), 2, A.data(), 3) == 0.,
          "Exact reconstruction must retain complex phases");
  U[0] = conv<T>(conv<fp64>(U[0]) + fp64(2.));
  require(check_answer_svd(3, 2, 2, U.data(), 3, V.data(), 2, A.data(), 3) == 4.,
          "Residual products and reductions must use FP64");
  require(check_answer_svd(3, 2, 0, U.data(), 3, V.data(), 2, A.data(), 3) == 91.,
          "Zero-rank residual must equal the input squared norm");
  require(fnorm(3, 2, A.data(), 3) == 91., "Input squared norm must use FP64");
  if constexpr (benchmark_is_complex<T>) {
    const T z = conv<T>(std::complex<double>(3., 4.));
    require(fnorm(1, 1, &z, 1) == 25., "Complex norm must include imaginary components");
  }
}

int main() try {
  check_storage<double>();
  check_storage<float>();
  check_storage<__half>();
  check_storage<cuDoubleComplex>();
  check_storage<cuComplex>();
  check_storage<__half2>();
  check_storage<std::complex<double>>();
  check_storage<std::complex<float>>();
  const float large = 1.e20f;
  const double norm = fnorm(1, 1, &large, 1);
  require(std::isfinite(norm) && norm == double(large) * double(large),
          "FP32 input norm must not overflow in FP32 arithmetic");
  const __half small = __double2half(1.e-4);
  const double small_norm = fnorm(1, 1, &small, 1);
  require(small_norm > 0. && small_norm == double(small) * double(small),
          "FP16 input norm must not underflow in FP16 arithmetic");
  std::vector<std::complex<double>> reference(8 * 4);
  std::vector<cuDoubleComplex> cuda_values(reference.size());
  std::vector<__half2> half_values(reference.size());
  matrix_generator<std::complex<double>>(1., 8, 4).generate_block(8, 4, reference.data(), 8);
  matrix_generator<cuDoubleComplex>(1., 8, 4).generate_block(8, 4, cuda_values.data(), 8);
  matrix_generator<__half2>(1., 8, 4).generate_block(8, 4, half_values.data(), 8);
  for (size_t i = 0; i < reference.size(); ++i) {
    require(conv<std::complex<double>>(cuda_values[i]) == reference[i], "Shared generator must preserve CUDA complex values");
    require(conv<std::complex<double>>(half_values[i]) == conv<std::complex<double>>(conv<__half2>(reference[i])),
            "Shared generator must use Hyacin half-complex conversion");
  }
  LIBRARY_CHECK(ncclSuccess);
  std::cout << "Hyacin FP64 residual and storage checks passed\n";
  return 0;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 1;
}
