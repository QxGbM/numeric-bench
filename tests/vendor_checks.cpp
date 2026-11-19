#include <vendor_common.hpp>
#include <iostream>

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

template <class Status>
void expect_failure(Status status) {
  bool threw = false;
  try { library_check(status, "injected_call", "test_source", 42); }
  catch (const std::runtime_error& error) {
    const std::string message(error.what());
    threw = message.find("test_source:42") != std::string::npos &&
      message.find("injected_call") != std::string::npos;
  }
  require(threw, "Library failure must report the call and source location");
}

template <class T>
void check_residual() {
  const std::vector<T> A{T(1.), T(2.), T(3.), T(4.), T(5.), T(6.)};
  std::vector<T> U = A;
  const T phase = [] {
    if constexpr (std::is_same_v<T, double>) return T(-1.);
    else return T(0., 1.);
  }();
  const std::vector<T> V{phase, T(0.), T(0.), T(1.)};
  for (int i = 0; i < 3; ++i) U[i] *= phase;
  require(vendor_check_answer_svd(3, 2, 2, U.data(), 3, V.data(), 2, A.data(), 3) == 0.,
    "Exact projection must have zero residual");
  require(vendor_check_answer_svd(3, 2, 0, U.data(), 3, V.data(), 2, A.data(), 3) == 91.,
    "Zero-rank residual must retain the input norm");
  U[0] += T(2.);
  require(std::abs(vendor_check_answer_svd(3, 2, 2, U.data(), 3, V.data(), 2, A.data(), 3) - 4.) < 1.e-12,
    "Residual must detect an incorrect projection");
}

int main() try {
  static_assert(vendor_type<double> == CUDA_R_64F);
  static_assert(vendor_type<std::complex<double>> == CUDA_C_64F);
  LIBRARY_CHECK(cudaSuccess);
  LIBRARY_CHECK(CUBLAS_STATUS_SUCCESS);
  LIBRARY_CHECK(CUSOLVER_STATUS_SUCCESS);
  expect_failure(cudaErrorInvalidValue);
  expect_failure(CUBLAS_STATUS_INVALID_VALUE);
  expect_failure(CUSOLVER_STATUS_INVALID_VALUE);
  setenv("NCCL_MULTI_RANK_GPU_ENABLE", "0", 1);
  enable_nccl_multi_rank_gpu();
  require(std::string(std::getenv("NCCL_MULTI_RANK_GPU_ENABLE")) == "1", "Multi-rank helper must enable NCCL GPU oversubscription");
  LIBRARY_CHECK(ncclSuccess);
  expect_failure(ncclInvalidArgument);
  // At fixed N, doubling M must double only the rectangular work.
  const double base = gesvd_flops(100., 10., 1.);
  const double taller = gesvd_flops(200., 10., 1.);
  require(std::abs((taller - base) - 40000.) < 1.e-9, "GESVD work must scale linearly with M");
  require(gesvd_flops(100., 10., 4.) == 4. * base, "Complex work scaling");
  require(std::isfinite(gesvd_flops(1.e9, 1.e6, 4.)), "Flop estimate must avoid integer overflow");
  static_assert(benchmark_flop_factor<double> == 1.);
  static_assert(benchmark_flop_factor<cuDoubleComplex> == 4.);
  static_assert(benchmark_flop_factor<std::complex<double>> == 4.);
  // A 4x3 QR costs 54; a rank-2 solve adds 2^2 * (3-2) = 4.
  require(interpolative_flops(4., 3., 2., 1.) == 58., "LRA must count rectangular TRSM and no explicit Q");
  require(interpolative_flops(4., 3., 3., 1.) == 54., "Full-rank LRA has no trailing TRSM");
  require(interpolative_flops(4., 3., 0., 1.) == 54., "QR work remains at zero selected rank");
  require(interpolative_flops(3., 4., 2., 1.) == 62., "Wide QR uses min(M,N) reflectors");
  require(interpolative_flops(4., 3., 2., 4.) == 232., "Complex LRA scaling");
  require(randomized_svd_flops(4., 3., 2., 0., 1.) == 184., "Right-vector randomized SVD estimate");
  require(randomized_svd_flops(4., 3., 2., 1., 1.) == 280., "Power iteration adds two matrix products");
  require(randomized_svd_flops(4., 3., 0., 0., 1.) == 0., "Zero-rank SVD work");
  require(randomized_svd_flops(4., 3., 2., 0., 4.) == 736., "Complex SVD scaling");
  require(std::isfinite(interpolative_flops(1.e9, 1.e6, 1.e5, 4.)), "LRA flop count avoids integer overflow");
  require(std::isfinite(randomized_svd_flops(1.e9, 1.e6, 1.e5, 0., 4.)), "SVD flop count avoids integer overflow");
  check_residual<double>();
  check_residual<std::complex<double>>();
  std::cout << "Vendor host checks passed\n";
  return 0;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 1;
}
