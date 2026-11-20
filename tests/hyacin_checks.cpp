#include <hyacin_common.hpp>
#include <iostream>

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

template <class T>
void check_storage() {
  const auto value = conv<T>(std::complex<double>(3., 4.));
  const auto expected = benchmark_is_complex<T> ? std::complex<double>(3., 4.) : std::complex<double>(3., 0.);
  require(conv<std::complex<double>>(value) == expected, "Storage conversion must retain supported components");
  require(conv<double>(value) == 3., "Conversion to real must retain the real component");
}

template <class T> struct DeviceInput {
  T* data = nullptr;
  explicit DeviceInput(const std::vector<T>& values) {
    LIBRARY_CHECK(cudaMalloc(reinterpret_cast<void**>(&data), values.size() * sizeof(T)));
    LIBRARY_CHECK(cudaMemcpy(data, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice));
  }
  ~DeviceInput() { cudaFree(data); }
  DeviceInput(const DeviceInput&) = delete;
  DeviceInput& operator=(const DeviceInput&) = delete;
};

void require_close(double actual, double expected, const char* message) {
  require(std::isfinite(actual) && std::abs(actual - expected) <= 1.e-12 * std::max(1., expected), message);
}

template <class T>
void check_projection(const hyacinHandle_t* handle) {
  // Padded leading dimensions must not contribute to the norm or projection.
  std::vector<T> A(10, conv<T>(999.)), V(6, conv<T>(999.)), adjoint(9, conv<T>(999.));
  for (int j = 0; j < 2; ++j) for (int i = 0; i < 3; ++i) {
    const auto value = std::complex<double>(1. + i + 3 * j, benchmark_is_complex<T> ? 1. : 0.);
    A[i + j * 5] = conv<T>(value);
    adjoint[j + i * 3] = conv<T>(std::conj(value));
  }
  const auto phase = benchmark_is_complex<T> ? std::complex<double>(0., 2.) : std::complex<double>(-2., 0.);
  V[0] = conv<T>(phase); V[1] = conv<T>(0.);
  V[3] = conv<T>(phase); V[4] = conv<T>(1.); // Nonorthogonal columns span the full space.
  DeviceInput<T> input(A), basis(V), transposed(adjoint);
  const double norm = benchmark_is_complex<T> ? 97. : 91.;
  const double residual = benchmark_is_complex<T> ? 80. : 77.;
  for (char op : {'N', 'C'}) {
    const T* matrix = op == 'N' ? input.data : transposed.data;
    const int lda = op == 'N' ? 5 : 3;
    require_close(hyacin_check_norm_fp64<T>(handle, op, 3, 2, matrix, lda, 0, nullptr, 0), norm,
                  "Rank-zero check must return the FP64 squared input norm without V");
    require_close(hyacin_check_norm_fp64(handle, op, 3, 2, matrix, lda, 1, basis.data, 3), residual,
                  "Rank-one projection must use the span of V, including complex phases");
    require_close(hyacin_check_norm_fp64(handle, op, 3, 2, matrix, lda, 2, basis.data, 3), 0.,
                  "Full-span projection must orthonormalize nonorthogonal V");
  }
  std::vector<T> afterA(A.size()), afterV(V.size());
  LIBRARY_CHECK(cudaMemcpy(afterA.data(), input.data, A.size() * sizeof(T), cudaMemcpyDeviceToHost));
  LIBRARY_CHECK(cudaMemcpy(afterV.data(), basis.data, V.size() * sizeof(T), cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < A.size(); ++i)
    require(conv<std::complex<double>>(afterA[i]) == conv<std::complex<double>>(A[i]), "Norm check must preserve A");
  for (size_t i = 0; i < V.size(); ++i)
    require(conv<std::complex<double>>(afterV[i]) == conv<std::complex<double>>(V[i]), "Norm check must preserve V");
}

void check_gpu() {
  struct Handle {
    hyacinHandle_t value{};
    Handle() { hyacinCreate(&value); }
    ~Handle() { hyacinDestroy(&value); }
  } handle;
  check_projection<double>(&handle.value);
  check_projection<float>(&handle.value);
  check_projection<__half>(&handle.value);
  check_projection<cuDoubleComplex>(&handle.value);
  check_projection<cuComplex>(&handle.value);
  check_projection<__half2>(&handle.value);
  check_projection<std::complex<double>>(&handle.value);
  check_projection<std::complex<float>>(&handle.value);
  const float large = 1.e20f;
  DeviceInput<float> largeInput(std::vector<float>{large});
  const double norm = hyacin_check_norm_fp64<float>(&handle.value, 'N', 1, 1, largeInput.data, 1, 0, nullptr, 0);
  require_close(norm, double(large) * double(large), "FP32 norm must not overflow in FP32 arithmetic");
  const __half small = __double2half(1.e-4);
  DeviceInput<__half> smallInput(std::vector<__half>{small});
  const double smallNorm = hyacin_check_norm_fp64<__half>(&handle.value, 'N', 1, 1, smallInput.data, 1, 0, nullptr, 0);
  const double expected = double(small) * double(small);
  require(smallNorm > 0. && std::abs(smallNorm / expected - 1.) < 1.e-12,
          "FP16 norm must not underflow in FP16 arithmetic");
  hyacinSync_TimerSegments(&handle.value, nullptr, 0);
  std::cout << "Hyacin library FP64 projection and norm checks passed\n";
}

int main(int argc, char* argv[]) try {
  check_storage<double>();
  check_storage<float>();
  check_storage<__half>();
  check_storage<cuDoubleComplex>();
  check_storage<cuComplex>();
  check_storage<__half2>();
  check_storage<std::complex<double>>();
  check_storage<std::complex<float>>();
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
  std::cout << "Hyacin storage checks passed\n";
  if (argc == 2 && std::string(argv[1]) == "--gpu") check_gpu();
  else require(argc == 1, "Usage: hyacin_checks.app [--gpu]");
  return 0;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 1;
}
