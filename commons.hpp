
#include <complex>
#include <vector>
#include <algorithm>
#include <numeric>
#include <limits>
#include <random>
#include <iostream>
#include <algorithm>
#include <omp.h>

#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#define MKL_COMPLEX_16 std::complex<double>
#define EIGEN_USE_MKL_ALL
#include <mkl.h>
#include <eigen3/Eigen/Dense>

void random_vector(int64_t N, double X[]) {
  std::mt19937_64 gen;
  std::normal_distribution<float> dist(0.f, 32.f);
  std::generate(X, &X[N], [&]() { return (int)dist(gen); });
}
