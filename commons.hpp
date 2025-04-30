
#include <complex>
#include <random>
#include <algorithm>

void random_vector(int64_t N, double X[]) {
  std::mt19937_64 gen;
  std::normal_distribution<float> dist(0.f, 32.f);
  std::generate(X, &X[N], [&]() { return (int)dist(gen); });
}
