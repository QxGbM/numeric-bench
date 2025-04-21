
#include <complex>
#include <random>
#include <algorithm>

void random_vector(int64_t N, double X[]) {
  std::mt19937_64 gen;
  std::normal_distribution<double> dist(0., 1.);
  std::generate(X, &X[N], [&]() { return dist(gen), dist(gen); });
}
