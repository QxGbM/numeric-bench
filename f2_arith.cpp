
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <quadmath.h>
#include <vector>
#include <algorithm>

inline void two_sum(double& a0, double& a1) {
  double sum = a0 + a1;
  double b = (sum - a0);
  double err = (a0 - (sum - b)) + (a1 - b);
  a0 = sum;
  a1 = err;
}

inline void two_prod(double& a0, double& a1) {
  double prod = a0 * a1;
  double err = std::fma(a0, a1, -prod);
  a0 = prod;
  a1 = err;
}

// a = sum[i=0;i=n-1](a_i * s_i);
inline void expansion_scale(int32_t n, double a[], double s[]) {
  for (int32_t i = 0; i < n; ++i) {
    two_prod(a[i], s[i]);
    for (int32_t j = 0; j < i; ++j)
      two_sum(a[i], s[j]);
  }
}

// a = sum[i=0;i=n-1](a_i + b_i);
void expansion_add(int32_t n, double a[], double b[]) {
  for (int32_t i = 0; i < n; ++i)
    for (int32_t j = 0; j <= i; ++j)
      two_sum(a[i], b[j]);
  for (int32_t i = 0; i < (n - 1); ++i)
    two_sum(a[i], a[i + 1]);
}

// c = c + sum[(i=0,j=0);(i=n-1,j=n-1)](a_i * b_j)
void expansion_fma(int32_t n, const double a[], const double b[], double c[]) {
  std::vector<double> i0(n), i1(n);

  for (int32_t i = 0; i < n; ++i) {
    std::copy(a, &a[n], i0.begin());
    std::fill(i1.begin(), i1.end(), b[i]);
    expansion_scale(n, i0.data(), i1.data());
    expansion_add(n, c, i0.data());
  }
  for (int32_t i = 0; i < (n - 1); ++i)
    two_sum(c[i], c[i + 1]);
}

int32_t main() {
  __float128 x0 = 8.0Q / 3.0Q;
  __float128 x1 = 1.0Q / 3.0Q;
  printf("a = %.40Qf\n", x0);
  printf("b = %.40Qf\n", x1);

  __float128 d = x0 * x1;

  std::vector<double> y(3), z(3), w(3, 0.);

  y[0] = double(x0);
  y[1] = double(x0 - __float128(y[0]));
  y[2] = double(x0 - __float128(y[0]) - __float128(y[1]));
  z[0] = double(x1);
  z[1] = double(x1 - __float128(z[0]));
  z[2] = double(x1 - __float128(z[0]) - __float128(z[1]));

  printf("y = %.20e %.20e %.20e\n", y[0], y[1], y[2]);
  printf("z = %.20e %.20e %.20e\n", z[0], z[1], z[2]);

  expansion_fma(3, y.data(), z.data(), w.data());
  printf("w = %.20e %.20e %.20e\n", w[0], w[1], w[2]);

  __float128 d2 = __float128(w[0]) + __float128(w[1]) + __float128(w[2]);
  printf("err = %.40Qe\n", (d - d2) / d);

  return 0;
}
