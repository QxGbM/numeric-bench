
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <quadmath.h>

void add_f2(double& a0, double& a1) {
  double sum = a0 + a1;
  double err = (a0 - sum) + a1;
  a0 = sum;
  a1 = err;
}

void mul_f2(double& a0, double& a1) {
  double prod = a0 * a1;
  double err = std::fma(a0, a1, -prod);
  a0 = prod;
  a1 = err;
}

int32_t main() {
  
  double x0 = 333;
  double x1 = 0.33333333333333333333333;

  __float128 f = 1.0Q / 3.0Q;
  printf("%.40Qf %.40lf\n", f, 1.0 / 3.0);

  __float128 d = __float128(x0) * __float128(x1);

  mul_f2(x0, x1);
  __float128 d2 = __float128(x0) + __float128(x1);

  printf("%.40Qe %.40Qe\n", d, std::abs(d - d2) / d);

  return 0;
}
