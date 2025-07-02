
#include <cstdint>
#include <cstdio>
#include <cmath>

#include <double_double.hpp>

double2 double_double2(double a) {
  return make_double2(a, 0.);
}

int32_t main() {
  double x0 = 2;
  double x1 = 1.0 / std::sqrt(x0);
  printf("a = %.40lf\n", x0);
  printf("b = 1./sqrt(a) = %.40lf\n", x1);

  double d = x0 * x1 * x1;

  double2 y, z, w, e;
  y = double_double2(x0);
  z = host::dd::frsqrt(y);
  w = double_double2(0.);
  e = double_double2(0.);

  printf("<double2> a = %.40le %.40le\n", y.x, y.y);
  printf("<double2> b = %.40le %.40le\n", z.x, z.y);

  w = host::dd::fma(y, z, e);
  w = host::dd::fma(w, z, e);
  printf("<float> a*b*b = %.40le %.40le\n", w.x, w.y);

  double d2 = double(w.x) + double(w.y);
  printf("<double> a*b*b = %.40le\n", d);
  printf("<double2 in double> a*b*b = %.40le\n", d2);
  printf("err = %.40le\n", (d - d2) / d);

  return 0;
}
