#include <cstdint>
#include <cstdio>
#include <cmath>

#include <quad_float.hpp>

float4 double_float4(double a) {
  float x = float(a);
  a -= double(x);
  float y = float(a);
  a -= double(y);
  float z = float(a);
  return float4 { x, y, z, 0.f };
}

int32_t main() {
  double x0 = 2000.0 / 19.0;
  double x1 = 1.0 / std::sqrt(x0);
  printf("a = %.20lf\n", x0);
  printf("b = 1./sqrt(a) = %.20lf\n", x1);

  double d = x0 * x1 * x1;

  float4 y, z, w, e;
  y = double_float4(x0);
  z = device::qf::frsqrt(y);
  w = double_float4(0.);
  e = double_float4(0.);

  printf("<float4> a = %.20le %.20le %.20le %.20le\n", y.x, y.y, y.z, y.w);
  printf("<float4> b = %.20le %.20le %.20le %.20le\n", z.x, z.y, z.z, z.w);

  w = device::qf::fma(y, z, e);
  w = device::qf::fma(w, z, e);
  printf("<float> a*b*b = %.20le %.20le %.20le %.20le\n", w.x, w.y, w.z, w.w);

  double d2 = double(w.x) + double(w.y) + double(w.z) + double(w.w);
  printf("<double> a*b*b = %.20le\n", d);
  printf("<float4 in double> a*b*b = %.20le\n", d2);
  printf("err = %.20le\n", (d - d2) / d);

  return 0;
}
