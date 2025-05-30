
#include <cstdint>
#include <cstdio>
#include <cmath>

void add_f2(float& a0, float& a1) {
  float sum = a0 + a1;
  float err = (a0 - sum) + a1;
  a0 = sum;
  a1 = err;
}

void mul_f2(float& a0, float& a1) {
  float c = 4097.f * a0;
  float a0_hi = c - (c - a0);
  float a0_lo = a0 - a0_hi;

  c = 4097.f * a1;
  float a1_hi = c - (c - a1);
  float a1_lo = a1 - a1_hi;

  float prod = a0 * a1;
  float err = ((a0_hi * a1_hi - prod) + a0_hi * a1_lo + a0_lo * a1_hi) + a0_lo * a1_lo;
  a0 = prod;
  a1 = err;
}

int32_t main() {
  
  float x0 = 1746.1233232;
  float x1 = 1.51231;

  double d = double(x0) * double(x1);

  mul_f2(x0, x1);
  double d2 = double(x0) + double(x1);

  printf("%.12e %.12e %.12e\n", x0, x1, std::abs(d - d2) / d);

  return 0;
}
