
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
  float prod = a0 * a1;
  float err = std::fma(a0, a1, -prod);
  a0 = prod;
  a1 = err;
}

int32_t main() {
  
  float x0 = 1746.1233232;
  float x1 = 1.51212313123135631;

  double d = double(x0) * double(x1);

  mul_f2(x0, x1);
  double d2 = double(x0) + double(x1);

  printf("%.12e %.12e %.12e\n", x0, x1, std::abs(d - d2) / d);

  return 0;
}
