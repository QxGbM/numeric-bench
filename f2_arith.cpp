
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <float4_fma.cuh>

/*inline void two_sum_f4(float4 a0, float4 a1, float4& sum, float4& err) {
  float4 err_x, err_y;
  sum.x = a0.x + a1.x;
  sum.y = a0.y + a1.y;
  sum.z = a0.z + a1.z;
  sum.w = a0.w + a1.w;

  err_x.x = sum.x - a0.x;
  err_x.y = sum.y - a0.y;
  err_x.z = sum.z - a0.z;
  err_x.w = sum.w - a0.w;

  a1.x = a1.x - err_x.x;
  a1.y = a1.y - err_x.y;
  a1.z = a1.z - err_x.z;
  a1.w = a1.w - err_x.w;

  err_y.x = sum.x - err_x.x;
  err_y.y = sum.y - err_x.y;
  err_y.z = sum.z - err_x.z;
  err_y.w = sum.w - err_x.w;

  a0.x = a0.x - err_y.x;
  a0.y = a0.y - err_y.y;
  a0.z = a0.z - err_y.z;
  a0.w = a0.w - err_y.w;

  err.x = a0.x + a1.x;
  err.y = a0.y + a1.y;
  err.z = a0.z + a1.z;
  err.w = a0.w + a1.w;
}

inline void two_prod_f4(float4 a0, float4 a1, float4& prod, float4& err) {
  prod.x = a0.x * a1.x;
  prod.y = a0.y * a1.y;
  prod.z = a0.z * a1.z;
  prod.w = a0.w * a1.w;

  err.x = std::fma(a0.x, a1.x, -prod.x);
  err.y = std::fma(a0.y, a1.y, -prod.y);
  err.z = std::fma(a0.z, a1.z, -prod.z);
  err.w = std::fma(a0.w, a1.w, -prod.w);
}

inline void renormalize(float4& a) {
  float sum = a.x + a.y;
  a.y += a.x - sum;
  a.x = sum;

  sum = a.y + a.z;
  a.z += a.y - sum;
  a.y = sum;

  sum = a.z + a.w;
  a.w += a.z - sum;
  a.z = sum;
}

// a = sum[i=0;i=n-1](a_i + b_i);
void float4_add(float4& a, float4& b) {
  float4 bi;
  two_sum_f4(a, b, a, bi);
  b = make_float4(0.f, bi.x, bi.y, bi.z);

  two_sum_f4(a, b, a, bi);
  b = make_float4(0.f, bi.x, bi.y, bi.z);

  two_sum_f4(a, b, a, bi);
  b = make_float4(0.f, bi.x, bi.y, bi.z);

  two_sum_f4(a, b, a, bi);
  b = make_float4(0.f, bi.x, bi.y, bi.z);
  renormalize(a);
}

void float4_fma(float4 a, float4 b, float4& c) {
  float4 bi, prod, err;
  bi = make_float4(b.x, b.x, b.x, b.x);
  two_prod_f4(a, bi, prod, err);
  float4_add(c, prod);
  float4_add(c, err);

  bi = make_float4(b.y, b.y, b.y, b.y);
  two_prod_f4(a, bi, prod, err);
  float4_add(c, prod);
  float4_add(c, err);

  bi = make_float4(b.z, b.z, b.z, b.z);
  two_prod_f4(a, bi, prod, err);
  float4_add(c, prod);
  float4_add(c, err);

  bi = make_float4(b.w, b.w, b.w, b.w);
  two_prod_f4(a, bi, prod, err);
  float4_add(c, prod);
  float4_add(c, err);
  
  renormalize(c);
}*/

float4 double_float4(double a) {
  float x = float(a);
  a -= double(x);
  float y = float(a);
  a -= double(y);
  float z = float(a);
  return float4 { x, y, z, 0.f };
}

int32_t main() {
  double x0 = 8.0 / 3.0;
  double x1 = 1.0 / 3.0;
  printf("a = %.20lf\n", x0);
  printf("b = %.20lf\n", x1);

  double d = x0 * x1;

  float4 y, z, w;
  y = double_float4(x0);
  z = double_float4(x1);
  w = double_float4(0.);

  printf("y = %.20e %.20e %.20e %.20e\n", y.x, y.y, y.z, y.w);
  printf("z = %.20e %.20e %.20e %.20e\n", z.x, z.y, z.z, z.w);

  w = float4_fma(y, z, w);
  printf("w = %.20e %.20e %.20e %.20e\n", w.x, w.y, w.z, w.w);

  double d2 = double(w.x) + double(w.y) + double(w.z) + double(w.w);
  printf("d = %.20le\n", d2);
  printf("err = %.20le\n", (d - d2) / d);

  return 0;
}
