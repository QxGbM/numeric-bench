
#include <cstdint>
#include <cstdio>
#include <cmath>

#include <float4.hpp>
#include <hyacinth.h>
#include <cuda_runtime_api.h>

float4 double_float4(double a) {
  float x = float(a);
  a -= double(x);
  float y = float(a);
  a -= double(y);
  float z = float(a);
  return float4 { x, y, z, 0.f };
}

int32_t main() {
  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

  double x0 = 2000.0 / 19.0;
  double x1 = 1.0 / std::sqrt(x0);
  printf("a = %.20lf\n", x0);
  printf("b = 1./sqrt(a) = %.20lf\n", x1);

  double d = x0 * x1 * x1;

  float4 *y, *z, *w;
  cudaMallocManaged((void**)&y, sizeof(float) * 4, cudaMemAttachGlobal);
  cudaMallocManaged((void**)&z, sizeof(float) * 4, cudaMemAttachGlobal);
  cudaMallocManaged((void**)&w, sizeof(float) * 4, cudaMemAttachGlobal);

  *y = double_float4(x0);
  *z = host::f4::rsqrt(*y);
  *w = double_float4(x0);

  printf("<float4> a = %.20e %.20e %.20e %.20e\n", y->x, y->y, y->z, y->w);
  printf("<float4> b = %.20e %.20e %.20e %.20e\n", z->x, z->y, z->z, z->w);

  scal_incx1_float4(stream, *z, 1, w);
  scal_incx1_float4(stream, *z, 1, w);
  cudaDeviceSynchronize();

  printf("<float4> a*b*b = %.20e %.20e %.20e %.20e\n", w->x, w->y, w->z, w->w);

  double d2 = double(w->x) + double(w->y) + double(w->z) + double(w->w);
  printf("<double> a*b*b = %.20le\n", d);
  printf("<float4 in double> a*b*b = %.20le\n", d2);
  printf("err = %.20le\n", (d - d2) / d);

  cudaFree(y);
  cudaFree(z);
  cudaFree(w);
  cudaStreamDestroy(stream);
  return 0;
}
