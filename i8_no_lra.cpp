
#include <commons.hpp>
#include <int_fp_encode.hpp>

int32_t main() {
  int32_t M = 400, N = 40;
  Eigen::MatrixXcd matA(M, N);
  random_vector(M * N * 2, (double*)matA.data());

  int3 test_i3 = make_int3(-13515313, -1515618, -4199848); //
  float2 test_f2 = device::int8::decode_scaled_int3_float2(test_i3);
  double test_double = double(test_f2.x) + double(test_f2.y);
  double ref_d = (double)test_i3.x + (double)test_i3.y * 128 + (double)test_i3.z * 16384;
  printf("%lf %lf %le\n", test_double, ref_d, test_double-ref_d);

  double ref_f = -2.81569195199e-10;
  int32_t expon;
  union { int3 code; int8_t byte[12]; } c;
  int64_t frac = 0;
  device::int8::encode_double_exp7_9xi8(ref_f, expon, c.code);

  for (int i = 0; i < 10; ++i) {
    int64_t e = c.byte[i];
    frac += e * (int64_t(1) << (7 * i));
  }

  double test_f = std::scalbln(double(frac), 7*expon);
  printf("%le %le %le %d %ld\n", test_f, ref_f, test_f-ref_f, expon, frac);

  return 0;
}
