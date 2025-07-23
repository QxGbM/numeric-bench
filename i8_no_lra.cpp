
#include <commons.hpp>
#include <int_fp_encode.hpp>

template<int order> double decode_int8(int8_t (&code)[order], int32_t expon) {
  double res = 0;
  for (int i = 0; i < order; ++i)
    res += std::scalbn(double(code[i]), 7*(i+expon));
  return res;
}

int32_t main() {
  int32_t M = 400, N = 40;
  Eigen::MatrixXcd matA(M, N);
  random_vector(M * N * 2, (double*)matA.data());

  int32_t test_i3[3]{ -13515313, -1515618, -4199848 }; //
  float test_f2[2];
  device::int8::decode_scaled_int3_float2(test_i3, test_f2);
  double test_double = double(test_f2[0]) + double(test_f2[1]);
  double ref_d = (double)test_i3[0] + (double)test_i3[1] * 128 + (double)test_i3[2] * 16384;
  printf("%lf %lf %le\n", test_double, ref_d, test_double-ref_d);

  double ref_f = -2.81569195111111418919199e-2;
  int32_t expon = device::int8::get_double_top_exp(ref_f), vec_e, rem;
  printf("%d %lf\n", expon, std::scalbn(ref_f, -expon));

  device::int8::fast_div7_i32x(expon, vec_e, rem);  
  printf("%d %lf\n", vec_e, std::scalbn(ref_f, -(vec_e*7)));

  union { uint32_t ext[5]; uint32_t code[3]; int8_t byte[20]; } c {};
  vec_e = vec_e - 6 + 1;
  device::int8::encode_double_exp7_9xi8(ref_f, expon, c.code);
  device::int8::align_expon<5>(c.ext, expon - vec_e);

  double test_f = decode_int8(c.byte, vec_e);
  printf("%le %le %le\n", test_f, ref_f, std::abs((test_f-ref_f)/ref_f));

  return 0;
}
