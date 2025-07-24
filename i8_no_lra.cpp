
#include <commons.hpp>
#include <int_fp_encode.hpp>

template<int order> double decode_int8(int8_t (&code)[order], int32_t expon) {
  double res = 0;
  int32_t carry = 0;
  int32_t m4 = order & 3, o4 = order - m4;

  for (int32_t i = 0; i < o4; i += 4) {
    int32_t c[4] = { int32_t(code[i]), int32_t(code[i+1]), int32_t(code[i+2]), int32_t(code[i+3]) };
    int32_t val = device::int8::decode_scaled_4xi32(c, carry);
    res += std::scalbn(double(val), 7*(i+expon));
  }

  int32_t c[4]{};
  for (int32_t i = 0; i < m4; ++i)
    c[i] = code[i+o4];
  int32_t val = device::int8::decode_scaled_4xi32(c, carry);
  res += std::scalbn(double(val), 7*(o4+expon));
  res += std::scalbn(double(carry), 7*(o4+expon+1));
  return res;
}

int32_t main() {
  int32_t M = 400, N = 40;
  Eigen::MatrixXcd matA(M, N);
  random_vector(M * N * 2, (double*)matA.data());

  int32_t test_i4[4]{ -13515313, 1515618, -4199848, 1561648 };
  int32_t carry = -156156916, carry_old = carry;
  int32_t i4 = device::int8::decode_scaled_4xi32(test_i4, carry);
  int64_t test_decode_i = int64_t(i4) + (int64_t(carry) << 28);
  int64_t ref_decode_i = int64_t(carry_old) + int64_t(test_i4[0]) + (int64_t(test_i4[1]) << 7) + (int64_t(test_i4[2]) << 14) + (int64_t(test_i4[3]) << 21);
  printf("%d %d\n", i4, carry);
  printf("%lld %lld %lld\n", test_decode_i, ref_decode_i, test_decode_i-ref_decode_i);

  double ref_f = -2.81569195111111418919199e-2;
  int32_t expon = device::int8::get_double_top_exp(ref_f), vec_e, rem;
  printf("%d %lf\n", expon, std::scalbn(ref_f, -expon));

  device::int8::fast_div7_i32x(expon, vec_e, rem);  
  printf("%d %lf\n", vec_e, std::scalbn(ref_f, -(vec_e*7)));

  union { uint32_t ext[5]; uint32_t code[3]; int8_t byte[20]; } c {};
  vec_e = vec_e - 10 + 1;
  device::int8::encode_double_exp7_9xi8(ref_f, expon, c.code);
  device::int8::align_expon<5>(c.ext, expon - vec_e);

  double test_f = decode_int8(c.byte, vec_e);
  printf("%le %le %le\n", test_f, ref_f, std::abs((test_f-ref_f)/ref_f));

  return 0;
}
