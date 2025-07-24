
#include <commons.hpp>
#include <int_fp_encode.hpp>

template<int order> double decode_int8(int8_t (&code)[order], int32_t expon) {
  double res = 0;
  int32_t carry = 0;
  int32_t m7 = order % 7, o7 = order - m7;

  for (int32_t i = 0; i < o7; i += 7) {
    int32_t c[7]{};
    for (int32_t j = 0; j < 7; ++j)
      c[j] = int32_t(code[i+j]);
    int64_t val = device::int8::decode_scaled_7xi32(c, carry);
    res += std::scalbn(double(val), 7*(i+expon));
  }

  int32_t c[7]{};
  for (int32_t i = 0; i < m7; ++i)
    c[i] = code[i+o7];
  int64_t val = device::int8::decode_scaled_7xi32(c, carry);
  res += std::scalbn(double(val), 7*(o7+expon));
  res += std::scalbn(double(carry), 7*(o7+expon+7));
  return res;
}

int32_t main() {
  int32_t M = 400, N = 40;
  Eigen::MatrixXcd matA(M, N);
  random_vector(M * N * 2, (double*)matA.data());

  int32_t test_i7[7]{ -1298998, 1165168, -156128, 115628, -165128, -798128, -128 };
  int32_t carry = 8591, carry_old = carry;
  int64_t i7 = device::int8::decode_scaled_7xi32(test_i7, carry);
  int64_t test_decode_i = int64_t(i7) + (int64_t(carry) << 49);
  int64_t ref_decode_i = int64_t(carry_old) + int64_t(test_i7[0]) + (int64_t(test_i7[1]) << 7) + (int64_t(test_i7[2]) << 14) + (int64_t(test_i7[3]) << 21)
    + (int64_t(test_i7[4]) << 28) + (int64_t(test_i7[5]) << 35) + (int64_t(test_i7[6]) << 42);
  printf("%lld %d\n", i7, carry);
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
