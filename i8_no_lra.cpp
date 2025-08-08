
#include <commons.hpp>
#include <int_fp_encode.hpp>

template<int32_t base, int32_t order> double decode_int(int8_t (&code)[order], int32_t expon) {
  double res = 0;
  for (int32_t i = 0; i < order; ++i)
    res += std::scalbn(double(code[i]), base*(i+expon));
  return res;
}

constexpr int32_t base = 6;

int32_t main() {
  int32_t M = 400, N = 40;
  Eigen::MatrixXcd matA(M, N);
  random_vector(M * N * 2, (double*)matA.data());

  double ref_f = -2.81569195111111418919199e-2;
  int32_t expon = device::int8::get_double_top_exp(ref_f), vec_e, rem;
  printf("%d %lf\n", expon, std::scalbn(ref_f, -expon));

  device::int8::fast_division_i32<base>(expon, vec_e, rem);  
  printf("%d %lf\n", vec_e, std::scalbn(ref_f, -(vec_e*base)));

  union { uint32_t ext[5]; uint32_t code[4]; int8_t byte[20]; } c {};
  int32_t order = 9;

  vec_e = vec_e - order + 1;
  device::int8::encode_double<base>(ref_f, expon, c.code);
  device::int8::align_expon<5>(c.ext, expon - vec_e);

  double test_f = decode_int<base>(c.byte, vec_e);
  printf("%le %le %le\n", test_f, ref_f, std::abs((test_f-ref_f)/ref_f));

  double epi = 1.e-11;
  int32_t bits = int32_t(std::ceil(-std::log2(epi)));
  int32_t bits_order = int32_t(std::ceil(-std::log2(epi) / base));

  printf("%d %d %.20le %.20le %.20le\n", bits, bits_order*base, epi, std::scalbn(1, -bits), std::scalbn(1, -base*bits_order));

  return 0;
}
