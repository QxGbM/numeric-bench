
#include <commons.hpp>

uint64_t fixed_point_mul_i15x4(const uint64_t* a, const uint64_t* b) {
  uint64_t ia0 = a[0];
  int32_t a0 = int16_t(ia0);
  int32_t a1 = int16_t(ia0 >> 16);
  int32_t a2 = int16_t(ia0 >> 32);
  int32_t a3 = int16_t(ia0 >> 48);

  uint64_t ib0 = b[0];
  int32_t b0 = int16_t(ib0);
  int32_t b1 = int16_t(ib0 >> 16);
  int32_t b2 = int16_t(ib0 >> 32);
  int32_t b3 = int16_t(ib0 >> 48);

  int32_t m0 = a0 * b0;
  int32_t m1 = a1 * b0 + a0 * b1;
  int32_t m2 = a2 * b0 + a1 * b1 + a0 * b2;
  int32_t m3 = a3 * b0 + a2 * b1 + a1 * b2 + a0 * b3;
  int32_t m4 = a3 * b1 + a2 * b2 + a1 * b3;

  a0 = m0 & 0x3FFF;
  a1 = m1 & 0x3FFF;
  a2 = m2 & 0x3FFF;
  a3 = m3 & 0x3FFF;

  m0 = m0 >> 14;
  m1 = (m1 >> 14) + a0;
  m2 = (m2 >> 14) + a1;
  m3 = (m3 >> 14) + a2;
  m4 = (m4 >> 14) + a3;

  a0 = m1 >> 14;
  a1 = m2 >> 14;
  a2 = m3 >> 14;
  a3 = m4 >> 14;

  m0 = m0 + a0;
  m1 = (m1 & 0x3FFF) + a1;
  m2 = (m2 & 0x3FFF) + a2;
  m3 = (m3 & 0x3FFF) + a3;

  uint32_t ci0 = (m1 << 16) | (m0 & 0xFFFF);
  uint32_t ci1 = (m3 << 16) | (m2 & 0xFFFF);
  uint64_t c = (uint64_t(ci1) << 32) | uint64_t(ci0);
  return c;
}

int32_t main() {  
  double b = 1531.2345678913354632132 / 1024;
  double c = 1.;
  for (int32_t i = 0; i < 6; ++i) {
    c = 0.5 * c * (3 - b * c * c);
  }
  double d = b * c * c;
  printf("%e\n", std::abs(d - 1.));

  b = -b;
  double b_iter = b;
  int16_t bp[4];
  for (int32_t i = 0; i < 4; ++i) {
    b_iter *= (1 << 14);
    bp[i] = int16_t(b_iter);
    b_iter -= bp[i];
  }

  b_iter = 1. / (1 << 14);
  double reb = 0;
  for (int32_t i = 0; i < 4; ++i) {
    reb += b_iter * bp[i];
    b_iter *= 1. / (1 << 14);
  }

  printf("%e\n", std::abs(reb - b) / b);
  printf("%d %d %d %d\n", bp[0], bp[1], bp[2], bp[3]);

  double c_iter = c * 0.5;
  int16_t cp[4];
  for (int32_t i = 0; i < 4; ++i) {
    c_iter *= (1 << 14);
    cp[i] = int16_t(c_iter);
    c_iter -= cp[i];
  }

  int16_t ap[4];
  uint64_t a = fixed_point_mul_i15x4((uint64_t*)bp, (uint64_t*)cp);
  ap[0] = int16_t(a);
  ap[1] = int16_t(a >> 16);
  ap[2] = int16_t(a >> 32);
  ap[3] = int16_t(a >> 48);

  b_iter = 2. / (1 << 14);
  double re_m = 0;
  for (int32_t i = 0; i < 4; ++i) {
    re_m += b_iter * ap[i];
    b_iter *= 1. / (1 << 14);
  }

  printf("%.12e\n", std::abs(re_m - b * c));

  return 0;
}
