
#include <commons.hpp>
#include <int_fp_encode.hpp>

void d2i(int32_t N, const double Xd[], int8_t Xi[]) {
  std::transform(Xd, &Xd[N], Xi, [](double e) { 
    int32_t i = (int32_t)std::roundf((float)e); return (int8_t)std::min(std::max(i, -127), 127); });
}

void i2d(int32_t N, const int8_t Xi[], double Xd[]) {
  std::transform(Xi, &Xi[N], Xd, [](float e) { return static_cast<double>(e); });
}

int32_t main() {
  int32_t M = 400, N = 40;
  Eigen::MatrixXcd matA(M, N);
  random_vector(M * N * 2, (double*)matA.data());

  Eigen::MatrixXcd ref = matA;
  double nrm = matA.lpNorm<Eigen::Infinity>();

  int3 test_i3 = make_int3(-13515313, -1515618, -4199848); //
  float2 test_f2 = host::int8::decode_scaled_int3_float2(test_i3);
  double test_double = double(test_f2.x) + double(test_f2.y);
  double ref_d = (double)test_i3.x * 16384 + (double)test_i3.y * 128 + (double)test_i3.z;
  printf("%lf %lf %lf\n", test_double, ref_d, test_double-ref_d);

  for (int8_t i = 0; i < 10; ++i) {
    Eigen::MatrixXcd matB = matA * (127. / nrm);

    std::vector<int8_t> matAi(M * N * 2);
    d2i(M * N * 2, (double*)matB.data(), matAi.data());

    Eigen::MatrixXcd reA(M, N);
    i2d(M * N * 2, matAi.data(), (double*)reA.data());

    double scale = nrm / 127.;
    matA -= reA * scale;
    nrm = scale;
    printf("iter: %d, nrm: %e, remainder: %e\n", i, scale, matA.norm() / ref.norm());
  }
  return 0;
}
