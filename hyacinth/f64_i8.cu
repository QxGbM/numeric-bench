
#include <hyacinth.h>
#include <cuda_runtime_api.h>
#include <thrust/pair.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/for_each.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/counting_iterator.h>

#include <vector>
#include <complex>

struct nrm_idx {
  const cuDoubleComplex* A;
  float N, rN, ld;
  nrm_idx(int32_t N, const cuDoubleComplex* A, int32_t ld) : A(A), N(N), rN(1.f / N), ld(ld) {}
  __device__ double operator()(int32_t idx) {
    float fidx = idx, x = floorf(rN * fidx), y = fmaf(-x, N, fidx);
    int32_t i = (int32_t)fmaf(x, ld, y);
    cuDoubleComplex e = A[i];
    return hypot(e.x, e.y);
  }
};

struct convert_idx2 {
  float N, rN, lda, ldb;
  convert_idx2(int32_t N, int32_t lda, int32_t ldb) : N(N), rN(1.f / N), lda(lda), ldb(ldb) {}
  __device__ thrust::pair<int32_t, int32_t> operator()(int32_t idx) {
    float fidx = idx, x = floorf(rN * fidx), y = fmaf(-x, N, fidx);
    return thrust::pair<int32_t, int32_t>(fmaf(x, lda, y), fmaf(x, ldb, y));
  }
};

struct convert_f64 {
  const double scale, *in;
  int8_t* out;
  convert_f64(double scale, const double* in, int8_t* out) : scale(scale), in(in), out(out) {}
  __device__ void operator()(thrust::pair<int32_t, int32_t> idx) {
    int32_t e = __double2int_rz(in[idx.first] * scale);
    out[idx.second] = (int8_t)(min(max(e, -128), 127));
  }
};

double f64_i8(cudaStream_t stream, int32_t M, int32_t N, const cuDoubleComplex* A, int32_t lda, int8_t* Ai8) {
  int32_t ld = align_up(M, 64);
  if (Ai8 == nullptr)
    return (double)ld;
  
  auto idx_iter = thrust::make_transform_iterator(thrust::make_counting_iterator(0), nrm_idx(M, A, lda));
  double nrm = thrust::reduce(thrust::cuda::par_nosync.on(stream), idx_iter, idx_iter + (M * N), 0., thrust::maximum<double>());

  auto idx2_iter = thrust::make_transform_iterator(thrust::make_counting_iterator(0), convert_idx2(M, 2 * lda, 2 * ld));
  thrust::for_each_n(thrust::cuda::par_nosync.on(stream), idx2_iter, M * N, convert_f64(128. / nrm, (const double*)A, Ai8));
  return nrm / 128.;
}
