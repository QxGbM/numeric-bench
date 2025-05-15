
#include <hyacinth.h>
#include <cuda_runtime_api.h>
#include <thrust/pair.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/for_each.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/counting_iterator.h>

struct convert_idx2 {
  float N, rN, lda, ldb;
  convert_idx2(int32_t N, int32_t lda, int32_t ldb) : N(N), rN(1.f / N), lda(lda), ldb(ldb) {}
  __device__ thrust::pair<int32_t, int32_t> operator()(int32_t idx) {
    float fidx = idx, x = floorf(rN * fidx), y = fmaf(-x, N, fidx);
    return thrust::pair<int32_t, int32_t>(fmaf(x, lda, y), fmaf(x, ldb, y));
  }
};

struct snrm_idx {
  const cuComplex* A;
  float N, rN, ld;
  snrm_idx(int32_t N, const cuComplex* A, int32_t ld) : A(A), N(N), rN(1.f / N), ld(ld) {}
  __device__ double operator()(int32_t idx) {
    float fidx = idx, x = floorf(rN * fidx), y = fmaf(-x, N, fidx);
    int32_t i = (int32_t)fmaf(x, ld, y);
    cuComplex e = A[i];
    return hypotf(e.x, e.y);
  }
};

struct convert_f32 {
  const float scale, *in;
  int8_t* out;
  convert_f32(float scale, const float* in, int8_t* out) : scale(scale), in(in), out(out) {}
  __device__ void operator()(thrust::pair<int32_t, int32_t> idx) {
    int32_t e = (int32_t)(in[idx.first] * scale);
    out[idx.second] = (int8_t)(min(max(e, -128), 127));
  }
};

float c_f32_i8(cudaStream_t stream, int32_t M, int32_t N, const cuComplex* A, int32_t lda, int8_t* Ai8, int32_t ldi) {
  auto idx_iter = thrust::make_transform_iterator(thrust::make_counting_iterator(0), snrm_idx(M, A, lda));
  double nrm = thrust::reduce(thrust::cuda::par_nosync.on(stream), idx_iter, idx_iter + (M * N), 0., thrust::maximum<double>());

  auto idx2_iter = thrust::make_transform_iterator(thrust::make_counting_iterator(0), convert_idx2(2 * M, 2 * lda, 2 * ldi));
  thrust::for_each_n(thrust::cuda::par_nosync.on(stream), idx2_iter, 2 * M * N, convert_f32(128.f / nrm, (const float*)A, Ai8));
  return nrm / 128.f;
}

struct dnrm_idx {
  const cuDoubleComplex* A;
  float N, rN, ld;
  dnrm_idx(int32_t N, const cuDoubleComplex* A, int32_t ld) : A(A), N(N), rN(1.f / N), ld(ld) {}
  __device__ double operator()(int32_t idx) {
    float fidx = idx, x = floorf(rN * fidx), y = fmaf(-x, N, fidx);
    int32_t i = (int32_t)fmaf(x, ld, y);
    cuDoubleComplex e = A[i];
    return hypot(e.x, e.y);
  }
};

struct convert_f64 {
  const double scale, *in;
  int8_t* out;
  convert_f64(double scale, const double* in, int8_t* out) : scale(scale), in(in), out(out) {}
  __device__ void operator()(thrust::pair<int32_t, int32_t> idx) {
    int32_t e = (int32_t)(in[idx.first] * scale);
    out[idx.second] = (int8_t)(min(max(e, -128), 127));
  }
};

double c_f64_i8(cudaStream_t stream, int32_t M, int32_t N, const cuDoubleComplex* A, int32_t lda, int8_t* Ai8, int32_t ldi) {
  auto idx_iter = thrust::make_transform_iterator(thrust::make_counting_iterator(0), dnrm_idx(M, A, lda));
  double nrm = thrust::reduce(thrust::cuda::par_nosync.on(stream), idx_iter, idx_iter + (M * N), 0., thrust::maximum<double>());

  auto idx2_iter = thrust::make_transform_iterator(thrust::make_counting_iterator(0), convert_idx2(2 * M, 2 * lda, 2 * ldi));
  thrust::for_each_n(thrust::cuda::par_nosync.on(stream), idx2_iter, 2 * M * N, convert_f64(128. / nrm, (const double*)A, Ai8));
  return nrm / 128.;
}
