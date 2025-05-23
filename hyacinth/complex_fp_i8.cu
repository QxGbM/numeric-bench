
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
    return thrust::pair<int32_t, int32_t>((int32_t)fmaf(x, lda, y), (int32_t)fmaf(x, ldb, y));
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
  const float scale;
  const cuComplex* in;
  int16_t* out;
  convert_f32(float scale, const cuComplex* in, int16_t* out) : scale(scale), in(in), out(out) {}
  __device__ void operator()(thrust::pair<int32_t, int32_t> idx) {
    int32_t e0 = (int32_t)(in[idx.first].x * scale);
    int32_t e1 = (int32_t)(in[idx.first].y * scale);
    uint32_t e = __vmins2(__vmaxs2((e1 << 16) | (uint16_t)e0, 0xFF81FF81), 0x007F007F);
    out[idx.second] = ((e >> 8) & 0xFF00) | (uint8_t)e;
  }
};

float c_f32_i8(cudaStream_t stream, int32_t M, int32_t N, const cuComplex* A, int32_t lda, int8_t* Ai8, int32_t ldi) {
  auto idx_iter = thrust::make_transform_iterator(thrust::make_counting_iterator(0), snrm_idx(M, A, lda));
  double nrm = thrust::reduce(thrust::cuda::par_nosync.on(stream), idx_iter, idx_iter + (M * N), 0., thrust::maximum<double>());

  auto idx2_iter = thrust::make_transform_iterator(thrust::make_counting_iterator(0), convert_idx2(M, lda, ldi));
  thrust::for_each_n(thrust::cuda::par_nosync.on(stream), idx2_iter, M * N, convert_f32(127.f / nrm, A, (int16_t*)Ai8));
  return nrm / 127.f;
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
  const double scale;
  const cuDoubleComplex* in;
  int16_t* out;
  convert_f64(double scale, const cuDoubleComplex* in, int16_t* out) : scale(scale), in(in), out(out) {}
  __device__ void operator()(thrust::pair<int32_t, int32_t> idx) {
    int32_t e0 = (int32_t)(in[idx.first].x * scale);
    int32_t e1 = (int32_t)(in[idx.first].y * scale);
    uint32_t e = __vmins2(__vmaxs2((e1 << 16) | (uint16_t)e0, 0xFF81FF81), 0x007F007F);
    out[idx.second] = ((e >> 8) & 0xFF00) | (uint8_t)e;
  }
};

double c_f64_i8(cudaStream_t stream, int32_t M, int32_t N, const cuDoubleComplex* A, int32_t lda, int8_t* Ai8, int32_t ldi) {
  auto idx_iter = thrust::make_transform_iterator(thrust::make_counting_iterator(0), dnrm_idx(M, A, lda));
  double nrm = thrust::reduce(thrust::cuda::par_nosync.on(stream), idx_iter, idx_iter + (M * N), 0., thrust::maximum<double>());

  auto idx2_iter = thrust::make_transform_iterator(thrust::make_counting_iterator(0), convert_idx2(M, lda, ldi));
  thrust::for_each_n(thrust::cuda::par_nosync.on(stream), idx2_iter, M * N, convert_f64(127. / nrm, A, (int16_t*)Ai8));
  return nrm / 127.;
}
