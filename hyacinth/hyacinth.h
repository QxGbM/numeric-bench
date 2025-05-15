
#pragma once

#include <stdint.h>
#include <cuComplex.h>
#include <cublas_v2.h>

int32_t align_up(
  int32_t ld,
  int32_t align
);

int32_t cpotrfp_gpu(
  cublasHandle_t handle,
  int32_t N,
  const cuComplex* A,
  int32_t lda,
  int32_t* ipiv,
  cuComplex* X,
  int32_t ldx,
  cuComplex* work
);

double f64_i8(
  cudaStream_t stream,
  int32_t M,
  int32_t N,
  const cuDoubleComplex* A,
  int32_t lda,
  int8_t* Ai8
);
