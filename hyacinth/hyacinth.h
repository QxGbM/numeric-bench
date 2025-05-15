
#pragma once

#include <stdint.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

int32_t cpotrfp_gpu(
  cublasHandle_t handle,
  int32_t N,
  const cuComplex* A,
  int32_t lda,
  int32_t* ipiv,
  cuComplex* X,
  int32_t ldx,
  cuComplex* work);

