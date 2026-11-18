# What this library does

This library contains some examples for using Hyacin for computing low-rank approximation

# Authors
Qianxiang Ma, RIKEN R-CCS. ma@rio.scrc.iir.isct.ac.jp / qianxiang.ma@riken.jp

# Vendor benchmarks

The vendor executables support `data=D` for real FP64 (the default) and `data=Z`
for complex FP64. This selects real or complex matrix generation, storage,
BLAS operations, and solver compute types throughout the benchmark.

```
./build/xgeqp3_magma.app M=4096 N=512 epi=1e-12 data=D
./build/xgesvd_cusolver.app M=4096 N=512 epi=1e-12 data=D
./build/xgesvdr_cusolver.app M=4096 N=512 K=64 oversampling=16 piters=1 data=D
CUDA_VISIBLE_DEVICES=0 ./build/xgevd_vendor.app M=4096 N=512 K=64 data=D
```

MAGMA accepts named arguments (`M=... N=... epi=... data=D`), matching the
other vendor benchmarks. The SYRK/HERK + EVD benchmark uses NCCL; set
`CUDA_VISIBLE_DEVICES` to the GPU list for its row-distributed run.

Vendor timings report the mean over `kernel_runs` in `common.hpp` after a warmup.
Inputs and MAGMA pivot flags are restored before each timed factorization; input
copies are excluded from kernel time. The EVD path rebuilds the Gram matrix on
every run. Its kernel, communication, and total times are averaged separately.
MAGMA's host time includes setup, validation, and all repetitions.

GESVD throughput uses the conventional estimate
`4*M*N^2 + 8*N^3 + 9*N^2` for real arithmetic, scaled by four for complex
arithmetic. This replaces the incorrect `M^2*N` term. These are algorithmic
estimates; bidiagonal QR work depends on convergence and vendor implementation.

CUDA runtime, cuBLAS, cuSOLVER, and NCCL failures report the failing call and source
location and cause a nonzero exit. Solver `info` values and MAGMA statuses are
also checked, including each timed run.

Build and run the host-side regression checks (no GPU required):

```
cmake --build build -j
ctest --test-dir build --output-on-failure
```

The checks cover real/complex dispatch, residual calculations, GESVD work scaling,
and library failure reporting. GPU numerical and timing validation still requires
a working CUDA device and driver.

# Hyacin benchmarks

The Hyacin executables are `xlra_hyacin.app`, `xsvd_hyacin.app`, and
`xsvd_1drow_hyacin.app`. Use `data=D` for real FP64 or `data=Z` for complex FP64.
For example, to run the default 2048-by-2048 problem:

```
./build/xlra_hyacin.app data=D
./build/xsvd_hyacin.app data=Z
CUDA_VISIBLE_DEVICES=0,1 ./build/xsvd_1drow_hyacin.app data=D
```

The distributed SVD uses the GPUs listed in `CUDA_VISIBLE_DEVICES`.

Hyacin LRA and SVD print an estimated per-run `[flops=...]` count and effective
GFlop/s computed as `flops / (kernel_ms * 1e6)`. Kernel time is averaged over
`kernel_runs`. These are equivalent conventional algorithm work estimates;
Hyacin's quantized kernels execute a different mix of operations.

For real arithmetic, with `r` the returned rank and `k = min(M,N)`:

- LRA: `2*M*N*k - (2/3)*k^3 + r^2*(N-r)`. This counts full GEQP3 without
  forming Q, followed by a left triangular solve on `r` by `N-r` right-hand sides.
- SVD: `4*M*N*r + r^2*(4*M + 2*N)`. This uses the same leading-order estimate as
  cuSOLVER GESVDR with `jobu=N`, `jobvt=S`, no oversampling, and no power
  iterations, at the achieved rank rather than the maximum rank budget `K`.

Complex arithmetic multiplies these counts by four. The MAGMA LRA and cuSOLVER
GESVDR benchmarks share these formulas. Distributed Hyacin SVD uses global `M`
and the existing time averaged across GPUs and repetitions, including the
communication segment; its throughput is for the whole problem.
