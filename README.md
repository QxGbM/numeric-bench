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

Vendor timings report the mean over `kernel_runs` in `benchmark_common.hpp` after a warmup.
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

# HSS construction benchmark

`hss_cusolver.app` constructs a complex FP64 HSS representation from a distributed
dense Helmholtz matrix and compares its matvec against the dense reference. It
uses MPI processes with NCCL GPU communication; local MPI ranks select GPUs from
`CUDA_VISIBLE_DEVICES` in round-robin order. Each active local MPI process must
use a distinct physical GPU; launching more active local processes than visible
GPUs with a shared device list is rejected by NCCL.

```
CUDA_VISIBLE_DEVICES=0,1 mpiexec -n 2 ./build/hss_cusolver.app 2048 512 1e-10
```

The positional arguments are matrix order, leaf size, and construction tolerance,
with defaults `2048`, `512`, and `1e-10`. `DenseMatrix::eps` is public and initialized
to `0.` (no tolerance-based truncation). Set it before construction; the benchmark
assigns `denseA.eps = epi` before `matA.evalMatVecA(denseA, cell.data())`.
Compression and evaluation read the stored tolerance rather than accepting a
separate tolerance argument. The reported error checks one deterministic random
right-hand side; it is not a full matrix-norm error bound.

All benchmarks require MPI and a Hyacin installation built with NCCL. CMake
always builds the dense and HSS targets and links MPI; no MPI/NCCL feature flags
are needed. Both `bootstrap_posix_fork` and `bootstrap_mpi` are always available
in `benchmark_common.hpp`. The distributed dense benchmarks currently call the POSIX
bootstrap and use `CUDA_VISIBLE_DEVICES`; HSS always uses MPI.

# Benchmark headers

`benchmark_common.hpp` contains the shared matrix generator, MPI/NCCL bootstrap,
status checks, timing settings, FP64 reference BLAS operations, and flop estimates.
It does not include Hyacin. `hyacin_common.hpp` includes that shared header and
adds the Hyacin precision mappings, real/complex FP16/FP32/FP64 conversions, and
FP64 residual helpers. The library residual checks also promote the stored inputs
to FP64 before computing QR, products, and norms. `vendor_common.hpp` includes the
shared header directly and provides the simpler real/complex FP64 vendor wrappers
and reference checks. Each benchmark includes its backend header; HSS remains
independent of all three headers.

# Multiple ranks per GPU

The distributed dense benchmarks (`xsvd_1drow_hyacin`, `xgevd_vendor`, and
`bootstrap_check`) enable `NCCL_MULTI_RANK_GPU_ENABLE=1` before NCCL initialization.
This requires NCCL 2.31 or newer. Use `ranks=3` or `ranks=4` to launch more POSIX
ranks than visible GPUs. Omitting `ranks` launches one rank on one GPU;
`ranks=0` explicitly selects one rank per visible GPU. Device assignment is `local_rank % device_count`, so all selected device
indices remain valid when ranks outnumber GPUs.

```
CUDA_VISIBLE_DEVICES=0,1 ./build/bootstrap_check.app ranks=3
CUDA_VISIBLE_DEVICES=0,1 ./build/xsvd_1drow_hyacin.app data=D ranks=4
CUDA_VISIBLE_DEVICES=0,1 ./build/xgevd_vendor.app data=Z ranks=4
```

HSS does not call this environment helper and retains its existing launch behavior.

# Fixed-rank HSS with RSVD

`hss_rsvd.app` shares all HSS construction, redistribution, communication, and
matvec code with `hss_cusolver.app` through `hss_common.hpp`. These headers are
independent of the dense benchmark headers. `DenseMatrixCustom`, defined in
`hss_rsvd.cu`, is local to that executable and derives from `DenseMatrix`; virtual compression and tolerance hooks
select its RSVD behavior when HSS holds a `DenseMatrix&`.

```
CUDA_VISIBLE_DEVICES=0,1 mpiexec -n 2 ./build/hss_rsvd.app 2048 512 200 0 0 1.15
```

Positional arguments are matrix order, leaf size, leaf rank, oversampling, power
iterations, and alpha. The defaults are `2048 512 200 0 0 1.15`.
`DenseMatrixCustom` publicly exposes `K`, `oversampling`, `power_iters`, and
`alpha`. At each existing level-transition hook, `K` becomes
`floor(double(K) * alpha)`. `eps` remains on the base class and is ignored by RSVD.
The GESVD executable retains its original `Nbody leaf_size eps` arguments and
its no-op tolerance-iteration hook.

RSVD calls `cusolverDnXgesvdr` with `jobu=N`, `jobvt=S` on the original block
shape, including wide blocks. It preserves the input and computes the compact
left factor as `A*V` with complex FP64 GEMM. Singular values are neither used to
scale the compact factor nor read for tail truncation; every compressed block
writes exactly `K` into its recipe. Invalid rank/oversampling combinations are
rejected rather than silently reducing `K`: cuSOLVER requires
[`K + oversampling < min(block rows, block cols)`](https://docs.nvidia.com/cuda/archive/13.3.1/cusolver/).
The same constraint must hold as ranks grow across levels.

GPU regression tests for tall/wide blocks, fixed rank on rank-deficient inputs,
`A*V`, overlapping compaction, ignored `eps`, and virtual rank iteration can be
built and run explicitly on a CUDA-capable host:

```
cmake -S . -B build -DENABLE_HSS_GPU_TESTS=ON
cmake --build build --target hss_rsvd_checks.app -j
ctest --test-dir build -R hss_rsvd_gpu_checks --output-on-failure
```
