# What this library does

This library contains some examples for using Hyacin for computing low-rank approximation

# Authors
Qianxiang Ma, RIKEN R-CCS. ma@rio.scrc.iir.isct.ac.jp / qianxiang.ma@riken.jp

Codex, OpenAI — The BEST AI coding assistant.

# Vendor benchmarks

The vendor executables support `data=D` for real FP64 (the default) and `data=Z`
for complex FP64. This selects real or complex matrix generation, storage,
BLAS operations, and solver compute types throughout the benchmark.

```
./build/xgeqp3_magma.app M=4096 N=512 eps=1e-12 data=D
./build/xgesvd_cusolver.app M=4096 N=512 eps=1e-12 data=D
./build/xgesvdr_cusolver.app M=4096 N=512 K=64 oversampling=16 piters=1 data=D
./build/xgesvj_magma_batch.app M=256 N=128 batchN=16 eps=1e-12 data=D
./build/xgersvd_kblas_batch.app M=256 N=256 K=64 batchN=16 eps=1e-12 data=D
CUDA_VISIBLE_DEVICES=0 ./build/xgevd_vendor.app M=4096 N=512 K=64 data=D
```

MAGMA accepts named arguments (`M=... N=... eps=... data=D`), matching the
other vendor benchmarks. The SYRK/HERK + EVD benchmark uses NCCL; set
`CUDA_VISIBLE_DEVICES` to the GPU list for its row-distributed run.

Vendor timings report the mean over `kernel_runs` in `benchmark_common.hpp` after a warmup.
Inputs and MAGMA pivot flags are restored before each timed factorization; input
copies are excluded from kernel time. The EVD path rebuilds the Gram matrix on
every run. Its kernel, communication, and total times are averaged separately.
MAGMA's host time includes setup, validation, and all repetitions.

GESVDR keeps `jobu=N`, `jobvt=S` for timing and reports its V-side projection
error as `err_AVVH`. An additional untimed `jobu=S`, `jobvt=S` decomposition forms
`B=U*S` with cuBLAS DGMM and reports `err_USVH = ||A-B*V^H||_F / ||A||_F`.
Both checks run for real and complex FP64.

`xgesvj_magma_batch.app` benchmarks `magma_dgesvj_batched_strided` or
`magma_zgesvj_batched_strided` on one GPU with `MagmaVec` for both U and V.
Defaults are `M=32 N=32 batchN=1024 eps=1e-12 data=D`. Both tall and wide blocks
are supported, including dimensions above 32. These MAGMA routines take
GPU-resident arrays despite having no `_gpu` suffix; they return economy-size
`M`-by-`min(M,N)` U and `N`-by-`min(M,N)` V factors. Dimensions, batch count, and
per-matrix strides must fit MAGMA's integer type. The shared generator
creates one `M`-by-`N*batchN` matrix, with consecutive groups of `N` columns
forming the batch.

Only validation uses `eps`: each block is truncated using the same reverse
cumulative squared-singular-value rule as the GESVD benchmark. `err_USVH`
reports `||B-A||_F/||A||_F` for the concatenated reconstruction
`B=[U1*S1*V1^H U2*S2*V2^H ...]`. The output includes minimum, mean, and maximum
retained rank. Timed calls compute economy-size singular vectors without truncation,
restore the input before each call, exclude copy time, and average over
`kernel_runs` using events on the MAGMA queue's CUDA stream. The driver's
internal workspace setup remains part of each timed call. Both the MAGMA return
status and every batch member's `info` are checked after each call.
The reported `flops` and GFlops use a GESVD-equivalent estimate summed over the
batch, `batchN * gesvd_flops(max(M,N), min(M,N), scalar_factor)`; actual Jacobi
work depends on convergence, and this strided driver does not expose sweep counts.

On the RTX A5000, MAGMA 2.10's default complex block-size tuning can exceed
available shared memory for some shapes, including `M=128 N=64 data=Z`.
The benchmark rejects a blocked path when its matrix/eigenvector storage alone
exceeds the GPU limit; MAGMA can otherwise print an error while returning
success and zero `info`. Complex `M=192 N=96` and `M=96 N=192` were verified on
this GPU. Other affected shapes require a change to MAGMA's tuning or use of its
expert interface with a smaller block size. The scalar `1x1` case is also rejected
because MAGMA 2.10's fused path fails an internal thread-count assertion.

`xgersvd_kblas_batch.app` profiles KBLAS GERSVD through
`kblasDsvd_full_batch_strided(..., SVD_random, ...)`. Defaults are
`M=256 N=256 K=64 batchN=16 eps=1e-12 data=D seed=0`. It uses the shared
generator for one `M`-by-`N*batchN` matrix and requests rank `K` for every block.
The installed KBLAS SVD interface supports real FP32/FP64 only; this benchmark
uses FP64 and explicitly rejects `data=Z`. Dimensions, batch count, strides, and
per-block workspace must fit KBLAS's integer interface, with
`1 <= K <= min(M,N)` and `K<N` so the timed call
uses the randomized algorithm. KBLAS's `K=N` fallback switches to ordinary Jacobi
and failed the tall-block workspace/reconstruction checks in this installation;
this benchmark rejects that boundary.
This interface exposes neither oversampling nor power iterations.

The installed FP64 batched QR has a **1024-row limit**, independent of alignment.
With `QR_SUPPORT_LARGE` enabled, its configured maximum is
`32768 / (4 * sizeof(double)) = 1024`. GERSVD applies QR to both an `M`-by-`K`
sample and an `N`-by-`K` projection, so the benchmark rejects either `M>1024` or
`N>1024` before GPU setup. Padding the leading dimension does not bypass this
check. The public header still documents 512 rows; the
[kernel configuration](https://github.com/ecrc/kblas-gpu/blob/master/src/batch_svd/qr_kernels.cuh#L22-L36)
and [QR argument check](https://github.com/ecrc/kblas-gpu/blob/master/src/batch_svd/batch_qr.cu#L940-L944)
give the effective limit in this installation.

KBLAS returns a left basis `U` and a scaled right factor `V=A^T*U`, so the
reconstruction is `U*V^T`. Singular values must not be multiplied into these
factors again. Validation sorts the returned singular values and the
corresponding factor columns, applies the same Frobenius-tail truncation rule
as GESVD, and reports the concatenated relative residual as `err_UVT`.
For `err_energy`, validation subtracts the sum of the squared retained
singular values from each block's squared Frobenius norm and floors negative
differences at zero. It sums the discarded energies across the batch,
takes the square root, and divides by the Frobenius norm of the full input.
`eps` affects validation only, and a rank cap can prevent the requested
tolerance from being reached.

The random state and queried workspace are allocated before timing. Each timed
call uses the original rank `K`, restores the input outside the measured interval,
and runs on the KBLAS handle's CUDA stream; reported kernel time is the mean over
`kernel_runs`. KBLAS's internal input copy and factor construction remain part of
the timed full-interface call. Return codes and CUDA completion are checked.
Flops are leading-order estimates for the randomized factorization plus its
left-basis rotation and final `A^T*U`; convergence-dependent inner Jacobi work is
not counted exactly.

CMake requires the installed `kblasgpu` package and links this benchmark through
`KBLASGPU::kblasgpu`. Its default search path is
`~/cmake_install/lib64/cmake/kblasgpu`; use `-Dkblasgpu_DIR=/path/to/package` to
select another installation. The package supplies its CUDA, BLAS, OpenMP, and
MAGMA dependencies.

GESVD throughput uses the conventional estimate
`4*M*N^2 + 8*N^3 + 9*N^2` for real arithmetic, scaled by four for complex
arithmetic. This replaces the incorrect `M^2*N` term. These are algorithmic
estimates; bidiagonal QR work depends on convergence and vendor implementation.

CUDA runtime, cuBLAS, cuSOLVER, and NCCL failures report the failing call and source
location and cause a nonzero exit. Solver `info` values and MAGMA statuses are
also checked, including each timed run.

Build and run all regression checks (CUDA libraries and a working GPU are required):

```
cmake --build build -j
ctest --test-dir build --output-on-failure
```

The checks cover real/complex dispatch, residual calculations, GESVD work scaling,
library failure reporting, and GPU compression. All dependencies are required;
the build always includes the host and GPU test targets.

Hyacin's GPU projection and norm checks run with CTest, or directly through
`./build/hyacin_checks.app --gpu`.
These cover all supported storage types, FP64 promotion, padded matrices,
adjoint inputs, and projection onto nonorthogonal bases.

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

`hss_cusolver.app` constructs a real or complex FP64 HSS representation from a distributed
dense Helmholtz matrix and compares its matvec against the dense reference. It
uses MPI processes with NCCL GPU communication; local MPI ranks select GPUs from
`CUDA_VISIBLE_DEVICES` in round-robin order. Each active local MPI process must
use a distinct physical GPU; launching more active local processes than visible
GPUs with a shared device list is rejected by NCCL.

```
CUDA_VISIBLE_DEVICES=0,1 mpiexec -n 2 ./build/hss_cusolver.app N=2048 leaf_size=512 eps=1e-10 data=D
```

All HSS benchmarks accept `N` (matrix order, default 2048), `leaf_size` (default
512), and `data=D` for real FP64 or `data=Z` for complex FP64 (default). The
real benchmark generates the real part of the same Helmholtz kernel. The GESVD and
Hyacin backends accept `eps` for construction tolerance (default `1e-10`). The
original positional forms also work, with `data` passed as a named argument.

`DenseMatrix<T>::eps` is public and initialized to `0.`. The benchmark sets it
before HSS construction. The reported error compares the HSS and dense
matvec results for one deterministic random right-hand side.

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
the typed wrapper for `hyacinCheckNorm`. The library residual checks promote the stored inputs
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

# Profiling POSIX ranks with Nsight Systems

For benchmarks using `bootstrap_posix_fork`, enable tracing of forked children
before `exec`:

```bash
CUDA_VISIBLE_DEVICES=0,1 nsys profile --trace-fork-before-exec=true ./build/bootstrap_check.app ranks=2
```

Without this option, a forked rank can abort under the profiler and leave the
other ranks waiting inside NCCL initialization. This command was verified with
Nsight Systems 2026.4.1. NVIDIA still documents limitations for
[profiling fork without exec](https://docs.nvidia.com/nsight-systems/UserGuide/index.html);
the MPI bootstrap is the more reliable profiling path.

# Fixed-rank HSS with RSVD

`hss_rsvd.app` shares all HSS construction, redistribution, communication, and
matvec code with `hss_cusolver.app` through `hss_common.hpp`. These headers are
independent of the dense benchmark headers. `DenseMatrixCustom`, defined in
`hss_rsvd.cu`, is local to that executable and derives from `DenseMatrix<T>`. Virtual compression and tolerance hooks select
its RSVD behavior when HSS holds a `DenseMatrix<T>&`.

```
CUDA_VISIBLE_DEVICES=0,1 mpiexec -n 2 ./build/hss_rsvd.app N=2048 leaf_size=512 K=200 oversampling=0 piters=0 alpha=1.15 data=D
```

`K` sets the leaf rank (default 200). `oversampling` and `piters` default to 0,
and `alpha` defaults to 1.15. At each level transition, the rank parameter
becomes `floor(double(K) * alpha)`. RSVD keeps the base class's `eps` field
unused.

RSVD calls `cusolverDnXgesvdr` with `jobu=N`, `jobvt=S` on the original block
shape, including wide blocks. It preserves the input and computes the compact
left factor as `A*V` with real or complex FP64 GEMM. Singular values are neither used to
scale the compact factor nor read for tail truncation; every compressed block
writes exactly `K` into its recipe. Invalid rank/oversampling combinations are
rejected rather than silently reducing `K`: cuSOLVER requires
[`K + oversampling < min(block rows, block cols)`](https://docs.nvidia.com/cuda/archive/13.3.1/cusolver/).
The same constraint must hold as ranks grow across levels.

GPU regression tests for tall/wide blocks, fixed rank on rank-deficient inputs,
`A*V`, overlapping compaction, ignored `eps`, and virtual rank iteration can be
built and run with the standard configuration:

```
cmake -S . -B build
cmake --build build --target hss_rsvd_checks.app -j
ctest --test-dir build -R hss_rsvd_gpu_checks --output-on-failure
```
# Tolerance-based HSS with Hyacin

`hss_hyacin.app` uses the same HSS construction and matvec code in
`hss_common.hpp`, with a local `DenseMatrixCustom` compression override in
`hss_hyacin.cu`.

```
CUDA_VISIBLE_DEVICES=0,1 mpiexec -n 2 ./build/hss_hyacin.app N=2048 leaf_size=512 eps=1e-10 data=Z
```

Compression uses the base class's public `eps` and retains the
base no-op tolerance-iteration hook. Hyacin fits the right basis directly for
both tall and wide blocks, caps the rank at `min(M,N)`, and writes its internally
truncated rank into the recipe. The input and basis use separate allocations,
so the preserved input supplies the compact left factor through `A*V` in the selected FP64 precision.

The matrix owns one Hyacin handle, initialized with the HSS stream, cuBLAS, and
cuSOLVER handles before `hyacinCreate`. This requires Hyacin's null-checking
resource initialization; ownership is managed entirely by Hyacin. HSS retains
its existing NCCL communication, while each Hyacin fit processes a local block.

Build `hss_hyacin_checks.app` and run
`ctest --test-dir build -R hss_hyacin_gpu_checks --output-on-failure` to check
tall/wide inputs, internal truncation, `A*V`, overlapping block compaction, and
borrowed-resource lifetime.
