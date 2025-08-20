#!/bin/bash
#YBATCH -r ad_1
#SBATCH -N 1
#SBATCH -J hyac
#SBATCH --time=01:00:00
#SBATCH --output hyac/rtx4096_65536x4096.out

. /etc/profile.d/modules.sh
module load cmake/4.1.0 cuda/12.8 intel/2022/mkl
M=65536
N=4096

~/hyacinth/build/dgeqp3_example.app
~/hyacinth/build/dgeqp3_example.app $M $N 1.e-16
~/hyacinth/build/dgeqp3_example.app $M $N 1.e-14
~/hyacinth/build/dgeqp3_example.app $M $N 1.e-12
~/hyacinth/build/dgeqp3_example.app $M $N 1.e-10
~/hyacinth/build/dgeqp3_example.app $M $N 1.e-8
~/hyacinth/build/dgeqp3_example.app $M $N 1.e-6
~/hyacinth/build/dgeqp3_example.app $M $N 1.e-4
~/hyacinth/build/dgeqp3_example.app $M $N 1.e-2
echo ----------------------------------------------------------------

~/hyacinth/build/sgeqp3_example.app
~/hyacinth/build/sgeqp3_example.app $M $N 1.e-7
~/hyacinth/build/sgeqp3_example.app $M $N 1.e-5
~/hyacinth/build/sgeqp3_example.app $M $N 1.e-3
~/hyacinth/build/sgeqp3_example.app $M $N 1.e-1
echo ----------------------------------------------------------------

~/hyacinth/build/zgeqp3_example.app
~/hyacinth/build/zgeqp3_example.app $M $N 1.e-16
~/hyacinth/build/zgeqp3_example.app $M $N 1.e-14
~/hyacinth/build/zgeqp3_example.app $M $N 1.e-12
~/hyacinth/build/zgeqp3_example.app $M $N 1.e-10
~/hyacinth/build/zgeqp3_example.app $M $N 1.e-8
~/hyacinth/build/zgeqp3_example.app $M $N 1.e-6
~/hyacinth/build/zgeqp3_example.app $M $N 1.e-4
~/hyacinth/build/zgeqp3_example.app $M $N 1.e-2
echo ----------------------------------------------------------------

~/hyacinth/build/cgeqp3_example.app
~/hyacinth/build/cgeqp3_example.app $M $N 1.e-7
~/hyacinth/build/cgeqp3_example.app $M $N 1.e-5
~/hyacinth/build/cgeqp3_example.app $M $N 1.e-3
~/hyacinth/build/cgeqp3_example.app $M $N 1.e-1
echo ----------------------------------------------------------------

~/hyacinth/build/dlra_example.app
~/hyacinth/build/dlra_example.app $M $N 1.e-16
~/hyacinth/build/dlra_example.app $M $N 1.e-14
~/hyacinth/build/dlra_example.app $M $N 1.e-12
~/hyacinth/build/dlra_example.app $M $N 1.e-10
~/hyacinth/build/dlra_example.app $M $N 1.e-8
~/hyacinth/build/dlra_example.app $M $N 1.e-6
~/hyacinth/build/dlra_example.app $M $N 1.e-4
~/hyacinth/build/dlra_example.app $M $N 1.e-2
echo ----------------------------------------------------------------

~/hyacinth/build/slra_example.app
~/hyacinth/build/slra_example.app $M $N 1.e-7
~/hyacinth/build/slra_example.app $M $N 1.e-5
~/hyacinth/build/slra_example.app $M $N 1.e-3
~/hyacinth/build/slra_example.app $M $N 1.e-1
echo ----------------------------------------------------------------

~/hyacinth/build/zlra_example.app
~/hyacinth/build/zlra_example.app $M $N 1.e-16
~/hyacinth/build/zlra_example.app $M $N 1.e-14
~/hyacinth/build/zlra_example.app $M $N 1.e-12
~/hyacinth/build/zlra_example.app $M $N 1.e-10
~/hyacinth/build/zlra_example.app $M $N 1.e-8
~/hyacinth/build/zlra_example.app $M $N 1.e-6
~/hyacinth/build/zlra_example.app $M $N 1.e-4
~/hyacinth/build/zlra_example.app $M $N 1.e-2
echo ----------------------------------------------------------------

~/hyacinth/build/clra_example.app
~/hyacinth/build/clra_example.app $M $N 1.e-7
~/hyacinth/build/clra_example.app $M $N 1.e-5
~/hyacinth/build/clra_example.app $M $N 1.e-3
~/hyacinth/build/clra_example.app $M $N 1.e-1
echo ----------------------------------------------------------------

~/numeric-bench/build/dgeqrf_cusolver.app
~/numeric-bench/build/dgeqrf_cusolver.app $M $N
~/numeric-bench/build/dgeqrf_cusolver.app $M $N

~/numeric-bench/build/sgeqrf_cusolver.app
~/numeric-bench/build/sgeqrf_cusolver.app $M $N
~/numeric-bench/build/sgeqrf_cusolver.app $M $N

~/numeric-bench/build/zgeqrf_cusolver.app
~/numeric-bench/build/zgeqrf_cusolver.app $M $N
~/numeric-bench/build/zgeqrf_cusolver.app $M $N

~/numeric-bench/build/cgeqrf_cusolver.app
~/numeric-bench/build/cgeqrf_cusolver.app $M $N
~/numeric-bench/build/cgeqrf_cusolver.app $M $N

echo ----------------------------------------------------------------

~/numeric-bench/build/dsvd_polar_cusolver.app
~/numeric-bench/build/dsvd_polar_cusolver.app $M $N
~/numeric-bench/build/dsvd_polar_cusolver.app $M $N

~/numeric-bench/build/ssvd_polar_cusolver.app
~/numeric-bench/build/ssvd_polar_cusolver.app $M $N
~/numeric-bench/build/ssvd_polar_cusolver.app $M $N

~/numeric-bench/build/zsvd_polar_cusolver.app
~/numeric-bench/build/zsvd_polar_cusolver.app $M $N
~/numeric-bench/build/zsvd_polar_cusolver.app $M $N

~/numeric-bench/build/csvd_polar_cusolver.app
~/numeric-bench/build/csvd_polar_cusolver.app $M $N
~/numeric-bench/build/csvd_polar_cusolver.app $M $N

echo ----------------------------------------------------------------

~/numeric-bench/build/dgeqp3_magma.app
~/numeric-bench/build/dgeqp3_magma.app $M $N
~/numeric-bench/build/dgeqp3_magma.app $M $N

~/numeric-bench/build/sgeqp3_magma.app
~/numeric-bench/build/sgeqp3_magma.app $M $N
~/numeric-bench/build/sgeqp3_magma.app $M $N

~/numeric-bench/build/zgeqp3_magma.app
~/numeric-bench/build/zgeqp3_magma.app $M $N
~/numeric-bench/build/zgeqp3_magma.app $M $N

~/numeric-bench/build/cgeqp3_magma.app
~/numeric-bench/build/cgeqp3_magma.app $M $N
~/numeric-bench/build/cgeqp3_magma.app $M $N
