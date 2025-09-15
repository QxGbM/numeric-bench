#!/bin/bash
#YBATCH -r h100_1
#SBATCH -N 1
#SBATCH -J bench
#SBATCH --time=04:00:00
#SBATCH --output hyac_results/h100_64000x4000_benches.out

. /etc/profile.d/modules.sh
module load cmake/4.1.0 cuda/12.8 intel/2022/mkl

M=64000
N=4000

nvidia-smi

~/numeric-bench/build/dgeqrf_cusolver.app $M $N
~/numeric-bench/build/sgeqrf_cusolver.app $M $N
~/numeric-bench/build/zgeqrf_cusolver.app $M $N
~/numeric-bench/build/cgeqrf_cusolver.app $M $N

echo ----------------------------------------------------------------

~/numeric-bench/build/dsvd_polar_cusolver.app $M $N
~/numeric-bench/build/ssvd_polar_cusolver.app $M $N
~/numeric-bench/build/zsvd_polar_cusolver.app $M $N
~/numeric-bench/build/csvd_polar_cusolver.app $M $N

echo ----------------------------------------------------------------

~/numeric-bench/build/dgeqp3_magma.app $M $N
~/numeric-bench/build/sgeqp3_magma.app $M $N
~/numeric-bench/build/zgeqp3_magma.app $M $N
~/numeric-bench/build/cgeqp3_magma.app $M $N

echo ----------------------------------------------------------------
