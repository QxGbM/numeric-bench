#!/bin/bash
#YBATCH -r a100_1
#SBATCH -N 1
#SBATCH -J hyac
#SBATCH --time=04:00:00
#SBATCH --output hyac_results/a100_64000x4000_r1.out

. /etc/profile.d/modules.sh
module load cmake/4.1.0 cuda/12.8 intel/2022/mkl

M=64000
N=4000
omega=0.01
sep=16
#for M in {16000,32000,64000,128000,256000,512000}
#for N in {1000,2000,4000,8000,16000,32000}
#do

~/hyacinth/build/dgeqp3_example.app $M $N 1.e-16
~/hyacinth/build/dgeqp3_example.app $M $N 1.e-14
~/hyacinth/build/dgeqp3_example.app $M $N 1.e-12
~/hyacinth/build/dgeqp3_example.app $M $N 1.e-10
~/hyacinth/build/dgeqp3_example.app $M $N 1.e-8
~/hyacinth/build/dgeqp3_example.app $M $N 1.e-6
~/hyacinth/build/dgeqp3_example.app $M $N 1.e-4
~/hyacinth/build/dgeqp3_example.app $M $N 1.e-2
echo ----------------------------------------------------------------

~/hyacinth/build/sgeqp3_example.app $M $N 1.e-7
~/hyacinth/build/sgeqp3_example.app $M $N 1.e-5
~/hyacinth/build/sgeqp3_example.app $M $N 1.e-3
~/hyacinth/build/sgeqp3_example.app $M $N 1.e-1
echo ----------------------------------------------------------------

~/hyacinth/build/zgeqp3_example.app $M $N 1.e-16
~/hyacinth/build/zgeqp3_example.app $M $N 1.e-14
~/hyacinth/build/zgeqp3_example.app $M $N 1.e-12
~/hyacinth/build/zgeqp3_example.app $M $N 1.e-10
~/hyacinth/build/zgeqp3_example.app $M $N 1.e-8
~/hyacinth/build/zgeqp3_example.app $M $N 1.e-6
~/hyacinth/build/zgeqp3_example.app $M $N 1.e-4
~/hyacinth/build/zgeqp3_example.app $M $N 1.e-2
echo ----------------------------------------------------------------

~/hyacinth/build/cgeqp3_example.app $M $N 1.e-7
~/hyacinth/build/cgeqp3_example.app $M $N 1.e-5
~/hyacinth/build/cgeqp3_example.app $M $N 1.e-3
~/hyacinth/build/cgeqp3_example.app $M $N 1.e-1
echo ----------------------------------------------------------------

~/hyacinth/build/dlra_example.app $M $N 1.e-16 $omega $sep
~/hyacinth/build/dlra_example.app $M $N 1.e-14 $omega $sep
~/hyacinth/build/dlra_example.app $M $N 1.e-12 $omega $sep
~/hyacinth/build/dlra_example.app $M $N 1.e-10 $omega $sep
~/hyacinth/build/dlra_example.app $M $N 1.e-8 $omega $sep
~/hyacinth/build/dlra_example.app $M $N 1.e-6 $omega $sep
~/hyacinth/build/dlra_example.app $M $N 1.e-4 $omega $sep
~/hyacinth/build/dlra_example.app $M $N 1.e-2 $omega $sep
echo ----------------------------------------------------------------

~/hyacinth/build/slra_example.app $M $N 1.e-7 $omega $sep
~/hyacinth/build/slra_example.app $M $N 1.e-5 $omega $sep
~/hyacinth/build/slra_example.app $M $N 1.e-3 $omega $sep
~/hyacinth/build/slra_example.app $M $N 1.e-1 $omega $sep
echo ----------------------------------------------------------------

~/hyacinth/build/zlra_example.app $M $N 1.e-16 $omega $sep
~/hyacinth/build/zlra_example.app $M $N 1.e-14 $omega $sep
~/hyacinth/build/zlra_example.app $M $N 1.e-12 $omega $sep
~/hyacinth/build/zlra_example.app $M $N 1.e-10 $omega $sep
~/hyacinth/build/zlra_example.app $M $N 1.e-8 $omega $sep
~/hyacinth/build/zlra_example.app $M $N 1.e-6 $omega $sep
~/hyacinth/build/zlra_example.app $M $N 1.e-4 $omega $sep
~/hyacinth/build/zlra_example.app $M $N 1.e-2 $omega $sep
echo ----------------------------------------------------------------

~/hyacinth/build/clra_example.app $M $N 1.e-7 $omega $sep
~/hyacinth/build/clra_example.app $M $N 1.e-5 $omega $sep
~/hyacinth/build/clra_example.app $M $N 1.e-3 $omega $sep
~/hyacinth/build/clra_example.app $M $N 1.e-1 $omega $sep
echo ----------------------------------------------------------------

#done
