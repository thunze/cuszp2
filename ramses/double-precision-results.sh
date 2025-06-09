#!/bin/bash -l
#SBATCH --partition=gpu
#SBATCH -G h100:1
#SBATCH --mem=256gb
#SBATCH --time=0:30:00

set -e
set -u
set -o pipefail
set -x

# Prepare environment for GPU
export CUDA_HOME=/projects/NEC/nvidia/cuda_11.8.0
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# When using OpenMP Offloading, make sure that kernels are running on the GPU
export OMP_TARGET_OFFLOAD=MANDATORY

# Give information about data mapping and kernel launch
export NVCOMPILER_ACC_NOTIFY=3

rundir=$PWD
workdir=/scratch/${USER}/hyfshishen-SC24-cuSZp2-85ce693/double-precision-results

pushd $workdir

python3 ./2-execution.py 1E-2
python3 ./2-execution.py 1E-3
python3 ./2-execution.py 1E-4

popd
