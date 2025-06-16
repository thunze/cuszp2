# cusZP2 Reproduction on RAMSES

- Original hardware used: NVIDIA A100
- RAMSES hardware used: NVIDIA H100, NVIDIA A30

## Prepare working directory

- Create working directory, using `/scratch` because it has a quota of 40 TB instead of 100 GB

```sh
cd /scratch
mkdir $USER
cd $USER
```

- Download AD/AE version of cusZP2

```sh
curl -sSL -o SC24-cuSZp2-cuSZp2-AD-AE-SC24.zip https://zenodo.org/records/13315526/files/hyfshishen/SC24-cuSZp2-cuSZp2-AD-AE-SC24.zip
unzip SC24-cuSZp2-cuSZp2-AD-AE-SC24.zip
cd hyfshishen-SC24-cuSZp2-85ce693
```

## Download datasets

- Fix dataset URLs by editing `./dataset-preparation.py` and replacing `https://` with `http://` for the last 3 URLs (`jetin_url`, `miranda_url`, `syntruss_url`)
- Download datasets for **main results**

```sh
python3 ./dataset-preparation.py
```

- Download datasets for **double-precision results**

```sh
cd double-precision-results
python3 0-dataset-preparation.py
cd ..
```

## Compilation

- Add the Ampere and Hopper architectures (compute capabilities `8.0` and `9.0`, i.e., codes `80` and `90`) to the target architectures to generate device code for by editing `./main-results/CMakeLists.txt` and `./double-precision-results/CMakeLists.txt`, and in each of these files, replacing `set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 75)` with `set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 75 80 90)`
  - [CUDA GPU Compute Capability Reference](https://developer.nvidia.com/cuda-gpus)
- Prepare environment variables for CUDA compilation

```sh
export CUDA_HOME=/projects/NEC/nvidia/cuda_11.8.0

export PATH="${CUDA_HOME}/bin${PATH:+:${PATH}}"
export CPATH="${CUDA_HOME}/include${CPATH:+:${CPATH}}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
```

- Compile binaries for **main results**

```sh
cd main-results
python3 0-compilation.py
cd ..
```

- Compile binaries for **double-precision results**

```sh
cd double-precision-results
python3 1-compilation.py
cd ..
```

## Execution

### Main results

- Create shell script `./main-results.sh` for the **main results** execution job with the following content

```sh
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
workdir=/scratch/${USER}/hyfshishen-SC24-cuSZp2-85ce693/main-results

pushd $workdir

python3 ./1-execution.py 1E-2
python3 ./1-execution.py 1E-3
python3 ./1-execution.py 1E-4

popd
```

- Submit the **main results** execution job to Slurm and watch the queue

```sh
sbatch ./main-results.sh
watch squeue --user $USER
```

- Find the job output in `slurm-<JOBID>.out` in the directory the job was started from

### Double precision results

- Create shell script `./double-precision-results.sh` for the **double-precision results** execution job with the following content

```sh
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
```

- Submit the **double-precision results** execution job to Slurm and watch the queue

```sh
sbatch ./double-precision-results.sh
watch squeue --user $USER
```

- Find the job output in `slurm-<JOBID>.out` in the directory the job was started from
