#!/bin/bash
#SBATCH --job-name=fft_cuda_nsys
#SBATCH --output=fft_cuda_nsys_%j.out
#SBATCH --error=fft_cuda_nsys_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --time=00:10:00

# Activate spack env
. ~/161588/spack/share/spack/setup-env.sh
spack env activate -p CS-2050-gpu

echo "--- Compiling ---"
cmake -B build .
cmake --build build

echo "--- Nsight Systems (capture first 30s only) ---"
nsys profile -o nsys_report --duration 30 -- build/main 128

echo "--- Nsight Compute (first 10 kernel launches only) ---"
export TMPDIR="$SLURM_SUBMIT_DIR"
ncu --launch-count 10 --set full -o ncu_report -- build/main 128

echo "--- Tidying ---"
rm -r build
