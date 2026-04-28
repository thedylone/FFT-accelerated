#!/bin/bash
#SBATCH --job-name=fft_cuda
#SBATCH --output=fft_cuda_%j.out
#SBATCH --error=fft_cuda_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --time=00:10:00

# Activate spack env
. ~/161588/spack/share/spack/setup-env.sh
spack env activate -p CS-2050-gpu

echo "--- Compiling and running ... ---"
cmake -B build .
cmake --build build

echo "FFT CUDA"
echo "--- GPU info ---"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

echo "block_size,n,time_ms" > results.csv

for BS in 32 64 128 256 512 1024; do
    OUTPUT=$(build/main $BS)
    echo "$OUTPUT"

    # Append only numeric data rows to CSV
    echo "$OUTPUT" | awk -F',' '$1~/^[0-9]+$/ { print }' >> results.csv
done

echo "--- Tidying ---"
rm -r build
